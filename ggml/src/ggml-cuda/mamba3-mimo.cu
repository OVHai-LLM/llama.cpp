#include <cstdint>
#include <algorithm>
#include "mamba3-mimo.cuh"

// Fused Olala Mamba3-MIMO mixer core (see ggml_mamba3_mimo in ggml.h).
// semantic reference: ggml_compute_forward_mamba3_mimo (ggml/src/ggml-cpu/ops.cpp).
//
// One block per (head, seq[, chunk]). Every token step builds its own operands
// from the raw projections in shared memory (B/C rms-norm + weight + bias +
// rotary, x/z MIMO scaling, softplus/exp/sigmoid discretization), then updates
// the state and, when producing outputs, applies the D skip, the SiLU(z) gate
// and the MIMO output combine. Warps own state rows p, lanes stride d, so the
// state update and the q.S contraction share registers and need no sync.
//
// short ubatches (decode): one kernel, serial over the tokens of the sequence.
// long ubatches (prefill): the recurrence is affine in S, so with chunks of
// M3_CHUNK tokens S_end(n) = P_n S_end(n-1) + L_n, where P_n is the product of
// the chunk's alphas and L_n the chunk-local state run from S = 0:
//   angles:  per (head, seq): serial phase cumsum -> per-token angle buffer
//   phase 1: per (chunk, head, seq): L_n and P_n                (parallel)
//   phase 2: per (head, seq): S_end scan, leaves chunk carry-ins (serial over chunks)
//   phase 3: per (chunk, head, seq): outputs from the carry-in  (parallel)

#define M3_CHUNK    32
#define M3_MAX_R    8
#define M3_NTHREADS 256

struct m3_params {
    const float * pdyn;   int64_t pdyn_s;   // token stride (floats)
    const float * pstat;  int64_t pstat_s;
    const float * bias;                     // (D_qk, 2R, H)
    const float * mxz;                      // (D_v, 2R, H)
    const float * mimo_o;                   // (D_v, R, H)
    const float * norms;                    // (D_qk, 2)
    const float * misc;                     // (H, 2)
    const float * s_in;   int64_t s_in_s;   // seq stride (floats)
    float       * y;                        // (D_v, H, n_tok)
    float       * s_out;  int64_t s_out_s;  // (n_embd_s, n_seqs), seq stride (floats)
    int32_t     * tile_done;                // (H, n_seqs) completion counter for in-place n_tiles > 1
    float       * ang;                      // (n_ang, H, n_tok) per-token angles (chunked path)
    float       * wstate;                   // (n_chunks, H, n_seqs) x SD (chunked path)
    float       * wdecay;                   // (n_chunks, H, n_seqs)
    int D_qk, R, D_v, H, n_ang;
    int T, n_seqs, n_chunks;
    int64_t n_embd_s, off_K, off_V, off_A;
    float eps, a_floor;
};

enum m3_mode { M3_SERIAL, M3_PHASE1, M3_PHASE3 };

static __device__ __forceinline__ float m3_softplus(float x) {
    return fmaxf(x, 0.0f) + logf(expf(-fabsf(x)) + 1.0f);
}

struct m3_coefs { float dt, alpha, beta, gamma; };

static __device__ __forceinline__ m3_coefs m3_token_coefs(const m3_params & P, int h, int64_t tok) {
    const float * ps = P.pstat + tok*P.pstat_s + (int64_t) h*(2*P.D_v + 3);
    m3_coefs c;
    c.dt    = m3_softplus(ps[2*P.D_v + 0] + P.misc[h]);
    const float a    = fminf(-m3_softplus(ps[2*P.D_v + 1]), -P.a_floor);
    c.alpha = expf(a*c.dt);
    const float trap = 1.0f/(1.0f + expf(-ps[2*P.D_v + 2]));
    c.beta  = (1.0f - trap)*c.dt*c.alpha;
    c.gamma = trap*c.dt;
    return c;
}

static __device__ __forceinline__ float m3_wrap(float a) {
    return a - 6.283185307179586f*rintf(a*0.15915494309189535f);
}

// builds k (and q, z if wanted) for token `tok` into shared memory from the raw
// projections; cs/sn must hold the token's rotary cos/sin, rms the 2R scales.
// v always. Caller syncs before and after.
static __device__ __forceinline__ void m3_build_operands(
        const m3_params & P, int h, int64_t tok, const float * cs, const float * sn, const float * rms,
        float * k, float * q, float * v, float * z, bool want_qz) {
    const int tid = threadIdx.x, nth = blockDim.x;
    const int D_qk = P.D_qk, R = P.R, D_v = P.D_v;
    const float * pd     = P.pdyn + tok*P.pdyn_s;
    const float * bias_h = P.bias + (int64_t) h*2*R*D_qk;
    const int     half   = D_qk/2, quarter = D_qk/4;

    const int n_kq = (want_qz ? 2 : 1)*R*D_qk;
    for (int j = tid; j < n_kq; j += nth) {
        const int g = j / D_qk;
        const int d = j % D_qk;
        const float * src = pd + g*D_qk;
        const float * w   = P.norms + (g < R ? 0 : D_qk);
        const float * b   = bias_h + g*D_qk;
        const float   sc  = rms[g];
        float out;
        if (d < quarter) {
            const float v0 = src[d]*sc*w[d] + b[d];
            const float v2 = src[d + half]*sc*w[d + half] + b[d + half];
            out = v0*cs[d] - v2*sn[d];
        } else if (d >= half && d < half + quarter) {
            const int i = d - half;
            const float v0 = src[i]*sc*w[i] + b[i];
            const float v2 = src[d]*sc*w[d] + b[d];
            out = v0*sn[i] + v2*cs[i];
        } else {
            out = src[d]*sc*w[d] + b[d];
        }
        if (g < R) {
            k[j] = out;
        } else {
            q[j - R*D_qk] = out;
        }
    }

    const float * ps   = P.pstat + tok*P.pstat_s + (int64_t) h*(2*D_v + 3);
    const float * mx_h = P.mxz + (int64_t) h*2*R*D_v;
    for (int j = tid; j < R*D_v; j += nth) {
        const int p = j % D_v;
        v[j] = ps[D_v + p]*mx_h[j];
        if (want_qz) {
            z[j] = ps[p]*mx_h[R*D_v + j];
        }
    }
}

// rms scales of the 2R groups of B/C of token `tok` (one warp per group)
static __device__ __forceinline__ void m3_rms(const m3_params & P, int64_t tok, float * rms, int n_groups) {
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE, nw = blockDim.x / WARP_SIZE;
    const float * pd = P.pdyn + tok*P.pdyn_s;
    for (int g = warp; g < n_groups; g += nw) {
        float s = 0.0f;
        for (int d = lane; d < P.D_qk; d += WARP_SIZE) {
            const float x = pd[g*P.D_qk + d];
            s += x*x;
        }
        s = warp_reduce_sum(s);
        if (lane == 0) {
            rms[g] = rsqrtf(s/P.D_qk + P.eps);
        }
    }
}

template <m3_mode mode, bool state_in_smem, int RT>
static __global__ void __launch_bounds__(M3_NTHREADS) mamba3_mimo_scan_cuda(const m3_params P) {
    const int h   = blockIdx.y;
    const int seq = blockIdx.z;
    const int nc  = blockIdx.x;

    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid % WARP_SIZE, warp = tid / WARP_SIZE, nw = nth / WARP_SIZE;

    constexpr int R = RT;
    const int D_qk = P.D_qk, D_v = P.D_v, H = P.H, n_ang = P.n_ang;
    const int64_t SD = (int64_t) D_qk*D_v;

    const int t0 = mode == M3_SERIAL ? 0   : nc*M3_CHUNK;
    const int tc = mode == M3_SERIAL ? P.T : min(M3_CHUNK, P.T - t0);
    const int64_t tok0 = (int64_t) seq*P.T;

    const float * s_in  = P.s_in  + seq*P.s_in_s;
    float       * s_out = P.s_out + seq*P.s_out_s;
    float       * W     = mode == M3_SERIAL ? nullptr : P.wstate + (((int64_t) seq*H + h)*P.n_chunks + nc)*SD;

    // shared: [ k x2 | q | v x2 | z | cs | sn | ang | rms | (S) ]
    extern __shared__ float smem[];
    float * kb  = smem;
    float * qb  = kb  + 2*R*D_qk;
    float * vb  = qb  + R*D_qk;
    float * zb  = vb  + 2*R*D_v;
    float * cs  = zb  + R*D_v;
    float * sn  = cs  + n_ang;
    float * ang = sn  + n_ang;
    float * rms = ang + n_ang;
    float * S   = state_in_smem ? rms + 2*R : (mode == M3_SERIAL ? s_out + h*SD : W);

    // initial state
    for (int64_t i = tid; i < SD; i += nth) {
        float s0;
        if constexpr (mode == M3_SERIAL) {
            s0 = s_in[h*SD + i];
        } else if constexpr (mode == M3_PHASE1) {
            s0 = 0.0f;
        } else {
            s0 = W[i]; // carry-in left by phase 2
        }
        if (state_in_smem || mode != M3_PHASE3) {
            S[i] = s0;
        }
    }
    if constexpr (mode == M3_SERIAL) {
        for (int i = tid; i < n_ang; i += nth) {
            ang[i] = s_in[P.off_A + h*n_ang + i];
        }
    }

    // previous token's k/v: from the cache at the sequence start, else rebuilt
    int prv = 1;
    if (t0 == 0) {
        for (int j = tid; j < R*D_qk; j += nth) {
            kb[prv*R*D_qk + j] = s_in[P.off_K + (int64_t) h*R*D_qk + j];
        }
        for (int j = tid; j < R*D_v; j += nth) {
            vb[prv*R*D_v + j] = s_in[P.off_V + (int64_t) h*R*D_v + j];
        }
    } else {
        const int64_t ptok = tok0 + t0 - 1;
        for (int i = tid; i < n_ang; i += nth) {
            float s, c;
            sincosf(__ldg(&P.ang[(ptok*H + h)*n_ang + i]), &s, &c);
            cs[i] = c; sn[i] = s;
        }
        m3_rms(P, ptok, rms, R);
        __syncthreads();
        m3_build_operands(P, h, ptok, cs, sn, rms, kb + prv*R*D_qk, qb, vb + prv*R*D_v, zb, false);
    }
    __syncthreads();

    float pa = 1.0f; // running alpha product (phase 1)
    const float D_h = P.misc[H + h];
    const float * mo_h = P.mimo_o + (int64_t) h*R*D_v;

    for (int t = 0; t < tc; ++t) {
        const int64_t tok = tok0 + t0 + t;
        const int cur = prv ^ 1;
        constexpr bool want_y = mode != M3_PHASE1;

        const m3_coefs cf = m3_token_coefs(P, h, tok);
        if constexpr (mode == M3_PHASE1) {
            pa *= cf.alpha;
        }

        for (int i = tid; i < n_ang; i += nth) {
            float a;
            if constexpr (mode == M3_SERIAL) {
                a = m3_wrap(ang[i] + tanhf(P.pdyn[tok*P.pdyn_s + 2*R*D_qk + i])*3.14159265358979f*cf.dt);
                ang[i] = a;
            } else {
                a = __ldg(&P.ang[(tok*H + h)*n_ang + i]);
            }
            float s, c;
            sincosf(a, &s, &c);
            cs[i] = c; sn[i] = s;
        }
        m3_rms(P, tok, rms, want_y ? 2*R : R);
        __syncthreads();

        float * k_c = kb + cur*R*D_qk;
        float * v_c = vb + cur*R*D_v;
        const float * k_p = kb + prv*R*D_qk;
        const float * v_p = vb + prv*R*D_v;
        m3_build_operands(P, h, tok, cs, sn, rms, k_c, qb, v_c, zb, want_y);
        __syncthreads();

        for (int p = warp; p < D_v; p += nw) {
            float vc[R], vp[R], acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                vc[r]  = v_c[r*D_v + p];
                vp[r]  = v_p[r*D_v + p];
                acc[r] = 0.0f;
            }
            float * Sp = S + (int64_t) p*D_qk;
            for (int d = lane; d < D_qk; d += WARP_SIZE) {
                float c_acc = 0.0f, p_acc = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    c_acc += vc[r]*k_c[r*D_qk + d];
                    p_acc += vp[r]*k_p[r*D_qk + d];
                }
                const float s = cf.alpha*Sp[d] + cf.beta*p_acc + cf.gamma*c_acc;
                Sp[d] = s;
                if constexpr (want_y) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        acc[r] += s*qb[r*D_qk + d];
                    }
                }
            }
            if constexpr (want_y) {
                float yp = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float o  = warp_reduce_sum(acc[r]) + D_h*vc[r];
                    const float zv = zb[r*D_v + p];
                    yp += mo_h[r*D_v + p]*o*(zv/(1.0f + expf(-zv)));
                }
                if (lane == 0) {
                    P.y[(tok*H + h)*D_v + p] = yp;
                }
            }
        }
        __syncthreads();
        prv = cur;
    }

    const bool last = mode == M3_SERIAL || (mode == M3_PHASE3 && nc == P.n_chunks - 1);
    if (last) {
        // k/v of the last token feed the next ubatch's trapezoid term
        for (int j = tid; j < R*D_qk; j += nth) {
            s_out[P.off_K + (int64_t) h*R*D_qk + j] = kb[prv*R*D_qk + j];
        }
        for (int j = tid; j < R*D_v; j += nth) {
            s_out[P.off_V + (int64_t) h*R*D_v + j] = vb[prv*R*D_v + j];
        }
    }
    if constexpr (mode == M3_SERIAL) {
        for (int i = tid; i < n_ang; i += nth) {
            s_out[P.off_A + h*n_ang + i] = ang[i];
        }
        if constexpr (state_in_smem) {
            for (int64_t i = tid; i < SD; i += nth) {
                s_out[h*SD + i] = S[i];
            }
        }
    } else if constexpr (mode == M3_PHASE3) {
        if (nc == P.n_chunks - 1) {
            for (int64_t i = tid; i < SD; i += nth) {
                s_out[h*SD + i] = S[i];
            }
        }
    }
    if constexpr (mode == M3_PHASE1) {
        if constexpr (state_in_smem) {
            for (int64_t i = tid; i < SD; i += nth) {
                W[i] = S[i];
            }
        }
        if (tid == 0) {
            P.wdecay[((int64_t) seq*H + h)*P.n_chunks + nc] = pa;
        }
    }
}

// per-token wrapped rotary phases for the chunked path; one block per (head, seq)
static __global__ void mamba3_mimo_angles_cuda(const m3_params P) {
    const int h = blockIdx.x, seq = blockIdx.y;
    const int R = P.R, D_qk = P.D_qk, H = P.H, n_ang = P.n_ang;
    const float dt_bias = __ldg(&P.misc[h]);
    for (int i = threadIdx.x; i < n_ang; i += blockDim.x) {
        float a = __ldg(&P.s_in[seq*P.s_in_s + P.off_A + h*n_ang + i]);
        for (int t = 0; t < P.T; ++t) {
            const int64_t tok = (int64_t) seq*P.T + t;
            const float * ps  = P.pstat + tok*P.pstat_s + (int64_t) h*(2*P.D_v + 3);
            const float   dt  = m3_softplus(__ldg(&ps[2*P.D_v]) + dt_bias);
            a = m3_wrap(a + tanhf(__ldg(&P.pdyn[tok*P.pdyn_s + 2*R*D_qk + i]))*3.14159265358979f*dt);
            P.ang[(tok*H + h)*n_ang + i] = a;
        }
        P.s_out[seq*P.s_out_s + P.off_A + h*n_ang + i] = a;
    }
}

// phase 2: serial scan over chunks 0..n_chunks-2; leaves each chunk's carry-in in W
// (final chunk carry-in is written to W[n_chunks-1] so PHASE3 produces S_out directly)
template <bool state_in_smem>
static __global__ void mamba3_mimo_chunk_scan_cuda(const m3_params P) {
    const int h = blockIdx.x, seq = blockIdx.y;
    const int tid = threadIdx.x, nth = blockDim.x;
    const int64_t SD = (int64_t) P.D_qk*P.D_v;

    const float * S_init = P.s_in  + seq*P.s_in_s  + h*SD;
    float       * W_last = P.wstate + (((int64_t) seq*P.H + h)*P.n_chunks + (P.n_chunks - 1))*SD;

    extern __shared__ float smem[];
    float * S = state_in_smem ? smem : W_last;

    for (int64_t i = tid; i < SD; i += nth) {
        S[i] = __ldg(S_init + i);
    }
    for (int n = 0; n < P.n_chunks - 1; ++n) {
        float     * W  = P.wstate + (((int64_t) seq*P.H + h)*P.n_chunks + n)*SD;
        const float pa = P.wdecay[((int64_t) seq*P.H + h)*P.n_chunks + n];
        for (int64_t i = tid; i < SD; i += nth) {
            const float local = W[i];
            W[i] = S[i];
            S[i] = pa*S[i] + local;
        }
    }
    if constexpr (state_in_smem) {
        for (int64_t i = tid; i < SD; i += nth) {
            W_last[i] = S[i];
        }
    }
}

#if !defined(GGML_USE_HIP)
// Register-resident fast path when D_v == 64 && D_qk == 128 && WARP_SIZE == 32 (Olala-7A1B):
// Keeps S in float4 s_reg[8] (0 bytes SMEM for S -> 4 blocks/SM on sm_86/89/90).
template <m3_mode mode, int RT>
static __global__ void __launch_bounds__(M3_NTHREADS, 4)
mamba3_mimo_scan_reg_64x128_cuda(const m3_params P) {
    constexpr int R = RT;
    constexpr int D_qk = 128, D_v = 64, ROWS_PER_WARP = 8;
    const int h   = blockIdx.y;
    const int seq = blockIdx.z;
    const int nc  = blockIdx.x;
    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int H = P.H, n_ang = P.n_ang;
    const int64_t SD = (int64_t) D_qk * D_v;

    const int t0 = (mode == M3_SERIAL) ? 0 : nc * M3_CHUNK;
    const int tc = (mode == M3_SERIAL) ? P.T : min(M3_CHUNK, P.T - t0);
    const int64_t tok0 = (int64_t) seq * P.T;

    const float * s_in  = P.s_in  + seq * P.s_in_s;
    float       * s_out = P.s_out + seq * P.s_out_s;
    float       * W     = (mode == M3_SERIAL) ? nullptr : P.wstate + (((int64_t) seq * H + h) * P.n_chunks + nc) * SD;

    extern __shared__ float smem[];
    float * kb  = smem;
    float * qb  = kb  + 2 * R * D_qk;
    float * vb  = qb  + R * D_qk;
    float * zb  = vb  + 2 * R * D_v;
    float * cs  = zb  + R * D_v;
    float * sn  = cs  + n_ang;
    float * ang = sn  + n_ang;
    float * rms = ang + n_ang;

    float4 s_reg[ROWS_PER_WARP];
#pragma unroll
    for (int rp = 0; rp < ROWS_PER_WARP; ++rp) {
        const int p = warp + rp * 8;
        if constexpr (mode == M3_SERIAL) {
            s_reg[rp] = __ldg(reinterpret_cast<const float4 *>(s_in + (int64_t) h * SD + p * D_qk) + lane);
        } else if constexpr (mode == M3_PHASE1) {
            s_reg[rp] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            s_reg[rp] = __ldg(reinterpret_cast<const float4 *>(W + p * D_qk) + lane);
        }
    }
    if constexpr (mode == M3_SERIAL) {
        for (int i = tid; i < n_ang; i += nth) ang[i] = __ldg(&s_in[P.off_A + h * n_ang + i]);
    }

    int prv = 1;
    if (t0 == 0) {
        for (int j = tid; j < R * D_qk; j += nth) kb[prv * R * D_qk + j] = __ldg(&s_in[P.off_K + (int64_t) h * R * D_qk + j]);
        for (int j = tid; j < R * D_v;  j += nth) vb[prv * R * D_v  + j] = __ldg(&s_in[P.off_V + (int64_t) h * R * D_v  + j]);
    } else {
        const int64_t ptok = tok0 + t0 - 1;
        for (int i = tid; i < n_ang; i += nth) {
            float s, c;
            sincosf(__ldg(&P.ang[(ptok * H + h) * n_ang + i]), &s, &c);
            cs[i] = c; sn[i] = s;
        }
        m3_rms(P, ptok, rms, R);
        __syncthreads();
        m3_build_operands(P, h, ptok, cs, sn, rms, kb + prv * R * D_qk, qb, vb + prv * R * D_v, zb, false);
    }
    __syncthreads();

    float pa = 1.0f;
    const float D_h = __ldg(&P.misc[H + h]);
    const float * mo_h = P.mimo_o + (int64_t) h * R * D_v;

    for (int t = 0; t < tc; ++t) {
        const int64_t tok = tok0 + t0 + t;
        const int cur = prv ^ 1;
        constexpr bool want_y = (mode != M3_PHASE1);

        const m3_coefs cf = m3_token_coefs(P, h, tok);
        if constexpr (mode == M3_PHASE1) pa *= cf.alpha;

        for (int i = tid; i < n_ang; i += nth) {
            float a;
            if constexpr (mode == M3_SERIAL) {
                a = m3_wrap(ang[i] + tanhf(__ldg(&P.pdyn[tok * P.pdyn_s + 2 * R * D_qk + i])) * 3.14159265358979f * cf.dt);
                ang[i] = a;
            } else {
                a = __ldg(&P.ang[(tok * H + h) * n_ang + i]);
            }
            float s, c;
            sincosf(a, &s, &c);
            cs[i] = c; sn[i] = s;
        }
        m3_rms(P, tok, rms, want_y ? 2 * R : R);
        __syncthreads();

        float * k_c = kb + cur * R * D_qk;
        float * v_c = vb + cur * R * D_v;
        const float * k_p = kb + prv * R * D_qk;
        const float * v_p = vb + prv * R * D_v;
        m3_build_operands(P, h, tok, cs, sn, rms, k_c, qb, v_c, zb, want_y);
        __syncthreads();

        float4 kc4[R], kp4[R], qv4[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            kc4[r] = reinterpret_cast<const float4 *>(k_c + r * D_qk)[lane];
            kp4[r] = reinterpret_cast<const float4 *>(k_p + r * D_qk)[lane];
            if constexpr (want_y) {
                qv4[r] = reinterpret_cast<const float4 *>(qb + r * D_qk)[lane];
            }
        }

#pragma unroll
        for (int rp = 0; rp < ROWS_PER_WARP; ++rp) {
            const int p = warp + rp * 8;
            float4 c_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 p_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float vc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                vc[r] = v_c[r * D_v + p];
                const float vp_r = v_p[r * D_v + p];
                c_acc.x = fmaf(vc[r], kc4[r].x, c_acc.x); c_acc.y = fmaf(vc[r], kc4[r].y, c_acc.y);
                c_acc.z = fmaf(vc[r], kc4[r].z, c_acc.z); c_acc.w = fmaf(vc[r], kc4[r].w, c_acc.w);
                p_acc.x = fmaf(vp_r,  kp4[r].x, p_acc.x); p_acc.y = fmaf(vp_r,  kp4[r].y, p_acc.y);
                p_acc.z = fmaf(vp_r,  kp4[r].z, p_acc.z); p_acc.w = fmaf(vp_r,  kp4[r].w, p_acc.w);
            }
            float4 s = s_reg[rp];
            s.x = fmaf(cf.alpha, s.x, fmaf(cf.beta, p_acc.x, cf.gamma * c_acc.x));
            s.y = fmaf(cf.alpha, s.y, fmaf(cf.beta, p_acc.y, cf.gamma * c_acc.y));
            s.z = fmaf(cf.alpha, s.z, fmaf(cf.beta, p_acc.z, cf.gamma * c_acc.z));
            s.w = fmaf(cf.alpha, s.w, fmaf(cf.beta, p_acc.w, cf.gamma * c_acc.w));
            s_reg[rp] = s;

            if constexpr (want_y) {
                float yp = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float dot = fmaf(s.x, qv4[r].x, fmaf(s.y, qv4[r].y, fmaf(s.z, qv4[r].z, s.w * qv4[r].w)));
                    const float o   = warp_reduce_sum(dot) + D_h * vc[r];
                    const float zv  = zb[r * D_v + p];
                    yp = fmaf(__ldg(&mo_h[r * D_v + p]) * o, zv / (1.0f + expf(-zv)), yp);
                }
                if (lane == 0) P.y[(tok * H + h) * D_v + p] = yp;
            }
        }
        __syncthreads();
        prv = cur;
    }

    const bool is_last = (mode == M3_SERIAL) || (mode == M3_PHASE3 && nc == P.n_chunks - 1);
    if (is_last) {
#pragma unroll
        for (int rp = 0; rp < ROWS_PER_WARP; ++rp) {
            const int p = warp + rp * 8;
            reinterpret_cast<float4 *>(s_out + (int64_t) h * SD + p * D_qk)[lane] = s_reg[rp];
        }
        for (int j = tid; j < R * D_qk; j += nth) s_out[P.off_K + (int64_t) h * R * D_qk + j] = kb[prv * R * D_qk + j];
        for (int j = tid; j < R * D_v;  j += nth) s_out[P.off_V + (int64_t) h * R * D_v  + j] = vb[prv * R * D_v  + j];
        if constexpr (mode == M3_SERIAL) {
            for (int i = tid; i < n_ang; i += nth) s_out[P.off_A + h * n_ang + i] = ang[i];
        }
    } else if constexpr (mode == M3_PHASE1) {
#pragma unroll
        for (int rp = 0; rp < ROWS_PER_WARP; ++rp) {
            const int p = warp + rp * 8;
            reinterpret_cast<float4 *>(W + p * D_qk)[lane] = s_reg[rp];
        }
        if (tid == 0) P.wdecay[((int64_t) seq * H + h) * P.n_chunks + nc] = pa;
    }
}
#endif // !defined(GGML_USE_HIP)


// ============================================================================
// Dedicated T == 1 Decode Kernel: Zero SMEM State Staging + Direct float4 RMW
// ============================================================================

template <int RT>
static __global__ void __launch_bounds__(M3_NTHREADS, (RT <= 4) ? 4 : 2)
mamba3_mimo_decode_t1_cuda(const m3_params P) {
    const int tile    = blockIdx.x;
    const int n_tiles = gridDim.x;
    const int h       = blockIdx.y;
    const int seq     = blockIdx.z;

    const int tid  = threadIdx.x;
    const int nth  = blockDim.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int nw   = nth / WARP_SIZE;

    constexpr int R = RT;
    const int D_qk  = P.D_qk;
    const int D_v   = P.D_v;
    const int H     = P.H;
    const int n_ang = P.n_ang;
    const int64_t SD  = (int64_t) D_qk * D_v;
    const int64_t tok = (int64_t) seq; // P.T == 1

    const float * s_in  = P.s_in  + seq * P.s_in_s;
    float       * s_out = P.s_out + seq * P.s_out_s;

    // Shared memory holds ONLY operands (~9.6 KB for R=4, D_qk=128, D_v=64) -> 4 blocks/SM in default 48 KB
    // Layout: [ k_c | k_p | q | v_c | v_p | z | cs | sn | ang | rms ]
    extern __shared__ float smem[];
    float * k_c = smem;
    float * k_p = k_c + R * D_qk;
    float * qb  = k_p + R * D_qk;
    float * v_c = qb  + R * D_qk;
    float * v_p = v_c + R * D_v;
    float * zb  = v_p + R * D_v;
    float * cs  = zb  + R * D_v;
    float * sn  = cs  + n_ang;
    float * ang = sn  + n_ang;
    float * rms = ang + n_ang;

    const m3_coefs cf = m3_token_coefs(P, h, tok);

    // Load previous k_p, v_p and update angles in parallel
    for (int j = tid; j < R * D_qk; j += nth) {
        k_p[j] = __ldg(s_in + P.off_K + (int64_t) h * R * D_qk + j);
    }
    for (int j = tid; j < R * D_v; j += nth) {
        v_p[j] = __ldg(s_in + P.off_V + (int64_t) h * R * D_v + j);
    }
    for (int i = tid; i < n_ang; i += nth) {
        const float a0 = __ldg(s_in + P.off_A + (int64_t) h * n_ang + i);
        const float a  = m3_wrap(a0 + tanhf(__ldg(P.pdyn + tok * P.pdyn_s + 2 * R * D_qk + i)) * 3.14159265358979f * cf.dt);
        ang[i] = a;
        float s, c;
        sincosf(a, &s, &c);
        cs[i] = c;
        sn[i] = s;
    }

    m3_rms(P, tok, rms, 2 * R);
    __syncthreads();

    m3_build_operands(P, h, tok, cs, sn, rms, k_c, qb, v_c, zb, true);
    __syncthreads();

    const float   D_h  = __ldg(P.misc + H + h);
    const float * mo_h = P.mimo_o + (int64_t) h * R * D_v;

    const int p_begin = (D_v * tile) / n_tiles;
    const int p_end   = (D_v * (tile + 1)) / n_tiles;

    const float * s_in_h  = s_in  + (int64_t) h * SD;
    float       * s_out_h = s_out + (int64_t) h * SD;
    const bool vec4_ok = ((D_qk & 3) == 0) &&
                         ((reinterpret_cast<uintptr_t>(s_in_h)  & 15u) == 0u) &&
                         ((reinterpret_cast<uintptr_t>(s_out_h) & 15u) == 0u);

    for (int p = p_begin + warp; p < p_end; p += nw) {
        float vc[R], vp[R], acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            vc[r]  = v_c[r * D_v + p];
            vp[r]  = v_p[r * D_v + p];
            acc[r] = 0.0f;
        }

        const float * sp_in  = s_in_h  + (int64_t) p * D_qk;
        float       * sp_out = s_out_h + (int64_t) p * D_qk;

        if (vec4_ok) {
            const float4 * sp_in4  = reinterpret_cast<const float4 *>(sp_in);
            float4       * sp_out4 = reinterpret_cast<float4 *>(sp_out);
            const int d4_max = D_qk >> 2;

            for (int d4 = lane; d4 < d4_max; d4 += WARP_SIZE) {
                // Single 128-bit coalesced load directly from global state (zero SMEM staging)
                const float4 s_old = __ldg(sp_in4 + d4);
                float4 c_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 p_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float4 kc = reinterpret_cast<const float4 *>(k_c + r * D_qk)[d4];
                    const float4 kp = reinterpret_cast<const float4 *>(k_p + r * D_qk)[d4];
                    c_acc.x = fmaf(vc[r], kc.x, c_acc.x);
                    c_acc.y = fmaf(vc[r], kc.y, c_acc.y);
                    c_acc.z = fmaf(vc[r], kc.z, c_acc.z);
                    c_acc.w = fmaf(vc[r], kc.w, c_acc.w);
                    p_acc.x = fmaf(vp[r], kp.x, p_acc.x);
                    p_acc.y = fmaf(vp[r], kp.y, p_acc.y);
                    p_acc.z = fmaf(vp[r], kp.z, p_acc.z);
                    p_acc.w = fmaf(vp[r], kp.w, p_acc.w);
                }

                float4 s_new;
                s_new.x = fmaf(cf.alpha, s_old.x, fmaf(cf.beta, p_acc.x, cf.gamma * c_acc.x));
                s_new.y = fmaf(cf.alpha, s_old.y, fmaf(cf.beta, p_acc.y, cf.gamma * c_acc.y));
                s_new.z = fmaf(cf.alpha, s_old.z, fmaf(cf.beta, p_acc.z, cf.gamma * c_acc.z));
                s_new.w = fmaf(cf.alpha, s_old.w, fmaf(cf.beta, p_acc.w, cf.gamma * c_acc.w));

                // Single 128-bit coalesced write directly to output state
                sp_out4[d4] = s_new;

#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float4 qv = reinterpret_cast<const float4 *>(qb + r * D_qk)[d4];
                    acc[r] = fmaf(s_new.x, qv.x, fmaf(s_new.y, qv.y, fmaf(s_new.z, qv.z, fmaf(s_new.w, qv.w, acc[r]))));
                }
            }
        } else {
            for (int d = lane; d < D_qk; d += WARP_SIZE) {
                float c_acc = 0.0f, p_acc = 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    c_acc = fmaf(vc[r], k_c[r * D_qk + d], c_acc);
                    p_acc = fmaf(vp[r], k_p[r * D_qk + d], p_acc);
                }
                const float s = fmaf(cf.alpha, __ldg(sp_in + d), fmaf(cf.beta, p_acc, cf.gamma * c_acc));
                sp_out[d] = s;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc[r] = fmaf(s, qb[r * D_qk + d], acc[r]);
                }
            }
        }

        float yp = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float o  = warp_reduce_sum(acc[r]) + D_h * vc[r];
            const float zv = zb[r * D_v + p];
            yp = fmaf(__ldg(mo_h + r * D_v + p) * o, zv / (1.0f + expf(-zv)), yp);
        }
        if (lane == 0) {
            P.y[(tok * H + h) * D_v + p] = yp;
        }
    }

    // Write metadata state tails: when P.tile_done != nullptr (in-place / aliased n_tiles > 1),
    // the last tile to finish its p loop writes off_K/V/A after all tiles have read them.
    bool write_meta = (tile == 0);
    if (P.tile_done != nullptr) {
        __shared__ int s_is_last;
        __syncthreads();
        if (tid == 0) {
            const int done = atomicAdd(&P.tile_done[seq * H + h], 1) + 1;
            s_is_last = (done == n_tiles) ? 1 : 0;
        }
        __syncthreads();
        write_meta = (s_is_last != 0);
    }
    if (write_meta) {
        for (int j = tid; j < R * D_qk; j += nth) {
            s_out[P.off_K + (int64_t) h * R * D_qk + j] = k_c[j];
        }
        for (int j = tid; j < R * D_v; j += nth) {
            s_out[P.off_V + (int64_t) h * R * D_v + j] = v_c[j];
        }
        for (int i = tid; i < n_ang; i += nth) {
            s_out[P.off_A + (int64_t) h * n_ang + i] = ang[i];
        }
    }
}

static void m3_launch_decode_t1(ggml_backend_cuda_context & ctx, m3_params & P, size_t smem_ops, cudaStream_t stream) {
    // Check alias safely using uintptr_t without undefined behavior
    const uintptr_t u_s_in    = reinterpret_cast<uintptr_t>(P.s_in);
    const uintptr_t u_s_out   = reinterpret_cast<uintptr_t>(P.s_out);
    const uintptr_t in_bytes  = static_cast<uintptr_t>(P.n_seqs) * static_cast<uintptr_t>(P.s_in_s)  * sizeof(float);
    const uintptr_t out_bytes = static_cast<uintptr_t>(P.n_seqs) * static_cast<uintptr_t>(P.s_out_s) * sizeof(float);
    const bool states_alias =
        (u_s_in == u_s_out) ||
        (u_s_in < u_s_out + out_bytes && u_s_out < u_s_in + in_bytes);

    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int max_tiles = std::min(8, P.D_v / 8); // keep >= 1 row per warp (8 warps/block)
    int n_tiles = 1;
    const int64_t base_blocks = (int64_t) P.H * P.n_seqs;
    if (max_tiles >= 2 && base_blocks < (int64_t) nsm * 2) {
        for (int t = 2; t <= max_tiles; t *= 2) {
            n_tiles = t;
            if (base_blocks * t >= (int64_t) nsm * 2) {
                break;
            }
        }
    }

    ggml_cuda_pool_alloc<int32_t> tile_done_alloc;
    if (states_alias && n_tiles > 1) {
        const size_t n_td = static_cast<size_t>(P.n_seqs) * static_cast<size_t>(P.H);
        tile_done_alloc.alloc(ctx.pool(), n_td);
        CUDA_CHECK(cudaMemsetAsync(tile_done_alloc.get(), 0, n_td * sizeof(int32_t), stream));
        P.tile_done = tile_done_alloc.get();
    } else {
        P.tile_done = nullptr;
    }

    const dim3 grid(n_tiles, P.H, P.n_seqs);

    switch (P.R) {
        case 1: mamba3_mimo_decode_t1_cuda<1><<<grid, M3_NTHREADS, smem_ops, stream>>>(P); break;
        case 2: mamba3_mimo_decode_t1_cuda<2><<<grid, M3_NTHREADS, smem_ops, stream>>>(P); break;
        case 4: mamba3_mimo_decode_t1_cuda<4><<<grid, M3_NTHREADS, smem_ops, stream>>>(P); break;
        case 8: mamba3_mimo_decode_t1_cuda<8><<<grid, M3_NTHREADS, smem_ops, stream>>>(P); break;
        default: GGML_ABORT("m3_launch_decode_t1: unsupported R=%d", P.R);
    }
}

template <m3_mode mode, int RT>
static void m3_launch_r(const m3_params & P, dim3 grid, size_t smem_ops, size_t smem_state, bool in_smem, cudaStream_t stream) {
#if !defined(GGML_USE_HIP)
    const bool aligned16 =
        ((reinterpret_cast<uintptr_t>(P.s_in)  & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(P.s_out) & 15u) == 0u) &&
        ((P.s_in_s  & 3) == 0) && ((P.s_out_s & 3) == 0) &&
        (P.wstate == nullptr || (reinterpret_cast<uintptr_t>(P.wstate) & 15u) == 0u);
    if (P.D_v == 64 && P.D_qk == 128 && aligned16 &&
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == 32) {
        mamba3_mimo_scan_reg_64x128_cuda<mode, RT><<<grid, M3_NTHREADS, smem_ops, stream>>>(P);
        return;
    }
#endif
    if (in_smem) {
        CUDA_SET_SHARED_MEMORY_LIMIT((mamba3_mimo_scan_cuda<mode, true, RT>), smem_ops + smem_state);
        mamba3_mimo_scan_cuda<mode, true, RT><<<grid, M3_NTHREADS, smem_ops + smem_state, stream>>>(P);
    } else {
        mamba3_mimo_scan_cuda<mode, false, RT><<<grid, M3_NTHREADS, smem_ops, stream>>>(P);
    }
}

template <m3_mode mode>
static void m3_launch(const m3_params & P, dim3 grid, size_t smem_ops, size_t smem_state, bool in_smem, cudaStream_t stream) {
    switch (P.R) {
        case 1: m3_launch_r<mode, 1>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        case 2: m3_launch_r<mode, 2>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        case 4: m3_launch_r<mode, 4>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        case 8: m3_launch_r<mode, 8>(P, grid, smem_ops, smem_state, in_smem, stream); break;
        default: GGML_ABORT("mamba3_mimo: unsupported MIMO rank %d", P.R);
    }
}

void ggml_cuda_op_mamba3_mimo(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float * fused_s_out,
        int64_t fused_s_out_stride) {
    const ggml_tensor * pdyn   = dst->src[0];
    const ggml_tensor * pstat  = dst->src[1];
    const ggml_tensor * state  = dst->src[7];

    m3_params P = {};
    P.D_qk  = (int) dst->src[2]->ne[0];
    P.R     = (int) dst->src[4]->ne[1];
    P.H     = (int) dst->src[4]->ne[2];
    P.D_v   = (int) dst->src[4]->ne[0];
    P.n_ang = P.D_qk/4;
    GGML_ASSERT(P.R <= M3_MAX_R);

    const int64_t n_tok = pdyn->ne[1];
    P.n_seqs = (int) state->ne[1];
    P.T      = (int) (n_tok / P.n_seqs);

    for (int i = 2; i < 7; ++i) {
        GGML_ASSERT(ggml_is_contiguous(dst->src[i]));
    }
    GGML_ASSERT(pdyn->nb[0] == sizeof(float) && pstat->nb[0] == sizeof(float) && state->nb[0] == sizeof(float));

    P.pdyn   = (const float *) pdyn->data;   P.pdyn_s  = pdyn->nb[1]/sizeof(float);
    P.pstat  = (const float *) pstat->data;  P.pstat_s = pstat->nb[1]/sizeof(float);
    P.bias   = (const float *) dst->src[2]->data;
    P.mxz    = (const float *) dst->src[3]->data;
    P.mimo_o = (const float *) dst->src[4]->data;
    P.norms  = (const float *) dst->src[5]->data;
    P.misc   = (const float *) dst->src[6]->data;
    P.s_in   = (const float *) state->data;  P.s_in_s  = state->nb[1]/sizeof(float);

    P.off_K    = (int64_t) P.H*P.D_v*P.D_qk;
    P.off_V    = P.off_K + (int64_t) P.H*P.R*P.D_qk;
    P.off_A    = P.off_V + (int64_t) P.H*P.R*P.D_v;
    P.n_embd_s = P.off_A + (int64_t) P.H*P.n_ang;

    P.y         = (float *) dst->data;
    P.s_out     = fused_s_out ? fused_s_out : (P.y + (int64_t) P.D_v * P.H * n_tok);
    P.s_out_s   = fused_s_out ? fused_s_out_stride : P.n_embd_s;
    P.tile_done = nullptr;

    P.eps     = ggml_get_op_params_f32(dst, 0);
    P.a_floor = ggml_get_op_params_f32(dst, 1);

    cudaStream_t stream = ctx.stream();

    const size_t smem_ops   = (size_t) (3*P.R*P.D_qk + 3*P.R*P.D_v + 3*P.n_ang + 2*P.R)*sizeof(float);
    const size_t smem_state = (size_t) P.D_qk*P.D_v*sizeof(float);
    const size_t smem_max   = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    GGML_ASSERT(smem_ops <= 48*1024);
    const bool in_smem = smem_ops + smem_state <= smem_max;

    if (P.T == 1 && (P.R == 1 || P.R == 2 || P.R == 4 || P.R == 8)) {
        P.n_chunks = 1;
        m3_launch_decode_t1(ctx, P, smem_ops, stream);
        return;
    }

    if (P.T < 2*M3_CHUNK) {
        P.n_chunks = 1;
        m3_launch<M3_SERIAL>(P, dim3(1, P.H, P.n_seqs), smem_ops, smem_state, in_smem, stream);
        return;
    }

    P.n_chunks = (P.T + M3_CHUNK - 1)/M3_CHUNK;
    const int64_t SD = (int64_t) P.D_qk*P.D_v;
    auto align256 = [](size_t n) -> size_t { return (n + 255u) & ~static_cast<size_t>(255u); };
    const size_t sz_ang    = align256((size_t) n_tok*P.H*P.n_ang * sizeof(float));
    const size_t sz_wstate = align256((size_t) P.n_chunks*P.n_seqs*P.H*SD * sizeof(float));
    const size_t sz_wdecay = align256((size_t) P.n_chunks*P.n_seqs*P.H * sizeof(float));
    ggml_cuda_pool_alloc<uint8_t> m3_scratch(ctx.pool(), sz_ang + sz_wstate + sz_wdecay);
    uint8_t * m3_base = m3_scratch.get();
    P.ang    = reinterpret_cast<float *>(m3_base); m3_base += sz_ang;
    P.wstate = reinterpret_cast<float *>(m3_base); m3_base += sz_wstate;
    P.wdecay = reinterpret_cast<float *>(m3_base);

    mamba3_mimo_angles_cuda<<<dim3(P.H, P.n_seqs), WARP_SIZE, 0, stream>>>(P);

    const dim3 grid_p1(P.n_chunks - 1, P.H, P.n_seqs);
    const dim3 grid_p3(P.n_chunks, P.H, P.n_seqs);
    m3_launch<M3_PHASE1>(P, grid_p1, smem_ops, smem_state, in_smem, stream);
    if (smem_state <= smem_max) {
        CUDA_SET_SHARED_MEMORY_LIMIT((mamba3_mimo_chunk_scan_cuda<true>), smem_state);
        mamba3_mimo_chunk_scan_cuda<true><<<dim3(P.H, P.n_seqs), M3_NTHREADS, smem_state, stream>>>(P);
    } else {
        mamba3_mimo_chunk_scan_cuda<false><<<dim3(P.H, P.n_seqs), M3_NTHREADS, 0, stream>>>(P);
    }
    m3_launch<M3_PHASE3>(P, grid_p3, smem_ops, smem_state, in_smem, stream);
}

// ---------------------------------------------------------------------------
// geodesic residual: one block per row


// Fast path: n <= 4096, 16-byte aligned float4 rows, 1x VRAM read + 2 __syncthreads() total
static __global__ void __launch_bounds__(256, 4)
geodesic_cuda_f4(
        const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
        const float * __restrict__ scale, const float * __restrict__ bias, const float inv_depth,
        const int n4, const int64_t ne1, const int64_t ne2,
        const int64_t sx1, const int64_t sx2, const int64_t sx3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3) {
    constexpr int MAX_VECS = 4; // 256 threads * 4 * float4 = up to 4096 floats in registers
    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1, i2 = (ir / ne1) % ne2, i3 = ir / (ne1 * ne2);

    const float4 * xr = reinterpret_cast<const float4 *>(x   + i1 * sx1 + i2 * sx2 + i3 * sx3);
    const float4 * gr = reinterpret_cast<const float4 *>(g   + i1 * sg1 + i2 * sg2 + i3 * sg3);
    float4       * yr = reinterpret_cast<float4 *>      (dst + i1 * sd1 + i2 * sd2 + i3 * sd3);

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int nw   = blockDim.x / WARP_SIZE;

    __shared__ float s_xx[8];
    __shared__ float s_xg[8];
    __shared__ float s_pp[8];

    float4 rx[MAX_VECS];
    float4 rg[MAX_VECS];
    float xx = 0.0f, xg = 0.0f;

    #pragma unroll
    for (int k = 0; k < MAX_VECS; ++k) {
        const int i = tid + k * blockDim.x;
        if (i < n4) {
            const float4 xv = __ldg(xr + i);
            const float4 gv = __ldg(gr + i);
            rx[k] = xv;
            rg[k] = gv;

            xx = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, fmaf(xv.z, xv.z, fmaf(xv.w, xv.w, xx))));
            xg = fmaf(xv.x, gv.x, fmaf(xv.y, gv.y, fmaf(xv.z, gv.z, fmaf(xv.w, gv.w, xg))));
        }
    }

    // Reduction #1 (fused xx + xg -> 1 __syncthreads)
    xx = warp_reduce_sum(xx);
    xg = warp_reduce_sum(xg);
    if (lane == 0) {
        s_xx[warp] = xx;
        s_xg[warp] = xg;
    }
    __syncthreads();

    xx = warp_reduce_sum(lane < nw ? s_xx[lane] : 0.0f);
    xg = warp_reduce_sum(lane < nw ? s_xg[lane] : 0.0f);

    const float x_norm_sq = fmaxf(xx, 1e-12f);
    const float coef      = xg / x_norm_sq;

    // Pass 2: purely from registers (zero VRAM reads, static register indexing)
    float pp = 0.0f;
    #pragma unroll
    for (int k = 0; k < MAX_VECS; ++k) {
        const int i = tid + k * blockDim.x;
        if (i < n4) {
            const float4 xv = rx[k];
            const float4 gv = rg[k];
            const float px = fmaf(-coef, xv.x, gv.x);
            const float py = fmaf(-coef, xv.y, gv.y);
            const float pz = fmaf(-coef, xv.z, gv.z);
            const float pw = fmaf(-coef, xv.w, gv.w);
            pp = fmaf(px, px, fmaf(py, py, fmaf(pz, pz, fmaf(pw, pw, pp))));
        }
    }

    // Reduction #2 (separate s_pp -> 1 __syncthreads, no reuse hazard)
    pp = warp_reduce_sum(pp);
    if (lane == 0) {
        s_pp[warp] = pp;
    }
    __syncthreads();

    pp = warp_reduce_sum(lane < nw ? s_pp[lane] : 0.0f);

    const float pi4      = 0.785398163f;
    const float tan_norm = fmaxf(sqrtf(pp), 1e-8f);
    const float xn       = sqrtf(x_norm_sq);
    float theta = fminf(tan_norm / fmaxf(xn, 1e-6f), pi4);
    theta = fminf((theta * __ldg(scale) + __ldg(bias)) * inv_depth, pi4);

    float sin_t, cos_t;
    sincosf(theta, &sin_t, &cos_t);
    const float c = cos_t;
    const float s = sin_t * xn / tan_norm;

    // Pass 3: compute & write float4 directly from registers (zero local memory spill)
    #pragma unroll
    for (int k = 0; k < MAX_VECS; ++k) {
        const int i = tid + k * blockDim.x;
        if (i < n4) {
            const float4 xv = rx[k];
            const float4 gv = rg[k];
            float4 out;
            out.x = fmaf(xv.x, c, fmaf(-coef, xv.x, gv.x) * s);
            out.y = fmaf(xv.y, c, fmaf(-coef, xv.y, gv.y) * s);
            out.z = fmaf(xv.z, c, fmaf(-coef, xv.z, gv.z) * s);
            out.w = fmaf(xv.w, c, fmaf(-coef, xv.w, gv.w) * s);
            yr[i] = out;
        }
    }
}

static __global__ void geodesic_cuda(
        const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
        const float * __restrict__ scale, const float * __restrict__ bias, const float inv_depth,
        const int n, const int64_t ne1, const int64_t ne2,
        const int64_t sx1, const int64_t sx2, const int64_t sx3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3) {
    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1, i2 = (ir/ne1) % ne2, i3 = ir/(ne1*ne2);
    const float * xr = x   + i1*sx1 + i2*sx2 + i3*sx3;
    const float * gr = g   + i1*sg1 + i2*sg2 + i3*sg3;
    float       * yr = dst + i1*sd1 + i2*sd2 + i3*sd3;

    __shared__ float red[32];
    __shared__ float bc[2];
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE, nw = blockDim.x / WARP_SIZE;

    auto block_sum = [&](float v) -> float {
        v = warp_reduce_sum(v);
        if (lane == 0) red[warp] = v;
        __syncthreads();
        v = lane < nw ? red[lane] : 0.0f;
        v = warp_reduce_sum(v);
        __syncthreads();
        return v;
    };

    float xx = 0.0f, xg = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        xx += xr[i]*xr[i];
        xg += xr[i]*gr[i];
    }
    xx = block_sum(xx);
    xg = block_sum(xg);
    const float x_norm_sq = fmaxf(xx, 1e-12f);
    const float coef      = xg/x_norm_sq;

    float pp = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float gp = gr[i] - coef*xr[i];
        pp += gp*gp;
    }
    pp = block_sum(pp);

    if (tid == 0) {
        const float pi4      = 0.785398163f;
        const float tan_norm = fmaxf(sqrtf(pp), 1e-8f);
        const float xn       = sqrtf(x_norm_sq);
        float theta = fminf(tan_norm/fmaxf(xn, 1e-6f), pi4);
        theta = fminf((theta*scale[0] + bias[0])*inv_depth, pi4);
        bc[0] = cosf(theta);
        bc[1] = sinf(theta)*xn/tan_norm;
    }
    __syncthreads();
    const float c = bc[0], s = bc[1];
    for (int i = tid; i < n; i += blockDim.x) {
        yr[i] = xr[i]*c + (gr[i] - coef*xr[i])*s;
    }
}

void ggml_cuda_op_geodesic(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = dst->src[1];
    GGML_ASSERT(x->nb[0] == sizeof(float) && g->nb[0] == sizeof(float));
    const int64_t nr = ggml_nrows(x);
    const int n = (int) x->ne[0];

    const bool aligned16 =
        ((n & 3) == 0) &&
        ((reinterpret_cast<uintptr_t>(x->data)   & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(g->data)   & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(dst->data) & 15u) == 0u) &&
        ((x->nb[1]   & 15u) == 0u) && ((x->nb[2]   & 15u) == 0u) && ((x->nb[3]   & 15u) == 0u) &&
        ((g->nb[1]   & 15u) == 0u) && ((g->nb[2]   & 15u) == 0u) && ((g->nb[3]   & 15u) == 0u) &&
        ((dst->nb[1] & 15u) == 0u) && ((dst->nb[2] & 15u) == 0u) && ((dst->nb[3] & 15u) == 0u);

    if (aligned16 && n <= 4096) {
        const int nthreads = (n <= 2048) ? 128 : 256;
        geodesic_cuda_f4<<<nr, nthreads, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) g->data, (float *) dst->data,
            (const float *) dst->src[2]->data, (const float *) dst->src[3]->data, ggml_get_op_params_f32(dst, 0),
            n >> 2, x->ne[1], x->ne[2],
            x->nb[1]/4, x->nb[2]/4, x->nb[3]/4,
            g->nb[1]/4, g->nb[2]/4, g->nb[3]/4,
            dst->nb[1]/4, dst->nb[2]/4, dst->nb[3]/4);
        return;
    }

    const int nthreads = n >= 1024 ? 512 : (n >= 256 ? 256 : 128);
    geodesic_cuda<<<nr, nthreads, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) g->data, (float *) dst->data,
        (const float *) dst->src[2]->data, (const float *) dst->src[3]->data, ggml_get_op_params_f32(dst, 0),
        n, x->ne[1], x->ne[2],
        x->nb[1]/4, x->nb[2]/4, x->nb[3]/4,
        g->nb[1]/4, g->nb[2]/4, g->nb[3]/4,
        dst->nb[1]/4, dst->nb[2]/4, dst->nb[3]/4);
}
