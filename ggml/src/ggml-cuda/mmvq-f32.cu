#include "mmvq-f32.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_QKK_64)

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int MMVQ_F32_WARP_SIZE     = 32;
constexpr int MMVQ_F32_NWARPS_SM     = 2;  // 64 threads/block -> eliminates wave tails on 28 SMs
constexpr int MMVQ_F32_THREADS_SM    = MMVQ_F32_WARP_SIZE * MMVQ_F32_NWARPS_SM; // 64
constexpr int MMVQ_F32_NWARPS_LG     = 4;  // 128 threads/block for vocab lm_head
constexpr int MMVQ_F32_THREADS_LG    = MMVQ_F32_WARP_SIZE * MMVQ_F32_NWARPS_LG; // 128
constexpr int MMVQ_F32_VOCAB_MIN_M   = 32768;
constexpr int MMVQ_F32_VOCAB_SMEM_K  = 4096;

static inline size_t align256(size_t n) {
    return (n + 255u) & ~static_cast<size_t>(255u);
}

__device__ __forceinline__ float warp_sum_f32(float val) {
#pragma unroll
    for (int offset = MMVQ_F32_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float act_relu2(float x) {
    const float v = (x > 0.0f) ? x : 0.0f;
    return v * v;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

template <bool X_IS_SMEM>
__device__ __forceinline__ float4 ld_x_f4(const float * __restrict__ p) {
    if constexpr (X_IS_SMEM) {
        return *reinterpret_cast<const float4 *>(p);
    } else {
        return __ldg(reinterpret_cast<const float4 *>(p));
    }
}

template <bool X_IS_SMEM>
__device__ __forceinline__ float2 ld_x_f2(const float * __restrict__ p) {
    if constexpr (X_IS_SMEM) {
        return *reinterpret_cast<const float2 *>(p);
    } else {
        return __ldg(reinterpret_cast<const float2 *>(p));
    }
}

template <bool X_IS_SMEM>
__device__ __forceinline__ float ld_x_f1(const float * __restrict__ p) {
    if constexpr (X_IS_SMEM) {
        return *p;
    } else {
        return __ldg(p);
    }
}

static_assert(sizeof(block_q5_K) == 176 && offsetof(block_q5_K, scales) == 4, "block_q5_K layout mismatch");
static_assert(sizeof(block_q4_K) == 144 && offsetof(block_q4_K, scales) == 4, "block_q4_K layout mismatch");

__device__ __forceinline__ void unpack_k4_scales_u4(
        uint4 hdr,
        int c,
        float & d1,
        float & min1,
        float & d2,
        float & min2) {
    const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(&hdr.x));
    const uint32_t s0 = hdr.y;
    const uint32_t s1 = hdr.z;
    const int shift = (c & 1) << 4;

    uint32_t sc1, sc2, m1, m2;
    if (c < 2) {
        sc1 = (s0 >> (shift + 0)) & 63u;
        sc2 = (s0 >> (shift + 8)) & 63u;
        m1  = (s1 >> (shift + 0)) & 63u;
        m2  = (s1 >> (shift + 8)) & 63u;
    } else {
        const uint32_t s2 = hdr.w;
        sc1 = ((s2 >> (shift +  0)) & 0x0Fu) | (((s0 >> (shift +  6)) & 3u) << 4);
        sc2 = ((s2 >> (shift +  8)) & 0x0Fu) | (((s0 >> (shift + 14)) & 3u) << 4);
        m1  = ((s2 >> (shift +  4)) & 0x0Fu) | (((s1 >> (shift +  6)) & 3u) << 4);
        m2  = ((s2 >> (shift + 12)) & 0x0Fu) | (((s1 >> (shift + 14)) & 3u) << 4);
    }

    d1   = dm.x * static_cast<float>(sc1);
    d2   = dm.x * static_cast<float>(sc2);
    min1 = dm.y * static_cast<float>(m1);
    min2 = dm.y * static_cast<float>(m2);
}


// ============================================================================
// Core Software-Pipelined Vectorized Dot Product Engine (1, 2, or 4 Rows/Warp)
// Issues ALL global weight loads across ROWS_PER_WARP before computing FMAs.
// ============================================================================

template <ggml_type TYPE, int ROWS_PER_WARP, bool X_IS_SMEM>
__device__ __forceinline__ void vec_dot_rows_f32(
        const uint8_t * __restrict__ row0_ptr,
        size_t                       row_stride,
        const float   * __restrict__ x_ptr,
        int K,
        int lane,
        float (&out)[ROWS_PER_WARP]) {
    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        acc[r] = 0.0f;
    }

    if constexpr (TYPE == GGML_TYPE_Q5_K) {
        const int nb = K / QK_K;
        const int c  = lane >> 3; // 0..3
        const int t  = lane & 7;  // 0..7

        for (int ib = 0; ib < nb; ++ib) {
            const int base_k  = ib * QK_K + 64 * c + 4 * t;
            const float4 x_lo = ld_x_f4<X_IS_SMEM>(x_ptr + base_k);
            const float4 x_hi = ld_x_f4<X_IS_SMEM>(x_ptr + base_k + 32);
            const float sum_lo = (x_lo.x + x_lo.y) + (x_lo.z + x_lo.w);
            const float sum_hi = (x_hi.x + x_hi.y) + (x_hi.z + x_hi.w);

            uint4    hdr[ROWS_PER_WARP];
            uint32_t qs_w[ROWS_PER_WARP];
            uint32_t qh_w[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q5_K & blk = reinterpret_cast<const block_q5_K *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                hdr[r]  = __ldg(reinterpret_cast<const uint4 *>(&blk));
                qs_w[r] = __ldg(reinterpret_cast<const uint32_t *>(blk.qs + 32 * c + 4 * t));
                qh_w[r] = __ldg(reinterpret_cast<const uint32_t *>(blk.qh + 4 * t));
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                float d1, min1, d2, min2;
                unpack_k4_scales_u4(hdr[r], c, d1, min1, d2, min2);

                const uint32_t hb1  = ((c <= 2) ? (qh_w[r] << (4 - 2 * c)) : (qh_w[r] >> 2)) & 0x10101010u;
                const uint32_t hb2  = ((c <  2) ? (qh_w[r] << (3 - 2 * c)) : (qh_w[r] >> (2 * c - 3))) & 0x10101010u;
                const uint32_t q1_4 = ( qs_w[r]       & 0x0F0F0F0Fu) | hb1;
                const uint32_t q2_4 = ((qs_w[r] >> 4) & 0x0F0F0F0Fu) | hb2;

                const float qx1 = fmaf(static_cast<float>( q1_4        & 0xFFu), x_lo.x,
                                  fmaf(static_cast<float>((q1_4 >>  8) & 0xFFu), x_lo.y,
                                  fmaf(static_cast<float>((q1_4 >> 16) & 0xFFu), x_lo.z,
                                       static_cast<float>( q1_4 >> 24         ) * x_lo.w)));
                const float qx2 = fmaf(static_cast<float>( q2_4        & 0xFFu), x_hi.x,
                                  fmaf(static_cast<float>((q2_4 >>  8) & 0xFFu), x_hi.y,
                                  fmaf(static_cast<float>((q2_4 >> 16) & 0xFFu), x_hi.z,
                                       static_cast<float>( q2_4 >> 24         ) * x_hi.w)));
                acc[r] = fmaf(d1, qx1, fmaf(-min1, sum_lo, acc[r]));
                acc[r] = fmaf(d2, qx2, fmaf(-min2, sum_hi, acc[r]));
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_Q4_K) {
        const int nb = K / QK_K;
        const int c  = lane >> 3;
        const int t  = lane & 7;

        for (int ib = 0; ib < nb; ++ib) {
            const int base_k  = ib * QK_K + 64 * c + 4 * t;
            const float4 x_lo = ld_x_f4<X_IS_SMEM>(x_ptr + base_k);
            const float4 x_hi = ld_x_f4<X_IS_SMEM>(x_ptr + base_k + 32);
            const float sum_lo = (x_lo.x + x_lo.y) + (x_lo.z + x_lo.w);
            const float sum_hi = (x_hi.x + x_hi.y) + (x_hi.z + x_hi.w);

            uint4    hdr[ROWS_PER_WARP];
            uint32_t qs_w[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q4_K & blk = reinterpret_cast<const block_q4_K *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                hdr[r]  = __ldg(reinterpret_cast<const uint4 *>(&blk));
                qs_w[r] = __ldg(reinterpret_cast<const uint32_t *>(blk.qs + 32 * c + 4 * t));
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                float d1, min1, d2, min2;
                unpack_k4_scales_u4(hdr[r], c, d1, min1, d2, min2);

                const uint32_t q1_4 =  qs_w[r]       & 0x0F0F0F0Fu;
                const uint32_t q2_4 = (qs_w[r] >> 4) & 0x0F0F0F0Fu;

                const float qx1 = fmaf(static_cast<float>( q1_4        & 0xFFu), x_lo.x,
                                  fmaf(static_cast<float>((q1_4 >>  8) & 0xFFu), x_lo.y,
                                  fmaf(static_cast<float>((q1_4 >> 16) & 0xFFu), x_lo.z,
                                       static_cast<float>( q1_4 >> 24         ) * x_lo.w)));
                const float qx2 = fmaf(static_cast<float>( q2_4        & 0xFFu), x_hi.x,
                                  fmaf(static_cast<float>((q2_4 >>  8) & 0xFFu), x_hi.y,
                                  fmaf(static_cast<float>((q2_4 >> 16) & 0xFFu), x_hi.z,
                                       static_cast<float>( q2_4 >> 24         ) * x_hi.w)));
                acc[r] = fmaf(d1, qx1, fmaf(-min1, sum_lo, acc[r]));
                acc[r] = fmaf(d2, qx2, fmaf(-min2, sum_hi, acc[r]));
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_Q6_K) {
        const int nb = K / QK_K;
        const int h  = lane >> 4;
        const int t  = lane & 15;
        const int is = t >> 3;

        for (int ib = 0; ib < nb; ++ib) {
            const int n = ib * QK_K + 128 * h + 2 * t;
            const float2 x0 = ld_x_f2<X_IS_SMEM>(x_ptr + n +  0);
            const float2 x1 = ld_x_f2<X_IS_SMEM>(x_ptr + n + 32);
            const float2 x2 = ld_x_f2<X_IS_SMEM>(x_ptr + n + 64);
            const float2 x3 = ld_x_f2<X_IS_SMEM>(x_ptr + n + 96);

            float    ds0[ROWS_PER_WARP], ds2[ROWS_PER_WARP], ds4[ROWS_PER_WARP], ds6[ROWS_PER_WARP];
            uint16_t ql0[ROWS_PER_WARP], ql1[ROWS_PER_WARP], qh0[ROWS_PER_WARP];

#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q6_K & blk = reinterpret_cast<const block_q6_K *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                const float d = __half2float(__ldg(&blk.d));
                const int8_t * __restrict__ sc = blk.scales + 8 * h + is;
                ds0[r] = d * static_cast<float>(__ldg(sc + 0));
                ds2[r] = d * static_cast<float>(__ldg(sc + 2));
                ds4[r] = d * static_cast<float>(__ldg(sc + 4));
                ds6[r] = d * static_cast<float>(__ldg(sc + 6));

                ql0[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.ql + 64 * h + 2 * t));
                ql1[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.ql + 64 * h + 32 + 2 * t));
                qh0[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.qh + 32 * h + 2 * t));
            }

#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const uint32_t ql0_0 = ql0[r] & 0xFFu;
                const uint32_t ql0_1 = ql0[r] >> 8;
                const uint32_t ql1_0 = ql1[r] & 0xFFu;
                const uint32_t ql1_1 = ql1[r] >> 8;
                const uint32_t qh0_0 = qh0[r] & 0xFFu;
                const uint32_t qh0_1 = qh0[r] >> 8;

                const float q1_0 = static_cast<float>(static_cast<int>((ql0_0 & 0x0Fu) | (((qh0_0 >> 0) & 3u) << 4)) - 32);
                const float q1_1 = static_cast<float>(static_cast<int>((ql0_1 & 0x0Fu) | (((qh0_1 >> 0) & 3u) << 4)) - 32);
                const float q2_0 = static_cast<float>(static_cast<int>((ql1_0 & 0x0Fu) | (((qh0_0 >> 2) & 3u) << 4)) - 32);
                const float q2_1 = static_cast<float>(static_cast<int>((ql1_1 & 0x0Fu) | (((qh0_1 >> 2) & 3u) << 4)) - 32);
                const float q3_0 = static_cast<float>(static_cast<int>((ql0_0 >> 4)     | (((qh0_0 >> 4) & 3u) << 4)) - 32);
                const float q3_1 = static_cast<float>(static_cast<int>((ql0_1 >> 4)     | (((qh0_1 >> 4) & 3u) << 4)) - 32);
                const float q4_0 = static_cast<float>(static_cast<int>((ql1_0 >> 4)     | (((qh0_0 >> 6) & 3u) << 4)) - 32);
                const float q4_1 = static_cast<float>(static_cast<int>((ql1_1 >> 4)     | (((qh0_1 >> 6) & 3u) << 4)) - 32);

                acc[r] = fmaf(ds0[r], fmaf(q1_0, x0.x, q1_1 * x0.y), acc[r]);
                acc[r] = fmaf(ds2[r], fmaf(q2_0, x1.x, q2_1 * x1.y), acc[r]);
                acc[r] = fmaf(ds4[r], fmaf(q3_0, x2.x, q3_1 * x2.y), acc[r]);
                acc[r] = fmaf(ds6[r], fmaf(q4_0, x3.x, q4_1 * x3.y), acc[r]);
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_Q8_0) {
        const int nb    = K / QK8_0;
        const int b_off = lane >> 3;
        const int t     = lane & 7;

        for (int ib = b_off; ib < nb; ib += 4) {
            const float4 x4 = ld_x_f4<X_IS_SMEM>(x_ptr + ib * QK8_0 + 4 * t);

            float    dv[ROWS_PER_WARP];
            uint16_t q01[ROWS_PER_WARP], q23[ROWS_PER_WARP];

#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q8_0 & blk = reinterpret_cast<const block_q8_0 *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                dv[r]  = __half2float(__ldg(&blk.d));
                q01[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.qs + 4 * t));
                q23[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.qs + 4 * t + 2));
            }

#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const float q0 = static_cast<float>(static_cast<int8_t>(q01[r] & 0xFFu));
                const float q1 = static_cast<float>(static_cast<int8_t>(q01[r] >> 8));
                const float q2 = static_cast<float>(static_cast<int8_t>(q23[r] & 0xFFu));
                const float q3 = static_cast<float>(static_cast<int8_t>(q23[r] >> 8));
                const float p  = fmaf(q0, x4.x, fmaf(q1, x4.y, fmaf(q2, x4.z, q3 * x4.w)));
                acc[r] = fmaf(dv[r], p, acc[r]);
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_F32) {
        const bool aligned = ((reinterpret_cast<uintptr_t>(row0_ptr) & 15u) == 0u) && ((row_stride & 15u) == 0u);
        if (aligned) {
            for (int k = lane * 4; k < K; k += MMVQ_F32_WARP_SIZE * 4) {
                const float4 x4 = ld_x_f4<X_IS_SMEM>(x_ptr + k);
                float4 w4[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    w4[r] = __ldg(reinterpret_cast<const float4 *>(row0_ptr + static_cast<size_t>(r) * row_stride + static_cast<size_t>(k) * sizeof(float)));
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    acc[r] = fmaf(w4[r].x, x4.x, acc[r]);
                    acc[r] = fmaf(w4[r].y, x4.y, acc[r]);
                    acc[r] = fmaf(w4[r].z, x4.z, acc[r]);
                    acc[r] = fmaf(w4[r].w, x4.w, acc[r]);
                }
            }
        } else {
            for (int k = lane; k < K; k += MMVQ_F32_WARP_SIZE) {
                const float xv = ld_x_f1<X_IS_SMEM>(x_ptr + k);
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const float * w = reinterpret_cast<const float *>(row0_ptr + static_cast<size_t>(r) * row_stride);
                    acc[r] = fmaf(__ldg(w + k), xv, acc[r]);
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_F16) {
        const bool aligned = ((reinterpret_cast<uintptr_t>(row0_ptr) & 7u) == 0u) && ((row_stride & 7u) == 0u);
        if (aligned) {
            for (int k = lane * 4; k < K; k += MMVQ_F32_WARP_SIZE * 4) {
                const float4 x4 = ld_x_f4<X_IS_SMEM>(x_ptr + k);
                uint2 p[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    p[r] = __ldg(reinterpret_cast<const uint2 *>(row0_ptr + static_cast<size_t>(r) * row_stride + static_cast<size_t>(k) * sizeof(half)));
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&p[r].x));
                    const float2 b = __half22float2(*reinterpret_cast<const half2 *>(&p[r].y));
                    acc[r] = fmaf(a.x, x4.x, acc[r]);
                    acc[r] = fmaf(a.y, x4.y, acc[r]);
                    acc[r] = fmaf(b.x, x4.z, acc[r]);
                    acc[r] = fmaf(b.y, x4.w, acc[r]);
                }
            }
        } else {
            for (int k = lane; k < K; k += MMVQ_F32_WARP_SIZE) {
                const float xv = ld_x_f1<X_IS_SMEM>(x_ptr + k);
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const half * w = reinterpret_cast<const half *>(row0_ptr + static_cast<size_t>(r) * row_stride);
                    acc[r] = fmaf(__half2float(__ldg(w + k)), xv, acc[r]);
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_BF16) {
        const bool aligned = ((reinterpret_cast<uintptr_t>(row0_ptr) & 7u) == 0u) && ((row_stride & 7u) == 0u);
        if (aligned) {
            for (int k = lane * 4; k < K; k += MMVQ_F32_WARP_SIZE * 4) {
                const float4 x4 = ld_x_f4<X_IS_SMEM>(x_ptr + k);
                uint2 p[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    p[r] = __ldg(reinterpret_cast<const uint2 *>(row0_ptr + static_cast<size_t>(r) * row_stride + static_cast<size_t>(k) * sizeof(uint16_t)));
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    acc[r] = fmaf(bf16_bits_to_f32(static_cast<uint16_t>(p[r].x & 0xFFFFu)), x4.x, acc[r]);
                    acc[r] = fmaf(bf16_bits_to_f32(static_cast<uint16_t>(p[r].x >> 16)),     x4.y, acc[r]);
                    acc[r] = fmaf(bf16_bits_to_f32(static_cast<uint16_t>(p[r].y & 0xFFFFu)), x4.z, acc[r]);
                    acc[r] = fmaf(bf16_bits_to_f32(static_cast<uint16_t>(p[r].y >> 16)),     x4.w, acc[r]);
                }
            }
        } else {
            for (int k = lane; k < K; k += MMVQ_F32_WARP_SIZE) {
                const float xv = ld_x_f1<X_IS_SMEM>(x_ptr + k);
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const uint16_t * w = reinterpret_cast<const uint16_t *>(row0_ptr + static_cast<size_t>(r) * row_stride);
                    acc[r] = fmaf(bf16_bits_to_f32(__ldg(w + k)), xv, acc[r]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        out[r] = warp_sum_f32(acc[r]);
    }
}

template <int ROWS_PER_WARP, bool X_IS_SMEM>
__device__ __forceinline__ void vec_dot_dispatch_rows(
        ggml_type                    type,
        const uint8_t * __restrict__ row0_ptr,
        size_t                       row_stride,
        const float   * __restrict__ x_ptr,
        int K,
        int lane,
        float (&out)[ROWS_PER_WARP]) {
    switch (type) {
        case GGML_TYPE_Q5_K: vec_dot_rows_f32<GGML_TYPE_Q5_K, ROWS_PER_WARP, X_IS_SMEM>(row0_ptr, row_stride, x_ptr, K, lane, out); break;
        case GGML_TYPE_Q6_K: vec_dot_rows_f32<GGML_TYPE_Q6_K, ROWS_PER_WARP, X_IS_SMEM>(row0_ptr, row_stride, x_ptr, K, lane, out); break;
        case GGML_TYPE_Q8_0: vec_dot_rows_f32<GGML_TYPE_Q8_0, ROWS_PER_WARP, X_IS_SMEM>(row0_ptr, row_stride, x_ptr, K, lane, out); break;
        case GGML_TYPE_Q4_K: vec_dot_rows_f32<GGML_TYPE_Q4_K, ROWS_PER_WARP, X_IS_SMEM>(row0_ptr, row_stride, x_ptr, K, lane, out); break;
        case GGML_TYPE_F16:  vec_dot_rows_f32<GGML_TYPE_F16,  ROWS_PER_WARP, X_IS_SMEM>(row0_ptr, row_stride, x_ptr, K, lane, out); break;
        case GGML_TYPE_BF16: vec_dot_rows_f32<GGML_TYPE_BF16, ROWS_PER_WARP, X_IS_SMEM>(row0_ptr, row_stride, x_ptr, K, lane, out); break;
        default:             vec_dot_rows_f32<GGML_TYPE_F32,  ROWS_PER_WARP, X_IS_SMEM>(row0_ptr, row_stride, x_ptr, K, lane, out); break;
    }
}

// ============================================================================
// 0. Dedicated High-Throughput LM Head Q6_K Kernel (8 Rows/Warp + SMEM x)
// ============================================================================

template <bool USE_SMEM>
__global__ __launch_bounds__(MMVQ_F32_THREADS_LG, 4)
void mul_mat_vec_q6_K_f32_vocab_kernel(
        const uint8_t * __restrict__ src0_bytes,
        const float   * __restrict__ x_global,
        float         * __restrict__ dst_f32,
        int     K,
        int     M,
        size_t  src0_nb1) {
    extern __shared__ float s_x[];

    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid     = warp_id * MMVQ_F32_WARP_SIZE + lane;

    if constexpr (USE_SMEM) {
        const float4 * __restrict__ x4 = reinterpret_cast<const float4 *>(x_global);
        float4       * __restrict__ s4 = reinterpret_cast<float4 *>(s_x);
        const int k4 = K >> 2;
        for (int i = tid; i < k4; i += MMVQ_F32_THREADS_LG) {
            s4[i] = __ldg(x4 + i);
        }
        __syncthreads();
    }

    const int row0 = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_LG + warp_id) * 8;
    if (row0 >= M) {
        return;
    }

    const float   * x_ptr    = USE_SMEM ? s_x : x_global;
    const uint8_t * row0_ptr = src0_bytes + static_cast<size_t>(row0) * src0_nb1;

    if (row0 + 8 <= M) {
        const int nb = K / QK_K;
        const int h  = lane >> 4;
        const int t  = lane & 15;
        const int is = t >> 3;

        float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        for (int ib = 0; ib < nb; ++ib) {
            const int n = ib * QK_K + 128 * h + 2 * t;
            const float2 x0 = ld_x_f2<USE_SMEM>(x_ptr + n +  0);
            const float2 x1 = ld_x_f2<USE_SMEM>(x_ptr + n + 32);
            const float2 x2 = ld_x_f2<USE_SMEM>(x_ptr + n + 64);
            const float2 x3 = ld_x_f2<USE_SMEM>(x_ptr + n + 96);

            // Process 8 rows in 4 software-pipelined pairs to keep registers <= 48/thread
#pragma unroll
            for (int p = 0; p < 8; p += 2) {
                const block_q6_K & b0 = reinterpret_cast<const block_q6_K *>(row0_ptr + static_cast<size_t>(p + 0) * src0_nb1)[ib];
                const block_q6_K & b1 = reinterpret_cast<const block_q6_K *>(row0_ptr + static_cast<size_t>(p + 1) * src0_nb1)[ib];

                const float d0 = __half2float(__ldg(&b0.d));
                const float d1 = __half2float(__ldg(&b1.d));
                const int8_t * __restrict__ sc0 = b0.scales + 8 * h + is;
                const int8_t * __restrict__ sc1 = b1.scales + 8 * h + is;

                const float a0_0 = d0 * static_cast<float>(__ldg(sc0 + 0));
                const float a0_2 = d0 * static_cast<float>(__ldg(sc0 + 2));
                const float a0_4 = d0 * static_cast<float>(__ldg(sc0 + 4));
                const float a0_6 = d0 * static_cast<float>(__ldg(sc0 + 6));

                const float a1_0 = d1 * static_cast<float>(__ldg(sc1 + 0));
                const float a1_2 = d1 * static_cast<float>(__ldg(sc1 + 2));
                const float a1_4 = d1 * static_cast<float>(__ldg(sc1 + 4));
                const float a1_6 = d1 * static_cast<float>(__ldg(sc1 + 6));

                const uint16_t u0_ql0 = __ldg(reinterpret_cast<const uint16_t *>(b0.ql + 64 * h + 2 * t));
                const uint16_t u0_ql1 = __ldg(reinterpret_cast<const uint16_t *>(b0.ql + 64 * h + 32 + 2 * t));
                const uint16_t u0_qh0 = __ldg(reinterpret_cast<const uint16_t *>(b0.qh + 32 * h + 2 * t));

                const uint16_t u1_ql0 = __ldg(reinterpret_cast<const uint16_t *>(b1.ql + 64 * h + 2 * t));
                const uint16_t u1_ql1 = __ldg(reinterpret_cast<const uint16_t *>(b1.ql + 64 * h + 32 + 2 * t));
                const uint16_t u1_qh0 = __ldg(reinterpret_cast<const uint16_t *>(b1.qh + 32 * h + 2 * t));

                auto mac_row = [&](uint16_t ql0, uint16_t ql1, uint16_t qh0,
                                   float ds0, float ds2, float ds4, float ds6, float & acc_r) {
                    const uint32_t ql0_0 = ql0 & 0xFFu, ql0_1 = ql0 >> 8;
                    const uint32_t ql1_0 = ql1 & 0xFFu, ql1_1 = ql1 >> 8;
                    const uint32_t qh0_0 = qh0 & 0xFFu, qh0_1 = qh0 >> 8;

                    const float q1_0 = static_cast<float>(static_cast<int>((ql0_0 & 0x0Fu) | (((qh0_0 >> 0) & 3u) << 4)) - 32);
                    const float q1_1 = static_cast<float>(static_cast<int>((ql0_1 & 0x0Fu) | (((qh0_1 >> 0) & 3u) << 4)) - 32);
                    const float q2_0 = static_cast<float>(static_cast<int>((ql1_0 & 0x0Fu) | (((qh0_0 >> 2) & 3u) << 4)) - 32);
                    const float q2_1 = static_cast<float>(static_cast<int>((ql1_1 & 0x0Fu) | (((qh0_1 >> 2) & 3u) << 4)) - 32);
                    const float q3_0 = static_cast<float>(static_cast<int>((ql0_0 >> 4)     | (((qh0_0 >> 4) & 3u) << 4)) - 32);
                    const float q3_1 = static_cast<float>(static_cast<int>((ql0_1 >> 4)     | (((qh0_1 >> 4) & 3u) << 4)) - 32);
                    const float q4_0 = static_cast<float>(static_cast<int>((ql1_0 >> 4)     | (((qh0_0 >> 6) & 3u) << 4)) - 32);
                    const float q4_1 = static_cast<float>(static_cast<int>((ql1_1 >> 4)     | (((qh0_1 >> 6) & 3u) << 4)) - 32);

                    acc_r = fmaf(ds0, fmaf(q1_0, x0.x, q1_1 * x0.y), acc_r);
                    acc_r = fmaf(ds2, fmaf(q2_0, x1.x, q2_1 * x1.y), acc_r);
                    acc_r = fmaf(ds4, fmaf(q3_0, x2.x, q3_1 * x2.y), acc_r);
                    acc_r = fmaf(ds6, fmaf(q4_0, x3.x, q4_1 * x3.y), acc_r);
                };

                mac_row(u0_ql0, u0_ql1, u0_qh0, a0_0, a0_2, a0_4, a0_6, acc[p + 0]);
                mac_row(u1_ql0, u1_ql1, u1_qh0, a1_0, a1_2, a1_4, a1_6, acc[p + 1]);
            }
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float sum = warp_sum_f32(acc[r]);
            if (lane == 0) {
                dst_f32[row0 + r] = sum;
            }
        }
    } else {
        for (int r = 0; r < 8 && row0 + r < M; ++r) {
            float s1[1];
            vec_dot_rows_f32<GGML_TYPE_Q6_K, 1, USE_SMEM>(
                row0_ptr + static_cast<size_t>(r) * src0_nb1, 0, x_ptr, K, lane, s1);
            if (lane == 0) {
                dst_f32[row0 + r] = s1[0];
            }
        }
    }
}

// ============================================================================
// 1. Vectorized GGML_OP_MUL_MAT_ID (NWARPS=2 -> High SM Saturation on sm_86)
// ============================================================================

template <ggml_type TYPE, int ROWS_PER_WARP>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 8)
void mul_mat_vec_q_f32_id_kernel(
        const uint8_t * __restrict__ src0_bytes,
        const uint8_t * __restrict__ src1_bytes,
        const uint8_t * __restrict__ ids_bytes,
        uint8_t       * __restrict__ dst_bytes,
        int     K,
        int     M,
        int     n_experts,
        int     ne11,
        size_t  src0_nb1,
        size_t  src0_nb2,
        size_t  src1_nb1,
        size_t  src1_nb2,
        size_t  ids_nb1,
        size_t  dst_nb1,
        size_t  dst_nb2) {
    const int lane        = threadIdx.x;
    const int warp_id     = threadIdx.y;
    const int row0        = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    const int expert_slot = static_cast<int>(blockIdx.y);
    const int token       = static_cast<int>(blockIdx.z);

    if (row0 >= M) {
        return;
    }

    const int32_t * ids_row =
        reinterpret_cast<const int32_t *>(ids_bytes + static_cast<size_t>(token) * ids_nb1);
    const int expert_id = __ldg(ids_row + expert_slot);

    float * dst_row = reinterpret_cast<float *>(
        dst_bytes + static_cast<size_t>(token) * dst_nb2
                  + static_cast<size_t>(expert_slot) * dst_nb1);

    if (expert_id < 0 || expert_id >= n_experts) {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (row0 + r < M) {
                    dst_row[row0 + r] = 0.0f;
                }
            }
        }
        return;
    }

    const int i11 = (ne11 == 1) ? 0 : (expert_slot % ne11);
    const float * x_ptr = reinterpret_cast<const float *>(
        src1_bytes + static_cast<size_t>(token) * src1_nb2
                   + static_cast<size_t>(i11)   * src1_nb1);

    const uint8_t * row0_ptr =
        src0_bytes + static_cast<size_t>(expert_id) * src0_nb2
                   + static_cast<size_t>(row0)      * src0_nb1;

    if (row0 + ROWS_PER_WARP <= M) {
        float s[ROWS_PER_WARP];
        vec_dot_rows_f32<TYPE, ROWS_PER_WARP, false>(row0_ptr, src0_nb1, x_ptr, K, lane, s);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                dst_row[row0 + r] = s[r];
            }
        }
    } else {
        for (int r = 0; r < ROWS_PER_WARP && row0 + r < M; ++r) {
            float s1[1];
            vec_dot_rows_f32<TYPE, 1, false>(row0_ptr + static_cast<size_t>(r) * src0_nb1, 0, x_ptr, K, lane, s1);
            if (lane == 0) {
                dst_row[row0 + r] = s1[0];
            }
        }
    }
}

template <ggml_type TYPE>
void launch_mmvq_f32_id(
        cudaStream_t stream,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
    const int K             = static_cast<int>(src0->ne[0]);
    const int M             = static_cast<int>(src0->ne[1]);
    const int n_experts     = static_cast<int>(src0->ne[2]);
    const int ne11          = static_cast<int>(src1->ne[1]);
    const int n_expert_used = static_cast<int>(ids->ne[0]);
    const int n_tokens      = static_cast<int>(ids->ne[1]);

    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);
    const int bx4 = (M + (MMVQ_F32_NWARPS_SM * 4) - 1) / (MMVQ_F32_NWARPS_SM * 4);
    const int bx2 = (M + (MMVQ_F32_NWARPS_SM * 2) - 1) / (MMVQ_F32_NWARPS_SM * 2);
    const int bx1 = (M + MMVQ_F32_NWARPS_SM - 1) / MMVQ_F32_NWARPS_SM;
    const int mult = n_expert_used * n_tokens;

    const uint8_t * src0_bytes = static_cast<const uint8_t *>(src0->data);
    const uint8_t * src1_bytes = static_cast<const uint8_t *>(src1->data);
    const uint8_t * ids_bytes  = static_cast<const uint8_t *>(ids->data);
    uint8_t       * dst_bytes  = static_cast<uint8_t *>(dst->data);

    // Require >= 224 blocks (8 full waves on 28 SMs) before widening to 4 rows/warp
    if (bx4 * mult >= 224) {
        const dim3 grid(static_cast<unsigned int>(bx4), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        mul_mat_vec_q_f32_id_kernel<TYPE, 4><<<grid, block, 0, stream>>>(
            src0_bytes, src1_bytes, ids_bytes, dst_bytes,
            K, M, n_experts, ne11,
            src0->nb[1], src0->nb[2], src1->nb[1], src1->nb[2],
            ids->nb[1], dst->nb[1], dst->nb[2]);
    } else if (bx2 * mult >= 112) {
        const dim3 grid(static_cast<unsigned int>(bx2), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        mul_mat_vec_q_f32_id_kernel<TYPE, 2><<<grid, block, 0, stream>>>(
            src0_bytes, src1_bytes, ids_bytes, dst_bytes,
            K, M, n_experts, ne11,
            src0->nb[1], src0->nb[2], src1->nb[1], src1->nb[2],
            ids->nb[1], dst->nb[1], dst->nb[2]);
    } else {
        const dim3 grid(static_cast<unsigned int>(bx1), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        mul_mat_vec_q_f32_id_kernel<TYPE, 1><<<grid, block, 0, stream>>>(
            src0_bytes, src1_bytes, ids_bytes, dst_bytes,
            K, M, n_experts, ne11,
            src0->nb[1], src0->nb[2], src1->nb[1], src1->nb[2],
            ids->nb[1], dst->nb[1], dst->nb[2]);
    }
}

// ============================================================================
// 2. Vectorized Dense GGML_OP_MUL_MAT (260+ GB/s Q5_K + Dedicated Vocab Q6_K)
// ============================================================================

template <ggml_type TYPE, int ROWS_PER_WARP>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 8)
void mul_mat_vec_q_f32_kernel(
        const uint8_t * __restrict__ src0_bytes,
        const uint8_t * __restrict__ src1_bytes,
        uint8_t       * __restrict__ dst_bytes,
        int     K,
        int     M,
        int     ne12,
        int     r2,
        int     r3,
        size_t  src0_nb1,
        size_t  src0_nb2,
        size_t  src0_nb3,
        size_t  src1_nb1,
        size_t  src1_nb2,
        size_t  src1_nb3,
        size_t  dst_nb1,
        size_t  dst_nb2,
        size_t  dst_nb3) {
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row0    = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    if (row0 >= M) {
        return;
    }

    const int i11  = static_cast<int>(blockIdx.y);
    const int bidx = static_cast<int>(blockIdx.z);
    const int i12  = bidx % ne12;
    const int i13  = bidx / ne12;
    const int i02  = i12 / r2;
    const int i03  = i13 / r3;

    const float * x_ptr = reinterpret_cast<const float *>(
        src1_bytes + static_cast<size_t>(i13) * src1_nb3
                   + static_cast<size_t>(i12) * src1_nb2
                   + static_cast<size_t>(i11) * src1_nb1);

    const uint8_t * row0_ptr =
        src0_bytes + static_cast<size_t>(i03)  * src0_nb3
                   + static_cast<size_t>(i02)  * src0_nb2
                   + static_cast<size_t>(row0) * src0_nb1;

    float * dst_row = reinterpret_cast<float *>(
        dst_bytes + static_cast<size_t>(i13) * dst_nb3
                  + static_cast<size_t>(i12) * dst_nb2
                  + static_cast<size_t>(i11) * dst_nb1);

    if (row0 + ROWS_PER_WARP <= M) {
        float s[ROWS_PER_WARP];
        vec_dot_rows_f32<TYPE, ROWS_PER_WARP, false>(row0_ptr, src0_nb1, x_ptr, K, lane, s);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                dst_row[row0 + r] = s[r];
            }
        }
    } else {
        for (int r = 0; r < ROWS_PER_WARP && row0 + r < M; ++r) {
            float s1[1];
            vec_dot_rows_f32<TYPE, 1, false>(row0_ptr + static_cast<size_t>(r) * src0_nb1, 0, x_ptr, K, lane, s1);
            if (lane == 0) {
                dst_row[row0 + r] = s1[0];
            }
        }
    }
}

template <ggml_type TYPE, int ROWS_PER_WARP, int BATCH_T>
__device__ __forceinline__ void vec_dot_rows_f32_batched(
        const uint8_t * __restrict__ row0_ptr,
        size_t                       row_stride,
        const float * const (&x_ptrs)[BATCH_T],
        int K,
        int lane,
        float (&out)[BATCH_T][ROWS_PER_WARP]);

template <ggml_type TYPE, int ROWS_PER_WARP, int BATCH_T>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 6)
void mul_mat_vec_q_f32_batched_kernel(
        const uint8_t * __restrict__ src0_bytes,
        const uint8_t * __restrict__ src1_bytes,
        uint8_t       * __restrict__ dst_bytes,
        int K, int M, int ne11, int ne12, int r2, int r3,
        size_t src0_nb1, size_t src0_nb2, size_t src0_nb3,
        size_t src1_nb1, size_t src1_nb2, size_t src1_nb3,
        size_t dst_nb1,  size_t dst_nb2,  size_t dst_nb3) {
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row0    = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    if (row0 >= M) return;

    const int tok0 = static_cast<int>(blockIdx.y) * BATCH_T;
    const int cnt  = min(BATCH_T, ne11 - tok0);
    const int bidx = static_cast<int>(blockIdx.z);
    const int i12  = bidx % ne12;
    const int i13  = bidx / ne12;
    const int i02  = i12 / r2;
    const int i03  = i13 / r3;

    const uint8_t * s1_base = src1_bytes + static_cast<size_t>(i13) * src1_nb3 + static_cast<size_t>(i12) * src1_nb2;
    const float * x_ptrs[BATCH_T];
#pragma unroll
    for (int b = 0; b < BATCH_T; ++b) {
        const int t = tok0 + ((b < cnt) ? b : 0);
        x_ptrs[b] = reinterpret_cast<const float *>(s1_base + static_cast<size_t>(t) * src1_nb1);
    }

    const uint8_t * row0_ptr = src0_bytes + static_cast<size_t>(i03) * src0_nb3 + static_cast<size_t>(i02) * src0_nb2 + static_cast<size_t>(row0) * src0_nb1;
    uint8_t       * d_base   = dst_bytes  + static_cast<size_t>(i13) * dst_nb3  + static_cast<size_t>(i12) * dst_nb2;

    const int n_rows = min(ROWS_PER_WARP, M - row0);
    float out[BATCH_T][ROWS_PER_WARP];
    if (n_rows == ROWS_PER_WARP) {
        vec_dot_rows_f32_batched<TYPE, ROWS_PER_WARP, BATCH_T>(row0_ptr, src0_nb1, x_ptrs, K, lane, out);
    } else {
        for (int r = 0; r < n_rows; ++r) {
            float out1[BATCH_T][1];
            vec_dot_rows_f32_batched<TYPE, 1, BATCH_T>(row0_ptr + static_cast<size_t>(r) * src0_nb1, 0, x_ptrs, K, lane, out1);
#pragma unroll
            for (int b = 0; b < BATCH_T; ++b) out[b][r] = out1[b][0];
        }
    }
    if (lane == 0) {
        for (int b = 0; b < cnt; ++b) {
            float * dst_row = reinterpret_cast<float *>(d_base + static_cast<size_t>(tok0 + b) * dst_nb1);
            for (int r = 0; r < n_rows; ++r) dst_row[row0 + r] = out[b][r];
        }
    }
}

template <ggml_type TYPE>
void launch_mmvq_f32(
        cudaStream_t stream,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    const int K    = static_cast<int>(src0->ne[0]);
    const int M    = static_cast<int>(src0->ne[1]);
    const int ne11 = static_cast<int>(src1->ne[1]);
    const int ne12 = static_cast<int>(src1->ne[2]);
    const int ne13 = static_cast<int>(src1->ne[3]);

    // Dedicated high-occupancy 8-row SMEM kernel for single-token Q6_K lm_head
    if constexpr (TYPE == GGML_TYPE_Q6_K) {
        if (M >= MMVQ_F32_VOCAB_MIN_M && ne11 == 1 && ne12 == 1 && ne13 == 1) {
            const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_LG, 1);
            const dim3 grid(static_cast<unsigned int>((M + 31) / 32), 1, 1);
            if (K <= MMVQ_F32_VOCAB_SMEM_K) {
                const size_t smem = static_cast<size_t>(K) * sizeof(float);
                mul_mat_vec_q6_K_f32_vocab_kernel<true><<<grid, block, smem, stream>>>(
                    static_cast<const uint8_t *>(src0->data),
                    static_cast<const float *>(src1->data),
                    static_cast<float *>(dst->data),
                    K, M, src0->nb[1]);
            } else {
                mul_mat_vec_q6_K_f32_vocab_kernel<false><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(src0->data),
                    static_cast<const float *>(src1->data),
                    static_cast<float *>(dst->data),
                    K, M, src0->nb[1]);
            }
            return;
        }
    }

    const int r2         = static_cast<int>(src1->ne[2] / src0->ne[2]);
    const int r3         = static_cast<int>(src1->ne[3] / src0->ne[3]);
    const int batch_mult = ne11 * ne12 * ne13;

    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);
    const int bx4 = (M + (MMVQ_F32_NWARPS_SM * 4) - 1) / (MMVQ_F32_NWARPS_SM * 4);
    const int bx2 = (M + (MMVQ_F32_NWARPS_SM * 2) - 1) / (MMVQ_F32_NWARPS_SM * 2);
    const int bx1 = (M + MMVQ_F32_NWARPS_SM - 1) / MMVQ_F32_NWARPS_SM;

    const uint8_t * src0_bytes = static_cast<const uint8_t *>(src0->data);
    const uint8_t * src1_bytes = static_cast<const uint8_t *>(src1->data);
    uint8_t       * dst_bytes  = static_cast<uint8_t *>(dst->data);

    // Batched weight-sharing GEMV for ne11 = 2..8 (avoids BATCH_T=8 register spills on pp32)
    if (ne11 >= 2) {
        if (ne11 == 2) {
            const dim3 grid(static_cast<unsigned int>(bx2), 1, static_cast<unsigned int>(ne12 * ne13));
            mul_mat_vec_q_f32_batched_kernel<TYPE, 2, 2><<<grid, block, 0, stream>>>(
                src0_bytes, src1_bytes, dst_bytes, K, M, ne11, ne12, r2, r3,
                src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]);
            return;
        }
        const int by = (ne11 + 3) / 4;
        const dim3 grid(static_cast<unsigned int>(bx2), static_cast<unsigned int>(by), static_cast<unsigned int>(ne12 * ne13));
        mul_mat_vec_q_f32_batched_kernel<TYPE, 2, 4><<<grid, block, 0, stream>>>(
            src0_bytes, src1_bytes, dst_bytes, K, M, ne11, ne12, r2, r3,
            src0->nb[1], src0->nb[2], src0->nb[3], src1->nb[1], src1->nb[2], src1->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]);
        return;
    }

    // For Q5_K/Q4_K, 2 rows/warp @ 64 threads maximizes occupancy (24-32 warps/SM) and kills wave tails
    if (bx4 * batch_mult >= 224) {
        const dim3 grid(static_cast<unsigned int>(bx4), static_cast<unsigned int>(ne11), static_cast<unsigned int>(ne12 * ne13));
        mul_mat_vec_q_f32_kernel<TYPE, 4><<<grid, block, 0, stream>>>(
            src0_bytes, src1_bytes, dst_bytes,
            K, M, ne12, r2, r3,
            src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[1], src1->nb[2], src1->nb[3],
            dst->nb[1],  dst->nb[2],  dst->nb[3]);
    } else if (bx2 * batch_mult >= 112) {
        const dim3 grid(static_cast<unsigned int>(bx2), static_cast<unsigned int>(ne11), static_cast<unsigned int>(ne12 * ne13));
        mul_mat_vec_q_f32_kernel<TYPE, 2><<<grid, block, 0, stream>>>(
            src0_bytes, src1_bytes, dst_bytes,
            K, M, ne12, r2, r3,
            src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[1], src1->nb[2], src1->nb[3],
            dst->nb[1],  dst->nb[2],  dst->nb[3]);
    } else {
        const dim3 grid(static_cast<unsigned int>(bx1), static_cast<unsigned int>(ne11), static_cast<unsigned int>(ne12 * ne13));
        mul_mat_vec_q_f32_kernel<TYPE, 1><<<grid, block, 0, stream>>>(
            src0_bytes, src1_bytes, dst_bytes,
            K, M, ne12, r2, r3,
            src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[1], src1->nb[2], src1->nb[3],
            dst->nb[1],  dst->nb[2],  dst->nb[3]);
    }
}

// ============================================================================
// 3. Fused GGML_OP_OLALA_MOE (4-Kernel Pipeline: Stage 1+1b+2 & Stage 5a+5b)
// ============================================================================

// Device-inline 1-Warp Top-K epilogue using __ldcg (bypasses L1 for cross-SM visibility)
template <int MAX_PER_LANE>
__device__ __forceinline__ void olala_topk_warp_inline(
        const float * __restrict__ logits,
        const float * __restrict__ exp_probs_b,
        int32_t     * __restrict__ out_ids,
        float       * __restrict__ out_w,
        int   n_expert,
        int   n_expert_used,
        float w_scale,
        int   lane) {
    float p_reg[MAX_PER_LANE];
    float s_reg[MAX_PER_LANE];

#pragma unroll
    for (int j = 0; j < MAX_PER_LANE; ++j) {
        const int e = lane + 32 * j;
        if (e < n_expert) {
            const float p = 1.0f / (1.0f + expf(-__ldcg(logits + e)));
            const float b = (exp_probs_b != nullptr) ? __ldg(exp_probs_b + e) : 0.0f;
            p_reg[j] = p;
            s_reg[j] = p + b;
        } else {
            p_reg[j] = 0.0f;
            s_reg[j] = -INFINITY;
        }
    }

    int32_t my_id   = 0;
    float   my_prob = 0.0f;
    float   w_sum   = 0.0f;

    for (int k = 0; k < n_expert_used; ++k) {
        float best_v = -INFINITY;
        float best_p = 0.0f;
        int   best_i = n_expert;

#pragma unroll
        for (int j = 0; j < MAX_PER_LANE; ++j) {
            const float v = s_reg[j];
            const int   e = lane + 32 * j;
            if (v > best_v || (v == best_v && e < best_i)) {
                best_v = v;
                best_p = p_reg[j];
                best_i = e;
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float v2 = __shfl_xor_sync(0xFFFFFFFFu, best_v, offset);
            const float p2 = __shfl_xor_sync(0xFFFFFFFFu, best_p, offset);
            const int   i2 = __shfl_xor_sync(0xFFFFFFFFu, best_i, offset);
            if (v2 > best_v || (v2 == best_v && i2 < best_i)) {
                best_v = v2;
                best_p = p2;
                best_i = i2;
            }
        }

        if (lane == k) {
            my_id   = best_i;
            my_prob = best_p;
        }
        w_sum += (best_i < n_expert) ? best_p : 0.0f;

        if ((best_i & 31) == lane) {
            const int slot_j = best_i >> 5;
#pragma unroll
            for (int j = 0; j < MAX_PER_LANE; ++j) {
                if (j == slot_j) {
                    s_reg[j] = -INFINITY;
                }
            }
        }
    }

    if (lane < n_expert_used) {
        const float scale = w_scale / fmaxf(w_sum, 6.103515625e-5f);
        out_ids[lane] = (my_id < n_expert) ? my_id : 0;
        out_w[lane]   = (my_id < n_expert) ? (my_prob * scale) : 0.0f;
    }
}

// Stage 1: Fused Router + Last-Block Top-K Epilogue + Latent Down + Shared Expert Up (ReLU^2)
// Compile-time specialization for TYPE_LD / TYPE_SH eliminates 7-way switch register bloat.
template <bool SPECIALIZED, ggml_type SPEC_LD, ggml_type SPEC_SH, bool FUSE_TOPK, int TOPK_PER_LANE>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 8)
void olala_stage1_fused_cur_kernel(
        const uint8_t * __restrict__ cur_bytes,
        size_t                       cur_nb1,
        const uint8_t * __restrict__ gate_inp_bytes,
        size_t                       gate_inp_nb1,
        ggml_type                    gate_inp_type,
        const float   * __restrict__ exp_probs_b,
        const uint8_t * __restrict__ latent_down_bytes,
        size_t                       latent_down_nb1,
        ggml_type                    latent_down_type,
        const uint8_t * __restrict__ up_shexp_bytes,
        size_t                       up_shexp_nb1,
        ggml_type                    up_shexp_type,
        float         * __restrict__ router_logits,
        int32_t       * __restrict__ router_done,
        int32_t       * __restrict__ expert_ids,
        float         * __restrict__ expert_weights,
        float         * __restrict__ latent_out,
        float         * __restrict__ h_shexp_out,
        int   n_embd,
        int   n_expert,
        int   n_expert_used,
        float w_scale,
        int   moe_latent_size,
        int   ff_shexp,
        int   blocks_router,
        int   blocks_latent) {
    constexpr int RPW = 2;
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int bx      = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.z);

    const float * x_ptr = reinterpret_cast<const float *>(
        cur_bytes + static_cast<size_t>(token) * cur_nb1);

    if (bx < blocks_router) {
        // Router block: never return early so __syncthreads() in epilogue is deadlock-free
        const int row0 = (bx * MMVQ_F32_NWARPS_SM + warp_id) * RPW;
        float * y_row  = router_logits + static_cast<size_t>(token) * n_expert;

        if (row0 < n_expert) {
            const uint8_t * r0 = gate_inp_bytes + static_cast<size_t>(row0) * gate_inp_nb1;
            if (row0 + RPW <= n_expert) {
                float s[RPW];
                if (gate_inp_type == GGML_TYPE_F32) {
                    vec_dot_rows_f32<GGML_TYPE_F32, RPW, false>(r0, gate_inp_nb1, x_ptr, n_embd, lane, s);
                } else if (gate_inp_type == GGML_TYPE_F16) {
                    vec_dot_rows_f32<GGML_TYPE_F16, RPW, false>(r0, gate_inp_nb1, x_ptr, n_embd, lane, s);
                } else {
                    vec_dot_rows_f32<GGML_TYPE_BF16, RPW, false>(r0, gate_inp_nb1, x_ptr, n_embd, lane, s);
                }
                if (lane == 0) {
                    y_row[row0 + 0] = s[0];
                    y_row[row0 + 1] = s[1];
                }
            } else {
                float s1[1];
                if (gate_inp_type == GGML_TYPE_F32) {
                    vec_dot_rows_f32<GGML_TYPE_F32, 1, false>(r0, 0, x_ptr, n_embd, lane, s1);
                } else if (gate_inp_type == GGML_TYPE_F16) {
                    vec_dot_rows_f32<GGML_TYPE_F16, 1, false>(r0, 0, x_ptr, n_embd, lane, s1);
                } else {
                    vec_dot_rows_f32<GGML_TYPE_BF16, 1, false>(r0, 0, x_ptr, n_embd, lane, s1);
                }
                if (lane == 0) {
                    y_row[row0] = s1[0];
                }
            }
        }

        if constexpr (FUSE_TOPK) {
            __shared__ int s_is_last;
            __threadfence(); // Order BOTH warps' router_logits writes to L2 first
            __syncthreads(); // Wait until Warp 1's __threadfence() is complete

            if (warp_id == 0 && lane == 0) {
                const int done = atomicAdd(&router_done[token], 1) + 1;
                s_is_last = (done == blocks_router) ? 1 : 0;
            }
            __syncthreads();

            if (s_is_last && warp_id == 0) {
                __threadfence(); // Acquire visibility of all other SMs' L2 writes before __ldcg
                olala_topk_warp_inline<TOPK_PER_LANE>(
                    y_row,
                    exp_probs_b,
                    expert_ids     + static_cast<size_t>(token) * n_expert_used,
                    expert_weights + static_cast<size_t>(token) * n_expert_used,
                    n_expert, n_expert_used, w_scale, lane);
            }
        }
        return;
    }

    if (bx < blocks_router + blocks_latent) {
        const int row0 = ((bx - blocks_router) * MMVQ_F32_NWARPS_SM + warp_id) * RPW;
        if (row0 >= moe_latent_size) {
            return;
        }
        const uint8_t * r0 = latent_down_bytes + static_cast<size_t>(row0) * latent_down_nb1;
        float * y_row = latent_out + static_cast<size_t>(token) * moe_latent_size;

        if (row0 + RPW <= moe_latent_size) {
            float s[RPW];
            if constexpr (SPECIALIZED) {
                vec_dot_rows_f32<SPEC_LD, RPW, false>(r0, latent_down_nb1, x_ptr, n_embd, lane, s);
            } else {
                vec_dot_dispatch_rows<RPW, false>(latent_down_type, r0, latent_down_nb1, x_ptr, n_embd, lane, s);
            }
            if (lane == 0) {
                y_row[row0 + 0] = s[0];
                y_row[row0 + 1] = s[1];
            }
        } else {
            float s1[1];
            if constexpr (SPECIALIZED) {
                vec_dot_rows_f32<SPEC_LD, 1, false>(r0, 0, x_ptr, n_embd, lane, s1);
            } else {
                vec_dot_dispatch_rows<1, false>(latent_down_type, r0, 0, x_ptr, n_embd, lane, s1);
            }
            if (lane == 0) {
                y_row[row0] = s1[0];
            }
        }
    } else {
        const int row0 = ((bx - blocks_router - blocks_latent) * MMVQ_F32_NWARPS_SM + warp_id) * RPW;
        if (row0 >= ff_shexp) {
            return;
        }
        const uint8_t * r0 = up_shexp_bytes + static_cast<size_t>(row0) * up_shexp_nb1;
        float * y_row = h_shexp_out + static_cast<size_t>(token) * ff_shexp;

        if (row0 + RPW <= ff_shexp) {
            float s[RPW];
            if constexpr (SPECIALIZED) {
                vec_dot_rows_f32<SPEC_SH, RPW, false>(r0, up_shexp_nb1, x_ptr, n_embd, lane, s);
            } else {
                vec_dot_dispatch_rows<RPW, false>(up_shexp_type, r0, up_shexp_nb1, x_ptr, n_embd, lane, s);
            }
            if (lane == 0) {
                y_row[row0 + 0] = act_relu2(s[0]);
                y_row[row0 + 1] = act_relu2(s[1]);
            }
        } else {
            float s1[1];
            if constexpr (SPECIALIZED) {
                vec_dot_rows_f32<SPEC_SH, 1, false>(r0, 0, x_ptr, n_embd, lane, s1);
            } else {
                vec_dot_dispatch_rows<1, false>(up_shexp_type, r0, 0, x_ptr, n_embd, lane, s1);
            }
            if (lane == 0) {
                y_row[row0] = act_relu2(s1[0]);
            }
        }
    }
}

// Fallback Stage 1b kernel only when n_expert > 512 or n_expert_used > 32
__global__ void olala_stage1b_topk_block_kernel(
        const float * __restrict__ router_logits,
        const float * __restrict__ exp_probs_b,
        int32_t     * __restrict__ expert_ids,
        float       * __restrict__ expert_weights,
        int   n_expert,
        int   n_expert_used,
        float w_scale) {
    extern __shared__ float smem[];
    float * s_probs  = smem;
    float * s_scores = smem + n_expert;

    __shared__ float s_red_val[256];
    __shared__ int   s_red_idx[256];

    const int tid   = threadIdx.x;
    const int token = static_cast<int>(blockIdx.x);
    const float * logits = router_logits + static_cast<size_t>(token) * n_expert;

    for (int e = tid; e < n_expert; e += blockDim.x) {
        const float p = 1.0f / (1.0f + expf(-logits[e]));
        const float b = (exp_probs_b != nullptr) ? exp_probs_b[e] : 0.0f;
        s_probs[e]  = p;
        s_scores[e] = p + b;
    }
    __syncthreads();

    int32_t * out_ids = expert_ids     + static_cast<size_t>(token) * n_expert_used;
    float   * out_w   = expert_weights + static_cast<size_t>(token) * n_expert_used;
    float     w_sum   = 0.0f;

    for (int k = 0; k < n_expert_used; ++k) {
        float best_val = -INFINITY;
        int   best_idx = n_expert;
        for (int e = tid; e < n_expert; e += blockDim.x) {
            const float v = s_scores[e];
            if (v > best_val || (v == best_val && e < best_idx)) {
                best_val = v;
                best_idx = e;
            }
        }
        s_red_val[tid] = best_val;
        s_red_idx[tid] = best_idx;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float v2 = s_red_val[tid + stride];
                const int   i2 = s_red_idx[tid + stride];
                if (v2 > s_red_val[tid] || (v2 == s_red_val[tid] && i2 < s_red_idx[tid])) {
                    s_red_val[tid] = v2;
                    s_red_idx[tid] = i2;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int chosen = s_red_idx[0];
            out_ids[k] = (chosen < n_expert) ? chosen : 0;
            const float pw = (chosen < n_expert) ? s_probs[chosen] : 0.0f;
            out_w[k] = pw;
            w_sum += pw;
            if (chosen < n_expert) {
                s_scores[chosen] = -INFINITY;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float scale = w_scale / fmaxf(w_sum, 6.103515625e-5f);
        for (int k = 0; k < n_expert_used; ++k) {
            out_w[k] *= scale;
        }
    }
}

// Stage 3: Routed Experts Up + ReLU^2 (Adaptive 1 / 2 / 4 Rows per Warp, NWARPS=2)
template <ggml_type TYPE_UP, int ROWS_PER_WARP>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 8)
void olala_stage3_routed_up_relu2_kernel(
        const uint8_t * __restrict__ up_exps_bytes,
        size_t                       up_nb1,
        size_t                       up_nb2,
        const float   * __restrict__ latent,
        const int32_t * __restrict__ expert_ids,
        float         * __restrict__ h_exp,
        int moe_latent_size,
        int ff_exp,
        int n_expert,
        int n_expert_used) {
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row0    = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    if (row0 >= ff_exp) {
        return;
    }

    const int slot      = static_cast<int>(blockIdx.y);
    const int token     = static_cast<int>(blockIdx.z);
    const int expert_id = __ldg(expert_ids + token * n_expert_used + slot);
    float * dst_row     = h_exp + (static_cast<size_t>(token) * n_expert_used + slot) * ff_exp;

    if (expert_id < 0 || expert_id >= n_expert) {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (row0 + r < ff_exp) {
                    dst_row[row0 + r] = 0.0f;
                }
            }
        }
        return;
    }

    const float   * x_ptr = latent + static_cast<size_t>(token) * moe_latent_size;
    const uint8_t * r0    = up_exps_bytes + static_cast<size_t>(expert_id) * up_nb2 + static_cast<size_t>(row0) * up_nb1;

    if (row0 + ROWS_PER_WARP <= ff_exp) {
        float s[ROWS_PER_WARP];
        vec_dot_rows_f32<TYPE_UP, ROWS_PER_WARP, false>(r0, up_nb1, x_ptr, moe_latent_size, lane, s);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                dst_row[row0 + r] = act_relu2(s[r]);
            }
        }
    } else {
        for (int r = 0; r < ROWS_PER_WARP && row0 + r < ff_exp; ++r) {
            float s1[1];
            vec_dot_rows_f32<TYPE_UP, 1, false>(r0 + static_cast<size_t>(r) * up_nb1, 0, x_ptr, moe_latent_size, lane, s1);
            if (lane == 0) {
                dst_row[row0 + r] = act_relu2(s1[0]);
            }
        }
    }
}

// Stage 4: Parallel Routed Experts Down (grid.y = n_expert_used, Adaptive 1 / 2 / 4 Rows per Warp, NWARPS=2)
template <ggml_type TYPE_DOWN, int ROWS_PER_WARP>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 8)
void olala_stage4_routed_down_parallel_kernel(
        const uint8_t * __restrict__ down_exps_bytes,
        size_t                       down_nb1,
        size_t                       down_nb2,
        const float   * __restrict__ h_exp,
        const int32_t * __restrict__ expert_ids,
        const float   * __restrict__ expert_weights,
        float         * __restrict__ y_partial,
        int ff_exp,
        int moe_latent_size,
        int n_expert,
        int n_expert_used) {
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row0    = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    if (row0 >= moe_latent_size) {
        return;
    }

    const int slot      = static_cast<int>(blockIdx.y);
    const int token     = static_cast<int>(blockIdx.z);
    const int expert_id = __ldg(expert_ids + token * n_expert_used + slot);
    const float weight  = __ldg(expert_weights + token * n_expert_used + slot);
    float * dst_row     = y_partial + (static_cast<size_t>(token) * n_expert_used + slot) * moe_latent_size;

    if (expert_id < 0 || expert_id >= n_expert) {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if (row0 + r < moe_latent_size) {
                    dst_row[row0 + r] = 0.0f;
                }
            }
        }
        return;
    }

    const float   * x_ptr = h_exp + (static_cast<size_t>(token) * n_expert_used + slot) * ff_exp;
    const uint8_t * r0    = down_exps_bytes + static_cast<size_t>(expert_id) * down_nb2 + static_cast<size_t>(row0) * down_nb1;

    if (row0 + ROWS_PER_WARP <= moe_latent_size) {
        float s[ROWS_PER_WARP];
        vec_dot_rows_f32<TYPE_DOWN, ROWS_PER_WARP, false>(r0, down_nb1, x_ptr, ff_exp, lane, s);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                dst_row[row0 + r] = weight * s[r];
            }
        }
    } else {
        for (int r = 0; r < ROWS_PER_WARP && row0 + r < moe_latent_size; ++r) {
            float s1[1];
            vec_dot_rows_f32<TYPE_DOWN, 1, false>(r0 + static_cast<size_t>(r) * down_nb1, 0, x_ptr, ff_exp, lane, s1);
            if (lane == 0) {
                dst_row[row0 + r] = weight * s1[0];
            }
        }
    }
}

// Stage 5: Fused 6-Slot Sum + ffn_latent_up + ffn_down_shexp -> Single Write to dst
// Compile-time specialization for SPEC_LU / SPEC_DS eliminates 7-way switch register bloat.
template <bool SPECIALIZED, ggml_type SPEC_LU, ggml_type SPEC_DS, int ROWS_PER_WARP>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 8)
void olala_stage5_fused_out_kernel(
        const uint8_t * __restrict__ latent_up_bytes,
        size_t                       lu_nb1,
        ggml_type                    lu_type,
        const float   * __restrict__ y_partial,
        const uint8_t * __restrict__ down_shexp_bytes,
        size_t                       ds_nb1,
        ggml_type                    ds_type,
        const float   * __restrict__ h_shexp,
        uint8_t       * __restrict__ dst_bytes,
        size_t                       dst_nb1,
        int moe_latent_size,
        int ff_shexp,
        int n_embd,
        int n_expert_used,
        bool has_shexp) {
    extern __shared__ float s_y[];

    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid     = warp_id * MMVQ_F32_WARP_SIZE + lane;
    const int token   = static_cast<int>(blockIdx.z);

    // Cooperatively reduce 6 L2-resident weighted slots into s_y using vectorized float4
    const float * tok_base = y_partial + static_cast<size_t>(token) * n_expert_used * moe_latent_size;
    const int k4_max = moe_latent_size >> 2;
    const float4 * tok_base4 = reinterpret_cast<const float4 *>(tok_base);
    float4       * s_y4      = reinterpret_cast<float4 *>(s_y);
    for (int k4 = tid; k4 < k4_max; k4 += MMVQ_F32_THREADS_SM) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 6
        for (int s = 0; s < n_expert_used; ++s) {
            const float4 v = __ldg(tok_base4 + static_cast<size_t>(s) * k4_max + k4);
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
        s_y4[k4] = sum;
    }
    __syncthreads();

    const int row0 = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    if (row0 >= n_embd) {
        return;
    }

    const uint8_t * r0_lu  = latent_up_bytes  + static_cast<size_t>(row0) * lu_nb1;
    const uint8_t * r0_ds  = has_shexp ? (down_shexp_bytes + static_cast<size_t>(row0) * ds_nb1) : nullptr;
    const float   * x_shex = has_shexp ? (h_shexp + static_cast<size_t>(token) * ff_shexp) : nullptr;
    float         * dst_row = reinterpret_cast<float *>(dst_bytes + static_cast<size_t>(token) * dst_nb1);

    if (row0 + ROWS_PER_WARP <= n_embd) {
        float s[ROWS_PER_WARP];
        if constexpr (SPECIALIZED) {
            vec_dot_rows_f32<SPEC_LU, ROWS_PER_WARP, true>(r0_lu, lu_nb1, s_y, moe_latent_size, lane, s);
            if (has_shexp) {
                float t[ROWS_PER_WARP];
                vec_dot_rows_f32<SPEC_DS, ROWS_PER_WARP, false>(r0_ds, ds_nb1, x_shex, ff_shexp, lane, t);
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    s[r] += t[r];
                }
            }
        } else {
            vec_dot_dispatch_rows<ROWS_PER_WARP, true>(lu_type, r0_lu, lu_nb1, s_y, moe_latent_size, lane, s);
            if (has_shexp) {
                float t[ROWS_PER_WARP];
                vec_dot_dispatch_rows<ROWS_PER_WARP, false>(ds_type, r0_ds, ds_nb1, x_shex, ff_shexp, lane, t);
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    s[r] += t[r];
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                dst_row[row0 + r] = s[r];
            }
        }
    } else {
        for (int r = 0; r < ROWS_PER_WARP && row0 + r < n_embd; ++r) {
            float s1[1];
            if constexpr (SPECIALIZED) {
                vec_dot_rows_f32<SPEC_LU, 1, true>(r0_lu + static_cast<size_t>(r) * lu_nb1, 0, s_y, moe_latent_size, lane, s1);
                if (has_shexp) {
                    float t1[1];
                    vec_dot_rows_f32<SPEC_DS, 1, false>(r0_ds + static_cast<size_t>(r) * ds_nb1, 0, x_shex, ff_shexp, lane, t1);
                    s1[0] += t1[0];
                }
            } else {
                vec_dot_dispatch_rows<1, true>(lu_type, r0_lu + static_cast<size_t>(r) * lu_nb1, 0, s_y, moe_latent_size, lane, s1);
                if (has_shexp) {
                    float t1[1];
                    vec_dot_dispatch_rows<1, false>(ds_type, r0_ds + static_cast<size_t>(r) * ds_nb1, 0, x_shex, ff_shexp, lane, t1);
                    s1[0] += t1[0];
                }
            }
            if (lane == 0) {
                dst_row[row0 + r] = s1[0];
            }
        }
    }
}

// --- Launch Helpers ---

template <ggml_type TYPE_UP>
void launch_stage3(
        cudaStream_t stream,
        const ggml_tensor * up_exps,
        const float * latent,
        const int32_t * expert_ids,
        float * h_exp,
        int moe_latent_size, int ff_exp, int n_expert, int n_expert_used, int n_tokens) {
    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);
    const int bx4 = (ff_exp + (MMVQ_F32_NWARPS_SM * 4) - 1) / (MMVQ_F32_NWARPS_SM * 4);
    const int bx2 = (ff_exp + (MMVQ_F32_NWARPS_SM * 2) - 1) / (MMVQ_F32_NWARPS_SM * 2);
    const int bx1 = (ff_exp + MMVQ_F32_NWARPS_SM - 1) / MMVQ_F32_NWARPS_SM;

    if (bx4 * n_expert_used * n_tokens >= 224) {
        const dim3 grid(static_cast<unsigned int>(bx4), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        olala_stage3_routed_up_relu2_kernel<TYPE_UP, 4><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(up_exps->data), up_exps->nb[1], up_exps->nb[2],
            latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used);
    } else if (bx2 * n_expert_used * n_tokens >= 112) {
        const dim3 grid(static_cast<unsigned int>(bx2), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        olala_stage3_routed_up_relu2_kernel<TYPE_UP, 2><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(up_exps->data), up_exps->nb[1], up_exps->nb[2],
            latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used);
    } else {
        const dim3 grid(static_cast<unsigned int>(bx1), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        olala_stage3_routed_up_relu2_kernel<TYPE_UP, 1><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(up_exps->data), up_exps->nb[1], up_exps->nb[2],
            latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used);
    }
}

template <ggml_type TYPE_DOWN>
void launch_stage4(
        cudaStream_t stream,
        const ggml_tensor * down_exps,
        const float * h_exp,
        const int32_t * expert_ids,
        const float * expert_weights,
        float * y_partial,
        int ff_exp, int moe_latent_size, int n_expert, int n_expert_used, int n_tokens) {
    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);
    const int bx4 = (moe_latent_size + (MMVQ_F32_NWARPS_SM * 4) - 1) / (MMVQ_F32_NWARPS_SM * 4);
    const int bx2 = (moe_latent_size + (MMVQ_F32_NWARPS_SM * 2) - 1) / (MMVQ_F32_NWARPS_SM * 2);
    const int bx1 = (moe_latent_size + MMVQ_F32_NWARPS_SM - 1) / MMVQ_F32_NWARPS_SM;

    if (bx4 * n_expert_used * n_tokens >= 224) {
        const dim3 grid(static_cast<unsigned int>(bx4), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        olala_stage4_routed_down_parallel_kernel<TYPE_DOWN, 4><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(down_exps->data), down_exps->nb[1], down_exps->nb[2],
            h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used);
    } else if (bx2 * n_expert_used * n_tokens >= 112) {
        const dim3 grid(static_cast<unsigned int>(bx2), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        olala_stage4_routed_down_parallel_kernel<TYPE_DOWN, 2><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(down_exps->data), down_exps->nb[1], down_exps->nb[2],
            h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used);
    } else {
        const dim3 grid(static_cast<unsigned int>(bx1), static_cast<unsigned int>(n_expert_used), static_cast<unsigned int>(n_tokens));
        olala_stage4_routed_down_parallel_kernel<TYPE_DOWN, 1><<<grid, block, 0, stream>>>(
            static_cast<const uint8_t *>(down_exps->data), down_exps->nb[1], down_exps->nb[2],
            h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used);
    }
}

template <bool SPECIALIZED, ggml_type SPEC_LU, ggml_type SPEC_DS>
void launch_stage5_specialized(
        cudaStream_t stream,
        const uint8_t * latent_up_bytes, size_t lu_nb1, ggml_type lu_type,
        const float * y_partial,
        const uint8_t * down_shexp_bytes, size_t ds_nb1, ggml_type ds_type,
        const float * h_shexp,
        uint8_t * dst_bytes, size_t dst_nb1,
        int moe_latent_size, int ff_shexp, int n_embd, int n_expert_used, int n_tokens, bool has_shexp) {
    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);
    const int bx4 = (n_embd + (MMVQ_F32_NWARPS_SM * 4) - 1) / (MMVQ_F32_NWARPS_SM * 4);
    const int bx2 = (n_embd + (MMVQ_F32_NWARPS_SM * 2) - 1) / (MMVQ_F32_NWARPS_SM * 2);
    const int bx1 = (n_embd + MMVQ_F32_NWARPS_SM - 1) / MMVQ_F32_NWARPS_SM;
    const size_t smem = static_cast<size_t>(moe_latent_size) * sizeof(float);

    if (bx4 * n_tokens >= 224) {
        const dim3 grid(static_cast<unsigned int>(bx4), 1, static_cast<unsigned int>(n_tokens));
        olala_stage5_fused_out_kernel<SPECIALIZED, SPEC_LU, SPEC_DS, 4><<<grid, block, smem, stream>>>(
            latent_up_bytes, lu_nb1, lu_type, y_partial,
            down_shexp_bytes, ds_nb1, ds_type, h_shexp,
            dst_bytes, dst_nb1, moe_latent_size, ff_shexp, n_embd, n_expert_used, has_shexp);
    } else if (bx2 * n_tokens >= 112) {
        const dim3 grid(static_cast<unsigned int>(bx2), 1, static_cast<unsigned int>(n_tokens));
        olala_stage5_fused_out_kernel<SPECIALIZED, SPEC_LU, SPEC_DS, 2><<<grid, block, smem, stream>>>(
            latent_up_bytes, lu_nb1, lu_type, y_partial,
            down_shexp_bytes, ds_nb1, ds_type, h_shexp,
            dst_bytes, dst_nb1, moe_latent_size, ff_shexp, n_embd, n_expert_used, has_shexp);
    } else {
        const dim3 grid(static_cast<unsigned int>(bx1), 1, static_cast<unsigned int>(n_tokens));
        olala_stage5_fused_out_kernel<SPECIALIZED, SPEC_LU, SPEC_DS, 1><<<grid, block, smem, stream>>>(
            latent_up_bytes, lu_nb1, lu_type, y_partial,
            down_shexp_bytes, ds_nb1, ds_type, h_shexp,
            dst_bytes, dst_nb1, moe_latent_size, ff_shexp, n_embd, n_expert_used, has_shexp);
    }
}

// ============================================================================
// P1: Batched Multi-Token Weight-Reuse & GPU-Grouped MoE (n_tokens = 9..512)
// ============================================================================

template <ggml_type TYPE, int ROWS_PER_WARP, int BATCH_T>
__device__ __forceinline__ void vec_dot_rows_f32_batched(
        const uint8_t * __restrict__ row0_ptr,
        size_t                       row_stride,
        const float * const (&x_ptrs)[BATCH_T],
        int K,
        int lane,
        float (&out)[BATCH_T][ROWS_PER_WARP]) {
    float acc[BATCH_T][ROWS_PER_WARP];
#pragma unroll
    for (int b = 0; b < BATCH_T; ++b) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            acc[b][r] = 0.0f;
        }
    }

    if constexpr (TYPE == GGML_TYPE_Q5_K) {
        const int nb = K / QK_K;
        const int c  = lane >> 3;
        const int t  = lane & 7;

        for (int ib = 0; ib < nb; ++ib) {
            const int base_k = ib * QK_K + 64 * c + 4 * t;
            float4 x_lo[BATCH_T], x_hi[BATCH_T];
            float  sum_lo[BATCH_T], sum_hi[BATCH_T];
#pragma unroll
            for (int b = 0; b < BATCH_T; ++b) {
                x_lo[b]   = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b] + base_k));
                x_hi[b]   = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b] + base_k + 32));
                sum_lo[b] = (x_lo[b].x + x_lo[b].y) + (x_lo[b].z + x_lo[b].w);
                sum_hi[b] = (x_hi[b].x + x_hi[b].y) + (x_hi[b].z + x_hi[b].w);
            }

            uint4    hdr[ROWS_PER_WARP];
            uint32_t qs_w[ROWS_PER_WARP];
            uint32_t qh_w[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q5_K & blk = reinterpret_cast<const block_q5_K *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                hdr[r]  = __ldg(reinterpret_cast<const uint4 *>(&blk));
                qs_w[r] = __ldg(reinterpret_cast<const uint32_t *>(blk.qs + 32 * c + 4 * t));
                qh_w[r] = __ldg(reinterpret_cast<const uint32_t *>(blk.qh + 4 * t));
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                float d1, min1, d2, min2;
                unpack_k4_scales_u4(hdr[r], c, d1, min1, d2, min2);
                const uint32_t hb1  = ((c <= 2) ? (qh_w[r] << (4 - 2 * c)) : (qh_w[r] >> 2)) & 0x10101010u;
                const uint32_t hb2  = ((c <  2) ? (qh_w[r] << (3 - 2 * c)) : (qh_w[r] >> (2 * c - 3))) & 0x10101010u;
                const uint32_t q1_4 = ( qs_w[r]       & 0x0F0F0F0Fu) | hb1;
                const uint32_t q2_4 = ((qs_w[r] >> 4) & 0x0F0F0F0Fu) | hb2;

                const float q10 = static_cast<float>( q1_4        & 0xFFu);
                const float q11 = static_cast<float>((q1_4 >>  8) & 0xFFu);
                const float q12 = static_cast<float>((q1_4 >> 16) & 0xFFu);
                const float q13 = static_cast<float>( q1_4 >> 24         );
                const float q20 = static_cast<float>( q2_4        & 0xFFu);
                const float q21 = static_cast<float>((q2_4 >>  8) & 0xFFu);
                const float q22 = static_cast<float>((q2_4 >> 16) & 0xFFu);
                const float q23 = static_cast<float>( q2_4 >> 24         );

#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    const float qx1 = fmaf(q10, x_lo[b].x, fmaf(q11, x_lo[b].y, fmaf(q12, x_lo[b].z, q13 * x_lo[b].w)));
                    const float qx2 = fmaf(q20, x_hi[b].x, fmaf(q21, x_hi[b].y, fmaf(q22, x_hi[b].z, q23 * x_hi[b].w)));
                    acc[b][r] = fmaf(d1, qx1, fmaf(-min1, sum_lo[b], acc[b][r]));
                    acc[b][r] = fmaf(d2, qx2, fmaf(-min2, sum_hi[b], acc[b][r]));
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_Q4_K) {
        const int nb = K / QK_K;
        const int c  = lane >> 3;
        const int t  = lane & 7;

        for (int ib = 0; ib < nb; ++ib) {
            const int base_k = ib * QK_K + 64 * c + 4 * t;
            float4 x_lo[BATCH_T], x_hi[BATCH_T];
            float  sum_lo[BATCH_T], sum_hi[BATCH_T];
#pragma unroll
            for (int b = 0; b < BATCH_T; ++b) {
                x_lo[b]   = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b] + base_k));
                x_hi[b]   = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b] + base_k + 32));
                sum_lo[b] = (x_lo[b].x + x_lo[b].y) + (x_lo[b].z + x_lo[b].w);
                sum_hi[b] = (x_hi[b].x + x_hi[b].y) + (x_hi[b].z + x_hi[b].w);
            }

            uint4    hdr[ROWS_PER_WARP];
            uint32_t qs_w[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q4_K & blk = reinterpret_cast<const block_q4_K *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                hdr[r]  = __ldg(reinterpret_cast<const uint4 *>(&blk));
                qs_w[r] = __ldg(reinterpret_cast<const uint32_t *>(blk.qs + 32 * c + 4 * t));
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                float d1, min1, d2, min2;
                unpack_k4_scales_u4(hdr[r], c, d1, min1, d2, min2);
                const uint32_t q1_4 =  qs_w[r]       & 0x0F0F0F0Fu;
                const uint32_t q2_4 = (qs_w[r] >> 4) & 0x0F0F0F0Fu;

                const float q10 = static_cast<float>( q1_4        & 0xFFu);
                const float q11 = static_cast<float>((q1_4 >>  8) & 0xFFu);
                const float q12 = static_cast<float>((q1_4 >> 16) & 0xFFu);
                const float q13 = static_cast<float>( q1_4 >> 24         );
                const float q20 = static_cast<float>( q2_4        & 0xFFu);
                const float q21 = static_cast<float>((q2_4 >>  8) & 0xFFu);
                const float q22 = static_cast<float>((q2_4 >> 16) & 0xFFu);
                const float q23 = static_cast<float>( q2_4 >> 24         );

#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    const float qx1 = fmaf(q10, x_lo[b].x, fmaf(q11, x_lo[b].y, fmaf(q12, x_lo[b].z, q13 * x_lo[b].w)));
                    const float qx2 = fmaf(q20, x_hi[b].x, fmaf(q21, x_hi[b].y, fmaf(q22, x_hi[b].z, q23 * x_hi[b].w)));
                    acc[b][r] = fmaf(d1, qx1, fmaf(-min1, sum_lo[b], acc[b][r]));
                    acc[b][r] = fmaf(d2, qx2, fmaf(-min2, sum_hi[b], acc[b][r]));
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_Q6_K) {
        const int nb = K / QK_K;
        const int h  = lane >> 4;
        const int t  = lane & 15;
        const int is = t >> 3;

        for (int ib = 0; ib < nb; ++ib) {
            const int n = ib * QK_K + 128 * h + 2 * t;
            float2 x0[BATCH_T], x1[BATCH_T], x2[BATCH_T], x3[BATCH_T];
#pragma unroll
            for (int b = 0; b < BATCH_T; ++b) {
                x0[b] = __ldg(reinterpret_cast<const float2 *>(x_ptrs[b] + n +  0));
                x1[b] = __ldg(reinterpret_cast<const float2 *>(x_ptrs[b] + n + 32));
                x2[b] = __ldg(reinterpret_cast<const float2 *>(x_ptrs[b] + n + 64));
                x3[b] = __ldg(reinterpret_cast<const float2 *>(x_ptrs[b] + n + 96));
            }
            float    ds0[ROWS_PER_WARP], ds2[ROWS_PER_WARP], ds4[ROWS_PER_WARP], ds6[ROWS_PER_WARP];
            uint16_t ql0[ROWS_PER_WARP], ql1[ROWS_PER_WARP], qh0[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q6_K & blk = reinterpret_cast<const block_q6_K *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                const float d = __half2float(__ldg(&blk.d));
                const int8_t * __restrict__ sc = blk.scales + 8 * h + is;
                ds0[r] = d * static_cast<float>(__ldg(sc + 0));
                ds2[r] = d * static_cast<float>(__ldg(sc + 2));
                ds4[r] = d * static_cast<float>(__ldg(sc + 4));
                ds6[r] = d * static_cast<float>(__ldg(sc + 6));
                ql0[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.ql + 64 * h + 2 * t));
                ql1[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.ql + 64 * h + 32 + 2 * t));
                qh0[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.qh + 32 * h + 2 * t));
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const uint32_t ql0_0 = ql0[r] & 0xFFu, ql0_1 = ql0[r] >> 8;
                const uint32_t ql1_0 = ql1[r] & 0xFFu, ql1_1 = ql1[r] >> 8;
                const uint32_t qh_0  = qh0[r] & 0xFFu, qh_1  = qh0[r] >> 8;

                const float q00 = static_cast<float>(static_cast<int>((ql0_0 & 0xFu) | ((qh_0 & 0x03u) << 4)) - 32);
                const float q01 = static_cast<float>(static_cast<int>((ql0_1 & 0xFu) | ((qh_1 & 0x03u) << 4)) - 32);
                const float q10 = static_cast<float>(static_cast<int>((ql1_0 & 0xFu) | ((qh_0 & 0x0Cu) << 2)) - 32);
                const float q11 = static_cast<float>(static_cast<int>((ql1_1 & 0xFu) | ((qh_1 & 0x0Cu) << 2)) - 32);
                const float q20 = static_cast<float>(static_cast<int>((ql0_0 >> 4)   | ((qh_0 & 0x30u)     )) - 32);
                const float q21 = static_cast<float>(static_cast<int>((ql0_1 >> 4)   | ((qh_1 & 0x30u)     )) - 32);
                const float q30 = static_cast<float>(static_cast<int>((ql1_0 >> 4)   | ((qh_0 & 0xC0u) >> 2)) - 32);
                const float q31 = static_cast<float>(static_cast<int>((ql1_1 >> 4)   | ((qh_1 & 0xC0u) >> 2)) - 32);

#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    acc[b][r] = fmaf(ds0[r], fmaf(q00, x0[b].x, q01 * x0[b].y), acc[b][r]);
                    acc[b][r] = fmaf(ds2[r], fmaf(q10, x1[b].x, q11 * x1[b].y), acc[b][r]);
                    acc[b][r] = fmaf(ds4[r], fmaf(q20, x2[b].x, q21 * x2[b].y), acc[b][r]);
                    acc[b][r] = fmaf(ds6[r], fmaf(q30, x3[b].x, q31 * x3[b].y), acc[b][r]);
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_Q8_0) {
        const int nb    = K / QK8_0;
        const int b_off = lane >> 3;
        const int t     = lane & 7;

        for (int ib = b_off; ib < nb; ib += 4) {
            float4 xb[BATCH_T];
#pragma unroll
            for (int b = 0; b < BATCH_T; ++b) {
                xb[b] = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b] + ib * QK8_0 + 4 * t));
            }
            float    dv[ROWS_PER_WARP];
            uint16_t q01[ROWS_PER_WARP], q23[ROWS_PER_WARP];
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const block_q8_0 & blk = reinterpret_cast<const block_q8_0 *>(row0_ptr + static_cast<size_t>(r) * row_stride)[ib];
                dv[r]  = __half2float(__ldg(&blk.d));
                q01[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.qs + 4 * t));
                q23[r] = __ldg(reinterpret_cast<const uint16_t *>(blk.qs + 4 * t + 2));
            }
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const float q0 = static_cast<float>(static_cast<int8_t>(q01[r] & 0xFFu));
                const float q1 = static_cast<float>(static_cast<int8_t>(q01[r] >> 8));
                const float q2 = static_cast<float>(static_cast<int8_t>(q23[r] & 0xFFu));
                const float q3 = static_cast<float>(static_cast<int8_t>(q23[r] >> 8));
#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    const float p = fmaf(q0, xb[b].x, fmaf(q1, xb[b].y, fmaf(q2, xb[b].z, q3 * xb[b].w)));
                    acc[b][r] = fmaf(dv[r], p, acc[b][r]);
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_F32) {
        bool aligned = true;
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if ((reinterpret_cast<uintptr_t>(row0_ptr + static_cast<size_t>(r) * row_stride) & 15u) != 0u) {
                aligned = false;
            }
        }
        if (aligned) {
            const int k4 = K >> 2;
            for (int i4 = lane; i4 < k4; i4 += MMVQ_F32_WARP_SIZE) {
                float4 xb[BATCH_T];
#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    xb[b] = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b]) + i4);
                }
                float4 wv[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    wv[r] = __ldg(reinterpret_cast<const float4 *>(row0_ptr + static_cast<size_t>(r) * row_stride) + i4);
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
#pragma unroll
                    for (int b = 0; b < BATCH_T; ++b) {
                        acc[b][r] = fmaf(wv[r].x, xb[b].x, fmaf(wv[r].y, xb[b].y, fmaf(wv[r].z, xb[b].z, fmaf(wv[r].w, xb[b].w, acc[b][r]))));
                    }
                }
            }
        } else {
            for (int i = lane; i < K; i += MMVQ_F32_WARP_SIZE) {
                float xb[BATCH_T];
#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    xb[b] = __ldg(x_ptrs[b] + i);
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const float w = __ldg(reinterpret_cast<const float *>(row0_ptr + static_cast<size_t>(r) * row_stride) + i);
#pragma unroll
                    for (int b = 0; b < BATCH_T; ++b) {
                        acc[b][r] = fmaf(w, xb[b], acc[b][r]);
                    }
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_F16) {
        bool aligned = true;
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if ((reinterpret_cast<uintptr_t>(row0_ptr + static_cast<size_t>(r) * row_stride) & 7u) != 0u) {
                aligned = false;
            }
        }
        if (aligned) {
            const int k4 = K >> 2;
            for (int i4 = lane; i4 < k4; i4 += MMVQ_F32_WARP_SIZE) {
                float4 xb[BATCH_T];
#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    xb[b] = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b]) + i4);
                }
                uint2 hv[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    hv[r] = __ldg(reinterpret_cast<const uint2 *>(row0_ptr + static_cast<size_t>(r) * row_stride) + i4);
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const float2 w01 = __half22float2(*reinterpret_cast<const half2 *>(&hv[r].x));
                    const float2 w23 = __half22float2(*reinterpret_cast<const half2 *>(&hv[r].y));
#pragma unroll
                    for (int b = 0; b < BATCH_T; ++b) {
                        acc[b][r] = fmaf(w01.x, xb[b].x, fmaf(w01.y, xb[b].y, fmaf(w23.x, xb[b].z, fmaf(w23.y, xb[b].w, acc[b][r]))));
                    }
                }
            }
        } else {
            for (int i = lane; i < K; i += MMVQ_F32_WARP_SIZE) {
                float xb[BATCH_T];
#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    xb[b] = __ldg(x_ptrs[b] + i);
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const float w = __half2float(__ldg(reinterpret_cast<const half *>(row0_ptr + static_cast<size_t>(r) * row_stride) + i));
#pragma unroll
                    for (int b = 0; b < BATCH_T; ++b) {
                        acc[b][r] = fmaf(w, xb[b], acc[b][r]);
                    }
                }
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_BF16) {
        bool aligned = true;
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if ((reinterpret_cast<uintptr_t>(row0_ptr + static_cast<size_t>(r) * row_stride) & 7u) != 0u) {
                aligned = false;
            }
        }
        if (aligned) {
            const int k4 = K >> 2;
            for (int i4 = lane; i4 < k4; i4 += MMVQ_F32_WARP_SIZE) {
                float4 xb[BATCH_T];
#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    xb[b] = __ldg(reinterpret_cast<const float4 *>(x_ptrs[b]) + i4);
                }
                uint2 hv[ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    hv[r] = __ldg(reinterpret_cast<const uint2 *>(row0_ptr + static_cast<size_t>(r) * row_stride) + i4);
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const float w0 = bf16_bits_to_f32(static_cast<uint16_t>(hv[r].x & 0xFFFFu));
                    const float w1 = bf16_bits_to_f32(static_cast<uint16_t>(hv[r].x >> 16));
                    const float w2 = bf16_bits_to_f32(static_cast<uint16_t>(hv[r].y & 0xFFFFu));
                    const float w3 = bf16_bits_to_f32(static_cast<uint16_t>(hv[r].y >> 16));
#pragma unroll
                    for (int b = 0; b < BATCH_T; ++b) {
                        acc[b][r] = fmaf(w0, xb[b].x, fmaf(w1, xb[b].y, fmaf(w2, xb[b].z, fmaf(w3, xb[b].w, acc[b][r]))));
                    }
                }
            }
        } else {
            for (int i = lane; i < K; i += MMVQ_F32_WARP_SIZE) {
                float xb[BATCH_T];
#pragma unroll
                for (int b = 0; b < BATCH_T; ++b) {
                    xb[b] = __ldg(x_ptrs[b] + i);
                }
#pragma unroll
                for (int r = 0; r < ROWS_PER_WARP; ++r) {
                    const float w = bf16_bits_to_f32(__ldg(reinterpret_cast<const uint16_t *>(row0_ptr + static_cast<size_t>(r) * row_stride) + i));
#pragma unroll
                    for (int b = 0; b < BATCH_T; ++b) {
                        acc[b][r] = fmaf(w, xb[b], acc[b][r]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int b = 0; b < BATCH_T; ++b) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            out[b][r] = warp_sum_f32(acc[b][r]);
        }
    }
}

__global__ void olala_group_experts_kernel(
        const int32_t * __restrict__ expert_ids,
        int32_t       * __restrict__ exp_offsets,
        int32_t       * __restrict__ sorted_pairs,
        int n_tokens,
        int n_expert,
        int n_expert_used) {
    __shared__ int32_t s_count[512];
    __shared__ int32_t s_cursor[512];

    const int tid   = threadIdx.x;
    const int total = n_tokens * n_expert_used;

    for (int e = tid; e < n_expert; e += blockDim.x) {
        s_count[e] = 0;
    }
    __syncthreads();

    for (int i = tid; i < total; i += blockDim.x) {
        const int e = expert_ids[i];
        if (e >= 0 && e < n_expert) {
            atomicAdd(&s_count[e], 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int32_t acc = 0;
        for (int e = 0; e < n_expert; ++e) {
            exp_offsets[e] = acc;
            s_cursor[e]    = acc;
            acc += s_count[e];
        }
        exp_offsets[n_expert] = acc;
    }
    __syncthreads();

    for (int i = tid; i < total; i += blockDim.x) {
        const int e = expert_ids[i];
        if (e >= 0 && e < n_expert) {
            const int pos   = atomicAdd(&s_cursor[e], 1);
            const int token = i / n_expert_used;
            const int slot  = i - token * n_expert_used;
            sorted_pairs[pos] = (token << 16) | slot;
        }
    }
}

template <ggml_type TYPE_UP, int ROWS_PER_WARP, int TILE_T = 4>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 6)
void olala_stage3_grouped_kernel(
        const uint8_t * __restrict__ up_exps_bytes,
        size_t up_nb1, size_t up_nb2,
        const float   * __restrict__ latent,
        const int32_t * __restrict__ exp_offsets,
        const int32_t * __restrict__ sorted_pairs,
        float         * __restrict__ h_exp,
        int moe_latent_size, int ff_exp, int n_expert_used) {
    const int e     = static_cast<int>(blockIdx.y);
    const int begin = __ldg(exp_offsets + e);
    const int end   = __ldg(exp_offsets + e + 1);
    if (begin >= end) return;

    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row0    = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    if (row0 >= ff_exp) return;

    const uint8_t * r0 = up_exps_bytes + static_cast<size_t>(e) * up_nb2 + static_cast<size_t>(row0) * up_nb1;
    const int n_rows = min(ROWS_PER_WARP, ff_exp - row0);

    for (int p = begin; p < end; p += TILE_T) {
        const int cnt = min(TILE_T, end - p);
        const float * x_ptrs[TILE_T];
        int pairs[TILE_T];
#pragma unroll
        for (int b = 0; b < TILE_T; ++b) {
            const int pair = (b < cnt) ? __ldg(sorted_pairs + p + b) : __ldg(sorted_pairs + p);
            pairs[b]  = pair;
            x_ptrs[b] = latent + static_cast<size_t>(pair >> 16) * moe_latent_size;
        }
        float out[TILE_T][ROWS_PER_WARP];
        if (n_rows == ROWS_PER_WARP) {
            vec_dot_rows_f32_batched<TYPE_UP, ROWS_PER_WARP, TILE_T>(r0, up_nb1, x_ptrs, moe_latent_size, lane, out);
        } else {
            for (int r = 0; r < n_rows; ++r) {
                float out1[TILE_T][1];
                vec_dot_rows_f32_batched<TYPE_UP, 1, TILE_T>(r0 + static_cast<size_t>(r) * up_nb1, 0, x_ptrs, moe_latent_size, lane, out1);
#pragma unroll
                for (int b = 0; b < TILE_T; ++b) out[b][r] = out1[b][0];
            }
        }
        if (lane == 0) {
            for (int b = 0; b < cnt; ++b) {
                const int tok  = pairs[b] >> 16;
                const int slot = pairs[b] & 0xFFFF;
                float * dst_row = h_exp + (static_cast<size_t>(tok) * n_expert_used + slot) * ff_exp;
                for (int r = 0; r < n_rows; ++r) {
                    dst_row[row0 + r] = act_relu2(out[b][r]);
                }
            }
        }
    }
}

template <ggml_type TYPE_DOWN, int ROWS_PER_WARP, int TILE_T = 4>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 6)
void olala_stage4_grouped_kernel(
        const uint8_t * __restrict__ down_exps_bytes,
        size_t down_nb1, size_t down_nb2,
        const float   * __restrict__ h_exp,
        const float   * __restrict__ expert_weights,
        const int32_t * __restrict__ exp_offsets,
        const int32_t * __restrict__ sorted_pairs,
        float         * __restrict__ y_partial,
        int ff_exp, int moe_latent_size, int n_expert_used) {
    const int e     = static_cast<int>(blockIdx.y);
    const int begin = __ldg(exp_offsets + e);
    const int end   = __ldg(exp_offsets + e + 1);
    if (begin >= end) return;

    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row0    = (static_cast<int>(blockIdx.x) * MMVQ_F32_NWARPS_SM + warp_id) * ROWS_PER_WARP;
    if (row0 >= moe_latent_size) return;

    const uint8_t * r0 = down_exps_bytes + static_cast<size_t>(e) * down_nb2 + static_cast<size_t>(row0) * down_nb1;
    const int n_rows = min(ROWS_PER_WARP, moe_latent_size - row0);

    for (int p = begin; p < end; p += TILE_T) {
        const int cnt = min(TILE_T, end - p);
        const float * x_ptrs[TILE_T];
        int   pairs[TILE_T];
        float weights[TILE_T];
#pragma unroll
        for (int b = 0; b < TILE_T; ++b) {
            const int pair = (b < cnt) ? __ldg(sorted_pairs + p + b) : __ldg(sorted_pairs + p);
            const int tok  = pair >> 16;
            const int slot = pair & 0xFFFF;
            const size_t idx = static_cast<size_t>(tok) * n_expert_used + slot;
            pairs[b]   = pair;
            weights[b] = __ldg(expert_weights + idx);
            x_ptrs[b]  = h_exp + idx * ff_exp;
        }
        float out[TILE_T][ROWS_PER_WARP];
        if (n_rows == ROWS_PER_WARP) {
            vec_dot_rows_f32_batched<TYPE_DOWN, ROWS_PER_WARP, TILE_T>(r0, down_nb1, x_ptrs, ff_exp, lane, out);
        } else {
            for (int r = 0; r < n_rows; ++r) {
                float out1[TILE_T][1];
                vec_dot_rows_f32_batched<TYPE_DOWN, 1, TILE_T>(r0 + static_cast<size_t>(r) * down_nb1, 0, x_ptrs, ff_exp, lane, out1);
#pragma unroll
                for (int b = 0; b < TILE_T; ++b) out[b][r] = out1[b][0];
            }
        }
        if (lane == 0) {
            for (int b = 0; b < cnt; ++b) {
                const int tok  = pairs[b] >> 16;
                const int slot = pairs[b] & 0xFFFF;
                float * dst_row = y_partial + (static_cast<size_t>(tok) * n_expert_used + slot) * moe_latent_size;
                for (int r = 0; r < n_rows; ++r) {
                    dst_row[row0 + r] = (weights[b] == 0.0f) ? 0.0f : (weights[b] * out[b][r]);
                }
            }
        }
    }
}

template <ggml_type TYPE_UP>
static void launch_stage3_grouped(
        cudaStream_t stream, const ggml_tensor * up_exps, const float * latent,
        const int32_t * exp_offsets, const int32_t * sorted_pairs, float * h_exp,
        int moe_latent_size, int ff_exp, int n_expert, int n_expert_used) {
    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);
    const int bx4 = (ff_exp + (MMVQ_F32_NWARPS_SM * 4) - 1) / (MMVQ_F32_NWARPS_SM * 4);
    const int bx2 = (ff_exp + (MMVQ_F32_NWARPS_SM * 2) - 1) / (MMVQ_F32_NWARPS_SM * 2);
    if (bx4 * n_expert >= 224) {
        olala_stage3_grouped_kernel<TYPE_UP, 4, 4><<<dim3(bx4, n_expert, 1), block, 0, stream>>>(
            static_cast<const uint8_t *>(up_exps->data), up_exps->nb[1], up_exps->nb[2],
            latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert_used);
    } else {
        olala_stage3_grouped_kernel<TYPE_UP, 2, 4><<<dim3(bx2, n_expert, 1), block, 0, stream>>>(
            static_cast<const uint8_t *>(up_exps->data), up_exps->nb[1], up_exps->nb[2],
            latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert_used);
    }
}

template <ggml_type TYPE_DOWN>
static void launch_stage4_grouped(
        cudaStream_t stream, const ggml_tensor * down_exps, const float * h_exp,
        const float * expert_weights, const int32_t * exp_offsets, const int32_t * sorted_pairs,
        float * y_partial, int ff_exp, int moe_latent_size, int n_expert, int n_expert_used) {
    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);
    const int bx4 = (moe_latent_size + (MMVQ_F32_NWARPS_SM * 4) - 1) / (MMVQ_F32_NWARPS_SM * 4);
    const int bx2 = (moe_latent_size + (MMVQ_F32_NWARPS_SM * 2) - 1) / (MMVQ_F32_NWARPS_SM * 2);
    if (bx4 * n_expert >= 224) {
        olala_stage4_grouped_kernel<TYPE_DOWN, 4, 4><<<dim3(bx4, n_expert, 1), block, 0, stream>>>(
            static_cast<const uint8_t *>(down_exps->data), down_exps->nb[1], down_exps->nb[2],
            h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert_used);
    } else {
        olala_stage4_grouped_kernel<TYPE_DOWN, 2, 4><<<dim3(bx2, n_expert, 1), block, 0, stream>>>(
            static_cast<const uint8_t *>(down_exps->data), down_exps->nb[1], down_exps->nb[2],
            h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert_used);
    }
}

__global__ void olala_reduce_slots_kernel(
        const float * __restrict__ y_partial,
        float       * __restrict__ y_reduced,
        int moe_latent_size, int n_expert_used, int n_tokens) {
    const int idx   = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = moe_latent_size * n_tokens;
    if (idx >= total) return;
    const int tok = idx / moe_latent_size;
    const int d   = idx - tok * moe_latent_size;
    float sum = 0.0f;
    const float * base = y_partial + static_cast<size_t>(tok) * n_expert_used * moe_latent_size + d;
    for (int s = 0; s < n_expert_used; ++s) {
        sum += __ldg(base + static_cast<size_t>(s) * moe_latent_size);
    }
    y_reduced[idx] = sum;
}

__global__ void olala_relu2_inplace_kernel(float * __restrict__ data, int total) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = act_relu2(data[idx]);
}

__global__ void olala_add_f32_kernel(
        float       * __restrict__ dst, size_t dst_nb1,
        const float * __restrict__ src, int n_embd, int n_tokens) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_embd * n_tokens) return;
    const int tok = idx / n_embd;
    const int d   = idx - tok * n_embd;
    float * row = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dst) + static_cast<size_t>(tok) * dst_nb1);
    row[d] += __ldg(src + idx);
}

} // namespace

// ============================================================================
// Public Entry Points
// ============================================================================

void ggml_cuda_mul_mat_vec_q_f32_id(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);

    cudaStream_t stream = ctx.stream();
    switch (src0->type) {
        case GGML_TYPE_F32:  launch_mmvq_f32_id<GGML_TYPE_F32> (stream, src0, src1, ids, dst); break;
        case GGML_TYPE_F16:  launch_mmvq_f32_id<GGML_TYPE_F16> (stream, src0, src1, ids, dst); break;
        case GGML_TYPE_BF16: launch_mmvq_f32_id<GGML_TYPE_BF16>(stream, src0, src1, ids, dst); break;
        case GGML_TYPE_Q8_0: launch_mmvq_f32_id<GGML_TYPE_Q8_0>(stream, src0, src1, ids, dst); break;
        case GGML_TYPE_Q5_K: launch_mmvq_f32_id<GGML_TYPE_Q5_K>(stream, src0, src1, ids, dst); break;
        case GGML_TYPE_Q6_K: launch_mmvq_f32_id<GGML_TYPE_Q6_K>(stream, src0, src1, ids, dst); break;
        case GGML_TYPE_Q4_K: launch_mmvq_f32_id<GGML_TYPE_Q4_K>(stream, src0, src1, ids, dst); break;
        default: GGML_ABORT("ggml_cuda_mul_mat_vec_q_f32_id: unsupported type %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_vec_q_f32(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();
    switch (src0->type) {
        case GGML_TYPE_Q8_0: launch_mmvq_f32<GGML_TYPE_Q8_0>(stream, src0, src1, dst); break;
        case GGML_TYPE_Q5_K: launch_mmvq_f32<GGML_TYPE_Q5_K>(stream, src0, src1, dst); break;
        case GGML_TYPE_Q6_K: launch_mmvq_f32<GGML_TYPE_Q6_K>(stream, src0, src1, dst); break;
        case GGML_TYPE_Q4_K: launch_mmvq_f32<GGML_TYPE_Q4_K>(stream, src0, src1, dst); break;
        default: GGML_ABORT("ggml_cuda_mul_mat_vec_q_f32: unsupported type %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_olala_moe(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * cur         = dst->src[0];
    const ggml_tensor * latent_down = dst->src[1];
    const ggml_tensor * gate_inp    = dst->src[2];
    const ggml_tensor * exp_probs_b = dst->src[3];
    const ggml_tensor * up_exps     = dst->src[4];
    const ggml_tensor * down_exps   = dst->src[5];
    const ggml_tensor * latent_up   = dst->src[6];
    const ggml_tensor * up_shexp    = dst->src[7];
    const ggml_tensor * down_shexp  = dst->src[8];

    const int32_t n_expert_used = ((const int32_t *) dst->op_params)[0];
    const float   w_scale       = ((const float   *) dst->op_params)[1];

    const int n_embd          = static_cast<int>(cur->ne[0]);
    const int n_tokens        = static_cast<int>(cur->ne[1]);
    const int moe_latent_size = static_cast<int>(latent_down->ne[1]);
    const int n_expert        = static_cast<int>(gate_inp->ne[1]);
    const int ff_exp          = static_cast<int>(up_exps->ne[1]);
    const bool has_shexp      = (up_shexp != nullptr && down_shexp != nullptr);
    const int ff_shexp        = has_shexp ? static_cast<int>(up_shexp->ne[1]) : 0;

    const bool can_fuse_topk = (n_expert <= 512 && n_expert_used <= 32);

    cudaStream_t stream = ctx.stream();

    if (n_tokens > 8) {
        // P1: Grouped GPU MoE for n_tokens = 9..512 (Tensor Core / MMQ dense + grouped experts, 0 host syncs)
        const size_t sz_logits   = align256(static_cast<size_t>(n_tokens) * n_expert        * sizeof(float));
        const size_t sz_latent   = align256(static_cast<size_t>(n_tokens) * moe_latent_size * sizeof(float));
        const size_t sz_ids      = align256(static_cast<size_t>(n_tokens) * n_expert_used   * sizeof(int32_t));
        const size_t sz_weights  = align256(static_cast<size_t>(n_tokens) * n_expert_used   * sizeof(float));
        const size_t sz_hexp     = align256(static_cast<size_t>(n_tokens) * n_expert_used * ff_exp          * sizeof(float));
        const size_t sz_ypartial = align256(static_cast<size_t>(n_tokens) * n_expert_used * moe_latent_size * sizeof(float));
        const size_t sz_hshexp   = has_shexp ? align256(static_cast<size_t>(n_tokens) * ff_shexp * sizeof(float)) : 0u;
        const size_t sz_offsets  = align256(static_cast<size_t>(n_expert + 1) * sizeof(int32_t));
        const size_t sz_pairs    = align256(static_cast<size_t>(n_tokens) * n_expert_used * sizeof(int32_t));
        const size_t sz_yred     = align256(static_cast<size_t>(n_tokens) * moe_latent_size * sizeof(float));
        const size_t sz_yshexp   = has_shexp ? align256(static_cast<size_t>(n_tokens) * n_embd * sizeof(float)) : 0u;

        const size_t sz_persist = sz_ids + sz_weights + sz_hshexp + sz_offsets + sz_pairs;
        const size_t sz_slot_a  = std::max(sz_logits, sz_hexp);
        const size_t sz_slot_b  = std::max(sz_latent, sz_ypartial);
        const size_t sz_slot_c  = std::max(sz_yred, sz_yshexp);

        ggml_cuda_pool_alloc<uint8_t> scratch(ctx.pool(), sz_persist + sz_slot_a + sz_slot_b + sz_slot_c);
        uint8_t * base           = scratch.get();
        int32_t * expert_ids     = reinterpret_cast<int32_t *>(base); base += sz_ids;
        float   * expert_weights = reinterpret_cast<float   *>(base); base += sz_weights;
        float   * h_shexp        = has_shexp ? reinterpret_cast<float *>(base) : nullptr;
        if (has_shexp) base += sz_hshexp;
        int32_t * exp_offsets    = reinterpret_cast<int32_t *>(base); base += sz_offsets;
        int32_t * sorted_pairs   = reinterpret_cast<int32_t *>(base); base += sz_pairs;

        uint8_t * slot_a         = base; base += sz_slot_a;
        uint8_t * slot_b         = base; base += sz_slot_b;
        uint8_t * slot_c         = base;

        float   * router_logits  = reinterpret_cast<float   *>(slot_a);
        float   * h_exp          = reinterpret_cast<float   *>(slot_a);
        float   * latent         = reinterpret_cast<float   *>(slot_b);
        float   * y_partial      = reinterpret_cast<float   *>(slot_b);
        float   * y_reduced      = reinterpret_cast<float   *>(slot_c);
        float   * y_shexp        = has_shexp ? reinterpret_cast<float *>(slot_c) : nullptr;

        auto make_f32_src = [&](void * data, int64_t ne0, int64_t ne1, size_t nb1) {
            ggml_tensor t = {};
            t.type   = GGML_TYPE_F32;
            t.ne[0]  = ne0; t.ne[1] = ne1; t.ne[2] = 1; t.ne[3] = 1;
            t.nb[0]  = sizeof(float);
            t.nb[1]  = nb1;
            t.nb[2]  = nb1 * static_cast<size_t>(ne1);
            t.nb[3]  = t.nb[2];
            t.data   = data;
            t.buffer = dst->buffer;
            t.op     = GGML_OP_NONE;
            return t;
        };

        auto make_f32_2d = [&](const ggml_tensor * w, const ggml_tensor * x, void * data, int64_t ne0, int64_t ne1, size_t nb1, int32_t prec0, int32_t prec_src1) {
            ggml_tensor t = make_f32_src(data, ne0, ne1, nb1);
            t.op     = GGML_OP_MUL_MAT;
            t.src[0] = const_cast<ggml_tensor *>(w);
            t.src[1] = const_cast<ggml_tensor *>(x);
            ggml_set_op_params_i32(&t, 0, prec0);
            ggml_set_op_params_i32(&t, 3, prec_src1);
            return t;
        };

        ggml_tensor t_cur = *cur;
        ggml_tensor t_router = make_f32_2d(gate_inp,    &t_cur, router_logits, n_expert,        n_tokens, static_cast<size_t>(n_expert)        * sizeof(float), static_cast<int32_t>(GGML_PREC_F32),     static_cast<int32_t>(GGML_PREC_F32));
        ggml_tensor t_latent = make_f32_2d(latent_down, &t_cur, latent,        moe_latent_size, n_tokens, static_cast<size_t>(moe_latent_size) * sizeof(float), static_cast<int32_t>(GGML_PREC_DEFAULT), static_cast<int32_t>(GGML_PREC_BF16));

        ggml_cuda_mul_mat_direct(ctx, gate_inp,    &t_cur, &t_router);
        ggml_cuda_mul_mat_direct(ctx, latent_down, &t_cur, &t_latent);
        if (has_shexp) {
            ggml_tensor t_hshexp = make_f32_2d(up_shexp, &t_cur, h_shexp, ff_shexp, n_tokens, static_cast<size_t>(ff_shexp) * sizeof(float), static_cast<int32_t>(GGML_PREC_DEFAULT), static_cast<int32_t>(GGML_PREC_BF16));
            ggml_cuda_mul_mat_direct(ctx, up_shexp, &t_cur, &t_hshexp);
            const int tot_sh = ff_shexp * n_tokens;
            olala_relu2_inplace_kernel<<<(tot_sh + 255) / 256, 256, 0, stream>>>(h_shexp, tot_sh);
        }

        const float * b_ptr = exp_probs_b ? static_cast<const float *>(exp_probs_b->data) : nullptr;
        const size_t topk_smem = 2u * static_cast<size_t>(n_expert) * sizeof(float);
        olala_stage1b_topk_block_kernel<<<static_cast<unsigned int>(n_tokens), 256, topk_smem, stream>>>(
            router_logits, b_ptr, expert_ids, expert_weights, n_expert, n_expert_used, w_scale);

        olala_group_experts_kernel<<<1, 256, 0, stream>>>(
            expert_ids, exp_offsets, sorted_pairs, n_tokens, n_expert, n_expert_used);

        switch (up_exps->type) {
            case GGML_TYPE_F32:  launch_stage3_grouped<GGML_TYPE_F32> (stream, up_exps, latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used); break;
            case GGML_TYPE_F16:  launch_stage3_grouped<GGML_TYPE_F16> (stream, up_exps, latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used); break;
            case GGML_TYPE_BF16: launch_stage3_grouped<GGML_TYPE_BF16>(stream, up_exps, latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used); break;
            case GGML_TYPE_Q8_0: launch_stage3_grouped<GGML_TYPE_Q8_0>(stream, up_exps, latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used); break;
            case GGML_TYPE_Q5_K: launch_stage3_grouped<GGML_TYPE_Q5_K>(stream, up_exps, latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used); break;
            case GGML_TYPE_Q6_K: launch_stage3_grouped<GGML_TYPE_Q6_K>(stream, up_exps, latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used); break;
            case GGML_TYPE_Q4_K: launch_stage3_grouped<GGML_TYPE_Q4_K>(stream, up_exps, latent, exp_offsets, sorted_pairs, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used); break;
            default: GGML_ABORT("olala_moe: unsupported up_exps type %s", ggml_type_name(up_exps->type));
        }

        switch (down_exps->type) {
            case GGML_TYPE_F32:  launch_stage4_grouped<GGML_TYPE_F32> (stream, down_exps, h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used); break;
            case GGML_TYPE_F16:  launch_stage4_grouped<GGML_TYPE_F16> (stream, down_exps, h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used); break;
            case GGML_TYPE_BF16: launch_stage4_grouped<GGML_TYPE_BF16>(stream, down_exps, h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used); break;
            case GGML_TYPE_Q8_0: launch_stage4_grouped<GGML_TYPE_Q8_0>(stream, down_exps, h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used); break;
            case GGML_TYPE_Q5_K: launch_stage4_grouped<GGML_TYPE_Q5_K>(stream, down_exps, h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used); break;
            case GGML_TYPE_Q6_K: launch_stage4_grouped<GGML_TYPE_Q6_K>(stream, down_exps, h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used); break;
            case GGML_TYPE_Q4_K: launch_stage4_grouped<GGML_TYPE_Q4_K>(stream, down_exps, h_exp, expert_weights, exp_offsets, sorted_pairs, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used); break;
            default: GGML_ABORT("olala_moe: unsupported down_exps type %s", ggml_type_name(down_exps->type));
        }

        const int tot_red = moe_latent_size * n_tokens;
        olala_reduce_slots_kernel<<<(tot_red + 255) / 256, 256, 0, stream>>>(
            y_partial, y_reduced, moe_latent_size, n_expert_used, n_tokens);

        ggml_tensor t_yred = make_f32_src(y_reduced, moe_latent_size, n_tokens, static_cast<size_t>(moe_latent_size) * sizeof(float));
        ggml_tensor t_dst  = *dst;
        t_dst.op     = GGML_OP_MUL_MAT;
        t_dst.src[0] = const_cast<ggml_tensor *>(latent_up);
        t_dst.src[1] = &t_yred;
        ggml_set_op_params_i32(&t_dst, 0, static_cast<int32_t>(GGML_PREC_DEFAULT));
        ggml_set_op_params_i32(&t_dst, 3, static_cast<int32_t>(GGML_PREC_BF16));
        ggml_cuda_mul_mat_direct(ctx, latent_up, &t_yred, &t_dst);

        if (has_shexp) {
            ggml_tensor t_hshexp = make_f32_src(h_shexp, ff_shexp, n_tokens, static_cast<size_t>(ff_shexp) * sizeof(float));
            ggml_tensor t_yshexp = make_f32_2d(down_shexp, &t_hshexp, y_shexp, n_embd,   n_tokens, static_cast<size_t>(n_embd)   * sizeof(float), static_cast<int32_t>(GGML_PREC_DEFAULT), static_cast<int32_t>(GGML_PREC_BF16));
            ggml_cuda_mul_mat_direct(ctx, down_shexp, &t_hshexp, &t_yshexp);
            const int tot_out = n_embd * n_tokens;
            olala_add_f32_kernel<<<(tot_out + 255) / 256, 256, 0, stream>>>(
                static_cast<float *>(dst->data), dst->nb[1], y_shexp, n_embd, n_tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const size_t sz_logits   = align256(static_cast<size_t>(n_tokens) * n_expert        * sizeof(float));
    const size_t sz_latent   = align256(static_cast<size_t>(n_tokens) * moe_latent_size * sizeof(float));
    const size_t sz_ids      = align256(static_cast<size_t>(n_tokens) * n_expert_used   * sizeof(int32_t));
    const size_t sz_weights  = align256(static_cast<size_t>(n_tokens) * n_expert_used   * sizeof(float));
    const size_t sz_hexp     = align256(static_cast<size_t>(n_tokens) * n_expert_used * ff_exp          * sizeof(float));
    const size_t sz_ypartial = align256(static_cast<size_t>(n_tokens) * n_expert_used * moe_latent_size * sizeof(float));
    const size_t sz_hshexp   = has_shexp ? align256(static_cast<size_t>(n_tokens) * ff_shexp * sizeof(float)) : 0u;
    const size_t sz_done     = can_fuse_topk ? align256(static_cast<size_t>(n_tokens) * sizeof(int32_t)) : 0u;

    const size_t sz_persist = sz_ids + sz_weights + sz_hshexp;
    const size_t sz_slot_a  = std::max(sz_logits + sz_done, sz_hexp);
    const size_t sz_slot_b  = std::max(sz_latent, sz_ypartial);

    ggml_cuda_pool_alloc<uint8_t> scratch(
        ctx.pool(),
        sz_persist + sz_slot_a + sz_slot_b);

    uint8_t * base           = scratch.get();
    int32_t * expert_ids     = reinterpret_cast<int32_t *>(base); base += sz_ids;
    float   * expert_weights = reinterpret_cast<float   *>(base); base += sz_weights;
    float   * h_shexp        = has_shexp ? reinterpret_cast<float *>(base) : nullptr;
    if (has_shexp) {
        base += sz_hshexp;
    }

    uint8_t * slot_a         = base; base += sz_slot_a;
    uint8_t * slot_b         = base;

    float   * router_logits  = reinterpret_cast<float   *>(slot_a);
    int32_t * d_router_done  = can_fuse_topk ? reinterpret_cast<int32_t *>(slot_a + sz_logits) : nullptr;
    float   * h_exp          = reinterpret_cast<float   *>(slot_a); // aliases router_logits after Stage 1/1b

    float   * latent         = reinterpret_cast<float   *>(slot_b);
    float   * y_partial      = reinterpret_cast<float   *>(slot_b); // aliases latent after Stage 3

    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);

    // 1. Fused Stage 1: Router + Latent Down + Shared Expert Up + Inline Top-K
    if (can_fuse_topk) {
        CUDA_CHECK(cudaMemsetAsync(d_router_done, 0, static_cast<size_t>(n_tokens) * sizeof(int32_t), stream));
    }
    {
        const int rows_per_blk  = MMVQ_F32_NWARPS_SM * 2;
        const int blocks_router = (n_expert        + rows_per_blk - 1) / rows_per_blk;
        const int blocks_latent = (moe_latent_size + rows_per_blk - 1) / rows_per_blk;
        const int blocks_shexp  = has_shexp ? ((ff_shexp + rows_per_blk - 1) / rows_per_blk) : 0;
        const dim3 grid(static_cast<unsigned int>(blocks_router + blocks_latent + blocks_shexp), 1, static_cast<unsigned int>(n_tokens));

        const float * b_ptr = exp_probs_b ? static_cast<const float *>(exp_probs_b->data) : nullptr;

        auto dispatch_stage1 = [&](auto fuse_topk_tag, auto topk_per_lane_tag) {
            constexpr bool FUSE_TK = decltype(fuse_topk_tag)::value;
            constexpr int  TPL     = decltype(topk_per_lane_tag)::value;

            if (latent_down->type == GGML_TYPE_Q5_K && (!has_shexp || up_shexp->type == GGML_TYPE_Q5_K)) {
                olala_stage1_fused_cur_kernel<true, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, FUSE_TK, TPL><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(cur->data), cur->nb[1],
                    static_cast<const uint8_t *>(gate_inp->data), gate_inp->nb[1], gate_inp->type,
                    b_ptr,
                    static_cast<const uint8_t *>(latent_down->data), latent_down->nb[1], latent_down->type,
                    has_shexp ? static_cast<const uint8_t *>(up_shexp->data) : nullptr,
                    has_shexp ? up_shexp->nb[1] : 0u,
                    has_shexp ? up_shexp->type : GGML_TYPE_F32,
                    router_logits, d_router_done, expert_ids, expert_weights, latent, h_shexp,
                    n_embd, n_expert, n_expert_used, w_scale, moe_latent_size, ff_shexp,
                    blocks_router, blocks_latent);
            } else if (latent_down->type == GGML_TYPE_Q8_0 && (!has_shexp || up_shexp->type == GGML_TYPE_Q8_0)) {
                olala_stage1_fused_cur_kernel<true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, FUSE_TK, TPL><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(cur->data), cur->nb[1],
                    static_cast<const uint8_t *>(gate_inp->data), gate_inp->nb[1], gate_inp->type,
                    b_ptr,
                    static_cast<const uint8_t *>(latent_down->data), latent_down->nb[1], latent_down->type,
                    has_shexp ? static_cast<const uint8_t *>(up_shexp->data) : nullptr,
                    has_shexp ? up_shexp->nb[1] : 0u,
                    has_shexp ? up_shexp->type : GGML_TYPE_F32,
                    router_logits, d_router_done, expert_ids, expert_weights, latent, h_shexp,
                    n_embd, n_expert, n_expert_used, w_scale, moe_latent_size, ff_shexp,
                    blocks_router, blocks_latent);
            } else {
                olala_stage1_fused_cur_kernel<false, GGML_TYPE_F32, GGML_TYPE_F32, FUSE_TK, TPL><<<grid, block, 0, stream>>>(
                    static_cast<const uint8_t *>(cur->data), cur->nb[1],
                    static_cast<const uint8_t *>(gate_inp->data), gate_inp->nb[1], gate_inp->type,
                    b_ptr,
                    static_cast<const uint8_t *>(latent_down->data), latent_down->nb[1], latent_down->type,
                    has_shexp ? static_cast<const uint8_t *>(up_shexp->data) : nullptr,
                    has_shexp ? up_shexp->nb[1] : 0u,
                    has_shexp ? up_shexp->type : GGML_TYPE_F32,
                    router_logits, d_router_done, expert_ids, expert_weights, latent, h_shexp,
                    n_embd, n_expert, n_expert_used, w_scale, moe_latent_size, ff_shexp,
                    blocks_router, blocks_latent);
            }
        };

        if (can_fuse_topk) {
            if (n_expert <= 256) {
                dispatch_stage1(std::true_type{}, std::integral_constant<int, 8>{});
            } else {
                dispatch_stage1(std::true_type{}, std::integral_constant<int, 16>{});
            }
        } else {
            dispatch_stage1(std::false_type{}, std::integral_constant<int, 8>{});
        }
    }

    // 2. Stage 1b fallback only when Top-K could not be fused into Stage 1
    if (!can_fuse_topk) {
        const float * b_ptr = exp_probs_b ? static_cast<const float *>(exp_probs_b->data) : nullptr;
        const size_t topk_smem = 2u * static_cast<size_t>(n_expert) * sizeof(float);
        olala_stage1b_topk_block_kernel<<<static_cast<unsigned int>(n_tokens), 256, topk_smem, stream>>>(
            router_logits, b_ptr, expert_ids, expert_weights, n_expert, n_expert_used, w_scale);
    }

    // 3. Stage 3: Routed Experts Up + ReLU^2 (Adaptive 1 / 2 / 4 rows/warp)
    switch (up_exps->type) {
        case GGML_TYPE_F32:  launch_stage3<GGML_TYPE_F32> (stream, up_exps, latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_F16:  launch_stage3<GGML_TYPE_F16> (stream, up_exps, latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_BF16: launch_stage3<GGML_TYPE_BF16>(stream, up_exps, latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q8_0: launch_stage3<GGML_TYPE_Q8_0>(stream, up_exps, latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q5_K: launch_stage3<GGML_TYPE_Q5_K>(stream, up_exps, latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q6_K: launch_stage3<GGML_TYPE_Q6_K>(stream, up_exps, latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q4_K: launch_stage3<GGML_TYPE_Q4_K>(stream, up_exps, latent, expert_ids, h_exp, moe_latent_size, ff_exp, n_expert, n_expert_used, n_tokens); break;
        default: GGML_ABORT("olala_moe: unsupported up_exps type %s", ggml_type_name(up_exps->type));
    }

    // 4. Stage 4: Parallel Routed Experts Down (grid.y = n_expert_used)
    switch (down_exps->type) {
        case GGML_TYPE_F32:  launch_stage4<GGML_TYPE_F32> (stream, down_exps, h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_F16:  launch_stage4<GGML_TYPE_F16> (stream, down_exps, h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_BF16: launch_stage4<GGML_TYPE_BF16>(stream, down_exps, h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q8_0: launch_stage4<GGML_TYPE_Q8_0>(stream, down_exps, h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q5_K: launch_stage4<GGML_TYPE_Q5_K>(stream, down_exps, h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q6_K: launch_stage4<GGML_TYPE_Q6_K>(stream, down_exps, h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used, n_tokens); break;
        case GGML_TYPE_Q4_K: launch_stage4<GGML_TYPE_Q4_K>(stream, down_exps, h_exp, expert_ids, expert_weights, y_partial, ff_exp, moe_latent_size, n_expert, n_expert_used, n_tokens); break;
        default: GGML_ABORT("olala_moe: unsupported down_exps type %s", ggml_type_name(down_exps->type));
    }

    // 5. Fused Stage 5: 6-Slot Sum + ffn_latent_up + ffn_down_shexp -> Single Write to dst
    const uint8_t * lu_data = static_cast<const uint8_t *>(latent_up->data);
    const uint8_t * ds_data = has_shexp ? static_cast<const uint8_t *>(down_shexp->data) : nullptr;
    const size_t    ds_nb1  = has_shexp ? down_shexp->nb[1] : 0u;
    const ggml_type ds_type = has_shexp ? down_shexp->type : GGML_TYPE_F32;

    if (latent_up->type == GGML_TYPE_Q5_K && (!has_shexp || ds_type == GGML_TYPE_Q5_K)) {
        launch_stage5_specialized<true, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K>(
            stream, lu_data, latent_up->nb[1], latent_up->type,
            y_partial, ds_data, ds_nb1, ds_type, h_shexp,
            static_cast<uint8_t *>(dst->data), dst->nb[1],
            moe_latent_size, ff_shexp, n_embd, n_expert_used, n_tokens, has_shexp);
    } else if (latent_up->type == GGML_TYPE_Q8_0 && (!has_shexp || ds_type == GGML_TYPE_Q8_0)) {
        launch_stage5_specialized<true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(
            stream, lu_data, latent_up->nb[1], latent_up->type,
            y_partial, ds_data, ds_nb1, ds_type, h_shexp,
            static_cast<uint8_t *>(dst->data), dst->nb[1],
            moe_latent_size, ff_shexp, n_embd, n_expert_used, n_tokens, has_shexp);
    } else {
        launch_stage5_specialized<false, GGML_TYPE_F32, GGML_TYPE_F32>(
            stream, lu_data, latent_up->nb[1], latent_up->type,
            y_partial, ds_data, ds_nb1, ds_type, h_shexp,
            static_cast<uint8_t *>(dst->data), dst->nb[1],
            moe_latent_size, ff_shexp, n_embd, n_expert_used, n_tokens, has_shexp);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// P3: CUDA Fused V-Layer (MMV, TPA_PREP, DIFF_GATE)
// ============================================================================

namespace {

struct olala_mmv_desc {
    const uint8_t * ptr[9];
    size_t          nb1[9];
    int             row_off[9];
    int             M[9];
    ggml_type       type[9];
    int             n_mats;
};

template <int MODE>
__global__ __launch_bounds__(MMVQ_F32_THREADS_SM, 8)
void olala_mmv_cuda_kernel(
        const float   * __restrict__ x_base, size_t x_nb1,
        float         * __restrict__ y_base, size_t y_nb1,
        olala_mmv_desc desc, int K) {
    constexpr int RPW = 2;
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int bx      = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.z);
    const int rows_per_blk = MMVQ_F32_NWARPS_SM * RPW;

    int acc_blk = 0, m = 0;
    for (; m < desc.n_mats; ++m) {
        const int blks_m = (desc.M[m] + rows_per_blk - 1) / rows_per_blk;
        if (bx < acc_blk + blks_m) {
            break;
        }
        acc_blk += blks_m;
    }
    if (m >= desc.n_mats) {
        return;
    }

    const int local_row = ((bx - acc_blk) * MMVQ_F32_NWARPS_SM + warp_id) * RPW;
    if (local_row >= desc.M[m]) {
        return;
    }

    const float   * x_ptr = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(x_base) + static_cast<size_t>(token) * x_nb1);
    float         * y_ptr = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(y_base) + static_cast<size_t>(token) * y_nb1) + desc.row_off[m];
    const uint8_t * r0    = desc.ptr[m] + static_cast<size_t>(local_row) * desc.nb1[m];

    if (local_row + RPW <= desc.M[m]) {
        float s[RPW];
        if constexpr (MODE == 0) {
            vec_dot_rows_f32<GGML_TYPE_Q5_K, RPW, false>(r0, desc.nb1[m], x_ptr, K, lane, s);
        } else if constexpr (MODE == 1) {
            if (desc.type[m] == GGML_TYPE_Q5_K)      vec_dot_rows_f32<GGML_TYPE_Q5_K, RPW, false>(r0, desc.nb1[m], x_ptr, K, lane, s);
            else if (desc.type[m] == GGML_TYPE_Q8_0) vec_dot_rows_f32<GGML_TYPE_Q8_0, RPW, false>(r0, desc.nb1[m], x_ptr, K, lane, s);
            else                                     vec_dot_rows_f32<GGML_TYPE_F32,  RPW, false>(r0, desc.nb1[m], x_ptr, K, lane, s);
        } else {
            vec_dot_dispatch_rows<RPW, false>(desc.type[m], r0, desc.nb1[m], x_ptr, K, lane, s);
        }
        if (lane == 0) {
            y_ptr[local_row + 0] = s[0];
            y_ptr[local_row + 1] = s[1];
        }
    } else {
        float s1[1];
        if constexpr (MODE == 0) {
            vec_dot_rows_f32<GGML_TYPE_Q5_K, 1, false>(r0, 0, x_ptr, K, lane, s1);
        } else if constexpr (MODE == 1) {
            if (desc.type[m] == GGML_TYPE_Q5_K)      vec_dot_rows_f32<GGML_TYPE_Q5_K, 1, false>(r0, 0, x_ptr, K, lane, s1);
            else if (desc.type[m] == GGML_TYPE_Q8_0) vec_dot_rows_f32<GGML_TYPE_Q8_0, 1, false>(r0, 0, x_ptr, K, lane, s1);
            else                                     vec_dot_rows_f32<GGML_TYPE_F32,  1, false>(r0, 0, x_ptr, K, lane, s1);
        } else {
            vec_dot_dispatch_rows<1, false>(desc.type[m], r0, 0, x_ptr, K, lane, s1);
        }
        if (lane == 0) {
            y_ptr[local_row] = s1[0];
        }
    }
}

__global__ void olala_tpa_prep_cuda_kernel(
        const float   * __restrict__ prd,
        const float   * __restrict__ pvd,
        const int32_t * __restrict__ posd,
        const float   * __restrict__ qn,
        const float   * __restrict__ kn,
        const float   * __restrict__ sc,
        float         * __restrict__ out,
        int H, int Hn, int R, int Dh, int Hs, int n_seqs, int T, int64_t ROW, int64_t pv_s,
        float eps, float wsize, float gate_b) {
    const int t  = static_cast<int>(blockIdx.x);
    const int hn = static_cast<int>(blockIdx.y);
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw   = nth >> 5;

    const int64_t n_seq_t = T / n_seqs;
    const int64_t s  = t / n_seq_t;
    const int64_t tt = t - s * n_seq_t;
    const int G = H / Hn;

    const int64_t OAK = (int64_t) H * Dh;
    const int64_t OAV = OAK + Hn * R;
    const int64_t OBK = OAV + Hn * R;
    const int64_t OBV = OBK + R * Dh;
    const int64_t OSK = OBV + R * Dh;
    const int64_t OSV = OSK + Hn;
    const int64_t OLAM = OSV + Hn;
    const int64_t OG = OLAM + Hn;

    const int64_t o_k    = (int64_t) T * H * Dh;
    const int64_t o_v    = o_k + (int64_t) T * Hn * Dh;
    const int64_t o_lam  = o_v + (int64_t) T * Hn * Dh;
    const int64_t o_g    = o_lam + (int64_t) T * Hn;
    const int64_t o_last = o_g + (int64_t) T * Hs * Dh;

    const float * p = prd + static_cast<int64_t>(t) * ROW;
    const float pos_val = static_cast<float>(__ldg(posd + t));
    const float pmask = fminf(fmaxf(pos_val, 0.0f), 1.0f);
    const float lp    = logf(fminf(fmaxf(pos_val + 1.0f, 1.0f), wsize));

    extern __shared__ float smem[];
    float * kr  = smem;
    float * vr  = kr + Dh;
    float * kp  = vr + Dh;
    float * vp  = kp + Dh;
    float * kx  = vp + Dh;
    __shared__ float s_red[8];

    const float inv_r = 1.0f / static_cast<float>(R);
    for (int d = tid; d < Dh; d += nth) {
        float sk = 0.0f, sv = 0.0f;
        for (int r = 0; r < R; ++r) {
            sk = fmaf(__ldg(&p[OBK + r * Dh + d]), __ldg(&p[OAK + hn * R + r]), sk);
            sv = fmaf(__ldg(&p[OBV + r * Dh + d]), __ldg(&p[OAV + hn * R + r]), sv);
        }
        const float k_cur = sk * inv_r;
        const float v_cur = sv * inv_r;
        kr[d] = k_cur;
        vr[d] = v_cur;

        if (tt == 0) {
            kp[d] = __ldg(&pvd[s * pv_s + hn * Dh + d]);
            vp[d] = __ldg(&pvd[s * pv_s + Hn * Dh + hn * Dh + d]);
        } else {
            const float * pm1 = prd + static_cast<int64_t>(t - 1) * ROW;
            float skp = 0.0f, svp = 0.0f;
            for (int r = 0; r < R; ++r) {
                skp = fmaf(__ldg(&pm1[OBK + r * Dh + d]), __ldg(&pm1[OAK + hn * R + r]), skp);
                svp = fmaf(__ldg(&pm1[OBV + r * Dh + d]), __ldg(&pm1[OAV + hn * R + r]), svp);
            }
            kp[d] = skp * inv_r;
            vp[d] = svp * inv_r;
        }
    }
    __syncthreads();

    if (tt == n_seq_t - 1) {
        for (int d = tid; d < Dh; d += nth) {
            out[o_last + s * 2 * Hn * Dh + hn * Dh + d]           = kr[d];
            out[o_last + s * 2 * Hn * Dh + Hn * Dh + hn * Dh + d] = vr[d];
        }
    }

    const float ak = (1.0f / (1.0f + expf(-__ldg(&p[OSK + hn])))) * pmask;
    const float av = (1.0f / (1.0f + expf(-__ldg(&p[OSV + hn])))) * pmask;
    const float bk = 1.0f - ak;
    const float bv = 1.0f - av;

    float ss_k = 0.0f;
    for (int d = tid; d < Dh; d += nth) {
        const float xk = fmaf(kp[d], ak, kr[d] * bk);
        kx[d] = xk;
        ss_k  = fmaf(xk, xk, ss_k);
        out[o_v + (static_cast<int64_t>(t) * Hn + hn) * Dh + d] = fmaf(vp[d], av, vr[d] * bv);
    }
    ss_k = warp_sum_f32(ss_k);
    if (lane == 0) {
        s_red[warp] = ss_k;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nw) ? s_red[lane] : 0.0f;
        v = warp_sum_f32(v);
        if (lane == 0) {
            s_red[0] = rsqrtf(v / static_cast<float>(Dh) + eps);
        }
    }
    __syncthreads();
    const float rs_k = s_red[0];
    for (int d = tid; d < Dh; d += nth) {
        out[o_k + (static_cast<int64_t>(t) * Hn + hn) * Dh + d] = (kx[d] * rs_k) * __ldg(kn + d);
    }

    for (int g = 0; g < G; ++g) {
        const int h = hn * G + g;
        const float * q = p + static_cast<int64_t>(h) * Dh;
        float ss_q = 0.0f;
        for (int d = tid; d < Dh; d += nth) {
            const float qv = __ldg(q + d);
            ss_q = fmaf(qv, qv, ss_q);
        }
        ss_q = warp_sum_f32(ss_q);
        if (lane == 0) {
            s_red[warp] = ss_q;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nw) ? s_red[lane] : 0.0f;
            v = warp_sum_f32(v);
            if (lane == 0) {
                s_red[0] = rsqrtf(v / static_cast<float>(Dh) + eps);
            }
        }
        __syncthreads();
        const float q_scale = s_red[0] * lp * __ldg(sc + h);
        for (int d = tid; d < Dh; d += nth) {
            out[(static_cast<int64_t>(t) * H + h) * Dh + d] = (__ldg(q + d) * q_scale) * __ldg(qn + d);
        }
        __syncthreads();
    }

    if (hn == 0) {
        for (int i = tid; i < Hn; i += nth) {
            out[o_lam + static_cast<int64_t>(t) * Hn + i] = 1.0f / (1.0f + expf(-__ldg(&p[OLAM + i])));
        }
        const int64_t g_len = static_cast<int64_t>(Hs) * Dh;
        for (int64_t i = tid; i < g_len; i += nth) {
            const float gv = __ldg(&p[OG + i]) + gate_b;
            out[o_g + static_cast<int64_t>(t) * g_len + i] = gv / (1.0f + expf(-gv));
        }
    }
}

__global__ void olala_diff_gate_cuda_kernel(
        const float * __restrict__ a,
        const float * __restrict__ lam, int64_t lam_s,
        const float * __restrict__ gs,  int64_t gs_s,
        float       * __restrict__ y,
        int Dh, int H, int Hn, int Hs, int snr, int T) {
    const int64_t total = static_cast<int64_t>(T) * Hs * Dh;
    const int64_t i     = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    const int64_t hs_dh = static_cast<int64_t>(Hs) * Dh;
    const int64_t t     = i / hs_dh;
    const int64_t j     = i - t * hs_dh;
    const int64_t hs    = j / Dh;
    const int64_t d     = j - hs * Dh;
    const int64_t hn    = hs / snr;
    const int64_t k     = hs - hn * snr;

    const float sig = __ldg(&a[(t * H + hn * (snr + 1) + k)   * Dh + d]);
    const float noi = __ldg(&a[(t * H + hn * (snr + 1) + snr) * Dh + d]);
    const float l   = __ldg(&lam[t * lam_s + hn]);
    const float g   = __ldg(&gs[t * gs_s + j]);
    y[i] = fmaf(-noi, l, sig) * g;
}

} // namespace

void ggml_cuda_op_olala_mmv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur = dst->src[0];
    const int K        = static_cast<int>(cur->ne[0]);
    const int n_tokens = static_cast<int>(cur->ne[1]);
    const int rows_per_blk = MMVQ_F32_NWARPS_SM * 2;

    olala_mmv_desc desc = {};
    int row_acc = 0;
    int total_blocks = 0;
    bool all_q5k = true;
    bool only_q5k_q8_f32 = true;

    for (int i = 1; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        const ggml_tensor * w = dst->src[i];
        const int idx = desc.n_mats++;
        const int M   = static_cast<int>(w->ne[1]);
        desc.ptr[idx]     = static_cast<const uint8_t *>(w->data);
        desc.nb1[idx]     = w->nb[1];
        desc.row_off[idx] = row_acc;
        desc.M[idx]       = M;
        desc.type[idx]    = w->type;
        if (w->type != GGML_TYPE_Q5_K) {
            all_q5k = false;
            if (w->type != GGML_TYPE_Q8_0 && w->type != GGML_TYPE_F32) {
                only_q5k_q8_f32 = false;
            }
        }
        row_acc += M;
        total_blocks += (M + rows_per_blk - 1) / rows_per_blk;
    }

    cudaStream_t stream = ctx.stream();
    const dim3 grid(static_cast<unsigned int>(total_blocks), 1, static_cast<unsigned int>(n_tokens));
    const dim3 block(MMVQ_F32_WARP_SIZE, MMVQ_F32_NWARPS_SM, 1);

    if (all_q5k) {
        olala_mmv_cuda_kernel<0><<<grid, block, 0, stream>>>(
            static_cast<const float *>(cur->data), cur->nb[1],
            static_cast<float *>(dst->data), dst->nb[1], desc, K);
    } else if (only_q5k_q8_f32) {
        olala_mmv_cuda_kernel<1><<<grid, block, 0, stream>>>(
            static_cast<const float *>(cur->data), cur->nb[1],
            static_cast<float *>(dst->data), dst->nb[1], desc, K);
    } else {
        olala_mmv_cuda_kernel<2><<<grid, block, 0, stream>>>(
            static_cast<const float *>(cur->data), cur->nb[1],
            static_cast<float *>(dst->data), dst->nb[1], desc, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_olala_tpa_prep(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * pr   = dst->src[0];
    const ggml_tensor * prev = dst->src[1];
    const ggml_tensor * pos  = dst->src[2];
    const float * qn = static_cast<const float *>(dst->src[3]->data);
    const float * kn = static_cast<const float *>(dst->src[4]->data);
    const float * sc = static_cast<const float *>(dst->src[5]->data);

    const int H      = ggml_get_op_params_i32(dst, 0);
    const int Hn     = ggml_get_op_params_i32(dst, 1);
    const int R      = ggml_get_op_params_i32(dst, 2);
    const int Dh     = ggml_get_op_params_i32(dst, 3);
    const int Hs     = ggml_get_op_params_i32(dst, 4);
    const int n_seqs = ggml_get_op_params_i32(dst, 5);
    const float eps    = ggml_get_op_params_f32(dst, 6);
    const float wsize  = ggml_get_op_params_f32(dst, 7);
    const float gate_b = ggml_get_op_params_f32(dst, 8);

    const int T        = static_cast<int>(pr->ne[1]);
    const int64_t ROW  = pr->ne[0];
    const int64_t pv_s = prev->nb[1] / sizeof(float);

    const size_t smem_bytes = static_cast<size_t>(5 * Dh) * sizeof(float);
    const dim3 grid(static_cast<unsigned int>(T), static_cast<unsigned int>(Hn), 1);
    olala_tpa_prep_cuda_kernel<<<grid, 128, smem_bytes, ctx.stream()>>>(
        static_cast<const float *>(pr->data),
        static_cast<const float *>(prev->data),
        static_cast<const int32_t *>(pos->data),
        qn, kn, sc,
        static_cast<float *>(dst->data),
        H, Hn, R, Dh, Hs, n_seqs, T, ROW, pv_s, eps, wsize, gate_b);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_olala_diff_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * attn = dst->src[0];
    const ggml_tensor * lam  = dst->src[1];
    const ggml_tensor * gs   = dst->src[2];

    const int Dh  = static_cast<int>(attn->ne[0]);
    const int H   = static_cast<int>(attn->ne[1]);
    const int T   = static_cast<int>(attn->ne[2]);
    const int Hn  = static_cast<int>(lam->ne[0]);
    const int Hs  = H - Hn;
    const int snr = Hs / Hn;

    const int64_t lam_s = lam->nb[1] / sizeof(float);
    const int64_t gs_s  = gs->nb[1]  / sizeof(float);
    const int64_t total = static_cast<int64_t>(T) * Hs * Dh;

    olala_diff_gate_cuda_kernel<<<static_cast<unsigned int>((total + 255) / 256), 256, 0, ctx.stream()>>>(
        static_cast<const float *>(attn->data),
        static_cast<const float *>(lam->data), lam_s,
        static_cast<const float *>(gs->data), gs_s,
        static_cast<float *>(dst->data),
        Dh, H, Hn, Hs, snr, T);
    CUDA_CHECK(cudaGetLastError());
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_QKK_64)
