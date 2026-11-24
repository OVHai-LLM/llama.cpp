#pragma once

#include "common.cuh"
#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_QKK_64)

#define GGML_CUDA_MMVQ_F32_DENSE_MAX_BATCH 8
#define GGML_CUDA_MMVQ_F32_ID_MAX_BATCH    32
#define GGML_CUDA_OLALA_MOE_MAX_BATCH      512

static inline bool ggml_cuda_supports_mmvq_f32_quant_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_K:
            return true;
        default:
            return false;
    }
}

static inline bool ggml_cuda_supports_mmvq_f32_id_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_K:
            return true;
        default:
            return false;
    }
}

static inline bool ggml_cuda_f32_rows_aligned16(const ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    if ((t->ne[0] & 3) != 0) {
        return false;
    }
    if ((t->nb[1] & 15u) != 0u || (t->nb[2] & 15u) != 0u || (t->nb[3] & 15u) != 0u) {
        return false;
    }
    if (t->view_src != nullptr && (t->view_offs & 15u) != 0u) {
        return false;
    }
    if (t->data != nullptr && (reinterpret_cast<uintptr_t>(t->data) & 15u) != 0u) {
        return false;
    }
    return true;
}

static inline bool ggml_cuda_quant_rows_aligned16(const ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    if (t->type == GGML_TYPE_Q5_K || t->type == GGML_TYPE_Q4_K) {
        if ((t->nb[1] & 15u) != 0u || (t->nb[2] & 15u) != 0u || (t->nb[3] & 15u) != 0u) {
            return false;
        }
        if (t->view_src != nullptr && (t->view_offs & 15u) != 0u) {
            return false;
        }
        if (t->data != nullptr && (reinterpret_cast<uintptr_t>(t->data) & 15u) != 0u) {
            return false;
        }
    }
    return true;
}

static inline bool ggml_cuda_should_use_mmvq_f32_id(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst = nullptr) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(ids);
    GGML_UNUSED(dst);
    return false;
#else
    if (ids == nullptr || ids->type != GGML_TYPE_I32 || src1 == nullptr || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst != nullptr && (dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float))) {
        return false;
    }
    if (!ggml_cuda_supports_mmvq_f32_id_type(src0->type)) {
        return false;
    }
    if (!ggml_cuda_quant_rows_aligned16(src0)) {
        return false;
    }
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens < 1 || n_tokens > GGML_CUDA_MMVQ_F32_ID_MAX_BATCH) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) ||
        src1->nb[0] != sizeof(float) ||
        ids->nb[0]  != sizeof(int32_t)) {
        return false;
    }
    if (!ggml_cuda_f32_rows_aligned16(src1)) {
        return false;
    }
    return true;
#endif
}

static inline bool ggml_cuda_should_use_mmvq_f32(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#else
    if (src1 == nullptr || dst == nullptr) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[2] * src1->ne[3] > 65535) {
        return false; // CUDA gridDim.z hardware limit guard
    }
    if (!ggml_cuda_supports_mmvq_f32_quant_type(src0->type)) {
        return false;
    }
    if (!ggml_cuda_quant_rows_aligned16(src0)) {
        return false;
    }
    const int64_t max_b = GGML_CUDA_MMVQ_F32_DENSE_MAX_BATCH;
    if (src1->ne[1] < 1 || src1->ne[1] > max_b) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) ||
        src1->nb[0] != sizeof(float) ||
        dst->nb[0]  != sizeof(float)) {
        return false;
    }
    if (!ggml_cuda_f32_rows_aligned16(src1)) {
        return false;
    }
    return true;
#endif
}

static inline bool ggml_cuda_supports_olala_moe(const ggml_tensor * dst) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(dst);
    return false;
#else
    if (dst == nullptr || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    const ggml_tensor * cur         = dst->src[0];
    const ggml_tensor * latent_down = dst->src[1];
    const ggml_tensor * gate_inp    = dst->src[2];
    const ggml_tensor * exp_probs_b = dst->src[3];
    const ggml_tensor * up_exps     = dst->src[4];
    const ggml_tensor * down_exps   = dst->src[5];
    const ggml_tensor * latent_up   = dst->src[6];
    const ggml_tensor * up_shexp    = dst->src[7];
    const ggml_tensor * down_shexp  = dst->src[8];

    if (!cur || !latent_down || !gate_inp || !up_exps || !down_exps || !latent_up) {
        return false;
    }
    if (cur->type != GGML_TYPE_F32 || cur->ne[1] < 1 || cur->ne[1] > GGML_CUDA_OLALA_MOE_MAX_BATCH ||
        cur->ne[2] != 1 || cur->ne[3] != 1) {
        return false;
    }
    if (!ggml_cuda_f32_rows_aligned16(cur)) {
        return false;
    }
    if (latent_down->ne[1] > 4096 || (latent_down->ne[1] & 3) != 0 || (up_exps->ne[1] & 3) != 0) {
        return false;
    }
    if (gate_inp->ne[1] > 4096) {
        return false; // Guard SMEM bounds in fallback topk kernel
    }
    if (cur->ne[1] > 8 && gate_inp->ne[1] > 512) {
        return false; // Grouped MoE SMEM histogram supports up to 512 experts
    }
    const int32_t n_expert_used = static_cast<const int32_t *>(dst->op_params)[0];
    if (n_expert_used < 1 || n_expert_used > gate_inp->ne[1]) {
        return false;
    }
    if (gate_inp->type != GGML_TYPE_F32 &&
        gate_inp->type != GGML_TYPE_F16 &&
        gate_inp->type != GGML_TYPE_BF16) {
        return false;
    }
    if (exp_probs_b != nullptr && exp_probs_b->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_cuda_supports_mmvq_f32_id_type(latent_down->type) ||
        !ggml_cuda_supports_mmvq_f32_id_type(up_exps->type)     ||
        !ggml_cuda_supports_mmvq_f32_id_type(down_exps->type)   ||
        !ggml_cuda_supports_mmvq_f32_id_type(latent_up->type)) {
        return false;
    }
    if (up_shexp != nullptr) {
        if (!ggml_cuda_supports_mmvq_f32_id_type(up_shexp->type) || (up_shexp->ne[1] & 3) != 0) {
            return false;
        }
    }
    if (down_shexp != nullptr && !ggml_cuda_supports_mmvq_f32_id_type(down_shexp->type)) {
        return false;
    }
    // Verify contiguous row strides and 16-byte row alignment for all non-null inputs
    for (int i = 0; i <= 8; ++i) {
        if (dst->src[i] != nullptr) {
            if (dst->src[i]->nb[0] != ggml_type_size(dst->src[i]->type)) {
                return false;
            }
            if (i >= 1 && !ggml_cuda_quant_rows_aligned16(dst->src[i])) {
                return false;
            }
        }
    }
    return true;
#endif
}

void ggml_cuda_mul_mat_vec_q_f32_id(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_q_f32(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

void ggml_cuda_op_olala_moe(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);

static inline bool ggml_cuda_supports_olala_mmv(const ggml_tensor * dst) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(dst);
    return false;
#else
    if (dst == nullptr || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    const ggml_tensor * cur = dst->src[0];
    if (cur == nullptr || cur->type != GGML_TYPE_F32 || cur->nb[0] != sizeof(float) ||
        cur->ne[1] < 1 || cur->ne[1] > 8 || cur->ne[2] != 1 || cur->ne[3] != 1 ||
        dst->ne[2] != 1 || dst->ne[3] != 1 || !ggml_cuda_f32_rows_aligned16(cur)) {
        return false;
    }
    int n_mats = 0;
    for (int i = 1; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        const ggml_tensor * w = dst->src[i];
        if (!ggml_cuda_supports_mmvq_f32_id_type(w->type) ||
            !ggml_cuda_quant_rows_aligned16(w) ||
            w->nb[0] != ggml_type_size(w->type)) {
            return false;
        }
        ++n_mats;
    }
    return n_mats >= 1 && n_mats <= 9;
#endif
}

static inline bool ggml_cuda_supports_olala_tpa_prep(const ggml_tensor * dst) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(dst);
    return false;
#else
    if (dst == nullptr || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int H  = ggml_get_op_params_i32(dst, 0);
    const int Hn = ggml_get_op_params_i32(dst, 1);
    const int R  = ggml_get_op_params_i32(dst, 2);
    const int Dh = ggml_get_op_params_i32(dst, 3);
    const int Hs = ggml_get_op_params_i32(dst, 4);
    const int n_seqs   = ggml_get_op_params_i32(dst, 5);
    const int factored = ggml_get_op_params_i32(dst, 9);
    if (factored != 0 || H <= 0 || Hn <= 0 || (H % Hn) != 0 || R <= 0 || R > 8 ||
        Dh <= 0 || Dh > 256 || Hs <= 0 || n_seqs <= 0) {
        return false;
    }
    for (int i = 0; i <= 5; ++i) {
        if (dst->src[i] == nullptr) {
            return false;
        }
    }
    return dst->src[0]->type == GGML_TYPE_F32 &&
           dst->src[0]->ne[1] >= 1 && (dst->src[0]->ne[1] % n_seqs) == 0 &&
           dst->src[1]->type == GGML_TYPE_F32 &&
           dst->src[1]->nb[0] == sizeof(float) && (dst->src[1]->nb[1] % sizeof(float)) == 0 &&
           dst->src[2]->type == GGML_TYPE_I32 &&
           dst->src[3]->type == GGML_TYPE_F32 &&
           dst->src[4]->type == GGML_TYPE_F32 &&
           dst->src[5]->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(dst->src[0]) &&
           ggml_is_contiguous(dst->src[2]) &&
           ggml_is_contiguous(dst->src[3]) &&
           ggml_is_contiguous(dst->src[4]) &&
           ggml_is_contiguous(dst->src[5]);
#endif
}

static inline bool ggml_cuda_supports_olala_diff_gate(const ggml_tensor * dst) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(dst);
    return false;
#else
    if (dst == nullptr || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (dst->src[i] == nullptr || dst->src[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst->src[i])) {
            return false;
        }
    }
    return true;
#endif
}

void ggml_cuda_mul_mat_direct(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

void ggml_cuda_op_olala_mmv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_olala_tpa_prep(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_olala_diff_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#else

// Stubs when building under HIP, MUSA, or GGML_QKK_64
static inline bool ggml_cuda_should_use_mmvq_f32_id(
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor * = nullptr) {
    return false;
}
static inline bool ggml_cuda_should_use_mmvq_f32(
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    return false;
}
static inline bool ggml_cuda_supports_olala_moe(const ggml_tensor *) {
    return false;
}
static inline bool ggml_cuda_supports_olala_mmv(const ggml_tensor *) {
    return false;
}
static inline bool ggml_cuda_supports_olala_tpa_prep(const ggml_tensor *) {
    return false;
}
static inline bool ggml_cuda_supports_olala_diff_gate(const ggml_tensor *) {
    return false;
}
inline void ggml_cuda_mul_mat_vec_q_f32_id(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {}
inline void ggml_cuda_mul_mat_vec_q_f32(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {}
inline void ggml_cuda_op_olala_moe(
        ggml_backend_cuda_context &, ggml_tensor *) {}
inline void ggml_cuda_op_olala_mmv(
        ggml_backend_cuda_context &, ggml_tensor *) {}
inline void ggml_cuda_op_olala_tpa_prep(
        ggml_backend_cuda_context &, ggml_tensor *) {}
inline void ggml_cuda_op_olala_diff_gate(
        ggml_backend_cuda_context &, ggml_tensor *) {}

#endif
