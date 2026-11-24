#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_mamba3_mimo(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float * fused_s_out = nullptr,
        int64_t fused_s_out_stride = 0);
void ggml_cuda_op_geodesic(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
