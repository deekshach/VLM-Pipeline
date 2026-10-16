#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Jetson study switch GGML_CUDA_JETSON_ACT_FUSE: RMS_NORM node `norm` followed by MUL node `mul_tensor`, written as
// BF16 into `out`. With out == nullptr only checks. Returns false (doing nothing) if the pair does not qualify.
bool ggml_cuda_jetson_rms_norm_mul_bf16(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
                                        const ggml_tensor * mul_tensor, nv_bfloat16 * out);
