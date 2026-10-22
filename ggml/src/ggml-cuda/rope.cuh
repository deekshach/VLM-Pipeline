#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// Jetson study switch GGML_CUDA_JETSON_ROPE_F16 (default off): the ViT's vision RoPE (Qwen2.5-VL M-RoPE) with one thread
// per rotated pair instead of a 256-thread block per 80-value row, and, when the rotated K is only read by a CPY to F16
// through a reshape/permute view (the cast ViT flash attention needs), written as that F16 tensor directly: the CPY
// is skipped and the F32 K never written.
bool ggml_cuda_jetson_rope_f16_enabled();
// ROPE node in vision mode that the Jetson kernel handles (F32, no frequency factors, contiguous output)
bool ggml_cuda_jetson_rope_vision_ok(const ggml_tensor * rope);
// CPY node `cpy` (to F16, contiguous) reads all of `rope` through a permutation view of its contiguous layout
bool ggml_cuda_jetson_rope_view_ok(const ggml_tensor * rope, const ggml_tensor * cpy);
// rope computed into rope->data (cpy == nullptr) or into cpy->data as F16, in the layout cpy produces
void ggml_cuda_jetson_rope_vision(ggml_backend_cuda_context & ctx, const ggml_tensor * rope, ggml_tensor * cpy);
