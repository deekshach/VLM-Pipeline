#pragma once

#include "common.cuh"

void ggml_cuda_op_visual_attn_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_pyramid_mask_gen(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
