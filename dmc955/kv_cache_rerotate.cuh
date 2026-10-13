#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct kv_reuse_params {
    int n_tokens;        // number of static tokens to copy
    int n_layers;        // number of KV layers
    int n_kv_heads;      // GQA head count
    int n_embd_head_k;   // dimension per head (K)
    int n_embd_head_v;   // dimension per head (V)
    int kv_size;         // total KV cache size (for stride computation)
    int rope_sections[4]; // [temporal, y, x, unused]
    float freq_base;
    int old_temporal_pos;
    int new_temporal_pos;
    bool v_trans;
    const int * src_cells;  // [n_tokens] GPU array of source cell indices
    const int * dst_cells;  // [n_tokens] GPU array of destination cell indices
};

// Launch KV cache reuse kernels for all layers on the given stream.
// k_tensors and v_tensors are CPU arrays of [n_layers] pointers to ggml_tensor,
// from which we extract the data pointers (which point to GPU memory).
void launch_kv_reuse(
    const kv_reuse_params & params,
    void ** k_data_ptrs,   // [n_layers] GPU data pointers for K tensors
    void ** v_data_ptrs,   // [n_layers] GPU data pointers for V tensors
    cudaStream_t stream);
