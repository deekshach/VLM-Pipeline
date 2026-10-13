#include "kv_cache_rerotate.cuh"
#include <cstdio>

// Kernel A: Copy K values with temporal RoPE re-rotation
// K layout (ggml): [n_embd_k_gqa, kv_size] where n_embd_k_gqa = n_embd_head_k * n_kv_heads
// For cell c, head h, pair p: offset = c * n_embd_k_gqa + h * n_embd_head_k + p * 2
// Grid:  (n_tokens, n_kv_heads)
// Block: (n_pairs) where n_pairs = n_embd_head_k / 2
__global__ void kv_cache_copy_rerotate_k(
    const int * __restrict__ src_cells,
    const int * __restrict__ dst_cells,
    half * __restrict__ k_data,
    int n_embd_head_k,
    int n_embd_k_gqa,     // = n_embd_head_k * n_kv_heads
    int rope_sec_temporal, // rope_sections[0]: temporal dim count (in pairs)
    float freq_base,
    int delta_pos)         // new_temporal_pos - old_temporal_pos
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int pair_idx  = threadIdx.x;

    const int n_pairs = n_embd_head_k / 2;
    if (pair_idx >= n_pairs) return;

    const int src_cell = src_cells[token_idx];
    const int dst_cell = dst_cells[token_idx];

    const int src_base = src_cell * n_embd_k_gqa + head_idx * n_embd_head_k + pair_idx * 2;
    const int dst_base = dst_cell * n_embd_k_gqa + head_idx * n_embd_head_k + pair_idx * 2;

    // Read source K values
    half k_re = k_data[src_base];
    half k_im = k_data[src_base + 1];

    // M-RoPE sections: pairs [0, sec_temporal) = temporal dimension
    // Y, X, and unused sectors: straight copy (spatial positions unchanged for static patches)
    if (pair_idx < rope_sec_temporal && delta_pos != 0) {
        // Temporal sector: apply RoPE correction
        float theta = (float)delta_pos * powf(freq_base, -2.0f * (float)pair_idx / (float)n_embd_head_k);
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float re = __half2float(k_re);
        float im = __half2float(k_im);

        float new_re = re * cos_t - im * sin_t;
        float new_im = re * sin_t + im * cos_t;

        k_re = __float2half(new_re);
        k_im = __float2half(new_im);
    }

    // Write to destination
    k_data[dst_base]     = k_re;
    k_data[dst_base + 1] = k_im;
}

// Kernel B (non-transposed V): Copy V values contiguously per cell
// V layout: [n_embd_v_gqa, kv_size] — each cell is a contiguous block of n_embd_v_gqa elements
// Grid:  (n_tokens)
// Block: (n_embd_v_gqa) — capped at 1024
__global__ void kv_cache_copy_v_nontrans(
    const int * __restrict__ src_cells,
    const int * __restrict__ dst_cells,
    half * __restrict__ v_data,
    int n_embd_v_gqa)
{
    const int token_idx = blockIdx.x;
    const int elem_idx  = threadIdx.x;

    if (elem_idx >= n_embd_v_gqa) return;

    const int src_cell = src_cells[token_idx];
    const int dst_cell = dst_cells[token_idx];

    v_data[dst_cell * n_embd_v_gqa + elem_idx] = v_data[src_cell * n_embd_v_gqa + elem_idx];
}

// Kernel B (transposed V): Copy V values with scattered layout
// When v_trans=true, element j of cell c is at: v_data[j * kv_size + c]
// Grid:  (n_tokens)
// Block: (n_embd_v_gqa) — capped at 1024
__global__ void kv_cache_copy_v_trans(
    const int * __restrict__ src_cells,
    const int * __restrict__ dst_cells,
    half * __restrict__ v_data,
    int n_embd_v_gqa,
    int kv_size)
{
    const int token_idx = blockIdx.x;
    const int j         = threadIdx.x;

    if (j >= n_embd_v_gqa) return;

    const int src_cell = src_cells[token_idx];
    const int dst_cell = dst_cells[token_idx];

    v_data[j * kv_size + dst_cell] = v_data[j * kv_size + src_cell];
}

void launch_kv_reuse(
    const kv_reuse_params & params,
    void ** k_data_ptrs,
    void ** v_data_ptrs,
    cudaStream_t stream)
{
    if (params.n_tokens <= 0 || params.n_layers <= 0) return;

    const int delta_pos = params.new_temporal_pos - params.old_temporal_pos;
    const int n_pairs = params.n_embd_head_k / 2;
    const int n_embd_k_gqa = params.n_embd_head_k * params.n_kv_heads;
    const int n_embd_v_gqa = params.n_embd_head_v * params.n_kv_heads;

    for (int il = 0; il < params.n_layers; il++) {
        // Launch K copy + re-rotation kernel
        {
            dim3 grid(params.n_tokens, params.n_kv_heads);
            dim3 block(n_pairs);

            kv_cache_copy_rerotate_k<<<grid, block, 0, stream>>>(
                params.src_cells,
                params.dst_cells,
                (half *)k_data_ptrs[il],
                params.n_embd_head_k,
                n_embd_k_gqa,
                params.rope_sections[0],
                params.freq_base,
                delta_pos);
        }

        // Launch V copy kernel
        {
            int block_size = (n_embd_v_gqa > 1024) ? 1024 : n_embd_v_gqa;
            dim3 grid(params.n_tokens);
            dim3 block(block_size);

            if (params.v_trans) {
                kv_cache_copy_v_trans<<<grid, block, 0, stream>>>(
                    params.src_cells,
                    params.dst_cells,
                    (half *)v_data_ptrs[il],
                    n_embd_v_gqa,
                    params.kv_size);
            } else {
                kv_cache_copy_v_nontrans<<<grid, block, 0, stream>>>(
                    params.src_cells,
                    params.dst_cells,
                    (half *)v_data_ptrs[il],
                    n_embd_v_gqa);
            }
        }
    }
}
