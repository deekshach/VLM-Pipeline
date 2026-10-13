#pragma once

#include <cuda_runtime.h>
#include <cstddef>

// ============================================================================
// Visual Pyramid CUDA Kernels for RTX 4090 (Architecture B)
//
// Three kernels for GPU-only pyramid KV prefill:
//   1. visual_attn_reduce  — reduce attention weights to per-token importance
//   2. sort_and_thresholds — CUB radix sort + per-layer threshold computation
//   3. pyramid_mask_gen    — generate per-layer drop masks on GPU
// ============================================================================

// Initialize pyramid GPU state: allocate device buffers, query CUB temp size
void pyramid_gpu_init(
    int n_visual,
    int n_layers,
    float ** d_scores,
    float ** d_scores_sorted,
    float ** d_thresholds,
    void  ** d_cub_temp,
    size_t * cub_temp_bytes,
    cudaStream_t stream = 0);

// Free pyramid GPU state
void pyramid_gpu_free(
    float * d_scores,
    float * d_scores_sorted,
    float * d_thresholds,
    void  * d_cub_temp);

// Kernel 1: Reduce [n_kv, n_heads_kv, n_q] attention to [n_visual] importance scores
void launch_visual_attn_reduce(
    const float * d_attn,       // attention weights on GPU [n_kv * n_heads_kv * n_q]
    float       * d_scores,     // output importance scores [n_visual]
    int n_heads_kv,
    int n_q,
    int n_kv,
    int visual_start,
    int n_visual,
    int tokens_per_frame,       // for frame 0 protection
    bool protect_frame0,
    cudaStream_t stream = 0);

// Kernel 2: Sort scores descending + compute per-layer thresholds
void launch_sort_and_thresholds(
    float * d_scores_sorted,    // [n_visual] — will be sorted in place
    const float * d_scores,     // [n_visual] — original unsorted scores
    float * d_thresholds,       // output [n_layers]
    void  * d_cub_temp,
    size_t  cub_temp_bytes,
    int n_visual,
    int n_layers,
    int clie_layer,
    float pvc_min,
    cudaStream_t stream = 0);

// Kernel 3: Generate drop mask for a single layer on GPU
void launch_pyramid_mask_gen(
    const float * d_base_mask,  // base KQ mask [n_kv * n_q] (can be nullptr for zero base)
    const float * d_scores,     // importance scores [n_visual]
    const float * d_thresholds, // per-layer thresholds [n_layers]
    float       * d_out_mask,   // output modified mask [n_kv * n_q]
    int n_q,
    int n_kv,
    int visual_start,
    int n_visual,
    int layer_idx,
    cudaStream_t stream = 0);

// Batch version: generate drop masks for ALL layers at once
void launch_pyramid_mask_gen_all_layers(
    const float * d_scores,     // importance scores [n_visual]
    const float * d_thresholds, // per-layer thresholds [n_layers]
    float       * d_drop_masks, // output [n_layers * n_kv * n_q]
    int n_q,
    int n_kv,
    int visual_start,
    int n_visual,
    int n_layers,
    int clie_layer,
    cudaStream_t stream = 0);
