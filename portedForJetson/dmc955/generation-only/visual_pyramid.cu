// Visual Pyramid CUDA Kernels for RTX 4090 (Architecture B)
//
// Implements GPU-only pyramid KV prefill scoring, threshold computation,
// and mask generation. Zero CPU-GPU syncs during prefill.

#include "visual_pyramid.cuh"
#include <cub/cub.cuh>
#include <cfloat>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Kernel 1: visual_attn_reduce_kernel
//
// Reduces [n_kv, n_heads_kv, n_q] attention weights to [n_visual] importance.
// One warp (32 threads) per visual position. Sums across all heads and queries.
// ============================================================================

__global__ void visual_attn_reduce_kernel(
    const float * __restrict__ attn,
    float       * __restrict__ scores,
    int n_heads_kv, int n_q, int n_kv,
    int visual_start, int n_visual,
    int tokens_per_frame, bool protect_frame0)
{
    int vis = blockIdx.x;
    if (vis >= n_visual) return;

    int kv_pos = visual_start + vis;

    float sum = 0.0f;
    // Attention tensor layout after ggml_soft_max_ext and permute:
    // [n_kv, n_heads_kv, n_q] with strides [1, n_kv, n_kv*n_heads_kv]
    // So element [kv, h, q] = attn[(h * n_q + q) * n_kv + kv]
    for (int h = 0; h < n_heads_kv; h++) {
        for (int q = threadIdx.x; q < n_q; q += blockDim.x) {
            sum += attn[(h * n_q + q) * n_kv + kv_pos];
        }
    }

    // Warp reduction
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    }

    if (threadIdx.x == 0) {
        // Protect frame 0: set importance to max float
        if (protect_frame0 && vis < tokens_per_frame) {
            scores[vis] = FLT_MAX;
        } else {
            scores[vis] = sum;
        }
    }
}

// ============================================================================
// Kernel 2a: compute_thresholds_kernel
//
// After CUB sort (descending), reads sorted scores and writes one threshold
// per layer using the cosine retention schedule.
// ============================================================================

__global__ void compute_thresholds_kernel(
    const float * __restrict__ sorted_scores_desc,
    float       * __restrict__ thresholds,
    int n_visual, int n_layers, int clie_layer, float pvc_min)
{
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= n_layers) return;

    if (il < clie_layer) {
        // Scout layers — no threshold needed (keep all)
        thresholds[il] = -FLT_MAX;
        return;
    }

    // Cosine schedule: ratio = min + (max - min) * 0.5 * (1 + cos(pi * il / (n_layers-1)))
    float ratio = pvc_min + (1.0f - pvc_min) * 0.5f
        * (1.0f + cosf((float)M_PI * (float)il / (float)(n_layers - 1)));
    int k = max(1, (int)(ratio * n_visual));

    // sorted_scores_desc[k-1] is the k-th largest score
    thresholds[il] = sorted_scores_desc[k - 1];
}

// ============================================================================
// Kernel 3: pyramid_mask_gen_kernel
//
// Generates per-layer modified KQ mask entirely on GPU.
// For visual positions below threshold: inject -INF.
// For all other positions: copy base mask (or 0 if no base).
// ============================================================================

__global__ void pyramid_mask_gen_kernel(
    const float * __restrict__ base_mask,   // can be nullptr
    const float * __restrict__ scores,
    const float * __restrict__ thresholds,
    float       * __restrict__ out_mask,
    int n_q, int n_kv,
    int visual_start, int n_visual, int layer_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_q * n_kv;
    if (idx >= total) return;

    int kv_pos = idx % n_kv;

    bool is_visual = (kv_pos >= visual_start &&
                      kv_pos <  visual_start + n_visual);
    bool drop = is_visual &&
                scores[kv_pos - visual_start] < thresholds[layer_idx];

    float base_val = base_mask ? base_mask[idx] : 0.0f;
    out_mask[idx] = drop ? -INFINITY : base_val;
}

// Batch version: one layer per grid Y dimension
__global__ void pyramid_mask_gen_all_kernel(
    const float * __restrict__ scores,
    const float * __restrict__ thresholds,
    float       * __restrict__ drop_masks,  // [n_layers, n_q, n_kv]
    int n_q, int n_kv,
    int visual_start, int n_visual,
    int n_layers, int clie_layer)
{
    int il = blockIdx.y;
    if (il < clie_layer || il >= n_layers) return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_q * n_kv;
    if (idx >= total) return;

    int kv_pos = idx % n_kv;

    bool is_visual = (kv_pos >= visual_start &&
                      kv_pos <  visual_start + n_visual);
    bool drop = is_visual &&
                scores[kv_pos - visual_start] < thresholds[il];

    size_t layer_offset = (size_t)il * n_q * n_kv;
    drop_masks[layer_offset + idx] = drop ? -INFINITY : 0.0f;
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void pyramid_gpu_init(
    int n_visual, int n_layers,
    float ** d_scores, float ** d_scores_sorted, float ** d_thresholds,
    void ** d_cub_temp, size_t * cub_temp_bytes,
    cudaStream_t stream)
{
    cudaMalloc(d_scores, n_visual * sizeof(float));
    cudaMalloc(d_scores_sorted, n_visual * sizeof(float));
    cudaMalloc(d_thresholds, n_layers * sizeof(float));

    // Query CUB temp storage size
    *cub_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeysDescending(
        nullptr, *cub_temp_bytes,
        *d_scores_sorted, *d_scores_sorted,
        n_visual, 0, 32, stream);
    cudaMalloc(d_cub_temp, *cub_temp_bytes);

    printf("[pyramid_gpu_init] n_visual=%d n_layers=%d cub_temp=%.1f KB\n",
           n_visual, n_layers, (float)*cub_temp_bytes / 1024.0f);
}

void pyramid_gpu_free(
    float * d_scores, float * d_scores_sorted,
    float * d_thresholds, void * d_cub_temp)
{
    if (d_scores)        cudaFree(d_scores);
    if (d_scores_sorted) cudaFree(d_scores_sorted);
    if (d_thresholds)    cudaFree(d_thresholds);
    if (d_cub_temp)      cudaFree(d_cub_temp);
}

void launch_visual_attn_reduce(
    const float * d_attn, float * d_scores,
    int n_heads_kv, int n_q, int n_kv,
    int visual_start, int n_visual,
    int tokens_per_frame, bool protect_frame0,
    cudaStream_t stream)
{
    if (n_visual <= 0) return;
    // One warp per visual token
    visual_attn_reduce_kernel<<<n_visual, 32, 0, stream>>>(
        d_attn, d_scores,
        n_heads_kv, n_q, n_kv,
        visual_start, n_visual,
        tokens_per_frame, protect_frame0);
}

void launch_sort_and_thresholds(
    float * d_scores_sorted, const float * d_scores,
    float * d_thresholds,
    void * d_cub_temp, size_t cub_temp_bytes,
    int n_visual, int n_layers, int clie_layer, float pvc_min,
    cudaStream_t stream)
{
    if (n_visual <= 0) return;

    // Copy scores to sorted buffer
    cudaMemcpyAsync(d_scores_sorted, d_scores,
                    n_visual * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // CUB sort descending
    cub::DeviceRadixSort::SortKeysDescending(
        d_cub_temp, cub_temp_bytes,
        d_scores_sorted, d_scores_sorted,
        n_visual, 0, 32, stream);

    // Compute per-layer thresholds
    int block = 32;
    int grid = (n_layers + block - 1) / block;
    compute_thresholds_kernel<<<grid, block, 0, stream>>>(
        d_scores_sorted, d_thresholds,
        n_visual, n_layers, clie_layer, pvc_min);
}

void launch_pyramid_mask_gen(
    const float * d_base_mask, const float * d_scores,
    const float * d_thresholds, float * d_out_mask,
    int n_q, int n_kv, int visual_start, int n_visual,
    int layer_idx, cudaStream_t stream)
{
    int total = n_q * n_kv;
    if (total <= 0) return;
    int block = 128;
    int grid = (total + block - 1) / block;
    pyramid_mask_gen_kernel<<<grid, block, 0, stream>>>(
        d_base_mask, d_scores, d_thresholds, d_out_mask,
        n_q, n_kv, visual_start, n_visual, layer_idx);
}

void launch_pyramid_mask_gen_all_layers(
    const float * d_scores, const float * d_thresholds,
    float * d_drop_masks,
    int n_q, int n_kv, int visual_start, int n_visual,
    int n_layers, int clie_layer, cudaStream_t stream)
{
    int total = n_q * n_kv;
    if (total <= 0) return;
    int block = 256;
    int grid_x = (total + block - 1) / block;
    dim3 grid(grid_x, n_layers);
    pyramid_mask_gen_all_kernel<<<grid, block, 0, stream>>>(
        d_scores, d_thresholds, d_drop_masks,
        n_q, n_kv, visual_start, n_visual,
        n_layers, clie_layer);
}
