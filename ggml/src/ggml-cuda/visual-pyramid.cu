// Visual Pyramid CUDA Kernels for Architecture B (RTX 4090)
// GGML_OP_VISUAL_ATTN_REDUCE: reduce attention → importance scores + thresholds
// GGML_OP_PYRAMID_MASK_GEN: generate per-layer drop mask on GPU

#include "visual-pyramid.cuh"
#include <cfloat>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Kernel: visual_attn_reduce
// One warp per visual token. Sums attention received across all heads and queries.
// ============================================================================

static __global__ void visual_attn_reduce_kernel(
    const float * __restrict__ attn,    // [n_kv, n_heads_kv, n_q]
    float       * __restrict__ output,  // [n_visual + n_layers]
    int n_heads_kv, int n_q, int n_kv,
    int visual_start, int n_visual, int n_layers, int clie_layer,
    float pvc_min, int tokens_per_frame, int protect_frame0)
{
    int vis = blockIdx.x;
    if (vis >= n_visual) return;

    int kv_pos = visual_start + vis;

    float sum = 0.0f;
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
        if (protect_frame0 && vis < tokens_per_frame) {
            output[vis] = FLT_MAX;
        } else {
            output[vis] = sum;
        }
    }
}

// ============================================================================
// Kernel: compute thresholds from sorted scores using cosine schedule
// ============================================================================

static __global__ void compute_thresholds_kernel(
    const float * __restrict__ scores,  // [n_visual] unsorted
    float       * __restrict__ output,  // write thresholds at offset n_visual
    int n_visual, int n_layers, int clie_layer, float pvc_min, int output_offset)
{
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= n_layers) return;

    if (il < clie_layer) {
        output[output_offset + il] = -FLT_MAX;
        return;
    }

    // Cosine schedule
    float ratio = pvc_min + (1.0f - pvc_min) * 0.5f
        * (1.0f + cosf((float)M_PI * (float)il / (float)(n_layers - 1)));
    int k = max(1, (int)(ratio * n_visual));

    // Simple O(n*k) selection for threshold (good enough for n_visual < 10K)
    // Find the k-th largest score
    float threshold = -FLT_MAX;
    int count_above = 0;

    // Binary search approach: estimate threshold
    float lo = -FLT_MAX, hi = FLT_MAX;
    for (int iter = 0; iter < 32; iter++) {
        float mid = (lo == -FLT_MAX) ? (hi == FLT_MAX ? 0.0f : hi - 1.0f) :
                    (hi == FLT_MAX ? lo + 1.0f : (lo + hi) * 0.5f);
        if (lo == -FLT_MAX && hi == FLT_MAX) {
            // First pass: find range
            float mn = FLT_MAX, mx = -FLT_MAX;
            for (int v = 0; v < n_visual; v++) {
                float s = scores[v];
                if (s < FLT_MAX) { // skip protected tokens
                    mn = fminf(mn, s);
                    mx = fmaxf(mx, s);
                }
            }
            lo = mn;
            hi = mx;
            mid = (lo + hi) * 0.5f;
        }
        count_above = 0;
        for (int v = 0; v < n_visual; v++) {
            if (scores[v] >= mid) count_above++;
        }
        if (count_above >= k) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    output[output_offset + il] = lo;
}

// ============================================================================
// Kernel: pyramid_mask_gen
// Generates drop mask for a single layer on GPU
// ============================================================================

static __global__ void pyramid_mask_gen_kernel(
    const float * __restrict__ base_mask,
    const float * __restrict__ scores,
    const float * __restrict__ thresholds,  // read threshold from device memory
    float       * __restrict__ out_mask,
    int total_elems, int n_kv,
    int visual_start, int n_visual, int layer_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    float threshold = thresholds[layer_idx];  // read from GPU memory (no host sync)
    int kv_pos = idx % n_kv;

    bool is_visual = (kv_pos >= visual_start &&
                      kv_pos <  visual_start + n_visual);
    bool drop = is_visual &&
                scores[kv_pos - visual_start] < threshold;

    out_mask[idx] = drop ? -INFINITY : base_mask[idx];
}

// ============================================================================
// Host dispatch functions
// ============================================================================

void ggml_cuda_op_visual_attn_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // kq_soft [n_kv, n_heads_kv, n_q]

    int visual_start    = ggml_get_op_params_i32(dst, 0);
    int n_visual        = ggml_get_op_params_i32(dst, 1);
    int n_layers        = ggml_get_op_params_i32(dst, 2);
    int clie_layer      = ggml_get_op_params_i32(dst, 3);
    float pvc_min;
    memcpy(&pvc_min, (int32_t *)dst->op_params + 4, sizeof(float));
    int tokens_per_frame = ggml_get_op_params_i32(dst, 5);
    int protect_frame0   = ggml_get_op_params_i32(dst, 6);

    int n_kv    = src0->ne[0];
    int n_heads = src0->ne[1];
    int n_q     = src0->ne[2];

    cudaStream_t stream = ctx.stream();

    // Step 1: Reduce attention to importance scores (output[0..n_visual-1])
    if (n_visual > 0) {
        visual_attn_reduce_kernel<<<n_visual, 32, 0, stream>>>(
            (const float *)src0->data, (float *)dst->data,
            n_heads, n_q, n_kv,
            visual_start, n_visual, n_layers, clie_layer,
            pvc_min, tokens_per_frame, protect_frame0);
    }

    // Step 2: Compute per-layer thresholds (output[n_visual..n_visual+n_layers-1])
    // Using a simple single-thread-per-layer kernel with binary search
    int block = 32;
    int grid = (n_layers + block - 1) / block;
    compute_thresholds_kernel<<<grid, block, 0, stream>>>(
        (const float *)dst->data,  // scores in first n_visual elements
        (float *)dst->data,        // thresholds at offset n_visual
        n_visual, n_layers, clie_layer, pvc_min, n_visual);
}

void ggml_cuda_op_pyramid_mask_gen(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // base KQ mask [n_kv, n_q]
    const ggml_tensor * src1 = dst->src[1];  // scores+thresholds [n_visual + n_layers]

    int visual_start = ggml_get_op_params_i32(dst, 0);
    int n_visual     = ggml_get_op_params_i32(dst, 1);
    int layer_idx    = ggml_get_op_params_i32(dst, 2);
    int n_layers     = ggml_get_op_params_i32(dst, 3);

    int n_kv = src0->ne[0];
    int n_q  = src0->ne[1];
    int total = n_kv * n_q;

    cudaStream_t stream = ctx.stream();

    // Zero CPU-GPU sync: threshold is read from device memory INSIDE the kernel
    // thresholds are at offset n_visual in src1
    int block = 256;
    int grid = (total + block - 1) / block;
    pyramid_mask_gen_kernel<<<grid, block, 0, stream>>>(
        (const float *)src0->data,
        (const float *)src1->data,                  // scores [n_visual]
        (const float *)src1->data + n_visual,       // thresholds [n_layers]
        (float *)dst->data,
        total, n_kv, visual_start, n_visual, layer_idx);
}
