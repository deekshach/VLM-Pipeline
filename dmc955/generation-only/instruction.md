# Pyramid KV Prefill — Implementation Plan for llama.cpp
## Qwen2.5-VL Video Token Dropping: Jetson Orin AGX + RTX 4090

---

# OVERVIEW

This document describes the full implementation of layer-wise visual token dropping (PyramidInfer + PureKV ideas) inside llama.cpp for Qwen2.5-VL video inference. Two hardware targets require different implementations due to fundamentally different memory architectures.

**Jetson Orin AGX** — unified memory, CPU and GPU share the same DRAM pool. A "readback" is a cache coherency flush, not a PCIe copy. Cost: ~10–30 μs. Strategy: two ggml graph submissions with one readback in between.

**RTX 4090 + i9** — discrete GPU with 24 GB GDDR6X connected over PCIe 4.0. Any `cudaMemcpy` D2H incurs ~5–15 μs latency plus serialization of the GPU pipeline. Strategy: everything stays on GPU using custom CUDA kernels registered as ggml ops. Zero CPU-GPU syncs during prefill.

Both implementations share the same algorithmic idea:
- Run the first 2 transformer layers normally (the "scout" layers)
- Use the attention weights from those 2 layers to score every visual token by importance
- For each subsequent layer, keep only the top-k% most important visual tokens, where k decreases with layer depth following a cosine schedule (pyramid shape)
- Drop visual tokens by injecting -INF into the KQ attention mask at dropped positions before softmax — so softmax zeroes them out naturally

---

# PART 1 — FILES TO MODIFY AND CREATE

## Files to modify (both targets)

```
src/llama.cpp                    — main graph builder, add pyramid variants
src/llama-context.cpp/h          — context holds pyramid state
ggml/include/ggml.h              — declare new ops (RTX only)
ggml/src/ggml.c                  — register new ops (RTX only)
ggml/src/ggml-cuda/ggml-cuda.cu  — CUDA dispatch for new ops (RTX only)
```

## Files to create

```
src/pyramid_prefill.h            — shared interface, hardware dispatch
src/pyramid_prefill.cpp          — Orin implementation (Architecture A)
ggml/src/ggml-cuda/visual-pyramid.cu  — RTX CUDA kernels (Architecture B)
tests/test_pyramid_prefill.cpp   — unit + integration tests
```

---

# PART 2 — SHARED CONSTANTS AND CONFIGURATION

These values apply to both targets. Put them in `pyramid_prefill.h`.

```
CLIE_LAYER      = 2       // scout uses layers 0 and 1; scoring happens at end of layer 1
PVC_MIN_RATIO   = 0.20    // deepest layer keeps 20% of visual tokens
PVC_MAX_RATIO   = 1.00    // shallowest layers keep 100%
FIRST_FRAME_PROTECT = true  // always keep all tokens from frame 0 (PureKV anchor rule)
```

The retention ratio for layer `il` out of `n_layers` total follows a cosine schedule:

```
pvC_ratio(il) = PVC_MIN + (PVC_MAX - PVC_MIN) * 0.5 * (1 + cos(pi * il / (n_layers - 1)))
```

This gives:
- Layer 0: 100% retained
- Layer 14 (mid): ~60% retained
- Layer 27 (last): 20% retained

The importance score for each visual token is computed by summing the attention weights it receives across all KV heads and all query positions in the scout layers. Visual tokens that accumulate less attention are less important to future generation and are safe to drop.

---

# PART 3 — JETSON ORIN IMPLEMENTATION (Architecture A)

## Step A1 — Track visual token positions during batch construction

In `clip.cpp`, after `clip_image_batch_encode()` encodes the visual patches into a float tensor, and before `llama_batch_add()` is called for each patch embedding, record the mapping from patch index to llama position index.

Add a struct to hold this mapping:

```cpp
// pyramid_prefill.h

struct VisualTokenRange {
    int seq_id;
    int start_pos;    // llama position of first visual token
    int n_tokens;     // total visual token count
    int n_frames;     // number of video frames
    int tokens_per_frame;  // n_tokens / n_frames
};
```

In the calling code (wherever `llama_batch_add` loops over visual embeddings), record `batch.n_tokens` before the loop as `start_pos` and `batch.n_tokens` after as `start_pos + n_tokens`. Store this in a `VisualTokenRange` that gets passed to the pyramid prefill function.

## Step A2 — Implement the scout graph builder

Add a new function in `src/llama.cpp` named `llm_build_qwen2vl_scout`. It is identical to `llm_build_qwen2vl` for the first `CLIE_LAYER` layers, except:

1. It only builds the graph for layers 0 through `CLIE_LAYER - 1` (i.e., layers 0 and 1)
2. At the end of the last scout layer's attention block, after `ggml_soft_max`, it keeps a pointer to the attention weight tensor as a named output
3. It still calls `llm_build_kv_store` for those layers so the KV cache is populated correctly

The function signature:

```cpp
static struct ggml_cgraph * llm_build_qwen2vl_scout(
    llama_context  & lctx,
    const llama_ubatch & ubatch,
    struct ggml_tensor ** out_attn_weights);  // output: pointer to attn weight tensor
```

The `out_attn_weights` pointer should point to the soft-maxed KQ tensor from layer `CLIE_LAYER - 1`. This tensor has shape `[n_kv, n_heads_kv, n_q]` in ggml storage order.

Call `ggml_build_forward_expand(gf, *out_attn_weights)` to ensure it is not optimized away by the graph allocator.

## Step A3 — Implement the readback and threshold computation

After `ggml_backend_graph_compute(ctx->backend_gpu, scout_gf)` returns, the scout has run. On Orin this sync is just a cache coherency point, not a DMA copy.

```cpp
// Read back: [n_kv, n_heads_kv, n_q] stored row-major in ggml
std::vector<float> attn_host(ggml_nelements(attn_weights_tensor));
ggml_backend_tensor_get(attn_weights_tensor, attn_host.data(), 0,
                        attn_host.size() * sizeof(float));
```

Reduce to per-visual-token importance score:

```cpp
// n_kv = total KV positions in the context
// n_heads_kv = 8 for Qwen2.5-VL-7B (GQA groups)
// n_q = number of query tokens = ubatch.n_tokens
// visual tokens occupy positions [vrange.start_pos, vrange.start_pos + vrange.n_tokens)

std::vector<float> importance(vrange.n_tokens, 0.f);

for (int q = 0; q < n_q; q++) {
    for (int h = 0; h < n_heads_kv; h++) {
        for (int v = 0; v < vrange.n_tokens; v++) {
            int kv_pos = vrange.start_pos + v;
            // ggml layout: element index = q * n_heads_kv * n_kv + h * n_kv + kv_pos
            importance[v] += attn_host[q * n_heads_kv * n_kv + h * n_kv + kv_pos];
        }
    }
}
```

If `FIRST_FRAME_PROTECT` is true, set `importance[v] = FLT_MAX` for all `v < tokens_per_frame`. This guarantees the anchor frame is never dropped regardless of threshold.

Compute per-layer thresholds using `std::nth_element` (O(n) average):

```cpp
// drop_masks[il] = flat float array of size n_q * n_kv
// Values: 0.0 for kept positions, -INFINITY for dropped positions
std::vector<std::vector<float>> drop_masks(n_layers);

std::vector<float> scratch = importance;

for (int il = CLIE_LAYER; il < n_layers; il++) {
    float ratio = pvC_ratio(il, n_layers);
    int k = std::max(1, (int)(ratio * vrange.n_tokens));  // keep top-k

    // Partition so that scratch[n_visual - k] is the kth-largest
    std::nth_element(scratch.begin(),
                     scratch.begin() + (vrange.n_tokens - k),
                     scratch.end());
    float threshold = scratch[vrange.n_tokens - k];
    scratch = importance;  // reset for next iteration

    // Build mask: only visual columns get -INF where importance < threshold
    drop_masks[il].assign((size_t)n_q * n_kv, 0.f);
    for (int v = 0; v < vrange.n_tokens; v++) {
        if (importance[v] < threshold) {
            int kv_pos = vrange.start_pos + v;
            for (int q = 0; q < n_q; q++) {
                drop_masks[il][q * n_kv + kv_pos] = -INFINITY;
            }
        }
    }
}
```

Layers 0 through `CLIE_LAYER - 1` get empty `drop_masks` (no visual dropping in scout layers).

## Step A4 — Implement the main graph builder with baked masks

Add a new function `llm_build_qwen2vl_with_masks` that takes the `drop_masks` vector and threads it into each attention block. This function is identical to `llm_build_qwen2vl` except in the attention block at layer `il >= CLIE_LAYER`:

```cpp
// After computing KQ_masked = ggml_add(ctx0, KQ, KQ_causal_mask):
if (il >= CLIE_LAYER && !drop_masks[il].empty()) {
    // Create a constant tensor from the CPU-computed drop mask
    struct ggml_tensor * vdrop = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, n_kv, n_q);
    // Mark as a constant so the graph allocator doesn't try to free it
    vdrop->flags |= GGML_TENSOR_FLAG_INPUT;
    // ggml will upload this to GPU when the graph is submitted
    ggml_backend_tensor_set(vdrop, drop_masks[il].data(), 0,
                            drop_masks[il].size() * sizeof(float));
    KQ_masked = ggml_add(ctx0, KQ_masked, vdrop);
}
// Then: KQ_soft = ggml_soft_max(ctx0, KQ_masked)
// Dropped positions become exp(-INF) = 0 in softmax output
```

## Step A5 — Integrate into llama_decode flow

In the calling site where `llama_decode` currently processes the visual batch, replace with:

```
1. Build and run scout graph  (llm_build_qwen2vl_scout)
2. Readback attention weights  (ggml_backend_tensor_get)
3. Compute importance scores and thresholds on CPU
4. Build main graph with masks  (llm_build_qwen2vl_with_masks)
5. Run main graph  (ggml_backend_graph_compute)
```

The text decode loop is unchanged — it runs after step 5 as normal.

## Step A6 — Handle multi-frame temporal protection

For video input with multiple frames, apply the following logic on top of the base importance scores before computing thresholds:

For each patch position `p` within a frame (0 to `tokens_per_frame - 1`), compute the inter-frame delta:

```cpp
// For frame f > 0, penalize tokens that are similar to previous frame
// by reducing their importance. High cosine similarity = redundant.
// This is done on the embedding tensors before they enter the model,
// so it complements (not replaces) the attention-based scoring.

// After computing importance[] from attention:
for (int f = 1; f < n_frames; f++) {
    for (int p = 0; p < tokens_per_frame; p++) {
        int curr_idx = f * tokens_per_frame + p;
        int prev_idx = (f - 1) * tokens_per_frame + p;
        // If similarity to prev frame is very high, halve the importance
        // cosine_sim computed from patch embeddings before encode
        if (frame_cosine_sim[f][p] > TEMPORAL_SIMILARITY_THRESHOLD) {
            importance[curr_idx] *= 0.5f;
        }
    }
}
// Then proceed to threshold computation as normal
```

`TEMPORAL_SIMILARITY_THRESHOLD` default: 0.95. Tune based on your video content.

---

# PART 4 — RTX 4090 IMPLEMENTATION (Architecture B)

## Step B1 — Add two new ggml ops

In `ggml/include/ggml.h`, add to the `ggml_op` enum:

```c
GGML_OP_VISUAL_ATTN_REDUCE,
GGML_OP_PYRAMID_MASK_GEN,
```

In `ggml/src/ggml.c`, in the `GGML_OP_COUNT` table, add names and shapes:

For `GGML_OP_VISUAL_ATTN_REDUCE`:
- Input src0: attn_weights tensor `[n_kv, n_heads_kv, n_q]`
- Output: 1D tensor `[n_visual]`
- op_params[0] = visual_start_pos (int32)
- op_params[1] = n_visual (int32)

For `GGML_OP_PYRAMID_MASK_GEN`:
- Input src0: base KQ mask `[n_kv, n_q]`
- Input src1: scores tensor `[n_visual]`
- Input src2: thresholds tensor `[n_layers]`
- Output: modified KQ mask `[n_kv, n_q]`
- op_params[0] = visual_start_pos (int32)
- op_params[1] = n_visual (int32)
- op_params[2] = layer_idx (int32)
- op_params[3] = n_layers (int32)
- op_params[4] = pvc_min_ratio as float bits cast to int32

Add node creation helper functions:

```c
struct ggml_tensor * ggml_visual_attn_reduce(
    struct ggml_context * ctx,
    struct ggml_tensor * attn_weights,
    int visual_start, int n_visual);

struct ggml_tensor * ggml_pyramid_mask_gen(
    struct ggml_context * ctx,
    struct ggml_tensor * base_kq_mask,
    struct ggml_tensor * scores,
    struct ggml_tensor * thresholds,
    int visual_start, int n_visual, int layer_idx, int n_layers);
```

These follow the same pattern as other ggml op helpers — set `result->op`, fill `result->src[]`, write op_params.

## Step B2 — Write the CUDA kernels

Create `ggml/src/ggml-cuda/visual-pyramid.cu`.

### Kernel 1: visual_attn_reduce_kernel

Reduces the `[n_kv, n_heads_kv, n_q]` attention weight tensor to `[n_visual]` importance scores. One block per visual position.

```cuda
__global__ void visual_attn_reduce_kernel(
    const float* __restrict__ attn,
    float*       __restrict__ scores,
    int n_heads_kv, int n_q, int n_kv,
    int visual_start, int n_visual)
{
    int vis = blockIdx.x;
    if (vis >= n_visual) return;
    int kv_pos = visual_start + vis;

    float sum = 0.f;
    for (int h = 0; h < n_heads_kv; h++) {
        for (int q = threadIdx.x; q < n_q; q += blockDim.x) {
            // ggml stores [n_kv, n_heads_kv, n_q] row-major
            sum += attn[q * n_heads_kv * n_kv + h * n_kv + kv_pos];
        }
    }
    // Warp reduction
    for (int off = warpSize / 2; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    if (threadIdx.x == 0)
        scores[vis] = sum;
}
// Launch config: <<<n_visual, 32, 0, stream>>>
```

### Kernel 2: compute_thresholds_kernel

Reads the sorted scores array (after a CUB sort) and writes one threshold per layer.

```cuda
__global__ void compute_thresholds_kernel(
    const float* __restrict__ sorted_scores_desc,
    float*       __restrict__ thresholds,
    int n_visual, int n_layers, float pvc_min)
{
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= n_layers) return;

    float t = il < 2 ? 0.f :  // scout layers — no threshold needed
        pvc_min + (1.f - pvc_min) * 0.5f *
        (1.f + cosf(M_PI * (float)il / (float)(n_layers - 1)));
    int k = max(1, (int)(t * n_visual));
    thresholds[il] = sorted_scores_desc[k - 1];
}
// Launch config: <<<(n_layers+31)/32, 32, 0, stream>>>
```

For the sort itself, use CUB's `DeviceRadixSort::SortKeysDescending`:

```cuda
void sort_scores_descending(
    float* d_scores,        // [n_visual] — will be sorted in-place
    void*  d_temp,          // CUB temp storage
    size_t temp_bytes,
    int    n_visual,
    cudaStream_t stream)
{
    cub::DeviceRadixSort::SortKeysDescending(
        d_temp, temp_bytes,
        d_scores, d_scores,  // in-place
        n_visual, 0, 32, stream);
}
```

Allocate `d_temp` once at context init using `cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_bytes, ...)` to query required size.

### Kernel 3: pyramid_mask_gen_kernel

Generates the per-layer modified KQ mask entirely on GPU.

```cuda
__global__ void pyramid_mask_gen_kernel(
    const float* __restrict__ base_mask,
    const float* __restrict__ scores,
    const float* __restrict__ thresholds,
    float*       __restrict__ out_mask,
    int n_q, int n_kv,
    int visual_start, int n_visual, int layer_idx)
{
    int kv_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (kv_pos >= n_kv) return;

    bool is_visual = (kv_pos >= visual_start &&
                      kv_pos <  visual_start + n_visual);
    bool drop = is_visual &&
                scores[kv_pos - visual_start] < thresholds[layer_idx];

    for (int q = 0; q < n_q; q++) {
        int idx = q * n_kv + kv_pos;
        out_mask[idx] = drop ? -INFINITY : base_mask[idx];
    }
}
// Launch config: <<<(n_kv+127)/128, 128, 0, stream>>>
```

## Step B3 — Dispatch in ggml-cuda.cu

In the large switch statement of `ggml_cuda_compute_forward` (or equivalent dispatch function), add:

```cpp
case GGML_OP_VISUAL_ATTN_REDUCE:
{
    int visual_start = ggml_get_op_params_i32(dst, 0);
    int n_visual     = ggml_get_op_params_i32(dst, 1);
    int n_kv    = src0->ne[0];
    int n_heads = src0->ne[1];
    int n_q     = src0->ne[2];

    visual_attn_reduce_kernel<<<n_visual, 32, 0, main_stream>>>(
        (const float*)src0->data, (float*)dst->data,
        n_heads, n_q, n_kv, visual_start, n_visual);
    break;
}

case GGML_OP_PYRAMID_MASK_GEN:
{
    int visual_start = ggml_get_op_params_i32(dst, 0);
    int n_visual     = ggml_get_op_params_i32(dst, 1);
    int layer_idx    = ggml_get_op_params_i32(dst, 2);
    int n_kv = src0->ne[0];
    int n_q  = src0->ne[1];

    // src0 = base KQ mask, src1 = scores, src2 = thresholds
    pyramid_mask_gen_kernel<<<(n_kv+127)/128, 128, 0, main_stream>>>(
        (const float*)src0->data,
        (const float*)src1->data,
        (const float*)src2->data,
        (float*)dst->data,
        n_q, n_kv, visual_start, n_visual, layer_idx);
    break;
}
```

## Step B4 — Persistent GPU state in the context

Add to `llama_context` (or a sub-struct for pyramid state):

```cpp
struct PyramidGPUState {
    struct ggml_tensor * d_scores          = nullptr;  // [n_visual], device
    struct ggml_tensor * d_scores_sorted   = nullptr;  // [n_visual], scratch for CUB sort
    struct ggml_tensor * d_thresholds      = nullptr;  // [n_layers], device
    void  * d_cub_temp                     = nullptr;
    size_t  cub_temp_bytes                 = 0;
    int     visual_start_pos               = 0;
    int     n_visual                       = 0;
    bool    initialized                    = false;
};
```

Initialize once before the first visual prefill using `ggml_new_tensor_1d` with a GPU backend buffer. Query CUB temp size with a null call.

## Step B5 — Modified graph builder for RTX 4090

The function `llm_build_qwen2vl_rtx4090` is identical to `llm_build_qwen2vl` with these changes inside the per-layer loop:

```
At layer il == CLIE_LAYER - 1 (end of layer 1 attention):
    1. Keep the KQ_soft tensor reference
    2. Insert VISUAL_ATTN_REDUCE node:
           scores_node = ggml_visual_attn_reduce(ctx0, KQ_soft, visual_start, n_visual)
           ggml_build_forward_expand(gf, scores_node)
    3. Insert CUB sort + threshold node (COMPUTE_THRESHOLDS op):
           thresholds_node output depends on scores_node
           ggml_build_forward_expand(gf, thresholds_node)
    4. Store scores_node and thresholds_node in PyramidGPUState

At layer il >= CLIE_LAYER (all remaining layers):
    1. After computing KQ_masked = ggml_add(KQ, KQ_causal_mask):
    2. Insert PYRAMID_MASK_GEN node:
           drop_mask = ggml_pyramid_mask_gen(ctx0,
               KQ_masked, scores_node, thresholds_node,
               visual_start, n_visual, il, n_layers)
    3. Use drop_mask as the new KQ_masked going into ggml_soft_max
```

Because ggml tracks tensor dependencies as a DAG, the graph allocator guarantees that `VISUAL_ATTN_REDUCE` and `COMPUTE_THRESHOLDS` complete before any `PYRAMID_MASK_GEN` node executes. No explicit CUDA stream sync is needed.

## Step B6 — Single graph submission

The RTX 4090 path submits exactly one graph:

```cpp
struct ggml_cgraph * gf = llm_build_qwen2vl_rtx4090(
    lctx, ubatch, pyramid_state);
ggml_backend_graph_compute(ctx->backend_gpu, gf);
// Done. No readback. No second submission.
```

---

# PART 5 — HARDWARE DISPATCH

Add a detection function called once at context init:

```cpp
// pyramid_prefill.h

enum class PyramidTarget {
    JETSON_ORIN,   // UMA, Architecture A
    RTX_DISCRETE,  // PCIe, Architecture B
    NONE           // pyramid disabled
};

PyramidTarget detect_hardware() {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    if (prop.integrated == 1) return PyramidTarget::JETSON_ORIN;
    // Compute capability 8.9 = RTX 4090 (Ada Lovelace)
    if (prop.major == 8 && prop.minor == 9) return PyramidTarget::RTX_DISCRETE;
    // Other discrete GPUs: use Orin path (safe fallback, one readback)
    return PyramidTarget::JETSON_ORIN;
}
```

In the graph builder, dispatch:

```cpp
switch (pyramid_target) {
    case PyramidTarget::JETSON_ORIN:
        return run_pyramid_prefill_orin(lctx, ubatch, vrange);
    case PyramidTarget::RTX_DISCRETE:
        return llm_build_qwen2vl_rtx4090(lctx, ubatch, pyramid_state);
    case PyramidTarget::NONE:
        return llm_build_qwen2vl(lctx, ubatch);  // original, no dropping
}
```

---

# PART 6 — IMPORTANT CORRECTNESS RULES

These rules are not optional. Violations cause silent accuracy loss.

**Rule 1 — Never drop text tokens.** The drop mask must only touch KV positions in `[visual_start_pos, visual_start_pos + n_visual)`. All positions outside this range must have 0.0 in the drop mask, not -INF.

**Rule 2 — Never drop from the KV cache itself, only from the attention mask.** The KV cache still stores all visual tokens. The pyramid only prevents dropped tokens from being attended to. Do not call `llama_kv_cache_seq_rm` during prefill — that is a decode-stage tool.

**Rule 3 — Protect frame 0.** Always set `FIRST_FRAME_PROTECT = true` for video input. This implements PureKV's "first frame as global anchor" rule. Dropping tokens from the first frame causes catastrophic accuracy loss on temporal reasoning tasks.

**Rule 4 — The KQ mask accumulates.** The ggml attention computes `softmax(QK^T / sqrt(d) + mask)`. When you add your drop mask, you're adding to the causal mask that's already there. So a position that's causally masked (already -INF) will remain -INF. A position that's not causally masked but has your -INF drop will correctly become -INF. This additive approach is safe and requires no special handling.

**Rule 5 — CLIE_LAYER applies to the scoring only, not to the dropping.** Layers 0 and 1 run with full visual token attention. Dropping starts at layer 2. Do not apply drop masks to the scout layers.

**Rule 6 — GQA head count.** Qwen2.5-VL-7B uses Grouped Query Attention with `n_heads_q = 28` and `n_heads_kv = 8`. The attention weight tensor shape is `[n_kv, 8, n_q]` — use `n_heads_kv = 8` in the reduction, not 28.

---

# PART 7 — PARAMETERS TO EXPOSE

These should be runtime-configurable, not compile-time constants. Expose them as command-line args or context parameters:

```
--pyramid-clie-layer N       default: 2   (scout depth, try 1–4)
--pyramid-pvc-min FLOAT      default: 0.20 (minimum retention, try 0.10–0.30)
--pyramid-temporal-thresh F  default: 0.95 (inter-frame cosine threshold)
--pyramid-protect-frame0     default: true
--no-pyramid                 disable entirely, use original graph
```

---

# PART 8 — TESTING STRATEGY

## Unit tests

Write in `tests/test_pyramid_prefill.cpp`.

Test 1 — pvC_ratio schedule: verify that `pvC_ratio(0, 28) == 1.0`, `pvC_ratio(27, 28) == PVC_MIN`, and values are monotonically decreasing.

Test 2 — Importance scoring: construct a synthetic `[n_kv, n_heads_kv, n_q]` attention weight tensor where visual positions 0–3 have high scores and positions 4–9 have low scores. Verify the reduce kernel outputs match expected values.

Test 3 — Threshold computation: given known importance scores, verify `nth_element` (Orin) and CUB sort (RTX) produce identical thresholds for the same input.

Test 4 — Mask correctness: for a known threshold, verify that the drop mask has -INF exactly at the expected KV positions and 0.0 everywhere else. Verify text positions are untouched.

Test 5 — Frame 0 protection: verify that setting `FIRST_FRAME_PROTECT` causes the first `tokens_per_frame` entries in importance to be FLT_MAX, which is always above any threshold.

## Integration tests

Test 6 — Parity with full attention on short sequences: run Qwen2.5-VL with `--pyramid-pvc-min 1.0` (keep everything) and verify logits match the original `llm_build_qwen2vl` output exactly. This validates that the mask injection path is correct.

Test 7 — Monotone KV usage: after prefill with pyramid enabled, count non-dropped visual tokens per layer. Verify the count is non-increasing with layer depth.

Test 8 — Accuracy on MVBench: run the model on a sample of MVBench tasks at pvc_min=0.20 and pvc_min=1.0. Report the delta. Acceptable regression: < 2 points on aggregate score.

Test 9 — Memory reduction: before and after enabling pyramid, measure peak KV cache allocation during prefill of a 16-frame video clip. Expect > 40% reduction.

Test 10 — Latency: profile prefill time for a 16-frame clip on each device. Expect > 30% prefill speedup on RTX 4090 (compute reduction). Orin speedup is primarily memory-bound.

---

# PART 9 — BUILD SYSTEM CHANGES

In `CMakeLists.txt`:

```cmake
# Add new source files
target_sources(llama PRIVATE
    src/pyramid_prefill.cpp
)

if (GGML_CUDA)
    target_sources(ggml-cuda PRIVATE
        ggml/src/ggml-cuda/visual-pyramid.cu
    )
    # CUB is shipped with CUDA toolkit, no extra dep needed
    # Ensure compute capability includes sm_87 (Orin) and sm_89 (RTX 4090)
    set_property(TARGET ggml-cuda PROPERTY
        CUDA_ARCHITECTURES "87;89")
endif()
```

---

# PART 10 — KNOWN ISSUES AND MITIGATIONS

**Issue 1 — ggml graph allocator may free intermediate tensors.** The attention weight tensor from the scout layers will be freed by the allocator after it's consumed by the forward pass. To prevent this, call `ggml_build_forward_expand(gf, attn_tensor)` explicitly, which marks it as a graph output and prevents deallocation.

**Issue 2 — CUB temp storage size.** Query the required temp storage size for CUB RadixSort before allocating. This must be done once at context init, not inside the graph builder. The sort input size (n_visual) may vary between calls if different videos have different frame counts — allocate for the maximum expected n_visual.

**Issue 3 — Qwen2.5-VL uses window attention in the vision encoder.** The visual tokens that enter the LLM backbone (after the projector) are full dense attention tokens — they are NOT subject to window attention at that stage. The pyramid dropping operates on the backbone layers, not the ViT encoder. Do not confuse `clip.cpp` processing with backbone processing.

**Issue 4 — CUDA Graphs compatibility.** llama.cpp uses CUDA Graphs for decode. The pyramid implementation described here is for prefill only, not decode. CUDA Graph capture happens after prefill is complete. No conflict exists, but ensure the pyramid does not execute during the decode graph capture phase. Add a guard: only call pyramid graph builders when `ubatch.n_tokens > 1` (prefill mode).

**Issue 5 — Position encoding.** Qwen2.5-VL uses M-RoPE (multi-dimensional rotary position encoding) with separate spatial and temporal position indices for visual tokens. The drop mask operates on the KQ weights after RoPE is applied — this is correct and requires no position encoding changes.

---

# SUMMARY TABLE

| Item | Jetson Orin | RTX 4090 |
|---|---|---|
| GPU submissions | 2 (scout + main) | 1 (unified DAG) |
| CPU-GPU syncs | 1 (between submissions) | 0 |
| Threshold computation | CPU, std::nth_element | GPU, CUB DeviceRadixSort |
| Mask generation | CPU bakes float arrays, uploads as constants | GPU kernel PYRAMID_MASK_GEN |
| New CUDA kernels | visual_attn_reduce only | + pyramid_mask_gen + compute_thresholds |
| New ggml ops | 0 | 2 (VISUAL_ATTN_REDUCE, PYRAMID_MASK_GEN) |
| New .cu files | 0 | 1 (visual-pyramid.cu) |
| New .cpp files | 1 (pyramid_prefill.cpp) | same file, RTX path |
| CUB dependency | No | Yes (ships with CUDA toolkit) |
| Expected prefill speedup | ~20–35% (memory bound) | ~35–55% (compute + memory) |
| Expected KV memory reduction | ~50% at pvc_min=0.20 | ~50% at pvc_min=0.20 |
| Risk level | Low | Medium (new op registration) |
| Implementation order | Do this first | Do after Orin validated |
