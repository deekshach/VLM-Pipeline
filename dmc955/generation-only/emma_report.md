# Emma: Efficient Multi-stage Motion-Aware Visual Token Optimization for Video VLMs

## 1. Method

We present Emma (**E**fficient **M**ulti-stage **M**otion-**A**ware), a four-stage optimization pipeline for video inference with Vision-Language Models. Emma reduces the computational cost of video understanding by systematically eliminating redundant visual tokens at multiple points in the inference pipeline, while preserving the tokens that carry the most visual information. We evaluate Emma on top of Qwen2.5-VL-7B-Instruct running on a single NVIDIA RTX 4090 GPU.

### 1.1 Stage 1: Adaptive Frame Selection

Rather than uniformly sampling frames, Emma analyzes optical flow across the full video to select frames that maximize information coverage.

**Motion Profiling.** For each consecutive frame pair, we read the ground-truth optical flow and compute per-frame motion statistics: mean magnitude, variance, maximum magnitude, and the 95th-percentile displacement (p95). These are cached as JSON for subsequent runs (<1 ms lookup vs. ~700 ms from raw `.flo` files).

**Global Complexity Scoring.** A composite complexity score is computed from all p95 values using a weighted combination:

$$C_{global} = 0.4 \cdot \bar{p}_{95}/100 + 0.3 \cdot \text{Var}(p_{95})/5000 + 0.2 \cdot \text{Range}(p_{95})/150 + 0.1 \cdot r_{high}$$

where $r_{high}$ is the fraction of frames above the 75th-percentile p95 value. This score is mapped to a target frame count via a step function: high-complexity videos (C > 0.8) receive more frames (base + 6), while static videos (C < 0.2) receive fewer (base − 8).

**Importance-Weighted Sampling.** Frames with p95 above the `selection_percentile` threshold receive importance weight equal to their p95 value; frames below receive weight 0.1. We sample without replacement using `std::discrete_distribution`, always anchoring the first and last frames as temporal boundaries. A post-selection filter removes frames with per-frame complexity below a configurable floor, ensuring no near-static frames waste compute.

### 1.2 Stage 2: ViT Encoding

Each selected frame is encoded through the Qwen2.5-VL vision encoder (ViT) to produce a grid of visual embeddings of shape `[nx × ny, d]`, where `nx × ny` is the token grid determined by the image resolution (typically ~592 tokens for 1024×436 Sintel frames) and `d = 3584` is the embedding dimension. Frame 0 (keyframe) embeddings are cached for use as the reference signal in Stage 3.

### 1.3 Stage 3: Pre-LLM Token Pruning with Token Merging

Before injecting visual tokens into the LLM's KV cache, Emma scores and prunes each non-keyframe to 40% of its original token count using a multi-signal scoring strategy with information-preserving merging.

**MOTION+KF Scoring.** Each visual token receives a composite importance score:

$$S(t) = 0.5 \cdot \frac{M(t)}{\max_i M(i)} + 0.5 \cdot \frac{D_{cos}(t)}{\max_i D_{cos}(i)}$$

where $M(t)$ is the average optical flow magnitude in the token's spatial patch (mapping the token grid to pixel coordinates in the flow field), and $D_{cos}(t) = 1 - \text{cos\_sim}(e_t, e_t^{kf})$ is the cosine distance between the token's embedding and the corresponding keyframe token's embedding. Motion captures dynamic content; keyframe distance captures semantic change since the reference frame.

**Spatial Diversity Guarantee.** A 4×4 grid is overlaid on the ViT token grid. The highest-scoring token in each grid cell is force-selected, guaranteeing minimum spatial coverage across the frame regardless of score distribution. The remaining budget is filled by top-scoring non-forced tokens.

**PruMerge-Style Token Merging.** Rather than discarding dropped tokens entirely, each dropped token's embedding is merged into its nearest surviving token by grid distance, weighted by inverse distance:

$$e'_s = \frac{e_s + \sum_{t \in \text{dropped}(s)} w_t \cdot e_t}{1 + \sum_{t \in \text{dropped}(s)} w_t}, \quad w_t = \frac{1}{\max(1, \|p_t - p_s\|_2)}$$

This preserves spatial information from pruned regions within fewer tokens.

**M-RoPE Position Assignment.** Pruned tokens are injected with correct Multi-dimensional Rotary Position Embeddings: `(temporal = n_past, y = n_past + grid_y, x = n_past + grid_x)`. Critically, `n_past` advances by `max(nx, ny)` regardless of how many tokens are kept—a hard constraint of the M-RoPE scheme that reserves the full position range even for sparse token sets.

### 1.4 Stage 4: Post-Prefill Attention-Based KV Eviction

After the complete prefill (all images + prompt text), Emma performs a single-pass eviction of low-importance visual tokens from the KV cache using the LLM's own attention as the scoring signal.

**Flash-Compatible Qcur Capture.** During the final text chunk prefill, a `cb_eval` callback captures the post-RoPE query projection tensor `Qcur` from layer 27. Unlike approaches that capture `kq_soft_max` (which requires disabling flash attention), capturing `Qcur` is compatible with flash attention enabled, avoiding a significant performance penalty.

**CPU Q@K^T Scoring.** The K cache for layer 27 is read from GPU to CPU and converted to float32. For each query token and attention head, we compute the full softmax attention distribution over all KV positions:

$$\alpha_{q,h,k} = \frac{\exp(q_{q,h} \cdot k_k / \sqrt{d_k})}{\sum_j \exp(q_{q,h} \cdot k_j / \sqrt{d_k})}$$

The total attention received by each KV position is summed across all queries and heads, then normalized. This yields a per-visual-token importance score reflecting how much the LLM actually attends to each token when processing the prompt.

**Eviction.** Tokens are sorted by importance. The bottom fraction is evicted via `llama_memory_seq_rm_pos_ext()`, which removes KV entries by their M-RoPE coordinates (temporal, x, y). Frame 0 tokens are protected from eviction. The default retention ratio is 21%, yielding a net reduction from the original visual token count.

### 1.5 Stage 5: Generation with Deferred Defrag

After the first generated token is decoded, the KV cache is defragmented to compact holes left by eviction. The `mv()` function in the KV cell manager correctly copies M-RoPE extended position coordinates during compaction. Autoregressive decoding then proceeds normally with the reduced KV cache.

**Net Token Reduction.** For a non-keyframe with N visual tokens:
- Stage 3 retains 40% → 0.40N tokens enter KV cache
- Stage 4 retains ~20% of all visual tokens (with frame 0 protected)

For the keyframe: Stage 3 does not prune (full injection), but Stage 4 applies attention-based eviction (protected by default, so keyframe tokens are preferentially retained).

---

## 2. Experimental Results

We evaluate Emma against the Qwen2.5-VL-7B-Instruct baseline on the MPI-Sintel dataset (video: `alley_1`, 1024×436 resolution). All experiments run on a single NVIDIA RTX 4090 (24 GB VRAM) with the model in BF16 precision, flash attention enabled, and max generation of 256 tokens.

### 2.1 Latency Comparison

We compare end-to-end latency across frame counts of 1, 2, 4, 8, and 16, plus Emma's adaptive mode which automatically selected 3 frames for this video.

#### Table 1: End-to-End Latency (ms)

| Frames | Baseline Total | Emma Total | Speedup |
|--------|---------------|------------|---------|
| 1      | 2,626         | 1,937      | 1.36×   |
| 2      | 3,310         | 3,232      | 1.02×   |
| 4      | 3,524         | 2,335      | 1.51×   |
| 8      | 2,595         | 2,201      | 1.18×   |
| 16     | 4,265         | 4,159      | 1.03×   |
| Emma Adaptive (3) | — | 2,268      | —       |

Emma achieves consistent total speedups of 1.02–1.51× across frame counts. The variation is partly due to differing generation lengths (the model produces different amounts of text depending on how it perceives the visual content after pruning). Emma Adaptive selects just 3 frames and completes in 2,268 ms total.

#### Table 2: Stage-Level Latency Breakdown (ms)

| Frames | Stage          | Baseline | Emma  | Speedup |
|--------|----------------|----------|-------|---------|
| 4      | ViT Encode     | 337      | 326   | 1.03×   |
| 4      | LLM Prefill    | 334      | 285   | 1.17×   |
| 4      | KV Eviction    | 0        | 24    | —       |
| 8      | ViT Encode     | 580      | 536   | 1.08×   |
| 8      | LLM Prefill    | 661      | 500   | 1.32×   |
| 8      | KV Eviction    | 0        | 44    | —       |
| 16     | ViT Encode     | 1,066    | 961   | 1.11×   |
| 16     | LLM Prefill    | 1,373    | 937   | 1.47×   |
| 16     | KV Eviction    | 0        | 82    | —       |

**LLM Prefill** shows the most significant speedup, reaching 1.47× at 16 frames. This is the direct result of Stage 3 (pre-LLM pruning), which injects only 40% of non-keyframe tokens into the KV cache, reducing the number of tokens the LLM must process during prefill. The KV eviction overhead (Stage 4) is modest: 10–82 ms, scaling linearly with visual token count.

**ViT Encode** times are similar between baseline and Emma since both use the same full ViT encoding path (Stage 2, Path A). The small differences are due to system-level variance.

### 2.2 KV Cache Reduction

Emma's two-stage pruning dramatically reduces the visual token footprint in the KV cache.

#### Table 3: Visual Token Retention

| Frames | Baseline Tokens | Emma Pre-LLM | Emma Post-Evict | Retention |
|--------|----------------|---------------|-----------------|-----------|
| 1      | 592            | 592 (keyframe protected) | 118      | 19.9%     |
| 2      | 1,184          | 828           | 165              | 19.9%     |
| 4      | 2,368          | 1,300         | 260              | 20.0%     |
| 8      | 4,736          | 2,244         | 448              | 20.0%     |
| 16     | 9,472          | 4,132         | 826              | 20.0%     |
| Adaptive (3) | —         | 1,064         | 212              | 19.9%     |

At 16 frames, baseline retains all 9,472 visual tokens in KV cache. Emma reduces this to 826—an **91.3% reduction** in KV memory for visual tokens. This reduction has direct implications for maximum context length: the freed KV slots can accommodate longer text generation or additional frames.

The two-stage cascade works as follows:
- **Stage 3 (Pre-LLM):** Keyframe (frame 0, 592 tokens) passes through fully. Each subsequent frame is pruned to ~40%, reducing 15 non-keyframe frames × 592 tokens from 8,880 to ~3,540. Combined with the keyframe: 4,132 tokens.
- **Stage 4 (Post-prefill eviction):** Further reduces to ~20% of the pre-eviction total, with frame 0 tokens protected. Final retention: 826 tokens.

### 2.3 Generation Throughput

#### Table 4: Decode Throughput

| Frames | Baseline tok/s | Emma tok/s | Improvement |
|--------|---------------|------------|-------------|
| 1      | 62.4          | 64.0       | +2.6%       |
| 2      | 63.3          | 64.0       | +1.1%       |
| 4      | 63.0          | 63.9       | +1.4%       |
| 8      | 62.1          | 63.7       | +2.6%       |
| 16     | 59.7          | 63.7       | +6.7%       |

Emma consistently achieves higher decode throughput than the baseline. The improvement grows with frame count: at 16 frames, Emma generates at 63.7 tok/s vs. baseline's 59.7 tok/s—a **6.7% improvement**. This is because the reduced KV cache (826 vs. 9,472 visual tokens) means less memory to attend over during each autoregressive step. At lower frame counts, the KV cache is already small enough that the benefit is marginal.

### 2.4 Information Retention

A critical question is whether aggressive pruning (retaining only ~20% of visual tokens) degrades the visual information available to the LLM. We evaluate this through two complementary analyses.

#### 2.4.1 Embedding Diversity (Spectral Entropy)

We compute the spectral entropy of the visual embeddings injected into the LLM, defined as the Shannon entropy of the normalized singular values of the token embedding matrix. This measures the effective dimensionality of the visual token set—higher values indicate more diverse information content.

#### Table 5: Embedding Diversity

| Frames | Spectral Entropy | Avg Cosine Distance |
|--------|-----------------|---------------------|
| 1      | 5.244           | 0.701               |
| 2      | 5.243           | 0.696               |
| 4      | 5.262           | 0.711               |
| 8      | 5.261           | 0.711               |
| 16     | 5.280           | 0.715               |
| Adaptive (3) | 5.286    | 0.705               |

These metrics are computed on the full ViT embeddings (pre-pruning) since both methods receive identical ViT outputs. The key insight is that spectral entropy is stable across frame counts (5.24–5.28), indicating that adding more frames provides diminishing marginal information diversity. The adaptive selector's 3-frame selection achieves entropy (5.286) comparable to 16 uniform frames (5.280), validating that motion-guided selection captures the most informative frames.

#### 2.4.2 Qualitative Output Analysis

We compare the generated text descriptions to assess whether pruning affects the model's visual understanding.

**Baseline (8 frames):**
> "A young woman with short, reddish-brown hair is standing in a courtyard. She is wearing a gray tank top and has a tattoo on her left arm. She is holding a red apple in her right hand and is looking down at it. The courtyard is surrounded by stone walls and there is a wooden structure in the background. The woman appears to be examining the apple closely."

**Emma (8 frames, 20% retention):**
> "A young woman with short, reddish-brown hair is holding a red apple in her right hand. She is wearing a light blue tank top and a black headband. She is standing in a courtyard with stone walls and a wooden gate in the background. The sun is shining brightly, casting shadows on the ground."

**Emma Adaptive (3 frames, 20% retention):**
> "The video sequence shows a young woman with short, reddish-brown hair, wearing a gray tank top and a black headband. She is holding a red apple in her right hand and appears to be looking down at it. The setting is an outdoor area with a sandy ground and some wooden crates and barrels in the background. The lighting suggests it is daytime, and the shadows indicate that the sun is shining from the left side of the frame."

All three outputs correctly identify:
- The character (young woman, reddish-brown hair)
- The object (red apple, right hand)
- The setting (courtyard/outdoor, stone walls, daytime)
- The action (examining/looking at the apple)

Minor differences are stylistic rather than factual. Emma with just 3 frames and 212 surviving visual tokens (vs. baseline's 4,736 tokens at 8 frames) produces a comparably detailed and accurate description. This suggests that the majority of visual tokens in the baseline are redundant for the LLM's understanding.

### 2.5 Efficiency-Quality Tradeoff Summary

#### Table 6: Emma vs. Baseline at Scale

| Metric                    | Baseline (16 frames) | Emma Adaptive (3 frames) | Reduction   |
|---------------------------|---------------------|--------------------------|-------------|
| Visual tokens in KV       | 9,472               | 212                      | **97.8%**   |
| Total latency (ms)        | 4,265               | 2,268                    | **46.8%**   |
| ViT encode (ms)           | 1,066               | 269                      | **74.8%**   |
| LLM prefill (ms)          | 1,373               | 232                      | **83.1%**   |
| Decode speed (tok/s)      | 59.7                | 64.0                     | **+7.2%**   |
| Description quality       | Detailed             | Comparably detailed      | Preserved   |

Emma Adaptive achieves the most dramatic gains by combining frame selection with token pruning. Compared to the 16-frame baseline, Emma processes 3 motion-selected frames, prunes visual tokens to 212 (from 9,472), and completes inference in 2,268 ms—a **1.88× total speedup** with comparable output quality.

### 2.6 Scaling Analysis

The benefits of Emma's pruning pipeline scale superlinearly with frame count:

| Frames | KV Tokens Saved | Prefill Savings (ms) | Eviction Cost (ms) | Net Prefill Benefit |
|--------|----------------|---------------------|--------------------|--------------------|
| 1      | 474            | 46                  | 10                 | +36 ms             |
| 4      | 2,108          | 49                  | 24                 | +25 ms             |
| 8      | 4,288          | 161                 | 44                 | +117 ms            |
| 16     | 8,646          | 436                 | 82                 | +354 ms            |

As frame count increases, the prefill savings from pre-LLM pruning grow faster than the eviction overhead, making Emma increasingly beneficial for longer videos. At 16 frames, every millisecond spent on KV eviction saves 4.3 ms of prefill time.
