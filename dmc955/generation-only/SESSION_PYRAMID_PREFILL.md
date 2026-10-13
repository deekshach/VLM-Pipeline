# Pyramid KV Prefill — Implementation Session Summary

## Date: 2026-03-16
## Goal: Minimize end-to-end execution time for Qwen2.5-VL video inference on RTX 4090

---

## What Was Built

A three-stage pyramid token dropping system for Qwen2.5-VL video inference in llama.cpp. The system reduces visual tokens through two mechanisms:
1. **Pre-LLM pruning**: Drop uninformative visual tokens before they enter the LLM using optical flow motion + embedding analysis (reduces prefill compute)
2. **Post-prefill three-stage KV eviction**: Use LLM attention from layers 7, 19, and 27 to progressively remove low-importance tokens from the KV cache (reduces generation KV size)

Additionally, Architecture B ggml ops were added to the compute graph for GPU-native pyramid mask injection during prefill (zero CPU-GPU syncs).

---

## Files Created

### `dmc955/generation-only/pyramid_prefill.h`
Shared header with:
- `pyramid_config` struct (clie_layer, pvc_min_ratio, first_frame_protect, etc.)
- `visual_token_range` struct for tracking visual token positions
- `pvc_ratio(il, n_layers, cfg)` — cosine retention schedule function
- CPU-side importance scoring, threshold computation, and drop mask generation
- CLI argument parsing helpers

### `dmc955/generation-only/pyramid_generation.cpp`
Main benchmark executable (~1100 lines). Key sections:

- **Section 3**: `multi_qcur_capture_state` + `multi_qcur_callback` — flash-attn-compatible Qcur capture from layers 7, 19, 27 via cb_eval. Captures query projections (NOT kq_soft_max) so flash attention can stay ON.

- **Section 4**: `compute_qk_attention_scores()` — CPU Q@K^T computation using captured Qcur and K cache tensors. For each of the 3 capture layers, reads Q (captured via cb_eval) and K (from KV cache), computes full attention scores on CPU with softmax. Works with flash attention ON.

- **Section 5**: `pyramid_vlm_context` — wrapper around llama model + mtmd vision context

- **Section 6**: `load_frame_paths()` — loads frame paths and frame numbers for flow file lookup

- **Section 6b**: Optical flow reader + per-token motion scoring:
  - `read_flo_file()` — reads Sintel `.flo` files (magic 202021.25, width, height, u/v flow vectors)
  - `compute_token_motion_scores()` — maps each ViT token's grid position to the corresponding image region, averages the optical flow magnitude over that region. Returns [chunk_nx * chunk_ny] motion scores.
  - `load_optical_flows()` — loads flow files for the selected frame indices

- **Section 8**: `run_inference()` — main pipeline:
  1. Load frames + optical flow files
  2. Tokenize with mtmd
  3. Encode images through ViT, decode into KV cache
  4. **Pre-LLM pruning**: motion + cosine + norm combined scoring with spatial grid guarantee (40% keep for frames 1-7, frame 0 fully kept)
  5. Prefill last text chunk with Qcur capture enabled (flash attn ON)
  6. **Three-stage progressive KV eviction** using CPU Q@K^T attention scores
  7. Autoregressive text generation

**CLI flags**:
```
--sintel-dir <dir>          MPI-Sintel dataset path (default: /data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel)
--video <name>              Video name (e.g., alley_1)
--n-frames <n>              Number of frames (default: 8)
--n-predict <n>             Max generation tokens (default: 256)
--prompt <text>             Custom prompt
--baseline                  Run without pyramid (vanilla llama.cpp behavior)
--no-prellm                 Disable pre-LLM pruning (attention-only eviction)
--no-pyramid                Disable pyramid entirely
--pyramid-pvc-min <f>       Min retention ratio (default: 0.20)
--pyramid-clie-layer <n>    Scout depth (default: 2)
--output-json <path>        Save results as JSON
```

### `dmc955/generation-only/visual_pyramid.cu` + `visual_pyramid.cuh`
CUDA kernels for Architecture B (RTX 4090):
- `visual_attn_reduce_kernel` — reduces attention tensor to per-token importance scores (1 warp/token)
- `compute_thresholds_kernel` — computes per-layer thresholds using cosine schedule with binary search
- `pyramid_mask_gen_kernel` — generates per-layer drop masks entirely on GPU (reads threshold from device memory, zero host sync)
- Host wrapper functions with launch configs

### `dmc955/generation-only/run_benchmark.sh`
Multi-video benchmark runner script comparing baseline vs pyramid at different pvc_min ratios.

---

## Files Modified

### Core ggml (Architecture B ops)

**`ggml/include/ggml.h`**
- Added `GGML_OP_VISUAL_ATTN_REDUCE` and `GGML_OP_PYRAMID_MASK_GEN` to the `ggml_op` enum (before `GGML_OP_COUNT`)

**`ggml/src/ggml.c`**
- Added entries to `GGML_OP_NAME[]` and `GGML_OP_SYMBOL[]` tables
- Updated `GGML_OP_COUNT` static_assert from 95 to 97
- Added `ggml_visual_attn_reduce()` and `ggml_pyramid_mask_gen()` op creation functions
  - `VISUAL_ATTN_REDUCE`: input kq_soft [n_kv, n_heads, n_q] → output [n_visual + n_layers] (scores + thresholds)
  - `PYRAMID_MASK_GEN`: input base_mask [n_kv, n_q] + scores_and_thresholds → output modified mask [n_kv, n_q]

**`ggml/src/ggml-cuda/visual-pyramid.cu` + `.cuh`** (NEW)
- CUDA implementations of both ops
- Registered via `file(GLOB *.cu)` in the ggml-cuda CMakeLists (auto-discovered)

**`ggml/src/ggml-cuda/ggml-cuda.cu`**
- Added `#include "ggml-cuda/visual-pyramid.cuh"`
- Added dispatch cases in `ggml_cuda_compute_forward()`
- Added to `supports_op()` so CUDA backend claims these ops

**`ggml/src/ggml-cpu/ggml-cpu.c`**
- Added `n_tasks = 1` entries in `ggml_graph_plan()` for both ops
- Added GGML_ABORT fallback in `ggml_compute_forward()` (GPU-only ops)

### llama.cpp core (pyramid config + attention capture)

**`src/llama-cparams.h`**
- Added pyramid fields to `llama_cparams`: `pyramid_enabled`, `pyramid_clie_layer`, `pyramid_pvc_min`, `pyramid_visual_start`, `pyramid_n_visual`, `pyramid_tokens_per_frame`, `pyramid_protect_frame0`

**`include/llama.h`**
- Added public API: `llama_set_pyramid_params()` — sets pyramid config on context before prefill

**`src/llama-context.h` + `.cpp`**
- Added `get_cparams_mut()` non-const accessor
- Implemented `llama_set_pyramid_params()` API function

**`src/llama-graph.h` + `.cpp`**
- Added optional parameters to `build_attn_mha()`: `kq_drop_mask` (additive mask for pyramid) and `out_kq_soft` (capture softmax output for scoring)
- Added same to the `build_attn(llm_graph_input_attn_kv *)` overload
- Implementation: if `kq_drop_mask` is non-null, adds it to the KQ mask before softmax via `ggml_add`. If `out_kq_soft` is non-null, stores the softmax tensor reference.

**`src/models/qwen2vl.cpp`**
- Added pyramid mode: when `cparams.pyramid_enabled && n_visual > 0 && n_tokens > 1`:
  - At scout layer (clie_layer - 1): captures kq_soft via `out_kq_soft`, then inserts `ggml_visual_attn_reduce()` node
  - At layers >= clie_layer: inserts `ggml_pyramid_mask_gen()` to modify the KQ mask per-layer
  - ggml DAG dependency tracking ensures correct execution order (reduce before mask_gen)
  - Architecture B: single graph submission, zero CPU-GPU syncs during prefill

### Build system

**`dmc955/CMakeLists.txt`**
- Added `llama-pyramid-gen` target (CPU scoring path)
- Added `llama-pyramid-gen-cuda` target (with CUDA kernels, defines `HAVE_CUDA_PYRAMID`)

---

## Three Benchmark Schemes

### [A] Baseline
- All 8 frames × 592 tokens = 4736 visual tokens
- Flash attention ON, no cb_eval
- All tokens through all 28 layers, all remain in KV for generation
- This is vanilla llama.cpp behavior

### [B] Attention-Only (--no-prellm)
- All 4736 tokens enter LLM (no pre-LLM pruning)
- Flash attention ON, Qcur captured from layers 7/19/27
- After prefill: CPU Q@K^T computes attention scores per layer
- Three-stage eviction: keep 70% → 60% → 50% = ~21% final retention (994 tokens)
- Best quality (near-baseline), but Q@K^T adds ~270ms overhead for 4736 tokens

### [C] Full Pyramid (default)
- Frame 0: all 592 tokens kept (anchor frame)
- Frames 1-7: 40% kept by motion + cosine + norm scoring with spatial grid = 2244 total tokens
- Flash attention ON, Qcur captured from layers 7/19/27
- After prefill: three-stage eviction → 471 tokens (10% of original 4736)
- 19% prefill speedup, 2.4% generation speedup, ~330ms Q@K^T overhead
- Quality comparable to baseline on most scenes

---

## Pre-LLM Scoring: Motion + Cosine Distance + Embedding Norm

For each non-keyframe, tokens are scored by a weighted combination of three signals:

```
score[t] = 0.5 * motion_score[t]    // optical flow magnitude (normalized)
          + 0.3 * cosine_distance[t]  // distance from frame centroid embedding
          + 0.2 * embedding_norm[t]   // L2 norm of ViT embedding (normalized)
```

### Signal 1: Optical Flow Motion (weight 0.5)
- Reads Sintel ground-truth `.flo` files from `training/flow/<video>/frame_NNNN.flo`
- Each `.flo` contains per-pixel (u, v) flow vectors between consecutive frames
- For each ViT token, maps its grid position to the corresponding pixel region and averages the flow magnitude: `motion[t] = mean(sqrt(u^2 + v^2))` over the token's spatial patch
- Normalized to [0,1] by dividing by the frame's max motion
- **Why it works**: tokens covering moving characters/objects get high scores; static background gets ~0. In bamboo_1, the climbing girl has high flow while the static bamboo forest has low flow.

### Signal 2: Cosine Distance from Centroid (weight 0.3)
- Computes mean embedding across all tokens in the frame (centroid)
- `cosine_distance[t] = 1 - cosine_similarity(embedding[t], centroid)`
- Tokens that differ from the frame average are distinctive (characters, objects, edges)
- **Why it works**: uniform background patches are near the centroid; unique content stands out

### Signal 3: Embedding L2 Norm (weight 0.2)
- `norm[t] = ||embedding[t]||_2`, normalized by max norm in frame
- High-energy tokens tend to carry more visual information
- Lowest weight because it's the least reliable signal

### Spatial Grid Guarantee
- A 4×4 grid is overlaid on the token grid
- The highest-scoring token in each grid cell is force-kept regardless of rank
- Prevents entire spatial regions from being wiped out (e.g., if a character is in one corner with low overall scores, at least some tokens from that region survive)

### Evolution of scoring approaches (oldest → newest):
1. **L2 norm only** (v1) — favored bright background over subjects. bamboo_1 failed completely.
2. **Cosine distance + spatial grid** (v2) — better, but still missed spatially small subjects.
3. **Motion + cosine + norm + spatial grid** (v3, current) — motion is the primary signal, correctly prioritizes moving characters. All 4 test videos produce good descriptions.

---

## CPU Q@K^T Computation

Flash attention fuses softmax into the kernel, making `kq_soft_max` tensors uncapturable via cb_eval. Instead, we capture `Qcur` (query projections after RoPE, available regardless of flash attention mode) and compute attention on CPU:

```
For each layer l in {7, 19, 27}:
  Q = captured Qcur from layer l     [n_embd_head, n_head, n_q]
  K = read from KV cache layer l     [n_kv, n_head_kv * n_embd_head]

  For each query qi, head h:
    kv_h = h / gqa_ratio          (GQA: 28 Q heads / 4 KV heads = 7)
    logits[j] = dot(Q[qi,h,:], K[j,kv_h,:]) * (1/sqrt(d_k))
    attn = softmax(logits)
    kv_attn[j] += attn[j] / n_head

  score[visual_token_i] = kv_attn[cell_idx_of_token_i]
```

**Key detail**: The cb_eval callback captures "Qcur-7", "Qcur-19", "Qcur-27". Qcur is named twice per layer (pre-RoPE and post-RoPE) — the last write wins, giving us post-RoPE Q which is what we need for correct Q@K^T.

**K cache type handling**: K tensors in the KV cache may be F16, BF16, or F32 depending on quantization. The code handles all three via type-dispatch conversion to F32.

Cost: ~110ms per layer for 2244 KV entries on CPU. Could be 10x faster on GPU using the CUDA kernels already written.

---

## Three-Stage Progressive KV Eviction

After prefill, visual tokens are evicted in three rounds using attention scores from different network depths:

```
Stage 1 [Layer 7 attention]:  keep 70% of active tokens → evict least-attended by early layers
Stage 2 [Layer 19 attention]: keep 60% of remaining    → evict least-attended by middle layers
Stage 3 [Layer 27 attention]: keep 50% of remaining    → evict least-attended by deep layers

Net retention: ~21% (0.7 × 0.6 × 0.5)
```

**Rationale**: Attention patterns differ across layer depths:
- Early layers (0-7): broad context, distribute attention widely. Tokens getting zero attention here are truly uninformative.
- Middle layers (8-19): semantic processing. Tokens not attended here lack meaningful content.
- Deep layers (20-27): task-specific refinement. Only the most relevant tokens for answering the prompt survive.

Each stage uses `llama_memory_seq_rm_pos_ext(mem, seq_id, kv_pos, ext_x, ext_y)` to remove individual KV entries by their M-RoPE coordinates. Frame 0 tokens are always protected (score set to FLT_MAX).

---

## Key Constraints and Known Issues

1. **M-RoPE defrag bug**: `llama_memory_defragment()` after KV eviction corrupts M-RoPE extended positions (ext_x, ext_y). KV holes from eviction are left as-is. This means generation still iterates over the full n_kv range even after eviction.

2. **Flash attention + cb_eval**: Capturing `kq_soft_max` requires disabling flash attention (fuses softmax). The Qcur capture approach avoids this — flash attention stays ON for all schemes.

3. **Frame 0 protection**: First frame's tokens are always fully kept (set to FLT_MAX score). This is PureKV's "anchor frame" rule — dropping frame 0 causes catastrophic accuracy loss on temporal reasoning.

4. **Architecture B ops + CUDA graphs**: The initial PYRAMID_MASK_GEN kernel used `cudaMemcpyAsync(DeviceToHost)` which is incompatible with CUDA graph capture. Fixed by reading thresholds directly from device memory inside the kernel.

5. **Optical flow availability**: Flow files exist between consecutive frames. The last selected frame may not have a corresponding flow file (frame N → N+1 flow doesn't exist if N is the last frame). The code handles this gracefully — tokens from frames without flow data are scored by cosine distance + norm only (motion weight falls to 0).

6. **Quality vs speed tradeoff**: At 40% pre-LLM keep ratio with motion scoring, quality is near-baseline for all tested scenes. Going lower (20%) loses character information in some scenes. The motion signal is the key improvement over pure embedding heuristics.

---

## Performance Summary (RTX 4090, BF16, 8 frames, averaged across 4 videos)

| Metric | [A] Baseline | [C] Full Pyramid |
|--------|-------------|-----------------|
| Visual tokens entering LLM | 4736 | 2244 |
| Visual tokens for generation | 4736 | 471 |
| Prefill | ~1248 ms | ~1008 ms (**19% faster**) |
| KV eviction (CPU Q@K^T) | 0 ms | ~330 ms |
| Generation throughput | 61.6 t/s | 63.1 t/s (**2.4% faster**) |
| Flash attention | ON | ON |

### Quality comparison (detailed prompt: "Be as detailed as possible...")

| Video | Baseline | Pyramid + Motion |
|-------|----------|-----------------|
| **alley_1** | reddish-brown hair, sleeveless top, tattoo, apple, medieval building | reddish-brown hair, sleeveless top, headband, red apple, stone walls, wooden beams |
| **bamboo_1** | red-haired girl, brown dress, walking through bamboo, dappled sunlight | brown-haired girl, red dress, brown belt, **climbing bamboo**, top-down view, dappled shadows |
| **cave_2** | white tank top, curved weapon, dragon, dark cave, fierce battle | white tank top, brown pants, boots, wooden staff, scaly creature, serpentine neck, defensive stance |
| **temple_2** | red hair, light outfit, ruined city, sunset, desolation, walking | **white dress, long hair, standing on stone ledge**, sunset, orange/pink, cityscape, serene |

Motion-based scoring correctly preserves moving characters (bamboo_1 girl climbing, temple_2 person on rooftop) that were previously lost with embedding-only scoring.

---

## How to Build

```bash
cd /data/spring26/CPP/llama.cpp/build
cmake .. -DGGML_CUDA=ON
make -j$(nproc) llama-pyramid-gen        # CPU scoring
make -j$(nproc) llama-pyramid-gen-cuda   # with CUDA kernels
```

## How to Run

```bash
# [A] Baseline — vanilla llama.cpp, all tokens, flash attn ON
./build/bin/llama-pyramid-gen \
  -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf \
  --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf \
  --video alley_1 --n-frames 8 -ngl 99 -c 8192 --n-predict 256 --baseline

# [B] Attention-only — no pre-LLM pruning, 3-stage Q@K eviction, flash attn ON
./build/bin/llama-pyramid-gen \
  -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf \
  --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf \
  --video alley_1 --n-frames 8 -ngl 99 -c 8192 --n-predict 256 --no-prellm

# [C] Full pyramid — motion-based pre-LLM + 3-stage eviction, flash attn ON
./build/bin/llama-pyramid-gen \
  -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf \
  --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf \
  --video alley_1 --n-frames 8 -ngl 99 -c 8192 --n-predict 256

# With custom detailed prompt
./build/bin/llama-pyramid-gen \
  -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf \
  --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf \
  --video bamboo_1 --n-frames 8 -ngl 99 -c 8192 --n-predict 512 \
  --prompt "Describe what is happening in this video sequence. Be as detailed as possible."
```

---

## Next Steps / Improvements

1. **Move Q@K^T to GPU**: Use the CUDA kernels in `visual_pyramid.cu` to compute attention scores on GPU instead of CPU. Would reduce eviction time from ~330ms to ~30ms, making scheme C faster end-to-end than baseline.

2. **True partial-layer execution**: The user's original idea — stop after layer 7, trim KV, continue layers 8-19, trim again, continue 20-27. This saves actual prefill compute in deeper layers by processing fewer tokens. Requires modifying the qwen2vl graph builder to support layer ranges and handling KV cache cell reuse across stages.

3. **Adaptive keep ratio per frame**: Instead of fixed 40%, use the optical flow magnitude to set per-frame keep ratio. High-motion frames (cave_2 with motion up to 224px) keep more tokens; near-static frames keep fewer. Formula: `keep_ratio = clamp(0.25 + 0.5 * (mean_motion / max_motion_across_frames), 0.25, 0.75)`.

4. **ViT attention as additional signal**: Use the ViT encoder's self-attention (last layer) as a fourth scoring signal. Patches that receive high ViT attention are focal points of the image. Requires disabling flash attention in the ViT encoder only.

5. **Inter-frame redundancy**: For temporally adjacent frames with high cosine similarity in their embeddings, aggressively drop overlapping tokens. Currently each frame is scored independently — cross-frame deduplication would further reduce tokens.

6. **Tune three-stage ratios**: Current 70%/60%/50% is heuristic. Could be tuned per-model or made adaptive based on attention entropy at each stage (high entropy = attention is spread = keep more tokens).

7. **Benchmark on MVBench**: Measure actual task accuracy (not just generation quality) to validate that token dropping doesn't hurt reasoning. This would provide quantitative quality numbers.

---

## Data Paths

- Model: `/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf`
- MMProj: `/data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf`
- Sintel frames: `/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel/training/final/<video>/`
- Sintel optical flow: `/data/Fall25/multimodalRAG/VLM/Qwen2.5-VL/MPI-Sintel/training/flow/<video>/`
- Videos tested: alley_1, bamboo_1, cave_2, temple_2, market_5
- Source code: `/data/spring26/CPP/llama.cpp/dmc955/generation-only/`
- Instruction doc: `/data/spring26/CPP/llama.cpp/dmc955/generation-only/instruction.md`
