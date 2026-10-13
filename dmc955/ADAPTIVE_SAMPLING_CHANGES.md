# Adaptive Spatio-Temporal VLM Optimizer for MPI-Sintel

## Overview

A C++ application built on top of llama.cpp's multimodal API (`libmtmd`) that uses optical flow motion vectors to **adaptively sample video frames** based on their temporal and spatial complexity, then runs vision-language model (VLM) inference using Qwen2.5-VL. Instead of uniformly sampling a fixed number of frames (baseline), the adaptive algorithm analyzes motion patterns to select fewer frames for static scenes and more frames for dynamic ones, reducing processing time while maintaining output quality.

## Architecture

```
MPI-Sintel Dataset (.flo + .png)
        |
        v
[Step 1] Motion Analysis         -- read .flo files, compute per-frame motion stats
        |
        v
[Step 2] Adaptive Frame Selection -- step function + weighted sampling
        |
        v
[Step 3] Post-Selection Filtering -- drop near-static frames (complexity < 0.05)
        |
        v
[Step 4] Spatial Patch Analysis   -- 4x4 grid analysis per selected frame
        |
        v
[Step 5] Spatial-Aware Skipping   -- drop frames with <25% active patches
        |
        v
[Step 6] VLM Inference            -- Qwen2.5-VL via llama.cpp mtmd API
        |
        v
[Step 7] Metrics & JSON Output    -- entropy, motion coverage, timing breakdown
```

## Files Created/Modified

| File | Description |
|------|-------------|
| `dmc955/adaptive_sintel_vlm.cpp` | Main C++ application (~1250 lines) |
| `dmc955/CMakeLists.txt` | Build configuration for the target |
| `dmc955/run_all_experiments.sh` | Batch script to run all 23 MPI-Sintel videos |
| `dmc955/compare_results.py` | Python script to compare adaptive vs baseline results |
| `CMakeLists.txt` (root) | Added `add_subdirectory(dmc955)` for build integration |

---

## Algorithm Details

### Step 1: Motion Analysis (`build_motion_profile`)

Reads MPI-Sintel `.flo` optical flow files (magic `202021.25f`, int32 width/height, then h*w interleaved float32 u,v pairs). For each frame, computes:

- **mean_magnitude**: Average pixel-wise optical flow magnitude
- **p95_magnitude**: 95th percentile of flow magnitudes
- **max_magnitude**: Maximum flow magnitude
- **variance**: Variance of flow magnitudes
- **high_motion_ratio**: Fraction of pixels exceeding `motion_threshold` (default: 2.0)

**Global complexity** is computed from p95 values across all frames using absolute normalization constants (matching the original Python implementation):

```
mean_norm  = min(mean(p95_values) / 100.0, 1.0)
var_norm   = min(var(p95_values) / 5000.0, 1.0)
range_norm = min((max(p95) - min(p95)) / 150.0, 1.0)
high_ratio = fraction of p95 values above the 75th percentile

global_complexity = 0.4 * mean_norm + 0.3 * var_norm + 0.2 * range_norm + 0.1 * high_ratio
```

Per-frame complexity uses min-max normalization (for JSON diagnostics only):
```
per_frame_complexity = 0.4 * mean_norm + 0.3 * var_norm + 0.2 * range_norm + 0.1 * high_motion_ratio
```

### Step 2: Adaptive Frame Selection (`select_frames`)

**Frame count determination** via aggressive step function based on global complexity:

| Global Complexity | Frame Count |
|-------------------|-------------|
| > 0.8 | min(base + 6, max_frames) |
| > 0.6 | base + 3 |
| > 0.4 | base |
| > 0.2 | max(base - 4, min_frames) |
| <= 0.2 | max(base - 8, min_frames) |

With defaults `base=8, min=3, max=24` (as used in experiments), this gives:

| Complexity Range | Frames Selected |
|------------------|-----------------|
| > 0.8 | 14 |
| 0.6 - 0.8 | 11 |
| 0.4 - 0.6 | 8 |
| 0.2 - 0.4 | 4 |
| <= 0.2 | 3 (minimum) |

**Importance weight thresholding**: Frames are weighted by their p95 magnitude. Frames below the 70th percentile of p95 values receive a reduced weight of 0.1; frames above keep their full p95 value as weight.

**Weighted sampling without replacement**: Uses `std::discrete_distribution` with deterministic seed (42). No first/last frame anchoring. Results sorted in temporal order.

### Step 3: Post-Selection Complexity Filtering

After probabilistic selection, removes frames with per-frame `complexity_score < 0.05` (near-static frames that slipped through). Always keeps at least `min_frames` by retaining the highest-complexity frames if filtering is too aggressive.

### Step 4: Spatial Patch Analysis (`analyze_spatial_patches`)

Divides each selected frame's optical flow into a 4x4 grid (16 patches). Each patch is evaluated with **3-criteria encoding decision**:

```
selected = (motion_ratio > spatial_threshold)    // fraction of high-motion pixels > 0.05
        OR (mean_motion > motion_threshold)       // mean magnitude > 2.0
        OR (max_motion > motion_threshold * 2.0)  // max magnitude > 4.0
```

Guarantees minimum 25% patches selected per frame (top patches by mean motion if threshold too aggressive).

### Step 5: Spatial-Aware Frame Skipping

Drops entire frames where less than 25% of spatial patches are active (mostly static frames). Always keeps at least `min_frames`. Rebuilds spatial analysis array for remaining frames.

### Baseline Mode

When `--baseline` flag is set, uses uniform frame selection (evenly spaced indices) with no motion analysis for frame selection. Still builds the motion profile for comparison metrics (motion coverage).

---

## Changes from Initial C++ Implementation

The initial C++ implementation had several algorithm differences from the original Python code. Here is every change made:

### Change 1: Complexity Normalization (Absolute vs Min-Max)

**Before (C++ v1):** Used min-max normalization within the video for global complexity. This inflated complexity scores (~0.3) because normalization was relative.

**After (Python-matched):** Uses absolute divisor constants on p95 values:
- `mean(p95) / 100`
- `var(p95) / 5000`
- `range(p95) / 150`

This produces much lower scores for MPI-Sintel data (~0.05-0.15 for calm videos), enabling the step function to correctly reduce frames.

### Change 2: Frame Count Formula (Linear vs Step Function)

**Before (C++ v1):** Linear interpolation `base + round(complexity * (max - base))`. This could **never go below base**, always selecting >= 8 frames.

**After (Python-matched):** Step function with discrete thresholds. Can reduce to `base - 4` or `base - 8` for low-complexity content, selecting as few as 3 frames.

### Change 3: Aggressive Step Function

**Before (Python-matched v1):** Step function used `-3/-6` reductions (base-3 and base-6).

**After (aggressive optimization):** Step function uses `-4/-8` reductions for more aggressive frame savings on low-complexity content.

### Change 4: Importance Weight Thresholding

**Before (C++ v1):** No thresholding. All frames had equal chance weighted by raw p95 magnitude.

**After (Python-matched):** Frames below the 70th percentile of p95 values receive weight 0.1 (near-zero), concentrating selection probability on high-motion frames.

### Change 5: First/Last Frame Anchoring Removed

**Before (C++ v1):** Always included first and last frames, potentially adding 2 extra frames beyond the step function count.

**After (Python-matched):** No forced anchoring. Frame selection is purely probabilistic based on motion weights.

### Change 6: Spatial Threshold Updated

**Before (C++ v1):** `spatial_threshold = 0.3` (30% of pixels must have motion).

**After (Python-matched):** `spatial_threshold = 0.05` (5%), matching the Python `patch_motion_ratio` parameter.

### Change 7: Default Parameter Updates

| Parameter | Before (v1) | After (final) |
|-----------|-------------|---------------|
| `min_frames` | 3 (was 6, then lowered) | 3 |
| `max_frames` | 15 | 24 |
| `base_frames` | 8 | 16 (but experiments use `--base-frames 8`) |
| `spatial_threshold` | 0.3 | 0.05 |
| `selection_percentile` | (not present) | 70.0 |

### Change 8: Post-Selection Complexity Filtering (New)

**Added:** After weighted sampling, remove frames with per-frame complexity < 0.05. This catches near-static frames that slipped through probabilistic sampling (e.g., static background frames in otherwise dynamic videos).

### Change 9: Spatial-Aware Frame Skipping (New)

**Added:** After spatial patch analysis, drop frames where fewer than 25% of patches are active. This provides a second pass of frame reduction informed by the spatial structure of motion.

### Change 10: Baseline Mode Added

**Added:** `--baseline` flag for uniform (evenly-spaced) frame sampling to enable fair A/B comparison. Baseline still computes motion profile for metric calculation (motion coverage).

---

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--sintel-dir <dir>` | `/data/Fall25/.../MPI-Sintel` | MPI-Sintel dataset root |
| `--video <name>` | `alley_1` | Video sequence name |
| `--min-frames <n>` | `3` | Minimum frames to select |
| `--max-frames <n>` | `24` | Maximum frames to select |
| `--base-frames <n>` | `16` | Base frame count for step function |
| `--motion-threshold <f>` | `2.0` | Motion magnitude threshold (pixels) |
| `--motion-percentile <f>` | `0.95` | Percentile for per-frame motion stats |
| `--selection-percentile <f>` | `70.0` | Percentile for importance weight thresholding |
| `--patch-grid <n>` | `4` | Spatial grid size (NxN) |
| `--spatial-threshold <f>` | `0.05` | Patch motion ratio threshold |
| `--output-json <path>` | auto-generated | Output JSON file path |
| `--baseline` | `false` | Use uniform frame sampling |

Standard llama.cpp flags (`-m`, `--mmproj`, `-p`, `-n`, `-ngl`, `-c`, etc.) are also supported.

---

## Build Instructions

```bash
# From llama.cpp root directory
cmake -B build
cmake --build build --target llama-adaptive-vlm

# Binary output
./build/bin/llama-adaptive-vlm
```

The `dmc955/CMakeLists.txt` links against `common`, `mtmd`, and `Threads::Threads`, following the pattern from `tools/mtmd/CMakeLists.txt`.

---

## Usage Examples

### Single Video (Adaptive)
```bash
./build/bin/llama-adaptive-vlm \
    -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-3B-Instruct-q8_0.gguf \
    --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
    --video alley_1 --base-frames 8 \
    -p "Describe the scene, motion patterns, and key visual elements in these video frames." \
    -n 150 -ngl 99 --output-json results/alley_1_adaptive.json
```

### Single Video (Baseline)
```bash
./build/bin/llama-adaptive-vlm \
    -m /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-3B-Instruct-q8_0.gguf \
    --mmproj /data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf \
    --video alley_1 --base-frames 8 --baseline \
    -p "Describe the scene, motion patterns, and key visual elements in these video frames." \
    -n 150 -ngl 99 --output-json results/alley_1_baseline.json
```

### All 23 Videos (Batch)
```bash
cd /data/spring26/CPP/llama.cpp/dmc955
./run_all_experiments.sh       # runs baseline + adaptive for all videos
python3 compare_results.py     # generates comparison tables and summary JSON
```

---

## Experiment Results (23 MPI-Sintel Videos)

### Summary Statistics

| Metric | Baseline | Adaptive | Change |
|--------|----------|----------|--------|
| **Avg frames** | 8.0 | 6.1 | **-23.9%** |
| **Avg encode time** | ~738 ms | ~575 ms | **1.28x speedup** |
| **Avg total time** | ~3277 ms | ~3061 ms | **1.07x speedup** |
| **Avg Shannon entropy** | 4.245 bits | 4.238 bits | -0.007 bits (negligible) |
| **Avg motion coverage** | 17.1% | 27.7% | **1.6x better** |

### Per-Video Frame Selection

| Video | Complexity | Baseline Frames | Adaptive Frames | Reduction |
|-------|-----------|-----------------|-----------------|-----------|
| alley_1 | 0.079 | 8 | 3 | 62.5% |
| alley_2 | 0.082 | 8 | 3 | 62.5% |
| ambush_2 | 0.852 | 8 | 14 | -75.0% (more frames) |
| ambush_4 | 0.785 | 8 | 11 | -37.5% |
| ambush_5 | 0.667 | 8 | 11 | -37.5% |
| ambush_6 | 0.776 | 8 | 11 | -37.5% |
| ambush_7 | 0.249 | 8 | 4 | 50.0% |
| bamboo_1 | 0.055 | 8 | 3 | 62.5% |
| bamboo_2 | 0.139 | 8 | 3 | 62.5% |
| bandage_1 | 0.108 | 8 | 3 | 62.5% |
| bandage_2 | 0.092 | 8 | 3 | 62.5% |
| cave_2 | 0.651 | 8 | 11 | -37.5% |
| cave_4 | 0.383 | 8 | 4 | 50.0% |
| market_2 | 0.142 | 8 | 3 | 62.5% |
| market_5 | 0.751 | 8 | 11 | -37.5% |
| market_6 | 0.432 | 8 | 8 | 0.0% |
| mountain_1 | 0.127 | 8 | 3 | 62.5% |
| shaman_2 | 0.053 | 8 | 3 | 62.5% |
| shaman_3 | 0.056 | 8 | 3 | 62.5% |
| sleeping_1 | 0.050 | 8 | 3 | 62.5% |
| sleeping_2 | 0.049 | 8 | 3 | 62.5% |
| temple_2 | 0.424 | 8 | 8 | 0.0% |
| temple_3 | 0.649 | 8 | 11 | -37.5% |

### Key Observations

1. **Low-complexity videos** (complexity < 0.2): Reduced to 3 frames (62.5% reduction, ~2.4x encode speedup). These are slow-moving scenes like `sleeping_1`, `shaman_2`, `bamboo_1`.

2. **Medium-complexity videos** (0.2-0.6): Reduced to 4-8 frames. `cave_4` (0.38) gets 4 frames, `market_6` (0.43) keeps 8.

3. **High-complexity videos** (>0.6): Increased to 11-14 frames. Action scenes like `ambush_2` (0.85) get 14 frames for better motion coverage.

4. **Quality preservation**: Shannon entropy difference is only -0.007 bits on average, meaning generated text quality is virtually identical.

5. **Motion coverage**: Adaptive achieves 27.7% vs 17.1% baseline (1.6x improvement) because it concentrates frames on high-motion segments rather than spreading them uniformly.

---

## JSON Output Format

Each run produces a JSON file with:

```json
{
  "video_name": "alley_1",
  "mode": "adaptive",
  "n_total_frames": 50,
  "n_selected_frames": 3,
  "global_complexity": 0.079,
  "selected_frames": [
    {"frame_index": 5, "motion_weight": 1.23, "complexity_score": 0.42}
  ],
  "motion_profile": [
    {"frame": 1, "mean_magnitude": 0.5, "p95_magnitude": 1.2, ...}
  ],
  "spatial_analysis": [
    {"frame_index": 5, "n_selected_patches": 12, "n_total_patches": 16, ...}
  ],
  "inference": {
    "generated_text": "...",
    "n_tokens": 150,
    "encode_time_ms": 305.0,
    "generation_time_ms": 2100.0
  },
  "metrics": {
    "motion_coverage": 0.115,
    "shannon_entropy": 4.186,
    "temporal_reduction": 0.94,
    "avg_spatial_savings": 0.25
  },
  "timing": {
    "flow_analysis_ms": 450.0,
    "frame_selection_ms": 0.1,
    "encode_ms": 305.0,
    "generation_ms": 2100.0,
    "total_ms": 2820.0
  },
  "params": { ... }
}
```

---

## Model Configuration

| Model | Path | Notes |
|-------|------|-------|
| Qwen2.5-VL 3B (q8_0) | `/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-3B-Instruct-q8_0.gguf` | Used for batch experiments |
| Qwen2.5-VL 3B mmproj | `/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf` | Vision encoder |
| Qwen2.5-VL 7B (BF16) | `/data/spring26/CPP/modelsRepo/Qwen2.5VL/Qwen2.5-VL-7B-Instruct-BF16.gguf` | Higher quality, slower |
| Qwen2.5-VL 7B mmproj | `/data/spring26/CPP/modelsRepo/Qwen2.5VL/mmproj-BF16.gguf` | Vision encoder for 7B |

---

## Inference Pipeline

The inference implementation mirrors `tools/mtmd/mtmd-cli.cpp`:

1. Load selected frame PNGs as bitmaps via `mtmd_helper_bitmap_init_from_file`
2. Build prompt with `<__media__>` markers (one per image) via `mtmd_default_marker()`
3. Format as chat message via `common_chat_format_single()`
4. Tokenize with vision context via `mtmd_tokenize()`
5. Evaluate chunks via `mtmd_helper_eval_chunks()`
6. Auto-regressive token generation loop with `common_sampler`
