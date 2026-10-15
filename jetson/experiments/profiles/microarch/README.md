# Microarchitectural effect of the switches (Nsight Compute, 2026-10-15)

`sudo bash jetson/scripts/run_microarch_ncu.sh`, then `python3 jetson/tools/ncu_compare.py` (full tables in
`compare.md`). ViT flash attention off, kernel replay with ncu's defaults (clocks locked to base, caches flushed
before every kernel). ViT: `jetson-vit-probe`, alley_1 frame 1 aligned to 1008x448, a window of 120 launches
starting at launch 80 (about 3 windowed-attention layers). "DRAM read" is `lts__d_sectors_fill_sysmem.sum` x 32 B:
bytes L2 fetched from main memory (writes are not counted).

## 1. Batched-window attention (`MTMD_VIT_WINDOW_BATCH`), full frame, per windowed layer

| kernel class | time base -> batched | DRAM read | L2 hit | L2 requests | tensor pipe | main stall |
|---|---|---|---|---|---|---|
| softmax | 12.1 -> 0.36 ms | 680 -> 9.4 MB | 33% -> 50% | 1,019 -> 19 MB | 0 -> 1% | long scoreboard 39% -> 38% |
| attention matmul (KQ, KQV) | 4.6 -> 0.38 ms | 404 -> 45 MB | 88% -> 66% | 1,647 -> 66 MB | 36% -> 13% | barrier / MIO throttle -> long scoreboard 52% |
| whole layer | 30.9 -> 14.5 ms (-53%) | 1,871 -> 808 MB (-57%) | | 4,940 -> 2,263 MB | | |

- The dense path reads the 340 MB FP32 score matrix twice per layer in softmax (680 MB: once for the max, once for
  the sum) and streams it through the KQV product; batched, the scores of all 36 windows are 9.4 MB, read once.
- Each (head, window) block is 16 KB of scores and 20 KB per Q/K/V, so the working set no longer depends on frame
  size; L2 hit for softmax rises from 33% to 50%.
- The attention products become small (64x64x80) batched GEMMs: latency-bound (long-scoreboard stalls, 13% tensor),
  but they are now 0.4 ms per layer. The 4 global layers are still dense (not in this launch window).
- Everything else in the layer is unchanged, so after batching the layer is ~66% elementwise kernels.

## 2. Window-cache crop size (base engine), per windowed layer

| input | time | DRAM read | per window: time / DRAM | softmax L2 hit | softmax DRAM per window |
|---|---|---|---|---|---|
| 9x4 windows (2304 patches) | 30.9 ms | 1,871 MB | 0.86 ms / 52 MB | 33% | 18.9 MB |
| 5x2 (640) | 6.8 ms | 320 MB | 0.68 ms / 32 MB | 62% | 3.0 MB |
| 1x2 (128) | 1.3 ms | 73 MB | 0.65 ms / 37 MB | 59% | 1.7 MB |

- Dense attention traffic grows with the square of the crop: per window, softmax reads 18.9 MB at full frame but
  3 MB at 5x2, where each head's 1.6 MB score block fits in the 4 MB L2 (hit rate 33% -> 62%). So smaller crops cut
  DRAM traffic per window by ~40%, which is the working-set effect of encoding fewer windows.
- With batched windows the attention cost per window no longer depends on the crop, so the window cache then only
  saves in proportion to the windows it skips.
- Elementwise kernels keep 49-52% L2 hit at every crop size: ncu flushes the caches before each kernel, so these
  numbers show only reuse inside one kernel. Whether a 3.3 MB activation (5x2) stays in L2 from producer to
  consumer is invisible here; measure it with `NCU_CACHE=none` (see the script).
- Small crops underfill the GPU: at 1x2 the weight GEMMs reach 12% occupancy and 28% tensor-pipe use (58% at full frame).

## 3. GEMM F32 output + bias fusion (`GGML_CUDA_JETSON_GEMM_F32OUT` + `_BIAS`), full frame, per layer

Per-layer figures use the structural kernel counts (7 GEMMs; 14 casts and 7 adds before, 7 casts and 2 residual
adds after), because the fixed launch window covers a different number of each kernel type in the two runs.

| kernels | time | DRAM read |
|---|---|---|
| casts | 3.54 -> 1.59 ms (14 -> 7 launches) | 163 -> 103 MB |
| adds | 1.87 -> 0.41 ms (7 -> 2 launches) | 132 -> 47 MB |
| GEMMs (7) | 4.30 -> 4.88 ms | 320 -> 356 MB |
| net | -2.8 ms per layer (-9%) | -109 MB per layer (-6%), plus the writes those kernels no longer make |

- The removed BF16->F32 casts and bias adds were load-stall-bound streaming kernels (long scoreboard 70-88%).
- The GEMMs themselves get 13% slower and read 11% more: they now write F32 (twice the bytes) and run the cuBLASLt
  heuristic's kernels; the tuned table (section 4) recovers part of this in the pipeline.

## 4. Tuned GEMM algorithm: LLM prefill down projection (3584 x 18944, 512 tokens)

| | cuBLASLt heuristic | tuned |
|---|---|---|
| kernel | 128x128 tiles, split-K 9 | 256x128 tiles, split-K 2 |
| grid / waves per SM | 28x4x9 = 1,008 CTAs / 31.5 | 14x4x2 = 112 CTAs / 7 |
| time | 6.70 ms | 2.61 ms (-61%) |
| tensor pipe | 25% | 65% |
| L2 requests | 1,149 MB | 826 MB |
| DRAM read | 258 MB | 217 MB (weights alone are 136 MB) |
| L2 hit | 87% | 74% |
| main stalls | MIO throttle 52%, barrier 15% | MIO throttle 41%, wait 29% |

- Larger tiles reuse each loaded weight/activation element for more outputs (fewer L2 requests per FLOP), and 2-way
  instead of 9-way split-K removes most partial-sum traffic and the barrier stalls of the split reduction. DRAM
  traffic ends close to the compulsory weight read.

## Limits

- One frame, one launch window; ncu serialises kernels with clocks at base and cold caches, so absolute times differ
  from the pipeline (use the nsys comparison, `../nsys/compare_switches.md`, for real time).
- The 4 global-attention layers and the LLM prefill under the other switches were not in these windows.
