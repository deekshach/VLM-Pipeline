# Jetson AGX Orin implementation

The video VLM pipeline (Qwen2.5-VL-7B BF16 through `libmtmd`) on a Jetson AGX Orin 64 GB, with every
optimization behind its own switch so that each one can be measured, and profiled, on its own. The
RTX 4090 study stays in `dmc955/`; nothing here changes its behaviour.

## Layout

| path | what |
|---|---|
| `jetson_vlm.cpp` | `llama-jetson-vlm`: the pipeline (one program: adaptive sampling, live OFA flow, window cache, engine switches) |
| `ofa_live_flow.h` | dense optical flow on the Orin's OFA engine through NVIDIA VPI, pipelined |
| `tools/gemm_tune.cpp` | `jetson-gemm-tune`: measures cuBLASLt algorithms per BF16 GEMM shape, writes the table |
| `tools/vit_shape_probe.cpp` | `jetson-vit-probe`: ViT alone, timing and embedding dumps per window-rectangle shape |
| `tools/nsys_stage_kernels.py` | GPU time of an nsys trace by stage (NVTX) and kernel class; compares traces |
| `tools/ncu_kernel_table.py` | per-kernel Nsight Compute summary (occupancy, caches, stalls) |
| `tools/analyze_switch_sweep.py` | summary of `scripts/run_switch_sweep.sh` |
| `scripts/` | switch sweep, `profile_nsys.sh`, `profile_ncu.sh` |
| `gemm/` | recorded GEMM shapes, the tuned table for this Orin, the tuning report |
| `experiments/` | every measurement of the Orin study (see below) |

Engine changes outside this directory are switches that default to off:

| file | switch |
|---|---|
| `tools/mtmd/models/qwen2vl.cpp`, `tools/mtmd/clip.cpp`, `tools/mtmd/clip-impl.h` | `MTMD_VIT_WINDOW_BATCH=1` |
| `ggml/src/ggml-cuda/jetson-gemm.cu(h)`, `ggml/src/ggml-cuda/ggml-cuda.cu` | `GGML_CUDA_JETSON_GEMM_F32OUT=1`, `GGML_CUDA_JETSON_GEMM_BIAS=1`, `GGML_CUDA_JETSON_GEMM_TABLE=<file>`, `GGML_CUDA_JETSON_GEMM_RECORD=<file>` |

## Build

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH CPATH=/usr/local/cuda-12.6/include
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-jetson-vlm jetson-gemm-tune jetson-vit-probe -j10
```

`LLAMA_BUILD_TESTS=OFF` because `include/llama.h` has a C++ default argument that breaks `tests/test-c.c`;
`CPATH` so C++ files of CUDA targets find `cuda_runtime.h`. `--live-flow` needs VPI (JetPack,
`/opt/nvidia/vpi3`); without it the program builds and the flag reports an error.

## Switches

All on `llama-jetson-vlm`; everything else is a standard llama.cpp option (`-m`, `--mmproj`, `-p`, `-n`, `-ngl`, ...).

| switch | what it changes | bottleneck it targets |
|---|---|---|
| `--baseline` | uniform frames instead of motion-guided sampling | (reference) |
| `--live-flow` / `--live-flow-serial` | optical flow computed on the OFA during the request (pipelined / one stage at a time) instead of read from `training/flow` | precomputed flow unavailable for real video |
| `--anchor-first` | frame 1 always selected and encoded while the flow is analysed | flow on the critical path |
| `--window-align` | frames resized to a multiple of the 112 px attention window (1024x436 -> 1008x448, 592 -> 576 tokens) | needed by the two below |
| `--window-cache` | re-encode only the windows whose pixels changed since they were last encoded; `--wc-threshold`, `--wc-halo`, `--wc-rekey-frac`, `--wc-cost-model` | ViT work on static content |
| `--vit-flash-attn on\|off` | ViT flash attention (`-fa` does not reach the ViT in this fork) | |
| `--vit-window-batch` | the 28 windowed ViT layers attend within each window as a batch of 64-patch sequences instead of dense attention under a block-diagonal mask (and the 21 MB mask is no longer built) | quadratic attention, 340 MB FP32 score matrix per layer without flash attention |
| `--gemm-f32-out` | BF16 GEMMs write F32 directly (cuBLASLt) instead of BF16 + a conversion kernel | elementwise passes over activations larger than L2 |
| `--gemm-bias-fuse` | the bias ADD after a BF16 GEMM goes into the GEMM epilogue (needs `--gemm-f32-out`) | elementwise passes |
| `--gemm-table <file>` | per-shape cuBLASLt algorithm measured on this Orin | GEMM tile choice (prefill down projection at 32% of peak) |
| `--gemm-record <file>` | append the BF16 GEMM shapes the run uses | input for `jetson-gemm-tune` |

The JSON output (`--output-json`) records every switch under `switches` and a TTFT breakdown under
`timing.breakdown` (analysis, ViT, image prefill, text prefill, window-cache actions per frame, live
flow stages, anchor).

### Tuning the GEMM table

```bash
GGML_CUDA_JETSON_GEMM_RECORD=shapes.txt build/bin/llama-jetson-vlm ... <the switches you will use>
build/bin/jetson-gemm-tune shapes.txt table.txt          # 32 candidates per shape, median of 10
build/bin/llama-jetson-vlm ... --gemm-table table.txt
```

Table entries are per exact shape and output configuration (F32 or BF16 output, bias or not); shapes
missing from the table fall back to the cuBLASLt heuristic. `gemm/table_orin.txt` was recorded on
`alley_1` and `ambush_2` with all GEMM configurations.

## Profiling which switch does what

```bash
# timeline + kernel classes per stage, no sudo; compare two configurations
bash jetson/scripts/profile_nsys.sh ref alley_1 --live-flow
bash jetson/scripts/profile_nsys.sh wb  alley_1 --live-flow --window-align --vit-window-batch
python3 jetson/tools/nsys_stage_kernels.py jetson/experiments/profiles/nsys/ref.sqlite jetson/experiments/profiles/nsys/wb.sqlite

# hardware counters for one stage (sudo; application replay, minutes per run)
sudo bash jetson/scripts/profile_ncu.sh wb vit alley_1 100 120 --live-flow --window-align --vit-window-batch
```

`profile_ncu.sh` uses application replay because the 15 GB model shares RAM with the GPU, and ncu's
default kernel replay runs out of memory backing it up. For the ViT alone,
`experiments/vit_shapes/run_ncu.sh` profiles `jetson-vit-probe` (vision tower only) with kernel replay.
Set `sudo jetson_clocks` before measuring; ncu runs trip the board's over-current throttling, normal
runs do not.

## Experiments

| directory | what |
|---|---|
| `experiments/baseline_f8.log` | first baseline run (dmc955 `llama-baseline-vlm`, 8 frames) |
| `experiments/overlap/` | ViT and prefill run concurrently on one GPU: no gain (0.99-1.02x) |
| `experiments/adaptive_sweep/` | adaptive vs uniform on 23 clips with Sintel ground-truth flow; `anchor/`: anchor-first; `ofa_flow/`: precomputed OFA flow vs ground truth |
| `experiments/flow_timing/` | optical-flow cost on CPU (OpenCV) vs OFA; OFA accuracy vs ground truth |
| `experiments/live_flow/` | live OFA flow in the pipeline: serial vs pipelined vs anchor-first, 23 clips |
| `experiments/vit_shapes/` | ViT cost per window-rectangle shape, flash attention on and off; kernel breakdown |
| `experiments/switch_sweep/` | each switch alone and all together (see below) |
| `experiments/ORIN_SESSION_HANDOFF.md` | setup notes and first results of the Orin study |

## Results

Jetson AGX Orin 64 GB, MAXN, clocks pinned (`jetson_clocks`), ViT flash attention off, 8 base frames,
TTFT excluding model load, mean over 6 Sintel clips (alley_1, sleeping_1, bandage_1: low motion;
temple_2: medium; ambush_5, cave_2: high). Full tables: `experiments/switch_sweep/summary.md`.

| configuration | mean TTFT | vs adaptive | ViT | image prefill | text prefill |
|---|---|---|---|---|---|
| uniform 8 frames, no optimizations | 14,060 ms | +57% | 8,260 | 4,853 | 745 |
| adaptive + live OFA flow (reference) | 8,973 ms | | 5,108 | 2,915 | 498 |
| + window align only | 8,736 ms | -3% | 4,899 | 2,857 | 493 |
| + window cache, threshold 24 | 8,569 ms | -4% | 4,704 | 2,872 | 496 |
| + window align + ViT window batch | 6,543 ms | -27% | 2,705 | 2,888 | 499 |
| + GEMM F32 output | 8,560 ms | -5% | 4,831 | 2,759 | 499 |
| + GEMM F32 output + bias fusion | 8,232 ms | -8% | 4,543 | 2,743 | 495 |
| + tuned GEMM table | 8,241 ms | -8% | 5,086 | 2,213 | 489 |
| + all of the above | 5,023 ms | -44% | 2,081 | 1,986 | 488 |
| + all of the above + anchor-first | 4,699 ms | -48% | 2,138 | 1,975 | 486 |

- **ViT window batch** halves the ViT without flash attention (dense 2304x2304 scores per head become
  36 windows of 64x64) and is exact. Per full frame, ViT alone: 966 -> 508 ms (flash attention off),
  489 -> 467 ms (on, where attention was already cheap).
- **GEMM F32 output + bias fusion** remove one conversion kernel and one bias-add kernel per BF16 GEMM
  (ViT and LLM): together -11% ViT time and -6% image prefill.
- **Tuned GEMM table** mostly speeds up image prefill (-24%): the LLM down projection
  (3584x18944, 512 tokens) goes from 6.7 to 2.5 ms per call (`gemm/tune_report.txt`).
- **Window cache** depends on its threshold (adaptive selection, `experiments/switch_sweep/wc_threshold/`):
  threshold 24 -> 8,569 ms (3 crops), 48 -> 7,298 ms (11 crops, 2 skips), 96 -> 5,816 ms (13 crops,
  9 skips). Higher thresholds reuse embeddings of windows that did change; the accuracy cost on the
  Orin is unmeasured.
- Live OFA flow costs about 0.4 s per request (pipelined); `experiments/live_flow/`.

## Caveats

- Generated text varies between identical runs in this build (same frames, same settings), so
  output-quality comparisons need several runs or a scored benchmark.
- `--vit-window-batch` gives bit-identical ViT embeddings (checked against the dense path, flash
  attention on and off). `--gemm-f32-out` changes a few numerically sensitive tokens (all GPU paths
  differ from a CPU F32-output reference by 3-6% relative L2); its effect on answers is unmeasured.
- OFA flow underestimates large motion, so adaptive sampling picks fewer frames on fast clips than
  with ground-truth flow (`experiments/adaptive_sweep/ofa_flow/comparison.md`).
- One run per configuration and clip; differences of a few hundred ms are within run-to-run noise.
