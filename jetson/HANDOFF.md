# Session handoff: VLM pipeline on Jetson AGX Orin (2026-10-13 to 2026-10-15)

Give this file to a new session to continue. It supersedes `experiments/ORIN_SESSION_HANDOFF.md`.
Full transcript of the session that produced it:
`/home/nvidia/.claude/projects/-home-nvidia-Desktop-Projects-VLM/5b030ded-dc0b-41a7-a96b-ff80f004e4f9.jsonl`.

## 1. Standing instructions from the user

- **All analysis is in terms of TTFT** (time to first token = image load + tokenize + ViT + prefill; decode excluded; model load excluded).
- **ViT flash attention OFF for experiments** (`--vit-flash-attn off`). In this fork `-fa off` does not reach the ViT:
  `tools/mtmd/mtmd.cpp:178` always passes AUTO to clip, and FA is only disabled when an eval callback is set.
  `llama-jetson-vlm --vit-flash-attn off` and `jetson-vit-probe ... off` install a no-op callback; check the log for
  "warmup: flash attention is disabled".
- **Every optimization must be an on/off switch**, so its microarchitectural effect can be measured on its own.
- **Jetson work lives in `VLM-Pipeline/jetson/`**, separate from the RTX 4090 study in `dmc955/`. Changes to shared
  engine files must be environment switches that default to off (4090 behaviour unchanged).
- The user can run sudo commands (`sudo jetson_clocks`, ncu); ask them to.
- `CLAUDE.md` points to upstream llama.cpp's `AGENTS.md` (AI agents should not write code for upstream contributions).
  This is the user's own research fork, at their request; this was disclosed.
- Persistent memory: `~/.claude/projects/-home-nvidia-Desktop-Projects-VLM/memory/` (flash attention off; jetson dir + switches).

## 2. Machine and paths

- Jetson AGX Orin Developer Kit, 64 GB unified RAM (61 GB visible), 12 CPU cores, 16 SMs, 4 MB L2, L4T R36.4.7, CUDA 12.6
  (`/usr/local/cuda-12.6`), VPI 3.2.4 (`/opt/nvidia/vpi3`), Nsight Systems 2024.5.4, Nsight Compute 2024.3.1.
  Power mode MAXN; run `sudo jetson_clocks` after every boot (GPU then pinned at 1300.5 MHz).
- No passwordless sudo; `/data` is root-only, so data lives in `/home/nvidia/data`.
- Data: `/home/nvidia/data/MPI-Sintel` (23 clips, `training/{clean,final,flow}`), `/home/nvidia/data/MSVD`
  (+ `MSVD_prepared`, 20 test clips, symlinks `training/final`, `training/flow`), `/home/nvidia/data/MPI-Sintel-ofa`
  (precomputed OFA flow, grid 2 / 4 levels, `training/clean` symlinked).
- Model: `/home/nvidia/data/models/Qwen2.5VL/{Qwen2.5-VL-7B-Instruct-BF16.gguf, mmproj-BF16.gguf}`; `dmc955/model_config.sh` sets `$MODEL`/`$MMPROJ`.
- Python env for MSVD prep: `/home/nvidia/data/envs/msvd-prep`. System `/usr/bin/python3` has VPI + OpenCV 4.5.4;
  the default `python3` is conda (matplotlib, numpy).
- Repos: `/home/nvidia/Desktop/Projects/VLM/VLM-Pipeline` (main; llama.cpp fork, branch `dmc955-vlm`, remote
  `https://github.com/deekshach/VLM-Pipeline.git`) and `/home/nvidia/Desktop/Projects/VLM-Profile/VLM-Profiling`
  (newer llama.cpp fork with the 4090 profiling study `dmc6955/` and its Jetson profiling framework `dmc6955/jetson/`).

## 3. Build (VLM-Pipeline)

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH CPATH=/usr/local/cuda-12.6/include
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-jetson-vlm jetson-gemm-tune jetson-vit-probe -j10
```
`LLAMA_BUILD_TESTS=OFF`: `include/llama.h:725` has a C++ default argument that breaks `tests/test-c.c`.
`CPATH`: C++ files of CUDA targets (`dmc955/kv_cache_rerotate.cuh`) need `cuda_runtime.h`.

## 4. Repository state

- Commit `d161e976` "jetson: Orin pipeline with switchable optimizations and engine switches" on `dmc955-vlm`,
  **not pushed** (no GitHub credentials on the board). The user pushes: `cd ~/Desktop/Projects/VLM/VLM-Pipeline && git push origin dmc955-vlm`.
- Work after that commit (microarch scripts, this file) may be uncommitted; check `git status`.
- `dmc955/` has **uncommitted local edits that must not be pushed**: the old-machine -> Jetson path rewrite in 49 files,
  and an older live-flow/anchor-first version of `dmc955/adaptive_sintel_vlm.cpp` + `dmc955/ofa_live_flow.h` +
  VPI lines in `dmc955/CMakeLists.txt`. `llama-jetson-vlm` supersedes them.
- `jetson/README.md` documents layout, switches, profiling workflow and results.

### Switches (`llama-jetson-vlm`)
| flag | engine env var | effect |
|---|---|---|
| `--baseline` | | uniform frames instead of adaptive sampling |
| `--live-flow` / `--live-flow-serial` | | optical flow on the OFA during the request (pipelined / staged) instead of `.flo` files |
| `--ofa-grid 2 --ofa-levels 4 --ofa-prep cpu --decode-threads 6` | | live-flow settings (defaults shown) |
| `--anchor-first` | | frame 1 always selected and encoded while flow is analysed |
| `--window-align` | | frames resized to multiples of 112 px (1024x436 -> 1008x448; 592 -> 576 tokens) |
| `--window-cache --wc-threshold T [--wc-halo --wc-rekey-frac --wc-cost-model]` | | re-encode only windows that changed since last encoded |
| `--vit-flash-attn on\|off` | (eval callback) | ViT flash attention |
| `--vit-window-batch` | `MTMD_VIT_WINDOW_BATCH=1` | windowed ViT layers as a batch of 64-patch windows (needs aligned frames) |
| `--gemm-f32-out` | `GGML_CUDA_JETSON_GEMM_F32OUT=1` | BF16 GEMMs via cuBLASLt write F32 directly |
| `--gemm-bias-fuse` | `GGML_CUDA_JETSON_GEMM_BIAS=1` | bias ADD folded into the GEMM epilogue (needs F32 out) |
| `--gemm-table F` | `GGML_CUDA_JETSON_GEMM_TABLE=F` | tuned per-shape cuBLASLt algorithm (`jetson/gemm/table_orin.txt`) |
| `--gemm-record F` | `GGML_CUDA_JETSON_GEMM_RECORD=F` | record GEMM shapes for `jetson-gemm-tune` |

Engine files: `tools/mtmd/models/qwen2vl.cpp`, `tools/mtmd/clip.cpp`, `tools/mtmd/clip-impl.h`,
`ggml/src/ggml-cuda/jetson-gemm.cu(h)`, `ggml/src/ggml-cuda/ggml-cuda.cu` (fusion hook + cuBLAS branch),
`ggml/src/ggml-cuda/CMakeLists.txt` (links cuBLASLt). NVTX ranges in the program: request, flow, select, anchor,
vit, prefill_img, prefill_txt, wc_diff, decode.

## 5. Findings, in order (all TTFT, Orin, MAXN)

1. **Baseline** (dmc955 `llama-baseline-vlm`, alley_1, 8 uniform frames, ViT FA on): ViT 4,848 ms (49.6%),
   image prefill 3,945 ms (40.4%), text 977 ms; TTFT ~9.8 s; decode 93 ms/token. ~6.6x slower than the 4090.
2. **ViT/prefill overlap on one GPU: no gain** (0.99-1.02x; each stage ~2x slower when concurrent).
3. **Adaptive sampling, Sintel ground-truth flow, 23 clips** (`experiments/adaptive_sweep/`): mean TTFT 10,320 -> 9,398 ms
   (-9%); low-motion clips -32..-48% (3-4 frames), high-motion +41..+90% (11-14 frames > 8 baseline).
4. **anchor-first** (frame 1 encoded during flow analysis): saved ~1.7 s when flow analysis cost 1.5 s; after the flow
   was made fast (below) it saves only ~0.2-0.3 s and changes the selection.
5. **Roofline + ncu with the VLM-Profile engine** (FA on, 2368 patches): ViT 19% of BF16 peak (4090: 50%), prefill 41%.
   ViT layer 20 ms = attention 38% / elementwise 39% / GEMMs 23%.
   - Attention: 184 regs/thread -> 2 blocks/SM -> 16% occupancy; 75% of stalls long-scoreboard (waiting on memory);
     tensor 9%, memory 21%, L2 hit 62%; K/V ~12 MB vs 4 MB L2 (4090: 72 MB L2 -> 65% of peak there).
   - Elementwise: activations 12 MB F32 > L2; L2 hit 33-53%; 64-89% long-scoreboard; 106-186 GB/s (near the 159-183
     GB/s measured ceiling). Per layer: 7 casts 1.5 ms, 5 bias adds 2.0 ms, 2 norms 1.0 ms, 2 K/V copies 0.9 ms.
   - RoPE: 28% achieved occupancy (100% possible), 21/32 active threads, 13% mem / 24% SM: inefficient on its own.
   - GEMMs: QKVO 87% of peak (fine); MLP cutlass 48-57% (mio/lg throttle); prefill down projection 32% of peak, split-K x9.
   - Measured ceilings: BF16 32.8 TFLOP/s, FP32 3.2, DRAM 159 (read) / 183 (copy) GB/s.
   - `--vit-window-batch` there (VLM-Profile's own implementation, FA on): ViT -26%, TTFT -14%.
6. **Over-current throttling** happens during ncu runs only (oc3 counter); normal runs: 0 events, GPU at 1300.5 MHz.
7. **This repo's engine differs** (VLM-Pipeline): with FA on, attention is only 11% of the ViT (it skips fully masked
   KV tiles), elementwise 61%, GEMM 28%. With **FA off**: full frame 975 ms vs 518 ms; attention 55% (softmax 39%,
   340 MB FP32 score matrix per layer), cost per window 15 ms (small crops) -> 27 ms (full frame), i.e. quadratic.
   Data: `experiments/vit_shapes/`.
8. **Window cache** (`dmc955/window_cache_vlm.cpp` policy): per-window mean |dRGB| vs the pixels each window was last
   encoded from (CPU, single thread), dirty bounding rectangle cropped and re-encoded, embeddings spliced into a host
   cache; clean windows never reach the GPU; LLM still prefills all tokens.
9. **Optical flow cost** (`experiments/flow_timing/`): Farneback CPU 137 ms/pair; DIS ultrafast 2.9 ms; OFA high grid 4
   ~12-18 ms. OFA grid 4 allows only 3 pyramid levels on 436-px frames (coarsest level >= 16 px); grid 2 + 4 levels
   handles large motion much better (ambush_2 per-frame corr 0.21 -> 0.85). Mean EPE vs GT 5.8 px (0.4-1.6 low motion,
   12-29 high). **OFA underestimates large motion -> fewer frames on fast clips** (14/23 same count; 77/104 frames
   identical, 100/104 within 2). `experiments/adaptive_sweep/ofa_flow/comparison.md`.
10. **Live OFA flow in the pipeline** (`experiments/live_flow/`, 23 clips, ViT FA off): serial 2,269 ms/clip vs pipelined
    322 ms; TTFT 10,548 (serial) -> 8,566 (pipelined) -> 8,391 (+anchor). Fixes needed: VPI buffer pool exhausted after
    ~40 frames (ring of 4 slots), host-memory wrappers stalled behind GPU work (copy into VPI images + CPU prep),
    full sort for p95 (nth_element: stats 316 -> 56 ms; GT file analysis 1,590 -> 330 ms).
11. **Switch sweep** (`experiments/switch_sweep/summary.md`; 6 clips; ViT FA off): reference (adaptive + live OFA)
    8,973 ms; + window batch 6,543 (-27%); + F32 out + bias 8,232 (-8%); + GEMM table 8,241 (-8%); all 5,023 (-44%);
    all + anchor 4,699 (-48%); uniform 8 frames 14,060. Window cache threshold 24 / 48 / 96 (adaptive): 8,569 / 7,298 /
    5,816 ms (3 / 11 / 13 crops) - accuracy cost of high thresholds unmeasured.
12. **Correctness**: window batch bit-identical (FA on and off). F32 out moves a few sensitive tokens (all GPU paths are
    3-6% rel. L2 from a CPU F32-output reference). Generation varies run to run even with identical settings.
13. **GEMM tuning** (`jetson/gemm/tune_report.txt`): 70 shapes; prefill down projection 3584x18944 (512 tokens)
    6.7 -> 2.5 ms (2.64x, no split-K); sum over shapes 1.33x.

## 6. Microarchitectural effects of the switches (nsys, alley_1, 3 frames, ViT FA off)

`experiments/profiles/nsys/compare_switches.md` (GPU kernel ms in the ViT / image prefill stages):

| ViT class | ref | window batch | F32 out + bias | GEMM table | all |
|---|---|---|---|---|---|
| softmax | 1192 | 159 | 1195 | 1258 | 159 |
| attention matmul (F32/TF32) | 481 | 86 | 477 | 496 | 86 |
| GEMM BF16 (+ split-K) | 302 (+132) | 287 (+129) | 331 (+134) | 425 (+3) | 428 (+0) |
| cast | 315 | 298 | 140 | 320 | 132 |
| add | 226 | 223 | 38 | 236 | 37 |
| ViT total | 3,050 | 1,568 | 2,715 | 3,150 | 1,226 |
| kernels launched | 3,903 | 3,987 | 2,547 | 3,903 | 2,631 |

Image prefill: split-K GEMM 1,097 -> 233 ms and non-split GEMM 188 -> 635 ms with the table (total 1,772 -> 1,356);
F32 out + bias: casts 253 -> 107, adds 35 -> 23; all: 1,170 ms.

Working sets vs the 4 MB L2 (computed):

| ViT input | activation F32 | FFN intermediate F32 | dense scores per layer (FA off) | batched-window scores |
|---|---|---|---|---|
| full frame (2304 patches) | 11.8 MB | 31.5 MB | 340 MB | 9.4 MB total, 16 KB per (head, window) |
| 5x2 crop (640) | 3.3 MB (fits) | 8.8 MB | 26 MB | 2.6 MB |
| 1x2 crop (128) | 0.66 MB | 1.8 MB | 1.0 MB (fits) | 0.5 MB |

- Window batch removes 87% of softmax and 82% of attention-matmul time; what remains is the 4 global layers (still
  dense, 340 MB scores each: 4/32 of 1,192 ms = 149 ms, matching the 159 ms left).
- F32 out + bias fusion remove 35% of kernel launches; ~4 B/element (cast) + ~8 B/element (bias add) of DRAM traffic
  per GEMM output, ~370 MB per full-frame ViT layer; GEMMs get slightly slower (+30 ms: F32 output is twice the bytes).
- GEMM table: the heuristic's split-K writes 9 partial F32 outputs (~66 MB, read back) for a 7.3 MB result; the tuned
  algorithm does not split. In the ViT the table removes split-K but is net neutral.
- **Not yet measured with counters**: L2 hit rates / DRAM bytes / stalls per switch. Script ready:
  `sudo bash jetson/scripts/run_microarch_ncu.sh`, then `python3 jetson/tools/ncu_compare.py jetson/experiments/profiles/microarch`.
  It collects `lts__t_sectors.sum` and `lts__d_sectors_fill_sysmem.sum` (DRAM read bytes = x32); these metric names
  were not verified (needs sudo) - if ncu rejects them, drop the `--metrics` part.

## 7. Open items / next steps

- Run the ncu microarch script (sudo) and interpret: expected L2 hit up / DRAM bytes down for batched windows and
  small crops; casts/adds gone with F32 out + bias; no split-K partials with the table.
- Push commit `d161e976` (+ later work) to GitHub.
- Window-cache threshold vs answer quality (needs repeated runs or a scored benchmark; generation is nondeterministic).
- Remaining ViT cost after all switches: the 4 global layers' dense attention (flash attention for those only?),
  elementwise passes (input casts F32->BF16 still there; norms, RoPE, K/V copies), RoPE kernel efficiency.
- OFA flow underestimates large motion: recalibrate the frame-count thresholds for OFA flow.
- GEMM table covers shapes recorded on alley_1 and ambush_2; record more clips for full coverage.
- MSVD has been prepared but not run through the new pipeline.
