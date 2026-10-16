# Optimizations: complete list (Jetson AGX Orin 64 GB)

Qwen2.5-VL-7B video VLM, BF16 weights throughout (no quantization). Every optimization is an off-by-default switch
of `llama-jetson-vlm` / the ggml-cuda engine, so each one's effect can be measured on its own.

Conventions:
- TTFT = image load + tokenize + ViT + prefill (model load and decode excluded).
- TTFT numbers are means over 6 Sintel clips (alley_1, ambush_5, bandage_1, cave_2, sleeping_1, temple_2) unless
  noted.
- ViT flash attention off; LLM flash attention AUTO (on).
- GPU at 1,300.5 MHz (MAXN + jetson_clocks).
- Microarchitectural numbers come from ncu (caches flushed before each kernel, two layers composed to a whole frame
  or ubatch) or from nsys kernel time; each is marked.

Sources:
- `experiments/switch_sweep/`: TTFT per switch.
- `experiments/profiles/ablation/`: Shapley attribution, nsys + ncu.
- `experiments/act_fusion/`: GEMM-input fusion and the GEMM configuration search.
- `experiments/live_flow/` and `experiments/adaptive_sweep/`.
- `experiments/overlap/`.
- `HANDOFF.md`.

## A. System level (what work reaches the GPU)

| # | Optimization | Switch | Stage | TTFT effect | Notes / microarch |
|---|---|---|---|---|---|
| A1 | Adaptive (motion-guided) frame sampling: the number of frames follows optical-flow motion instead of a fixed 8 | default (vs `--baseline`) | whole pipeline | 23 clips, ground-truth flow: 10,320 -> 9,398 ms (-9%); low-motion clips -32..-48% (3-4 frames), high-motion +41..+90% (11-14 frames) | Prefill is close to linear in frames. In the last 512-token ubatch only attention grows with the KV length (18 / 29 / 54 ms for 2 / 4 / 8 frames). Prefill attention keeps a 95-96% L2 hit rate: no L2 cliff. |
| A2 | Live optical flow on the OFA (NVIDIA VPI), pipelined with frame decode | `--live-flow` (`--live-flow-serial`: staged) | flow analysis | flow 2,269 -> 322 ms/clip (serial -> pipelined); TTFT 10,548 -> 8,566 ms (23 clips) | The OFA is a separate engine, so it overlaps without cost. Grid 2 + 4 pyramid levels; grid 4 allowed only 3 levels on 436-px frames. Mean EPE 5.8 px vs ground truth. Underestimates large motion, so fast clips get fewer frames. |
| A3 | Live-flow engineering | part of A2 | flow analysis | included in A2 | VPI buffer pool as a ring of 4 slots (the pool was exhausted after ~40 frames). Copies into VPI images plus CPU prep (host-memory wrappers had stalled behind GPU work). p95 via `nth_element` instead of a full sort: stats 316 -> 56 ms, ground-truth flow-file analysis 1,590 -> 330 ms. 6 decode threads. |
| A4 | Anchor-first: frame 1 always selected and encoded while the flow is analysed | `--anchor-first` | ViT + prefill overlapped with flow | with all other switches: 5,023 -> 4,699 ms (-6%) | Saved ~1.7 s when the flow cost 1.5 s; after A2, ~0.2-0.3 s. Changes the frame selection. |
| A5 | Window alignment: frames resized to multiples of the 112 px attention window (1024x436 -> 1008x448) | `--window-align` | ViT + prefill | 8,973 -> 8,736 ms (-3%) | 592 -> 576 tokens per frame; needed by A6 and B1. |
| A6 | Window cache: only windows whose pixels changed (mean abs dRGB per window) are re-encoded; cached embeddings are spliced in for the rest | `--window-cache --wc-threshold T` (`--wc-halo`, `--wc-rekey-frac`, `--wc-cost-model`) | ViT | threshold 24 / 48 / 96: 8,569 / 7,298 / 5,816 ms (3 / 11 / 13 crops) | Approximate: it reuses stale embeddings, and the accuracy at high thresholds is unmeasured. Largest ViT lever in the ablation (62% of the ViT saving, Shapley, full frame -> 5x2 crop). Without B1, each window of a crop costs 35% less DRAM because the working set fits the L2 (softmax L2 hit 39 -> 59%). In the real run, reuse between kernels is ~20% at 1x2 crops vs ~1% at full frame. The LLM still prefills all tokens. |

## B. ViT (vision encoder)

| # | Optimization | Switch | ViT effect | Microarch effect | Correctness |
|---|---|---|---|---|---|
| B1 | Batched-window attention: the 28 windowed layers attend per 64-patch window as a batch instead of dense attention under a block-diagonal mask (the 21 MB mask is no longer built) | `--vit-window-batch` | nsys: -455 ms/frame alone (30% Shapley share); TTFT 8,973 -> 6,543 ms (-27%) | Score matrix 340 MB/layer -> 16 KB per (head, window). DRAM -14 GB/frame (softmax -8.6 GB, attention matmul -5.4 GB). Softmax L2 hit 39 -> 50%. The attention matmul goes from barrier-bound (16% occupancy, 36% tensor) to latency-bound, 148 -> 29 ms/frame. The 4 global layers stay dense. No effect at 1x2 crops. | bit-identical |
| B2 | GEMM F32 output: cuBLASLt writes F32 directly instead of BF16 + a conversion kernel | `--gemm-f32-out` | -49 ms/frame alone (4%) | Casts -1.5 GB DRAM; GEMM +0.35 GB (writes F32). | Moves a few sensitive tokens; all GPU paths are 3-6% relative L2 from a CPU F32-output reference. |
| B3 | Bias fusion: the bias ADD goes into the GEMM epilogue | `--gemm-bias-fuse` (needs B2) | -30 ms/frame (3%) | Adds -2.0 GB DRAM. With B2: 3,903 -> 2,547 kernels launched; long-scoreboard stalls 39 -> 32%. | as B2 |
| B4 | Tuned GEMM table: per-shape cuBLASLt algorithm measured on this Orin | `--gemm-table` | -7 ms/frame (1%) | L2 requests -3.1 GB (less split-K partial-sum traffic). | Accumulation order changes. |
| B5 | GEMM-input fusion: RMS norm x weight and SwiGLU write the BF16 GEMM input directly; works under CUDA graph capture | `--act-fuse` (the same engine switch as C2) | 9x4 410 -> 382 ms (-7%); 5x2 113 -> 101 ms (-11%) | Casts 44 -> 6 ms (9x4). | bit-identical |

ViT cumulative, nsys:
- One full frame (9x4 windows): 979 ms (no switches) -> 412 ms (B1-B4) -> 382 ms (+B5).
- 5x2 crop: 208 -> 113 -> 101 ms.

## C. LLM prefill

| # | Optimization | Switch | Prefill effect (8 frames = 4,608 tokens, nsys) | Microarch effect | Correctness |
|---|---|---|---|---|---|
| C1 | Tuned GEMM table: per-shape algorithms, chiefly removing split-K x9 on the down projection | `--gemm-table` | 4,292 -> 3,164 ms alone (80% of the B2-B4 prefill saving); down projection 6.7 -> 2.5 ms (2.64x) | Split-K GEMM 2,863 -> 678 ms. Tensor pipe 50 -> 86%. L2 requests -33 GB per ubatch (-22%), DRAM unchanged: the prefill GEMMs are limited by L2 traffic and instruction issue (MIO throttle), not by DRAM. | Accumulation order changes; same top token. |
| C2 | GEMM-input fusion: the norm output is written once as BF16 and shared by Q/K/V and gate/up; SwiGLU writes BF16; no F32 intermediates | `--act-fuse` | 2,830 -> 2,552 ms (-9.8%) | Casts 278 -> 26 ms; SwiGLU 192 -> 169 ms (half the bytes written); norm -5 ms. Why: a 512-token activation is 7.3 MB F32 (39 MB for the FFN intermediate), larger than the 4 MB L2. Measured reuse between kernels was only 1-2%, so every hand-off was a DRAM round trip. | bit-identical (hash of the last token's logits) |
| C3 | GEMM F32 output (B2 applied to the LLM) | `--gemm-f32-out` | -265 ms (18%) | Casts -378 ms / -1.4 GB; split-K reads +2.4 GB (F32 partial sums). | as B2 |
| C4 | Bias fusion (B3 applied to the LLM; QKV biases) | `--gemm-bias-fuse` | -31 ms (2%) | Adds -0.27 GB. | as B2 |
| C5 | Full cuBLASLt configuration search (tuner mode 2): every algorithm x tile x stages x split-K x reduction x swizzle (900-2,400 per shape) | `--gemm-table` rows | K/V projection -12%, gate/up -2%; end to end -0.15% | The down projection stays at 63% of BF16 peak (gate/up: 82%). Its limit is the K = 18,944 reduction, not the last wave on 16 SMs. | Accumulation order changes. |
| C6 | Convert once: each F32 GEMM input converted to BF16 once and shared | `--act-dedup` | 2,820 -> 2,753 ms alone; nothing on top of C2 | Removes the repeated Q/K/V and gate/up conversions. | bit-identical |

Prefill cumulative, 8 frames (nsys): 4,292 ms -> 2,834 ms (C1 + C3 + C4) -> 2,552 ms (+C2), -41% in total.

## D. Engine infrastructure (enables or supports the above)

- D1. ViT flash-attention switch (`--vit-flash-attn on|off`). This fork's mtmd ignores the flash-attention setting
  for the ViT; it is disabled through an eval-callback hook.
- D2. cuBLASLt BF16 GEMM path (`ggml/src/ggml-cuda/jetson-gemm.cu`). It carries the F32-output, bias-epilogue,
  tuned-table and BF16-activation-cache switches.
- D3. Per-shape GEMM tuner (`jetson-gemm-tune`). `--gemm-record` collects the shapes; mode 0 times the heuristic
  picks, mode 1 adds swizzle variants, mode 2 searches the full configuration space.
- D4. Per-stream cuBLASLt workspace (bug fix). One workspace per device was shared, so split-K GEMMs running
  concurrently corrupted it or deadlocked. The serial pipeline was not affected.
- D5. Per-thread CUDA stream priority (`ggml_backend_cuda_set_thread_stream_priority`), used for the overlap study.
- D6. Probes:
  - `jetson-vit-probe`: per-crop ViT; embeddings dump.
  - `jetson-prefill-probe`: N image tokens, last-ubatch NVTX range, logits hash.
  - `jetson-overlap-probe`.
  - Ablation tooling: `run_ablation.sh`, `ablation_analyze.py` (Shapley attribution).

## E. Evaluated and not adopted (negative or neutral results)

| Idea | Result | Why |
|---|---|---|
| ViT / prefill overlap on one GPU (full frames; then crops + all switches + stream priorities) | 0.99-1.02x, then 1-2% | GPU work is conserved: the prefill GEMMs fill all 16 SMs at 63-82% of tensor peak. The prioritized stage runs at 1.07-1.18x its solo time and the other absorbs the rest. Pipelining only pays with a separate engine (the OFA, A2). |
| L2-aware GEMM tile order (CTA swizzle) | 0.2% | Prefill GEMMs read 1.4-1.5x their weights from DRAM (the 4 MB L2 drops weight panels between tile passes). They are compute-bound, so fewer bytes buy no time. |
| One ubatch per frame (`-ub 576`, with tuned 576-token shapes) | image prefill -2.8% (alley_1, uniform 8 frames) | Removes the 64-token tail ubatch per frame (~90-100 ms each). But 576 tokens = 4.5 tiles of 128, so the GEMMs are 30% slower for 12.5% more tokens. |
| Persisting L2 (access-policy window) | not implemented | The data worth pinning (the 3.7 MB BF16 activation) is about the size of the whole L2. |
| Quantization | excluded by design | BF16 only. |

## F. Cumulative TTFT (6 clips)

| Configuration | TTFT | vs uniform |
|---|---|---|
| Uniform 8 frames, no optimizations | 14,060 ms | - |
| Adaptive sampling + live OFA flow (A1 + A2) | 8,973 ms | -36% |
| + A5, A6 (threshold 24), B1, B2/C3, B3/C4, B4/C1 | 5,023 ms | -64% |
| + anchor-first (A4) | 4,699 ms | -67% |
| + GEMM-input fusion (B5/C2) | 4,407 ms | -69% |
| Uniform 8 frames + all switches | 7,234 ms | -49% |

Command (all switches):

    llama-jetson-vlm -m <llm.gguf> --mmproj <mmproj.gguf> --video <clip> --base-frames 8 -p "<prompt>" -ngl 99 \
        --vit-flash-attn off --live-flow --anchor-first --window-cache --wc-threshold 24 --vit-window-batch \
        --gemm-f32-out --gemm-bias-fuse --gemm-table jetson/gemm/table_orin.txt --act-fuse

## G. Known limitations / open items

- The accuracy of the window cache at higher thresholds is unmeasured; generation also varies from run to run with
  identical settings.
- OFA flow underestimates large motion, so fast clips get fewer frames; the frame-count thresholds need
  recalibration for OFA flow.
- The ViT's 4 global layers remain dense attention (flash attention off).
- Prefill: each frame still pays a 64-token tail ubatch. Packing frames into full 512-token ubatches would need the
  marker tokens between frames passed as embeddings.
- Not yet fused in prefill: the residual add + RMS norm, and the conversion of the O-projection input (the
  attention output).
- B5/C2 are evidenced by nsys kernel time; ncu DRAM-byte counts for them need a sudo run.
- `--act-fuse` assumes that all readers of a fused tensor are in the same graph split, which holds with the whole
  model on the GPU (the ViT is one split; the LLM is the CPU embedding lookup plus one GPU split).
- The RTX 4090 study (`dmc955/`) is separate and not included here.
