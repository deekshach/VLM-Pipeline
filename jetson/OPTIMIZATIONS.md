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

## H. Why each optimization: the microarchitectural rationale on Jetson

### H.0 The platform constraints that drive every decision

| Resource | Orin (this board) | Why it matters here |
|---|---|---|
| GPU | Ampere sm_87, 16 SMs, 64 tensor cores, locked at 1,300.5 MHz | Few SMs: a grid of a few dozen tiles already fills the GPU, and partial last waves and split-K choices matter. |
| BF16 tensor throughput | Theoretical dense: 64 TC x 512 FLOP/clk x 1.3 GHz = 42.6 TFLOP/s. An early microbenchmark measured 32.8, but the tuned prefill gate/up GEMM sustains 35.5 (= 2 x 512 x 3584 x 18944 FLOP in 1.956 ms). | Percent-of-peak figures in this file use 42.6. |
| L2 | 4 MB, shared by all SMs (RTX 4090: 72 MB) | Smaller than almost every activation tensor. This is the single most important constraint (H.1). |
| DRAM | 256-bit LPDDR5: 204.8 GB/s theoretical; measured 159 GB/s read, 183 GB/s copy. Unified: CPU and GPU share it. | Every pass over an activation that misses L2 costs DRAM time, and CPU-side work competes for the same bandwidth. |
| Ridge point | 35.5 TFLOP/s / 159 GB/s = ~220 FLOP per DRAM byte | Below it a kernel is memory-bound (H.1). |
| Other engines | OFA (optical-flow accelerator), 12 Arm Cortex-A78AE CPU cores | Work on these engines runs beside the SMs without using them. |

### H.1 The resulting bound classes

1. **Elementwise kernels.** Casts, bias adds, norms, RoPE, SwiGLU and copies do 0.25-1 FLOP per byte, so they are
   purely DRAM-bound.
   - Measured on the baseline ViT: 106-186 GB/s, i.e. at the 159-183 GB/s ceiling.
   - 64-89% of warp stalls are long-scoreboard (waiting on memory).
   - L2 hit 33-53%. A 33% hit rate means only the writes hit; every read misses.
2. **Prefill weight GEMMs at 512 tokens are compute-bound.** Each BF16 weight byte is reused 512 times, which is
   2 x 512 / 2 = 512 FLOP per byte, above the ridge point.
3. **The same GEMMs at 64 tokens are memory-bound.** 64 FLOP per byte, below the ridge point: the whole 13 GB of
   weights is streamed for 64 tokens.
4. **Producer-consumer reuse through L2 almost never happens at full size.** Working sets that matter:

   | Tensor | Size | vs the 4 MB L2 |
   |---|---|---|
   | ViT full frame (2,304 patches) activation | 11.8 MB F32 | 3x |
   | ViT full frame FFN intermediate | 31.5 MB | 8x |
   | ViT full frame dense scores (flash attention off) | 340 MB per layer | 85x |
   | LLM 512-token activation | 7.3 MB F32 | 1.8x |
   | LLM 512-token FFN intermediate | 39 MB F32 | 10x |

   Measured warm vs flushed caches:
   - Reuse between kernels is 1-3% of DRAM bytes (full frame and LLM).
   - At a 1x2 crop the working sets fit, and reuse rises to ~20%.

   So the only ways to cut elementwise cost on this chip are:
   - remove the pass;
   - write fewer bytes per pass;
   - make the working set smaller than 4 MB.

   Keeping data in L2 longer is not one of them.

### H.2 Per optimization

**A1 Adaptive frame sampling.**
- At baseline, TTFT is 90% per-frame work: ViT 49.6% + image prefill 40.4% (~1.2 s per frame on Orin, ~6.6x the
  RTX 4090).
- Both stages are linear in frames, and neither has a cache effect that fewer frames would unlock:
  - Prefill GEMMs are compute-bound at a fixed cost per 512-token ubatch; every class except attention is constant
    across 2 / 4 / 8 frames.
  - Prefill flash attention keeps a 95-96% L2 hit rate even when the KV per layer (2.4 -> 9.4 MB) outgrows the
    4 MB L2. It streams K/V tiles, so there is no cliff to stay under.
- Frame count is therefore the only lever that scales both dominant stages at once, and it costs only flow
  analysis.

**A2 Live flow on the OFA.**
- Real video has no precomputed flow, and computing it on the CPU or SMs takes compute from the pipeline:
  - CPU Farneback: 137 ms per pair, x 49 pairs.
  - DIS (2.9 ms per pair): cheaper but less accurate.
- Orin has a dedicated optical-flow engine that does not use SMs, CPU cores or (beyond its frames) much DRAM
  bandwidth. Running flow there and pipelining it with CPU frame decode (6 threads) puts the three engines (CPU,
  OFA, SMs) on separate parts of the work.
- Grid 2 with 4 pyramid levels: at grid 4 the 436-px frames allow only 3 levels (the coarsest level must stay at
  least 16 px), which underestimates large motion.

**A3 Live-flow engineering.**
- **Host-memory wrappers → copies into VPI images.** On a unified-memory SoC, VPI images wrapping host memory were
  synchronized against the busy GPU stream, so CPU-side flow work queued behind GPU kernels. Copying into
  VPI-owned images decouples them.
- **Fixed ring of 4 slots.** VPI's buffer pool was exhausted after about 40 frames.
- **`nth_element` for p95.** Statistics run on the A78AE cores on the critical path, so the O(n log n) full sort
  (316 ms) became O(n) (56 ms).

**A4 Anchor-first.**
- During flow analysis the SMs are idle: the work is on the CPU and OFA.
- Frame 1 is always selected, so encoding it then fills an idle engine. This is overlap across engines, the kind
  that works on this chip (contrast E, overlap on the SMs).

**A5 Window alignment.**
- The Qwen2.5-VL ViT attends within 112 px windows: 8 x 8 patches of 14 px, i.e. 4 x 4 merged tokens of 28 px.
- At 1024 x 436 the frame edges produce partial windows, which have to be padded and masked. Resizing to 1008 x 448
  makes every window complete and identical (64 patches). Without that:
  - window batching (B1) could not form one batch of equal-size windows;
  - the cache (A6) could not crop on window boundaries.
- Tokens per frame also drop 592 -> 576 (-2.7%), which shortens the LLM prefill.

**A6 Window cache.**
- With ViT flash attention off, the cost per encoded window grows with frame size (15 ms for small crops, 27 ms
  for a full frame), because dense attention is quadratic in the number of patches.
- In video most windows do not change between selected frames. Re-encoding only the dirty rectangle cuts work
  linearly and attention work super-linearly.
- It also shrinks the working set below the L2 size:
  - A 5x2 crop's activation is 3.3 MB; a 1x2 crop's dense scores are 1.0 MB.
  - Without B1, each encoded window then costs 35% less DRAM (softmax L2 hit 39 -> 59%).
  - In the real run, reuse between kernels reaches ~20% at 1x2, vs ~1% for a full frame.
- The dirty test is a cheap single-thread CPU diff (~0.2-0.4 ms per frame); clean windows never touch the GPU.

**B1 Batched-window attention.**
- With flash attention off, every one of the 32 layers built the full dense score matrix: 2,304 x 2,304 x 16
  heads x 4 B = 340 MB per layer, 85x the L2. Softmax alone was 39% of the ViT and attention 55%.
- 28 of the 32 layers only need the block-diagonal part (64-patch windows), so 35/36 of those scores were computed
  only to be masked away.
- Running the windowed layers as a batch of 64 x 64 problems makes each (head, window) score tile 16 KB, which
  stays on chip. Measured effects:
  - DRAM -14 GB per frame;
  - softmax L2 hit 39 -> 50%;
  - the attention matmul went from barrier-bound at 16% occupancy (36% tensor) to 29 instead of 148 ms per frame;
  - the 21 MB mask is no longer built.
- The 4 global layers stay dense by design and are now most of the remaining attention time.

**B2 / C3 GEMM F32 output.**
- ggml keeps activations in F32, so the stock path ran cuBLAS with BF16 output and then a separate BF16 -> F32
  conversion kernel.
- That kernel is pure DRAM traffic over a tensor larger than L2. At baseline, the ViT's 7 casts took 1.5 ms per
  layer, and in the LLM, casts were 79 ms per 512-token ubatch.
- cuBLASLt can write F32 straight from the F32 accumulators in its epilogue. One full read + write pass per GEMM
  disappears, and precision rises.
- The GEMM writes twice the output bytes (+0.35 GB per ViT frame), much less than the pass it removes.
- In the prefill, split-K partial sums also become F32 (+2.4 GB of reads); time still drops 265 ms. The F32
  partial sums are part of what motivated C1.

**B3 / C4 Bias fusion.**
- Each bias add is a separate kernel that reads and writes a full activation (larger than L2) for one add per
  element. The ViT had 5 per layer (2.0 ms per layer at baseline).
- The cuBLASLt bias epilogue adds the bias while the accumulators are still in registers, removing the pass and a
  kernel launch:
  - kernels launched per run fell 3,903 -> 2,547 with B2;
  - long-scoreboard stalls in the ViT fell 39 -> 32%.

**B4 / C1 Tuned GEMM table.**
- cuBLASLt's heuristic tends to pick split-K for narrow outputs, which helps big GPUs fill many SMs. On Orin that
  is the wrong trade.
- The prefill down projection (3,584 x 512 output, K = 18,944) ran split-K x9:
  - partial sums go through workspace memory and are reduced by a second pass;
  - the GEMMs were bound by L2 traffic and instruction issue (MIO throttle 33%, tensor pipe 50%), at 32% of peak.
- Measuring every offered algorithm per shape on the board itself picked tilings with little or no split-K.
  Effects:
  - tensor pipe 50 -> 86%;
  - L2 requests -33 GB per ubatch (-22%);
  - DRAM unchanged, confirming the limit was on-chip traffic, not DRAM;
  - down projection 6.7 -> 2.5 ms.
- In the ViT the same change mostly cuts L2 requests (-3.1 GB per frame).

**B5 / C2 GEMM-input fusion (`--act-fuse`).**
- After B2-B4, the largest remaining elementwise class was the F32 -> BF16 conversion of every GEMM input (cuBLAS
  needs BF16 operands; ggml hands it F32). In the LLM:
  - the attention-norm output was converted three times (Q, K, V) and the FFN-norm output twice (gate, up);
  - SwiGLU wrote a 39 MB F32 tensor that a cast then read back to write 19.4 MB of BF16.
- These passes cannot be served from L2: the tensors are 1.8-10x the L2 size, and measured reuse between kernels
  was 1-2%. Each was a full DRAM round trip, with the elementwise signature: L2 hit 33% (writes only), 69-88%
  long-scoreboard.
- With L2 capacity ruled out, the producers (RMS norm x weight, SwiGLU) now write the BF16 value the GEMMs need,
  in the same kernel that computes it. Every reader shares one BF16 copy, and the F32 tensor is never written.
  - Same arithmetic and rounding point, so the output is bit-identical.
  - Casts 278 -> 26 ms per 8-frame prefill; SwiGLU writes half the bytes (192 -> 169 ms).
- The ViT runs as a captured CUDA graph, so the BF16 buffers are allocated once and never freed while a graph may
  reference them.
- Persisting L2 (pinning the activation) was rejected: the 3.7 MB BF16 activation is almost the whole 4 MB L2.

**C5 Full cuBLASLt configuration search.**
- Hypothesis from the 16-SM geometry: the down projection's 56 output tiles (256 x 128) are 3.5 waves on 16 SMs,
  so the last wave runs half-empty (it sits at 63% of peak vs 82% for gate/up). The heuristic offers only ~8
  configurations per shape.
- Enumerating every tile, stage count, split-K and reduction scheme (900-2,400 configurations) tests this
  directly.
- Result: the small K/V GEMM, whose few tiles make it sensitive to tile choice, gained 12%. The down projection did
  not move even with split-K variants that make the tile count a multiple of 16. Its limit is the long
  K = 18,944 reduction per tile, not wave quantization.

**C6 Convert once (`--act-dedup`).**
- This was the first, simpler form of C2: it removes the redundant conversions (each a 7.3 MB read + 3.7 MB write
  per 512 tokens) without touching the producer kernels.
- It is subsumed by C2.

**D4 Per-stream cuBLASLt workspace.**
- Split-K GEMMs keep partial sums and reduction counters in the workspace. With one workspace per device, two
  models running GEMMs concurrently on different streams shared it: this corrupted the sums or deadlocked (the
  GPU sat busy at ~6 W).
- Found while testing overlap. The workspace is now per stream.

### H.3 Why the rejected ideas were tried, and why they failed

- **ViT / prefill overlap.**
  - Why tried: the ViT's crop kernels run at low occupancy (12% for the 1x2 GEMMs), which suggested idle SM
    capacity.
  - Why it failed: the prefill GEMMs leave none. Their 256 x 128 tiles take one CTA per SM with most of the shared
    memory, and they run at 63-82% of tensor peak. The two models can only time-share the 16 SMs, so total work
    is conserved: the prioritized stage runs at 1.07-1.18x its solo time and the other absorbs the rest.
  - On this chip overlap only pays across different engines (A2, A4).
- **CTA swizzle.**
  - Why tried: prefill GEMMs read 1.4-1.5x their weights from DRAM (gate/up 205 MB vs 140 MB compulsory). The 4 MB
    L2 drops a weight panel before the next token tile reads it (L2 requests ~4x the weights, 75% hit).
  - Why it failed: reordering tiles could cut those bytes, but the GEMMs are compute-bound (H.1 point 2), so the
    bytes cost no time (0.2%).
- **One ubatch per frame (`-ub 576`).**
  - Why tried: a frame's 576 tokens split as 512 + 64, and the 64-token ubatch is memory-bound (H.1 point 3). It
    streams all the weights for 64 tokens, ~90-100 ms per frame.
  - Why it gained little: 576 = 4.5 tiles of 128 tokens, so the 576-token GEMMs waste half a tile per row of
    tiles and run 30% slower for 12.5% more tokens. The net gain was only 2.8%.
  - What would work: packing frames into full 512-token ubatches.
