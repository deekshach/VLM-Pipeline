# Research agenda — VLM inference on an edge GPU

Self-contained. Assumes no prior context on this project. Compiled 2026-09-16 from
the measurements in `FINDINGS.md`; method in `METHODOLOGY.md`, derivations in
`BOTTLENECK_FUNDAMENTALS.md`.

Scope for the work proposed here: **lossless** — no pruning, no quantization, both
of which need accuracy evaluation. The baseline is **stock llama.cpp at its
defaults**, because the question is how much change stock needs to become optimal.

---

# Part I — Background

## 1. The system under test

**Model.** Qwen2.5-VL-7B-Instruct, BF16, unquantized.

| component | geometry |
|---|---|
| LLM | 28 layers, n_embd 3584, 28 heads × 128 dim, **n_head_kv 4** (GQA ratio 7) |
| FFN intermediate | 18944 |
| vocabulary | 152064 |
| vision tower | 32 layers, patch 14, 2×2 merge, head dim 80 |
| weights on device | **15.24 GB** |

**Engine.** llama.cpp + libmtmd (`b55dcdef5`). `baseline_vlm.cpp` uses the same
eight libmtmd calls as the canonical `mtmd-cli.cpp`, so nothing custom sits in the
measured path. The engine is part of what is being characterized, not a confound.

**Hardware.** RTX 4090, sm_89 (Ada), 128 SMs.

| resource | value | why it matters here |
|---|---|---|
| DRAM bandwidth | 1008 GB/s | decode's wall |
| L2 cache | **72 MB** | the *other* wall (pinning ruled out — see G2) |
| BF16 tensor peak | 165.2 TFLOP/s | idle 87% of the request |
| registers / SM | 65,536 | caps GEMM occupancy |
| shared memory / SM | ~100 KB | caps ViT GEMM occupancy |
| max warps / SM | 48 (12 per scheduler) | GEMMs get 1–2 |

**Workload.** MPI-Sintel `alley_1`, 1024×436 → 1036×448 → 74×32 patches of 14 px →
2×2 merge → **592 visual tokens per frame**. 128 generated tokens, `--ignore-eos`
(BF16 is numerically non-deterministic; generation length otherwise varies 9–128
tokens for identical input, making decode timing meaningless).

## 2. What the pipeline actually does

```
frames → PNG decode + resize (CPU)
       → mtmd_tokenize: ONE CHUNK PER FRAME, text chunks between them
       → mtmd_helper_eval_chunks:  [ViT encode → LLM prefill] × N, INTERLEAVED
       → decode loop: sample → llama_decode, one token at a time
```

Three facts about this structure that the rest of the document depends on:

1. **ViT and prefill interleave per frame** — they are not two blocks. They run on
   two different CUDA streams (clip backend and LLM backend).
2. **Each frame is encoded as an independent still image.** There is no temporal
   patch merging, so llama.cpp emits 592 tokens/frame where reference Qwen2-VL
   (`temporal_patch_size=2`) would emit 296.
3. **M-RoPE**: `n_past` advances by `max(nx,ny)` = 37 per frame, *independent of
   content* — not by token count. 13 frames → n_past 538 while the KV holds 7,696
   visual tokens. All image tokens in a frame share one temporal position.

## 3. Three fundamentals

Everything in Part III derives from these.

**F1 — The roofline.** Performance = min(peak_FLOPS, AI × bandwidth), where AI is
*arithmetic intensity* in FLOP per byte moved. The curves cross at the **ridge
point**:

```
165.2 TFLOP/s ÷ 1008 GB/s = 164 FLOP/byte
```

Below it, memory-bound — no kernel tuning helps. Above it, compute-bound.

**F2 — For a prefill GEMM, AI *is* the token count.** For `C[M,N] = A[M,K]·B[K,N]`
with weights dominating traffic, FLOPs = 2MNK and bytes ≈ 2KN, so **AI ≈ M**. Each
weight byte is read once and reused once per token. Verified on the real FFN shape
(K=3584, N=18944): M=2 → AI 2.0; M=80 → 77.9; M=512 → 437.7; M=2048 → 1219.

> **`n_ubatch` is therefore an arithmetic-intensity knob**, and any prefill ubatch
> below ~164 tokens is memory-bound by construction.

**F3 — A high cache hit rate is a bandwidth *amplifier*.** If L2 carries rate R at
hit rate h, DRAM sees only R(1−h). So L2 saturates before DRAM when
(1−h) < peak_DRAM/peak_L2. With L2 peak ≈ 2.5× DRAM peak, **L2 binds whenever
h > ~60%**. Tested against every read-reuse kernel measured: 8/8 correct.

This is the least intuitive of the three and the one that resolved the largest
open question in the prior characterization.

---

# Part II — What the measurements show

All at the **f8** operating point (8 frames, 4,736 visual tokens) unless stated.

## 4. Where the time goes

| stage | GPU busy | share | scaling in frames N |
|---|---|---|---|
| ViT encode | 419.3 ms | 12.5% | `−2.02 + 52.843·N` (R² 0.999945, **exactly linear**) |
| LLM prefill | 786.9 ms | 23.5% | `8.25 + 92.593·N + 0.5739·N²` (R² 0.999985) |
| LLM decode | 2,070.0 ms | 64.1% | flat at fixed token count (+3.3% over N=1…16) |

**The stage mix inverts with frame count** — at 1 frame the request is 93% decode;
at 16 frames it is 47% decode and 53% encode. Every encode-side optimisation
scales its value with frame count; decode-side work does not. Quote the frame
count with any encode-side result.

## 5. Whole-phase hardware counters (nsys, 100% coverage, no sampling caveat)

| counter | encode | decode |
|---|---|---|
| SMs Active | 88.0% | 93.0% |
| **SM Issue** | **16.1%** | **10.9%** |
| **Tensor Active** | **26.3%** | **0.0%** |
| DRAM total | 44.5% | **89.3%** |
| Unallocated warp slots | **71.6%** | 8.3% |
| PCIe RX / TX | 1.9% / 1.3% | 1.0% / 1.1% |

`SMs Active` 88–93% is what a casual look calls "busy". `SM Issue` 11–16% is what
is actually happening. Request-weighted **Tensor Active = 12.8%**.

## 6. Two memory walls, per launch shape (ncu, f8)

Coverage: ViT 82.1%, prefill 86.5%, decode 97.5% — per launch shape, quotable only
where launches of one shape agree.

| population | share of stage | binding unit |
|---|---|---|
| ViT GEMMs | 45.9% | **L2 64–75%** (tensor only 43–44%) |
| ViT elementwise | 25.3% | **DRAM 81–94%** |
| ViT flash-attn hd80 | 15.7% | latency-bound — nothing above 60% |
| ViT `rms_norm_f32` | 3.4% | latency-bound |
| prefill big GEMMs | 38.9% | **L2 69–74%** (tensor 48%) |
| prefill `cutlass_relu`, `mul_mat_f` | 24.6% | **DRAM 78–93%** |
| prefill attention | 5.2% | **L2, saturating to 82%** with depth |
| decode GEMV | 93% | **DRAM 88–97%** |
| decode attention | 3.2% | **L2 70–72%** |

Decode is exactly at the DRAM roofline: 15.24 GB ÷ 901 GB/s = 16.91 ms/token
predicted vs **16.82 measured (100.5%)**.

## 7. Occupancy ceilings, set by tile shape

| kernel family | regs | blk/SM | **limiter** | occupancy | warps/sched | eligible |
|---|---|---|---|---|---|---|
| prefill `ampere_*` | 230 | 2 | **registers** | 16.7% | 2.00 | 0.15 |
| ViT `cutlass` | 156 | 1 | **shared memory** | **8.3%** | **1.00** | 0.12 |
| decode `mul_mat_vec_f` | 39 | — | warps | 76–95% | 12.00 | 0.11–0.45 |

230 regs × 128 threads = 29,440; 65,536/29,440 → 2 blocks. ViT's cutlass uses
72 KB shared memory against ~100 KB/SM → 1 block. **Achieved occupancy equals
theoretical everywhere**, so these are ceilings, not tuning failures.

> Low occupancy alone is **not** a defect. Big-tile tensor GEMMs deliberately trade
> occupancy for register blocking and hide MMA latency with ILP, not warp count.
> The claim that something is lost rests on the *conjunction*: tensor 48% while it
> is the highest-utilised pipe, L2 at 69%, 0.15 eligible warps of 2 resident, and
> `CPIStall` reporting 11.7 cycles on an oversubscribed math pipe.

---

# Part III — The gaps

Ordered by (value × tractability). Each states the observation, the mechanism, why
it is a research question rather than a bug report, and what would settle it.

## G1 — Arithmetic intensity is a scheduling parameter, and the engine ignores it

**Observation.** Prefill performs **39 LLM graph evaluations to process 7,749
tokens**: 13 × 512, 13 × 80, 12 × 2, 1 × 29. Cost per token: 113 µs at 512 tokens,
264 µs at 80, **8,930 µs at 2** — a 79× spread.

**Mechanism.** By F2, AI = tokens per ubatch; by F1 the ridge is 164. **26 of the
39 evaluations are below the ridge by construction.** Each evaluation re-streams
the full 15.24 GB weight set, so total weight traffic is 594 GB = **69% of the
DRAM traffic measured during encode**. A 2-token ubatch does 0.3 GFLOP while
dragging 135.8 MB of weights: it is a decode step wearing a prefill costume.

Two independent causes:
- `n_ubatch` defaults to 512 while a frame is 592 tokens → every frame splits into
  512 + 80. **Validated fix: `-ub 640` saves 90 ms/request, decode unchanged, zero
  code.** Upstream knows — `mtmd-helper.cpp:275` reads *"TODO: n_ubatch must be
  enough to hold the image"*.
- Twelve 2-token `<|vision_end|><|vision_start|>` separator chunks, each running
  all 28 layers: **214 ms/request for 24 tokens**. Not fixable by a flag —
  `llama_batch` carries `token` **or** `embd`, so a separator cannot share a
  ubatch with image embeddings.

**Why it is research, not a bug report.** The general claim is *the roofline ridge
is a scheduling constraint, and it is a computable per-(model, GPU) constant*. Any
interleaved-modality engine — video, audio, multi-image, interleaved documents —
produces a *distribution* of ubatch sizes, and no engine's batch splitter is
ridge-aware. The kernel-level signature is visible and checkable: **prefill is the
only stage whose launch-shape count is context-dependent** (75 → 131 shapes from 1
to 16 frames, while ViT holds ~28 and decode ~15).

**Settles it.** `-ub` sweep 512/592/1024/2048 with per-stage attribution; predict
the prefill shape count collapses at ≥592. Then a ridge-aware chunk packer.

**Excess available: 383 ms/request (9.3%), of which 90 ms is already banked.**

---

## G2 — The L2 wall: PROVEN BY INTERVENTION; the pinning sub-gap is DEAD

> **RESOLVED 2026-09-16.** The `--cache-control none` experiment below settled this.
> Cross-kernel L2 reuse is real and large, but it buys **no time** for the
> L2-bound kernels — which proves the L2 wall causally and simultaneously kills
> L2-residency control as an intervention. The tiling half of the question
> survives and merges into G4. Raw logs: `results/cache_control/`.

**Observation.** The GEMMs that should be compute-bound are not, and they are not
DRAM-bound either — they are **L2-bandwidth-bound**:

| | tensor% | DRAM% | **L2 thr%** | L2 hit |
|---|---|---|---|---|
| ViT `cutlass @1184` | 43.6 | 12.3 | **65.1** | 97.0 |
| prefill `ampere_64x128 @1184` | 48.4 | 33.0 | **68.8** | 91.5 |
| prefill `ampere_128x64 @896` | 48.2 | 36.1 | **73.7** | 90.6 |
| prefill attention at depth | 52.8 | 10.1 | **82.1** | 97.7 |

Prefill at 512 tokens has AI = 438, **2.7× above the ridge** — it *should* be
solidly compute-bound. It reaches 48% of tensor peak.

**Mechanism.** F3. The 90–98% hit rate is not evidence of health; it is *why* L2
binds — traffic that would have gone to DRAM is served by a unit with finite
bandwidth. ncu's own `SOLBottleneck` rule says so in the CSV: *"Memory is more
heavily utilized than Compute: … identify the L2 bottleneck."* Prefill attention
is the cleanest case: as context deepens, L2 throughput climbs 4.8 → 75.7 → 82.1%
while DRAM **falls** 37 → 10%.

**The sub-question that makes it research.** Is the L2 traffic *reducible*? The
4090 has 72 MB of L2 and a 512-token activation tile is only 3.7 MB, so this is
not a capacity problem — it is bandwidth against a hit rate so high that L2
absorbs nearly all traffic. Whether larger tiles or fewer passes over the
activation tile cut L2 traffic per FLOP is **unmeasured**.

**The blocking caveat.** ncu defaults to `--cache-control all`, which flushes
L1/L2 before every replay pass. **Every hit rate quoted here is intra-kernel reuse
only**; cross-kernel reuse is excluded by construction — and cross-kernel reuse is
exactly what an L2-residency intervention would exploit.

### The experiment (2026-09-16)

ncu defaults to `--cache-control all`, flushing L1/L2 before every replay pass, so
every hit rate above is intra-kernel reuse only. Re-measured with
`--cache-control none`. **Methodological trap avoided**: with no flush *and*
multi-pass replay, passes 2+ see the cache warmed by pass 1 of the same kernel,
which inflates the hit rate as an artifact. A metric set small enough to fit in
**one pass** was used, so there is no replay and no self-warming.

| kernel | L2 hit (flush → none) | DRAM read | **Δ duration** |
|---|---|---|---|
| ViT `cutlass` @1184 | 97.16 → 98.28 | 18.57 → 10.15 MB (**−45.3%**) | **−0.0%** |
| ViT `silu_gated` @31635 | 33.35 → **53.72** | 65.17 → 45.06 MB (−30.8%) | **−11.3%** |
| prefill `ampere_64x128` @1184 | 91.42 → 91.67 | 141.51 → 137.20 MB (−3.0%) | −0.5% |
| prefill attn hd128 @256 | 96.23 → 98.91 | 11.40 → 3.80 MB (**−66.7%**) | −5.0% |

Three launches per cell; the a-priori sizing held (the ViT GEMM recovers 8.4 MB
against a predicted 6–15 MB working set; prefill's GEMM barely moves because its
135.8 MB of weights cannot fit in L2 regardless).

### What it proves

**The ViT GEMM loses 45.3% of its DRAM traffic and runs 0.0% faster.** That is a
causal demonstration that these kernels are **L2-bandwidth-bound, not
DRAM-bound** — far stronger than the throughput-percentage inference that
originally established the L2 wall.

### What it kills

**L2 residency control is not a viable intervention here.** Pinning changes what
is *resident*; it does not increase the rate at which L2 *serves*. The one kernel
that gains real time is `silu_gated` (−11.3%), and it gains because it is
DRAM-saturated at 94% — but **production never flushes**, so that gain is already
being realised. `none` is the realistic case; `all` was the artifact. Strike the
49.5 MB pinnable capability from the agenda.

### What survives

*Is L2 traffic per FLOP reducible by a different tile?* Untouched by this
experiment, which changed residency and not the access pattern. **Merged into G4.**

### Correction it forces on the survey

Absolute DRAM figures and durations for memory-bound kernels are **pessimistic**,
having been measured under flush — real `silu_gated` is 53.7% L2 hit at 73.8 µs,
not 33.4% at 83.1 µs. The *classification* of every kernel is unchanged, and the
L2-bound classification is strengthened. Keep `--cache-control all` as the default
for per-kernel comparison (it is reproducible); quote a no-flush number alongside
any claim about absolute bytes or time.

---

## G3 — Cross-modal scheduling: three phases, three resource vectors, executed serially

**Observation.** ViT and prefill are independent per frame yet overlap only
**14%** (58.6 ms of 419.3 ms at f8; 578 ms/request ceiling at 13 frames). And the
phases have genuinely different resource profiles — encode 44.5% DRAM / 26.3%
tensor, decode 89.3% DRAM / **0.0%** tensor.

**Mechanism, part 1 — a false dependency.** `mtmd_helper_eval_chunks`
(`mtmd-helper.cpp:393`) is a sequential host loop calling a synchronous
`eval_chunk_single(…, n_past, &n_past)`. The accumulator creates the dependency,
but under M-RoPE `n_past` advances by 37 per frame *independent of content*, so
every frame's positions are computable upfront. Frame N+1's ViT does not need
frame N's prefill. (A stronger hypothesis — that `llama_set_causal_attn` toggling
forces serialization — was tested and rejected: `mtmd_decode_use_non_causal` is
GEMMA3-only and false for Qwen2.5-VL.)

**Mechanism, part 2 — how to combine phases correctly.** A phase's bottleneck is a
**vector**, and two phases combine by **resource-time product** (utilisation ×
wall time), *not* by summing utilisations — those are rates over different
durations, and adding them is a category error. Co-running cannot finish faster
than `max_r(total resource-ms)`:

| resource | encode (1850 ms) | decode (2153 ms) | total |
|---|---|---|---|
| **DRAM** | 43.8% → 810 | 89.4% → 1925 | **2735 ← binds** |
| L2 | 59.6% → 1103 | 42.6% → 917 | 2020 |
| tensor | 26.9% → 498 | 0.0% → 0 | 498 |
| SM issue | 16.3% → 302 | 11.1% → 239 | 541 |

Serial 4003 ms → bound 2735 ms = **1.46× ceiling**, DRAM-binding. Not the naive
2×, and not ~0 — an earlier version of this analysis wrongly claimed the latter.

**The result that makes it interesting.** Halve decode's bytes (q8_0) and the
bound moves to **1.98×**, with **L2** binding instead of DRAM. **Quantization and
co-scheduling are multiplicative, not alternatives** — each relieves the resource
that limits the other. That claim is only available with the full resource vector,
which is the point of measuring one.

**Settles it.** Two concurrent inference streams; prediction: DRAM pins near 100%
and the speedup lands ≤ 1.46×. Separately, a pipelined `mtmd_helper` — land the
no-op refactor (pre-compute the whole `n_past` schedule, assert it matches) before
touching concurrency.

---

## G4 — Vendor-library tile selection on multimodal shapes

**Observation.** The GEMMs are capped at 8.3–16.7% occupancy, by **different
resources per family** — registers for prefill's `ampere_*`, shared memory for
ViT's `cutlass`. Achieved equals theoretical, so the tile shape sets the ceiling.

**Why it is a question.** These are cuBLAS *heuristic-selected* kernels for shapes
like 512 × 3584 × 18944. Nobody chose them for this shape, and cuBLAS cannot be
recompiled. The testable hypothesis: **vendor heuristics do worst on the
non-power-of-two token counts multimodal models produce** — 592, not 512.

**Honest scoping.** Only ggml's own kernels are reachable in-tree. llama.cpp
already exposes the exact knob in `fattn-mma-f16.cuh`: a per-shape
`fattn_mma_config` with an `occupancy` field feeding `__launch_bounds__`, currently
2 for both (80,80) and (128,128), with `static_assert(occupancy <= 8)`. And there
is **no Ada table** — sm_89 falls through to the Ampere one, while the file carries
a `// TODO tune specifically for Volta`, so per-arch retuning is accepted practice
that has not been done for this GPU. Addressable share: the two flash-attention
kernels, ~199 ms/request (~5%). The cuBLAS GEMMs are not reachable this way.

**Settles it.** Sweep `occupancy` {2,3,4} and `nbatch_fa`; separately, quantify the
heuristic gap with the CUTLASS profiler standalone — no llama.cpp changes needed
to *measure* it.

---

## G5 — Decode's hard wall, and what is actually available under a lossless constraint

**Observation.** Decode is 64% of the f8 request, sits at 88–97% of DRAM peak, and
uses **zero** tensor cores.

**Mechanism.** AI = 1 FLOP/byte against a 164 ridge — 0.006×. A GEMV has no reuse
dimension, so MMA cannot be used at all. The roofline prediction matches measured
to 0.5%.

**What this rules out.** No kernel tuning, no prefetching (it is bandwidth-
saturated, not latency-bound), no occupancy fix (8.3% unallocated warps, 76–95%
achieved). Decode cannot be fixed by scheduling *within* a request.

**What remains within a lossless scope.** Only G3 — overlapping decode with
another request's encode, where the tensor cores and ~55% of DRAM are idle. Note
that batching raises AI to B FLOP/byte, but even B = 64 is 0.39× the ridge: it
makes the tensor cores *usable*, not *useful*.

**Beyond the lossless scope** (recorded so the boundary is explicit): quantization
is near-deterministic here because decode time *is* bytes/bandwidth — predicted
1.9× for q8_0, 3.4× for q4_K_M. And the 2× visual-token inflation (§2.2) is an
engine deficiency, not a model limitation, but changing it changes outputs.

---

## G6 — Operating-point sensitivity is itself a finding

**Observation.** The kernel mix is workload-dependent, and not smoothly:

| frames | ViT shapes | prefill shapes | decode shapes |
|---|---|---|---|
| 1 | 28 | 75 | 15 |
| 8 | 31 | 110 | 16 |
| 16 | 27 | 131 | 15 |

Worse, **decode's attention changes kernel family**: `flash_attn_ext_vec @756` at
f8 and f13, but `flash_attn_ext_f16<128,128,1,8> @128` plus a stream-K fixup at
f16 — same build, same model. The dispatch (`fattn.cu:368`) gates the vector
kernel on `K->ne[1] % FATTN_KQ_STRIDE == 0` (stride 256), and the KV length is the
only workload-dependent term.

**Why it matters.** ncu targets, coverage figures and every per-kernel claim are
valid only at the operating point measured. A single-point characterization — the
norm in this literature — silently generalises across a discontinuity. This is a
methodological contribution as well as a caution: it says what a VLM
characterization must report that an LLM one need not.

---

# Part IV — Ruled out, and the constraints

## What has been measured and eliminated

| hypothesis | verdict |
|---|---|
| PCIe / host transfer is a bottleneck | **no** — 1.9% RX, 1.3% TX |
| The CPU is a bottleneck | **no** — decode GPU busy 97%, CPU runs 7.35 ms ahead |
| Decode is occupancy-limited | **no** — purely bandwidth |
| ViT frame batching would help | **no** — 7.7–9.3 waves, achieved == theoretical |
| Software prefetching would help decode | **no** — bandwidth-saturated, not latency-bound |
| ViT and decode are complementary | **partly** — DRAM-limited, 1.46× not 2× |

## The lossless budget

| item | prize | effort | status |
|---|---|---|---|
| `-ub ≥ 592` (G1) | **90 ms/request** | one CLI flag | **validated** |
| ViT/prefill pipelining (G3) | **578 ms/request** | days | not started |
| separator fusion (G1) | **214 ms/request** | days | not started |
| CUDA graph latch | **51 ms/request** | ~10 lines | diagnosed |
| fattn `occupancy` (G4) | ~199 ms addressable | hours | not started |
| phase co-scheduling (G3) | **1.46×** throughput | zero code to test | not started |
| ~~L2 residency (G2)~~ | **none** | — | **ruled out 2026-09-16** |

**933 ms of a 4,101 ms request = 22.7%**, before any co-scheduling gain, without
touching model outputs.

## Recommended order

1. ~~`--cache-control none` re-profile~~ — **done 2026-09-16**; cancelled the
   L2-pinning work and upgraded the L2 wall from inferred to proven.
2. **`-ub` sweep** — extends a validated win into a curve, and tests G1's
   kernel-level prediction at the same time.
3. **Two concurrent streams** — tests G3's 1.46× bound with zero code.
4. **Pipelining** (G3) — the largest single prize; land the no-op refactor first.
5. **Separator fusion** (G1), then **fattn occupancy / tiling** (G4).

Steps 2–3 cost about a day and produce two decisions.

---

# Appendix — the load-bearing numbers

| quantity | value | source |
|---|---|---|
| ridge point | 164 FLOP/byte | 165.2 TF ÷ 1008 GB/s |
| decode roofline check | 16.91 predicted / 16.82 measured ms/token | nsys hw counters |
| f8 request | 3,344.9 ms GPU | `stage_attrib.py` |
| ViT slope | 52.843 ms/frame, R² 0.999945 | 9-point sweep |
| prefill N² coefficient | 0.5739 ms/frame² | 9-point sweep |
| prefill evaluations (f13) | 39 for 7,749 tokens | `rms_norm_f32` grid |
| `-ub 640` saving | 90 ms/request, decode unchanged | 3-round A/B |
| ViT/prefill overlap | 14.0% (f8) | `stage_attrib.py` |
| co-scheduling ceiling | 1.46× (1.98× with q8_0) | resource-time product |
| L2 capacity / pinnable | 72 MB / 49.5 MB | `cudaGetDeviceProperties` |
| KV per token per layer | 2,048 B | 2 × 4 heads × 128 dim × 2 B |
| ncu coverage (f8) | ViT 82.1%, prefill 86.5%, decode 97.5% | per launch shape |
