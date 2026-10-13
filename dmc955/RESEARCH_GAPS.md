# Research gaps in this execution method — grounded in the measurements

Qwen2.5-VL-7B-BF16 on llama.cpp/libmtmd, RTX 4090 (sm_89), MPI-Sintel, 13 frames,
128 tokens. Every claim below cites a specific measurement in
`profiling_results/handshake_7B.*` (nsys) or `ncu_2026-09-02_7B-BF16/` (ncu).
Method caveats for both live in `PROFILING_HANDOFF.md` §6a.

Written 2026-09-15. Ranked by (value x confidence), not by novelty.

---

## The one-paragraph version

The request splits 47.5% encode / 52.5% decode, and **the two halves fail on
opposite microarchitectural axes**. Decode runs at 1 FLOP/byte against a 164
FLOP/byte ridge point — it streams the entire 15.2 GB weight set once per token
and uses **zero** tensor cores. Encode has 592 FLOP/byte, 3.6x *above* the ridge,
so it should be tensor-bound — yet it reaches only 26.9% Tensor Active, blocked by
L2 bandwidth (69-74% of peak) and a register-capped occupancy that leaves **71.6%
of warp slots on active SMs empty**. Across the whole request the 4090's dominant
compute unit is idle **87%** of the time. Separately, the ViT and the LLM prefill
are independent per frame but execute in lockstep, overlapping only 13% — 578
ms/request of pure scheduling loss.

---

## Hardware counters, whole-phase, 100% coverage

From nsys `GPU_METRICS` (21,530 samples/phase) — independent of the ncu sampling
and free of its coverage caveats. Both requests agree to <1%.

| counter | encode | decode |
|---|---|---|
| SMs Active | 88.0% | 93.0% |
| **SM Issue** | **16.3%** | **11.1%** |
| **Tensor Active** | **26.9%** | **0.0%** |
| DRAM read | 31.6% | 88.3% |
| DRAM write | 12.2% | 1.0% |
| **DRAM total** | **43.8%** | **89.4%** |
| Compute warps in flight | 6.11e6 | 21.84e6 |
| **Unallocated warps in active SMs** | **15.42e6 (71.6%)** | 1.98e6 (8.3%) |
| PCIe RX / TX | 1.9% / 1.3% | 1.0% / 1.1% |

Read this table twice. `SMs Active` 88-93% is what a naive look at the GPU reports
as "busy". `SM Issue` 11-16% is what it is actually doing. The gap is the whole
story.

These also **independently confirm the 4c correction**: encode DRAM is 43.8%, not
the 16.17% that the "complementary-bottleneck" proposal was built on. The
ncu-weighted estimate was 42.2% — agreement to 1.6 points.

---

## Gap 1 — ViT and prefill are independent per frame but run in lockstep

**Value: 578 ms/request = 14.1%. Confidence: high. No accuracy cost.**

`mtmd_helper_eval_chunks` runs `[ViT encode frame N -> LLM prefill frame N]` x 13
serially. Measured on the encode window of request 1:

| | ms |
|---|---|
| ViT busy (stream 17) | 664.9 |
| prefill busy (stream 15) | 1271.4 |
| sum of work | 1936.3 |
| union (wall) | 1849.7 |
| **actual overlap** | **86.6 — only 13% of the ViT** |
| GPU fully idle inside encode | 98.3 (5.0%) |

If the ViT were fully hidden behind prefill, encode would be `max(665, 1271)` =
1271 ms instead of 1850 — **578 ms/request, 14.1% of a 4101 ms request**. That is
11x the CUDA-graph prize (§4b) and larger than anything else on this list.

**Why it should be possible**: frame N+1's ViT does not depend on frame N's
prefill. Only the KV *append order* matters, and under M-RoPE each frame's
positions are known a priori (`n_past` advances by `max(nx,ny)` = 37 per frame,
independent of content). So the dependency being enforced is an artifact of the
helper's loop structure, not of the model.

**Why the headroom is real**: during encode, DRAM is at 43.8%, Tensor at 26.9%,
SM Issue at 16.3%, and 71.6% of warp slots are empty. There is room for more
concurrent work on every axis.

**Why it will not be the full 578 ms**: ViT and prefill are *both* L2-leaning
(54.8% and 59.5% time-weighted), summing to ~114% of L2 — L2 becomes the contended
resource. Expect a substantial fraction, not all. **That is the experiment**: it is
the one pairing in this workload where the shared resource is not already
saturated, unlike ViT+decode (133% DRAM, measured).

---

## Gap 2 — decode is 1 FLOP/byte and uses zero tensor cores

**Value: 52.5% of request time. Confidence: highest — this is exact.**

Decode time is **entirely** the weight stream, confirmed to 0.5%:

```
measured                16.82 ms/token   (2153 ms / 128 tokens)
DRAM achieved            901 GB/s        (89.4% of 1008 GB/s peak, nsys hw)
bytes moved per token   15.16 GB
LLM weight set          15.24 GB
predicted               16.91 ms/token   -> 100.5% of measured
```

Every token drags the whole model across the bus. Arithmetic intensity is 2 FLOP
per weight element over 2 bytes per BF16 element = **1 FLOP/byte**, against a 4090
ridge of 165.2 TFLOPS / 1008 GB/s = **164 FLOP/byte**. Decode sits at **0.006x the
ridge**, which is why `Tensor Active` reads exactly **0.0%**.

Directions, with what the data says about each:

| direction | predicted effect | what the data supports |
|---|---|---|
| quantization | q8_0 ~1.9x, q4_K_M ~3.4x on decode | decode time *is* bytes/BW, so this is near-deterministic. Highest-certainty win available. |
| batch / speculative decode | weight streams amortised over B tokens | raises intensity to B FLOP/byte — but **even B=64 is only 0.39x ridge**, so it never becomes compute-bound; it reduces streams, it does not reach the tensor cores |
| request-level batching | natural for streaming video | the VLM-specific angle: a video stream is a queue of requests over the *same* weights |

The honest framing for a paper: **decode cannot be fixed by scheduling, only by
moving fewer bytes.** Note also that decode is *not* occupancy-limited (only 8.3%
unallocated warps, achieved occupancy 91-97%) — it is purely bandwidth. Do not
propose occupancy fixes for decode.

---

## Gap 3 — prefill has the intensity to be tensor-bound and reaches 43-48%

> **FULLY WORKED IN `GAP3_PREFILL.md`** (2026-09-15). It turned out to be *two*
> problems. Headline: 2/3 of prefill's LLM evaluations run below the roofline ridge
> purely from batch chunking — **383 ms/request of excess (9.3%)**, of which
> **90 ms/request is already recovered by one flag (`-ub 640`), measured**. The
> kernel-efficiency story below applies only to the 60% that is properly batched.
> Governing rule: for a prefill ubatch, arithmetic intensity **= n_tokens**, and
> the 4090 ridge is 164 FLOP/byte — so any ubatch under ~164 tokens is
> memory-bound by construction. 26 of 39 evaluations per request are under it.

**Value: 31% of request time (prefill). Confidence: high. The sharpest anomaly.**

A prefill GEMM processes 592 tokens per chunk, so each weight is reused 592 times:
**592 FLOP/byte, 3.6x above the ridge point.** It should be solidly compute-bound.
It is not:

| | ncu, per launch | hw counter, whole phase |
|---|---|---|
| tensor pipe | 48.2-48.4% | 26.9% (encode incl. ViT) |
| L2 throughput | **68.8-73.8%** | — |
| DRAM | 32.4-37.2% | 43.8% |
| theoretical occupancy | 16.7% (**register-limited**) | — |
| warps/scheduler (hw max 12) | 2.00 | — |
| eligible warps/scheduler | 0.15 | — |
| issue slots busy | 10.1-10.2% | 16.3% |
| unallocated warp slots | — | **71.6%** |

Two microarch parameters bind simultaneously:

1. **L2 bandwidth.** ncu's own `SOLBottleneck` rule on these launches reads
   *"Memory is more heavily utilized than Compute: ... identify the L2
   bottleneck."* The 90-97% L2 hit rate is not evidence of health — it is *why*
   L2 is the limit: traffic that would have gone to DRAM is served by a unit with
   finite bandwidth. And because ncu defaults to `--cache-control all` (L1/L2
   flushed per replay pass), these hit rates are **intra-kernel reuse only** — the
   real figure can only be higher, so the L2 pressure is *understated*.
2. **Register-capped occupancy.** `TheoreticalOccupancy` rule: 2.00 warps/scheduler
   against a hardware maximum of 12, limited by registers per thread (230). With
   1-2 resident warps and ~11.7-cycle stalls on an oversubscribed tensor pipe
   (`CPIStall`), the scheduler issues once every ~10 cycles. Achieved occupancy
   already equals the theoretical ceiling — this is a *ceiling*, not a failure to
   reach it.

**The research question**: these are cuBLAS/cutlass kernels selected for generic
shapes. Is `ampere_bf16_s1688gemm_bf16_64x128` the right tile for
592 x 3584 x 3584 on sm_89, given that it lands at 16.7% occupancy and 69% L2? A
tile shape chosen for *this* shape — trading register tiling for occupancy, or
larger tiles to cut L2 traffic per FLOP — is a concrete, measurable experiment.
This is the one place in the pipeline where real compute is being left on the
table rather than being structurally unreachable.

The ViT's GEMMs are the same story, worse: `cutlass @1184` runs at **1.00
warp/scheduler**, 8.3% theoretical occupancy, 64.6% L2, 43.4% tensor.

---

## Gap 4 — the engine emits 2x the visual tokens the reference model would

**Value: touches ViT, prefill and KV simultaneously. Confidence: high.**

llama.cpp encodes each video frame as an **independent still image**. Reference
Qwen2-VL uses `temporal_patch_size=2`, pairing adjacent frames. So llama.cpp emits
**592 tokens/frame where the reference emits 296**.

This is visible in the transfer trace, not just the token count: the dominant H2D
copy during encode is **12 x 22.43 MB**, and one frame's patch tensor is
2368 x 1176 x 4 B = **11.14 MB**. The upload is exactly 2x the patch data, i.e.
the frame is being temporally duplicated before it is even encoded.

Consequences, all measured: ViT is *exactly* linear in frames (62.7 ms/frame,
content-independent to +/-0.3 ms), so halving tokens halves ViT; prefill is
super-linear (+6.2% per frame, because each frame attends over all prior visual
tokens), so halving tokens cuts prefill by *more* than half; and KV occupancy
halves (7,696 -> 3,848 visual tokens at 13 frames).

This is an **engine deficiency, not a model limitation** — which makes it the most
defensible "the system, not the weights" contribution in the whole
characterization. It is also the honest baseline caveat: any token-pruning result
measured here starts from 2x the reference token count and must say so.

---

## Gap 5 — the L2 wall itself

**Value: sets the ceiling on Gaps 1 and 3. Confidence: medium-high.**

New in §4c: ViT GEMMs 63-73% of L2 peak, prefill GEMMs 69-74%, prefill attention
**saturating to 85% at context depth** (67.5 -> 78.2 -> 85.0% as KV grows, while
DRAM *falls* 17.0 -> 9.5% and tensor never exceeds 54.7%). Decode by contrast uses
only ~41% of L2.

Open question: **L2 traffic per FLOP.** The 4090 has 72 MB of L2 and one prefill
chunk's activations are only 592 x 3584 x 2 B = 4.2 MB, so this is not a capacity
problem — it is bandwidth against a hit rate so high that L2 absorbs nearly all
traffic. Whether that is reducible (larger tiles, fewer passes over the activation
tile) is unmeasured and is the natural follow-on to Gap 3.

Caveat to state whenever quoting these: `--cache-control all` means the measured
hit rates exclude inter-kernel reuse, so they are conservative.

---

## Gap 6 — CUDA graphs are latched off for the entire LLM

**Value: 50.7 ms/request = 1.24%. Confidence: high. Fully explained.**

See §4b. The graph cache is keyed on `cgraph->nodes[0]`, which is a fixed address
in a reused buffer, so prefill and decode share one cache slot *and* one
`disable_due_to_too_many_updates` latch. The VLM's shape-varying prefill trips it
after 4 evaluations and nothing ever clears it, so the 2.15 s shape-stable decode
runs entirely ungraphed. The ViT keeps graphs at 100% because `clip.cpp` has its
own buffer, hence its own key.

Small, but it is a *clean, fully-mechanised* result about the engine — good for a
systems paper even at 1.24%.

---

## Gap 7 — host-staging amplification (platform risk, not a bottleneck here)

**Value: ~0 on this box; potentially large on Jetson. Confidence: low-medium.**

578.6 MB of H2D per encode (all on stream 14) for a pixel input of roughly 18 MB —
about a 32x amplification. On the 4090 this is **not** a bottleneck: PCIe RX sits
at 1.9% of peak and memcpy costs 46.5 ms of a 1948 ms encode.

But `portedForJetson/` exists in this tree, and on Jetson host and device share
one LPDDR controller — that traffic would contend with the DRAM the GPU is already
driving to 89% during decode. Worth re-measuring there before assuming it stays
free. I have identified only the `12 x 22.43 MB` component (the temporally
duplicated patch tensor, Gap 4); the rest — a 14 x 7.34 MB set, a 12 x 5.57 MB
set, and a descending per-frame ladder from 15.73 MB down to 1.39 MB — is
unexplained and would need instrumentation to attribute.

---

## Ruled out — negative results worth reporting

Stating these prevents re-litigating them, and two contradict earlier claims.

- **PCIe is not a bottleneck.** 1.9% RX / 1.3% TX at peak. The 16.7 GB of total
  H2D is dominated by model load, not per-request work.
- **The CPU is not a bottleneck.** Decode GPU busy 97.0%, CPU runs ~7.35 ms ahead
  of the GPU, median inter-kernel gap 0.64 us.
- **Occupancy is not decode's problem.** 8.3% unallocated warps, achieved
  occupancy 91-97%. Purely bandwidth.
- **ViT frame batching would buy nothing.** 7.71-9.25 waves/SM and achieved
  occupancy equals theoretical — there is no tail effect to recover. (Retracted
  claim; see §5.)
- **ViT + decode co-scheduling is DRAM-limited, not free — but not worthless
  either.** Measured DRAM 43.8% (encode) and 89.4% (decode). By resource-time
  product the ceiling is **1.46x**, not the naive 2x, and not ~0 as an earlier
  draft of this file claimed (it summed utilisations across unequal durations —
  a category error; see T3). The original ~18% figure in §9 rested on a ViT DRAM
  value of 16.17% that ignored the DRAM-saturated elementwise quarter.

---

---

# Recast in the MICRO'24 register  (2026-09-15)

> **`BOTTLENECK_FUNDAMENTALS.md` derives all of this from first principles** —
> the roofline ridge, why AI equals the token count, and why a high cache hit rate
> is a bandwidth amplifier. Read that for the *why*; this section is the ranking.

The gaps above are ordered by measured prize. This section re-orders them by
whether they are a **microarchitectural contribution** — finding, named mechanism,
counter evidence, an available knob, and a claim that generalises past this engine
— which is the bar `RELATED_MICRO24_DLRM.md` sets. Some of the biggest prizes above
are engineering fixes; some smaller ones are better research.

## The thesis the data supports

> Multimodal inference on an edge GPU is not one workload but three, whose
> arithmetic intensities differ by **500x** (1 vs 512 FLOP/byte) against a single
> 164 FLOP/byte ridge point. The engine's chunking and scheduling are oblivious to
> where each lands relative to that ridge, and **that obliviousness — not kernel
> tuning — is where the performance goes.**

Every number needed to support it is already measured.

## Tier 1 — genuine microarchitecture contributions

### T1. Chunk-boundary-induced arithmetic-intensity collapse
*The strongest new framing available to us.*

- **Finding**: 39 LLM graph evaluations to prefill 7,749 tokens; **26 of 39 sit
  below the ridge**; 383 ms/request excess; weight re-streaming 39 x 15.24 GB =
  594 GB = **69% of the encode phase's measured DRAM traffic**.
- **Mechanism**: for a prefill GEMM, arithmetic intensity **= tokens per ubatch**.
  Interleaved-modality chunking produces a *distribution* of ubatch sizes
  (512/80/29/2 here), and llama.cpp's batch splitter is oblivious to the ridge.
- **Knob**: ridge-aware chunk packing. Partially validated: `-ub 640` = 90 ms.
- **Generalises**: to any interleaved modality — video, audio, multi-image,
  interleaved image-text documents — and the ridge is a computable per-(model, GPU)
  constant, so the rule is portable, not tuned.
- **Settles it**: `-ub` sweep 512/592/1024/2048 + the separator-fusion prototype.

### T2. The visual KV working set as an L2-residency design parameter

- **Finding**: prefill attention's L2 throughput climbs 67.5 -> 78.2 -> **85.0%**
  with context depth while DRAM *falls* 17.0 -> 9.5% and L2 hit reaches 98.3%.
  Per-layer visual KV = 7,696 tok x 2 KB = **15.8 MB**, against 72 MB of L2
  (49.5 MB pinnable, queried).
- **Mechanism**: the visual KV is write-once/read-many and its size is set by the
  *model's token policy* (tokens/frame x frames x layers x KV dtype). While it fits
  L2, prefill is **L2-bandwidth**-bound; past that it must become DRAM-bound.
  **There is a cliff, and its location is a design parameter.**
- **Knob**: KV dtype, tokens/frame, frame count — and L2 residency control.
- **Generalises**: any long-visual-context model; the cliff formula is analytic.
- **Settles it**: KV-dtype sweep (f16/q8_0/q4_0) x frame count to locate the cliff.
  Prerequisite: re-profile with `--cache-control none`, because our L2 hit rates
  are intra-kernel only (§6a).
- **Why it matters beyond this paper**: it tells the pruning work what it is
  actually buying. Pruning visual tokens shrinks this working set — so it may move
  prefill *off* the L2 wall, a second-order benefit nobody has quantified.

### T3. Phase-level multi-resource complementarity — and a negative result

- **Finding**, whole-phase hardware counters at 100% coverage:

  | | DRAM | L2 | Tensor | SM Issue |
  |---|---|---|---|---|
  | encode | 43.8% | ~52% | 26.9% | 16.3% |
  | decode | 89.4% | ~43% | **0.0%** | 11.1% |

- **Mechanism**: a phase's bottleneck is a *vector*, not a scalar, and the right
  way to combine two phases is the **resource-time product** (utilisation x wall
  time), not the sum of utilisations — utilisations are rates over *different*
  durations, so adding them is a category error. Co-running cannot finish faster
  than `max_r(total resource-ms for r)`:

  | resource | encode (1850 ms) | decode (2153 ms) | total resource-ms |
  |---|---|---|---|
  | **DRAM** | 43.8% -> 810 | 89.4% -> 1925 | **2735  <- binds** |
  | L2 | 59.6% -> 1103 | 42.6% -> 917 | 2020 |
  | tensor | 26.9% -> 498 | 0.0% -> 0 | 498 |
  | SM issue | 16.3% -> 302 | 11.1% -> 239 | 541 |

  Serial wall 4003 ms; lower bound 2735 ms -> **max speedup 1.46x (31.7% of GPU
  time removed)**, DRAM-bound.
- **CORRECTION (supersedes an earlier claim in this file and in §4c).** I
  previously wrote that ViT+decode "sum to 133% DRAM, therefore they collide and
  the gain is ~nothing". That summed utilisations, which is wrong. DRAM does become
  the *binding* resource, but it still admits **1.46x**. The naive hope was 2x; the
  honest ceiling is 1.46x; it is not ~0.
- **The real negative result, restated**: the pairing is limited by DRAM, not by
  the complementarity of compute vs memory. The original argument (ViT is
  compute-bound at 16.17% DRAM) was built on a figure that ignored the ViT's
  DRAM-saturated elementwise quarter; measured, the ViT is 43.8% DRAM.
- **Composition is the interesting part**: relieve DRAM (q8_0 weights ~halve
  decode's bytes) and the bound moves to **1.98x**, with **L2** becoming the shared
  limit. So quantization and co-scheduling are *multiplicative* — each relieves the
  resource that limits the other. That is a result worth stating explicitly.
- **Knob**: co-scheduling driven by the full resource vector.
- **Settles it**: two concurrent inference processes; compare aggregate throughput
  against 2x serial *and* check whether DRAM pins near 100% as predicted.

## Tier 2 — strong results, more engineering than microarchitecture

### T4. Interface-enforced cross-modal serialization — 578 ms/request
ViT and prefill overlap only **13%** (86.6 ms of 664.9 ms). Mechanism now precise:
`mtmd_helper_eval_chunks` (`mtmd-helper.cpp:393`) is a sequential host loop calling
a synchronous `eval_chunk_single(..., n_past, &n_past)`. The `n_past` accumulator
creates a **false** data dependency — under M-RoPE `n_past` advances by
`max(nx,ny)` = 37 per frame *independent of content*, so every frame's positions
are computable upfront. The serialization is an artifact of the API's shape, not of
the model or the hardware. (Checked and rejected a better hypothesis:
`llama_set_causal_attn` toggling would force it, but `mtmd_decode_use_non_causal`
is GEMMA3-only and false for Qwen2.5-VL.)

### T5. Vendor-library tile selection on non-power-of-two multimodal shapes
cuBLAS picks kernels whose tile shape caps occupancy at **16.7%** (registers, 230/thread)
and **8.3%** (shared memory, 72 KB/block) for shapes like 512 x 3584 x 18944.
Achieved occupancy equals theoretical, so these are ceilings, not tuning failures.
Hypothesis worth testing: heuristics do worst on the odd token counts multimodal
models produce (592, not 512). Measurable standalone with the CUTLASS profiler — no
llama.cpp changes needed to quantify the gap.

### T6. M-RoPE prevents token-granular KV reclamation
All image tokens in a frame share one temporal position, so `llama_memory_seq_rm`
can only evict at **frame** granularity. Consequence for the project's own pruning
work: you can prune visual-token *compute* but not the *KV footprint*. A clean
architectural negative result, and it bounds what token pruning can deliver.

## Tier 3 — real fixes, weak as research contributions

- **CUDA graph latch** (§4b): fully mechanised, 1.24%. A good short paper section,
  not a thesis.
- **2x visual-token inflation** (Gap 4): large and important, but an engine feature
  gap rather than a microarchitectural finding. Belongs in the baseline caveats.
- **Host-staging amplification** (Gap 7): 32x, but latency-hidden here. Only
  becomes a result on Jetson.

## What is missing to make any of this publishable

Three things, in order:

1. **Regime sweeps.** Every Tier-1 gap needs a curve, not a point. This is the
   single largest methodological gap (`RELATED_MICRO24_DLRM.md` §2).
2. **`--cache-control none` re-profile.** Every L2 claim currently excludes
   cross-kernel reuse by construction, which is exactly the reuse T2 is about.
3. **A second architecture.** `portedForJetson/` is a stronger generalisation axis
   than the MICRO'24 paper's own A100 -> H100, because shared LPDDR changes the
   results qualitatively rather than quantitatively.

## Suggested order of work

0. **`-ub 640` — already validated, 90 ms/request, one flag.** See
   `GAP3_PREFILL.md`. Do this first; it costs nothing.
1. **Gap 1** — biggest win, no accuracy cost, and the resource analysis says the
   headroom is there. Needs a pipelined `mtmd_helper_eval_chunks` (two backends
   already exist on separate streams, so the mechanism is in place).
2. **Gap 2 via quantization** — near-deterministic prediction, `llama-quantize`
   already built. Validates the roofline model at the same time.
3. **Gap 4** — largest algorithmic lever, and it reframes every pruning result in
   the project.
4. **Gap 3** — highest microarchitectural interest, hardest to execute (kernel
   tiling work).
5. **Gap 6** — small but complete; cheap to land.
