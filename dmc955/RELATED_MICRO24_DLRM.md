# Transferring the MICRO'24 DLRM methodology to the VLM pipeline

**Paper**: R. Jain, V. M. Bhasi, A. Jog, A. Sivasubramaniam, M. T. Kandemir,
C. R. Das, *"Pushing the Performance Envelope of DNN-based Recommendation Systems
Inference on GPUs"*, MICRO 2024. IEEE Xplore 10764622 · open preprint
[arXiv:2410.22249](https://arxiv.org/abs/2410.22249).

Written 2026-09-15. Cross-references `PROFILING_HANDOFF.md` (§4b, §4c, §6a),
`RESEARCH_GAPS.md`, `GAP3_PREFILL.md`.

> IEEE Xplore blocks automated fetch (HTTP 418). Everything below is from the
> arXiv preprint of the same work; if the camera-ready differs in detail,
> re-check against the Xplore PDF.

---

## 1. What they did

**Structure**: decompose DLRM inference into 4 stages (embedding, bottom MLP,
feature interaction, top MLP) -> show one stage dominates (embedding, up to **3.2x**
slowdown) -> characterize it with ncu across *input regimes* -> derive three
optimizations, each from a named counter -> ablate, combine, re-validate on
another GPU.

**Their characterization**, on A100:

| observation | counter | value |
|---|---|---|
| low occupancy | registers/thread | 74 -> theoretical occupancy **37.5%** (24 of 64 warps) |
| memory-latency-bound | long scoreboard stalls | 1 cycle (one-item) -> **18.6** (random) |
| " | issued warps/scheduler | 0.77 -> **0.24** |
| " | achieved bandwidth | 329.5 GB/s of ~2 TB/s peak -> *not* bandwidth-bound |
| cache collapse | L1 hit / L2 hit | 98.7% -> 19% / 99.5% -> **7.7%** |

**Their three optimizations**, each tied to one of those:

| technique | mechanism | motivated by | gain |
|---|---|---|---|
| **OptMT** | `-maxrregcount` 74 -> 42 regs, 24 -> 40 warps/SM | low occupancy | +53% |
| **SW prefetching** | `prefetch.global.L1` PTX, 4 buffer variants (reg/shared/local/L1), distance tuned empirically (~10) | residual long-latency stalls | +97% |
| **L2 pinning** | Ampere L2 residency control, 30 MB reserved, `prefetch.global.L2::evict_last`, top-60k hot indices found offline | power-law access distribution | +62% |
| combined | | | **+103% embedding, +77% end-to-end** |

Then re-tuned on H100, where the optimal warp count moved 40 -> 32. **That last
detail matters most for us** (see §4).

---

## 2. The methodology worth copying

Five moves. We already do three of them, arguably better.

| their move | our status |
|---|---|
| stage decomposition + dominance claim | **Done, and stronger** — 3 stages with whole-phase hardware counters at 100% coverage (`RESEARCH_GAPS.md`), not just time shares |
| latency-bound vs bandwidth-bound discriminator | **Done per kernel** — `ncu_bottleneck.py` names the binding unit for every launch shape. Ours is finer-grained: they classify a stage, we classify a launch shape |
| per-launch reporting, no averaging | **Done** — this is exactly our §6 methodology rule |
| **regime sweep to isolate the mechanism** | **MISSING — our biggest methodological gap** |
| ablation -> combination -> synergy -> second GPU | **Missing** |

### The regime sweep is the gap

Their entire argument rests on sweeping one axis (access locality: one-item ->
high-hot -> medium-hot -> low-hot -> random). The 3.2x spread across that axis is
what turns "the embedding stage is slow" into "the embedding stage is
*memory-latency-bound*, and here is the knob". A single operating point cannot
make that argument.

We have characterized **one** operating point in depth (alley_1, 13 frames, 592
tokens/frame, `-ub 512`). The VLM axes worth sweeping, cheapest first:

1. **`n_ubatch`** — 512 / 592 / 1024 / 2048. Directly varies arithmetic intensity,
   which `GAP3_PREFILL.md` shows is *the* governing quantity. Already part-done:
   `-ub 640` gave 90 ms/request.
2. **tokens per frame** (image resolution) — 296 / 592 / 1184. Moves ViT cost,
   prefill cost and KV footprint together. Also the honest way to state the
   2x-token-inflation result (Gap 4).
3. **frame count** — 5 / 8 / 10 / 13. Partly done; isolates prefill's
   super-linearity from ViT's linearity.
4. **KV cache dtype** — f16 / q8_0 / q4_0. Moves attention's L2 working set
   without touching anything else. This is the closest analogue to *their*
   locality axis, and it is the one that would let us make a claim about the L2
   wall rather than just report it.
5. **weight quantization** — BF16 / q8_0 / q4_K_M. Validates the decode roofline
   model (predicts 1.9x / 3.4x).

Axis 4 is the one I would prioritise for a paper: it is the only sweep that varies
the L2 pressure we identified in §4c while holding compute constant.

---

## 3. Do their three techniques transfer? Honestly: partially

This is where I would push back on a straight copy. Their techniques are all
**latency-hiding**, and their bottleneck was latency-bound (329 GB/s of 2 TB/s).
Our dominant bottleneck is not.

### 3.1 OptMT — applies, but only to ~5% of the request

Our occupancy numbers are *worse* than theirs (they had 37.5%; we have 16.7% and
8.3%), so the observation transfers. The lever mostly does not:

| kernel family | share | occupancy | limiter | OptMT applicable? |
|---|---|---|---|---|
| cuBLAS `ampere_*` / `cutlass_*` | 38% prefill, 46% ViT | 8.3-16.7% | registers / shared mem | **No — closed binary.** Cannot recompile cuBLAS |
| ggml `flash_attn_ext_f16<80>` (ViT) | 15.8% of ViT | 25.0% | registers (168) | **Yes** |
| ggml `flash_attn_ext_f16<128>` (prefill) | 7.4% of prefill | 16.7% | registers (212) | **Yes** |
| ggml `mul_mat_f` | 15% prefill | 29.2% | registers (92) | Pointless — 93.7% DRAM; delete the calls instead (`GAP3_PREFILL.md`) |
| ggml `mul_mat_vec_f` (decode) | 92.8% decode | **91-97%** | warps | **No — already maximal**, and DRAM-bound |

Addressable share: ~105 ms (ViT attn) + ~94 ms (prefill attn) = **199 ms/request,
4.9%**. A 50% win there is 2.4% end-to-end. Real, but not a paper's headline.

**The good news — the knob already exists and is untuned for our GPU.**
`ggml/src/ggml-cuda/fattn-mma-f16.cuh:10-24` defines:

```c
struct fattn_mma_config {
    int  nthreads;        // threads per block
    int  occupancy;       // <-- OptMT: feeds __launch_bounds__(nthreads, occupancy)
    int  nbatch_fa;
    int  nbatch_K2, nbatch_V2, nbatch_combine;
    int  nstages_target;  // <-- prefetching: 2 == cp.async preload
    bool Q_in_reg;        // <-- register residency
};
```

fed to `__launch_bounds__` at line 1460. Current entries (lines 44-47, 59-62):

```
CONFIG_CASE( 80,  80, ncols, nthreads=128, occupancy=2, ...)   // ViT
CONFIG_CASE(128, 128, ncols, nthreads=128, occupancy=2, ...)   // prefill / decode
```

`occupancy = 2` means "fit >= 2 blocks of 128 threads per SM" = 8 warps = 16.7%,
and `static_assert(occupancy <= 8)` — so the table permits up to 4x more. ncu
confirms the effect exactly: prefill hd128 sits at 212 regs, 2 blocks/SM, 16.7%.

Decisively: the tables are `_ampere`, `_turing`, `_volta`, `_rdna` — **there is no
Ada table, and sm_89 falls through to the Ampere one.** The file even carries
`// TODO tune specifically for Volta`, so per-arch retuning is an accepted practice
here that simply has not been done for Ada. That is precisely the phenomenon the
paper demonstrated when A100's optimal 40 warps became 32 on H100.

**Experiment**: sweep `occupancy` over {2,3,4} for (80,80) and (128,128), rebuild,
measure ViT/prefill attention. It is a one-line-per-entry change, it is *the*
paper's OptMT in llama.cpp's own idiom, and it is unexplored.

### 3.2 Software prefetching — narrow applicability

Their motivation was long-scoreboard stalls in a gather-reduce. We have the same
stall class in decode — ncu `CPIStall`: *"71.0 cycles stalled waiting for a
scoreboard dependency on a L1TEX operation"* — but **decode is at 89-97% of DRAM
peak.** You cannot prefetch your way out of a bandwidth wall; the data is already
arriving as fast as the bus allows. Prefetching decode would be a category error,
and saying so is worth a paragraph in a paper.

The one kernel matching their diagnosis is **ViT `flash_attn_ext_f16<80>`**, where
ncu's own `SOLBottleneck` reads *"low compute throughput **and** memory bandwidth
utilization ... below 60% of peak"* — i.e. genuinely latency-bound, the only such
kernel in our pipeline. 15.8% of ViT.

And again the knob exists: `nstages_target` is already 2 (the maximum, meaning
cp.async preload is on). So llama.cpp is *already* doing their prefetching here.
The open question is the *distance* analogue — `nbatch_fa` (128 for ncols=8, 64
otherwise), which sets how many KV rows are staged per iteration. That is the
tunable, not whether to prefetch at all.

### 3.3 L2 pinning — the most interesting transfer, with a prerequisite

Hardware support confirmed on our GPU (queried directly):

```
NVIDIA GeForce RTX 4090 (sm_89)
l2CacheSize                : 72.0 MB
persistingL2CacheMaxSize   : 49.5 MB   <-- pinnable
accessPolicyMaxWindowSize  : 128.0 MB
```

49.5 MB pinnable vs their 30 MB on A100. And we have an L2 *bandwidth* wall
(§4c): ViT GEMMs 63-73% of L2 peak, prefill GEMMs 69-74%, prefill attention
**saturating to 85%** at context depth.

The motivating observation is already in the survey: the ViT's DRAM-saturated
elementwise kernels (`silu_gated` 94% DRAM, `k_bin_bcast` 91%, `convert_unary`
81-89%) run at **~33% L2 hit rate while the GEMMs beside them reach 97%** — the
original MANIFEST hypothesised they were evicting the GEMMs' working set. That is
exactly the situation L2 residency control exists for, and there are two framings:

- **pin** the reused set (`evict_last` on the KV / GEMM operands), or
- **stream** the polluters (`cudaAccessPropertyStreaming` / `evict_first` on the
  elementwise kernels' outputs, which are written once and consumed once).

I would try the second first: it is a smaller change, and the elementwise kernels
are ggml's own.

**Sizing sanity check** — the KV working set actually fits:

```
KV per token per layer = 2 x 4 heads x 128 dim x 2 B  = 2 KB
one layer, 13 frames   = 7,696 tok x 2 KB             = 15.8 MB   (fits in 49.5 MB)
all 28 layers                                          = 431 MB   (does not)
```

Since attention runs layer by layer, a per-layer 15.8 MB window is the natural
pin granularity, and it fits comfortably.

**The prerequisite, and it is a real one**: our ncu runs used the default
`--cache-control all`, which **flushes L1/L2 before every replay pass**. So the
97%-vs-33% hit-rate contrast is *intra-kernel* reuse only, and the cross-kernel
eviction hypothesis that motivates pinning **is not verifiable from the data we
have** (`PROFILING_HANDOFF.md` §6a). Before building anything:

> Re-profile the ViT GEMM and one elementwise kernel with
> `--cache-control none`, and compare L2 hit rate against the flushed run. That
> difference *is* the cross-kernel reuse, and its size is the ceiling on what
> pinning can recover.

That is one afternoon and it decides whether §3.3 is a contribution or a dead end.

---

## 4. Where our problem is structurally different — and why that is the better paper

Their headline is a **kernel inefficiency**: a latency-bound embedding kernel at
37.5% occupancy, fixable in-kernel for +103%.

Ours is two **hard walls plus two scheduling pathologies**:

| | our finding | is a PTX-level fix available? |
|---|---|---|
| decode, 52.5% of request | 1 FLOP/byte vs a 164 FLOP/byte ridge; 89-97% of DRAM peak; **Tensor Active 0.0%** | **No.** Only fewer bytes (quantization) or higher intensity (batching) |
| ViT+prefill GEMMs | L2-bound at 63-74%, occupancy capped by tile shape in closed cuBLAS binaries | Not without replacing the GEMM |
| **39 LLM evals for 7,749 tokens** | 383 ms/request excess; 26 of 39 ubatches below the ridge | **No — it is a chunking bug.** `-ub 640` already recovers 90 ms |
| **ViT/prefill overlap 13%** | 578 ms/request; they are independent per frame but run in lockstep | **No — it is a scheduling bug** |

If we copied their three techniques verbatim we would be optimising **4.9%** of
the request and leaving a measured **961 ms/request** (Gap 1 + Gap 3a, 23.4%) on
the table. The characterization says the wins are at the scheduling, chunking and
precision layer, not the PTX layer.

So the paper to write is *methodologically* theirs and *substantively* ours:

> Same rigour — stage decomposition, hardware counters, regime sweeps,
> latency-vs-bandwidth discrimination, ablation, second architecture — applied to
> a pipeline whose answer turns out to be "the engine schedules and chunks the
> work wrongly", not "the kernel is untuned".

That is a *stronger* claim than a kernel-tuning result, because it generalises
across engines, and it is the kind of finding only a characterization study
surfaces.

### Our advantage on their weakest axis

Their generalization check is A100 -> H100: same vendor, same memory technology,
one generation apart. We have `portedForJetson/`. Jetson Orin shares one LPDDR
controller between host and device, which makes several of our results change
qualitatively rather than quantitatively:

- decode's 89% DRAM wall meets a ~200 GB/s bus instead of 1008 GB/s;
- the 578 MB/encode host staging (free here at 1.9% PCIe) starts contending with
  the very DRAM decode is saturating;
- the L2 is far smaller, so the L2 wall and any pinning result move.

A 4090 -> Orin comparison is a much sharper generalization story than
A100 -> H100.

---

## 5. Concrete plan

**Phase A — close the methodology gap (no new code)**
1. `n_ubatch` sweep 512/592/1024/2048 — arithmetic-intensity axis. Extends the
   validated `-ub 640` result into a curve.
2. KV-dtype sweep f16/q8_0/q4_0 — the L2-pressure axis, our analogue of their
   locality sweep.
3. Weight-quantization sweep BF16/q8_0/q4_K_M — validates the decode roofline.
4. Re-profile two kernels with `--cache-control none` — decides whether L2
   pinning is worth pursuing at all.

**Phase B — the structural optimizations (our actual contribution)**
5. Ship `-ub >= 592`. Validated: 90 ms/request.
6. Frame-level ViT/prefill pipelining (Gap 1). Largest single prize, 578 ms
   ceiling, no accuracy cost.
7. Separator-token fusion into the image ubatch (Gap 3a). 214 ms.

**Phase C — the PTX-level techniques, as a secondary result**
8. OptMT: sweep `occupancy` {2,3,4} for fattn (80,80) and (128,128); consider a
   `ggml_cuda_fattn_mma_get_config_ada`. Directly their technique, in llama.cpp's
   own idiom, on a table never tuned for sm_89.
9. `nbatch_fa` sweep — the prefetch-distance analogue.
10. L2 streaming hints on the ViT elementwise kernels — **only if step 4 shows
    real cross-kernel reuse to protect.**

**Phase D — generalization**
11. Repeat A5/B on Jetson Orin.

Phases A and B are where the measured prizes are. Phase C is what makes it legible
to a MICRO/ISCA audience as microarchitectural work. Doing C without A is how you
end up optimising 4.9% of the request.
