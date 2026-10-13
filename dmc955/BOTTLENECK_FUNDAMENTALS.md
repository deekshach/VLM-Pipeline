# Why the VLM pipeline is slow — derived from fundamentals

Qwen2.5-VL-7B-BF16 on llama.cpp/libmtmd, RTX 4090 (sm_89), MPI-Sintel `alley_1`,
13 frames, 592 visual tokens/frame, 128 generated tokens.

This document derives the bottlenecks from first principles rather than listing
them. `RESEARCH_GAPS.md` ranks them by prize; this explains *why* each one exists
and what makes it addressable.

Sources: `profiling_results/handshake_7B.sqlite` (nsys, incl. `GPU_METRICS`
hardware counters at 100% phase coverage), `ncu_2026-09-02_7B-BF16/` (ncu, per
launch shape), plus the validated A/B in `GAP3_PREFILL.md`. Method caveats:
`PROFILING_HANDOFF.md` §6a.

Written 2026-09-15.

---

## Part 0 — Three fundamentals

Everything below follows from these three.

### F1. The roofline and the ridge point

Achievable performance is `min(peak_FLOPS, AI x bandwidth)`, where **AI** is
*arithmetic intensity* — FLOPs performed per byte moved from memory (FLOP/byte).
The two limits cross at the **ridge point**:

```
ridge = 165.2 TFLOP/s / 1008 GB/s = 164 FLOP/byte      (RTX 4090, BF16 tensor)
```

Below 164 FLOP/byte a kernel is **memory-bound**: the math units idle waiting on
data, and no amount of kernel tuning helps. Above it, **compute-bound**.

This single number organises the whole pipeline.

### F2. For a prefill GEMM, arithmetic intensity IS the token count

For `C[M,N] = A[M,K] . B[K,N]` with the weight matrix `B` dominating traffic:

```
FLOPs = 2.M.N.K          bytes ~= 2.K.N          =>   AI ~= M
```

i.e. **AI ~= the number of tokens in the ubatch**, because each weight byte is
read once and reused once per token. Verified against the real FFN shape
(K = 3584, N = 18944):

| M (tokens) | weights as % of traffic | AI | verdict |
|---|---|---|---|
| 2 | 100% | **2.0** | memory-bound |
| 80 | 97% | **77.9** | memory-bound |
| 512 | 85% | **437.7** | compute-bound (2.7x ridge) |
| 592 | 84% | 494.8 | compute-bound |
| 2048 | 60% | 1219.4 | compute-bound |

(The approximation `AI ~= M` is exact for small M and ~15% optimistic at M = 512,
where activations and output stop being negligible.)

**Consequence: `n_ubatch` is an arithmetic-intensity knob.** Any prefill ubatch
below ~164 tokens is memory-bound *by construction*.

### F3. A high cache hit rate is a bandwidth AMPLIFIER, not just a latency saver

This is the least intuitive of the three and it explains the L2 wall.

If L2 serves traffic at rate `R` with hit rate `h`, then DRAM only sees `R(1-h)`.
So L2 saturates before DRAM when:

```
1/peak_L2  >  (1-h)/peak_DRAM     <=>     (1-h) < peak_DRAM/peak_L2
```

With L2 peak ~2.5x DRAM peak on this part, **L2 binds whenever h > ~60%**:

| peak_L2 / peak_DRAM | L2 binds above hit rate |
|---|---|
| 2.0x | 50% |
| 2.5x | 60% |
| 3.0x | 67% |

Tested against every read-reuse kernel measured in the survey — it partitions all
eight correctly:

| kernel | hit rate | L2% | DRAM% | predicted | actual |
|---|---|---|---|---|---|
| ViT cutlass @1184 | 97.0 | 64.6 | 16.9 | L2 | **L2** |
| ViT ampere @370 | 95.0 | 64.0 | 19.6 | L2 | **L2** |
| prefill ampere @1184 | 91.5 | 68.8 | 32.4 | L2 | **L2** |
| prefill ampere @896 | 90.3 | 73.8 | 37.1 | L2 | **L2** |
| prefill attn (depth s900) | 98.3 | 85.0 | 9.5 | L2 | **L2** |
| decode flash_attn_vec | 85.6 | 77.8 | 49.4 | L2 | **L2** |
| decode mul_mat_vec @18944 | 1.1 | 41.5 | 95.9 | DRAM | **DRAM** |
| decode mul_mat_vec @3584 | 25.0 | 40.4 | 94.3 | DRAM | **DRAM** |

**Scope**: holds for read-reuse kernels (GEMMs, attention). The write-streaming
elementwise kernels do *not* obey it — their reported "hit rate" reflects
write-allocate behaviour, not reuse. Do not apply the rule to them.

This is the derivable mechanism behind the "two memory walls" result in §4c: it
was observed there, and is *predicted* here.

---

## Part 1 — Tier 1: genuine microarchitecture contributions

### T1. Chunk-boundary-induced arithmetic-intensity collapse

**Observation.** Prefill runs **39 LLM graph evaluations to process 7,749 tokens**:

| ubatch | evals | GPU ms | us/token |
|---|---|---|---|
| 512 | 13 | 753.4 | 113.2 |
| 80 | 13 | 274.4 | 263.8 |
| 29 | 1 | 18.4 | 633.5 |
| **2** | 12 | 214.3 | **8,930** |

(Recovered from the trace by grouping `rms_norm_f32` launches — its grid equals
the ubatch's token count — and timing stream-15 busy between each run's endpoints.)

**The fundamental.** By F2, AI = tokens/ubatch; by F1 the ridge is 164. So **26 of
the 39 evaluations sit below the ridge by construction** — memory-bound no matter
how good the kernel is. A 2-token ubatch performs 0.3 GFLOP while dragging 135.8
MB of weights: it is a decode step wearing a prefill costume, which is exactly why
it costs **79x** the marginal per-token rate.

**Why chunking causes it.** Interleaved modality forces chunk boundaries; each
boundary becomes a separate `llama_decode`, and each call re-streams the entire
15.24 GB weight set:

```
39 evaluations x 15.24 GB              = 594 GB
measured DRAM traffic during encode    = 860 GB
  -> weight re-streaming is 69% of it
```

llama.cpp's batch splitter has no notion of the ridge.

**Opportunity.** Ridge-aware packing — keep every ubatch above ~164 tokens. Two
independent causes, one already fixed:

- **`n_ubatch` = 512 against 592 tokens/frame** -> every frame splits into 512 + 80.
  mtmd submits all 592 in one `llama_decode`; llama.cpp chops it internally.
  Upstream already knows: `mtmd-helper.cpp:275` carries *"TODO @ngxson: ...
  n_ubatch must be enough to hold the image"*.
  **VALIDATED: `-ub 640` -> encode 1944 -> 1854 ms, 90 ms/request, decode unchanged.**
- **12 x 2-token separator chunks** (`<|vision_end|><|vision_start|>` between
  frames), each running all 28 layers: **214 ms/request for 24 tokens**. Cannot be
  fixed by a flag — `llama_batch` carries `token` **or** `embd`, so merging them
  into the image ubatch needs a mixed token/embedding path in `mtmd_helper`.

Total excess vs the 512-token marginal rate: **383 ms/request = 9.3%**.

**Why it generalises.** The claim is not "llama.cpp has a bug" but *the roofline
ridge is a scheduling constraint, and it is a computable per-(model, GPU)
constant*. It applies to any interleaved modality — video, audio, multi-image,
interleaved image-text documents — and the threshold is derived, not tuned.

### T2. The visual KV cache as an L2-residency design parameter

*Highest novelty per unit of effort.*

**Observation.** Prefill attention does not become "more compute-bound" with
context depth, as originally reported — it **saturates the L2**:

| context depth | duration | DRAM% | **L2%** | L2 hit | tensor% |
|---|---|---|---|---|---|
| s60 | 58.6 us | 17.0 | 67.5 | 94.7 | 42.5 |
| s320 | 140.6 us | 13.5 | 78.2 | 97.4 | 50.9 |
| s900 | 360.7 us | 9.5 | **85.0** | 98.3 | 54.7 |

DRAM *falls* 17.0 -> 9.5% as the KV grows. That inversion is the tell.

**The fundamental.** By F3, once hit rate passes ~60% the traffic has migrated to
L2 and the L2 ceiling binds. The KV cache is the ideal case for this — written
once, read many times — and it **fits**:

```
KV bytes per token per layer = 2(K,V) x 4 heads x 128 dim x 2 B  = 2,048 B
13 frames (7,696 tokens), one layer                              = 15.8 MB
L2 capacity                                                      = 72 MB
  of which pinnable (cudaDeviceProp.persistingL2CacheMaxSize)    = 49.5 MB
```

**The cliff is analytic.** L2 holds `72 MB / 2,048 B` = **35,156 tokens ~= 59
frames** at 592 tokens/frame. Beyond that, attention must fall off L2 and become
DRAM-bound. With q8_0 KV (1 B/element) the cliff moves to **119 frames**.

**Why this is the most interesting item.** The working-set size is not a fixed
hardware property — it is set by the *model's token policy*:

```
working set = tokens/frame  x  frames  x  layers  x  KV dtype bytes
```

All four are design parameters you control. Sintel clips are 20-50 frames, so this
workload sits just *below* the cliff — which makes the cliff directly testable by
sweeping frame count. It also tells the project's own pruning work what it is
buying: halving visual tokens doubles the frames held before the cliff, a
second-order benefit nobody has quantified.

**Two caveats, one of which gates the whole item.**

1. Weight streaming (135.8 MB per FFN GEMM) competes for the same L2, so effective
   KV residency is below 72 MB. The analytic cliff is an upper bound.
2. **Our measured hit rates are intra-kernel only.** ncu defaults to
   `--cache-control all`, which flushes L1/L2 before every replay pass, so
   cross-kernel reuse is excluded *by construction* — and cross-kernel reuse is
   precisely what T2 is about. **Re-profiling one GEMM and one elementwise kernel
   with `--cache-control none` decides whether T2 is real.**

### T3. Phase-level multi-resource scheduling

**Observation** — nsys `GPU_METRICS`, whole-phase, 100% coverage, both requests
agreeing to <1%:

| counter | encode | decode |
|---|---|---|
| SMs Active | 88.0% | 93.0% |
| SM Issue | 16.3% | 11.1% |
| Tensor Active | 26.9% | **0.0%** |
| DRAM (rd+wr) | 43.8% | 89.4% |
| L2 (time-weighted) | ~59.6% | ~42.6% |

**The fundamental.** A phase's bottleneck is a **vector**, not a scalar. Two
phases combine by **resource-time product** (utilisation x wall time) — *not* by
summing utilisations, because utilisations are rates over different durations and
adding them is a category error. Co-running cannot finish faster than
`max_r(total resource-ms for r)`:

| resource | encode (1850 ms) | decode (2153 ms) | total resource-ms |
|---|---|---|---|
| **DRAM** | 43.8% -> 810 | 89.4% -> 1925 | **2735  <- binds** |
| L2 | 59.6% -> 1103 | 42.6% -> 917 | 2020 |
| tensor | 26.9% -> 498 | 0.0% -> 0 | 498 |
| SM issue | 16.3% -> 302 | 11.1% -> 239 | 541 |

```
serial wall  4003 ms   ->   lower bound 2735 ms   =   1.46x max speedup
```

**CORRECTION.** An earlier draft of this analysis stated that ViT+decode "sum to
133% DRAM, therefore they collide and the gain is ~nothing." That summed
utilisations and was wrong. DRAM does become the **binding** resource, but it
still admits **1.46x**. The naive hope was 2x; the honest ceiling is 1.46x; it is
not zero. `RESEARCH_GAPS.md` T3 and `PROFILING_HANDOFF.md` §4c/§9 carry the
corrected table.

**The composition result — the genuinely interesting part.** Halve decode's bytes
with q8_0 weights and the bound moves to **1.98x**, with **L2** binding instead of
DRAM. So quantization and co-scheduling are **multiplicative, not alternatives** —
each relieves the resource that limits the other. That claim is only available
with the full resource vector, which is the point of the whole exercise.

**Falsifiable prediction for the experiment** (two concurrent inference
processes): DRAM pins near 100% and the speedup lands at or below 1.46x.

---

## Part 2 — Tier 2: strong results, shallower mechanism

### T4. Interface-enforced cross-modal serialization — 578 ms/request

ViT and prefill overlap only **13%** (86.6 ms of 664.9 ms ViT busy; union 1849.7
ms against 1936.3 ms of work). Full pipelining would give `max(665, 1271)` = 1271
ms instead of 1850 — **578 ms/request, 14.1%**, the largest raw prize in the
project.

**Mechanism.** `mtmd_helper_eval_chunks` (`mtmd-helper.cpp:393`) is a plain
sequential loop calling a synchronous `eval_chunk_single(..., n_past, &n_past)`.
The `n_past` accumulator creates a **false** data dependency: under M-RoPE,
`n_past` advances by `max(nx,ny)` = 37 per frame *independent of content*, so
every frame's positions are computable upfront. Frame N+1's ViT does not depend on
frame N's prefill.

**A stronger hypothesis, tested and rejected**: `llama_set_causal_attn` is global
context state, and toggling it around image decode *would* force serialization —
but `mtmd_decode_use_non_causal` returns true only for GEMMA3 and is **false for
Qwen2.5-VL**. So the serialization is API shape, not an architectural constraint.
That is why this sits in Tier 2 despite the largest prize.

**Headroom check** (why it is not hopeless): during encode, DRAM is 43.8%, tensor
26.9%, SM issue 16.3%, and 71.6% of warp slots are unallocated. The limit will be
L2 — ViT ~54.8% plus prefill ~59.5% exceeds 100% — so expect a substantial
fraction of 578 ms, not all of it.

### T5. Occupancy ceilings from closed-library tile selection

**The fundamental first, because the naive reading is wrong.** Latency hiding
requires concurrency = latency x throughput, and that concurrency can come from
*either* warps (thread-level parallelism) *or* independent work per warp
(instruction-level parallelism). Big-tile tensor GEMMs deliberately trade
occupancy for register blocking and deep MMA pipelining. **So 16.7% occupancy is
normal, and "low occupancy therefore slow" is bad reasoning.**

The ceilings are exact, and the two kernel families are limited by *different*
hardware resources (ncu `Block Limit *`, minimum binds):

| kernel | block | regs | blocks/SM | limiter | warps/SM | occupancy | warps/sched |
|---|---|---|---|---|---|---|---|
| prefill `ampere_64x128` @1184 | 128 | 230 | 2 | **REGISTERS** | 8 | 16.7% | 2.00 |
| prefill `ampere_128x64` @896 | 128 | 230 | 2 | **REGISTERS** | 8 | 16.7% | 2.00 |
| ViT `cutlass` @1184 | 128 | 156 | 1 | **SHARED MEM** | 4 | **8.3%** | **1.00** |

Arithmetic: 230 regs x 128 threads = 29,440; 65,536 / 29,440 = 2.22 -> 2 blocks.
The ViT kernel uses 72 KB of shared memory against ~100 KB/SM -> 1 block. Achieved
occupancy equals theoretical in every case, so these are **ceilings set by tile
shape**, not failures to reach one.

**The evidence something IS lost** is the conjunction, not the occupancy alone:

1. tensor pipe at only 48.4% while being the *highest-utilised* pipe;
2. L2 at 68.8-73.8% — ncu's `SOLBottleneck` names it the bottleneck;
3. **0.15 eligible warps per scheduler** out of 2 resident, i.e. ~92% of cycles
   nothing can issue;
4. `CPIStall`: 11.7 cycles stalled on an "oversubscribed math pipeline".

Read together: 2 warps cannot cover MMA latency *while* L2 is 69% busy feeding
operands. If the tensor pipe were the true limit it would read near 100%, not 48%.

**Opportunity and its limit.** These are cuBLAS heuristic-selected kernels for
shapes like 512 x 3584 x 18944 — nobody chose them for this shape, and cuBLAS
cannot be recompiled. Testable hypothesis: *vendor heuristics do worst on the odd
token counts multimodal models produce* (592, not a power of two). Quantifiable
standalone with the CUTLASS profiler, no llama.cpp changes needed to measure the
gap.

### T6. M-RoPE forecloses token-granular KV reclamation

All image tokens in a frame share one temporal position under M-RoPE, so
`llama_memory_seq_rm` can only evict at **frame** granularity, never per token.

Consequence for the project's own pruning work: you can prune visual-token
**compute** but not the **KV footprint**. Combined with T2 this is sharp — pruning
cannot move you back from the L2 residency cliff once crossed, it can only reduce
the work done before reaching it. A clean architectural negative result that bounds
what token pruning can deliver.

---

## Part 3 — The single gating measurement

Re-profile one GEMM and one elementwise kernel with **`--cache-control none`** and
compare L2 hit rate against the flushed run. That difference *is* the cross-kernel
reuse.

It matters because every L2 claim in this document currently excludes cross-kernel
reuse by construction, and:

- it decides whether **T2** is a contribution or a dead end;
- it sizes the ceiling on any L2-pinning work (49.5 MB is pinnable on this part);
- it tests the survey's original hypothesis that the DRAM-saturated elementwise
  kernels (33% L2 hit) are evicting the GEMMs' working set (97% L2 hit) — a claim
  that is **not verifiable** from the data we have.

One afternoon, and it either unlocks the highest-novelty item or kills it.

---

## Appendix — the numbers this document depends on

| quantity | value | source |
|---|---|---|
| ridge point | 164 FLOP/byte | 165.2 TFLOP/s / 1008 GB/s |
| request breakdown | encode 1850 ms / decode 2153 ms / 4101 ms total | nsys, `RESEARCH_GAPS.md` |
| decode roofline check | 16.91 predicted vs 16.82 ms/token measured (100.5%) | nsys hw counters |
| prefill evaluations | 39 for 7,749 tokens | `rms_norm_f32` grid, `GAP3_PREFILL.md` |
| `-ub 640` saving | 90 ms/request, decode unchanged | validated A/B, 3 rounds |
| L2 capacity / pinnable | 72 MB / 49.5 MB | `cudaGetDeviceProperties` on the box |
| KV per token per layer | 2,048 B | 2 x 4 heads x 128 dim x 2 B |
| co-scheduling ceiling | 1.46x (DRAM-bound), 1.98x with q8_0 | resource-time product |
