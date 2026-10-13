# Microarchitectural optimization plan — lossless only

Scope: **no pruning, no quantization** (both need accuracy evaluation). Everything
here is bit-identical work reorganised, or a kernel launch configuration change —
so the only validation needed is performance, not quality.

Derivations in `BOTTLENECK_FUNDAMENTALS.md`. Written 2026-09-15.

---

## The frame of reference

The baseline is **stock llama.cpp at its defaults** (`-ub 512`, graphs on, fa on).
That is deliberate: the question is how much change stock needs to become optimal,
so tuning the baseline first would hide part of the gap being measured.

Rank the work by how much of llama.cpp it touches — that ladder *is* the result:

| rung | what changes | cost |
|---|---|---|
| 0 | nothing — stock defaults | the baseline |
| 1 | **one CLI flag** (`-ub >= 592`) | zero code — **90 ms/request, measured** |
| 2 | small contained patches (graph-cache key, fattn `occupancy` table) | ~10 lines each |
| 3 | real engineering (ViT/prefill pipelining, separator fusion) | days |

Rung 1 is the strongest single finding available: the engine ships misconfigured
for this workload class, and upstream knows — `mtmd-helper.cpp:275` reads
*"TODO @ngxson: ... n_ubatch must be enough to hold the image"*.

## How the system actually gets faster — three mechanisms, in dependency order

Revised 2026-09-16 with measured realization ratios. The items are **not
independent**, so the order matters and the totals are not a sum.

### M1 — Stop doing work that should not happen  (chunking)

Prefill runs **39 LLM graph evaluations for 7,749 tokens**; 26 of them are below
the 164 FLOP/byte ridge, and each re-streams the full 15.24 GB weight set. Modelled
excess **383 ms**. We fixed half of it and learned the discount:

```
-ub 640    predicted 157 ms  ->  MEASURED 90 ms     realization ratio 0.57
separator fusion  212 ms x 0.57  ->  ~122 ms        (conservative: separators are
                                                     largely serial, so likely more)
```

**M1 ≈ 212 ms**, and it shrinks prefill busy from 1271 → 1060 ms, which changes
what M2 can achieve.

### M2 — Overlap what remains  (pipelining, then co-scheduling)

*Within a request*: ViT 665 and prefill 1060 are independent per frame but overlap
only 13-14%. Perfect pipelining gives `max(665, 1060)` = 1060 ms encode.

The ceiling is **L2, not scheduling**: ViT ~55% + prefill ~60% of L2 peak = ~115%.
Discounting to 55% realization: **encode 1638 → 1320 ms, a further 318 ms.**

*Across requests*: co-schedule encode(N+1) with decode(N). Resource-time bound:

| | serial | co-run bound | ceiling |
|---|---|---|---|
| baseline | 4003 ms | 2735 ms | **1.46×** (DRAM) |
| after M1+M2 | 3473 ms | 2503 ms | **1.39×** (DRAM) |

Note the interaction: **M1 and M2 partially compete.** Shrinking the request leaves
less encode work to hide behind decode, so the co-scheduling multiplier falls.
Do not add these gains naively.

### M3 — Do the remaining work better  (kernels)

| item | prize |
|---|---|
| CUDA graph latch (decode gap 0.64 µs → 0.06 µs) | **51 ms** |
| fattn `occupancy` retune, ~199 ms addressable | ~60 ms at a 30% win |

Deliberately last: overlapping redundant work is wasteful, and tuning kernels that
should not run at all is worse.

### Combined, by objective

| objective | baseline | after M1+M2+M3 | factor |
|---|---|---|---|
| **latency** (f13 request) | 4,101 ms | **~3,461 ms** | **1.18×** (15.6% faster) |
| **throughput** (server) | — | 1.18× × ~1.4× co-scheduling | **~1.66×** |
| **tensor utilisation** | 12.8% request-weighted | ~18% under co-scheduling | — |

**That is the honest lossless ceiling.** It is modest because **decode is 64% of
the request at a hard physical wall** — 88-97% of DRAM peak, roofline matched to
0.5%, zero tensor use, and nothing in M1-M3 touches it.

### The lever outside the lossless scope

Decode time *is* bytes ÷ bandwidth, so quantization is near-deterministic:
**q8_0 ~1.9×, q4_K_M ~3.4× on 64% of the request** — larger than everything above
combined. And it *composes*: relieving DRAM raises the co-scheduling ceiling from
1.46× to **1.98×**, with L2 binding instead. Quantization and co-scheduling are
multiplicative, not alternatives. Recorded here so the boundary of the lossless
result is explicit; it needs accuracy evaluation before it counts.

---

## The lossless budget

| item | prize | effort | status |
|---|---|---|---|
| `-ub >= 592` (T1a) | **90 ms/request** | one flag | **validated** |
| ViT/prefill pipelining (T4) | **578 ms/request** | days | not started |
| separator chunk fusion (T1b) | **214 ms/request** | days | not started |
| CUDA graph latch (§4b) | **51 ms/request** | hours | diagnosed |
| fattn `occupancy` retune (T5) | ~199 ms addressable | hours | not started |
| phase co-scheduling (T3) | **1.46x throughput** | zero code to *test* | not started |
| ~~L2 residency control~~ | **none** | ruled out 2026-09-16 | closed |

Serial latency items total **933 ms of a 4101 ms request = 22.7%**, before any
co-scheduling gain. None of it touches model outputs.

---

## Step 0 — fix the harness first (half a day)

Do this before anything else. Two problems will otherwise bite you.

### 0a. You cannot lock clocks

```
$ nvidia-smi -lgc 2100,2100
The current user does not have permission to change clocks for GPU 0
```

So drift control is procedural: `--warmup-iters` or discard round 0, and always
run >= 3 rounds. **Your measured noise floor is good** — from the `-ub` A/B,
steady-state rounds repeat to ~1 ms within a process and ~0.3% (~6 ms on encode)
across processes. So **effects above ~15 ms are detectable**; anything smaller
needs more rounds or an nsys trace instead of wall clock.

### 0b. Your stage attribution breaks exactly when pipelining works

`bench_serving_summarize.py` slices ViT vs prefill by parsing mtmd's host-side log
lines:

```python
RE_VIT     = r"image slice encoded in (\d+) ms"
RE_PREFILL = r"image decoded \(batch \d+/\d+\) in (\d+) ms"
```

Those are wall-clock spans around *synchronous* calls. The moment ViT and prefill
overlap (T4 — the biggest item on the list), the two spans overlap too, they stop
summing to `encode_ms`, and `residual_ms` goes negative. **The measurement you
would use to evaluate the optimization is invalidated by the optimization.**

Fix it now, while the baseline is still serial and you can cross-check: attribute
stages **GPU-side from the nsys trace** — stream 17 = ViT, stream 15 = LLM, decode
begins at the first `mul_mat_vec_f`. That definition survives pipelining. You
already have the query; it is the `gaps()`/phase-window code used throughout
`BOTTLENECK_FUNDAMENTALS.md`.

### 0c. One command per configuration — BUILT

Both tools exist now and are validated:

- **`stage_attrib.py <trace>`** — GPU-side ViT/prefill/decode attribution from an
  nsys trace. Detects the ViT and LLM streams **by kernel identity**
  (`rope_vision` / `mul_mat_vec_f`), never by stream number, because the clip
  backend moves from stream 17 to 16 when CUDA graphs are disabled — a hardcoded
  17 silently reports zero ViT time. Reports overlap as a first-class number.
- **`measure.sh <label> [-- extra args]`** — runs one config under nsys, calls
  `stage_attrib.py`, appends a row to `results/experiments.csv` tagged with the
  git SHA.

Validated against the existing characterization: on `handshake_7B` it reproduces
ViT 664.9/676.6 ms, prefill 1271.5/1282.4, overlap 86.7 (13.0%), decode 2153.1,
request 4101.1 — matching the hand-derived numbers and the published 4101 ms.
Cross-checked on the graphs-off trace where ViT sits on stream 16.

The hardware counters matter as much as the timings: they tell you *which resource
moved*, which is what makes a result microarchitectural rather than anecdotal.

**Which tool for which question:**

| question | tool |
|---|---|
| is A faster than B (wall clock) | `bench_serving.sh` — no profiler attached |
| where did the time go, per stage | `measure.sh` -> `stage_attrib.py` |
| did ViT/prefill overlap improve | `measure.sh` (the host-log split cannot answer this) |
| which resource moved | `measure.sh` (GPU_METRICS columns) |

Never quote wall-clock from `measure.sh` — nsys adds tracing overhead.

---

## Step 1 — three things to run today, zero code

Each answers one decision. Run all three before writing any code.

### 1a. `-ub` sweep — extends a validated result into a curve

```bash
for UB in 512 592 640 1024 2048; do
  measure.sh ub$UB -- -ub $UB
done
```

**Predicted**: 512 -> 640 gives the known 90 ms. Beyond 592 the remainder split is
already gone, so further gains must come from the GEMM itself getting more
efficient at larger M (AI rises 438 -> 1219). **Watch `Tensor Active` in encode**:
if it climbs with `-ub`, larger tiles genuinely help and T5 is worth pursuing; if
it is flat, the GEMM is L2-limited regardless of M and T5 is a dead end.

That is the cheapest possible test of T5's premise.

Also check memory: larger ubatch grows the compute buffer. Note any `-ub` that
fails to allocate at `-c 16384`.

### 1b. Two concurrent processes — tests the 1.46x co-scheduling bound

Zero code. Run two `llama-baseline-vlm` serving processes at once on different
videos and compare aggregate throughput to 2x serial.

**Predicted** (`BOTTLENECK_FUNDAMENTALS.md` T3): DRAM pins near 100% and aggregate
throughput lands **at or below 1.46x**, not 2x. If you measure materially above
1.46x, the resource-time model is wrong and I want to know why. If you measure
~1.0x, something else serialises (context switching, memory pressure) and that is
itself a finding.

Caveat: two processes each load a 14.19 GB model into 24 GB of VRAM — that will
not fit. Use two *sequences in one process* (`llama-batched-bench` is built), or
the 3B model for this test only, and say which.

### 1c. `--cache-control none` re-profile — gates T2 entirely

```bash
ncu --set full --cache-control none --clock-control base \
    --kernel-id ':17:regex:cutlass_80_tensorop_bf16_s16816gemm_bf16:' -c 4 ...
ncu --set full --cache-control none --clock-control base \
    --kernel-id ':17:regex:unary_gated_op_kernel:' -c 4 ...
```

Compare L2 hit rate against the existing flushed runs (97.0% and 33.5%). The
**difference is the cross-kernel reuse**, which every current L2 number excludes by
construction.

- If the GEMM's hit rate rises materially -> there is a working set worth
  protecting, L2 pinning (49.5 MB pinnable) is live, T2 proceeds.
- If it barely moves -> reuse is intra-kernel only, pinning has nothing to pin,
  and T2 dies here. **That is a good outcome too** — it saves you weeks.

---

## Step 2 — contained code changes

### 2a. CUDA graph latch — 51 ms, hours, self-contained

Fully diagnosed in `PROFILING_HANDOFF.md` §4b. The graph cache is keyed on
`cgraph->nodes[0]` (`ggml-cuda.cu:2979`), which is a fixed address in a reused
buffer, so prefill and decode share one slot **and one sticky
`disable_due_to_too_many_updates` latch** (`common.cuh:1162`). VLM prefill trips it
after 4 evaluations and nothing ever clears it, so the shape-stable 2.15 s decode
runs entirely ungraphed.

Two candidate fixes, both small:
- give the cache a key that separates shapes (e.g. include `n_nodes` or the ubatch
  token count), so decode gets its own entry and its own latch; or
- clear the latch when the graph shape goes stable again.

**Predicted**: ~51 ms/request (21 ms prefill + 30 ms decode), and the decode median
inter-kernel gap drops 0.64 us -> ~0.06 us. That gap is the direct check — measure
it in the trace, not just the wall clock, because 51 ms is only 3x your noise
floor.

Good first task: contained, mechanised, and the verification is unambiguous.

### 2b. fattn `occupancy` retune — hours, tests T5 cheaply

`ggml/src/ggml-cuda/fattn-mma-f16.cuh` already exposes exactly the knob the
MICRO'24 paper calls OptMT:

```c
struct fattn_mma_config {
    int nthreads;
    int occupancy;       // -> __launch_bounds__(nthreads, occupancy), line 1460
    ...
    int nstages_target;  // 2 == cp.async preload (already max)
};
```

Current entries are `occupancy = 2` for both (80,80) and (128,128), and
`static_assert(occupancy <= 8)` permits more. **There is no Ada table** — sm_89
falls through to `ggml_cuda_fattn_mma_get_config_ampere`, and the file carries a
`// TODO tune specifically for Volta`, so per-arch retuning is accepted practice
here that simply has not been done for your GPU.

Sweep `occupancy` over {2,3,4} for those two shapes, rebuild, measure. Targets:
ViT `flash_attn_ext_f16<80>` (15.8% of ViT, and **the only latency-bound kernel in
the pipeline** — ncu says both compute *and* bandwidth below 60%) and prefill
`flash_attn_ext_f16<128>` (7.4% of prefill, L2-saturating at 85%).

Addressable ~199 ms/request. Note cuBLAS GEMMs are **not** reachable this way —
closed binary — so this caps out around 5% of the request. Do it because it is
cheap and it is the paper's technique in llama.cpp's own idiom, not because it is
the big win.

---

## Step 3 — the real engineering

### 3a. ViT/prefill pipelining — 578 ms, the largest prize

`mtmd_helper_eval_chunks` (`mtmd-helper.cpp:393`) is a sequential loop over chunks
calling a synchronous `eval_chunk_single(..., n_past, &n_past)`. Measured overlap
is **86.6 ms of 664.9 ms of ViT = 13%**.

The dependency is **false**: frame N+1's ViT does not need frame N's prefill, and
under M-RoPE `n_past` advances by `max(nx,ny)` = 37 per frame *independent of
content*, so every frame's positions are computable up front.

Shape of the change: pre-compute all chunk positions, then software-pipeline —
encode frame N+1 on the clip backend while the LLM prefills frame N. Both backends
already run on separate CUDA streams (17 and 15), so the mechanism exists; what is
missing is that the host blocks on each in turn.

Sequence I would follow:
1. Pre-compute the `n_past` schedule for all chunks; assert it matches the current
   sequential values exactly. **Land this alone first** — it is a no-op refactor
   that removes the false dependency and is independently verifiable.
2. Double-buffer the image embedding output so frame N+1 can encode while frame N
   prefills.
3. Overlap, then measure GPU-side overlap in the trace (Step 0b).

**Predicted**: overlap rises from 13%; the ceiling is `max(665, 1271)` = 1271 ms
against today's 1850. Do **not** expect all 578 ms — ViT and prefill are both
L2-leaning (~54.8% and ~59.5%), so L2 becomes the shared limit above ~100%. A
realistic first target is 250-350 ms. Watch `L2%` in the encode window: if it pins
near saturation, you have found the real ceiling and that is a publishable curve.

### 3b. Separator chunk fusion — 214 ms

12 x 2-token `<|vision_end|><|vision_start|>` chunks, each running the full 28-layer
stack and streaming all 15.24 GB of weights: **8,930 us/token, 79x the marginal
rate**.

The obstacle is structural: `llama_batch` carries `token` **or** `embd`, not both,
so a separator cannot share a ubatch with image embeddings. The workable route is
to look up the separator tokens' embedding rows and concatenate them into the
image `embd` batch, making one 594-token ubatch instead of 2 + 592.

Do this **after** 3a — pipelining changes the chunk loop you would be modifying,
and doing both at once makes attribution impossible.

---

## Ordering, and why

```
Step 0  harness            (half day)   -- everything else depends on it
  |
Step 1  1a -ub sweep       (1 hour)     -- extends a validated win; also tests T5's premise
        1b co-scheduling   (2 hours)    -- tests the 1.46x bound, zero code
        1c cache-control   (2 hours)    -- lives-or-dies decision for T2
  |
Step 2  2a graph latch     (hours)      -- good first code task, unambiguous check
        2b fattn occupancy (hours)      -- cheap, and 1a already told you if it's worth it
  |
Step 3  3a pipelining      (days)       -- the 578 ms; do the no-op refactor first
        3b separator fusion(days)       -- after 3a, never concurrently
```

Step 1 costs one day total and produces three decisions, two of which can cancel
later work. That is the highest-value day available.

## Rules for every experiment

1. **Discard round 0**, run >= 3 rounds. Noise floor ~0.3% (~6 ms on encode).
2. **Report which resource moved**, not just the time. A latency win with no
   counter movement is probably noise or drift.
3. **Change one thing.** The `-ub` and pipelining changes both alter the chunk
   loop; never land them together.
4. **`--ignore-eos` always** — BF16 is non-deterministic and generation length
   otherwise varies 9-128 tokens (`PROFILING_HANDOFF.md` §4).
5. Re-splitting a batch changes floating-point reduction order, so outputs may
   differ bit-wise. That is numerical noise, not an accuracy regression — but say
   so explicitly when reporting, so nobody mistakes it for one.
