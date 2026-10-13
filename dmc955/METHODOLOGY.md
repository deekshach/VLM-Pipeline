# Characterization methodology

The procedure of record for profiling the VLM pipeline. Supersedes the ad-hoc
parts of `PROFILING_HANDOFF.md` §6 while keeping all six of its validity rules.

Last updated 2026-09-15.

---

## 0. The rule everything else serves

**A kernel name is not a kernel — and neither is `(name, grid)`.**

The same name launches at several grids that behave completely differently, and at
a *single* grid it can still be two different kernels. The discriminating key is

```
(name, grid, block, registersPerThread, sharedMemory, duration cluster)
```

Every one of those six was needed by a real case in this workload:

| discriminator | case |
|---|---|
| grid | `mul_mat_vec_f` runs at 18944 / 3584 / 512 / 152064 |
| block | `cpy_scalar` runs 64 threads, `rms_norm_f32` runs 1024 |
| registers | prefill `mul_mat_f @112` has regs 80 and regs 92 — different template |
| shared memory | decode `mul_mat_vec_f @3584` has smem 128 and 256 |
| **duration cluster** | decode grid 3584 at *identical* regs and smem is bimodal: 28.8 us (n=7040) and 144.3 us (n=3585) — q_proj vs down_proj, 25.7 MB vs 136 MB of weights. **nsys records no problem size**, so duration is the only trace-visible discriminator |

Averaging across any of these produces numbers that describe no actual launch.

---

## 1. Fix and state the operating point

The kernel mix is **workload-dependent**, so a measurement without its operating
point is not quotable:

| frames | ViT shapes | prefill shapes | decode shapes |
|---|---|---|---|
| 1 | 28 | 75 | 15 |
| 8 | 31 | 110 | 16 |
| 16 | 27 | 131 | 15 |

Prefill fragments with context (ubatch splits × attention grids tracking KV
depth). Worse, **decode's attention changes kernel family**: `flash_attn_ext_vec
@756` at f8 and f13, but `flash_attn_ext_f16<128,128,1,8> @128` plus a stream-K
fixup at f16 — same build, same model. The dispatch gates on
`K->ne[1] % FATTN_KQ_STRIDE == 0` (`fattn.cu:368`, stride 256).

So ncu targets **do not transfer across frame counts**. Re-plan per operating point.

The current point: `alley_1`, 8 frames (4,736 visual tokens), 7B BF16 + mmproj,
`-c 16384`, `-ub 512` (stock default — the baseline is stock llama.cpp),
`-fa on`, `--temp 0 --seed 42`, `--ignore-eos`.

`--ignore-eos` is load-bearing, not cosmetic: BF16 is numerically
non-deterministic and generation length otherwise varies 9-128 tokens for
identical input, which makes decode timing meaningless.

---

## 2. Stage attribution — GPU-side, from nsys

`stage_attrib.py` owns **the single definition of the phase boundaries**; every
tool slicing a trace comes through `phase_windows()` so the definitions cannot
drift.

- **ViT** = the clip stream inside an encode window.
- **prefill** = the LLM stream from the encode start **until decode begins** —
  *not* until the last ViT kernel. The final frame's prefill runs after the vision
  tower finishes; cutting there silently drops ~8% of the stage.
- **decode** = the LLM stream from the first decode GEMV onward.

Two rules learned the hard way:

- **Never hardcode the stream number.** The clip backend sits on stream 17 with
  CUDA graphs on and **stream 16 with them off**. Streams are detected by kernel
  identity — ViT is whichever carries `rope_vision`, LLM whichever carries
  `mul_mat_vec_f`.
- **Do not split stages by host log lines.** `bench_serving_summarize.py` parses
  `"image slice encoded in N ms"` / `"image decoded (batch…) in N ms"`, which are
  wall-clock spans around *synchronous* calls. They are correct only while the
  phases are serial — the moment ViT and prefill overlap (the 578 ms
  optimisation), they stop summing to `encode_ms` and `residual_ms` goes negative.
  The measurement that would evaluate the optimisation is invalidated by it.

Use `bench_serving.sh` for wall-clock A/B (no profiler attached) and `measure.sh`
→ `stage_attrib.py` for stage questions. Never quote wall time from a traced run.

---

## 3. Kernel census — the full enumeration

`kernel_census.py <trace>` lists every launch shape per stage with grid, block,
registers, shared memory, invocations/request, ms/request, share and **cumulative**
share, so "what covers 90% of this stage" is read straight off the table. Duration
clusters are split and flagged.

This replaces `ncu_targets.py shapes`, which is welded to one trace
(`FRAME_KERNELS = 1270`, `N_FRAMES = 13` hardcoded), truncates at `--top 14`, and
never reports block size.

---

## 4. ncu target selection — from the census, not by hand

Walk each stage's time-ordered shape list down to a coverage goal. **Do not guess
which kernel "represents" a stage.**

The hand-written list in `run_ncu_survey.sh` missed **17.0% of the ViT stage**, and
three of the six gaps were *the same kernel name at a different grid* than the one
profiled (`k_bin_bcast @33152`, `convert_unary @31635` twice). `rms_norm_f32
@2368` (3.41%) and `cpy_scalar @47360` (3.35%) were never sampled at all.

Mechanics:

- **ViT vs prefill are separated by CUDA stream** (`--kernel-id ':17:regex:…:'`),
  because they share kernel names — `convert_unary`, `k_bin_bcast`,
  `unary_gated_op_kernel`, `rms_norm_f32` all appear in both.
- **prefill vs decode are separated by name**, verified phase-exclusive
  (`mul_mat_f` 0 in decode, `mul_mat_vec_f` 0 in prefill, head dim 80 = ViT vs
  128 = LLM).
- **ncu forbids `--launch-skip` with a stream-filtered `--kernel-id`.** Targets
  needing `-s` must use plain `-k regex:` on a phase-exclusive name.
- **One window can serve several grids of one name** — ncu `-k` matches by name,
  so a mixed window costs one run instead of several *and* supplies the per-grid
  consistency check.
- **Anything with real drift is sampled as a curve, never a point.** Prefill
  attention spans 24.4-201.5 us at one grid at f8; a single sample is meaningless.

---

## 5. The metric set — and why each column exists

This is the part that changed most. The old table was `dur / DRAM% / SM% / L2% /
occ% / waves`, and two of those six were actively misleading.

| column | metric | why |
|---|---|---|
| dur | Duration | — |
| **DRAM%** | DRAM Throughput | the DRAM wall |
| **L2%** | **L2 Cache Throughput** | **the column that was missing.** The old "L2%" was *hit rate* |
| L2 hit | L2 Hit Rate | keep it, but it is not a saturation measure |
| SM% | Compute (SM) Throughput | aggregates *all* pipes — understates a tensor GEMM |
| **tensor%** | from the `HighPipeUtilization` rule | the honest compute number for a GEMM |
| **elig** | Eligible Warps Per Scheduler | distinguishes *saturated* from *cannot issue* |
| **issue%** | Issue Slots Busy | same |
| occ / th | Achieved / Theoretical Occupancy | achieved == theoretical means a *ceiling*, not a tuning failure |
| **limiter** | min of Block Limit Registers / Shared Mem / Warps / SM | names the capping resource |
| waves | Waves Per SM | <1.0 means underfilled; every throughput figure is then meaningless |

**Why L2 throughput is the decisive addition.** A high hit rate is what *makes*
L2 the binding unit: traffic that would have gone to DRAM is served by a unit with
finite bandwidth. If L2 peak is ~2.5x DRAM peak, **L2 binds whenever hit rate >
~60%** — and that threshold correctly partitions every read-reuse kernel measured
so far (ViT GEMMs 95-97% hit → L2-bound; decode GEMV 1-25% hit → DRAM-bound).

`ncu_bottleneck.py` reads all of these and names the binding unit per launch:
`SATURATED on X (>=80%)`, `X-leaning (>=60%)`, `LATENCY-BOUND (nothing >60%)`, or
`UNDERFILLED (<1 wave)` — which is reported first, since a partially-filled grid
makes every other figure meaningless.

Also read ncu's own `SOLBottleneck` / `CPIStall` / `TheoreticalOccupancy` rule
prose. It is in the CSV and it named the L2 bottleneck on the prefill GEMMs all
along.

---

## 6. Validity rules (all six carried forward, one added)

1. **Report per launch, never the mean.** The `ID` column is the launch index.
2. **Launch consistency IS the validity check.** Launches of one shape agreeing
   within a few percent is evidence; a wide spread means the sample is incoherent
   and must not be quoted. `vit__k_bin_bcast` in the old survey sampled
   36.5 / 19.8 / 20.4 us — a 65% spread straddling two duration clusters — and the
   MANIFEST then quoted launch 0 as representative.
3. **State stage coverage, computed per launch shape.** Summing a *name's* whole
   share onto the one grid sampled is how 82% got reported as 91%.
4. **`-c N` takes the FIRST N matching launches**, unrepresentative wherever cost
   grows with context. Use a positioned `-s` there.
5. **Percentiles, not min/max.** p10-p90, not the outliers.
6. **Re-plan after any build change** — and (new) **after any operating-point
   change**, per §1.
7. **(new) State the replay defaults.** ncu defaults to `--clock-control base`
   (clocks locked, so ncu durations are *not* comparable to nsys — take time
   shares from nsys only) and `--cache-control all` (L1/L2 flushed per replay
   pass, so hit rates are **intra-kernel reuse only**; cross-kernel reuse is
   excluded by construction). Measured 2026-09-16: without the flush, DRAM reads
   fall 45-67% for the GEMMs and attention, but their **duration does not change**
   — keep `all` as the reproducible default for comparison, and quote a no-flush
   number alongside any claim about absolute bytes or time. If you do run
   `--cache-control none`, use a metric set that fits in **one pass**: with
   multi-pass replay, passes 2+ see the cache warmed by pass 1 of the same kernel.

---

## 7. Tool map

| tool | question |
|---|---|
| `bench_serving.sh` | wall-clock latency, no profiler |
| `measure.sh` | one config → nsys trace + stage table + hw counters, one CSV row |
| `stage_attrib.py` | ViT/prefill/decode ms, overlap, GPU_METRICS per phase |
| `kernel_census.py` | every launch shape per stage, share + cumulative |
| `run_ncu_f8.sh` | the ncu survey at this operating point |
| `ncu_bottleneck.py` | per-launch limiter verdict from the reports |
| `plot_framesweep.py` | static SVG/PNG figures |

Deprecated: `ncu_targets.py` (hardcoded to the 13-frame trace), `ncu_report.py`
(globs `ncu_*.csv`, matches nothing, prints 0% coverage silently; its `mem%` field
holds GB/s labelled as a percent).

---

## 8. Whole-phase counters — free, and no coverage caveat

nsys `--gpu-metrics-device=0` records hardware counters across the entire run:
`SMs Active`, `SM Issue`, `Tensor Active`, `DRAM read/write`, `Compute Warps in
Flight`, `Unallocated Warps in Active SMs`, PCIe. Sampled ~21.5k times per phase,
100% coverage, independent of ncu's target selection.

Use it to sanity-check every ncu stage-level claim. It is how the ViT DRAM figure
was corrected from 16.17% (GEMMs only) to 43.8% (whole phase) — and the
ncu-weighted estimate then agreed to 1.6 points.

Caveat: `GPC Clock Frequency` (metricId 0) exports as garbage/negative on this
box. Ignore it.
