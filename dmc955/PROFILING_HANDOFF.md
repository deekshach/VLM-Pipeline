# VLM Profiling — Handoff Document

Self-contained context for the microarchitectural characterization of vanilla
Qwen2.5-VL on llama.cpp. Written to be pasted into a fresh session.

Last updated 2026-09-15.

---

## 1. What this is

Characterizing where time goes in **unoptimized** VLM video inference on an edge
C++ stack, as the baseline for later optimization work (adaptive sampling, visual
token pruning, KV reuse — all deliberately out of scope here).

The subject is the *system*: Qwen2.5-VL weights on llama.cpp/libmtmd. The engine
is part of what is being characterized, not a confound.

**Verified standard**: `baseline_vlm.cpp` uses the same eight libmtmd functions as
the canonical `tools/mtmd/mtmd-cli.cpp`, with an identical `mtmd_helper_eval_chunks`
call. Nothing custom sits in the measured path.

---

## 2. Environment

| | |
|---|---|
| GPU | RTX 4090, sm_89, 128 SMs, ~1008 GB/s peak, driver 580.173.02 |
| CPU | i9-14900K — P-cores 0-15 (`cpu_core`), E-cores 16-31 (`cpu_atom`) |
| kernel | 6.8.0-138-generic |
| llama.cpp | b55dcdef5 (dirty) |
| model | Qwen2.5-VL-7B-Instruct-BF16 + mmproj-BF16 (pinned in `model_config.sh`) |
| arch | LLM 28 layers, n_embd 3584, n_head_kv 4 · ViT 32 layers, patch 14, merge 2 |
| dataset | MPI-Sintel `training/clean`, 23 videos, 20-50 frames each |

### Traps that silently produce wrong data

- **`nsys`, `ncu`, `ncu-ui` on PATH are 2022-era** (CUDA 11.8). Use
  `/usr/local/cuda-12.6/bin/` — 2024.4.2 / 2024.3.1. A 2022 `ncu-ui` cannot open
  a 2024 `.ncu-rep`. `/opt/nvidia/nsight-systems/2024.5.4` is ARM-only (Jetson).
- **`cmake` on PATH is a broken pip shim** (`ModuleNotFoundError: No module named
  'cmake'`). Use `/usr/bin/cmake`.
- **`perf` needs `sudo sysctl -w kernel.perf_event_paranoid=1`** and it **resets
  on reboot**.
- **`nsys stats` caches a `.sqlite`** and refuses to run if it is older than the
  `.nsys-rep`. Pass `--force-export=true`.
- **GPU clocks idle at 210 MHz of 3105.** `--warmup-iters` covers the ramp;
  `nvidia-smi -lgc 2100,2100` removes drift entirely.

---

## 3. The pipeline

```
frames -> PNG decode + resize (CPU)
       -> mtmd_tokenize: ONE CHUNK PER FRAME
       -> mtmd_helper_eval_chunks:  [ViT encode -> LLM prefill] x N frames, INTERLEAVED
       -> decode loop: sample -> llama_decode, 1 token at a time
```

- **Geometry**: 1024x436 -> resized 1036x448 -> 74x32 patches of 14px -> 2x2 merge
  -> **592 tokens per frame**. Confirmed in `clip.cpp:2993`: `n_patches =
  (nx/28) * (ny/28)`.
- **ViT and prefill interleave per frame**, they are not two blocks. Verified in
  both the log ordering and the nsys timeline.
- **No temporal patch merging.** llama.cpp encodes each frame as an independent
  still image — N frames give N encoder calls, 592 tokens each, no step at even N.
  Reference Qwen2-VL uses `temporal_patch_size=2`, so llama.cpp emits **2x the
  visual tokens** the reference would. State this when reporting.
- **CUDA streams**: 15 = LLM backend, 17 = clip/ViT backend. They overlap 180.7 ms
  during encode; stream 17 is **completely idle during decode**.
- **M-RoPE**: `n_past` advances by `max(nx,ny)` = 37 per frame, not by token count.
  13 frames -> n_past 538 while the KV holds 7696 visual tokens. Do not compute KV
  occupancy from `n_past`.

---

## 4. Verified results

Basis: 460 measured requests, 7B BF16, constant-rate sampling (2 fps against an
assumed 8 fps source), 13 frames for 19 of 23 videos, 128 tokens fixed-length.

### Stage latency (mean +/- 95% CI)

| stage | ms | % of request |
|---|---|---|
| ViT encode | 749.5 +/- 14.1 | 18.3% |
| LLM prefill | 775.7 +/- 15.2 | 18.9% |
| residual (tokenize, PNG, text) | 260.4 | 6.3% |
| LLM decode | 2190.3 +/- 4.6 | 53.4% |
| **request** | **4101** | 58.5 tok/s |

### Scaling with frame count

| frames | vis tokens | ViT ms/frame | prefill ms/frame | decode ms |
|---|---|---|---|---|
| 5 | 2960 | 62.4 | 61.3 | 2153.6 |
| 8 | 4736 | 62.5 | 62.6 | 2161.2 |
| 10 | 5920 | 63.0 | 64.0 | 2176.1 |
| 13 | 7696 | 62.7 | 65.1 | 2196.5 |

- **ViT is exactly linear** — content-independent to +/-0.3 ms across 23 videos.
- **Prefill is super-linear** (+6.2%) — each frame attends over all prior visual
  tokens.
- **Decode grows +2.0% with context** at fixed token count — the isolated KV effect.

### Kernel structure

| stage | launches | distinct kernels | top-1 share | kernels for 80% |
|---|---|---|---|---|
| ViT | 16,497 / request | 18 | 29.3% | 6 |
| prefill | 31,452 / request | 19 | 25.9% | 6 |
| decode | 50,911 / request | 9 | **92.8%** | **1** |

Decode: 398 kernels/token = 28 layers x ~14, dominated by `mul_mat_vec_f`.

### ncu — per-launch, grouped by grid  (see `ncu_2026-09-02_7B-BF16/MANIFEST.md`)

20 reports, 80 per-launch samples, targets chosen from the nsys (kernel,grid)
census. Coverage: ViT ~91%, prefill ~75% + attention curve, decode ~96%.

**Decode — memory wall on every shape. HIGH CONFIDENCE.**

| grid | dur us | DRAM% | SM% | L2% | occ% | share of decode |
|---|---|---|---|---|---|---|
| 18944 | 291.3 | **95.9** | 14.1 | 1.1 | 95.6 | 49.2% |
| 3584 | 149.5 | **94.5** | 11.2 | 25.1 | 91.3 | 34.6% |
| 152064 | 1.1 | **96.9** | 19.1 | 0.8 | 96.9 | 7.0% |
| 512 | 6.8 | 55.4 | 13.3 | 35.9 | 57.5 | 1.9% |

Three shapes covering 91% of decode at 94-97% of DRAM peak, agreement inside 0.2%.

**ViT — splits into two populations. HIGH CONFIDENCE (but see 4c for the
corrected verdict on the first population).**

~~Compute-bound with excellent locality~~ — **L2-bandwidth-leaning**:
`cutlass_s16816gemm @1184` 64.6% of L2 peak (tensor only 43.4%, DRAM 16.9%);
`ampere_128x64 @370` 64.0% L2 (tensor 44.2%). `flash_attn hd80 @384` is
**latency-bound** — nothing above 60%. The 94-97% figures quoted here are L2
*hit rate*, not throughput; see 4c.

Memory-bound elementwise, **~24% of the stage and as saturated as decode**:
`silu_gated` 94.2% DRAM, `k_bin_bcast` 91.2%, `convert_unary f32->bf16` 89.2%,
`convert_unary bf16->f32` 81.5% — all at ~33% L2 hit while the GEMMs beside them
reach 97%. Likely evicting the GEMMs' working set.

**Prefill — nothing is saturated. NEEDS EXPLANATION.**

| kernel | grid | DRAM% | SM% | L2% | waves |
|---|---|---|---|---|---|
| ampere_64x128 | 1184 | 32.4 | 44.9 | 91.5 | 4.62 |
| ampere_128x64 | 896 | 37.1 | 47.9 | 90.3 | 3.50 |
| cutlass_relu | 592 | 78.5 | 29.3 | 84.7 | 2.31 |
| mul_mat_f | 592 | **93.7** | 13.5 | 6.8 | 2.31 |
| ampere_128x128 | 112 | 26.8 | 41.2 | 93.0 | 0.44 |

~~The two largest (38% of prefill) sit at 45-48% SM and 32-37% DRAM — neither
limit.~~ **ANSWERED in 4c**: they are at **69-74% of L2 peak**. The table above
reports L2 *hit rate* in the column labelled L2%; L2 *throughput* was captured but
never read.

**Prefill attention — a curve, not a number.** Drifts +229% across the phase.
~~It becomes *more compute-bound* as context grows.~~ **Backwards — it saturates
the L2**: 67.5% -> 78.2% -> **85.0%** of L2 peak at depth, while tensor never
exceeds 54.7% and DRAM *falls* 17.0 -> 9.5%. See 4c. The s640 point (19.6 us) is
an outlier — flash attention launches several blocks per layer, so launch index
does not map cleanly onto context depth.

### Other measured facts

- **Dtype marshalling**: `convert_unary` f32<->bf16 is **16.3% of ViT** (more than
  ViT attention's 15.8%), 6.0% of prefill, ~0% of decode. Decode's absence proves
  it is avoidable, not intrinsic.
- **CUDA graphs: the ViT uses them for everything, the LLM is permanently latched
  off after 4 evaluations.** Not "barely used" — see section 4b. 30
  `cudaGraphLaunch` against 171,040 `cudaLaunchKernel` looks like near-zero use,
  but each graph launch replaces 1,270 kernel launches: 33,020 of the ViT's
  33,020 kernels are graph-launched, and 169,168 of the LLM's 172,360 are not.
- **CPU is not the bottleneck**: decode GPU busy 97.0%, CPU runs ~7.35 ms ahead of
  the GPU (launch->execute latency), inter-kernel gaps median 0.64 us.
- **Flash attention** (measured on 3B): disabling it costs **+77% prefill, +23% TTFT**.
  The pruning variants force it off (`streaming_vlm.cpp:3930`), so never compare an
  `fa=on` baseline against an `fa=off` variant.
- **BF16 is numerically non-deterministic**: generation length varied 9-128 tokens
  for identical input. `--ignore-eos` forces exactly `n_predict` tokens and drops
  decode sd to 2.3%. **Required** for comparable decode timing.

---

## 4b. CUDA graphs — why the LLM never gets them  (2026-09-15)

Answers the §9 open question "Why do CUDA graphs fall back?". Mechanism traced in
the source, confirmed in `handshake_7B.nsys-rep`, and verified with a controlled
A/B (`profiling_results/graphab_{on,off}.nsys-rep`).

### The finding

The two backends behave completely differently, and the aggregate counter hides it:

| backend | stream | kernels/run | graph-launched | median inter-kernel gap |
|---|---|---|---|---|
| clip / ViT | 17 | 33,020 | **33,020 (100%)** | **0.064 us** |
| LLM (prefill + decode) | 15 | 172,360 | 3,192 (1.9%, warmup only) | **0.640 us** |

The ViT's 26 graph launches each replace 1,270 kernel launches — one per frame,
13 frames x 2 requests. The LLM gets 4 graph captures during startup and then
**never uses a graph again for the rest of the process**: last stream-15
graph-launched kernel at 679 ms of an 8,863 ms trace, covering two full requests
and 101,902 decode kernel launches.

### The mechanism — a sticky latch on a single-slot cache

Three pieces of stock llama.cpp (unmodified in this tree) combine:

1. **The graph cache is keyed on `cgraph->nodes[0]`**
   (`ggml-cuda.cu:2979`). `llm_graph_result::reset`
   (`llama-graph.cpp:745`) rebuilds every LLM graph into the *same reused*
   `buf_compute_meta`, so `nodes[0]` is the same address every time. **Prefill and
   decode therefore share one `ggml_cuda_graph` slot** — and one latch. The clip
   backend has its *own* `buf_compute_meta` (`clip.cpp:147`, sized at
   `clip.cpp:2023`), so it gets a different key and a separate latch. That is the
   entire reason the ViT survives and the LLM does not — and because every Sintel
   frame is the same 1036x448 geometry, the ViT's shapes never change,
   `number_consecutive_updates` resets to 0 on every evaluation, and its latch is
   never armed.

2. **Every shape change forces a graph update.**
   `ggml_cuda_graph_update_required` compares `ne[]`, `nb[]` and `data` pointers
   per node. The mtmd eval sequence is warmup -> text prefill -> 592-token image
   prefill -> ... -> 1-token decode: consecutive evaluations essentially never
   match. (`n_nodes` alone moves 1091 -> 959; see `handshake_7B.runlog:158,198`.)

3. **Four consecutive updates disable graphs permanently.**
   `ggml_cuda_graph::record_update` (`common.cuh:1162`) sets
   `disable_due_to_too_many_updates` at `number_consecutive_updates >= 4`. There is
   **no path that ever clears it** — `is_enabled()` returns false for the life of
   the context.

So the latch trips before the second frame is even prefilled, and the 2.15-second,
perfectly shape-stable decode phase that follows — the ideal CUDA-graph workload,
21,635 identical iterations — is disabled as collateral damage.

### Trace evidence

`cudaStreamBeginCapture` occurs exactly 5 times in the whole run: 4 on the LLM
stream (-18.6, 405.4, 590.9, 652.7 ms) and 1 on the clip stream. That is precisely
"initial capture + 3 more consecutive updates = 4", then silence. No other sticky
disable path applies: the 4090 is sm_89 (passes the Ampere check) and
`GGML_CUDA_DISABLE_GRAPHS` is unset.

### Controlled A/B — `GGML_CUDA_DISABLE_GRAPHS=1`

Falsifiable prediction: if the LLM is *already* ungraphed, setting the env var
must change the ViT and leave prefill/decode bit-identical. It does.

| | graphs ON | graphs OFF | delta |
|---|---|---|---|
| `cudaGraphLaunch` | 30 | 0 | — |
| `cudaLaunchKernel` | 171,040 | 200,340 | +29,300 (the graph-hidden ones) |
| stream-15 plain kernels | **169,168** | **169,168** | **0** |
| ViT median gap | 0.064 us | 0.672 us | **10.5x** |
| ViT in-episode idle (2 req) | 81.68 ms | 106.70 ms | +25.0 ms |
| decode median gap | 0.640 us | 0.640 / 0.671 us | **0** |
| decode idle (2 req) | 75.28 / 76.58 ms | 75.28 / 73.62 ms | **0** |

Decode is untouched by the flag — it had no graphs to lose. The ViT pays 12.5
ms/request. Crucially this gives a **measured** launch-overhead constant on the
same kernels under the same conditions: **0.608 us saved per launch** when
graph-launched, not an inference across stages.

### What the latch costs

| phase | plain launches/request | recoverable GPU idle |
|---|---|---|
| prefill | 34,350 | 21.0 ms |
| decode | 50,951 | 29.6 ms |
| **total** | **85,301** | **50.7 ms/request (1.24%)** |

Reproduced independently on both requests in the trace (50.6 / 50.8 ms). For
decode alone: 31.0 ms of a 2,153 ms phase = **1.44% of decode**.

**This supersedes "~121 ms/request (3%)".** That figure was the CPU-side wall time
of `cudaLaunchKernel` (241.97 ms / 2 requests), which is *hidden*: the CPU runs
~7.35 ms ahead of the GPU, so launch cost only becomes visible as the inter-kernel
gap. 1.24% is the honest number, and it is a ceiling, not a projection.

### Caveats before quoting this

- 50.7 ms is **gap-closing only**. It assumes a fixed LLM reaches the ViT's
  measured 0.064 us floor on every launch, which is the best case.
- It is a *systems* finding about llama.cpp's graph cache, not about Qwen2.5-VL.
  Any interleaved multi-shape workload on this engine hits it; pure-text decode
  does not, because nothing perturbs the shape for 4 consecutive evaluations.
- At 1.24% it is well below the complementary-bottleneck opportunity (~18%) in §9.
  Its value is as a *clean, fully-explained* result, not as a big win.

---

## 4c. The two memory walls — ncu re-analysis  (2026-09-15)

Answers "Prefill — nothing is saturated. NEEDS EXPLANATION." **No new data**: same
20 `.ncu-rep` files, columns that were captured but never read. Full detail and
the per-launch table in `ncu_2026-09-02_7B-BF16/MANIFEST.md`; regenerate with
`python3 ncu_bottleneck.py [--rules]`.

### What was wrong

The column labelled **"L2%" was `L2 Hit Rate`, not L2 throughput** — it says where
traffic landed, not whether that unit is full. And SOL `Memory Throughput` is
max(DRAM, L2, L1), so reading only `DRAM Throughput` reports near-nothing for any
kernel served out of L2 — exactly the 90-97%-hit-rate GEMMs. ncu's own
`SOLBottleneck` rule was sitting in the CSV saying *"Memory is more heavily
utilized than Compute: ... identify the **L2 bottleneck**."*

### The corrected picture

**The pipeline has two memory walls, not one.**

| population | share of stage | binding unit |
|---|---|---|
| ViT GEMMs | 45.9% of ViT | **L2 63-73%** (tensor 43-44, DRAM 17-21) |
| ViT elementwise | 25.3% of ViT | **DRAM 81-94%** |
| ViT flash-attn hd80 | 15.8% of ViT | latency-bound, nothing >60% |
| prefill big GEMMs | 38.2% of prefill | **L2 69-74%** (tensor 48, DRAM 33-37) |
| prefill cutlass_relu | 15.2% of prefill | **DRAM 78-83%** |
| prefill mul_mat_f @592 | 17.3% of prefill | **DRAM 94%** |
| prefill attention | 7.4% of prefill | **L2, saturating to 85%** at depth |
| decode mul_mat_vec_f | 92.8% of decode | **DRAM 94-97%** — unchanged |
| decode flash_attn_ext_vec | 4.7% of decode | **L2 78-80%**, DRAM only 50% |

Decode's DRAM wall stands exactly as published. ViT and prefill are substantially
an **L2** wall, which the hit-rate column made invisible.

### Why the GEMMs saturate nothing on the SM side

They cannot issue fast enough to. Occupancy is **register-limited** and achieved
occupancy already equals the ceiling, so this is not an occupancy-achievement
problem:

| | prefill @1184 | prefill @896 | ViT cutlass @1184 |
|---|---|---|---|
| theoretical occupancy | 16.7% | 16.7% | **8.3%** |
| warps/scheduler (hw max 12) | 2.00 | 2.00 | **1.00** |
| eligible warps/scheduler | 0.15 | 0.15 | 0.12 |
| issue slots busy | 10.2% | 10.1% | 11.6% |

With 1-2 resident warps per scheduler and ~11.7-cycle stalls on an oversubscribed
tensor pipe (`CPIStall`), the scheduler issues once every ~10 cycles. Normal for
big-tile tensor GEMMs — but it means **SM% will never look like a bottleneck for
these kernels, and reading SM% as headroom is a mistake.**

### This breaks the complementary-bottleneck premise (see §9)

That proposal rests on the ViT being **16.17% DRAM** — a figure covering its GEMMs
only, ignoring the elementwise quarter this same survey measured at 81-94% DRAM.
Time-weighted over each stage's covered share:

| stage | coverage | DRAM% | L2% |
|---|---|---|---|
| ViT | 82.2% (profiled shapes only) | **34.9** | 54.8 |
| ViT, incl. estimate for the 16.3% unprofiled | 98.5% | **42.2** | 51.4 |
| prefill | 79.1% | 52.8 | 59.5 |
| decode | 97.1% | 92.4 | 42.6 |

The ViT's unprofiled 16.3% is almost all elementwise/copy work (`rms_norm @2368`,
`cpy_scalar @47360`, `k_bin_bcast @33152`, `convert_unary @31635`); assuming it
behaves like its measured siblings at ~87% DRAM gives the second row. Either way
the conclusion holds.

**Combine these by resource-TIME product, not by summing utilisations** —
utilisations are rates over different durations, so adding them is a category
error (an earlier draft of this section did exactly that). Co-running cannot
finish faster than `max_r(total resource-ms)`:

| resource | encode (1850 ms) | decode (2153 ms) | total resource-ms |
|---|---|---|---|
| **DRAM** | 43.8% -> 810 | 89.4% -> 1925 | **2735  <- binds** |
| L2 | 59.6% -> 1103 | 42.6% -> 917 | 2020 |
| tensor | 26.9% -> 498 | 0.0% -> 0 | 498 |
| SM issue | 16.3% -> 302 | 11.1% -> 239 | 541 |

Serial 4003 ms -> bound 2735 ms = **max 1.46x**, DRAM-binding. The naive hope was
2x; the honest ceiling is 1.46x. It is *not* ~0, which is what "they sum to 133%,
so they collide" wrongly implied. Halving decode's bytes with q8_0 moves the bound
to **1.98x** with L2 binding instead — so quantization and co-scheduling are
multiplicative, not alternatives.

**Coverage caveat** (found 2026-09-15 while auditing target selection): the
ViT figures above are NOT the ones in this survey's own tables. The MANIFEST
credited each elementwise kernel's **per-name total** to the single grid that was
profiled — `k_bin_bcast` "8.5% @11840" is really 5.42% @11840 plus 3.08% at grid
**33152**, which was never profiled; same for both `convert_unary` rows. True ViT
coverage is **82.2%, not 91%**. Still above the 70% bar for a stage-level claim,
so no ViT conclusion changes — but the number was wrong. See §6a.

The experiment is still worth running, now against a falsifiable prediction: **if
DRAM is the shared limit, co-running should pin DRAM near 100% and deliver well
under the claimed ~18% ceiling.**

### Tooling defects found (both still live in `ncu_report.py`)

- Its glob is `ncu_*.csv`; survey files are `<phase>__<kernel>__<share>.csv`. It
  matches nothing and prints **0% coverage for all three stages with no error** —
  which is why the MANIFEST tables were assembled by hand. Verify: run it.
- Its `WANT` table matches **bare metric names**, but `Memory Throughput` appears
  twice — SOL (`68.85 %`) and Memory Workload Analysis (`318.66 Gbyte/s`). Last
  row wins, so its `mem%` holds GB/s labelled as a percent. Key on
  `(section, metric)`, as `ncu_bottleneck.py` does.

Neither defect touched a published number — the affected columns were never
quoted — but both must be fixed before the tool is trusted again.

---

## 5. RETRACTED / do not trust

- **"prefill GEMM underfills the GPU at 0.19 waves"** — RETRACTED, and the
  2026-09-02 survey shows it was **backwards**. That figure was the mean of two
  grid-16 launches and one grid-112, a value no launch had, and ncu never sampled
  grid 592. Measured properly, `mul_mat_f @592` (55% of that kernel's time) runs
  at **93.7% DRAM** — saturated, not idle.
- **"decode is 60% memory-bound"** — superseded. Early `--set basic` smoke test on
  a different kernel in a toy workload. Real figure 94-97% across three shapes.
- **"ViT has a 2.31-wave tail effect"** — wrong. 7.71-9.25 waves, achieved
  occupancy equals theoretical. ViT frame batching would buy nothing.
- **All ncu reports predating 2026-09-02** — deleted. They mixed two naming
  schemes, several workloads, and averaged across heterogeneous grid shapes.
  `ncu_2026-09-02_7B-BF16/` is the only valid ncu data.
- **`nsys`/`perf` data in `profiling_results/base_timeline.*` and `graphs_on.*`**
  — collected on 3B q8_0, stale since the 7B BF16 pin. `handshake_7B.*` is current.

- **"ViT is compute-bound with excellent locality"** — RETRACTED 2026-09-15. Its
  GEMMs run at 43-44% tensor against **63-73% of L2 peak**. The 97% figure quoted
  as evidence was L2 *hit rate*, which is what makes the L2 binding, not evidence
  that it isn't. See 4c.
- **"Prefill's two largest GEMMs are limited by neither SM nor DRAM"** —
  RETRACTED 2026-09-15. They are at 69-74% of L2 peak. See 4c.
- **"Prefill attention becomes more compute-bound as context grows"** — RETRACTED
  2026-09-15. It becomes **L2-saturated** (85% at depth); tensor never exceeds
  54.7%. See 4c.
- **"The ViT is 16.17% DRAM"**, used to argue decode and ViT are complementary —
  RETRACTED 2026-09-15. That covers the ViT's GEMMs only; stage-weighted it is
  **39.1% DRAM**. See 4c.

### Superseded by the 2026-09-02 survey

- Stage coverage is no longer 45%/25% — it is now ~91% ViT, ~75% prefill, ~96%
  decode, so ViT and prefill stage-level statements are defensible.
- Marshalling is not merely "16.3% of ViT time" — those kernels are **bandwidth-
  saturated at 81-94% DRAM**, which reframes the fusion opportunity: it is
  removing DRAM traffic, not removing launches.

## 6. ncu methodology — the rule that matters

**A kernel name is not a kernel — and neither is (name, grid).** The same name is
launched at several grid sizes that behave completely differently, and at a
*single* grid it can still be two different kernels. Averaging across them
produces numbers that describe no actual launch.

The discriminating key is the **launch shape**:

```
(kernel name, grid, registersPerThread, sharedMemory, duration cluster)
```

Each of those four extra dimensions was needed by a real case in this workload:

| discriminator | case |
|---|---|
| grid | `mul_mat_vec_f` runs at 18944 / 3584 / 512 / 152064 |
| registers | prefill `flash_attn_ext_f16` grid 256 has regs 140 / 168 / 212 |
| shared mem | decode `mul_mat_vec_f` grid 3584 has smem 128 / 256 |
| **duration cluster** | decode grid 3584, *identical* regs and smem, is **bimodal**: 28.8 us (n=14080) and 144.1 us (n=7170) |

That last one is `q_proj` and `down_proj` sharing a grid. They differ only in the
weight they stream — 3584x3584 = 25.7 MB versus 18944x3584 = 136 MB, which at
~1008 GB/s is ~25 us against ~135 us, matching the measured clusters. **nsys
records no problem size**, so duration is the only trace-visible discriminator.
Profiling "`mul_mat_vec_f` grid 3584" without splitting it mixes two kernels
whose costs differ 5x.

### The harness now does this for you

```bash
python3 ncu_targets.py shapes                  # distinct launch shapes per phase
bash profile.sh ncuplan --coverage 86          # -> ncu_targets.sh (generated)
bash profile.sh ncu                            # run them, each self-verifying
python3 ncu_report.py                          # per-shape, with stage coverage
```

- `ncu_targets.py` derives targets from `handshake_7B.sqlite`: it picks shapes by
  time-dominance, then solves for the **smallest ncu launch window** that contains
  >= 3 launches of each wanted shape, biased to mid-phase.
- Selection is constrained to the shape's **interquartile duration band**, so the
  profiled launches are representative rather than whatever `-c N` happened to hit.
- One window can serve several shapes of the same kernel name — ncu `-k` matches by
  name, so a mixed window costs one run instead of several *and* supplies the
  per-shape consistency check.
- Every run **verifies** the grids it actually profiled against the prediction and
  fails loudly if launch numbering drifted. Verified correct on all targets.

### Rules that still apply

1. **Report per launch, never the mean.** `ncu --import r.ncu-rep --page details
   --csv`; the `ID` column is the launch index.
2. **Launch consistency IS the validity check.** Launches of one shape agreeing
   within a few percent is evidence; a wide spread means the sample is incoherent
   and must not be quoted. `ncu_report.py` computes the spread and flags it.
3. **State stage coverage.** `ncu_report.py` prints the share of each stage backed
   by a consistent per-shape sample, and refuses to call it a stage-level claim
   below 70%.
4. **`-c N` takes the FIRST N matching launches**, which are unrepresentative
   wherever cost grows with context. The planner emits a mid-phase `-s`.
5. **Percentiles, not min/max.** ViT's cutlass GEMM looks like it spans 143-340 us
   on min/max; its p10-p90 is 144-149 us. A handful of outliers otherwise
   manufacture a fake distribution.
6. **Re-plan after any build change.** Launch numbering and kernel names are
   specific to the build. A q8_0 build uses a different kernel family entirely.

### 6a. Audit of how the ncu targets were actually chosen  (2026-09-15)

**There are two selection methods in this repo, and the survey used the weaker
one.** Worth knowing before trusting a coverage number.

**(1) `ncu_targets.py plan` — the documented planner.** Builds a census of
distinct *launch shapes* `(name, grid, regs, smem, duration cluster)`, walks it
down by time-dominance to a coverage goal, then runs a sliding-window solver for
the **shortest** window in ncu's own launch numbering that contains >= 3 in-phase
launches of every wanted shape, constrained to each shape's interquartile
duration band and biased toward mid-phase. Emits the predicted grid of every
launch so the report can be verified. Output: `ncu_targets.sh`, 21 targets.

**(2) `run_ncu_survey.sh` — what actually produced `ncu_2026-09-02_7B-BF16/`.**
20 hand-written targets with hand-picked selectors. **13 of the 20 use a bare
`-c N`**, i.e. the FIRST N matching launches — the exact thing §6 rule 4 warns
against. Only `prefill__mul_mat_f` (`-s 400`), the four attention depths and the
lm_head variant use a positioned `-s`.

It also contains **no verification step**, so the MANIFEST's claim that "every run
verifies the grids it actually profiled against the prediction ... verified correct
on all targets" is not reproducible from the script that generated the data.

#### Where the difference actually bit

- **Coverage was overstated for ViT: 91% claimed, 82.2% real** (see 4c). Cause is
  per-name-vs-per-grid attribution, not target choice.
- **Four ViT shapes worth 16.3% were never profiled**: `rms_norm_f32 @2368`
  (3.41%), `cpy_scalar @47360` (3.35%), `k_bin_bcast @33152` (3.08%),
  `convert_unary @31635` (2.89% + 1.67%), `flash_attn_stream_k_fixup @24576`
  (1.86%). The generated `ncu_targets.sh` *did* include `rms_norm_f32`; the
  hand-written script dropped it.
- **One genuine launch-consistency violation.**
  `vit__k_bin_bcast_add__9pct` sampled 36.5 / 19.8 / 20.4 us — a **65% spread**,
  well past the 15% incoherence bar. The planner had flagged that shape "3x spread
  at one grid: interleaved problem sizes, target the median"; bare `-c 3` walked
  straight into it, and the MANIFEST then quoted **launch 0** (36.5 us, 91.2% DRAM)
  as the representative value. Both clusters are DRAM-saturated (91.2 vs 88-89%),
  so the *conclusion* survives — but the duration and share are not quotable.

#### Where it is fine, and why

- **Decode is exemplary.** Coverage 97.1%, and every shape has p10 ~= p50 ~= p90
  with rho <= 0.12 — genuinely drift-free, so first-N sampling is as good as
  mid-phase. The bimodal grid 3584 (149 us vs 30 us, 134% raw spread) is the
  **documented** q_proj/down_proj case and was correctly reported as two separate
  shapes; a flat spread check flags it, duration clustering resolves it.
- **Prefill GEMMs are defensible**: rho 0.19-0.30 and p90/p10 < 1.05, so they do
  not drift and first-N is representative. Coverage 79.1%.
- **Prefill attention was handled correctly** — the one shape with real drift
  (11x spread at one grid, rho 0.45) is the one that got a positioned `-s` curve.

So the selection was right where it mattered most and sloppy at the margins. The
fix is to run the planner instead of the hand list, and to compute coverage from
`launch_shapes`, never by summing a name's share onto one grid.

#### Replay defaults, which change how every number reads

`run_ncu_survey.sh` passes no `--clock-control` or `--cache-control`, so ncu
defaults apply:

- **`--clock-control base`**: clocks are LOCKED. This is why `--warmup-iters 0`
  plus first-N sampling is not a clock-ramp confound — but it also means **ncu
  durations are not comparable to nsys durations**. All time shares must come from
  nsys; only ratios *within* an ncu report are meaningful.
- **`--cache-control all`**: L1 and L2 are FLUSHED before each replay pass. So the
  90-97% L2 hit rates in 4c are **intra-kernel reuse only** — inter-kernel reuse is
  excluded by construction. That makes the L2-wall finding conservative: in the
  real pipeline L2 hit rate can only be higher and DRAM traffic lower.
- **`--replay-mode kernel`** (the log warns about memory save/restore): each kernel
  is replayed several times to collect all metric passes. Fine for per-kernel
  counters, meaningless for anything cross-kernel.

### Phase boundaries

`vit` = stream 17 inside an encode window. `prefill` = stream 15 from the encode
start **until decode begins** — *not* until the last ViT kernel: the final frame's
prefill runs after the vision tower finishes, and cutting there silently drops
~8% of prefill (it was hiding 5.02% of the stage in one kernel alone).
`decode` = stream 15 from the first decode GEMV onward.

### Phase attribution without NVTX

Verified phase-exclusive against the nsys trace (counts per request):

| kernel | ViT (stream 17) | LLM (stream 15) | in encode | in decode |
|---|---|---|---|---|
| `mul_mat_vec_f` | 0 | 43273 | 0 | 21619 |
| `flash_attn_ext_vec` | 0 | 7168 | 0 | 3582 |
| `rope_vision` | 1664 | 0 | 832 | 0 |
| `flash_attn_ext_f16<(int)80` | 832 | 0 | 416 | 0 |
| `flash_attn_ext_f16<(int)128` | 0 | 2268 | 1019 | 0 |
| `mul_mat_f<` | 0 | 5211 | 2392 | 0 |
| `cutlass..._relu` | 0 | 2272 | 972 | 0 |

Head dim 80 = ViT, 128 = LLM. `_vec_` = batch-1 decode. **These names are BF16-specific
— a q8_0 build uses a completely different kernel family** (`mul_mat_vec_q`,
`quantize_q8_1`), which invalidated the entire target list once already.

**The durable fix is NVTX ranges** (~4 lines in `baseline_vlm.cpp`, 2 in
`tools/mtmd/mtmd-helper.cpp`), then `ncu --nvtx-include "decode/"` profiles every
kernel in a phase regardless of name. Not yet implemented.

---

## 7. Files

All under `/data/spring26/CPP/llama.cpp/dmc955/`.

| file | role |
|---|---|
| `baseline_vlm.cpp` | the vanilla VLM -> `llama-baseline-vlm` |
| `model_config.sh` | THE only definition of which weights are used |
| `bench_serving.sh` | latency: one loaded model, request queue, GPU trace |
| `bench_serving_summarize.py` | per-stage stats, cold start, tail, drift |
| `dump_characterization.py` | CSV export |
| `bench_vanilla.sh` | latency: process-per-config, for comparing variants |
| `profile.sh` | nsys / perf / ncu harness |
| `REPRODUCE.md` | how to re-run the characterization |
| `characterization/` | current CSVs (7B, fps sampling) |
| `characterization_3b-q8_0_fixed8/` | archived 3B data |
| `profiling_results/` | nsys traces and perf data |
| `ncu_2026-09-02_7B-BF16/` | **the only valid ncu data** — see its MANIFEST.md |
| `run_ncu_survey.sh` | regenerates that survey (resumable) |
| `ncu_bottleneck.py` | **names the binding unit per launch** — the 4c re-analysis |
| `RESEARCH_GAPS.md` | **ranked research gaps with the hw counters behind them** |
| `BOTTLENECK_FUNDAMENTALS.md` | **why each bottleneck exists, derived from the roofline** |
| `OPTIMIZATION_PLAN.md` | **lossless microarch work, ordered — start here** |
| `GAP3_PREFILL.md` | Gap 3 worked in full: ubatch/arithmetic-intensity + kernel limits |
| `RELATED_MICRO24_DLRM.md` | MICRO'24 DLRM paper: methodology to copy, techniques that transfer |
| `profiling_results/graphab_{on,off}.nsys-rep` | the §4b CUDA-graph A/B pair |

**Reports worth citing**: `handshake_7B.nsys-rep` (14.9 MB, CPU sampling + context
switches + GPU metrics, 2 full requests), `graphab_{on,off}.nsys-rep` (the §4b
paired A/B, run back-to-back on an idle GPU), and everything in
`ncu_2026-09-02_7B-BF16/`. Ignore `base_timeline.*` / `graphs_on.*` (3B q8_0).

Regenerate the §4b pair (~4 min, both runs under nsys, identical args):

```bash
source model_config.sh
for g in "" 1; do
  [[ -n "$g" ]] && export GGML_CUDA_DISABLE_GRAPHS=1 || unset GGML_CUDA_DISABLE_GRAPHS
  /usr/local/cuda-12.6/bin/nsys profile -t cuda,osrt --cuda-graph-trace=node \
    -o profiling_results/graphab_${g:-on}  --force-overwrite=true --stats=false \
    ../build/bin/llama-baseline-vlm -m "$MODEL" --mmproj "$MMPROJ" \
    --sintel-dir /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel \
    --serve-videos alley_1 --serve-rounds 2 --fps 2.0 --source-fps 8.0 \
    -n 128 -ngl 99 -c 16384 --temp 0 --seed 42 -fa on --no-flow-analysis \
    --ignore-eos --warmup-iters 0
done
```

**Gotcha**: with graphs off the clip backend lands on **stream 16**, not 17 — any
stream-17 filter silently returns nothing on the `off` trace. Attribute by
comparing `graphNodeId IS NULL` in `CUPTI_ACTIVITY_KIND_KERNEL`, and note that
`nsys` has no `sqlite3` CLI on this box: query the export from Python.

---

## 8. Commands

```bash
cd /data/spring26/CPP/llama.cpp/dmc955

# build (cmake on PATH is broken)
bash profile.sh build

# pre-flight: clocks + measured contention
bash profile.sh gpustate

# latency characterization (~35 min, 483 requests)
VIDS=$(cd /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel/training/clean \
       && ls -d */ | tr -d '/' | paste -sd,)
bash bench_serving.sh --videos "$VIDS" --rounds 21 --fps 2.0 --source-fps 8.0

# timeline, production config (graphs ON, --cuda-graph-trace=node)
bash profile.sh nsys
bash profile.sh kernels

# CPU counters, scoped to one phase via perf --control FIFOs
REGION=decode bash profile.sh perf     # all|inference|encode|decode

# per-kernel microarchitecture
NCU_SET=basic bash profile.sh ncu      # survey: 145 metrics
bash profile.sh ncu decode_gemv        # deep: 614 metrics
```

**Load-bearing flags** (the run is invalid without them): `--ignore-eos`
(BF16 non-determinism), `--source-fps 8.0` (Sintel clips are only 0.8-2.1 s),
`--rounds 21` (round 0 discarded as warmup), `--temp 0 --seed 42`, `-fa on`,
`--no-flow-analysis`.

**Sanity checks**: every measured request has `n_tokens == 128`; `residual_ms` is
small and **positive** (it cross-checks the JSON timer against the log parse — a
strongly negative value means the parse picked up iterations the timer did not);
SM clock median equal in both halves of the run.

---

## 9. Open questions

> **See `RESEARCH_GAPS.md`** for the ranked version of this list with the nsys
> hardware counters (SM Issue, Tensor Active, DRAM, unallocated warps) behind each
> item. The biggest gap is not in this section: ViT and prefill are independent per
> frame yet overlap only 13%, worth **578 ms/request (14.1%)**.

**Highest value, no code needed:**
- **Complementary-bottleneck scheduling.** *Premise corrected in 4c — read that
  first.* The ViT is **39-44% DRAM**, not 16.17% (the old figure ignored its
  DRAM-saturated elementwise quarter). By resource-TIME product the ceiling for
  encode||decode is **1.46x, DRAM-binding** — below the naive 2x but a real prize,
  and it rises to **1.98x** if quantization halves decode's bytes. Run it as two
  concurrent processes; prediction: DRAM pins near 100%, speedup <= 1.46x.
- **Batch scaling.** SM at 13.58% means ~7x compute headroom. `llama-batched-bench`
  and `llama-speculative` are already built.

**Needs the corrected ncu methodology:**
- ~~Re-profile ViT and prefill by (kernel, grid) to reach ~70% stage coverage.~~
  **Done** — coverage is 91/84/97%.
- ~~Is `mul_mat_f` grid 592 well-filled or not?~~ **Answered**: 93.7% DRAM, 2.31
  waves — saturated. (Grid 112 and 16 *are* underfilled: 0.44 and 0.06 waves.)
- **New, from 4c**: is the L2 wall on ViT/prefill GEMMs addressable? The traffic is
  already at 90-97% hit, so it is not a locality problem — it is L2 *bandwidth*.
  Worth checking whether larger tiles or fewer passes reduce L2 traffic per FLOP.

**Engineering:**
- ~~Why do CUDA graphs fall back?~~ **ANSWERED — see section 4b.** The follow-up is
  the fix: give the LLM graph cache a key that separates prefill from decode (or
  reset the latch on a shape change), then re-measure. Predicted ~51 ms/request.
- Can the f32<->bf16 conversion be fused away? (~110 ms/request in the ViT)
- NVTX ranges, to make phase attribution durable.

**Methodological:**
- Quantization sweep (`llama-quantize` is built) to validate that decode time
  scales purely with bytes per token. Predicts q8_0 ~1.9x, q4_K_M ~3.4x on decode.
