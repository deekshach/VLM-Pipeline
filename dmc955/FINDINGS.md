# All findings — VLM pipeline characterization

Qwen2.5-VL-7B-Instruct-BF16 + mmproj on llama.cpp/libmtmd, RTX 4090 (sm_89),
MPI-Sintel `alley_1`. Stock llama.cpp at defaults is the baseline throughout.

Compiled 2026-09-16. Method: `METHODOLOGY.md`. Derivations:
`BOTTLENECK_FUNDAMENTALS.md`.

---

## A. Operating points measured

| tag | frames | visual tokens | source | `-ub` | `-n` | status |
|---|---|---|---|---|---|---|
| **f8** | 8 | 4,736 | `results/trace_f8.nsys-rep` + `ncu_2026-09-15_f8/` | 512 | 128 / 16 (ncu) | **current** |
| f1…f16 | 1-16 | 592-9,472 | `results/trace_f*.nsys-rep` | 512 | 128 | sweep |
| f13 | 13 | 7,696 | `profiling_results/handshake_7B.*` + `ncu_2026-09-02_7B-BF16/` | 512 | 128 | prior |

**The kernel mix is operating-point dependent** — ncu targets do not transfer
(see §K3). Every number below carries its point.

---

## B. Stage latency — f8

| stage | GPU busy | wall span | share of request |
|---|---|---|---|
| ViT encode | 419.3 ms | — | 12.5% |
| LLM prefill | 786.9 ms | — | 23.5% |
| encode (union) | 1,147.6 ms | **1,202.4 ms** | 35.9% |
| LLM decode | 2,070.0 ms | **2,142.4 ms** | 64.1% |
| **request** | | **3,344.9 ms** | 58.5 tok/s |

ViT/prefill overlap: **58.6 ms = 14.0% of ViT**.

## C. Scaling laws (1→16 frames, R² from 9 points)

| stage | fit | R² | reading |
|---|---|---|---|
| ViT | `−2.02 + 52.843·N` | **0.999945** | exactly linear, ~zero intercept — no cross-frame work |
| prefill | `+8.25 + 92.593·N + 0.5739·N²` | **0.999985** | the N² term is attention over prior visual tokens |
| prefill (linear only) | `−19.24 + 102.427·N` | 0.999210 | curvature is real, not fit noise |
| decode | `2129.4 + 4.678·N` | 0.48 | flat within noise; +3.3% over N=1…16 |

Marginal cost of one more frame: ViT constant 52.84 ms; prefill **93.74 → 110.96 ms
(+18.4%)** from N=1 to N=16.

**Stage mix inverts with frame count:**

| N | ViT | prefill | decode |
|---|---|---|---|
| 1 | 2.2% | 4.3% | **93.1%** |
| 8 | 12.5% | 23.5% | 64.1% |
| 16 | 18.0% | 34.8% | **47.4%** |

Encode-side optimisations scale their value with frame count; decode-side do not.
Overlap saturates at ~14.5% — pipelining does not solve itself at larger N.

---

## D. Whole-phase hardware counters (nsys `GPU_METRICS`, 100% coverage)

| counter | encode | decode |
|---|---|---|
| SMs Active | 88.0% | 93.0% |
| **SM Issue** | **16.1%** | **10.9%** |
| **Tensor Active** | **26.3%** | **0.0%** |
| DRAM total | **44.5%** | **89.3%** |
| Unallocated warp slots | **71.6%** | 8.3% |
| PCIe RX / TX | 1.9% / 1.3% | 1.0% / 1.1% |

`SMs Active` 88-93% is what looks "busy". `SM Issue` 11-16% is what is happening.
Request-weighted **Tensor Active = 12.8%** — the dominant compute unit idles ~87%
of the request.

---

## E. Roofline

| quantity | value |
|---|---|
| ridge point (BF16 tensor) | 165.2 TFLOP/s ÷ 1008 GB/s = **164 FLOP/byte** |
| decode arithmetic intensity | **1** (0.006× ridge) |
| prefill AI | **= tokens per ubatch** (512 → 438; 80 → 78; 2 → 2) |
| decode measured | 16.82 ms/token |
| decode predicted (15.24 GB ÷ 901 GB/s) | **16.91 ms/token = 100.5%** |

Decode streams essentially the whole model once per token. Even batch 64 reaches
only 0.39× ridge — batching makes tensor cores *usable*, not *useful*.

---

## F. Kernel census — f8

| stage | GPU busy/req | distinct launch shapes | shapes for 90% |
|---|---|---|---|
| ViT | 419.3 ms | **31** | 12 |
| prefill | 786.9 ms | **110** | 16 |
| decode | 2,070.0 ms | **16** | 4 |

Shape count vs frames — **prefill is the only context-dependent stage**:

| frames | ViT | prefill | decode |
|---|---|---|---|
| 1 | 28 | **75** | 15 |
| 8 | 31 | **110** | 16 |
| 16 | 27 | **131** | 15 |

---

## G. Per-kernel microarchitecture — f8 (fresh, `ncu_2026-09-15_f8/`)

Coverage, per launch shape, quotable = passes the consistency gate:
**ViT 82.1%, prefill 86.5%, decode 97.5%.**

### G1. Decode — memory wall on every shape

| grid | dur µs | DRAM% | L2 thr% | L2 hit | SM% | occ% | waves | share |
|---|---|---|---|---|---|---|---|---|
| 18944 | 312.0 | **95.6** | 39.0 | 1.1 | 13.2 | 93.2 | 24.67 | 49.95% |
| 3584 down_proj | 179.1 | **93.6** | 34.1 | 25.1 | 9.3 | 76.4 | 4.67 | 25.14% |
| 3584 q_proj | 30.2 | **88.3** | 38.4 | 7.9 | 20.2 | 83.2 | 4.67 | 9.84% |
| 152064 lm_head | 1170.0 | **96.7** | 41.3 | 0.8 | 18.6 | 95.2 | 198 | 7.18% |
| 512 | 6.8 | 55.4 | 24.6 | 35.8 | 13.4 | 59.9 | **0.67** | 1.97% |
| flash_attn_vec @756 | 22.4 | 45.9 | **69.5-71.7** | 85.9 | 17.4 | 22.6/25 | 1.97 | 3.23% |

92% of decode at 88-97% of DRAM peak. **New:** decode attention is L2-leaning
(70-72%), not DRAM — invisible without an L2-throughput column.

### G2. ViT — GEMMs are L2-bound, not compute-bound

| kernel | grid | dur µs | DRAM% | **L2 thr%** | L2 hit | SM% | **tensor%** | w/sched | verdict |
|---|---|---|---|---|---|---|---|---|---|
| cutlass_s16816gemm | 1184 | 175.0 | 12.3 | **65.1** | 97.0 | 41.8 | 43.6 | **1.00** | L2-leaning |
| cutlass_s16816gemm | 592 | 177.2 | 14.6 | **74.5** | 97.5 | 41.5 | 44.0 | 1.00 | L2-leaning |
| ampere_128x64 | 370 | 66.8 | 14.8 | **64.1** | 94.8 | 41.7 | 44.4 | 2.00 | L2-leaning |
| flash_attn hd80 | 384 | 275.6 | 16.7 | 55.0 | 94.7 | 47.9 | 37.6 | 3.00 | **latency-bound** |

Tensor pipe never exceeds 44% while L2 runs 64-75%.

Elementwise — saturated, as previously found:

| kernel | grid | dur µs | DRAM% | L2 thr% | L2 hit |
|---|---|---|---|---|---|
| silu_gated | 31635 | 91.9 | **92.5-94.0** | 31.2 | 33.4 |
| convert f32→bf16 | 11840 | 16.0 | **81.9-82.5** | 34.4 | 33.7 |
| convert bf16→f32 | 31635 | 38.8 | **81.4-82.5** | 44.2 | 66.8 |
| k_bin_bcast | 11840 | 15.5 / 29.0 | 81-90 | 39-46 | 33-51 |

Two kernels never profiled before:

| kernel | grid | share | result |
|---|---|---|---|
| `rms_norm_f32` | 2368 | 3.41% | **latency-bound** — 37.8% DRAM, 18.6% L2, nothing >60%; 8 warps/sched, 1.29 eligible |
| `cpy_scalar` | 47360 | 3.35% | **incoherent** — 32.1/32.2/49.3 µs, 45% spread, two populations. Not quotable |

### G3. Prefill — resolved: L2-bound, not "nothing"

| kernel | grid | dur µs | DRAM% | **L2 thr%** | L2 hit | SM% | tensor% | waves | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ampere_64x128 | 1184 | 535.1 | 33.0 | **68.8** | 91.5 | 44.9 | 48.4 | 4.62 | L2-leaning |
| ampere_128x64 | 896 | 501.7 | 36.1 | **73.7** | 90.6 | 48.0 | 48.2 | 3.50 | L2-leaning |
| cutlass_relu | 592 | 206.4 | 78.4 | 50.0 | 84.7 | 29.2 | 34.9 | 2.31 | DRAM-leaning |
| cutlass_relu | 896 | 205.7 | **81.7** | 51.1 | 83.7 | 29.4 | 32.0 | 3.50 | DRAM-saturated |
| mul_mat_f | 592 | 178.5 | **93.0** | 34.1 | 12.7 | 11.7 | – | 2.31 | DRAM-saturated |
| ampere_128x128 | 112 | 111.5 | 27.0 | 42.8 | 93.0 | 41.0 | 47.8 | **0.44** | **underfilled** |

### G4. Prefill attention — an L2 curve, not a compute curve

| skip | dur µs | DRAM% | **L2 thr%** | L2 hit | SM% | tensor% |
|---|---|---|---|---|---|---|
| 40 | 9.9 | 6.8 | 4.8 | 73.3 | 2.1 | – |
| 150 | 84.0 | 15.5 | **75.7** | 95.8 | 46.0 | 47.5 |
| 280 | 10.4 | 37.1 | 22.0 | 63.4 | 5.6 | – (outlier) |
| **400** | 176.3 | **10.1** | **82.1** | 97.7 | 51.6 | 52.8 |

L2 climbs 4.8 → 75.7 → 82.1 while DRAM **falls** 37 → 10. It saturates the L2;
tensor caps at 52.8%. s280 is the same artifact as the old s640 — flash attention
launches several blocks per layer, so skip index ≠ context depth.

---

## H. The binding-unit rule (derived, then validated)

A high hit rate is a **bandwidth amplifier**: if L2 carries rate R at hit rate h,
DRAM sees R(1−h). With L2 peak ≈ 2.5× DRAM peak, **L2 binds whenever h > ~60%**.
Validated on 8 read-reuse kernels, 8/8 correct.

| population | share | binding unit |
|---|---|---|
| ViT GEMMs | 45.9% of ViT | **L2 64-75%** |
| ViT elementwise | 25.3% of ViT | **DRAM 81-94%** |
| ViT flash-attn | 15.7% of ViT | latency-bound |
| prefill big GEMMs | 38.9% of prefill | **L2 69-74%** |
| prefill cutlass_relu / mul_mat_f | 24.6% of prefill | **DRAM 78-93%** |
| prefill attention | 5.2% of prefill | **L2, saturating to 82%** |
| decode GEMV | 93% of decode | **DRAM 88-97%** |
| decode attention | 3.2% of decode | **L2 70-72%** |

**The pipeline has two memory walls, not one.** Scope: read-reuse kernels; the
write-streaming elementwise ones do not obey the rule.

---

## I. Occupancy ceilings — set by tile shape, different resource per family

| kernel | block | regs | blk/SM | **limiter** | warps/SM | occ | w/sched | eligible |
|---|---|---|---|---|---|---|---|---|
| prefill `ampere_*` | 128 | 230 | 2 | **REGISTERS** | 8 | 16.7% | 2.00 | 0.15 |
| prefill `cutlass_relu` | 128 | 88 | 2 | **SHARED MEM** | 8 | 16.7% | 2.00 | 0.11 |
| ViT `cutlass` | 128 | 156 | 1 | **SHARED MEM** | 4 | **8.3%** | **1.00** | 0.12 |
| `mul_mat_f` | 224 | 92 | 2 | REGISTERS | 14 | 29.2% | 3.50 | 0.04 |
| decode `mul_mat_vec_f` | 256 | 39 | — | WARPS | 48 | 76-95% | 12.00 | 0.11-0.45 |
| elementwise | 256 | 16-20 | — | WARPS | 48 | 100% | 12.00 | 0.24-0.84 |

230 regs × 128 threads = 29,440; 65,536/29,440 = 2 blocks. ViT's cutlass uses
72 KB smem against ~100 KB/SM → 1 block. Achieved == theoretical everywhere, so
these are **ceilings, not tuning failures**. Low occupancy alone is NOT a defect —
big-tile GEMMs trade it for register blocking. The defect claim needs the
conjunction: tensor 48% + L2 69% + 0.15 eligible warps + `CPIStall` 11.7 cycles on
an oversubscribed math pipe.

---

## J. Structural / engine findings

| # | finding | measured | mechanism |
|---|---|---|---|
| J1 | **ViT and prefill run in lockstep** though independent per frame | overlap **14.0%** at f8; **578 ms/request** ceiling at f13 | `mtmd_helper_eval_chunks` is a sequential host loop; the `n_past` accumulator is a *false* dependency — M-RoPE advances 37/frame independent of content |
| J2 | **Prefill does 39 LLM evaluations for 7,749 tokens** (f13) | 13×512, 13×80, 12×2, 1×29; **383 ms/request excess (9.3%)** | AI = tokens/ubatch; 26 of 39 below the 164 ridge. Weight re-streaming 594 GB = **69% of encode DRAM traffic** |
| J2a | `n_ubatch` 512 vs 592 tokens/frame → 512+80 split | **`-ub 640` saves 90 ms/request**, decode unchanged | CLI flag, zero code. Upstream TODO at `mtmd-helper.cpp:275` |
| J2b | 12 × 2-token `<vision_end><vision_start>` separators | **214 ms/request for 24 tokens** (8,930 µs/token, 79×) | `llama_batch` carries `token` **or** `embd`, so they cannot share an image ubatch |
| J3 | **CUDA graphs latched off for the whole LLM** | **50.7 ms/request**; ViT 100% graphed, LLM 0% | Cache keyed on `cgraph->nodes[0]` (fixed address) → prefill and decode share one sticky `disable_due_to_too_many_updates` latch, tripped after 4 evals, never cleared |
| J4 | **2× visual-token inflation** | 592 tokens/frame vs reference 296; patch upload 22.43 MB vs 11.14 MB needed | no temporal patch merging — each frame encoded as an independent still |
| J5 | **M-RoPE forecloses token-granular KV reclamation** | — | all image tokens in a frame share one temporal position; `llama_memory_seq_rm` evicts at frame granularity only |
| J6 | Host staging amplification | 578 MB H2D/encode for ~18 MB of pixels (32×) | free here (PCIe 1.9%); a risk on Jetson's shared LPDDR |

---

## K. Corrections to earlier claims

| # | retracted | correct | why it was wrong |
|---|---|---|---|
| K1 | "ViT is compute-bound with excellent locality" | **L2-leaning, 64-75% of L2 peak**; tensor 43-44% | the "97% L2" quoted as evidence was **hit rate**, which is what *makes* L2 bind |
| K2 | "Prefill: nothing is saturated" | **69-74% of L2 peak**; ncu's own rule names the L2 bottleneck | L2 *throughput* was captured by `--set full` and never read |
| K3 | "Prefill attention becomes more compute-bound with depth" | it **saturates the L2** (82%); tensor caps at 52.8% | DRAM *falling* 37→10% was misread as compute-bound |
| K4 | "ViT is 16.17% DRAM" → decode/ViT are complementary | **44.5% DRAM**; co-scheduling ceiling is **1.46×** (→1.98× with q8_0) | the figure covered ViT's GEMMs only and ignored its DRAM-saturated elementwise quarter |
| K5 | "ViT+decode collide, gain ≈ 0" (my own earlier claim) | **1.46×** | summing *utilisations* across unequal durations is a category error; the right quantity is the resource-**time** product |
| K6 | "ncu coverage: ViT 91%" | **82.2%** | per-*name* share credited to the single grid sampled |
| K7 | "CUDA graphs barely used (30 launches)" | ViT **100%** graphed, LLM **0%** | each graph launch replaces 1,270 kernel launches |
| K8 | "prefill GEMM underfills at 0.19 waves" | `mul_mat_f @592` is at **93% DRAM**, 2.31 waves | mean of two grid-16 launches and one grid-112; ncu never sampled grid 592 |

---

## L. Methodology and tooling defects found

| # | defect | consequence |
|---|---|---|
| L1 | `run_ncu_survey.sh` hand-written targets | **17.0% of ViT never profiled**; 3 of 6 gaps were the same name at a different grid |
| L2 | `vit__k_bin_bcast` sampled 36.5/19.8/20.4 µs | 65% spread straddling two clusters; MANIFEST quoted launch 0 as representative |
| L3 | `run_ncu_survey.sh` has no verification step | "every run verifies the grids" not reproducible from the script |
| L4 | `ncu_report.py` globs `ncu_*.csv` | matches nothing on the survey dir, prints **0% coverage silently** |
| L5 | `ncu_report.py` matches bare metric names | `Memory Throughput` exists twice; its `mem%` holds **GB/s labelled as a percent** |
| L6 | `ncu_targets.py` hardcodes `FRAME_KERNELS=1270`, `N_FRAMES=13` | mis-slices any other frame count |
| L7 | `bench_serving_summarize.py` splits stages by host log lines | breaks precisely when ViT/prefill overlap — the optimisation invalidates its own measurement |
| L8 | nsys `StringIds` contains the model's generated text | truncated mid-UTF-8 → `OperationalError` on unrelated queries (fixed: lenient `text_factory`) |
| L9 | clip backend moves stream 17 → 16 when graphs are off | a hardcoded stream filter silently reports **zero ViT time** |
| L10 | `-c N` takes the FIRST N launches | hit again in this survey: 3 prefill targets landed on the 29-token text chunk instead of the image chunks (−4.7% coverage) |

---

## L2. Cross-kernel L2 reuse — `--cache-control none` experiment (2026-09-16)

ncu defaults to `--cache-control all`, flushing L1/L2 before each replay pass, so
every hit rate in §G is **intra-kernel reuse only**. Re-measured without the flush.
Trap avoided: no-flush + multi-pass replay lets passes 2+ see the cache warmed by
pass 1 of the *same* kernel. A one-pass metric set was used, so there is no replay.

| kernel | L2 hit (all → none) | DRAM read | **Δ duration** |
|---|---|---|---|
| ViT `cutlass` @1184 | 97.16 → 98.28 | 18.57 → 10.15 MB (**−45.3%**) | **−0.0%** |
| ViT `silu_gated` @31635 | 33.35 → **53.72** | 65.17 → 45.06 MB (−30.8%) | **−11.3%** |
| prefill `ampere_64x128` @1184 | 91.42 → 91.67 | 141.51 → 137.20 MB (−3.0%) | −0.5% |
| prefill attn hd128 @256 | 96.23 → 98.91 | 11.40 → 3.80 MB (**−66.7%**) | −5.0% |

3 launches/cell. Raw: `results/cache_control/`.

| conclusion | evidence |
|---|---|
| **Cross-kernel reuse is real and large** | DRAM reads fall 45-67% for the GEMM and attention |
| **The L2 wall is proven causally, not inferred** | ViT GEMM loses 45.3% of DRAM traffic and runs **0.0% faster** — it is L2-bandwidth-bound |
| **L2 pinning is ruled out** | pinning changes residency, not the rate L2 serves; the only kernel gaining time (`silu`, −11.3%) is DRAM-bound and already gets it, since production never flushes |
| **Survey absolutes are pessimistic for memory-bound kernels** | real `silu_gated` is 53.7% hit / 73.8 µs, not 33.4% / 83.1 µs. Classifications unchanged |
| **Still open** | whether L2 traffic *per FLOP* is reducible by a different tile — untouched here; merged into the tiling question |

A-priori sizing held: the ViT GEMM recovers 8.4 MB against a predicted 6-15 MB
working set; prefill's GEMM barely moves because its 135.8 MB of weights cannot
fit in L2 regardless.

---

## M. Quantified opportunity (lossless only — no pruning, no quantization)

| item | prize | effort | status |
|---|---|---|---|
| `-ub ≥ 592` | **90 ms/request** | one CLI flag | **validated** |
| ViT/prefill pipelining | **578 ms/request** (f13 ceiling) | days | not started |
| separator chunk fusion | **214 ms/request** | days | not started |
| CUDA graph latch | **51 ms/request** | ~10 lines | diagnosed |
| fattn `occupancy` retune | ~199 ms addressable | hours | not started |
| phase co-scheduling | **1.46×** throughput (1.98× with q8_0) | zero code to test | not started |
| ~~L2 residency control~~ | **none** | ruled out 2026-09-16 (§L2) | closed |

Serial items total **933 ms of a 4,101 ms request (22.7%)** at f13, before any
co-scheduling gain.

---

## N. Ruled out — negative results

| claim | measurement |
|---|---|
| PCIe is a bottleneck | **no** — 1.9% RX, 1.3% TX |
| CPU is a bottleneck | **no** — decode GPU busy 97.0%, CPU runs 7.35 ms ahead |
| Decode is occupancy-limited | **no** — 8.3% unallocated warps, 76-95% achieved. Purely bandwidth |
| ViT frame batching would help | **no** — 7.71-9.25 waves, achieved == theoretical |
| Prefetching would help decode | **no** — it is bandwidth-saturated, not latency-bound |
| ViT+decode are complementary | **partly** — DRAM-limited, 1.46× not 2× |

---

## O. Open / unmeasured

1. ~~`--cache-control none` re-profile~~ — **done** (§L2): pinning ruled out, L2
   wall upgraded from inferred to proven.
2. **`-ub 640` sweep** — rung-1 curve; also predicts prefill's shape count collapses.
3. **Four mis-targeted ncu shapes** — prefill silu/convert @37888, ViT
   `k_bin_bcast @33152`, `flash_attn_stream_k_fixup @24576`.
4. **`cpy_scalar @47360`** — incoherent sample, needs re-profiling.
5. **Two concurrent processes** — test the 1.46× bound.
6. **Jetson** — where PCIe/DRAM findings change qualitatively.
