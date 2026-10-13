# Qwen2.5-VL Baseline Profile

Benchmark + microarchitectural characterisation of the **vanilla** VLM path
(`baseline_vlm.cpp`), measured 2026-08-28 on `e5-cse-340-01`.

Web version (same content, nicer tables): https://claude.ai/code/artifact/4a557e65-8de0-4962-aa2b-935b8ca940e3

| | |
|---|---|
| GPU | RTX 4090, sm_89, driver 580.173.02 |
| CPU | i9-14900K (P-cores 0-15, E-cores 16-31), kernel 6.8.0-138 |
| Model | Qwen2.5-VL-3B-Instruct q8_0 + mmproj f16 |
| llama.cpp | b55dcdef5 (dirty) |
| Workload | Sintel `alley_1`, 8 frames, `-n 128`, greedy, seed 42, `-c 16384` |

---

## Summary

1. **Flash attention is a comparison-breaking confound.** Vanilla runs with it
   enabled; the pruning variants force it off to capture attention. That alone
   costs **+77% prefill**, a handicap the optimisation must repay before showing
   any gain.
2. **Decode is memory-bound and immovable.** ~43% of GPU time at ~560 GB/s
   against the 4090's ~1008 GB/s ceiling. Only reading fewer bytes helps.
3. **Visual-token pruning targets the smaller half.** ViT + prefill is ~34% of
   GPU time at this operating point. Decide explicitly whether you optimise
   TTFT or end-to-end.
4. **The baseline is genuinely vanilla** — verified, not assumed. One piece of
   reporting-only optical-flow work was removed from the measured path; it was
   20% of the CPU profile and 89% of all branch mispredictions.

---

## Phase attribution

GPU kernel time by phase (CUDA graphs disabled for clean attribution):

| phase | share |
|---|---|
| LLM decode | ~43% |
| LLM prefill | ~17% |
| ViT encode | ~17% |
| marshalling + shared ops | ~23% |

Wall-clock split (flash attention on):

| phase | ms | what it is |
|---|---|---|
| ViT | 404 | vision encoder over 8 frames |
| prefill | 233 | visual tokens into the KV cache |
| residual | 89 | tokenise + text-chunk decode |
| **TTFT** | **732** | time to first token |
| decode | 671 | 121 tokens @ 180.3 tok/s |

The split is free: `mtmd-helper` already logs `image slice encoded in N ms`
(ViT) and `image decoded (batch i/n) in N ms` (prefill). The harness's own
`encode_ms` conflates them; `bench_summarize.py` separates them by parsing
those lines. No instrumentation needed.

---

## FINDING: the flash-attention confound

Vanilla resolves `-fa auto` to **enabled**. `streaming_vlm.cpp:3930` forces
`LLAMA_FLASH_ATTN_TYPE_DISABLED` whenever `--enable-pruning` or correlation
analysis is on — it must, because flash attention fuses the softmax the
`cb_eval` capture depends on.

| flash attn | ViT | prefill | TTFT | gen | tok/s |
|---|---|---|---|---|---|
| on (vanilla default) | 420 | 239 | 758 | 693 | 174.6 |
| off (pruning regime) | 414 | 424 | 934 | 735 | 164.6 |
| **delta** | — | **+77%** | **+23%** | +6% | -5.7% |

ViT is unaffected — `-fa off` only touches the LLM. So a pruning run starts
~190 ms of prefill in the hole before it prunes anything. A genuine win reads
as flat; a small regression reads as a large one.

**Mitigation:** flash attention is now a sweep axis, not a setting. Compare
pruning/correlation runs against the `fa=off` rows; compare eviction-only and
stock-decode runs against `fa=on`.

---

## Is the baseline vanilla?

Yes. Uniform frame selection, all frames fully encoded through stock `mtmd`
every run, no embedding cache, no patch cache, no token reduction, no pruning,
no KV reuse. Those live in separate binaries.

- `select_frames()` (motion-guided, `baseline_vlm.cpp:368`) and
  `analyze_spatial_patches()` (`:535`) are **defined but never called** — dead
  code from the shared file layout across variants.
- Order of operations settles it: frames are chosen at `:1037` by
  `select_frames_uniform(n_total_frames, base_frames)`, which takes two integers
  and no motion data. `build_motion_profile()` runs *after*, at `:1043`.
- `adaptive_vlm_context` is a copy-pasted struct name; it does plain model +
  `mtmd` init.
- Determinism confirmed empirically: under `--temp 0`, two runs produced
  byte-identical text and the same 121 tokens.

### The optical-flow work that shouldn't have been there

`build_motion_profile()` read ~50 `.flo` files and sorted a full 1024x436 field
per frame for one p95 value, purely for a `motion_coverage` reporting metric.
It could not influence inference (runs after selection) but dominated the CPU
profile.

Now behind `--no-flow-analysis` — opt-in, so other scripts keep their behaviour,
with `params.flow_analysis_skipped` in the JSON so `motion_coverage: 0` is never
misread as a measurement. Inference timings were unchanged with it removed
(ViT 404 vs 418 ms, prefill 233 vs 238, identical 121 tokens).

---

## GPU microarchitecture

Top kernels by total GPU time (graphs disabled). Phase tags derive from kernel
identity: `mul_mat_vec_q` appears only in batch-1 decode, cutlass/ampere GEMMs
only in the ViT, `flash_attn_ext_vec` only in decode.

| phase | kernel | time % | launches | avg us |
|---|---|---|---|---|
| decode | `mul_mat_vec_q<q8_0,1,true>` | 33.0 | 25,804 | 16.8 |
| prefill | `mul_mat_q<q8_0,128>` | 8.5 | 1,992 | 56.2 |
| ViT | `cutlass_80_tensorop_h16816gemm_128x256` | 5.2 | 528 | 130.2 |
| ViT | `flash_attn_ext_f16<80,80,64,1>` | 3.8 | 256 | 197.3 |
| decode | `flash_attn_ext_vec<128,1>` | 3.8 | 4,320 | 11.7 |
| ViT | `ampere_h1688gemm_256x64` | 3.8 | 1,024 | 48.9 |
| prefill | `mul_mat_q<q8_0,80>` | 3.3 | 1,992 | 21.8 |
| shared | `k_bin_bcast<op_add>` | 3.2 | 7,078 | 5.9 |
| prefill | `flash_attn_ext_f16<128,128,8,8>` | 2.5 | 612 | 53.8 |
| shared | `quantize_q8_1` | 2.4 | 28,287 | 1.1 |
| shared | `rms_norm_f32<1024>` | 2.3 | 10,610 | 2.8 |

### Decode is bandwidth-bound

A representative `mul_mat_vec_q` under `ncu`: **60.1% memory throughput vs 18.9%
compute (SM) throughput**. The arithmetic agrees independently:
121 tokens x 3.2 GB of q8_0 weights / 0.69 s ~= **560 GB/s** against ~1008 GB/s
peak. Decode reads the entire weight set once per token — no kernel tuning
changes that, only reading fewer bytes (smaller quant, speculative decoding,
batching).

`quantize_q8_1` plus the f16<->f32 `convert_unary` pair are ~6% of GPU time
across 32k launches. That is data marshalling, not mathematics.

---

## CPU microarchitecture

P-core counters, pinned to CPUs 0-15. Before/after shows the effect of removing
the reporting-only flow analysis.

| counter | with flow analysis | vanilla path only |
|---|---|---|
| wall time | 3.62 s | 2.75 s |
| IPC | 1.78 | 2.04 |
| branch misses | 131.1 M (1.95%) | 14.3 M (0.25%) |
| L1-dcache miss | 1.57% | 2.00% |
| LLC load miss | 81.6% | 80.3% |
| dTLB load miss | 0.01% | 0.02% |
| top hotspot | `std::__introsort_loop` 15.0% | `ggml_backend_cuda_synchronize` |

**89% of all branch mispredictions in the binary were the sort**, not the VLM.
The real inference path is almost perfectly predictable at 0.25%. The high LLC
miss rate is not alarming: only 43 M LLC loads out of 9.58 B L1 loads reach that
level — the signature of streaming data with no reuse, exactly right for feeding
tensors to a GPU.

### What was hiding underneath

```
libcuda.so                              10.31%
  └─ ggml_backend_cuda_synchronize        9.93%
       ├─ ggml_backend_sched_synchronize  7.73%
       │    └─ llama_context::synchronize 5.11%
       │         └─ common_sampler_sample 5.00%
       └─ ggml_backend_sched_graph_compute 2.62%
```

The dominant CPU cost is **blocking on GPU synchronisation inside the sampler** —
the CPU stalling on the GPU every decode step. Same story as the GPU side:
decode is latency-serialised, one token at a time, and neither the CPU nor the
SMs are the limit.

**Caveat:** the multiplexing percentages in `perf` output (41-75%) mean twelve
events were time-multiplexed across fewer hardware counters and scaled up. Sound
estimates, not exact counts. For an exact ratio, use a smaller event group.

---

## Running the harnesses

| file | role |
|---|---|
| `bench_vanilla.sh` | macro benchmark — videos x frames x flash-attn x reps |
| `bench_summarize.py` | parses runs into table + CSV, splits ViT from prefill |
| `profile.sh` | micro profiling — nsys timeline, perf counters, ncu roofline |
| `baseline_vlm.cpp` | the vanilla path; gained `--no-flow-analysis` |

```bash
# macro: full sweep, ~30 min (120 runs)
bash bench_vanilla.sh
bash bench_vanilla.sh --quick              # 1 config, both fa modes, ~2 min
bash bench_vanilla.sh --fa off --reps 5

# micro
bash profile.sh build
bash profile.sh nsys                       # graphs OFF - clean attribution
bash profile.sh nsys-real                  # graphs ON  - true wall-clock
bash profile.sh kernels                    # top kernels from existing report
bash profile.sh perf                       # CPU counters + hotspots
bash profile.sh ncu                        # all 5 phases (slow)
bash profile.sh ncu decode_gemv            # just one
FA=off bash profile.sh ncu                 # the pruning variants' regime
```

Both harnesses run the identical workload so profiler output cross-references
directly against benchmark numbers.

**What the benchmark controls for:** model load and flow analysis (excluded —
reads per-phase timers, never `total_ms`, which includes a 4.5 GB GGUF load);
sampling noise (greedy + fixed seed => deterministic, identical token counts);
cold start (one discarded warmup per config); drift (N reps, flags spread >5%);
CUDA graphs (enabled in the benchmark, disabled in the nsys attribution pass —
never compare across those two).

Five `ncu` targets aim at the measured hot kernels rather than a guessed regex:
`vit_gemm`, `vit_attn`, `prefill_gemm`, `decode_gemv`, `decode_attn`. Read
Speed-Of-Light, then Roofline, then Occupancy, then Warp State. For
`decode_gemv` check DRAM sector efficiency in Memory Workload Analysis.

---

## Environment gotchas

Each of these silently produced wrong or absent data before being found.

| trap | fix |
|---|---|
| `nsys`/`ncu` on PATH are the CUDA 11.8 builds (2022.4 / 2022.3) — profiling an Ada card on a 580 driver | pin `/usr/local/cuda-12.6/bin/{nsys,ncu}`. `/opt/nvidia/nsight-systems/2024.5.4` is ARM-only (Jetson cross-profiling) |
| `cmake` on PATH is a broken pip shim (`ModuleNotFoundError: No module named 'cmake'`) | use `/usr/bin/cmake` |
| `kernel.perf_event_paranoid=4` denies even user-space profiling | `sudo sysctl -w kernel.perf_event_paranoid=1` — whitelisted, but **resets on reboot** |
| build cache predated the `llama-baseline-vlm` target | `/usr/bin/cmake -S . -B build` to reconfigure |
| hybrid CPU: `perf stat -d -d -d` schedules every event on both PMUs, half return `<not counted>` | name events `cpu_core/.../` explicitly and `taskset -c 0-15` |
| peak VRAM does not scale with frame count | expected — KV cache preallocated at 576 MiB for `n_ctx=16384`. Size `-c` per config to measure real KV growth |

---

## Open items

- **NVTX phase annotation.** Phase attribution is currently inferred from kernel
  names. Works for the unambiguous ones, breaks for shared kernels (`rms_norm`,
  `k_bin_bcast`, `quantize_q8_1` appear in every phase; `flash_attn_ext_f16`
  spans ViT and prefill). NVTX ranges would let nsys attribute every kernel
  exactly and let `ncu --nvtx-include` filter by phase. ~4 lines in
  `baseline_vlm.cpp` + 2 in `tools/mtmd/mtmd-helper.cpp` at the existing timers.
  Inert outside a profiler.
- **Model inconsistency.** `baseline_vlm.cpp` defaults to 7B BF16 and
  `run_streaming_benchmark.sh` uses 7B BF16, but the benchmark and profiling
  harnesses use 3B q8_0. Fine in isolation, invalid if cross-compared.
- **VRAM sizing.** Size `-c` per config, or compute KV growth analytically, if
  visual-token memory scaling is a result you need.
- **Counter precision.** Split `PERF_EVENTS` into hardware-sized groups if a
  specific ratio needs to be exact rather than multiplexed.
