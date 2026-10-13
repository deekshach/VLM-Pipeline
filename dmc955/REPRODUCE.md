# Reproducing the vanilla VLM characterization

Result: per-stage latency (ViT encode / LLM prefill / decode) for the unoptimized
Qwen2.5-VL pipeline on llama.cpp, 460 measured requests, 2026-08-31.

Output lives in `characterization/`.

---

## One command

```bash
cd /data/spring26/CPP/llama.cpp/dmc955

# 0. build (note: cmake on PATH is a broken pip shim, use /usr/bin/cmake)
bash profile.sh build

# 1. pre-flight — GPU clocks and real contention
bash profile.sh gpustate

# 2. the sweep (~33 min)
VIDS=$(cd /data/spring26/CPP/modelsRepo/MPI_bkp/MPI-Sintel/training/clean \
       && ls -d */ | tr -d '/' | paste -sd,)
bash bench_serving.sh --videos "$VIDS" --rounds 21 --fps 2.0 --source-fps 8.0

# 3. export CSVs
S=serving_results/serve_7B-BF16_faon_fps2.0@src8.0_r21
python3 dump_characterization.py "$S" characterization
```

Step 2 prints the analysis itself; `bench_serving_summarize.py "$S"` re-prints it
from saved data without re-running anything.

---

## Files

| file | role |
|---|---|
| `baseline_vlm.cpp` | the vanilla VLM -> binary `llama-baseline-vlm` |
| `model_config.sh` | **the only** definition of which weights are used |
| `bench_serving.sh` | driver: one loaded model, back-to-back request queue, GPU trace |
| `bench_serving_summarize.py` | per-stage stats, cold-start, tail, drift |
| `dump_characterization.py` | exports the CSVs |
| `profile.sh` | microarchitecture (nsys / perf / ncu) — not used for these numbers |

---

## Settings, and why each one

| setting | value | why |
|---|---|---|
| model | Qwen2.5-VL-7B-Instruct-BF16 | pinned in `model_config.sh`; decode is bandwidth-bound so quantization moves the dominant phase ~2x — it must not vary |
| mmproj | mmproj-BF16.gguf | matches the 7B tower |
| `--fps 2.0` | constant-rate | Qwen's reference policy: frame count follows duration |
| `--source-fps 8.0` | reinterpret capture rate | Sintel is 24 fps and only 0.8-2.1 s long; assuming 8 fps stretches clips to 2.5-6.25 s so 2 fps sampling yields 5-13 frames instead of 2-4 |
| `--serve-rounds 21` | 21 | round 0 is discarded as warmup -> exactly 20 measured rounds per video |
| `--ignore-eos` | on (`IGNORE_EOS=1`) | **required.** 7B BF16 is numerically non-deterministic; generation length varied 9-128 tokens for identical input. Suppressing EOG forces exactly `n_predict` tokens and drops decode sd from meaningless to 2.3% |
| `-n 128` | 128 | fixed generation length |
| `--temp 0 --seed 42` | greedy | identical kernel sequence every run |
| `-fa on` | flash attention | production config. **Compare against `-fa off` only if the variant forces it off** — that alone costs +77% prefill |
| `-c 16384` | context | 13 frames x 592 = 7696 visual tokens, plus headroom |
| `-ngl 99` | all layers on GPU | |
| `--no-flow-analysis` | on | skips reporting-only optical flow: ~20% of the CPU profile and 89% of branch misses, and it never touched inference |
| CUDA graphs | **enabled** | production-accurate. `nsys --cuda-graph-trace=node` gives per-kernel attribution without disabling them |

Frame counts are derived, not fixed: 50-frame clips -> 13 frames, and the four
short clips give 5, 5, 8, 10 — a free four-point scaling curve inside one run.

---

## Environment

| | |
|---|---|
| GPU | RTX 4090, sm_89, driver 580.173.02 |
| CPU | i9-14900K (P-cores 0-15, E-cores 16-31) |
| kernel | 6.8.0-138-generic |
| llama.cpp | b55dcdef5 (dirty) |
| dataset | MPI-Sintel `training/clean`, 23 videos, 20-50 frames each |

Traps that silently produce wrong data:

- `cmake` on PATH is a broken pip shim -> use `/usr/bin/cmake`
- `nsys`/`ncu` on PATH are CUDA 11.8-era (2022) -> use `/usr/local/cuda-12.6/bin/`
- `perf` needs `sudo sysctl -w kernel.perf_event_paranoid=1`, **resets on reboot**
- GPU clocks idle at 210 MHz of 3105; warmup covers the ramp, `nvidia-smi -lgc 2100,2100` removes drift entirely

---

## What the numbers should look like

```
 frames  vis tok  vids    ViT ms  /frame   prefill  /frame     TTFT   decode  tok/s
      5     2960     2     311.9    62.4     306.3    61.3    742.1   2153.6   59.5
      8     4736     1     500.2    62.5     501.1    62.6   1189.8   2161.2   59.2
     10     5920     1     629.6    63.0     640.1    64.0   1504.0   2176.1   58.9
     13     7696    19     815.0    62.7     846.7    65.1   1962.4   2196.5   58.3

ViT 749.5+/-14.1 (18.3%) | prefill 775.7+/-15.2 (18.9%) | decode 2190.3+/-4.6 (53.4%)
of a 4101 ms request. All 460 requests at exactly 128 tokens.
```

Sanity checks that must hold, or something is wrong:

- every measured request has `n_tokens == 128` (else `--ignore-eos` did not apply)
- `residual_ms` is small and **positive** — it is `encode_ms` minus the log-parsed
  ViT and prefill, i.e. a cross-check between two independent sources. Strongly
  negative means the log parse picked up iterations the timer did not
- SM clock median is the same in both halves (else thermal throttling)

---

## Caveats

- **EOS suppression makes generated text degenerate** past the natural stopping
  point. This is a latency measurement; ignore `shannon_entropy`.
- **llama.cpp does not implement Qwen's temporal patch merging.** Each frame is
  encoded as an independent still image (verified: N frames -> N encoder calls,
  592 tokens each, no step at even N). The reference would emit half these tokens.
  State this when reporting.
- `decode_ms` is comparable across videos **only** because generation length is
  fixed. Without `--ignore-eos` it is not.
