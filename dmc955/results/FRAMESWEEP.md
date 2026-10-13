# Per-stage latency vs frame count

`alley_1`, 7B BF16, `-n 128 --ignore-eos`, `-c 16384`, `-ub 512` (default),
`-fa on`, 3 requests per point. GPU-side attribution via `stage_attrib.py`
(streams detected by kernel identity). Generated 2026-09-15 by:

```bash
for F in 1 2 3 4 5 6 7 8 16; do FRAMES=$F ROUNDS=3 bash measure.sh f$F; done
```

Raw: `results/framesweep.csv`, traces `results/trace_f*.nsys-rep`.

## Measured

| N | vis tok | ViT ms | ms/frame | prefill ms | ms/frame | encode wall | decode ms | overlap % | request ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 592 | 51.4 | 51.40 | 99.9 | 99.90 | 158.7 | 2142.7 | 8.2 | 2301.4 |
| 2 | 1184 | 107.3 | 53.65 | 199.7 | 99.85 | 312.1 | 2175.3 | 11.4 | 2487.4 |
| 3 | 1776 | 156.6 | 52.20 | 290.5 | 96.83 | 450.6 | 2154.6 | 12.3 | 2605.2 |
| 4 | 2368 | 208.0 | 52.00 | 385.8 | 96.45 | 595.5 | 2137.7 | 12.9 | 2733.3 |
| 5 | 2960 | 261.1 | 52.22 | 484.7 | 96.94 | 745.6 | 2138.4 | 13.4 | 2884.0 |
| 6 | 3552 | 313.5 | 52.25 | 584.1 | 97.35 | 896.8 | 2148.6 | 13.5 | 3045.3 |
| 7 | 4144 | 367.3 | 52.47 | 685.0 | 97.86 | 1049.1 | 2138.0 | 13.8 | 3187.1 |
| 8 | 4736 | 419.3 | 52.41 | 786.9 | 98.36 | 1202.4 | 2142.4 | 14.0 | 3344.9 |
| 16 | 9472 | 845.2 | 52.83 | 1636.5 | 102.28 | 2470.6 | 2230.3 | 14.5 | 4700.9 |

## Fitted scaling laws

```
ViT     = -2.02  +  52.843 N                       R2 = 0.999945     LINEAR
prefill = +8.25  +  92.593 N  +  0.5739 N^2        R2 = 0.999985     QUADRATIC
          (linear fit only reaches R2 = 0.999210)
decode  = 2129.4 +   4.678 N                       R2 = 0.48   (flat within noise)
```

- **ViT is linear to five decimal places**, 52.84 ms/frame, intercept ~0. It does
  no cross-frame work at all — each frame is an independent still image. This is
  also the direct signature of the missing temporal patch merging.
- **Prefill has a real N^2 term**: 0.5739 ms/frame^2. That is attention over all
  prior visual tokens, and it is what makes prefill super-linear. The quadratic
  term contributes 147 ms of prefill's 1636 ms at N=16 (9.0%). Adding the
  quadratic lifts R2 from 0.99921 to 0.99999, so the curvature is real, not fit
  noise.
- **Marginal cost of one more frame** grows: prefill 93.74 ms at N=1 -> 110.96 ms
  at N=16, **+18.4%**. ViT's marginal cost is constant at 52.84 ms.
- **Decode is flat at fixed token count**, 2137-2175 ms for N=1..8, rising to
  2230 ms at N=16 (+3.3% over the whole range). That is the isolated KV-context
  effect; the poor R2 just reflects that the trend is small relative to noise.

Cross-check against the independent 13-frame characterization: the fits predict
ViT 684.9 ms (measured 670.7, -2.1%) and prefill 1308.9 ms (measured 1276.9,
-2.4%). Different sampling policy and run, so agreement to ~2% is good.

## The stage mix inverts with frame count

| N | ViT | prefill | decode |
|---|---|---|---|
| 1 | 2.2% | 4.3% | **93.1%** |
| 16 | 18.0% | 34.8% | **47.4%** |

**This is the most consequential result for planning.** At one frame the request
is essentially pure decode and nothing in the encode path is worth optimising. By
16 frames encode is 52.6% of the request. Every encode-side optimisation
(pipelining, ubatch sizing, separator fusion, L2 work) scales its value with frame
count; decode-side work does not.

Quote the frame count with any encode-side speedup, always.

## Overlap grows with N but saturates low

ViT/prefill overlap rises 8.2% -> 14.5% across the sweep and is clearly
asymptotic, not trending toward anything useful. It grows because overlap happens
at frame boundaries and there are N-1 of them, so it scales with N at the same
rate ViT does. **The pipelining opportunity (T4) is therefore not something that
solves itself at larger N** — the fraction is stuck near 14%.

## Caveats

- **All points run at `-ub 512`, which is llama.cpp's default — this is the
  intended baseline, not a contaminated one.** The research question is how much
  change stock llama.cpp needs, so the stock configuration is the correct
  reference. Every frame therefore pays the 512+80 ubatch split
  (`GAP3_PREFILL.md`), and that is a property of the engine as shipped.

  The caveat is about **attribution, not validity**: 92.593 ms/frame is a property
  of model + hardware + *configuration*, and `-ub 640` (a CLI flag, no code change)
  moves it. Write "92.6 ms/frame at stock defaults" and the number is exact; write
  "prefill costs 92.6 ms/frame on a 4090" and it is wrong. The N^2 term is
  unaffected either way — the split is per-frame, so it contributes linearly.

  A `-ub 640` sweep belongs beside this one as the **rung-1 optimisation curve**
  (does the 90 ms/request win scale with frame count?), not as a replacement
  baseline.
- nsys tracing overhead is included; these are *relative* stage numbers, not
  wall-clock latency. Use `bench_serving.sh` for wall clock.
- One outlier: decode at N=2 reads 2175.3 ms, ~1.6% above its neighbours. Within
  run-to-run variance, not a trend.
- N=16 needs 9,472 visual tokens; `-c 16384` caps the sweep at 27 frames. Past
  that raise `NCTX`.
