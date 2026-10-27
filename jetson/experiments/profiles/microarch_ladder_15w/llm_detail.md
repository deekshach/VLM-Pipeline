**LLM prefill GEMMs per projection** (M = 512 tokens, mean of layers 1-12; split-K = grid z)

| projection (N x K) | step | tile | grid N x M x split | waves | GPU time µs | tensor pipe | L2 hit | L2 MB | DRAM MB | top stalls (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q (3584 x 3584) | base | 128x128 | 28 x 4 x 3 | 28.0 | 3,104 | 86% | 85.6% | 224 | 57 | wait 3.2, math_pipe_throttle 1.8, selected 1.0 |
| Q (3584 x 3584) | table | 256x128 | 14 x 4 x 1 | 9.3 | 3,185 | 90% | 56.9% | 162 | 70 | wait 3.7, math_pipe_throttle 3.4, selected 1.0 |
| Q (3584 x 3584) | design | 256x128 | 14 x 4 x 1 | 9.3 | 3,184 | 90% | 56.9% | 162 | 70 | wait 3.7, math_pipe_throttle 3.4, selected 1.0 |
| K (512 x 3584) | base | 128x128 | 4 x 4 x 1 | 1.3 | 484 | 90% | 78.9% | 30 | 13 | wait 3.3, math_pipe_throttle 2.1, selected 1.0 |
| K (512 x 3584) | table | 128x128 | 16 x 1 x 1 | 2.7 | 519 | 85% | 50.8% | 30 | 15 | wait 3.0, selected 1.0, long_scoreboard 0.1 |
| K (512 x 3584) | design | 128x128 | 16 x 1 x 1 | 2.7 | 519 | 85% | 50.3% | 30 | 15 | wait 3.0, selected 1.0, long_scoreboard 0.1 |
| V (512 x 3584) | base | 128x128 | 4 x 4 x 1 | 1.3 | 484 | 90% | 79.1% | 30 | 12 | wait 3.3, math_pipe_throttle 2.1, selected 1.0 |
| V (512 x 3584) | table | 128x128 | 16 x 1 x 1 | 2.7 | 520 | 85% | 50.1% | 30 | 15 | wait 3.0, selected 1.0, long_scoreboard 0.1 |
| V (512 x 3584) | design | 128x128 | 16 x 1 x 1 | 2.7 | 520 | 85% | 49.9% | 30 | 15 | wait 3.0, selected 1.0, long_scoreboard 0.1 |
| O (3584 x 3584) | base | 128x128 | 28 x 4 x 3 | 28.0 | 3,105 | 86% | 85.6% | 224 | 57 | wait 3.2, math_pipe_throttle 1.8, selected 1.0 |
| O (3584 x 3584) | table | 256x128 | 14 x 4 x 1 | 9.3 | 3,144 | 91% | 56.9% | 162 | 70 | wait 3.8, math_pipe_throttle 3.5, selected 1.0 |
| O (3584 x 3584) | design | 256x128 | 14 x 4 x 1 | 9.3 | 3,206 | 89% | 54.4% | 169 | 77 | wait 3.7, math_pipe_throttle 3.4, selected 1.0 |
| gate (18944 x 3584) | base | 128x128 | 148 x 4 x 3 | 148.0 | 16,163 | 86% | 85.3% | 1184 | 309 | wait 3.2, math_pipe_throttle 1.8, selected 1.0 |
| gate (18944 x 3584) | table | 256x128 | 74 x 4 x 1 | 49.3 | 15,618 | 91% | 57.5% | 854 | 363 | wait 3.8, math_pipe_throttle 3.5, selected 1.0 |
| gate (18944 x 3584) | design | 256x128 | 74 x 4 x 1 | 49.3 | 15,614 | 91% | 57.5% | 854 | 363 | wait 3.8, math_pipe_throttle 3.5, selected 1.0 |
| up (18944 x 3584) | base | 128x128 | 148 x 4 x 3 | 148.0 | 16,162 | 86% | 85.3% | 1184 | 309 | wait 3.2, math_pipe_throttle 1.8, selected 1.0 |
| up (18944 x 3584) | table | 256x128 | 74 x 4 x 1 | 49.3 | 15,613 | 91% | 57.5% | 854 | 363 | wait 3.8, math_pipe_throttle 3.5, selected 1.0 |
| up (18944 x 3584) | design | 256x128 | 74 x 4 x 1 | 49.3 | 15,609 | 91% | 57.5% | 854 | 363 | wait 3.8, math_pipe_throttle 3.5, selected 1.0 |
| down (3584 x 18944) | base | 128x128 | 28 x 4 x 3 | 28.0 | 41,595 | 33% | 78.2% | 1105 | 473 | mio_throttle 11.8, barrier 3.4, wait 3.3 |
| down (3584 x 18944) | table | 256x128 | 14 x 4 x 2 | 18.7 | 16,487 | 86% | 55.1% | 837 | 376 | wait 3.9, math_pipe_throttle 3.5, mio_throttle 1.3 |
| down (3584 x 18944) | design | 256x128 | 14 x 4 x 2 | 18.7 | 16,564 | 86% | 54.6% | 844 | 383 | wait 3.9, math_pipe_throttle 3.5, mio_throttle 1.3 |

**LLM prefill attention** (flash_attn_ext_f16, one kernel per layer, mean of layers 1-12)

| step | GPU time µs | tensor pipe | issue active | achieved occupancy | regs/thread | smem/block KB | occupancy limit (blocks/SM: regs, smem, warps) | L1 hit | L2 hit | L2 MB | DRAM MB | DRAM GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 15,327 | 51% | 38% | 17% | 212 | 36 | 2, 2, 12 | 0.3% | 95.5% | 951 | 56 | 4 |
| table | 15,344 | 51% | 38% | 17% | 212 | 36 | 2, 2, 12 | 0.3% | 95.5% | 951 | 57 | 4 |
| design | 15,404 | 51% | 38% | 17% | 212 | 36 | 2, 2, 12 | 0.3% | 95.5% | 951 | 57 | 4 |

*Attention warp stalls (warps per issue, base → design)*

| stall reason | base | design |
|---|---|---|
| wait | 1.52 | 1.52 |
| selected | 1.00 | 1.00 |
| short_scoreboard | 0.53 | 0.53 |
| math_pipe_throttle | 0.50 | 0.51 |
| mio_throttle | 0.48 | 0.48 |
| not_selected | 0.35 | 0.35 |
| long_scoreboard | 0.29 | 0.29 |
| dispatch_stall | 0.19 | 0.20 |
| barrier | 0.12 | 0.13 |
| lg_throttle | 0.05 | 0.05 |
| **sum** | 5.07 | 5.08 |
