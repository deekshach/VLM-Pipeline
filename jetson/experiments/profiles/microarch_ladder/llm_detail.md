**LLM prefill GEMMs per projection** (M = 512 tokens, mean of layers 1-12; split-K = grid z)

| projection (N x K) | step | tile | grid N x M x split | waves | GPU time µs | tensor pipe | L2 hit | L2 MB | DRAM MB | top stalls (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q (3584 x 3584) | base | 128x128 | 28 x 4 x 2 | 7.0 | 434 | 77% | 90.3% | 217 | 39 | wait 3.2, math_pipe_throttle 1.4, mio_throttle 1.0 |
| Q (3584 x 3584) | table | 256x128 | 14 x 4 x 1 | 3.5 | 439 | 86% | 75.0% | 162 | 40 | wait 3.7, math_pipe_throttle 3.3, selected 1.0 |
| Q (3584 x 3584) | design | 256x128 | 14 x 4 x 1 | 3.5 | 439 | 86% | 75.0% | 162 | 40 | wait 3.7, math_pipe_throttle 3.4, selected 1.0 |
| K (512 x 3584) | base | 128x64 | 4 x 8 x 1 | 1.0 | 94 | 55% | 91.6% | 45 | 7 | mio_throttle 2.7, wait 2.3, short_scoreboard 1.6 |
| K (512 x 3584) | table | 128x128 | 16 x 1 x 1 | 1.0 | 70 | 74% | 75.8% | 31 | 7 | wait 3.0, selected 1.0, long_scoreboard 0.5 |
| K (512 x 3584) | design | 128x128 | 16 x 1 x 1 | 1.0 | 72 | 73% | 75.8% | 31 | 7 | wait 3.0, selected 1.0, long_scoreboard 0.5 |
| V (512 x 3584) | base | 128x64 | 4 x 8 x 1 | 1.0 | 94 | 55% | 91.6% | 45 | 7 | mio_throttle 2.6, wait 2.3, short_scoreboard 1.6 |
| V (512 x 3584) | table | 128x128 | 16 x 1 x 1 | 1.0 | 71 | 74% | 75.8% | 31 | 7 | wait 3.0, selected 1.0, long_scoreboard 0.5 |
| V (512 x 3584) | design | 128x128 | 16 x 1 x 1 | 1.0 | 71 | 74% | 75.8% | 31 | 7 | wait 3.0, selected 1.0, long_scoreboard 0.5 |
| O (3584 x 3584) | base | 128x128 | 28 x 4 x 2 | 7.0 | 434 | 77% | 90.2% | 217 | 39 | wait 3.2, math_pipe_throttle 1.4, mio_throttle 1.0 |
| O (3584 x 3584) | table | 256x128 | 14 x 4 x 1 | 3.5 | 433 | 86% | 75.0% | 162 | 40 | wait 3.8, math_pipe_throttle 3.4, selected 1.0 |
| O (3584 x 3584) | design | 256x128 | 14 x 4 x 1 | 3.5 | 460 | 80% | 71.7% | 169 | 48 | wait 3.7, math_pipe_throttle 3.4, selected 1.0 |
| gate (18944 x 3584) | base | 128x128 | 148 x 4 x 2 | 37.0 | 2,455 | 82% | 91.4% | 1145 | 177 | wait 3.2, math_pipe_throttle 1.8, selected 1.0 |
| gate (18944 x 3584) | table | 256x128 | 74 x 4 x 1 | 18.5 | 2,029 | 86% | 75.9% | 854 | 206 | wait 3.8, math_pipe_throttle 3.4, selected 1.0 |
| gate (18944 x 3584) | design | 256x128 | 74 x 4 x 1 | 18.5 | 2,065 | 86% | 75.9% | 854 | 206 | wait 3.8, math_pipe_throttle 3.4, selected 1.0 |
| up (18944 x 3584) | base | 128x128 | 148 x 4 x 2 | 37.0 | 2,385 | 82% | 91.4% | 1145 | 178 | wait 3.2, math_pipe_throttle 1.8, selected 1.0 |
| up (18944 x 3584) | table | 256x128 | 74 x 4 x 1 | 18.5 | 2,032 | 86% | 75.9% | 854 | 206 | wait 3.8, math_pipe_throttle 3.4, selected 1.0 |
| up (18944 x 3584) | design | 256x128 | 74 x 4 x 1 | 18.5 | 2,040 | 86% | 75.9% | 854 | 206 | wait 3.8, math_pipe_throttle 3.4, selected 1.0 |
| down (3584 x 18944) | base | 128x128 | 28 x 4 x 9 | 31.5 | 6,748 | 25% | 87.6% | 1149 | 255 | mio_throttle 14.1, barrier 4.0, wait 3.2 |
| down (3584 x 18944) | table | 256x128 | 14 x 4 x 2 | 7.0 | 2,662 | 65% | 73.3% | 837 | 225 | mio_throttle 5.3, wait 3.9, math_pipe_throttle 2.5 |
| down (3584 x 18944) | design | 256x128 | 14 x 4 x 2 | 7.0 | 2,702 | 64% | 72.4% | 844 | 231 | mio_throttle 5.3, wait 3.9, math_pipe_throttle 2.5 |

**LLM prefill attention** (flash_attn_ext_f16, one kernel per layer, mean of layers 1-12)

| step | GPU time µs | tensor pipe | issue active | achieved occupancy | regs/thread | smem/block KB | occupancy limit (blocks/SM: regs, smem, warps) | L1 hit | L2 hit | L2 MB | DRAM MB | DRAM GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 2,022 | 50% | 38% | 17% | 212 | 36 | 2, 2, 12 | 0.3% | 96.5% | 951 | 45 | 22 |
| table | 2,021 | 50% | 38% | 17% | 212 | 36 | 2, 2, 12 | 0.3% | 96.5% | 951 | 45 | 22 |
| design | 2,022 | 50% | 38% | 17% | 212 | 36 | 2, 2, 12 | 0.3% | 96.5% | 951 | 45 | 22 |

*Attention warp stalls (warps per issue, base → design)*

| stall reason | base | design |
|---|---|---|
| wait | 1.53 | 1.53 |
| selected | 1.00 | 1.00 |
| short_scoreboard | 0.53 | 0.53 |
| math_pipe_throttle | 0.52 | 0.52 |
| mio_throttle | 0.48 | 0.48 |
| long_scoreboard | 0.39 | 0.39 |
| not_selected | 0.35 | 0.35 |
| dispatch_stall | 0.20 | 0.20 |
| barrier | 0.15 | 0.14 |
| lg_throttle | 0.05 | 0.05 |
| **sum** | 5.24 | 5.23 |
