
**ViT window-attention layer (mean of layers 1-6)**

| step | kernels | GPU time µs | DRAM MB | L2 MB | instructions (M) | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | achieved occupancy |
|---|---|---|---|---|---|---|---|---|---|---|
| stock kernels | 39 | 15,897 | 909.4 | 2534.9 | 400.2 | 57 | 18% | 43% | 45% | 53% |
| + batched window attention | 38 | 14,416 (-9%) | 810.7 | 2234.5 | 336.0 | 56 | 17% | 42% | 45% | 56% |
| + F32 output, bias in epilogue | 24 | 10,845 (-32%) | 647.0 | 1872.9 | 239.1 | 60 | 23% | 42% | 46% | 43% |
| + tuned GEMM table | 24 | 10,704 (-33%) | 593.8 | 1844.4 | 236.1 | 55 | 23% | 43% | 46% | 43% |
| + GEMM-input fusion | 18 | 9,217 (-42%) | 505.0 | 1681.0 | 199.7 | 55 | 26% | 43% | 48% | 37% |
| + F16 vision RoPE | 17 | 8,022 (-50%) | 491.3 | 1659.6 | 190.2 | 61 | 30% | 48% | 54% | 42% |
| + residual in GEMM (every switch) | 16 | 7,870 (-50%) | 485.3 | 1636.0 | 185.4 | 62 | 30% | 48% | 54% | 41% |

*ViT window-attention layer (mean of layers 1-6): per kernel class, stock → every switch*

| class | GPU time µs | kernels | DRAM MB | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | occupancy | top stall (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|
| cast | 3,521 → 183 | 14 → 1 | 163.4 → 11.8 | 46 → 64 | 2% → 2% | 40% → 41% | 33% → 34% | 85% → 84% | long_scoreboard 16.6 → long_scoreboard 15.8 |
| GEMM | 3,051 → 4,576 | 6 → 7 | 239.0 → 304.1 | 78 → 66 | 60% → 51% | 58% → 48% | 74% → 71% | 16% → 17% | wait 2.6 → mio_throttle 3.1 |
| add | 2,427 → 205 | 9 → 1 | 169.4 → 23.6 | 70 → 115 | 2% → 2% | 34% → 32% | 52% → 61% | 89% → 93% | long_scoreboard 27.5 → long_scoreboard 29.4 |
| attention | 1,703 → 258 | 2 → 1 | 121.7 → 23.7 | 71 → 92 | 22% → 8% | 50% → 28% | 49% → 54% | 24% → 25% | wait 1.7 → long_scoreboard 4.9 |
| rope | 1,425 → 704 | 2 → 2 | 23.6 → 23.7 | 17 → 34 | 1% → 1% | 24% → 74% | 12% → 25% | 28% → 95% | long_scoreboard 7.0 → wait 3.6 |
| GEMM split-K | 1,294 → — | 1 → — | 82.1 → — | 63 → — | 40% → — | 38% → — | 67% → — | 17% → — | mio_throttle 4.9 → — |
| norm | 956 → 921 | 2 → 2 | 23.6 → 23.6 | 25 → 26 | 2% → 1% | 34% → 36% | 17% → 15% | 64% → 64% | long_scoreboard 6.3 → long_scoreboard 5.9 |
| copy | 878 → 440 | 2 → 1 | 23.6 → 11.8 | 27 → 27 | 2% → 2% | 66% → 66% | 15% → 15% | 63% → 63% | wait 3.3 → wait 3.3 |
| glu | 642 → 583 | 1 → 1 | 63.0 → 63.0 | 98 → 108 | 1% → 1% | 38% → 43% | 52% → 48% | 85% → 85% | long_scoreboard 18.8 → long_scoreboard 15.7 |

**ViT full-attention layer (layer 7)**

| step | kernels | GPU time µs | DRAM MB | L2 MB | instructions (M) | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | achieved occupancy |
|---|---|---|---|---|---|---|---|---|---|---|
| stock kernels | 38 | 16,458 | 812.3 | 2532.3 | 423.0 | 49 | 19% | 43% | 45% | 52% |
| + batched window attention | 38 | 16,345 (-1%) | 811.5 | 2549.8 | 423.0 | 50 | 19% | 44% | 45% | 52% |
| + F32 output, bias in epilogue | 24 | 12,779 (-22%) | 647.9 | 2170.0 | 326.2 | 51 | 24% | 44% | 47% | 40% |
| + tuned GEMM table | 24 | 12,606 (-23%) | 594.2 | 2139.5 | 323.1 | 47 | 24% | 45% | 46% | 41% |
| + GEMM-input fusion | 18 | 11,134 (-32%) | 505.4 | 1986.7 | 286.8 | 45 | 27% | 45% | 48% | 35% |
| + F16 vision RoPE | 17 | 9,995 (-39%) | 491.6 | 1947.5 | 277.3 | 49 | 30% | 50% | 53% | 39% |
| + residual in GEMM (every switch) | 16 | 9,770 (-41%) | 487.2 | 1950.7 | 272.5 | 50 | 31% | 50% | 53% | 37% |

*ViT full-attention layer (layer 7): per kernel class, stock → every switch*

| class | GPU time µs | kernels | DRAM MB | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | occupancy | top stall (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|
| cast | 3,532 → 185 | 14 → 1 | 163.4 → 11.8 | 46 → 64 | 2% → 2% | 40% → 41% | 33% → 34% | 85% → 84% | long_scoreboard 16.6 → long_scoreboard 15.7 |
| GEMM | 3,079 → 4,589 | 6 → 7 | 238.8 → 304.7 | 78 → 66 | 59% → 51% | 57% → 48% | 73% → 71% | 16% → 17% | wait 2.6 → mio_throttle 3.1 |
| add | 2,447 → 204 | 9 → 1 | 169.4 → 23.6 | 69 → 116 | 2% → 2% | 34% → 31% | 51% → 61% | 89% → 93% | long_scoreboard 27.8 → long_scoreboard 29.5 |
| attention | 2,190 → 2,146 | 1 → 1 | 24.7 → 25.0 | 11 → 12 | 30% → 30% | 52% → 52% | 51% → 51% | 25% → 25% | wait 2.0 → wait 2.0 |
| rope | 1,432 → 699 | 2 → 2 | 23.6 → 23.7 | 17 → 34 | 1% → 1% | 24% → 74% | 12% → 25% | 28% → 95% | long_scoreboard 6.9 → wait 3.6 |
| GEMM split-K | 1,296 → — | 1 → — | 82.1 → — | 63 → — | 40% → — | 38% → — | 67% → — | 17% → — | mio_throttle 4.9 → — |
| norm | 957 → 924 | 2 → 2 | 23.6 → 23.6 | 25 → 26 | 2% → 1% | 34% → 36% | 17% → 15% | 64% → 64% | long_scoreboard 6.2 → long_scoreboard 5.9 |
| copy | 878 → 440 | 2 → 1 | 23.6 → 11.8 | 27 → 27 | 2% → 2% | 66% → 66% | 15% → 15% | 63% → 63% | wait 3.3 → wait 3.3 |
| glu | 647 → 584 | 1 → 1 | 63.0 → 63.0 | 97 → 108 | 1% → 1% | 38% → 43% | 51% → 49% | 85% → 85% | long_scoreboard 18.5 → long_scoreboard 15.6 |

**LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12)**

| step | kernels | GPU time µs | DRAM MB | L2 MB | instructions (M) | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | achieved occupancy |
|---|---|---|---|---|---|---|---|---|---|---|
| stock kernels | 34 | 19,366 | 1022.8 | 5419.3 | 310.7 | 53 | 39% | 48% | 54% | 32% |
| + F32 output, bias in epilogue | 24 | 18,091 (-7%) | 1092.9 | 5449.3 | 263.6 | 60 | 41% | 47% | 59% | 27% |
| + tuned GEMM table | 24 | 12,740 (-34%) | 992.2 | 4213.9 | 238.0 | 78 | 56% | 63% | 58% | 31% |
| + GEMM-input fusion | 18 | 11,547 (-40%) | 912.2 | 4073.8 | 207.8 | 79 | 62% | 66% | 60% | 25% |
| + residual in GEMM (every switch) | 16 | 11,344 (-41%) | 900.6 | 4044.4 | 201.9 | 79 | 63% | 66% | 60% | 24% |

*LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12): per kernel class, stock → every switch*

| class | GPU time µs | kernels | DRAM MB | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | occupancy | top stall (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|
| GEMM split-K | 12,457 → 2,702 | 5 → 1 | 687.3 → 231.3 | 55 → 86 | 51% → 64% | 50% → 64% | 57% → 51% | 16% → 17% | mio_throttle 8.0 → mio_throttle 5.3 |
| cast | 2,843 → 116 | 14 → 1 | 133.7 → 7.3 | 47 → 63 | 2% → 2% | 40% → 41% | 33% → 33% | 86% → 84% | long_scoreboard 16.5 → long_scoreboard 15.3 |
| attention | 2,022 → 2,022 | 1 → 1 | 45.1 → 45.2 | 22 → 22 | 50% → 50% | 50% → 50% | 73% → 73% | 17% → 17% | wait 1.5 → wait 1.5 |
| glu | 798 → 714 | 1 → 1 | 77.6 → 77.6 | 97 → 109 | 1% → 1% | 39% → 44% | 51% → 48% | 86% → 86% | long_scoreboard 18.4 → long_scoreboard 15.1 |
| add | 394 → — | 5 → — | 38.8 → — | 99 → — | 2% → — | 37% → — | 56% → — | 92% → — | long_scoreboard 23.9 → — |
| norm | 334 → 315 | 2 → 2 | 14.7 → 14.7 | 44 → 47 | 1% → 1% | 28% → 32% | 30% → 25% | 65% → 65% | long_scoreboard 10.4 → long_scoreboard 8.2 |
| rope | 282 → 281 | 2 → 2 | 8.4 → 8.4 | 30 → 30 | 1% → 1% | 40% → 40% | 20% → 20% | 38% → 38% | long_scoreboard 3.9 → long_scoreboard 3.9 |
| GEMM | 188 → 5,147 | 2 → 6 | 15.0 → 514.0 | 80 → 100 | 55% → 85% | 51% → 81% | 77% → 67% | 15% → 16% | mio_throttle 2.7 → wait 3.7 |
| copy | 47 → 47 | 2 → 2 | 2.1 → 2.1 | 45 → 45 | 1% → 1% | 52% → 53% | 24% → 24% | 85% → 86% | long_scoreboard 11.1 → long_scoreboard 11.1 |
