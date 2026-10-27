
**ViT window-attention layer (mean of layers 1-6)**

| step | kernels | GPU time µs | DRAM MB | L2 MB | instructions (M) | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | achieved occupancy |
|---|---|---|---|---|---|---|---|---|---|---|
| stock kernels | 39 | 95,122 | 1034.7 | 2598.5 | 404.7 | 11 | 24% | 58% | 43% | 48% |
| + batched window attention | 38 | 83,763 (-12%) | 938.9 | 2279.0 | 340.5 | 11 | 24% | 57% | 42% | 51% |
| + F32 output, bias in epilogue | 24 | 70,597 (-26%) | 832.5 | 1972.0 | 244.4 | 12 | 29% | 55% | 46% | 40% |
| + tuned GEMM table | 24 | 65,620 (-31%) | 673.9 | 1842.8 | 236.1 | 10 | 29% | 56% | 45% | 41% |
| + GEMM-input fusion | 18 | 58,188 (-39%) | 580.2 | 1679.4 | 199.7 | 10 | 33% | 56% | 48% | 36% |
| + F16 vision RoPE | 17 | 53,999 (-43%) | 569.0 | 1658.1 | 190.2 | 11 | 35% | 59% | 51% | 40% |
| + residual in GEMM (every switch) | 16 | 53,172 (-44%) | 565.7 | 1634.5 | 185.4 | 11 | 36% | 59% | 51% | 39% |

*ViT window-attention layer (mean of layers 1-6): per kernel class, stock → every switch*

| class | GPU time µs | kernels | DRAM MB | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | occupancy | top stall (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|
| cast | 17,950 → 945 | 14 → 1 | 163.4 → 11.8 | 9 → 12 | 2% → 2% | 56% → 56% | 21% → 21% | 84% → 84% | long_scoreboard 8.9 → long_scoreboard 8.9 |
| GEMM split-K | 16,835 → — | 5 → — | 184.2 → — | 11 → — | 61% → — | 60% → — | 70% → — | 16% → — | wait 2.7 → — |
| GEMM | 15,477 → 32,238 | 2 → 7 | 259.9 → 384.5 | 17 → 12 | 55% → 58% | 55% → 57% | 77% → 72% | 17% → 17% | mio_throttle 2.6 → mio_throttle 2.6 |
| attention | 12,477 → 1,138 | 2 → 1 | 124.0 → 23.7 | 10 → 21 | 24% → 14% | 57% → 50% | 55% → 47% | 25% → 24% | wait 1.8 → wait 1.7 |
| add | 9,963 → 865 | 9 → 1 | 169.3 → 23.6 | 17 → 27 | 4% → 4% | 67% → 65% | 36% → 45% | 89% → 90% | long_scoreboard 9.7 → long_scoreboard 10.5 |
| copy | 6,953 → 3,475 | 2 → 1 | 23.6 → 11.8 | 3 → 3 | 2% → 2% | 70% → 70% | 6% → 6% | 63% → 63% | wait 3.3 → wait 3.3 |
| norm | 6,434 → 6,244 | 2 → 2 | 23.6 → 23.6 | 4 → 4 | 2% → 1% | 42% → 43% | 17% → 18% | 63% → 63% | long_scoreboard 3.7 → long_scoreboard 3.7 |
| rope | 6,107 → 5,395 | 2 → 2 | 23.6 → 23.7 | 4 → 4 | 1% → 1% | 46% → 79% | 11% → 10% | 37% → 95% | wait 2.6 → not_selected 3.7 |
| glu | 2,927 → 2,872 | 1 → 1 | 63.0 → 63.0 | 22 → 22 | 2% → 2% | 63% → 64% | 36% → 31% | 85% → 85% | long_scoreboard 7.6 → long_scoreboard 7.2 |

**ViT full-attention layer (layer 7)**

| step | kernels | GPU time µs | DRAM MB | L2 MB | instructions (M) | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | achieved occupancy |
|---|---|---|---|---|---|---|---|---|---|---|
| stock kernels | 38 | 99,031 | 935.6 | 2671.3 | 427.5 | 9 | 26% | 58% | 44% | 47% |
| + batched window attention | 38 | 99,039 (+0%) | 966.6 | 2671.3 | 427.5 | 10 | 26% | 58% | 44% | 47% |
| + F32 output, bias in epilogue | 24 | 83,036 (-16%) | 836.0 | 2358.5 | 331.4 | 10 | 30% | 56% | 48% | 38% |
| + tuned GEMM table | 24 | 80,892 (-18%) | 676.0 | 2234.2 | 323.1 | 8 | 30% | 57% | 47% | 38% |
| + GEMM-input fusion | 18 | 73,437 (-26%) | 584.3 | 2071.2 | 286.8 | 8 | 33% | 57% | 50% | 34% |
| + F16 vision RoPE | 17 | 69,240 (-30%) | 567.9 | 2045.9 | 277.3 | 8 | 35% | 59% | 52% | 37% |
| + residual in GEMM (every switch) | 16 | 68,415 (-31%) | 568.1 | 2026.2 | 272.5 | 8 | 36% | 59% | 52% | 36% |

*ViT full-attention layer (layer 7): per kernel class, stock → every switch*

| class | GPU time µs | kernels | DRAM MB | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | occupancy | top stall (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|
| cast | 17,959 → 945 | 14 → 1 | 163.4 → 11.8 | 9 → 12 | 2% → 2% | 56% → 56% | 21% → 21% | 84% → 83% | long_scoreboard 8.9 → long_scoreboard 8.9 |
| GEMM split-K | 16,830 → — | 5 → — | 184.7 → — | 11 → — | 61% → — | 60% → — | 70% → — | 16% → — | wait 2.7 → — |
| attention | 16,369 → 16,386 | 1 → 1 | 23.7 → 23.7 | 1 → 1 | 33% → 33% | 58% → 58% | 56% → 56% | 25% → 25% | wait 2.0 → wait 2.0 |
| GEMM | 15,480 → 32,226 | 2 → 7 | 260.6 → 386.9 | 17 → 12 | 55% → 58% | 55% → 57% | 77% → 72% | 17% → 17% | mio_throttle 2.6 → mio_throttle 2.6 |
| add | 9,963 → 867 | 9 → 1 | 169.3 → 23.6 | 17 → 27 | 4% → 4% | 67% → 65% | 36% → 45% | 89% → 90% | long_scoreboard 9.7 → long_scoreboard 10.5 |
| copy | 6,954 → 3,477 | 2 → 1 | 23.6 → 11.8 | 3 → 3 | 2% → 2% | 70% → 70% | 6% → 6% | 63% → 63% | wait 3.3 → wait 3.3 |
| norm | 6,434 → 6,245 | 2 → 2 | 23.6 → 23.6 | 4 → 4 | 2% → 1% | 42% → 43% | 17% → 18% | 63% → 63% | long_scoreboard 3.7 → long_scoreboard 3.7 |
| rope | 6,115 → 5,393 | 2 → 2 | 23.6 → 23.7 | 4 → 4 | 1% → 1% | 46% → 79% | 11% → 10% | 37% → 95% | wait 2.6 → not_selected 3.7 |
| glu | 2,926 → 2,874 | 1 → 1 | 63.0 → 63.0 | 22 → 22 | 2% → 2% | 63% → 65% | 36% → 31% | 85% → 85% | long_scoreboard 7.6 → long_scoreboard 7.2 |

**LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12)**

| step | kernels | GPU time µs | DRAM MB | L2 MB | instructions (M) | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | achieved occupancy |
|---|---|---|---|---|---|---|---|---|---|---|
| stock kernels | 34 | 120,248 | 1562.3 | 5437.5 | 314.7 | 13 | 47% | 57% | 38% | 29% |
| + F32 output, bias in epilogue | 24 | 112,560 (-6%) | 1607.3 | 5516.0 | 267.2 | 14 | 50% | 57% | 41% | 25% |
| + tuned GEMM table | 24 | 85,386 (-29%) | 1543.4 | 4213.1 | 238.0 | 18 | 67% | 75% | 45% | 27% |
| + GEMM-input fusion | 18 | 79,202 (-34%) | 1468.7 | 4073.0 | 207.8 | 19 | 72% | 77% | 47% | 23% |
| + residual in GEMM (every switch) | 16 | 78,266 (-35%) | 1453.4 | 4043.6 | 201.9 | 19 | 73% | 77% | 47% | 22% |

*LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12): per kernel class, stock → every switch*

| class | GPU time µs | kernels | DRAM MB | DRAM GB/s | tensor pipe active | SM throughput | memory throughput | occupancy | top stall (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|
| GEMM split-K | 80,128 → 16,564 | 5 → 1 | 1205.7 → 383.3 | 15 → 23 | 59% → 86% | 59% → 84% | 39% → 44% | 17% → 17% | mio_throttle 6.3 → wait 3.9 |
| attention | 15,327 → 15,404 | 1 → 1 | 56.1 → 56.7 | 4 → 4 | 51% → 51% | 50% → 50% | 58% → 58% | 17% → 17% | wait 1.5 → wait 1.5 |
| cast | 14,810 → 591 | 14 → 1 | 133.7 → 7.3 | 9 → 12 | 2% → 2% | 56% → 56% | 20% → 20% | 84% → 84% | long_scoreboard 8.9 → long_scoreboard 8.8 |
| glu | 3,654 → 3,505 | 1 → 1 | 77.6 → 77.6 | 21 → 22 | 2% → 2% | 63% → 65% | 35% → 31% | 85% → 85% | long_scoreboard 7.6 → long_scoreboard 7.2 |
| norm | 1,965 → 1,952 | 2 → 2 | 14.7 → 14.7 | 7 → 8 | 2% → 1% | 40% → 41% | 20% → 20% | 65% → 65% | long_scoreboard 4.9 → long_scoreboard 4.5 |
| add | 1,786 → — | 5 → — | 38.8 → — | 22 → — | 4% → — | 66% → — | 39% → — | 90% → — | long_scoreboard 10.0 → — |
| rope | 1,381 → 1,373 | 2 → 2 | 8.4 → 8.4 | 6 → 6 | 1% → 1% | 50% → 50% | 16% → 16% | 39% → 39% | wait 3.2 → wait 3.2 |
| GEMM | 969 → 38,652 | 2 → 6 | 25.1 → 903.3 | 26 → 23 | 90% → 90% | 79% → 88% | 53% → 48% | 14% → 16% | wait 3.3 → wait 3.7 |
| copy | 228 → 225 | 2 → 2 | 2.1 → 2.1 | 9 → 9 | 2% → 2% | 71% → 71% | 17% → 17% | 85% → 85% | long_scoreboard 6.8 → long_scoreboard 6.8 |
