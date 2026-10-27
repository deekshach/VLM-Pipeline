
## ViT window-attention layer (mean of layers 1-6)

Stock layer: 15,897 µs.

### + batched window attention (layer 15,897 → 14,416 µs, -9.3%)

*attention: -1,444 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 1.0 |
| GPU time µs / layer | 1,703 | 259 |
| kernel (largest) | flash_attn_ext_f16 / flash_attn_mask_to_KV_max | flash_attn_ext_f16 |
| launch (largest) | 576x1x1 x 128 thr, 12.0 waves | 576x1x1 x 128 thr, 12.0 waves |
| SM throughput | 50% | 29% |
| memory throughput | 49% | 54% |
| L2 throughput | 47% | 54% |
| issue slots used | 50% | 29% |
| tensor pipe | 22% | 8% |
| FMA / ALU / LSU pipe | 20% / 42% / 30% | 11% / 24% / 14% |
| occupancy achieved / theoretical | 24% / 26% (registers-limited) | 25% / 25% (registers-limited) |
| warp latency per issue (cycles) | 5.7 | 10.1 |
| L1 / L2 hit | 7% / 64% | 10% / 39% |
| DRAM MB / layer | 121.7 | 23.7 |
| DRAM GB/s | 71 | 91 |
| L2 MB / layer | 338.8 | 38.4 |
| instructions M / layer | 69.6 | 5.4 |
| top stalls (warps per issue) | wait 1.7, long_scoreboard 1.3, selected 1.0 | long_scoreboard 4.9, wait 1.6, selected 1.0 |


### + F32 output, bias in epilogue (layer 14,416 → 10,845 µs, -24.8%)

*GEMM: +291 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 6.0 | 6.0 |
| GPU time µs / layer | 3,038 | 3,328 |
| kernel (largest) | cutlass::Kernel2 / ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_sta | cutlass::Kernel2 / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ |
| launch (largest) | 144x2x1 x 256 thr, 18.0 waves | 144x2x1 x 256 thr, 18.0 waves |
| SM throughput | 58% | 53% |
| memory throughput | 74% | 72% |
| L2 throughput | 58% | 58% |
| issue slots used | 19% | 19% |
| tensor pipe | 60% | 55% |
| FMA / ALU / LSU pipe | 1% / 10% / 19% | 1% / 10% / 19% |
| occupancy achieved / theoretical | 16% / 17% (registers-limited) | 16% / 17% (registers-limited) |
| warp latency per issue (cycles) | 10.1 | 10.1 |
| L1 / L2 hit | 50% / 76% | 50% / 77% |
| DRAM MB / layer | 238.3 | 251.7 |
| DRAM GB/s | 78 | 76 |
| L2 MB / layer | 1010.0 | 1067.3 |
| instructions M / layer | 46.5 | 50.7 |
| top stalls (warps per issue) | wait 2.6, mio_throttle 2.4, math_pipe_throttle 1.6 | wait 2.7, mio_throttle 2.3, math_pipe_throttle 1.3 |

*GEMM split-K: +89 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 1.0 | 1.0 |
| GPU time µs / layer | 1,291 | 1,380 |
| kernel (largest) | cutlass::Kernel2 | cutlass::Kernel2 |
| launch (largest) | 72x2x2 x 256 thr, 18.0 waves | 72x2x2 x 256 thr, 18.0 waves |
| SM throughput | 38% | 36% |
| memory throughput | 68% | 65% |
| L2 throughput | 49% | 50% |
| issue slots used | 13% | 13% |
| tensor pipe | 41% | 38% |
| FMA / ALU / LSU pipe | 1% / 6% / 15% | 1% / 7% / 15% |
| occupancy achieved / theoretical | 17% / 17% (registers-limited) | 17% / 17% (registers-limited) |
| warp latency per issue (cycles) | 14.3 | 14.4 |
| L1 / L2 hit | 73% / 83% | 72% / 82% |
| DRAM MB / layer | 82.1 | 88.3 |
| DRAM GB/s | 64 | 64 |
| L2 MB / layer | 299.5 | 317.6 |
| instructions M / layer | 13.7 | 14.5 |
| top stalls (warps per issue) | mio_throttle 4.9, lg_throttle 2.8, wait 2.4 | mio_throttle 4.8, lg_throttle 2.8, wait 2.5 |

*add: -1,990 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 9.0 | 2.0 |
| GPU time µs / layer | 2,394 | 404 |
| kernel (largest) | k_bin_bcast | k_bin_bcast |
| launch (largest) | 14x2304x1 x 128 thr, 168.0 waves | 11520x1x1 x 128 thr, 60.0 waves |
| SM throughput | 35% | 33% |
| memory throughput | 52% | 61% |
| L2 throughput | 52% | 61% |
| issue slots used | 35% | 33% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 16% / 29% / 12% | 15% / 27% / 11% |
| occupancy achieved / theoretical | 90% / 100% (warps-limited) | 93% / 100% (warps-limited) |
| warp latency per issue (cycles) | 31.9 | 33.7 |
| L1 / L2 hit | 57% / 49% | 33% / 33% |
| DRAM MB / layer | 169.4 | 47.2 |
| DRAM GB/s | 71 | 117 |
| L2 MB / layer | 324.5 | 70.8 |
| instructions M / layer | 62.1 | 10.0 |
| top stalls (warps per issue) | long_scoreboard 27.0, wait 2.2, selected 1.0 | long_scoreboard 28.8, wait 2.2, selected 1.0 |

*cast: -1,954 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 14.0 | 7.0 |
| GPU time µs / layer | 3,531 | 1,577 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 30780x1x1 x 256 thr, 320.6 waves | 30780x1x1 x 256 thr, 320.6 waves |
| SM throughput | 40% | 41% |
| memory throughput | 33% | 34% |
| L2 throughput | 33% | 34% |
| issue slots used | 40% | 41% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 20% / 31% / 9% | 21% / 32% / 10% |
| occupancy achieved / theoretical | 85% / 100% (warps-limited) | 84% / 100% (warps-limited) |
| warp latency per issue (cycles) | 25.1 | 23.9 |
| L1 / L2 hit | 7% / 52% | 15% / 33% |
| DRAM MB / layer | 163.4 | 102.3 |
| DRAM GB/s | 46 | 65 |
| L2 MB / layer | 336.7 | 153.6 |
| instructions M / layer | 91.1 | 41.6 |
| top stalls (warps per issue) | long_scoreboard 16.7, wait 3.3, short_scoreboard 1.5 | long_scoreboard 15.7, wait 3.4, short_scoreboard 1.5 |


### + tuned GEMM table (layer 10,845 → 10,704 µs, -1.3%)

*GEMM: +1,211 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 6.0 | 7.0 |
| GPU time µs / layer | 3,328 | 4,540 |
| kernel (largest) | cutlass::Kernel2 / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ | cutlass::Kernel2 / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ |
| launch (largest) | 144x2x1 x 256 thr, 18.0 waves | 72x2x1 x 256 thr, 9.0 waves |
| SM throughput | 53% | 49% |
| memory throughput | 72% | 70% |
| L2 throughput | 58% | 54% |
| issue slots used | 19% | 17% |
| tensor pipe | 55% | 52% |
| FMA / ALU / LSU pipe | 1% / 10% / 19% | 1% / 9% / 18% |
| occupancy achieved / theoretical | 16% / 17% (registers-limited) | 17% / 17% (registers-limited) |
| warp latency per issue (cycles) | 10.1 | 11.4 |
| L1 / L2 hit | 50% / 77% | 56% / 82% |
| DRAM MB / layer | 251.7 | 286.7 |
| DRAM GB/s | 76 | 63 |
| L2 MB / layer | 1067.3 | 1356.4 |
| instructions M / layer | 50.7 | 62.0 |
| top stalls (warps per issue) | wait 2.7, mio_throttle 2.3, math_pipe_throttle 1.3 | mio_throttle 3.2, wait 2.6, lg_throttle 1.5 |

*GEMM split-K: -1,380 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 1.0 | — |
| GPU time µs / layer | 1,380 | — |
| kernel (largest) | cutlass::Kernel2 | — |
| launch (largest) | 72x2x2 x 256 thr, 18.0 waves | — |
| SM throughput | 36% | — |
| memory throughput | 65% | — |
| L2 throughput | 50% | — |
| issue slots used | 13% | — |
| tensor pipe | 38% | — |
| FMA / ALU / LSU pipe | 1% / 7% / 15% | — |
| occupancy achieved / theoretical | 17% / 17% (registers-limited) | — |
| warp latency per issue (cycles) | 14.4 | — |
| L1 / L2 hit | 72% / 82% | — |
| DRAM MB / layer | 88.3 | — |
| DRAM GB/s | 64 | — |
| L2 MB / layer | 317.6 | — |
| instructions M / layer | 14.5 | — |
| top stalls (warps per issue) | mio_throttle 4.8, lg_throttle 2.8, wait 2.5 | — |


### + GEMM-input fusion (layer 10,704 → 9,217 µs, -13.9%)

*cast: -1,395 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 7.0 | 1.0 |
| GPU time µs / layer | 1,578 | 184 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 30780x1x1 x 256 thr, 320.6 waves | 11520x1x1 x 256 thr, 120.0 waves |
| SM throughput | 41% | 40% |
| memory throughput | 34% | 34% |
| L2 throughput | 34% | 34% |
| issue slots used | 41% | 40% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 21% / 32% / 10% | 20% / 31% / 9% |
| occupancy achieved / theoretical | 84% / 100% (warps-limited) | 84% / 100% (warps-limited) |
| warp latency per issue (cycles) | 23.9 | 24.3 |
| L1 / L2 hit | 15% / 33% | 16% / 33% |
| DRAM MB / layer | 102.3 | 11.8 |
| DRAM GB/s | 65 | 64 |
| L2 MB / layer | 153.6 | 17.7 |
| instructions M / layer | 41.6 | 4.8 |
| top stalls (warps per issue) | long_scoreboard 15.7, wait 3.4, short_scoreboard 1.5 | long_scoreboard 16.0, wait 3.4, short_scoreboard 1.5 |


### + F16 vision RoPE (layer 9,217 → 8,022 µs, -13.0%)

*copy: -442 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 1.0 |
| GPU time µs / layer | 882 | 440 |
| kernel (largest) | cpy_scalar | cpy_scalar |
| launch (largest) | 46080x1x1 x 64 thr, 180.0 waves | 46080x1x1 x 64 thr, 180.0 waves |
| SM throughput | 66% | 66% |
| memory throughput | 15% | 15% |
| L2 throughput | 15% | 15% |
| issue slots used | 66% | 66% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 32% / 43% / 3% | 32% / 43% / 3% |
| occupancy achieved / theoretical | 63% / 67% (shared memory-limited) | 63% / 67% (shared memory-limited) |
| warp latency per issue (cycles) | 11.2 | 11.2 |
| L1 / L2 hit | 15% / 33% | 15% / 33% |
| DRAM MB / layer | 23.6 | 11.8 |
| DRAM GB/s | 27 | 27 |
| L2 MB / layer | 35.4 | 17.7 |
| instructions M / layer | 47.2 | 23.6 |
| top stalls (warps per issue) | wait 3.3, long_scoreboard 2.8, not_selected 1.2 | wait 3.3, long_scoreboard 2.8, not_selected 1.3 |

*rope: -725 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 2.0 |
| GPU time µs / layer | 1,431 | 705 |
| kernel (largest) | rope_vision | rope_vision_jetson |
| launch (largest) | 36864x1x1 x 256 thr, 384.0 waves | 5760x1x1 x 256 thr, 60.0 waves |
| SM throughput | 24% | 74% |
| memory throughput | 12% | 25% |
| L2 throughput | 12% | 25% |
| issue slots used | 24% | 74% |
| tensor pipe | 1% | 1% |
| FMA / ALU / LSU pipe | 10% / 18% / 5% | 30% / 55% / 5% |
| occupancy achieved / theoretical | 28% / 100% (warps-limited) | 95% / 100% (registers-limited) |
| warp latency per issue (cycles) | 13.6 | 15.3 |
| L1 / L2 hit | 34% / 51% | 30% / 44% |
| DRAM MB / layer | 23.6 | 23.7 |
| DRAM GB/s | 17 | 34 |
| L2 MB / layer | 47.8 | 44.1 |
| instructions M / layer | 27.3 | 41.4 |
| top stalls (warps per issue) | long_scoreboard 7.0, wait 2.6, selected 1.0 | wait 3.6, not_selected 3.3, long_scoreboard 2.3 |


### + residual in GEMM (every switch) (layer 8,022 → 7,870 µs, -1.9%)

*add: -201 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 1.0 |
| GPU time µs / layer | 405 | 205 |
| kernel (largest) | k_bin_bcast | k_bin_bcast |
| launch (largest) | 11520x1x1 x 128 thr, 60.0 waves | 11520x1x1 x 128 thr, 60.0 waves |
| SM throughput | 32% | 32% |
| memory throughput | 61% | 61% |
| L2 throughput | 61% | 61% |
| issue slots used | 32% | 32% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 15% / 27% / 11% | 15% / 27% / 11% |
| occupancy achieved / theoretical | 93% / 100% (warps-limited) | 93% / 100% (warps-limited) |
| warp latency per issue (cycles) | 34.1 | 34.2 |
| L1 / L2 hit | 33% / 33% | 33% / 33% |
| DRAM MB / layer | 47.2 | 23.6 |
| DRAM GB/s | 116 | 115 |
| L2 MB / layer | 70.8 | 35.4 |
| instructions M / layer | 10.0 | 5.0 |
| top stalls (warps per issue) | long_scoreboard 29.2, wait 2.2, selected 1.0 | long_scoreboard 29.4, wait 2.2, selected 1.0 |


## LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12)

Stock layer: 19,366 µs.

### + F32 output, bias in epilogue (layer 19,366 → 18,091 µs, -6.6%)

*GEMM split-K: +424 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | 5.0 |
| GPU time µs / layer | 12,457 | 12,881 |
| kernel (largest) | ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_sta | ampere_s16816gemm_bf16_128x128_ldg8_stages_32x5_ / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ |
| launch (largest) | 28x4x9 x 128 thr, 31.5 waves | 28x4x9 x 128 thr, 31.5 waves |
| SM throughput | 50% | 48% |
| memory throughput | 57% | 62% |
| L2 throughput | 57% | 62% |
| issue slots used | 14% | 13% |
| tensor pipe | 51% | 48% |
| FMA / ALU / LSU pipe | 0% / 7% / 12% | 0% / 6% / 11% |
| occupancy achieved / theoretical | 16% / 17% (registers-limited) | 16% / 17% (registers-limited) |
| warp latency per issue (cycles) | 19.3 | 19.8 |
| L1 / L2 hit | 0% / 89% | 0% / 87% |
| DRAM MB / layer | 687.3 | 817.8 |
| DRAM GB/s | 55 | 63 |
| L2 MB / layer | 3872.4 | 4073.3 |
| instructions M / layer | 129.9 | 128.1 |
| top stalls (warps per issue) | mio_throttle 8.0, wait 3.2, barrier 2.4 | mio_throttle 8.1, wait 3.2, barrier 2.6 |

*add: -143 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | 2.0 |
| GPU time µs / layer | 394 | 251 |
| kernel (largest) | k_bin_bcast | k_bin_bcast |
| launch (largest) | 7168x1x1 x 128 thr, 37.3 waves | 7168x1x1 x 128 thr, 37.3 waves |
| SM throughput | 37% | 32% |
| memory throughput | 56% | 62% |
| L2 throughput | 56% | 62% |
| issue slots used | 37% | 32% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 18% / 32% / 13% | 15% / 28% / 11% |
| occupancy achieved / theoretical | 92% / 100% (warps-limited) | 92% / 100% (warps-limited) |
| warp latency per issue (cycles) | 29.4 | 33.4 |
| L1 / L2 hit | 45% / 40% | 33% / 33% |
| DRAM MB / layer | 38.8 | 29.4 |
| DRAM GB/s | 99 | 117 |
| L2 MB / layer | 63.3 | 44.1 |
| instructions M / layer | 10.2 | 6.2 |
| top stalls (warps per issue) | long_scoreboard 23.9, wait 2.3, selected 1.0 | long_scoreboard 28.3, wait 2.2, selected 1.0 |

*cast: -1,564 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 14.0 | 7.0 |
| GPU time µs / layer | 2,843 | 1,279 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 37888x1x1 x 256 thr, 394.7 waves | 37888x1x1 x 256 thr, 394.7 waves |
| SM throughput | 40% | 41% |
| memory throughput | 33% | 34% |
| L2 throughput | 33% | 34% |
| issue slots used | 40% | 41% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 20% / 31% / 9% | 21% / 32% / 10% |
| occupancy achieved / theoretical | 86% / 100% (warps-limited) | 84% / 100% (warps-limited) |
| warp latency per issue (cycles) | 24.9 | 24.0 |
| L1 / L2 hit | 7% / 52% | 15% / 33% |
| DRAM MB / layer | 133.7 | 82.9 |
| DRAM GB/s | 47 | 65 |
| L2 MB / layer | 277.1 | 124.4 |
| instructions M / layer | 75.0 | 33.7 |
| top stalls (warps per issue) | long_scoreboard 16.5, wait 3.3, short_scoreboard 1.5 | long_scoreboard 15.8, wait 3.4, short_scoreboard 1.5 |


### + tuned GEMM table (layer 18,091 → 12,740 µs, -29.6%)

*GEMM: +4,885 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 6.0 |
| GPU time µs / layer | 189 | 5,074 |
| kernel (largest) | ampere_s16816gemm_bf16_128x64_ldg8_relu_stages_3 | ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_ / ampere_s16816gemm_bf16_256x128_ldg8_relu_stages_ |
| launch (largest) | 4x8x1 x 128 thr, 1.0 waves | 74x4x1 x 256 thr, 18.5 waves |
| SM throughput | 50% | 82% |
| memory throughput | 78% | 67% |
| L2 throughput | 78% | 67% |
| issue slots used | 17% | 19% |
| tensor pipe | 54% | 85% |
| FMA / ALU / LSU pipe | 0% / 10% / 17% | 0% / 7% / 17% |
| occupancy achieved / theoretical | 15% / 17% (shared memory-limited) | 16% / 16% (registers-limited) |
| warp latency per issue (cycles) | 10.1 | 10.1 |
| L1 / L2 hit | 0% / 92% | 0% / 76% |
| DRAM MB / layer | 15.0 | 506.6 |
| DRAM GB/s | 80 | 100 |
| L2 MB / layer | 90.3 | 2091.3 |
| instructions M / layer | 2.5 | 75.6 |
| top stalls (warps per issue) | mio_throttle 2.7, wait 2.3, short_scoreboard 1.6 | wait 3.7, math_pipe_throttle 3.3, selected 1.0 |

*GEMM split-K: -10,218 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | 1.0 |
| GPU time µs / layer | 12,881 | 2,662 |
| kernel (largest) | ampere_s16816gemm_bf16_128x128_ldg8_stages_32x5_ / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ | ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_ |
| launch (largest) | 28x4x9 x 128 thr, 31.5 waves | 14x4x2 x 256 thr, 7.0 waves |
| SM throughput | 48% | 64% |
| memory throughput | 62% | 51% |
| L2 throughput | 62% | 51% |
| issue slots used | 13% | 14% |
| tensor pipe | 48% | 65% |
| FMA / ALU / LSU pipe | 0% / 6% / 11% | 0% / 4% / 12% |
| occupancy achieved / theoretical | 16% / 17% (registers-limited) | 17% / 17% (registers-limited) |
| warp latency per issue (cycles) | 19.8 | 14.2 |
| L1 / L2 hit | 0% / 87% | 0% / 73% |
| DRAM MB / layer | 817.8 | 225.4 |
| DRAM GB/s | 63 | 85 |
| L2 MB / layer | 4073.3 | 836.9 |
| instructions M / layer | 128.1 | 29.4 |
| top stalls (warps per issue) | mio_throttle 8.1, wait 3.2, barrier 2.6 | mio_throttle 5.3, wait 3.9, math_pipe_throttle 2.5 |


### + GEMM-input fusion (layer 12,740 → 11,547 µs, -9.4%)

*cast: -1,156 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 7.0 | 1.0 |
| GPU time µs / layer | 1,271 | 115 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 37888x1x1 x 256 thr, 394.7 waves | 7168x1x1 x 256 thr, 74.7 waves |
| SM throughput | 41% | 41% |
| memory throughput | 34% | 33% |
| L2 throughput | 34% | 33% |
| issue slots used | 41% | 41% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 21% / 32% / 10% | 21% / 32% / 10% |
| occupancy achieved / theoretical | 84% / 100% (warps-limited) | 85% / 100% (warps-limited) |
| warp latency per issue (cycles) | 23.8 | 23.7 |
| L1 / L2 hit | 15% / 33% | 15% / 33% |
| DRAM MB / layer | 82.9 | 7.3 |
| DRAM GB/s | 65 | 64 |
| L2 MB / layer | 124.4 | 11.0 |
| instructions M / layer | 33.7 | 3.0 |
| top stalls (warps per issue) | long_scoreboard 15.5, wait 3.4, short_scoreboard 1.5 | long_scoreboard 15.3, wait 3.4, short_scoreboard 1.5 |


### + residual in GEMM (every switch) (layer 11,547 → 11,344 µs, -1.8%)

*add: -251 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | — |
| GPU time µs / layer | 251 | — |
| kernel (largest) | k_bin_bcast | — |
| launch (largest) | 7168x1x1 x 128 thr, 37.3 waves | — |
| SM throughput | 33% | — |
| memory throughput | 62% | — |
| L2 throughput | 62% | — |
| issue slots used | 33% | — |
| tensor pipe | 2% | — |
| FMA / ALU / LSU pipe | 15% / 28% / 11% | — |
| occupancy achieved / theoretical | 92% / 100% (warps-limited) | — |
| warp latency per issue (cycles) | 33.1 | — |
| L1 / L2 hit | 33% / 33% | — |
| DRAM MB / layer | 29.4 | — |
| DRAM GB/s | 117 | — |
| L2 MB / layer | 44.1 | — |
| instructions M / layer | 6.2 | — |
| top stalls (warps per issue) | long_scoreboard 28.1, wait 2.2, selected 1.0 | — |

