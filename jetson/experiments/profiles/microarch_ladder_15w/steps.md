
## ViT window-attention layer (mean of layers 1-6)

Stock layer: 95,122 µs.

### + batched window attention (layer 95,122 → 83,763 µs, -11.9%)

*attention: -11,340 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 1.0 |
| GPU time µs / layer | 12,477 | 1,137 |
| kernel (largest) | flash_attn_ext_f16 / flash_attn_mask_to_KV_max | flash_attn_ext_f16 |
| launch (largest) | 576x1x1 x 128 thr, 32.0 waves | 576x1x1 x 128 thr, 32.0 waves |
| SM throughput | 57% | 50% |
| memory throughput | 55% | 47% |
| L2 throughput | 21% | 39% |
| issue slots used | 57% | 50% |
| tensor pipe | 24% | 14% |
| FMA / ALU / LSU pipe | 23% / 47% / 33% | 20% / 41% / 25% |
| occupancy achieved / theoretical | 25% / 25% (registers-limited) | 24% / 25% (registers-limited) |
| warp latency per issue (cycles) | 5.2 | 5.8 |
| L1 / L2 hit | 3% / 66% | 9% / 38% |
| DRAM MB / layer | 124.0 | 23.7 |
| DRAM GB/s | 10 | 21 |
| L2 MB / layer | 357.8 | 38.4 |
| instructions M / layer | 69.6 | 5.4 |
| top stalls (warps per issue) | wait 1.8, selected 1.0, long_scoreboard 0.6 | wait 1.7, long_scoreboard 1.1, selected 1.0 |


### + F32 output, bias in epilogue (layer 83,763 → 70,597 µs, -15.7%)

*GEMM: +2,573 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 2.0 |
| GPU time µs / layer | 15,465 | 18,037 |
| kernel (largest) | cutlass::Kernel2 | cutlass::Kernel2 |
| launch (largest) | 144x2x1 x 256 thr, 48.0 waves | 144x2x1 x 256 thr, 48.0 waves |
| SM throughput | 55% | 51% |
| memory throughput | 78% | 75% |
| L2 throughput | 26% | 28% |
| issue slots used | 19% | 20% |
| tensor pipe | 55% | 52% |
| FMA / ALU / LSU pipe | 1% / 9% / 21% | 2% / 11% / 22% |
| occupancy achieved / theoretical | 17% / 17% (registers-limited) | 17% / 17% (registers-limited) |
| warp latency per issue (cycles) | 10.3 | 9.9 |
| L1 / L2 hit | 74% / 49% | 74% / 46% |
| DRAM MB / layer | 263.1 | 294.9 |
| DRAM GB/s | 17 | 16 |
| L2 MB / layer | 513.9 | 551.1 |
| instructions M / layer | 29.2 | 32.3 |
| top stalls (warps per issue) | mio_throttle 2.6, wait 2.3, math_pipe_throttle 1.6 | wait 2.4, mio_throttle 2.4, lg_throttle 1.4 |

*GEMM split-K: +935 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | 5.0 |
| GPU time µs / layer | 16,826 | 17,761 |
| kernel (largest) | cutlass::Kernel2 / ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_sta | cutlass::Kernel2 / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ |
| launch (largest) | 72x2x2 x 256 thr, 48.0 waves | 72x2x2 x 256 thr, 48.0 waves |
| SM throughput | 60% | 57% |
| memory throughput | 70% | 70% |
| L2 throughput | 33% | 37% |
| issue slots used | 22% | 22% |
| tensor pipe | 61% | 58% |
| FMA / ALU / LSU pipe | 1% / 12% / 20% | 2% / 13% / 20% |
| occupancy achieved / theoretical | 16% / 17% (registers-limited) | 16% / 17% (registers-limited) |
| warp latency per issue (cycles) | 10.1 | 9.9 |
| L1 / L2 hit | 40% / 83% | 39% / 81% |
| DRAM MB / layer | 185.5 | 230.4 |
| DRAM GB/s | 11 | 13 |
| L2 MB / layer | 842.3 | 931.5 |
| instructions M / layer | 35.5 | 38.0 |
| top stalls (warps per issue) | wait 2.7, mio_throttle 2.5, lg_throttle 1.3 | wait 2.7, mio_throttle 2.6, lg_throttle 1.3 |

*add: -8,236 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 9.0 | 2.0 |
| GPU time µs / layer | 9,966 | 1,730 |
| kernel (largest) | k_bin_bcast | k_bin_bcast |
| launch (largest) | 14x2304x1 x 128 thr, 448.0 waves | 11520x1x1 x 128 thr, 160.0 waves |
| SM throughput | 67% | 65% |
| memory throughput | 36% | 45% |
| L2 throughput | 36% | 45% |
| issue slots used | 67% | 65% |
| tensor pipe | 4% | 4% |
| FMA / ALU / LSU pipe | 31% / 56% / 22% | 30% / 55% / 22% |
| occupancy achieved / theoretical | 89% / 100% (warps-limited) | 90% / 100% (warps-limited) |
| warp latency per issue (cycles) | 15.7 | 16.3 |
| L1 / L2 hit | 59% / 48% | 33% / 33% |
| DRAM MB / layer | 169.3 | 47.2 |
| DRAM GB/s | 17 | 27 |
| L2 MB / layer | 323.1 | 70.8 |
| instructions M / layer | 62.1 | 10.0 |
| top stalls (warps per issue) | long_scoreboard 9.7, wait 2.4, not_selected 1.1 | long_scoreboard 10.4, wait 2.4, selected 1.0 |

*cast: -9,759 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 14.0 | 7.0 |
| GPU time µs / layer | 17,945 | 8,186 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 30780x1x1 x 256 thr, 855.0 waves | 30780x1x1 x 256 thr, 855.0 waves |
| SM throughput | 56% | 56% |
| memory throughput | 21% | 21% |
| L2 throughput | 21% | 21% |
| issue slots used | 56% | 56% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 28% / 43% / 13% | 28% / 43% / 13% |
| occupancy achieved / theoretical | 84% / 100% (warps-limited) | 83% / 100% (warps-limited) |
| warp latency per issue (cycles) | 17.5 | 17.5 |
| L1 / L2 hit | 5% / 52% | 12% / 33% |
| DRAM MB / layer | 163.4 | 102.3 |
| DRAM GB/s | 9 | 13 |
| L2 MB / layer | 336.6 | 153.5 |
| instructions M / layer | 91.1 | 41.6 |
| top stalls (warps per issue) | long_scoreboard 8.9, wait 3.5, short_scoreboard 1.5 | long_scoreboard 8.9, wait 3.5, short_scoreboard 1.5 |


### + tuned GEMM table (layer 70,597 → 65,620 µs, -7.0%)

*GEMM: +14,134 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 7.0 |
| GPU time µs / layer | 18,037 | 32,171 |
| kernel (largest) | cutlass::Kernel2 | cutlass::Kernel2 / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ |
| launch (largest) | 144x2x1 x 256 thr, 48.0 waves | 72x2x1 x 256 thr, 24.0 waves |
| SM throughput | 51% | 57% |
| memory throughput | 75% | 72% |
| L2 throughput | 28% | 28% |
| issue slots used | 20% | 20% |
| tensor pipe | 52% | 58% |
| FMA / ALU / LSU pipe | 2% / 11% / 22% | 1% / 10% / 20% |
| occupancy achieved / theoretical | 17% / 17% (registers-limited) | 17% / 17% (registers-limited) |
| warp latency per issue (cycles) | 9.9 | 10.3 |
| L1 / L2 hit | 74% / 46% | 57% / 78% |
| DRAM MB / layer | 294.9 | 366.8 |
| DRAM GB/s | 16 | 11 |
| L2 MB / layer | 551.1 | 1355.4 |
| instructions M / layer | 32.3 | 62.0 |
| top stalls (warps per issue) | wait 2.4, mio_throttle 2.4, lg_throttle 1.4 | mio_throttle 2.6, wait 2.6, math_pipe_throttle 1.6 |

*GEMM split-K: -17,761 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | — |
| GPU time µs / layer | 17,761 | — |
| kernel (largest) | cutlass::Kernel2 / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ | — |
| launch (largest) | 72x2x2 x 256 thr, 48.0 waves | — |
| SM throughput | 57% | — |
| memory throughput | 70% | — |
| L2 throughput | 37% | — |
| issue slots used | 22% | — |
| tensor pipe | 58% | — |
| FMA / ALU / LSU pipe | 2% / 13% / 20% | — |
| occupancy achieved / theoretical | 16% / 17% (registers-limited) | — |
| warp latency per issue (cycles) | 9.9 | — |
| L1 / L2 hit | 39% / 81% | — |
| DRAM MB / layer | 230.4 | — |
| DRAM GB/s | 13 | — |
| L2 MB / layer | 931.5 | — |
| instructions M / layer | 38.0 | — |
| top stalls (warps per issue) | wait 2.7, mio_throttle 2.6, lg_throttle 1.3 | — |


### + GEMM-input fusion (layer 65,620 → 58,188 µs, -11.3%)

*cast: -7,227 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 7.0 | 1.0 |
| GPU time µs / layer | 8,172 | 945 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 30780x1x1 x 256 thr, 855.0 waves | 11520x1x1 x 256 thr, 320.0 waves |
| SM throughput | 56% | 56% |
| memory throughput | 21% | 21% |
| L2 throughput | 21% | 21% |
| issue slots used | 56% | 56% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 28% / 43% / 13% | 28% / 43% / 13% |
| occupancy achieved / theoretical | 84% / 100% (warps-limited) | 84% / 100% (warps-limited) |
| warp latency per issue (cycles) | 17.4 | 17.5 |
| L1 / L2 hit | 12% / 33% | 12% / 33% |
| DRAM MB / layer | 102.3 | 11.8 |
| DRAM GB/s | 13 | 12 |
| L2 MB / layer | 153.5 | 17.7 |
| instructions M / layer | 41.6 | 4.8 |
| top stalls (warps per issue) | long_scoreboard 8.9, wait 3.5, short_scoreboard 1.5 | long_scoreboard 8.9, wait 3.5, short_scoreboard 1.5 |


### + F16 vision RoPE (layer 58,188 → 53,999 µs, -7.2%)

*copy: -3,475 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 1.0 |
| GPU time µs / layer | 6,949 | 3,474 |
| kernel (largest) | cpy_scalar | cpy_scalar |
| launch (largest) | 46080x1x1 x 64 thr, 480.0 waves | 46080x1x1 x 64 thr, 480.0 waves |
| SM throughput | 70% | 70% |
| memory throughput | 6% | 6% |
| L2 throughput | 6% | 6% |
| issue slots used | 70% | 70% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 33% / 45% / 3% | 33% / 45% / 3% |
| occupancy achieved / theoretical | 63% / 67% (shared memory-limited) | 63% / 67% (shared memory-limited) |
| warp latency per issue (cycles) | 10.7 | 10.7 |
| L1 / L2 hit | 15% / 33% | 15% / 33% |
| DRAM MB / layer | 23.6 | 11.8 |
| DRAM GB/s | 3 | 3 |
| L2 MB / layer | 35.4 | 17.7 |
| instructions M / layer | 47.2 | 23.6 |
| top stalls (warps per issue) | wait 3.3, long_scoreboard 1.7, not_selected 1.5 | wait 3.3, long_scoreboard 1.7, not_selected 1.5 |

*rope: -714 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 2.0 |
| GPU time µs / layer | 6,109 | 5,395 |
| kernel (largest) | rope_vision | rope_vision_jetson |
| launch (largest) | 36864x1x1 x 256 thr, 1024.0 waves | 5760x1x1 x 256 thr, 160.0 waves |
| SM throughput | 46% | 79% |
| memory throughput | 11% | 10% |
| L2 throughput | 9% | 10% |
| issue slots used | 46% | 79% |
| tensor pipe | 1% | 1% |
| FMA / ALU / LSU pipe | 19% / 34% / 10% | 32% / 58% / 5% |
| occupancy achieved / theoretical | 37% / 100% (warps-limited) | 95% / 100% (registers-limited) |
| warp latency per issue (cycles) | 9.2 | 14.4 |
| L1 / L2 hit | 34% / 50% | 31% / 43% |
| DRAM MB / layer | 23.6 | 23.7 |
| DRAM GB/s | 4 | 4 |
| L2 MB / layer | 47.4 | 43.8 |
| instructions M / layer | 27.3 | 41.4 |
| top stalls (warps per issue) | wait 2.6, long_scoreboard 2.5, selected 1.0 | not_selected 3.7, wait 3.6, math_pipe_throttle 1.8 |


### + residual in GEMM (every switch) (layer 53,999 → 53,172 µs, -1.5%)

*add: -865 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 1.0 |
| GPU time µs / layer | 1,730 | 865 |
| kernel (largest) | k_bin_bcast | k_bin_bcast |
| launch (largest) | 11520x1x1 x 128 thr, 160.0 waves | 11520x1x1 x 128 thr, 160.0 waves |
| SM throughput | 65% | 65% |
| memory throughput | 45% | 45% |
| L2 throughput | 45% | 45% |
| issue slots used | 65% | 65% |
| tensor pipe | 4% | 4% |
| FMA / ALU / LSU pipe | 30% / 55% / 22% | 30% / 55% / 22% |
| occupancy achieved / theoretical | 90% / 100% (warps-limited) | 90% / 100% (warps-limited) |
| warp latency per issue (cycles) | 16.3 | 16.4 |
| L1 / L2 hit | 33% / 33% | 33% / 33% |
| DRAM MB / layer | 47.2 | 23.6 |
| DRAM GB/s | 27 | 27 |
| L2 MB / layer | 70.8 | 35.4 |
| instructions M / layer | 10.0 | 5.0 |
| top stalls (warps per issue) | long_scoreboard 10.4, wait 2.4, selected 1.0 | long_scoreboard 10.5, wait 2.4, selected 1.0 |


## LLM prefill layer, last 512-token ubatch of 8 frames (mean of layers 1-12)

Stock layer: 120,248 µs.

### + F32 output, bias in epilogue (layer 120,248 → 112,560 µs, -6.4%)

*GEMM split-K: +1,139 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | 5.0 |
| GPU time µs / layer | 80,128 | 81,267 |
| kernel (largest) | ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_sta | ampere_s16816gemm_bf16_128x128_ldg8_stages_32x5_ / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ |
| launch (largest) | 28x4x3 x 128 thr, 28.0 waves | 28x4x3 x 128 thr, 28.0 waves |
| SM throughput | 59% | 58% |
| memory throughput | 39% | 40% |
| L2 throughput | 38% | 40% |
| issue slots used | 17% | 17% |
| tensor pipe | 59% | 58% |
| FMA / ALU / LSU pipe | 1% / 8% / 14% | 1% / 8% / 14% |
| occupancy achieved / theoretical | 17% / 17% (registers-limited) | 17% / 17% (registers-limited) |
| warp latency per issue (cycles) | 15.8 | 16.0 |
| L1 / L2 hit | 0% / 82% | 0% / 81% |
| DRAM MB / layer | 1205.7 | 1310.4 |
| DRAM GB/s | 15 | 16 |
| L2 MB / layer | 3920.9 | 4170.0 |
| instructions M / layer | 134.5 | 132.3 |
| top stalls (warps per issue) | mio_throttle 6.3, wait 3.2, barrier 1.9 | mio_throttle 6.3, wait 3.3, barrier 1.9 |

*add: -667 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | 2.0 |
| GPU time µs / layer | 1,786 | 1,120 |
| kernel (largest) | k_bin_bcast | k_bin_bcast |
| launch (largest) | 7168x1x1 x 128 thr, 99.6 waves | 7168x1x1 x 128 thr, 99.6 waves |
| SM throughput | 66% | 66% |
| memory throughput | 39% | 44% |
| L2 throughput | 39% | 44% |
| issue slots used | 66% | 66% |
| tensor pipe | 4% | 4% |
| FMA / ALU / LSU pipe | 31% / 55% / 22% | 31% / 55% / 22% |
| occupancy achieved / theoretical | 90% / 100% (warps-limited) | 90% / 100% (warps-limited) |
| warp latency per issue (cycles) | 16.0 | 16.1 |
| L1 / L2 hit | 45% / 40% | 33% / 33% |
| DRAM MB / layer | 38.8 | 29.4 |
| DRAM GB/s | 22 | 26 |
| L2 MB / layer | 63.1 | 44.1 |
| instructions M / layer | 10.2 | 6.2 |
| top stalls (warps per issue) | long_scoreboard 10.0, wait 2.4, not_selected 1.0 | long_scoreboard 10.2, wait 2.4, selected 1.0 |

*cast: -8,189 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 14.0 | 7.0 |
| GPU time µs / layer | 14,810 | 6,621 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 37888x1x1 x 256 thr, 1052.4 waves | 37888x1x1 x 256 thr, 1052.4 waves |
| SM throughput | 56% | 56% |
| memory throughput | 20% | 21% |
| L2 throughput | 20% | 21% |
| issue slots used | 56% | 56% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 28% / 43% / 13% | 28% / 43% / 13% |
| occupancy achieved / theoretical | 84% / 100% (warps-limited) | 84% / 100% (warps-limited) |
| warp latency per issue (cycles) | 17.5 | 17.5 |
| L1 / L2 hit | 5% / 52% | 12% / 33% |
| DRAM MB / layer | 133.7 | 82.9 |
| DRAM GB/s | 9 | 13 |
| L2 MB / layer | 277.0 | 124.3 |
| instructions M / layer | 75.0 | 33.7 |
| top stalls (warps per issue) | long_scoreboard 8.9, wait 3.5, short_scoreboard 1.5 | long_scoreboard 8.9, wait 3.5, short_scoreboard 1.5 |


### + tuned GEMM table (layer 112,560 → 85,386 µs, -24.1%)

*GEMM: +37,607 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | 6.0 |
| GPU time µs / layer | 990 | 38,598 |
| kernel (largest) | ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ | ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_ / ampere_s16816gemm_bf16_256x128_ldg8_relu_stages_ |
| launch (largest) | 4x4x1 x 128 thr, 1.3 waves | 74x4x1 x 256 thr, 49.3 waves |
| SM throughput | 78% | 88% |
| memory throughput | 54% | 48% |
| L2 throughput | 54% | 37% |
| issue slots used | 20% | 20% |
| tensor pipe | 88% | 91% |
| FMA / ALU / LSU pipe | 0% / 10% / 19% | 0% / 7% / 18% |
| occupancy achieved / theoretical | 14% / 17% (registers-limited) | 16% / 16% (registers-limited) |
| warp latency per issue (cycles) | 7.2 | 9.5 |
| L1 / L2 hit | 0% / 78% | 0% / 57% |
| DRAM MB / layer | 26.2 | 895.6 |
| DRAM GB/s | 26 | 23 |
| L2 MB / layer | 60.9 | 2091.2 |
| instructions M / layer | 1.9 | 75.6 |
| top stalls (warps per issue) | wait 3.2, math_pipe_throttle 2.0, selected 1.0 | wait 3.7, math_pipe_throttle 3.3, selected 1.0 |

*GEMM split-K: -64,780 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 5.0 | 1.0 |
| GPU time µs / layer | 81,267 | 16,487 |
| kernel (largest) | ampere_s16816gemm_bf16_128x128_ldg8_stages_32x5_ / ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_ | ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_ |
| launch (largest) | 28x4x3 x 128 thr, 28.0 waves | 14x4x2 x 256 thr, 18.7 waves |
| SM throughput | 58% | 85% |
| memory throughput | 40% | 44% |
| L2 throughput | 40% | 35% |
| issue slots used | 17% | 18% |
| tensor pipe | 58% | 86% |
| FMA / ALU / LSU pipe | 1% / 8% / 14% | 0% / 6% / 16% |
| occupancy achieved / theoretical | 17% / 17% (registers-limited) | 17% / 17% (registers-limited) |
| warp latency per issue (cycles) | 16.0 | 10.7 |
| L1 / L2 hit | 0% / 81% | 0% / 55% |
| DRAM MB / layer | 1310.4 | 375.9 |
| DRAM GB/s | 16 | 23 |
| L2 MB / layer | 4170.0 | 836.8 |
| instructions M / layer | 132.3 | 29.4 |
| top stalls (warps per issue) | mio_throttle 6.3, wait 3.3, barrier 1.9 | wait 3.9, math_pipe_throttle 3.5, mio_throttle 1.3 |


### + GEMM-input fusion (layer 85,386 → 79,202 µs, -7.2%)

*cast: -6,027 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 7.0 | 1.0 |
| GPU time µs / layer | 6,618 | 590 |
| kernel (largest) | convert_unary | convert_unary |
| launch (largest) | 37888x1x1 x 256 thr, 1052.4 waves | 7168x1x1 x 256 thr, 199.1 waves |
| SM throughput | 56% | 56% |
| memory throughput | 21% | 20% |
| L2 throughput | 21% | 20% |
| issue slots used | 56% | 56% |
| tensor pipe | 2% | 2% |
| FMA / ALU / LSU pipe | 28% / 43% / 13% | 28% / 43% / 13% |
| occupancy achieved / theoretical | 84% / 100% (warps-limited) | 84% / 100% (warps-limited) |
| warp latency per issue (cycles) | 17.5 | 17.5 |
| L1 / L2 hit | 12% / 33% | 12% / 33% |
| DRAM MB / layer | 82.9 | 7.3 |
| DRAM GB/s | 13 | 12 |
| L2 MB / layer | 124.3 | 11.0 |
| instructions M / layer | 33.7 | 3.0 |
| top stalls (warps per issue) | long_scoreboard 8.9, wait 3.5, short_scoreboard 1.5 | long_scoreboard 8.8, wait 3.5, short_scoreboard 1.5 |


### + residual in GEMM (every switch) (layer 79,202 → 78,266 µs, -1.2%)

*add: -1,124 µs per layer*

| metric | before | after |
|---|---|---|
| kernels / layer | 2.0 | — |
| GPU time µs / layer | 1,124 | — |
| kernel (largest) | k_bin_bcast | — |
| launch (largest) | 7168x1x1 x 128 thr, 99.6 waves | — |
| SM throughput | 65% | — |
| memory throughput | 43% | — |
| L2 throughput | 43% | — |
| issue slots used | 65% | — |
| tensor pipe | 4% | — |
| FMA / ALU / LSU pipe | 30% / 54% / 22% | — |
| occupancy achieved / theoretical | 90% / 100% (warps-limited) | — |
| warp latency per issue (cycles) | 16.4 | — |
| L1 / L2 hit | 33% / 33% | — |
| DRAM MB / layer | 29.4 | — |
| DRAM GB/s | 26 | — |
| L2 MB / layer | 44.1 | — |
| instructions M / layer | 6.2 | — |
| top stalls (warps per issue) | long_scoreboard 10.5, wait 2.4, selected 1.0 | — |

