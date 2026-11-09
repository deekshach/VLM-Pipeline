| GEMM | shape (m n k f32 bias) | algorithm | kernel | grid | registers | shared B / block | local load B | local store B |
|---|---|---|---|---|---|---|---|---|
| Q (LLM) | 3584 512 3584 1 1 | heuristic | ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_32x5_tn | (28, 4, 2) | 226 | 82,944 | 0 | 0 |
| Q (LLM) | 3584 512 3584 1 1 | tuned | ampere_s16816gemm_bf16_256x128_ldg8_relu_stages_64x3_tn | (14, 4, 1) | 238 | 148,480 | 0 | 0 |
| K, V (LLM) | 512 512 3584 1 1 | heuristic | ampere_s16816gemm_bf16_128x64_ldg8_relu_stages_32x6_tn | (4, 8, 1) | 142 | 74,752 | 0 | 0 |
| K, V (LLM) | 512 512 3584 1 1 | tuned | Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x128_64x5_tn_align8> | (16, 1, 1) | 242 | 164,864 | 0 | 0 |
| O (LLM) | 3584 512 3584 1 0 | heuristic | ampere_s16816gemm_bf16_128x128_ldg8_stages_32x5_tn | (28, 4, 2) | 226 | 82,944 | 0 | 0 |
| O (LLM) | 3584 512 3584 1 0 | tuned | ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_tn | (14, 4, 1) | 238 | 148,480 | 0 | 0 |
| gate, up (LLM) | 18944 512 3584 1 0 | heuristic | ampere_s16816gemm_bf16_128x128_ldg8_stages_32x5_tn | (148, 4, 2) | 226 | 82,944 | 0 | 0 |
| gate, up (LLM) | 18944 512 3584 1 0 | tuned | ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_tn | (74, 4, 1) | 238 | 148,480 | 0 | 0 |
| down (LLM) | 3584 512 18944 1 0 | heuristic | ampere_s16816gemm_bf16_128x128_ldg8_stages_32x5_tn | (28, 4, 9) | 226 | 82,944 | 0 | 0 |
| down (LLM) | 3584 512 18944 1 0 | tuned | ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_tn | (14, 4, 2) | 238 | 148,480 | 0 | 0 |
| Q, K, V, O (ViT) | 1280 2304 1280 1 1 | heuristic | ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_32x5_tn | (10, 18, 1) | 226 | 82,944 | 0 | 0 |
| Q, K, V, O (ViT) | 1280 2304 1280 1 1 | tuned | ampere_s16816gemm_bf16_128x128_ldg8_relu_stages_32x5_tn | (10, 18, 1) | 226 | 82,944 | 0 | 0 |
| gate, up (ViT) | 3420 2304 1280 1 1 | heuristic | Kernel2<cutlass_80_tensorop_s16816gemm_bf16_256x128_64x3_tn_align2> | (144, 2, 1) | 232 | 148,480 | 0 | 0 |
| gate, up (ViT) | 3420 2304 1280 1 1 | tuned | Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x256_64x3_tn_align2> | (72, 4, 1) | 232 | 148,480 | 0 | 0 |
| down (ViT) | 1280 2304 3420 1 1 | heuristic | Kernel2<cutlass_80_tensorop_s16816gemm_bf16_256x128_64x3_tn_align2> | (72, 2, 2) | 232 | 148,480 | 0 | 0 |
| down (ViT) | 1280 2304 3420 1 1 | tuned | Kernel2<cutlass_80_tensorop_s16816gemm_bf16_128x256_64x3_tn_align2> | (72, 2, 1) | 232 | 148,480 | 0 | 0 |
