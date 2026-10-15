6 clips: alley_1, ambush_5, bandage_1, cave_2, sleeping_1, temple_2

| configuration | mean TTFT ms | vs reference | analysis | ViT | image prefill | text prefill | frames | ViT tokens | crops / skips |
|---|---|---|---|---|---|---|---|---|---|
| uniform 8 frames, no optimizations | 14060 | +57% | 0 | 8260 | 4853 | 745 | 8.0 | 4736 | 0 / 0 |
| uniform 8 frames + window cache, window batch, GEMM switches | 7700 | -14% | 0 | 3416 | 3318 | 730 | 8.0 | 4608 | 0 / 0 |
| adaptive + live OFA flow (reference) | 8973 | +0% | 395 | 5108 | 2915 | 498 | 4.8 | 2861 | 0 / 0 |
| + window align only | 8736 | -3% | 414 | 4899 | 2857 | 493 | 4.8 | 2784 | 0 / 0 |
| + window cache (threshold 24) | 8569 | -4% | 417 | 4704 | 2872 | 496 | 4.8 | 2688 | 3 / 0 |
| + window align + ViT window batch | 6543 | -27% | 378 | 2705 | 2888 | 499 | 4.8 | 2784 | 0 / 0 |
| + GEMM F32 output | 8560 | -5% | 413 | 4831 | 2759 | 499 | 4.8 | 2861 | 0 / 0 |
| + GEMM F32 output + bias fusion | 8232 | -8% | 394 | 4543 | 2743 | 495 | 4.8 | 2861 | 0 / 0 |
| + tuned GEMM table | 8241 | -8% | 397 | 5086 | 2213 | 489 | 4.8 | 2861 | 0 / 0 |
| + all of the above | 5023 | -44% | 388 | 2081 | 1986 | 488 | 4.8 | 2688 | 3 / 0 |
| + all of the above + anchor-first | 4699 | -48% | 402 | 2138 | 1975 | 486 | 4.8 | 2752 | 1 / 0 |

TTFT per clip (ms):
| configuration | alley_1 | ambush_5 | bandage_1 | cave_2 | sleeping_1 | temple_2 |
|---|---|---|---|---|---|---|
| uniform | 14026 | 13987 | 14019 | 14067 | 14190 | 14073 |
| uniform_all | 7712 | 7682 | 7698 | 7681 | 7738 | 7686 |
| adaptive | 5856 | 14366 | 5805 | 14404 | 5890 | 7516 |
| adaptive_align | 5690 | 14002 | 5672 | 14003 | 5698 | 7350 |
| adaptive_wcache | 5712 | 13546 | 5264 | 14007 | 5297 | 7590 |
| adaptive_winbatch | 4307 | 10327 | 4249 | 10259 | 4283 | 5833 |
| adaptive_f32out | 5858 | 13609 | 5513 | 13691 | 5505 | 7182 |
| adaptive_f32out_bias | 5577 | 13135 | 5325 | 13176 | 5278 | 6898 |
| adaptive_gemmtable | 5632 | 13061 | 5352 | 13162 | 5322 | 6918 |
| adaptive_all | 3539 | 7823 | 3230 | 8004 | 3253 | 4289 |
| adaptive_all_anchor | 3053 | 7508 | 3033 | 7636 | 3027 | 3936 |
