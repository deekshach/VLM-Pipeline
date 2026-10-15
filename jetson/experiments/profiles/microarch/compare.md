
## prefill down projection 3584x18944, 512 tokens
| config | class | kernels | time us | DRAM read MB | L2 requests MB | L2 hit | occupancy | tensor % | mem % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|
| heuristic | GEMM split-K | 1 | 6699 | 258.4 | 1149.2 | 87% | 17% | 25 | 38 | mio_throttle 52%, barrier 15% |
| heuristic | **all** | | **6699** | **258.4** | **1149.2** | | | | | |
| tuned | GEMM split-K | 1 | 2612 | 217.1 | 825.8 | 74% | 17% | 65 | 51 | mio_throttle 41%, wait 29% |
| tuned | **all** | | **2612** | **217.1** | **825.8** | | | | | |

## ViT, 1x2 windows
| config | class | kernels | time us | DRAM read MB | L2 requests MB | L2 hit | occupancy | tensor % | mem % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|
| base | GEMM | 18 | 1078 | 98.4 | 233.7 | 71% | 12% | 28 | 63 | lg_throttle 24%, wait 22% |
| base | cast | 42 | 818 | 27.4 | 56.8 | 51% | 84% | 1 | 24 | long_scoreboard 61%, wait 14% |
| base | add | 27 | 543 | 28.5 | 57.0 | 50% | 84% | 2 | 36 | long_scoreboard 75%, wait 9% |
| base | GEMM split-K | 3 | 453 | 30.9 | 101.2 | 82% | 15% | 18 | 61 | mio_throttle 41%, lg_throttle 25% |
| base | rope | 6 | 273 | 4.0 | 8.1 | 51% | 30% | 0 | 11 | long_scoreboard 53%, wait 20% |
| base | norm | 6 | 212 | 4.0 | 8.5 | 53% | 63% | 1 | 13 | long_scoreboard 31%, barrier 18% |
| base | copy | 6 | 191 | 4.0 | 19.7 | 69% | 61% | 2 | 34 | wait 30%, long_scoreboard 28% |
| base | attn matmul | 6 | 134 | 9.2 | 23.5 | 80% | 22% | 13 | 46 | long_scoreboard 17%, mio_throttle 15% |
| base | glu | 3 | 115 | 10.5 | 15.8 | 34% | 84% | 1 | 48 | long_scoreboard 66%, wait 14% |
| base | softmax | 3 | 98 | 3.4 | 8.4 | 59% | 92% | 1 | 53 | long_scoreboard 32%, short_scoreboard 28% |
| base | **all** | | **3915** | **220.1** | **532.7** | | | | | |
| base | **per layer** (3 layers) | | **1305** | **73.4** | **177.6** | | | | | |
| f32bias | GEMM | 28 | 1825 | 157.0 | 385.3 | 73% | 13% | 26 | 62 | lg_throttle 23%, wait 22% |
| f32bias | GEMM split-K | 5 | 764 | 52.5 | 173.3 | 82% | 15% | 18 | 61 | mio_throttle 40%, lg_throttle 23% |
| f32bias | cast | 33 | 624 | 27.2 | 41.3 | 34% | 84% | 1 | 23 | long_scoreboard 61%, wait 15% |
| f32bias | rope | 9 | 413 | 6.0 | 12.1 | 51% | 30% | 0 | 11 | long_scoreboard 53%, wait 20% |
| f32bias | norm | 10 | 351 | 6.7 | 14.2 | 53% | 63% | 1 | 13 | long_scoreboard 31%, barrier 18% |
| f32bias | copy | 8 | 255 | 5.3 | 26.4 | 69% | 61% | 2 | 34 | wait 30%, long_scoreboard 29% |
| f32bias | glu | 5 | 190 | 17.5 | 26.4 | 34% | 84% | 1 | 49 | long_scoreboard 66%, wait 14% |
| f32bias | attn matmul | 8 | 176 | 12.2 | 31.4 | 80% | 22% | 14 | 46 | long_scoreboard 16%, mio_throttle 14% |
| f32bias | add | 10 | 172 | 13.1 | 19.9 | 34% | 84% | 2 | 41 | long_scoreboard 74%, wait 9% |
| f32bias | softmax | 4 | 130 | 4.5 | 11.1 | 59% | 91% | 1 | 53 | long_scoreboard 31%, short_scoreboard 28% |
| f32bias | **all** | | **4901** | **302.1** | **741.3** | | | | | |
| f32bias | **per layer** (4 layers) | | **1225** | **75.5** | **185.3** | | | | | |
| winbatch | GEMM | 18 | 1081 | 98.4 | 233.7 | 71% | 12% | 28 | 63 | lg_throttle 24%, wait 22% |
| winbatch | cast | 42 | 814 | 27.4 | 56.9 | 51% | 84% | 1 | 24 | long_scoreboard 61%, wait 14% |
| winbatch | add | 25 | 505 | 26.5 | 53.6 | 51% | 84% | 2 | 37 | long_scoreboard 75%, wait 9% |
| winbatch | GEMM split-K | 3 | 451 | 30.9 | 101.0 | 82% | 15% | 19 | 61 | mio_throttle 41%, lg_throttle 25% |
| winbatch | rope | 6 | 275 | 4.0 | 8.1 | 51% | 30% | 0 | 11 | long_scoreboard 53%, wait 20% |
| winbatch | copy | 6 | 191 | 4.0 | 19.6 | 69% | 61% | 2 | 34 | wait 30%, long_scoreboard 28% |
| winbatch | norm | 5 | 176 | 3.3 | 7.1 | 53% | 62% | 2 | 13 | long_scoreboard 31%, barrier 18% |
| winbatch | glu | 3 | 115 | 10.5 | 15.8 | 34% | 84% | 1 | 48 | long_scoreboard 65%, wait 14% |
| winbatch | attn matmul | 6 | 111 | 7.6 | 12.9 | 70% | 19% | 9 | 38 | long_scoreboard 29%, barrier 15% |
| winbatch | softmax | 3 | 78 | 1.6 | 3.2 | 51% | 60% | 1 | 38 | long_scoreboard 34%, short_scoreboard 30% |
| winbatch | other | 3 | 16 | 0.0 | 0.1 | 54% | 16% | 0 | 1 | no_instruction 54%, wait 24% |
| winbatch | **all** | | **3813** | **214.2** | **512.0** | | | | | |
| winbatch | **per layer** (3 layers) | | **1271** | **71.4** | **170.7** | | | | | |

## ViT, 5x2 windows
| config | class | kernels | time us | DRAM read MB | L2 requests MB | L2 hit | occupancy | tensor % | mem % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|
| base | softmax | 3 | 5795 | 89.0 | 235.7 | 62% | 65% | 0 | 25 | short_scoreboard 21%, barrier 18% |
| base | GEMM | 21 | 4903 | 370.6 | 1365.3 | 76% | 16% | 38 | 66 | wait 21%, lg_throttle 20% |
| base | cast | 42 | 3054 | 136.2 | 281.2 | 51% | 85% | 1 | 31 | long_scoreboard 68%, wait 14% |
| base | add | 27 | 2115 | 141.3 | 272.6 | 49% | 88% | 2 | 49 | long_scoreboard 85%, wait 8% |
| base | rope | 6 | 1220 | 19.7 | 39.9 | 51% | 28% | 0 | 12 | long_scoreboard 56%, wait 20% |
| base | attn matmul | 6 | 1213 | 109.7 | 393.7 | 86% | 16% | 36 | 79 | barrier 18%, wait 17% |
| base | norm | 6 | 833 | 19.7 | 40.0 | 51% | 63% | 2 | 16 | long_scoreboard 30%, barrier 19% |
| base | copy | 6 | 774 | 19.7 | 94.5 | 69% | 62% | 2 | 41 | wait 32%, long_scoreboard 27% |
| base | glu | 3 | 515 | 52.5 | 78.9 | 33% | 84% | 1 | 53 | long_scoreboard 72%, wait 13% |
| base | **all** | | **20421** | **958.5** | **2801.8** | | | | | |
| base | **per layer** (3 layers) | | **6807** | **319.5** | **933.9** | | | | | |
| f32bias | GEMM | 33 | 8560 | 631.7 | 2255.1 | 75% | 16% | 36 | 63 | wait 22%, lg_throttle 20% |
| f32bias | softmax | 4 | 7693 | 118.7 | 314.2 | 62% | 65% | 0 | 25 | short_scoreboard 21%, barrier 18% |
| f32bias | cast | 33 | 2242 | 135.6 | 203.9 | 33% | 84% | 2 | 32 | long_scoreboard 67%, wait 15% |
| f32bias | rope | 9 | 1824 | 29.6 | 59.9 | 51% | 28% | 0 | 12 | long_scoreboard 55%, wait 21% |
| f32bias | attn matmul | 8 | 1609 | 146.3 | 524.9 | 86% | 16% | 36 | 80 | barrier 18%, wait 17% |
| f32bias | norm | 10 | 1391 | 32.9 | 66.6 | 51% | 63% | 2 | 16 | long_scoreboard 30%, barrier 19% |
| f32bias | copy | 8 | 1027 | 26.3 | 125.9 | 69% | 62% | 2 | 41 | wait 32%, long_scoreboard 28% |
| f32bias | glu | 5 | 859 | 87.6 | 131.4 | 33% | 84% | 1 | 53 | long_scoreboard 72%, wait 13% |
| f32bias | add | 10 | 602 | 65.6 | 98.5 | 33% | 89% | 2 | 58 | long_scoreboard 85%, wait 8% |
| f32bias | **all** | | **25808** | **1274.2** | **3780.4** | | | | | |
| f32bias | **per layer** (4 layers) | | **6452** | **318.6** | **945.1** | | | | | |
| winbatch | GEMM | 21 | 4900 | 369.4 | 1365.3 | 76% | 16% | 38 | 66 | wait 21%, lg_throttle 20% |
| winbatch | cast | 42 | 3060 | 136.2 | 281.2 | 51% | 85% | 1 | 31 | long_scoreboard 68%, wait 14% |
| winbatch | add | 25 | 1985 | 131.5 | 256.1 | 50% | 88% | 2 | 49 | long_scoreboard 85%, wait 8% |
| winbatch | rope | 6 | 1215 | 19.7 | 39.9 | 51% | 28% | 0 | 12 | long_scoreboard 56%, wait 20% |
| winbatch | copy | 6 | 783 | 19.7 | 97.7 | 71% | 62% | 2 | 42 | wait 31%, long_scoreboard 30% |
| winbatch | norm | 5 | 691 | 16.5 | 33.3 | 51% | 63% | 2 | 16 | long_scoreboard 30%, barrier 19% |
| winbatch | glu | 3 | 513 | 52.5 | 78.9 | 33% | 84% | 1 | 53 | long_scoreboard 72%, wait 13% |
| winbatch | attn matmul | 6 | 341 | 37.5 | 55.5 | 66% | 20% | 13 | 60 | long_scoreboard 40%, barrier 16% |
| winbatch | softmax | 3 | 318 | 7.9 | 15.8 | 50% | 62% | 1 | 45 | long_scoreboard 37%, short_scoreboard 30% |
| winbatch | other | 3 | 16 | 0.0 | 0.1 | 61% | 16% | 0 | 1 | no_instruction 41%, drain 27% |
| winbatch | **all** | | **13823** | **791.0** | **2223.7** | | | | | |
| winbatch | **per layer** (3 layers) | | **4608** | **263.7** | **741.2** | | | | | |

## ViT, 9x4 windows
| config | class | kernels | time us | DRAM read MB | L2 requests MB | L2 hit | occupancy | tensor % | mem % | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|
| base | softmax | 3 | 36315 | 2038.5 | 3057.7 | 33% | 66% | 0 | 29 | long_scoreboard 39%, barrier 18% |
| base | attn matmul | 6 | 13909 | 1212.7 | 4940.8 | 88% | 16% | 36 | 85 | barrier 21%, mio_throttle 17% |
| base | cast | 42 | 10625 | 490.1 | 1010.2 | 52% | 85% | 2 | 33 | long_scoreboard 70%, wait 14% |
| base | GEMM | 18 | 9068 | 714.4 | 3030.0 | 76% | 16% | 58 | 74 | wait 30%, mio_throttle 25% |
| base | add | 27 | 7204 | 508.1 | 973.6 | 49% | 90% | 2 | 52 | long_scoreboard 88%, wait 7% |
| base | rope | 6 | 4282 | 70.9 | 143.5 | 51% | 28% | 1 | 12 | long_scoreboard 56%, wait 21% |
| base | GEMM split-K | 3 | 3835 | 246.3 | 898.5 | 83% | 17% | 38 | 68 | mio_throttle 37%, lg_throttle 21% |
| base | norm | 6 | 2867 | 70.9 | 142.2 | 50% | 64% | 2 | 17 | long_scoreboard 31%, barrier 20% |
| base | copy | 6 | 2766 | 70.8 | 340.6 | 69% | 62% | 2 | 41 | wait 31%, long_scoreboard 29% |
| base | glu | 3 | 1912 | 189.1 | 283.7 | 33% | 86% | 1 | 52 | long_scoreboard 73%, wait 12% |
| base | **all** | | **92783** | **5611.7** | **14820.9** | | | | | |
| base | **per layer** (3 layers) | | **30928** | **1870.6** | **4940.3** | | | | | |
| f32bias | softmax | 4 | 49609 | 2717.9 | 4077.0 | 33% | 66% | 0 | 29 | long_scoreboard 40%, barrier 18% |
| f32bias | attn matmul | 8 | 18614 | 1623.0 | 6587.8 | 88% | 16% | 36 | 85 | barrier 21%, mio_throttle 17% |
| f32bias | GEMM | 28 | 16172 | 1236.0 | 5077.1 | 76% | 16% | 53 | 72 | wait 30%, mio_throttle 26% |
| f32bias | cast | 34 | 7716 | 499.8 | 750.2 | 33% | 84% | 2 | 34 | long_scoreboard 69%, wait 15% |
| f32bias | rope | 10 | 7164 | 118.2 | 239.1 | 51% | 28% | 1 | 12 | long_scoreboard 56%, wait 21% |
| f32bias | GEMM split-K | 5 | 6846 | 441.1 | 1588.2 | 82% | 17% | 36 | 65 | mio_throttle 36%, lg_throttle 21% |
| f32bias | norm | 9 | 4329 | 106.3 | 213.3 | 50% | 64% | 2 | 17 | long_scoreboard 31%, barrier 19% |
| f32bias | copy | 8 | 3731 | 94.4 | 453.8 | 69% | 62% | 2 | 41 | wait 30%, long_scoreboard 30% |
| f32bias | glu | 5 | 3220 | 315.2 | 472.9 | 33% | 85% | 1 | 52 | long_scoreboard 73%, wait 12% |
| f32bias | add | 9 | 1824 | 212.4 | 318.7 | 33% | 93% | 2 | 61 | long_scoreboard 89%, wait 7% |
| f32bias | **all** | | **119225** | **7364.4** | **19778.0** | | | | | |
| f32bias | **per layer** (4 layers) | | **29806** | **1841.1** | **4944.5** | | | | | |
| winbatch | cast | 41 | 10482 | 478.3 | 992.5 | 52% | 85% | 2 | 32 | long_scoreboard 70%, wait 14% |
| winbatch | GEMM | 17 | 8099 | 640.3 | 2773.1 | 77% | 16% | 59 | 74 | wait 30%, mio_throttle 24% |
| winbatch | add | 27 | 7340 | 508.1 | 973.6 | 49% | 90% | 2 | 51 | long_scoreboard 88%, wait 7% |
| winbatch | rope | 6 | 4292 | 70.9 | 143.5 | 51% | 28% | 1 | 12 | long_scoreboard 56%, wait 21% |
| winbatch | GEMM split-K | 3 | 3848 | 245.4 | 898.6 | 83% | 17% | 38 | 68 | mio_throttle 37%, lg_throttle 21% |
| winbatch | copy | 6 | 2729 | 70.8 | 350.4 | 71% | 62% | 2 | 43 | long_scoreboard 31%, wait 31% |
| winbatch | norm | 5 | 2401 | 59.1 | 118.5 | 50% | 64% | 2 | 17 | long_scoreboard 31%, barrier 19% |
| winbatch | glu | 3 | 1934 | 189.1 | 283.7 | 33% | 85% | 1 | 51 | long_scoreboard 73%, wait 12% |
| winbatch | attn matmul | 6 | 1129 | 134.7 | 198.7 | 66% | 20% | 13 | 66 | long_scoreboard 52%, barrier 14% |
| winbatch | softmax | 3 | 1087 | 28.3 | 56.7 | 50% | 63% | 1 | 47 | long_scoreboard 38%, short_scoreboard 30% |
| winbatch | other | 3 | 18 | 0.0 | 0.1 | 87% | 16% | 0 | 2 | no_instruction 38%, drain 31% |
| winbatch | **all** | | **43359** | **2425.0** | **6789.4** | | | | | |
| winbatch | **per layer** (3 layers) | | **14453** | **808.3** | **2263.1** | | | | | |
