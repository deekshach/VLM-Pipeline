
## vit  (GPU kernel ms)
| class | pm_30w_f8 |
|---|---|
| softmax | 10620.6 |
| attn matmul | 3996.7 |
| GEMM | 2144.5 |
| GEMM split-K | 2387.1 |
| cast | 2611.3 |
| add | 1443.6 |
| norm | 922.8 |
| rope | 1029.3 |
| copy | 998.0 |
| glu | 429.1 |
| other | 11.8 |
| **total** | 26594.7 |
| kernels | 10408 |

## prefill_img  (GPU kernel ms)
| class | pm_30w_f8 |
|---|---|
| attention | 1256.1 |
| GEMM | 132.0 |
| GEMM split-K | 12713.6 |
| cast | 2069.0 |
| add | 238.8 |
| norm | 263.0 |
| rope | 191.1 |
| copy | 35.9 |
| glu | 508.9 |
| **total** | 17408.5 |
| kernels | 15040 |

## prefill_txt  (GPU kernel ms)
| class | pm_30w_f8 |
|---|---|
| attention | 38.3 |
| GEMM | 1012.8 |
| GEMM split-K | 240.6 |
| cast | 11.4 |
| add | 10.5 |
| norm | 6.9 |
| rope | 4.9 |
| copy | 3.6 |
| glu | 6.2 |
| **total** | 1335.1 |
| kernels | 5604 |

## decode  (GPU kernel ms)
| class | pm_30w_f8 |
|---|---|
| attention | 31.8 |
| GEMM | 1562.6 |
| norm | 6.8 |
| rope | 3.4 |
| copy | 3.5 |
| **total** | 1608.2 |
| kernels | 3184 |

## unattributed  (GPU kernel ms)
| class | pm_30w_f8 |
|---|---|
| attention | 1.5 |
| GEMM | 146.0 |
| add | 0.8 |
| norm | 0.6 |
| rope | 0.3 |
| copy | 0.3 |
| glu | 0.3 |
| **total** | 149.8 |
| kernels | 590 |
