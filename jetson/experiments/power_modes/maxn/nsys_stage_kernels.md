
## vit  (GPU kernel ms)
| class | pm_maxn_f8 |
|---|---|
| softmax | 3190.5 |
| attn matmul | 1267.6 |
| GEMM | 812.9 |
| GEMM split-K | 343.1 |
| cast | 828.5 |
| add | 613.1 |
| norm | 260.5 |
| rope | 386.4 |
| copy | 256.6 |
| glu | 154.3 |
| other | 3.0 |
| **total** | 8116.5 |
| kernels | 10408 |

## prefill_img  (GPU kernel ms)
| class | pm_maxn_f8 |
|---|---|
| attention | 331.8 |
| GEMM | 498.8 |
| GEMM split-K | 2867.7 |
| cast | 668.9 |
| add | 91.9 |
| norm | 84.0 |
| rope | 55.4 |
| copy | 12.2 |
| glu | 177.9 |
| **total** | 4788.7 |
| kernels | 15264 |

## prefill_txt  (GPU kernel ms)
| class | pm_maxn_f8 |
|---|---|
| attention | 20.1 |
| GEMM | 642.1 |
| GEMM split-K | 32.5 |
| cast | 3.9 |
| add | 6.4 |
| norm | 3.8 |
| rope | 2.1 |
| copy | 2.2 |
| glu | 2.6 |
| **total** | 715.6 |
| kernels | 5688 |

## decode  (GPU kernel ms)
| class | pm_maxn_f8 |
|---|---|
| attention | 17.5 |
| GEMM | 700.3 |
| norm | 4.3 |
| rope | 2.0 |
| copy | 2.1 |
| **total** | 726.3 |
| kernels | 3184 |

## unattributed  (GPU kernel ms)
| class | pm_maxn_f8 |
|---|---|
| attention | 0.8 |
| GEMM | 82.0 |
| add | 0.5 |
| norm | 0.3 |
| rope | 0.1 |
| copy | 0.2 |
| glu | 0.1 |
| **total** | 84.1 |
| kernels | 590 |
