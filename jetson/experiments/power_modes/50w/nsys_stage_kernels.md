
## vit  (GPU kernel ms)
| class | pm_50w_f8 |
|---|---|
| softmax | 4382.9 |
| attn matmul | 1811.2 |
| GEMM | 1216.6 |
| GEMM split-K | 507.0 |
| cast | 1091.9 |
| add | 719.9 |
| norm | 367.9 |
| rope | 509.1 |
| copy | 386.4 |
| glu | 190.0 |
| other | 4.6 |
| **total** | 11187.5 |
| kernels | 10408 |

## prefill_img  (GPU kernel ms)
| class | pm_50w_f8 |
|---|---|
| attention | 496.4 |
| GEMM | 689.9 |
| GEMM split-K | 4529.5 |
| cast | 878.2 |
| add | 113.6 |
| norm | 112.3 |
| rope | 79.2 |
| copy | 16.4 |
| glu | 224.8 |
| **total** | 7140.4 |
| kernels | 15264 |

## prefill_txt  (GPU kernel ms)
| class | pm_50w_f8 |
|---|---|
| attention | 27.8 |
| GEMM | 712.3 |
| GEMM split-K | 50.3 |
| cast | 5.4 |
| add | 8.3 |
| norm | 5.2 |
| rope | 3.0 |
| copy | 3.0 |
| glu | 3.3 |
| **total** | 818.8 |
| kernels | 5688 |

## decode  (GPU kernel ms)
| class | pm_50w_f8 |
|---|---|
| attention | 23.2 |
| GEMM | 795.2 |
| norm | 5.6 |
| rope | 2.8 |
| copy | 2.8 |
| **total** | 829.7 |
| kernels | 3184 |

## unattributed  (GPU kernel ms)
| class | pm_50w_f8 |
|---|---|
| attention | 1.1 |
| GEMM | 92.0 |
| add | 0.6 |
| norm | 0.5 |
| rope | 0.2 |
| copy | 0.2 |
| glu | 0.2 |
| **total** | 94.8 |
| kernels | 590 |
