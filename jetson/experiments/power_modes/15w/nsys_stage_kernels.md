
## vit  (GPU kernel ms)
| class | pm_15w_f8 |
|---|---|
| softmax | 21197.8 |
| attn matmul | 7227.0 |
| GEMM | 6633.3 |
| GEMM split-K | 2181.8 |
| cast | 5177.2 |
| add | 2746.7 |
| norm | 1839.7 |
| rope | 1794.1 |
| copy | 1983.4 |
| glu | 823.4 |
| other | 23.4 |
| **total** | 51627.9 |
| kernels | 10408 |

## prefill_img  (GPU kernel ms)
| class | pm_15w_f8 |
|---|---|
| attention | 2522.8 |
| GEMM | 637.3 |
| GEMM split-K | 21150.8 |
| cast | 4099.6 |
| add | 453.4 |
| norm | 517.0 |
| rope | 375.6 |
| copy | 71.5 |
| glu | 976.8 |
| **total** | 30804.8 |
| kernels | 15040 |

## prefill_txt  (GPU kernel ms)
| class | pm_15w_f8 |
|---|---|
| attention | 60.4 |
| GEMM | 1784.2 |
| GEMM split-K | 395.7 |
| cast | 21.4 |
| add | 16.7 |
| norm | 10.7 |
| rope | 7.6 |
| copy | 5.5 |
| glu | 11.1 |
| **total** | 2313.4 |
| kernels | 5604 |

## decode  (GPU kernel ms)
| class | pm_15w_f8 |
|---|---|
| attention | 47.9 |
| GEMM | 3084.8 |
| add | 0.1 |
| norm | 10.3 |
| rope | 5.2 |
| copy | 5.0 |
| **total** | 3153.3 |
| kernels | 3184 |

## unattributed  (GPU kernel ms)
| class | pm_15w_f8 |
|---|---|
| attention | 2.4 |
| GEMM | 264.8 |
| add | 1.2 |
| norm | 0.9 |
| rope | 0.5 |
| copy | 0.4 |
| glu | 0.6 |
| **total** | 270.8 |
| kernels | 590 |
