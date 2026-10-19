| stage | wall ms | GPU busy ms | idle | kernels | mean kernel us | median gap us | p90 gap us | idle in gaps > 1 ms |
|---|---|---|---|---|---|---|---|---|
| vit | 8761 | 8117 | 7.4% | 10408 | 780 | 0.0 | 0.0 | 0% |
| prefill_img | 4919 | 4789 | 2.7% | 15264 | 314 | 1.4 | 2.0 | 18% |
| prefill_txt | 751 | 716 | 4.7% | 5688 | 126 | 1.4 | 1.4 | 12% |
| decode | 646 | 636 | 1.7% | 2809 | 226 | 1.4 | 1.4 | 0% |
