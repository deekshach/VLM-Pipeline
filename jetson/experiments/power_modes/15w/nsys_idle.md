| stage | wall ms | GPU busy ms | idle | kernels | mean kernel us | median gap us | p90 gap us | idle in gaps > 1 ms |
|---|---|---|---|---|---|---|---|---|
| vit | 52271 | 51628 | 1.2% | 10408 | 4960 | 0.0 | 10.0 | 0% |
| prefill_img | 31044 | 30805 | 0.8% | 15040 | 2048 | 3.2 | 6.6 | 17% |
| prefill_txt | 2374 | 2313 | 2.5% | 5604 | 413 | 3.2 | 3.3 | 0% |
| decode | 2786 | 2760 | 0.9% | 2799 | 986 | 3.2 | 3.3 | 39% |
