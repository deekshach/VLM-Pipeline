| stage | wall ms | GPU busy ms | idle | kernels | mean kernel us | median gap us | p90 gap us | idle in gaps > 1 ms |
|---|---|---|---|---|---|---|---|---|
| vit | 27036 | 26595 | 1.6% | 10408 | 2555 | 0.0 | 6.9 | 0% |
| prefill_img | 17572 | 17408 | 0.9% | 15040 | 1157 | 2.1 | 4.4 | 18% |
| prefill_txt | 1376 | 1335 | 3.0% | 5604 | 238 | 2.1 | 2.1 | 0% |
| decode | 1426 | 1409 | 1.2% | 2799 | 503 | 2.1 | 2.1 | 7% |
