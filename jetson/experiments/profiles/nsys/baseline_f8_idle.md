| stage | wall ms | GPU busy ms | idle | kernels | mean kernel us | median gap us | p90 gap us | idle in gaps > 1 ms |
|---|---|---|---|---|---|---|---|---|
| vit | 9489 | 9089 | 4.2% | 10408 | 873 | 0.0 | 0.2 | 0% |
| prefill_img | 5347 | 5211 | 2.6% | 15264 | 341 | 1.3 | 2.0 | 22% |
| prefill_txt | 824 | 781 | 5.2% | 5688 | 137 | 1.4 | 1.4 | 15% |
| decode | 683 | 669 | 2.0% | 2811 | 238 | 1.4 | 1.4 | 11% |
