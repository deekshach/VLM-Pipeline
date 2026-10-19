| stage | wall ms | GPU busy ms | idle | kernels | mean kernel us | median gap us | p90 gap us | idle in gaps > 1 ms |
|---|---|---|---|---|---|---|---|---|
| vit | 11726 | 11187 | 4.6% | 10408 | 1075 | 0.0 | 0.0 | 0% |
| prefill_img | 7308 | 7140 | 2.3% | 15264 | 468 | 1.7 | 2.4 | 18% |
| prefill_txt | 865 | 819 | 5.4% | 5688 | 144 | 1.7 | 1.8 | 12% |
| decode | 745 | 730 | 2.0% | 2813 | 259 | 1.7 | 1.8 | 15% |
