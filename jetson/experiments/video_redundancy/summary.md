| dataset | clips | frames | sampled | intra | inter | skip | new_info | same30 | same35 | same40 | info_top25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| msvd | 100 | 1804 | 19 | 0.465 | 0.492 | 0.043 | 0.854 | 0.314 | 0.23 | 0.145 | 0.329 |
| sintel | 23 | 69 | 4 | 0.587 | 0.38 | 0.0332 | 0.911 | 0.094 | 0.0491 | 0.0345 | 0.361 |
| charades | 10 | 538 | 54.8 | 0.289 | 0.668 | 0.0427 | 0.661 | 0.549 | 0.462 | 0.311 | 0.34 |
| nextqa | 10 | 910 | 92 | 0.597 | 0.396 | 0.00671 | 1.01 | 0.132 | 0.0721 | 0.0327 | 0.291 |
| videomme | 10 | 1288 | 130 | 0.413 | 0.397 | 0.19 | 0.671 | 0.45 | 0.393 | 0.336 | 0.41 |
| perception | 8 | 506 | 64.2 | 0.215 | 0.752 | 0.0323 | 0.639 | 0.619 | 0.481 | 0.339 | 0.341 |
| tacos | 9 | 137 | 16.2 | 0.0153 | 0.54 | 0.445 | 0.195 | 0.943 | 0.927 | 0.897 | 0.445 |
| all | 170 | 5252 | 31.9 | 0.44 | 0.491 | 0.0689 | 0.804 | 0.343 | 0.268 | 0.194 | 0.344 |

Means over clips; frames = sampled frames after each clip's first. Shares 0-1; new_info = a frame's size coded after the previous sampled frame / coded on its own; info_top25 = share of a clip's new information in its top quarter of frames.
