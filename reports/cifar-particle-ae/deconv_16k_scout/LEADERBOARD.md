# Plain deconvolution, 16k particles, no normalization

| Step | FID50k |
|---|---:|
| 5000 | 77.2284 |
| 10000 | 46.6522 |
| 15000 | 37.6181 |
| 20000 | 33.3280 |
| 25000 | 30.6990 |
| 30000 | 28.8087 |
| 35000 | 27.0096 |
| 40000 | 26.2232 |

Best sampled: 26.2232 at 40000; final: 26.2232. Training: 23.94 minutes.

| Historical reference | Steps | FID50k |
|---|---:|---:|
| Residual CNN, 16k expanded at 10k | 40000 | 17.0982 |
| Residual CNN, 16k best | 80000 | 15.7527 |

New G/E/D head and 16,384 independent particle rows trained from scratch. Frozen pretrained D features and shared noise scale match the historical recipe. Historical CNN runs expanded a trained 1,024-particle model at 10k; these are contextual references, not a matched architecture ablation.
