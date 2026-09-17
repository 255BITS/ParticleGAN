# Frozen-checkpoint noise intervention

No retraining. All network weights, raw component means, and calibrated initial spacing stay fixed. Only sampling sigma_rel changes. Baseline rows reproduce the final evaluation. Noise toggling changes RNG consumption, so individual draw paths are not paired.

| Trained model | Sampling sigma_rel | Conditional SW1 ↓ | HQ % | Modes | Core width (ideal 1) |
|---|---:|---:|---:|---:|---:|
| gan_atoms | 0.0 | 0.7595 | 93.46 | 77 | 0.854 |
| gan_atoms | 0.025 | 0.7585 | 93.09 | 78 | 1.263 |
| gan_mog | 0.0 | 0.9032 | 95.19 | 77 | 0.451 |
| gan_mog | 0.025 | 0.9025 | 95.19 | 77 | 0.701 |
| ddgan_atoms | 0.0 | 0.1172 | 77.86 | 100 | 0.989 |
| ddgan_atoms | 0.025 | 0.1115 | 77.88 | 100 | 0.978 |
| ddgan_mog | 0.0 | 0.1080 | 79.75 | 100 | 0.900 |
| ddgan_mog | 0.025 | 0.1190 | 79.57 | 100 | 0.889 |
