# Frozen-checkpoint noise intervention

No retraining. All network weights, raw component means, and calibrated initial spacing stay fixed. Only sampling sigma_rel changes. Baseline rows reproduce the final evaluation. Noise toggling changes RNG consumption, so individual draw paths are not paired.

| Trained model | Sampling sigma_rel | Conditional SW1 ↓ | HQ % | Modes | Core width (ideal 1) |
|---|---:|---:|---:|---:|---:|
| gan_atoms | 0.0 | 0.5784 | 96.40 | 96 | 0.559 |
| gan_atoms | 0.025 | 0.5771 | 92.24 | 96 | 1.219 |
| gan_mog | 0.0 | 0.6002 | 96.84 | 100 | 0.453 |
| gan_mog | 0.025 | 0.5984 | 94.21 | 99 | 1.120 |
| ddgan_atoms | 0.0 | 0.1884 | 5.17 | 28 | 9.317 |
| ddgan_atoms | 0.025 | 0.1856 | 5.21 | 26 | 9.363 |
| ddgan_mog | 0.0 | 0.1762 | 4.96 | 28 | 9.425 |
| ddgan_mog | 0.025 | 0.1817 | 5.24 | 24 | 9.456 |
