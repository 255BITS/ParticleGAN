# Particle expansion scout

2/2 certified. Same E-only 10k parent; initial FID50k 19.4482.

| Arm | Initial FID | Midpoint FID | Final FID | Train minutes | Wall minutes | Sibling latent RMS |
|---|---:|---:|---:|---:|---:|---:|
| split_16384 | 19.4480 | 18.3401 | 18.0136 | 7.46 | 11.12 | 0.14442 |
| split_8192 | 19.4479 | 18.6479 | 18.1602 | 7.28 | 10.87 | 0.13054 |

Cloned centers preserve initial normalization and fixed sigma. Per-row Adam moments copied; rates unchanged. Reference-count variance/covariance correction preserves the initial particle regularizer. Additional centers change sampling exposure and optimizer dynamics. Paired original data/noise streams audited. Sibling distances and image variation are not semantic coverage metrics.

## Existing matched-parent benchmarks

| Centers | FID15k | FID20k | Train minutes |
|---|---:|---:|---:|
| 1024 | 19.8699 | 19.8932 | 7.25 |
| 4096 | 18.9357 | 18.5207 | 7.37 |
| 8192 | 18.6479 | 18.1602 | 7.28 |
| 16384 | 18.3401 | 18.0136 | 7.46 |
