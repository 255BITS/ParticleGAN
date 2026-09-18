# Particle expansion scout

2/2 certified. SMOKE ONLY: 128-image FID is not a benchmark.

| Arm | Initial FID | Midpoint FID | Final FID | Train minutes | Wall minutes | Sibling latent RMS |
|---|---:|---:|---:|---:|---:|---:|
| split_8192 | 143.6044 | 143.8337 | 143.8337 | 0.05 | 0.49 | 0.00005 |
| split_16384 | 143.6228 | 143.9325 | 143.9325 | 0.05 | 0.54 | 0.00004 |

Cloned centers preserve initial normalization and fixed sigma. Per-row Adam moments copied; rates unchanged. Reference-count variance/covariance correction preserves the initial particle regularizer. Additional centers change sampling exposure and optimizer dynamics. Paired original data/noise streams audited. Sibling distances and image variation are not semantic coverage metrics.
