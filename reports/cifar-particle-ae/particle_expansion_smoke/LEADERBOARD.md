# Particle expansion scout

2/2 certified. SMOKE ONLY: 128-image FID is not a benchmark.

| Arm | Initial FID | Midpoint FID | Final FID | Train minutes | Wall minutes | Sibling latent RMS |
|---|---:|---:|---:|---:|---:|---:|
| control_1024 | 143.6053 | 143.8138 | 143.8138 | 0.05 | 0.49 | 0.00000 |
| split_4096 | 143.6062 | 143.8914 | 143.8914 | 0.05 | 0.50 | 0.00007 |

Cloned centers preserve initial normalization and fixed sigma. Per-row Adam moments copied; rates unchanged. Reference-count variance/covariance correction preserves the initial particle regularizer. Additional centers change sampling exposure and optimizer dynamics. Paired original data/noise streams audited. Sibling distances and image variation are not semantic coverage metrics.
