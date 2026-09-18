# Particle expansion scout

2/2 certified. Same E-only 10k parent; initial FID50k 19.4482.

| Arm | Initial FID | Midpoint FID | Final FID | Train minutes | Wall minutes | Sibling latent RMS |
|---|---:|---:|---:|---:|---:|---:|
| split_4096 | 19.4481 | 18.9357 | 18.5207 | 7.37 | 10.94 | 0.11283 |
| control_1024 | 19.4481 | 19.8699 | 19.8932 | 7.25 | 10.82 | 0.00000 |

Cloned centers preserve initial normalization and fixed sigma. Per-row Adam moments copied; rates unchanged. Reference-count variance/covariance correction preserves the initial particle regularizer. Additional centers change sampling exposure and optimizer dynamics. Paired original data/noise streams audited. Sibling distances and image variation are not semantic coverage metrics.
