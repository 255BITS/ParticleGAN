# Validation before production

- Analytic Gaussian overlap tests: 2 passed in 2.86 seconds on GPU 1. Identical components give one bit of ambiguity; two separated Gaussians reproduce the analytic nearest-center error and lower error at smaller sigma.
- Six full frozen-checkpoint evaluations completed with valid pipeline certificates. Both baseline FIDs reproduced within 0.00002, and historical density/coverage reproduced exactly. Weights, sigma, checkpoint bytes and prior RNG/exposure state were unchanged by the probes.
- Actual 16-update full-state resume smokes passed for both training interventions. Frozen live/EMA center tensors and prior Adam state were bitwise unchanged; prior update RMS was zero. G/D/E changed. Reduced-noise live/EMA sigma equals parent sigma times 0.75; centers and G/D/E update. Frozen discriminator features and fixed sigma checks passed.
- First freeze smoke completed numerically but its source certificate was invalidated by final source/log-path edits while it ran. That result was not promoted. Both smokes were rerun/certified against the final frozen source before production.
- Production freeze-centers fork restored the exact 80k checkpoint and began finite updates with original learning rates. No existing trainer, lib, or particlegan source was edited.

Smoke FID128 values are not benchmark measurements. Production jobs retain FID50k and 10k reconstruction evaluation. Each 5k evaluation records latent geometry using an independent RNG with training precision flags restored afterward.
