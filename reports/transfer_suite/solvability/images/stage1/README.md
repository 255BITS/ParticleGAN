# Image GAN solvability: shared configurations

Seed0, unchanged live quality/coverage gates, 24 observations and five final passing checks. Architecture and budget changes are explicit. Supervised controls do not enter GAN ranking. Previously reserved residual bars are now seen development data; no fresh holdout is evaluated.

| Shared card | Scope | Sustained / 4 | Stripes | Bars | Blobs | Intensity | CPU seconds |
| --- | --- | ---: | --- | --- | --- | --- | ---: |
| residual16 | architecture changed | 4/4 | PASS 2/2 · 100.0% | PASS 4/4 · 93.8% | PASS 4/4 · 100.0% | PASS 2/2 · 100.0% | 37.2 |
| residual16_r1r2_01 | architecture changed | 4/4 | PASS 2/2 · 100.0% | PASS 4/4 · 93.8% | PASS 4/4 · 100.0% | PASS 2/2 · 93.8% | 29.5 |
| r1r2_1 | same architecture/budget | 2/4 | PASS 2/2 · 100.0% | FAIL 4/4 · 84.4% | PASS 4/4 · 93.8% | FAIL 0/2 · 9.4% | 30.1 |
| vanilla_r1r2_01 | same architecture/budget | 1/4 | PASS 2/2 · 100.0% | FAIL 3/4 · 100.0% | FAIL 4/4 · 87.5% | FAIL 0/2 · 21.9% | 31.9 |
| r1r2_01 | same architecture/budget | 1/4 | PASS 2/2 · 100.0% | FAIL 3/4 · 93.8% | FAIL 2/4 · 87.5% | FAIL 0/2 · 3.1% | 31.0 |
| baseline | same architecture/budget | 1/4 | FAIL 1/2 · 100.0% | FAIL 2/4 · 100.0% | PASS 4/4 · 96.9% | FAIL 0/2 · 0.0% | 26.3 |
| fixed_prior | same architecture/budget | 1/4 | FAIL 1/2 · 100.0% | FAIL 2/4 · 65.6% | PASS 4/4 · 96.9% | FAIL 0/2 · 0.0% | 39.1 |
| baseline_1200 | budget doubled | 0/4 | FAIL 1/2 · 93.8% | FAIL 3/4 · 100.0% | FAIL 4/4 · 100.0% | FAIL 0/2 · 3.1% | 51.0 |
| prior_lr10 | same architecture/budget | 0/4 | FAIL 1/2 · 93.8% | FAIL 1/4 · 90.6% | FAIL 3/4 · 100.0% | FAIL 0/2 · 0.0% | 42.3 |
| transpose24 | architecture changed | 0/4 | FAIL 1/2 · 100.0% | FAIL 2/4 · 90.6% | FAIL 2/4 · 75.0% | FAIL 0/2 · 0.0% | 33.1 |
| cap_coeff03 | same architecture/budget | 0/4 | FAIL 1/2 · 100.0% | FAIL 2/4 · 78.1% | FAIL 1/4 · 87.5% | FAIL 0/2 · 0.0% | 28.8 |
| lsgan_vanilla | same architecture/budget | 0/4 | FAIL 1/2 · 100.0% | FAIL 1/4 · 100.0% | FAIL 0/4 · 0.0% | FAIL 0/2 · 0.0% | 32.6 |

Each cell reports sustained verdict, final quality-qualified modes and HQ. A final pass with an insufficient passing suffix remains FAIL. Missing tasks cannot form a shared winner.

| Supervised expressivity control (not a GAN) | Result | Confirmed step |
| --- | --- | ---: |
| img_stripes2 | PASS 2/2 · 100.0% | 225 |
| img_bars4 | PASS 4/4 · 100.0% | 275 |
| img_blobs4 | FAIL 3/4 · 53.1% | — |
| img_intensity2 | PASS 2/2 · 100.0% | 250 |

Every exact spec, failure, curve, action trace and source/runtime hash is retained in the episode JSON.gz artifacts.
