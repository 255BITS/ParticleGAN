# Conditional variation: diversity/fidelity table

One frozen EMA checkpoint. 512 held-out inputs, eight draws each, all 28 unordered pairs per input. Ordered by noise strength, not a universal quality ranking.

| Sampling condition | Unique /8 | Pair pixel RMSE /255 | Feature diversity /unrelated recon | Input MSE | MSE increase | Own anchor nearest |
|---|---:|---:|---:|---:|---:|---:|
| z_X + sigma × 0 × noise | 1.0 | 0.00 | 0.0% | 0.06296 | +0.0% | 100.00% |
| z_X + sigma × 0.25 × noise | 8.0 | 5.60 | 9.6% | 0.06388 | +1.5% | 99.98% |
| z_X + sigma × 0.5 × noise | 8.0 | 11.07 | 23.1% | 0.06661 | +5.8% | 99.34% |
| z_X + sigma × 1 × noise | 8.0 | 21.49 | 45.4% | 0.07716 | +22.6% | 82.23% |
| z_X + sigma × 2 × noise | 8.0 | 39.54 | 72.8% | 0.11375 | +80.7% | 28.27% |
| p[k] + sigma × noise | 8.0 | 42.43 | 70.2% | 0.36786 | +484.3% | 0.49% |

Feature diversity = mean pairwise cosine distance in normalized Inception pool2048 features, divided by the distance between unrelated deterministic reconstructions.
Unrelated reference: cosine distance 0.295645, pixel RMSE 78.13/255. Unrelated real images have cosine distance 0.404142.
Own-anchor retention retrieves the nearest of 512 deterministic reconstructions in feature space. It is not original-image identity or class accuracy.
MSE uses [-1,1] float pixels. Uniqueness and feature extraction use uint8 images. Zero-noise cosine distance has a ~1e-8 floating-point floor despite identical pixels.
Runtime 29.8 sec; peak 1.76 GiB. No training, image grids, visual inspection, or checkpoint changes.
