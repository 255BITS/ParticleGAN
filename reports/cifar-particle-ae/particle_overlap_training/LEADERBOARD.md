# Certified overlap intervention results

| Model | FID50k | Density | Coverage | Latent confusion |
|---|---:|---:|---:|---:|
| 80k parent | 15.7527 | 0.64686 | 64.26% | 0.0092% |
| 100k unchanged | 16.4609 | 0.59324 | 63.36% | 0.0244% |
| 100k freeze_centers | 16.5846 | 0.63840 | 65.09% | 0.0092% |
| 100k reduced_noise | 19.1835 | 0.60198 | 58.90% | 0.0824% |

All endpoints use identical 50k FID and 10k real/fake k5 coverage protocols. Latent confusion uses 32,768 matched samples. Parent is an 80k reference, not a matched-duration arm.

| Arm | Step | FID50k | Median nearest-center distance |
|---|---:|---:|---:|
| freeze_centers | 85000 | 15.7695 | 2.1637 |
| freeze_centers | 90000 | 15.7493 | 2.1637 |
| freeze_centers | 95000 | 16.6345 | 2.1637 |
| freeze_centers | 100000 | 16.5846 | 2.1637 |
| reduced_noise | 85000 | 16.4907 | 2.2424 |
| reduced_noise | 90000 | 16.7851 | 2.3363 |
| reduced_noise | 95000 | 18.3403 | 2.0299 |
| reduced_noise | 100000 | 19.1835 | 1.6733 |

Best resumed freeze point: 15.7493 at90k, practically tied with original80k15.7527. Best resumed reduced-noise point:16.4907 at85k; its starting sampling-only FID was15.9090. No new training queued.
