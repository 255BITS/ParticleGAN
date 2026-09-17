# Direct CIFAR particle autoencoder leaderboard

Final EMA weights, 10k updates, FID50k ascending. One shared seed; labels unused.

| Arm | FID50k ↓ | Feature variance /real | Test MSE ↓ | Effective /1024 | Offset RMS | Train min | Total min | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bounded | 33.266 | 1.081 | 0.06337 | 109.5 | 2.040 | 8.61 | 10.84 | 5.32 |
| gan | 81.411 | 1.020 | — | — | — | 7.99 | 10.32 | 5.32 |

Feature variance is a coarse spread diagnostic, not measured mode coverage.
MSE uses [-1,1] pixels and all 10k unaugmented CIFAR test images. Generation FID uses the existing CIFAR train50k reference.
Historical conditional DDGAN results are not a matched control.

## Reconstruction ablations

| Evaluation | Test MSE ↓ | Change from predicted |
|---|---:|---:|
| recon | 0.063365 | +0.0% |
| zero_offset | 0.364833 | +475.8% |
| random_offset | 0.367605 | +480.1% |
| shuffled_particle | 0.073319 | +15.7% |

Per-image error p90 0.10366, p99 0.15860; aggregate PSNR 18.00 dB.
Used 363/1024 particles; hard usage TV 0.771, hard–soft usage TV 0.366.
Offset saturation (abs >2.9): 11.31%; conditional mean RMS 1.241.

Verified: identical shared configs, initialization, pretrained weights, final data/prior RNG states, complete evaluation budgets,
fixed sigma, frozen feature weights, saved reconstruction errors/counts, and current-source completion certificates.

## Same-count diagnostic audit

The final checkpoints were additionally evaluated at 5k samples to distinguish training deterioration from sample-count effects. No updates were made.

| Arm | FID5k at 2500 | At 5000 | At 7500 | At 10000 (audit) |
|---|---:|---:|---:|---:|
| gan | 31.685 | 27.486 | 25.951 | 85.361 |
| bounded | 31.308 | 28.512 | 25.267 | 37.918 |

Audit sample grids reproduce the original final grids within one uint8 quantization level. Checkpoint hashes are unchanged.
