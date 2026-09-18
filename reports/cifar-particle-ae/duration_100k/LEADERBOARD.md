# CIFAR AE-GAN: duration_100k

1/1 certified runs complete. Target: FID50k below 13.

| Rank | Run | G params | Updates (start → end) | Final FID50k ↓ | Test MSE ↓ | Updates/s | New train min |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | n08 | 645,123 | 30,000 → 100,000 | 20.806 | 0.03371 | 22.36 | 52.17 |

N=8 double-backprop bcap (coefficient multiplied by 8); scratch encoder; frozen pretrained discriminator features. Width 32 for D/E throughout. Same seed for configuration control; no seed sweep. FID uses 50k generated images, CIFAR train50k reference, TF-compatible Inception and EMA weights. Reconstruction uses 10k test images. Both GPUs run concurrently, so timing includes shared-host effects.

Historical width32 baseline: FID50k **19.611 at 10k**, **20.105 at 20k**, **19.439 at 30k** ([audited curve](../lazy-long/README.md)). This is a historical reference, not a repeated control.

## Learning curves

| Run | Step | FID50k ↓ | Test MSE ↓ |
|---|---:|---:|---:|
| n08 | 40,000 | 19.424 | 0.04220 |
| n08 | 50,000 | 18.901 | 0.04022 |
| n08 | 60,000 | 19.977 | 0.03861 |
| n08 | 70,000 | 20.355 | 0.03715 |
| n08 | 80,000 | 20.340 | 0.03642 |
| n08 | 90,000 | 21.720 | 0.03584 |
| n08 | 100,000 | 20.806 | 0.03371 |

## Recommendation

Best final measurement: **n08, FID50k 20.806**. The remaining gap to 13 is 7.806.
Continuation changed FID by +1.367 relative to the saved 30k checkpoint. Use the last several FID50k measurements to judge remaining progress; improving reconstruction alone does not establish that generation FID will catch up.
Selection uses final measurements; intermediate values describe the curve. Review both track reports before selecting the next long run.
