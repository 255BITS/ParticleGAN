# CIFAR AE-GAN: capacity_scout

2/2 certified runs complete. Target: FID50k below 13.

| Rank | Run | G params | Updates (start → end) | Final FID50k ↓ | Test MSE ↓ | Updates/s | New train min |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | g64_deep | 4,072,963 | 0 → 20,000 | 25.778 | 0.04244 | 13.87 | 24.04 |
| 2 | g64 | 2,291,715 | 0 → 20,000 | 35.440 | 0.04964 | 17.51 | 19.03 |

N=8 double-backprop bcap (coefficient multiplied by 8); scratch encoder; frozen pretrained discriminator features. Width 32 for D/E throughout. Same seed for configuration control; no seed sweep. FID uses 50k generated images, CIFAR train50k reference, TF-compatible Inception and EMA weights. Reconstruction uses 10k test images. Both GPUs run concurrently, so timing includes shared-host effects.

Historical width32 baseline: FID50k **19.611 at 10k**, **20.105 at 20k**, **19.439 at 30k** ([audited curve](../lazy-long/README.md)). This is a historical reference, not a repeated control.

## Learning curves

| Run | Step | FID50k ↓ | Test MSE ↓ |
|---|---:|---:|---:|
| g64_deep | 10,000 | 37.000 | 0.06275 |
| g64_deep | 20,000 | 25.778 | 0.04244 |
| g64 | 10,000 | 21.020 | 0.05106 |
| g64 | 20,000 | 35.440 | 0.04964 |

## Recommendation

Best final measurement: **g64_deep, FID50k 25.778**. The remaining gap to 13 is 12.778.
Extend the leading capacity configuration if its improvement is meaningful and its curve supports further training. Compare its time per update with the duration track. If both remain near or above the historical 20k result (20.105), these scouts do not support spending more compute on width/depth alone.
Selection uses final measurements; intermediate values describe the curve. Review both track reports before selecting the next long run.
