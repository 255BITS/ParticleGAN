# CIFAR AE-GAN: duration_100k

0/1 certified runs complete. Target: FID50k below 13.

| Rank | Run | G params | Updates (start → end) | Final FID50k ↓ | Test MSE ↓ | Updates/s | New train min |
|---:|---|---:|---|---:|---:|---:|---:|

N=8 double-backprop bcap (coefficient multiplied by 8); scratch encoder; frozen pretrained discriminator features. Width 32 for D/E throughout. Same seed for configuration control; no seed sweep. FID uses 50k generated images, CIFAR train50k reference, TF-compatible Inception and EMA weights. Reconstruction uses 10k test images. Both GPUs run concurrently, so timing includes shared-host effects.

Historical width32 baseline: FID50k **19.611 at 10k**, **20.105 at 20k**, **19.439 at 30k** ([audited curve](../lazy-long/README.md)). This is a historical reference, not a repeated control.

## Learning curves

| Run | Step | FID50k ↓ | Test MSE ↓ |
|---|---:|---:|---:|

## Recommendation

Provisional: incomplete or uncertified runs: n08.
