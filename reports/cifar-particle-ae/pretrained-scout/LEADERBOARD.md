# CIFAR-10 AE-GAN pretrained encoder scouts

4/4 certified runs complete; 5000 updates each. Ranked by final EMA FID5,000 (lower is better).

| Rank | Config | FID ↓ | Test MSE ↓ | Effective particles ↑ | Offset saturation | Train min |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 01_scratch_control | 26.523 | 0.08674 | 236.5 | 37.5% | 4.28 |
| 2 | 03_pretrained_recon03 | 28.134 | 0.11235 | 61.3 | 8.9% | 4.32 |
| 3 | 02_pretrained | 28.325 | 0.10083 | 45.9 | 0.4% | 4.44 |
| 4 | 04_pretrained_half_lr | 31.785 | 0.12134 | 180.7 | 14.5% | 4.48 |

One shared seed; no seed sweep. FID uses unconditional generated images and the CIFAR-10 train50k reference. Reconstruction uses the first 1,000 unaugmented test images; labels are unused. FID5k is a scouting metric and cannot be compared directly with historical FID50k. Particle usage describes encoder routing, not unconditional class coverage.

## Interpretation

The scratch control and pretrained baseline differ only in encoder architecture/initialization. The pretrained backbone is frozen through layer3, with trainable spatial query and offset heads. The reconstruction-weight variant changes 1.0 to 0.3; the learning-rate variant halves G/E, D and prior rates together. Every arm retains the existing pretrained discriminator.

Pretrained baseline minus scratch FID: +1.802. This is one matched trajectory, not evidence of seed-to-seed reliability.

- 01_scratch_control: FID changed -5.699 over the last evaluation interval (negative means improvement).
- 03_pretrained_recon03: FID changed -7.963 over the last evaluation interval (negative means improvement).
- 02_pretrained: FID changed -4.887 over the last evaluation interval (negative means improvement).
- 04_pretrained_half_lr: FID changed -11.543 over the last evaluation interval (negative means improvement).

## Recommendation

Select **01_scratch_control** by final scout FID (26.523). Inspect its sample grids, reconstruction and late FID trend before the longer run. Small scout gaps may reflect finite-sample noise.

Prepared `winner_long.yaml`: 30k updates, FID50k at the end, all 10k test reconstructions, and retained evaluation checkpoints. This starts a fresh longer trajectory with the winning configuration and same seed; it does not resume the scout checkpoint. It is not launched automatically.

```sh
.venv/bin/python -u experiments/follow_grid.py \
  --root runs/cifar_particle_ae/pretrained_long \
  --log runs/cifar_particle_ae/pretrained_long/PIPELINE.log -- \
  --configs 'reports/cifar-particle-ae/pretrained-scout/winner_long.yaml' --gpus 0 --workers_per_gpu 1 \
  --python .venv/bin/python --trainer experiments/train_cifar_particle_ae.py
```
