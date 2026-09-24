# Noiseless Fourier-5 network-horizon probes

The shared noiseless public core had passed all 19 older toys under their frozen budgets. To test whether its native `grid100` Gaussian fit needed more G/D optimization, two predeclared variants changed only `name` and the global `network_lr_horizon_cap` from 1600 to 2400 or 3200. Every older host has at most 1600 updates, so this cap change does not change their LR schedule; a fresh all-22 replay would still be required before any shared-gate claim. Both native variants used seed 1234, 7,000 updates, Fourier 5, the generic affine-square generator, five terminal checks, and an independent 100,000-sample holdout on frozen source `1c1a086`.

| Network horizon | Config SHA-256 | Final modes | Final HQ | Final mass TV | Holdout HQ | Holdout center RMS | Holdout covariance trace bias | Terminal passes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2400 | `2693872c…` | 100/100 | .9529 | .05585 | .95100 | .463σ | −.203 | 0/5 |
| 3200 | `79a70a1d…` | 100/100 | .9584 | .06010 | .95671 | .400σ | −.108 | 0/5 |

Both runs completed the full budget and failed the original coverage and accuracy gates. The larger horizon improved final HQ slightly, but neither reached the original .97 HQ threshold; their holdout center and covariance errors also exceed the fidelity bounds. Full evidence, including all samples, checkpoints, event logs, holdouts, resolved configs, manifest, and V2 source archives, is in affine-noiseless-horizon-v1 (`artifacts/toy100-accuracy/affine-noiseless-horizon-v1`). All 110 copied files matched RAM originals by SHA-256, and both relocated runs independently regraded as valid failures with verified source archives.
