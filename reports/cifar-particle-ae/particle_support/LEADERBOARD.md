# Read-only particle-support diagnostics

3/3 certified; standard-sampler FID reproduction within 0.01 for every parent.

| Checkpoint | Noise x0 FID50k | Original x1 FID50k | Noise x2 FID50k | Within-particle feature variation |
|---|---:|---:|---:|---:|
| parent_10k | 41.8580 | 19.4481 | 21.7837 | 33.06% |
| control_20k | 42.4706 | 20.3039 | 21.7199 | 32.11% |
| half_g_20k | 43.4404 | 20.7005 | 23.5012 | 33.26% |

Noise x0 repeatedly samples the 1024 particle centers, so its FID describes a discrete empirical generator. Noise x2 is an inference-only distribution change. These are not training results. Within-particle variation is a balanced ANOVA over 128 particles and 16 draws per particle, not a class-coverage or recall score.
