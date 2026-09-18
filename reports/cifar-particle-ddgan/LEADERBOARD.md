# CIFAR particle DDGAN comparison

Final EMA at exactly 10k training updates. FID uses 50k generated images; lower is better.
Unconditional, same fixed-sigma MoG prior and learning rates. One shared seed, no best-checkpoint selection.

| Model | FID50k ↓ | FID5k ↓ | Feature variance /real | Train min | Total min | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| Direct GAN | 19.483 | 24.549 | 1.071 | 7.91 | 11.55 | 5.32 |
| Particle AE-GAN | 20.054 | 24.691 | 1.078 | 8.77 | 12.16 | 5.32 |
| DDGAN + particle AE | 43.233 | 47.380 | 0.992 | 10.65 | 15.13 | 5.34 |
| DDGAN | 49.475 | 53.585 | 0.951 | 9.10 | 13.26 | 5.33 |

Particle AE arms use deterministic bounded encoders plus reconstruction; no KL or variational objective.
Feature variance is a coarse spread metric, not semantic mode coverage. Cross-architecture compute and parameter counts differ.

## Same-count learning curves

| Model | FID5k at 2500 | 5000 | 7500 | 10000 |
|---|---:|---:|---:|---:|
| Direct GAN | 35.093 | 28.344 | 26.175 | 24.549 |
| Particle AE-GAN | 32.212 | 27.304 | 25.230 | 24.691 |
| DDGAN | 85.443 | 64.139 | 59.835 | 53.585 |
| DDGAN + particle AE | 71.607 | 53.199 | 49.392 | 47.380 |

## DDGAN clean prediction at fixed noisy input

MSE[-1,1] on all 10k test images; each t has a different corruption level. These are not latent-only reconstructions.

| t | DDGAN prior | DDGAN + particle AE encoded | Shuffled code | Prior code | Zero offset | Shuffled particle |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.017820 | 0.013682 | 0.014355 | 0.017835 | 0.017814 | 0.014167 |
| 2 | 0.056135 | 0.037830 | 0.044232 | 0.058420 | 0.057768 | 0.040818 |
| 3 | 0.193615 | 0.085023 | 0.270489 | 0.221854 | 0.181240 | 0.112945 |
| 4 | 0.409462 | 0.097218 | 0.398152 | 0.480671 | 0.317364 | 0.151379 |

Direct Particle AE-GAN test MSE 0.078996, PSNR 17.04dB; zero-offset 0.304904, random-offset 0.305342, shuffled-particle 0.128659.

| Encoder | Used /1024 | Effective /1024 | Offset RMS | Saturated coordinates |
|---|---:|---:|---:|---:|
| Particle AE-GAN | 611 | 295.1 | 2.391 | 31.07% |
| DDGAN + particle AE | 657 | 433.4 | 2.601 | 50.70% |

## Numerical variation

512 test inputs, eight draws each; DDGAN holds X_t and t fixed and measures predicted clean images before transition noise.
Own-anchor retrieval is relative to deterministic reconstructions, not semantic identity accuracy.

| Model | t | Code | Pair pixel RMSE (0–255 levels) | Feature cosine distance | MSE | Own-anchor retrieval |
|---|---:|---|---:|---:|---:|---:|
| Particle AE-GAN | 0 | encoded + 0sigma | 0.00 | 0.0000 | 0.078328 | 100.00% |
| Particle AE-GAN | 0 | encoded + 0.5sigma | 6.96 | 0.0383 | 0.079843 | 99.78% |
| Particle AE-GAN | 0 | encoded + 1sigma | 13.62 | 0.0867 | 0.084145 | 94.46% |
| DDGAN | 1 | prior | 2.97 | 0.0302 | 0.018033 | — |
| DDGAN | 2 | prior | 5.57 | 0.0578 | 0.055874 | — |
| DDGAN | 3 | prior | 30.89 | 0.1721 | 0.190916 | — |
| DDGAN | 4 | prior | 17.88 | 0.1023 | 0.420075 | — |
| DDGAN + particle AE | 1 | encoded + 0sigma | 0.00 | 0.0000 | 0.013833 | 100.00% |
| DDGAN + particle AE | 1 | encoded + 0.5sigma | 0.86 | 0.0048 | 0.013851 | 100.00% |
| DDGAN + particle AE | 1 | encoded + 1sigma | 1.52 | 0.0125 | 0.013898 | 100.00% |
| DDGAN + particle AE | 1 | prior | 3.82 | 0.0428 | 0.017965 | 83.30% |
| DDGAN + particle AE | 2 | encoded + 0sigma | 0.00 | 0.0000 | 0.037806 | 100.00% |
| DDGAN + particle AE | 2 | encoded + 0.5sigma | 1.15 | 0.0068 | 0.037836 | 100.00% |
| DDGAN + particle AE | 2 | encoded + 1sigma | 2.14 | 0.0171 | 0.037922 | 100.00% |
| DDGAN + particle AE | 2 | prior | 9.68 | 0.1070 | 0.058328 | 30.20% |
| DDGAN + particle AE | 3 | encoded + 0sigma | 0.00 | 0.0000 | 0.084872 | 100.00% |
| DDGAN + particle AE | 3 | encoded + 0.5sigma | 3.16 | 0.0098 | 0.085182 | 100.00% |
| DDGAN + particle AE | 3 | encoded + 1sigma | 6.22 | 0.0264 | 0.086051 | 99.93% |
| DDGAN + particle AE | 3 | prior | 45.51 | 0.2575 | 0.220514 | 1.03% |
| DDGAN + particle AE | 4 | encoded + 0sigma | 0.00 | 0.0000 | 0.098704 | 100.00% |
| DDGAN + particle AE | 4 | encoded + 0.5sigma | 4.06 | 0.0169 | 0.099179 | 100.00% |
| DDGAN + particle AE | 4 | encoded + 1sigma | 8.01 | 0.0435 | 0.100591 | 99.90% |
| DDGAN + particle AE | 4 | prior | 73.40 | 0.2964 | 0.483151 | 0.22% |

Verified: current-source completion certificates, full budgets, pair initializations and RNG states, cross-architecture data/prior RNG states, fixed sigma/frozen critic, per-image reconstruction errors, and four checkpoint hashes per arm.
