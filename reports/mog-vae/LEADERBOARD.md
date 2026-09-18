# Particle VAE toy leaderboard

Twelve prespecified configurations; one shared seed; final online weights at 6,000 updates.
Generation uses 100k decoder means G(z). Rank: coverage, then HQ, then SW1. Width should approach 1.
This rank is generation-focused; reconstruction and posterior usefulness have separate tradeoffs.

| Rank | Run | Modes /100 | HQ % ↑ | Width /real ≈1 | Balance KL ↓ | SW1 ↓ | Sampled recon MSE ↓ | MAP MSE ↓ | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 02_categorical_no_kl_gan | 95 | 76.01 | 0.6676 | 0.2297 | 0.293 | 0.004246 | 0.004429 | 40.46 |
| 2 | cat_sharp_obs003_gan | 90 | 72.92 | 0.6097 | 0.2974 | 0.3057 | 0.005626 | 0.006311 | 40.51 |
| 3 | local_sharp_obs010_gan | 85 | 66.6 | 0.6697 | 0.3967 | 0.3189 | 0.005963 | 0.005665 | 43.06 |
| 4 | local_sharp_obs003_gan | 85 | 63.42 | 0.7493 | 0.3722 | 0.2849 | 0.006273 | 0.006381 | 41.95 |
| 5 | 01_ae_gan | 84 | 73.5 | 0.6707 | 0.4072 | 0.3946 | 0.004629 | 0.004629 | 38.61 |
| 6 | cat_sharp_obs030_gan | 82 | 71.63 | 0.5191 | 0.3852 | 0.3048 | 0.005232 | 0.00545 | 40.66 |
| 7 | 00_gan | 80 | 59.31 | 0.7714 | 0.4172 | 0.2755 | — | — | 29.25 |
| 8 | cat_broad_obs030_gan | 77 | 70.28 | 0.6207 | 0.5356 | 0.5236 | 0.009971 | 0.005342 | 39.75 |
| 9 | cat_broad_obs003_gan | 74 | 68.78 | 0.5315 | 0.5844 | 0.5313 | 0.01499 | 0.01485 | 40.53 |
| 10 | cat_sharp_obs010_gan | 69 | 59.73 | 0.5275 | 0.5618 | 0.3478 | 0.008228 | 0.008255 | 40.32 |
| 11 | cat_broad_obs010_gan | 65 | 54.29 | 0.6368 | 0.6145 | 0.513 | 0.01118 | 0.0106 | 39.42 |
| 12 | cat_sharp_obs010_no_gan | 62 | 42.16 | 5.189 | 0.884 | 0.3526 | 0.009846 | 0.009735 | 19.6 |

## Posterior and variation

Eight draws/input on 8,192 held-out inputs. Pair RMS is Euclidean output distance; same-mode checks input-mode retention.
Conditional effective K is exp(mean categorical entropy). MI is a held-out categorical mutual-information estimate.
AE has no posterior: zero pair RMS is expected; its soft routing probabilities are only gradient diagnostics.

| Run | KL categorical | KL local | Conditional effective K | Aggregate effective K (sampled) | MI nats | Pair RMS | Same-mode % | Shuffled MSE | Local std /sigma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 02_categorical_no_kl_gan | 4.824 | 0 | 3.215 | 320.8 | 4.604 | 0.04216 | 100 | 16.5 | 1 |
| cat_sharp_obs003_gan | 4.819 | 0 | 3.228 | 321.7 | 4.604 | 0.03864 | 100 | 16.23 | 1 |
| local_sharp_obs010_gan | 4.747 | 0.0001038 | 3.471 | 346 | 4.605 | 0.04499 | 100 | 16.08 | 0.997 |
| local_sharp_obs003_gan | 4.811 | 0.001244 | 3.257 | 324.7 | 4.605 | 0.04551 | 100 | 16.05 | 0.9817 |
| 01_ae_gan | 0 | 0 | 39.81 | 189.4 | — | 0 | 100 | 16.55 | 0 |
| cat_sharp_obs030_gan | 4.76 | 0 | 3.428 | 341.2 | 4.603 | 0.03856 | 100 | 16.42 | 1 |
| cat_broad_obs030_gan | 4.772 | 0 | 3.385 | 319.2 | 4.549 | 0.1379 | 99.17 | 16.58 | 1 |
| cat_broad_obs003_gan | 4.857 | 0 | 3.111 | 295.7 | 4.557 | 0.07322 | 97.6 | 16.58 | 1 |
| cat_sharp_obs010_gan | 4.724 | 0 | 3.551 | 353.7 | 4.604 | 0.03659 | 100 | 15.96 | 1 |
| cat_broad_obs010_gan | 4.848 | 0 | 3.139 | 300.3 | 4.563 | 0.09277 | 98.81 | 16.36 | 1 |
| cat_sharp_obs010_no_gan | 5.222 | 0 | 2.159 | 214.4 | 4.6 | 0.0268 | 100 | 16.7 | 1 |

## Likelihood check

These samples include the decoder likelihood noise: G(z)+tau*noise. They are distinct from decoder means above.
No-KL control has an evaluable bound but does not optimize it. ELBO column is negative ELBO in nats, lower is better.

| Run | Tau | Negative ELBO ↓ | Predictive modes | Predictive HQ % ↑ |
|---|---:|---:|---:|---:|
| 02_categorical_no_kl_gan | 0.1 | 2.481 | 100 | 27.54 |
| cat_sharp_obs003_gan | 0.03 | 5.895 | 95 | 67.12 |
| local_sharp_obs010_gan | 0.1 | 2.576 | 99 | 25.74 |
| local_sharp_obs003_gan | 0.03 | 6.607 | 96 | 58.58 |
| cat_sharp_obs030_gan | 0.3 | 4.248 | 96 | 4.332 |
| cat_broad_obs030_gan | 0.3 | 4.313 | 93 | 4.352 |
| cat_broad_obs003_gan | 0.03 | 16.34 | 81 | 63.43 |
| cat_sharp_obs010_gan | 0.1 | 2.78 | 99 | 23.57 |
| cat_broad_obs010_gan | 0.1 | 3.199 | 98 | 23.48 |
| cat_sharp_obs010_no_gan | 0.1 | 3.439 | 96 | 16.45 |

## Learning curves and cost

At 2k/4k generation uses 20k samples; final uses 100k. SW1 always uses 8192.

| Run | 2k modes / HQ% / SW1 | 4k | 6k | Total process s | Peak GiB |
|---|---|---|---|---:|---:|
| 02_categorical_no_kl_gan | 77 / 54.85 / 0.3768 | 87 / 72.16 / 0.3125 | 95 / 76.01 / 0.2930 | 42 | 0.114 |
| cat_sharp_obs003_gan | 66 / 49.90 / 0.3048 | 73 / 49.12 / 0.3287 | 90 / 72.92 / 0.3057 | 42.14 | 0.114 |
| local_sharp_obs010_gan | 77 / 65.53 / 0.3701 | 86 / 72.74 / 0.3275 | 85 / 66.60 / 0.3189 | 44.61 | 0.114 |
| local_sharp_obs003_gan | 71 / 59.48 / 0.3288 | 88 / 71.76 / 0.2957 | 85 / 63.42 / 0.2849 | 43.5 | 0.114 |
| 01_ae_gan | 86 / 80.79 / 0.5033 | 99 / 92.66 / 0.4497 | 84 / 73.50 / 0.3946 | 40.37 | 0.1012 |
| cat_sharp_obs030_gan | 72 / 63.68 / 0.3738 | 87 / 77.85 / 0.3185 | 82 / 71.63 / 0.3048 | 42.22 | 0.114 |
| 00_gan | 85 / 64.59 / 0.2917 | 93 / 76.55 / 0.2687 | 80 / 59.31 / 0.2755 | 30.97 | 0.101 |
| cat_broad_obs030_gan | 94 / 80.17 / 0.5733 | 81 / 66.99 / 0.5719 | 77 / 70.28 / 0.5236 | 41.34 | 0.114 |
| cat_broad_obs003_gan | 54 / 50.49 / 0.6422 | 62 / 64.39 / 0.5329 | 74 / 68.78 / 0.5313 | 42.06 | 0.114 |
| cat_sharp_obs010_gan | 54 / 49.42 / 0.3815 | 87 / 72.93 / 0.3061 | 69 / 59.73 / 0.3478 | 41.85 | 0.114 |
| cat_broad_obs010_gan | 64 / 60.90 / 0.6252 | 84 / 75.17 / 0.5562 | 65 / 54.29 / 0.5130 | 41 | 0.114 |
| cat_sharp_obs010_no_gan | 99 / 43.54 / 0.3736 | 99 / 61.20 / 0.3617 | 62 / 42.16 / 0.3526 | 21.15 | 0.1136 |

Total training: 7.57 GPU minutes; total child-process time: 7.89 minutes (includes evaluation/I/O).

All 12 completion certificates, configurations, matched initializations/data/prior RNGs, fixed sigma, 36 checkpoint hashes, and per-input reconstruction metrics verified.
No seed-only runs, statistical superiority claim, or automatic promotion. See PROTOCOL.md for objective and caveats.


## Matched-count late-regression audit

Read-only 4k checkpoints evaluated with the same 100k sample count and RNG as final 6k; no additional training.

| Run | 4k modes | 4k HQ % | 6k modes | 6k HQ % |
|---|---:|---:|---:|---:|
| 00_gan | 94 | 76.78 | 80 | 59.31 |
| 01_ae_gan | 99 | 92.67 | 84 | 73.50 |
| 02_categorical_no_kl_gan | 91 | 71.81 | 95 | 76.01 |
| cat_sharp_obs003_gan | 76 | 48.87 | 90 | 72.92 |

AE-GAN was substantially stronger at 4k; endpoint ranking is not a stability or general superiority claim.
