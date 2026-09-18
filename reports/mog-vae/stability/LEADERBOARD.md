# Particle VAE lower-LR leaderboard

Five prespecified configurations; one shared seed; final online weights at 6,000 updates.
Generation uses 100k decoder means G(z). Rank: coverage, then HQ, then SW1. Width should approach 1.
This rank is generation-focused; reconstruction and posterior usefulness have separate tradeoffs.

| Rank | Run | Modes /100 | HQ % ↑ | Width /real ≈1 | Balance KL ↓ | SW1 ↓ | Sampled recon MSE ↓ | MAP MSE ↓ | Train s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_ae_gan | 97 | 89.71 | 0.4673 | 0.3188 | 0.5438 | 0.002341 | 0.002341 | 37.96 |
| 2 | 02_categorical_no_kl_gan | 96 | 92.92 | 0.5349 | 0.1975 | 0.3831 | 0.00722 | 0.00707 | 39.86 |
| 3 | 00_gan | 96 | 75.49 | 0.6003 | 0.1791 | 0.1685 | — | — | 28.7 |
| 4 | hard_constant_kl_gan | 88 | 85.21 | 0.5157 | 0.3994 | 0.5459 | 0.003264 | 0.003236 | 40.26 |
| 5 | cat_sharp_obs003_gan | 86 | 82.5 | 0.4007 | 0.3235 | 0.3213 | 0.003884 | 0.003613 | 40.31 |

## Posterior and variation

Eight draws/input on 8,192 held-out inputs. Pair RMS is Euclidean output distance; same-mode checks input-mode retention.
Conditional effective K is exp(mean categorical entropy). MI is a held-out categorical mutual-information estimate.
AE has no posterior: zero pair RMS is expected; its soft routing probabilities are only gradient diagnostics.

| Run | KL categorical | KL local | Conditional effective K | Aggregate effective K (sampled) | MI nats | Pair RMS | Same-mode % | Shuffled MSE | Local std /sigma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_ae_gan | 0 | 0 | 33.93 | 189.1 | — | 0 | 100 | 16.43 | 0 |
| 02_categorical_no_kl_gan | 4.838 | 0 | 3.171 | 307.1 | 4.576 | 0.03216 | 98.03 | 16.38 | 1 |
| hard_constant_kl_gan | 5.991 | 0 | 1 | 194 | 5.268 | 0.01066 | 100 | 16.5 | 1 |
| cat_sharp_obs003_gan | 4.776 | 0 | 3.371 | 335.3 | 4.602 | 0.02946 | 100 | 16.51 | 1 |

## Likelihood check

These samples include the decoder likelihood noise: G(z)+tau*noise. They are distinct from decoder means above.
No-KL control has an evaluable bound but does not optimize it. ELBO column is negative ELBO in nats, lower is better.

| Run | Tau | Negative ELBO ↓ | Predictive modes | Predictive HQ % ↑ |
|---|---:|---:|---:|---:|
| 02_categorical_no_kl_gan | 0.03 | 7.685 | 96 | 80.7 |
| hard_constant_kl_gan | 0.03 | 4.443 | 99 | 75.76 |
| cat_sharp_obs003_gan | 0.03 | 3.916 | 94 | 73.99 |

## Learning curves and cost

Every evaluation uses 100k prior samples. SW1 always uses 8192.

| Run | 2k modes / HQ% / SW1 | 4k | 6k | Total process s | Peak GiB |
|---|---|---|---|---:|---:|
| 01_ae_gan | 82 / 62.80 / 0.6466 | 96 / 87.49 / 0.5789 | 97 / 89.71 / 0.5438 | 39.74 | 0.1012 |
| 02_categorical_no_kl_gan | 87 / 88.12 / 0.4461 | 60 / 49.38 / 0.4279 | 96 / 92.92 / 0.3831 | 41.43 | 0.114 |
| 00_gan | 90 / 80.98 / 0.2195 | 97 / 73.08 / 0.1702 | 96 / 75.49 / 0.1685 | 30.41 | 0.101 |
| hard_constant_kl_gan | 89 / 84.46 / 0.6421 | 96 / 87.13 / 0.5915 | 88 / 85.21 / 0.5459 | 41.82 | 0.114 |
| cat_sharp_obs003_gan | 90 / 79.30 / 0.3676 | 95 / 81.23 / 0.3536 | 86 / 82.50 / 0.3213 | 41.85 | 0.114 |

Total training: 3.12 GPU minutes; total child-process time: 3.25 minutes (includes evaluation/I/O).

All 5 completion certificates, configurations, matched initializations/data/prior RNGs, fixed sigma, 15 checkpoint hashes, and per-input reconstruction metrics verified.
No seed-only runs, statistical superiority claim, or automatic promotion. See PROTOCOL.md for objective and caveats.


## Comparison with previous full learning rates

Same seed/initialization and final 6k update budget. New learning rates are half; the new round has 100k at every evaluation.
Old no-KL likelihood tau was .1; new .03 does not affect its training or decoder-mean metrics.

| Run | Old modes / HQ % / MSE | New modes / HQ % / MSE |
|---|---|---|
| 01_ae_gan | 84 / 73.50 / 0.004629 | 97 / 89.71 / 0.002341 |
| 02_categorical_no_kl_gan | 95 / 76.01 / 0.004246 | 96 / 92.92 / 0.00722 |
| 00_gan | 80 / 59.31 / — | 96 / 75.49 / — |
| cat_sharp_obs003_gan | 90 / 72.92 / 0.005626 | 86 / 82.50 / 0.003884 |

Hard posterior has KL=log(400) constant, omitted only from optimization and retained in ELBO diagnostics.
Its categorical effective K is one; only within-particle noise is stochastic. Encoder routing uses a biased straight-through surrogate.
