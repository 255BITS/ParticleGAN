# MisGAN toy: protocol and metrics

Reproduces full MisGAN (Li, Jiang & Marlin, ICLR 2019, "MisGAN: Learning from
Incomplete Data with GANs"): a data generator, a mask generator and an imputer,
trained jointly with this repo's recipe. Results: [FINDINGS.md](FINDINGS.md).

```bash
experiments/misgan_pipeline.sh              # grid, ppost + cost, leaderboards (GPUS=0,1 WORKERS=6)
tail -f results/misgan/PIPELINE.log         # the whole study
tail -f runs/misgan/mcar_p50__misgan.log    # one run: one line per eval
python experiments/misgan_ppost.py          # ppost + imputation cost from checkpoints
python experiments/analyze_misgan.py        # re-print tables, refresh FINDINGS.md
```

## Data

- The repo's 100-Gaussian grid in 2D (centers {-4.5..4.5}^2, sigma 0.03,
  `lib.toy_models.sample_100gaussians`), lifted to D = 8 by a fixed random
  orthonormal A (8x2, QR of a seeded Gaussian matrix), plus isotropic noise
  eta = 0.01 sigma in 8D so the data is not exactly rank 2.
- Standardized per coordinate with the mean and std of the *observed*
  training entries. The fill value is tau = 0 (the observed mean).
- A fixed pool of 20,000 incomplete training rows, whose masks are fixed. The
  oracle sees the same rows complete. The test set has 10,000 clean rows with
  their own masks from the same mechanism. A second clean sample of 10,000
  rows gives the sliced-W1 floor.
- Samples go back to 2D with A^T for the grid metrics.

## Missingness (independent of x, as MisGAN assumes)

| mechanism | masks |
| --- | --- |
| `mcar_p20`, `mcar_p50`, `mcar_p80` | each coordinate missing independently with probability p |
| `block` | 4 sensor-group patterns. Observed sets {0-4} (p = 0.4), {3-7} (0.3), {0,1,2,5,6,7} (0.2) and {7} (0.1). The first three jointly observe every coordinate pair, so the joint is identifiable. The last one leaves 10% of rows with a single linear view of x2. |

Two generic observed coordinates pin x2 (rank 2), so imputation is ambiguous
only for rows with at most one observed coordinate. Under `mcar_p80` that is
half the rows, under `mcar_p50` 3.5%, under `mcar_p20` about 0.1%, and under
`block` 10%.

## Model (experiments/misgan_toy.py)

Networks are the toy MLPs from `lib/toy_models.py` (3x128 LeakyReLU). D_x and
D_i use Fourier input features; D_m uses none.

| pair | generator | critic sees | prior |
| --- | --- | --- | --- |
| data | G_x(z_x) -> x in R^8 | f(x, m) = x*m + tau(1-m) | recipe particles, z_dim 2 |
| mask | G_m(z_m) -> sigmoid(logits / 0.66) | masks | recipe particles, z_dim 8 |
| imputer | G_i([x*m, m, z_i]) -> x_hat = m*x + (1-m)*G_i | x_hat vs complete G_x(z) | recipe particles, z_dim 8 |

- **Imputer noise.** The paper's omega is replaced by a particle draw z_i, fed
  as an extra input next to (x*m, m). This is cleaner than mapping z_i to an
  8-dim omega and filling the missing coordinates with it, because then the
  prior's own dimension sets how much randomness the imputer has. The risk is
  that the imputer ignores z_i and imputes deterministically. The `istd` and
  `itv` metrics below catch that.
- **Loss.** Every critic uses the recipe's RpGAN loss (`recipe.make_loss()`).
  The per-generator objectives follow the official code: G_m minimizes
  L_m + alpha L_x, G_x minimizes L_x + beta L_i, and G_i minimizes L_i, with
  alpha = 0.2 and beta = 0.1. They come from one scalar,
  `L_m + alpha*(L_x + beta*L_i)`. Each generator appears only in its own terms,
  and Adam ignores the constant factor this puts on G_x and G_i.
- **Recipe components.** Each pair gets its own prior, optimizers and critic
  penalty from `get_recipe`, `recipe.make_prior`, `recipe.make_optimizers` and
  `recipe.make_critic_penalty`. Each pair also gets the LR schedule
  (`scale_learning_rates`), annealed critic input noise and EMA weights. G_x
  gets the recipe's generator output noise, in the D_x branch only.
- **Loop.** The loop is the plain one: critic losses, then zero_grad, backward
  and step for the three critics, then the same for the three generators.
  Budget: the recipe default of 7,000 updates at batch 2,048.

## Arms

| arm | where | D_x real vs fake | imputer reference |
| --- | --- | --- | --- |
| `oracle` | all | complete x vs G_x | true complete rows |
| `zerofill` | all | f_0(x, m) vs raw G_x (plain GAN on filled data) | G_x |
| `misgan` | all | f(x, m) vs f(G_x, G_m) | G_x |
| `misgan_realmask` | all | fakes masked with masks resampled from the training pool | G_x |
| `misgan_paired` | all | each fake masked with its RpGAN-paired real row's mask | G_x |
| `misgan_gauss` | all | as `misgan`, with frozen-Gaussian priors for all three generators (`make_prior(learnable=False)`) | G_x |
| `misgan_hard` | mcar_p50 | as `misgan`, with G_m masks binarized by a straight-through estimator (tests the soft-mask critic shortcut) | G_x |
| `misgan_detach` | all | as `misgan`, but G_m's masks are detached inside L_x, so G_m trains on L_m alone | G_x |
| `misgan_long` | mcar_p50 | `misgan` with 3x the budget (21,000 updates). The G/D LR horizon cap stays at 1,600, while the prior LR and the noise schedules stretch with the budget | G_x |

G_m and D_m train in every arm. G_m gets L_x gradient only where its masks
enter D_x (`misgan`, `misgan_gauss`, `misgan_hard`). The seed is fixed, and
arms differ only in substance.

### Round 2: imputing with G_x itself

These arms drop G_i/D_i. They all build on `misgan_detach` (G_m's masks are
detached in L_x) and run on every mechanism.

| arm | extra pieces | imputation |
| --- | --- | --- |
| `aegan_recon_w{0.1,1}` | Encoder E([x*m, m, z2]) -> z_x, where z2 comes from its own recipe particle prior (z_dim 8). E is trained with the masked reconstruction w*\|\|m (G_x(E(.)) - x)\|\|^2, which also trains G_x. A latent critic D_z (its own recipe-built pair) scores E's outputs against draws from G_x's prior, so E stays inside the prior's support. | m*x + (1-m)*G_x(E(x*m, m, z2)), 16 draws of z2 |
| `aegan_ce` | E([x*m, m]) -> logits over G_x's 20,000-particle table. It is trained with cross-entropy against the stop-grad particle-posterior weights from the current G_x: softmax(-\|\|m (G_x(z_k) - x)\|\|^2 / 2 sigma^2). sigma is refit every 500 updates by the held-out-coordinate rule below. This is a single-pass, amortized ppost. | sample k from E's categorical, fill from G_x(z_k) |

Round-2 generator scalar: `L_m + alpha*(L_x + beta*L_i + L_z + w*rec + CE)`.
Each term reaches only its own generators.

**`ppost` (no training), `experiments/misgan_ppost.py`.**
- Draw M latents from a trained run's EMA prior and compute x_k = G_x(z_k).
- Each test row resamples k with w_k ~ exp(-||m (x_k - x_obs)||^2 / 2 sigma^2)
  and fills its missing coordinates from x_k.
- sigma maximizes the log-likelihood of one held-out observed coordinate per
  training row, over 4,096 training rows and a log grid from 0.005 to 3. No
  test data or ground truth is used.
- The grid is M in {256, 4096}, each without and with refinement (`r`).
  Refinement takes 20 Adam steps on z, from the recipe's
  `make_generator_optimizer` with lr 0.01, fitting the observed coordinates
  from the resampled particle.
- Sources: `oracle`, `misgan`, `misgan_realmask`, `misgan_detach`,
  `misgan_gauss`, `aegan_recon_w*` and `aegan_ce`.
- The main leaderboard shows `ppost:<run>` at M = 4096 without refinement
  (fixed before the runs). The per-mechanism ppost table shows all four
  configurations.

**Cost.** `ms_1k` is the wall-clock time to impute 1,000 test rows with 16 draws
each, on one otherwise idle GPU. It is measured after the grid, from the EMA
checkpoints (`ckpt.pt`), and excludes ppost's one-time sigma fit.
`tests/test_misgan_toy.py` checks that ppost with exact mixture samples in
place of G_x matches the Bayes-optimal imputer.

## Metrics

The trainer evaluates the EMA weights every 500 updates and writes one line
per eval. The final values go to `results/misgan/runs/<run>/summary.json`.

| key | meaning |
| --- | --- |
| `modes`, `hq` | G_x: modes with at least 10 samples within 3 sigma, out of 100, and % of samples within 3 sigma (10k samples, 2D via A^T) |
| `swd` | G_x: sliced W1 (256 projections) to the clean test set, in 8D standardized units |
| `off` | G_x: mean distance from the data plane (raw units; clean data about 0.0007) |
| `m_mae` | G_m: mean abs error of the per-coordinate missing rate (masks thresholded at 0.5) |
| `m_tv`, `m_tvk` | G_m: TV to the exact mask distribution over all 256 patterns (block), and TV of the observed-count histogram (MCAR) |
| `m_soft` | G_m: mean min(m, 1-m) of the raw sigmoid output (0 = binary) |
| `acc` | imputation mode accuracy: impute each test row with its real mask, map to 2D, check nearest mode = true mode (mean over K = 16 draws) |
| `acc_lo` | `acc` on the ambiguous rows (at most 1 observed coordinate) |
| `itv` | per-row TV between the modes the K = 16 imputations land in and the exact Bayes posterior over modes for that row, averaged over rows |
| `istd` | per-row std over the K draws on missing coordinates (0 = deterministic imputer) |
| `rmse` | RMSE on missing coordinates (standardized units, over draws) |
| `imodes`, `ihq` | modes and 3-sigma % of the imputed test set (first draw) |
| `ms_1k` | wall-clock ms to impute 1,000 test rows with 16 draws (post hoc, idle GPU) |

The baselines use no training and run on the same test rows:

- **mean**: fill with 0.
- **kNN**: k = 5, nan-Euclidean distance on the coordinates both rows observe,
  against the incomplete training pool (sklearn `KNNImputer` semantics).
- **Bayes-optimal**: the generative model is known, so each missing block is
  sampled from the exact conditional. That is a posterior over the 100 modes
  (a linear-Gaussian likelihood of the observed coordinates), then x2 from
  that mode's Gaussian posterior, then x_M = A_M x2 + noise. Its sampled `acc`
  is the ceiling for a stochastic imputer. The `gap` column is measured from
  it. Its MAP accuracy, the deterministic ceiling, is printed under each
  table. Its `itv` is the finite-K floor, about 0 when the posterior is one
  mode.

The leaderboard ranks by `acc`, then `hq`, then `swd`.
