# MisGAN toy: findings

Protocol: [README.md](README.md). There are 42 trained runs plus post-hoc ppost
on 32 of their G_x, all on one seed, on the branch `misgan-toy`. Every arm
differs in substance. Runs are deterministic: `mcar_p50__misgan` gave identical
numbers in the pilot and in all three grid passes. Treat differences of about
0.01 in `acc` as noise. At `mcar_p20` only 16 test rows are ambiguous, so
`acc_lo` there comes in steps of 1/16 and is not informative.

## Round 2: imputing with G_x itself (no G_i)

### Verdicts on the stated predictions

- **(a) aegan_recon's acc_lo >= the Bayes-sampled acc_lo: refuted.** Both
  weights score `acc_lo` 0.14 at `mcar_p50` (Bayes 0.555), 0.11 to 0.12 on
  `block` (Bayes 0.777) and 0.085 at `mcar_p80` (Bayes 0.408). That is no
  better than the round-1 G_i. At `mcar_p20` it scores 0.69 to 0.94 against
  0.31, but on 16 rows. The imputations are only roughly valid. On rows with
  one observed coordinate, the RMS residual of G_x(E(.)) on that coordinate is
  0.038 at `mcar_p50` and 0.018 on `block` (standardized units). ppost's
  residual is 0.009 and 0.008, and the within-mode sigma is about 0.02. A
  loose fit that is not weighted by the posterior lands on neighbouring modes
  whose projections also match. The Sum p^2 bound only holds if the picks
  follow the posterior (or the MAP), and E does neither.
- **(b) aegan_recon's itv/istd are worse than ppost's on the same G_x:
  confirmed** wherever G_x is good. At `mcar_p50`, itv is 0.089 to 0.093
  against 0.019 to 0.023, and istd 0.008 to 0.011 against 0.047 to 0.051 (the
  Bayes istd is 0.026). On `block`, itv is 0.084 against 0.011 to 0.013. E
  ignores z2, the same noise-channel hijack as G_i. At `mcar_p80` the two are
  equal (about 0.5) because G_x itself is bad.
- **(c) aegan_recon's G_x hq at p80 beats the earlier 4-17%: refuted.** It
  reaches 5.3% with 21 modes (w = 1) and 2.8% with 6 modes (w = 0.1). The
  reconstruction signal does not rescue G_x at p80. At `mcar_p50` it hurts:
  51 to 53% against 78.3% for its base, `misgan_detach`. On `block` and at
  `mcar_p20` it has no effect (99.7 to 99.8%).
- **(d) ppost closes most of the ambiguous-row gap wherever G_x is good:
  confirmed.** At M = 4096 without refinement, the share of the `acc_lo` gap
  between the run's own G_i and Bayes that ppost closes is:
  - `mcar_p50`: 1.03 on the oracle G_x (`acc_lo` 0.568 vs Bayes 0.555, itv
    0.009 vs 0.006), 0.83 on `misgan_realmask`, 0.73 on `misgan_detach` and
    0.48 on `misgan` (hq 43%).
  - `block`: 0.93 to 1.00 on every particle G_x.
  - `mcar_p80`: 0.99 with the oracle G_x, but at most 0.17 with G_x trained on
    incomplete data, because those G_x are broken.

  Overall accuracy follows. ppost on a MisGAN G_x reaches 0.969 to 0.977 at
  `mcar_p50`, against 0.89 to 0.93 for the trained imputers and 0.985 for
  Bayes. On `block` it reaches 0.973 to 0.976, against 0.91 and 0.978. How
  well ppost does tracks G_x quality (hq) almost exactly.
- **(e) aegan_ce is about equal to ppost at single-pass cost: refuted on both
  counts.**
  - Accuracy: 0.598 vs 0.969 for ppost on the same G_x at `mcar_p50`, 0.897
    vs 0.971 on `block`, and 0.285 vs 0.348 at `mcar_p80`.
  - Cost: 25 to 36 ms per 1k rows against 0.6 to 1.3 ms for ppost at
    M = 4096. The 20,000-way softmax costs more than ppost's 4,096-column
    distance matrix.
  - Cause: E's categorical is far too flat, with a median entropy of 5.3 to
    6.4 nats (about 200 to 600 effective particles). Its fills miss the
    observed coordinates (RMS residual 0.25 at `mcar_p50` vs 0.013 for
    ppost). Likely contributors are a target that drifts as G_x and the
    particles move, 20,000 classes on a 128-wide network, and sampling the
    EMA table with an E trained on the live table.
- **(f) Particles need fewer samples M than a Gaussian prior for the same ppost
  accuracy: not supported, and confounded.**
  - At M = 256 the two are equal or the Gaussian is ahead: `mcar_p50` 0.891
    (particles) vs 0.901 (Gaussian), `block` 0.888 vs 0.900. With about 2.5
    samples per mode, M = 256 is sample-starved for every G_x.
  - Going to M = 4096 helps particle G_x more on `mcar_p50` and `block`:
    +0.08 to +0.09 against +0.02 to +0.06 for the Gaussian. At `mcar_p80` both
    gain about +0.08.
  - That difference reflects G_x quality, not sample efficiency. The Gaussian
    G_x is collapsed (14 to 25 modes, 3 to 6% within 3 sigma) at every rate.
  - A clean test needs two G_x of equal quality, which this repo's Gaussian
    control does not provide.

### Other round-2 findings

- **ppost is the best imputer in the study and the cheapest trained one.** It
  needs no imputer training, and it costs 0.6 to 1.3 ms per 1k rows (16
  draws). The trained G_i costs 0.4 to 1.0 ms, kNN 13 to 53 ms, and the
  unoptimized per-pattern Bayes loop 2 to 84 ms. The kernel width sigma chosen
  by held-out coordinates is 0.010 on good G_x and grows to 0.02 to 0.11 on
  degraded ones (p80, Gaussian). That is a free, ground-truth-free signal of G_x quality.
- **Refinement is not worth it.** Twenty latent steps help only at M = 256
  (`mcar_p50` misgan 0.891 -> 0.943), and never reach unrefined M = 4096. At
  M = 4096 they change `acc` by at most 0.002 where G_x is good, and by up to
  +0.07 on the broken p80 G_x. They cost 11 to 16 ms per 1k rows instead of
  about 1.
- **Imputing through G_x makes G_x the bottleneck.** In round 1 the oracle G_i
  was about as bad as the MisGAN G_i, which pointed at the imputer objective.
  With ppost the oracle G_x is the best source everywhere, because the
  imputation quality is now the generator's quality.
- **Round-1 conclusions hold, with more mechanisms.** `misgan_detach` and
  `misgan_gauss` now run everywhere:
  - `misgan_detach` equals `misgan` on `block` and at `mcar_p20`, beats it at
    `mcar_p50`, and is worse at `mcar_p80`.
  - The Gaussian prior collapses G_x on every mechanism (14 to 25 modes, or 1
    at p80).

### Recommended next experiments

1. **Make ppost the imputer, and put the effort into G_x at high missingness.**
   At p80 the problem is G_x (5 to 17% within 3 sigma). Try `misgan_detach` or
   `misgan_realmask` with a longer budget and a few complete anchor rows (32
   and 128).
2. **Choose checkpoints and runs by ppost's held-out-coordinate likelihood.**
   Check whether that label-free score ranks G_x the same way as `acc` and `hq`
   across the 32 sources already here, at no training cost.
3. **Amortize ppost properly, if amortizing at all.** The target could be
   top-k particles rather than the full 20,000-way softmax, E could be trained
   on a frozen, finished G_x instead of a moving target, and the live table
   could be used consistently. ppost at 1 ms per 1k rows leaves little to gain,
   so try this only if M must grow a lot.
4. **For aegan_recon, weight the fit by the posterior.** For example, add a
   ppost-sampled particle as a target for E on ambiguous rows, or an entropy
   or diversity term on z2. Otherwise drop it: it neither imputes nor
   generates better than ppost on `misgan_detach`.
5. **A fair particles-vs-Gaussian sample-efficiency test.** Use a Gaussian
   G_x that covers every mode (for example an overcomplete z or a longer
   budget), then compare ppost accuracy against M at equal G_x hq.

## Round 1 summary (separate imputer G_i)

1. **MisGAN gets close to the Bayes-optimal imputer only where imputation is
   nearly deterministic.** Its mode-accuracy gap is 0.009 at `mcar_p20`,
   0.068 on `block`, 0.094 at `mcar_p50` and 0.274 at `mcar_p80`. Almost all
   of the gap comes from the ambiguous rows, those with at most one observed
   coordinate. At `mcar_p50`, `mcar_p80` and on `block` (at `mcar_p20` fewer
   than 10 test rows are ambiguous), every GAN imputer scores `acc_lo` 0.03 to
   0.15 there, against
   0.41 to 0.78 for the Bayes-optimal imputer and 0.22 to 0.49 for kNN. On
   rows where two or more coordinates are observed, the imputers are close to
   perfect.

2. **The gap comes from the imputer objective, not from learning with missing
   data.** The oracle imputer, trained against true complete rows, has the
   same gap: 0.921 vs 0.985 at `mcar_p50`, 0.911 vs 0.978 on `block`, and
   0.461 vs 0.701 at `mcar_p80`. The best MisGAN runs match it or beat it
   within noise. D_i only compares the marginal of the imputations with
   complete data, and a few ambiguous rows barely move that marginal.

3. **The imputer ignores its particle input when most rows are determined.**
   At `mcar_p20`, `mcar_p50` and `block`, the per-row std over 16 draws
   (`istd`) is 0.001 to 0.017. The Bayes-optimal imputer gets 0.026 at
   `mcar_p50` and 0.062 on `block`. The imputer is effectively deterministic
   on exactly the rows where it should sample. `itv` is therefore about
   1 - `acc`, compared with a Bayes floor of 0.006. At `mcar_p80` the imputer
   does use its noise (`istd` 0.30 to 0.38, the Bayes value is 0.33), but the
   draws land in the wrong modes: `itv` is 0.48 to 0.53 against 0.18.

4. **G_m's gradient from L_x is what hurts G_x.** The mask generator itself is
   fine: at `mcar_p50`, `m_mae` is 0.018 and `m_soft` is about 0. At
   `mcar_p50`, the arms where G_m gets no L_x gradient all do much better on
   generation: `misgan_detach` reaches 3-sigma 78.3%, `misgan_realmask` 85.6%
   and `misgan_paired` 91.5%, against 43.5% for `misgan`. The distance off the
   data plane falls from 0.138 to 0.03 to 0.05. `misgan_detach` also has the
   best imputation accuracy in the 7k-step runs (0.925). With the
   L_m + alpha L_x coupling, G_m and G_x cooperate: G_m learns to hide the
   coordinates G_x gets wrong, and its missing-rate error triples
   (0.006 -> 0.018). At `mcar_p20` and on `block` the arms are within noise of
   each other.

5. **Paired masking gives no clear gain over resampled real masks.** Paired vs
   realmask: 3-sigma 91.5 vs 85.6% at `mcar_p50` but 13.6 vs 16.5% at
   `mcar_p80`, and imputation accuracy within 0.012 everywhere. Pairing masks
   in the RpGAN pairing is not worth keeping as a separate variant.

6. **Particles matter for G_x, not for imputation accuracy.** With
   frozen-Gaussian priors on all three generators (`misgan_gauss`,
   `mcar_p50`), G_x collapses to 14 modes and 4.9% within 3 sigma. With
   particles it covers 100 modes at 43.5%. Imputation accuracy barely changes
   (0.888 vs 0.891) because the determined rows dominate it. The Gaussian arm
   also cannot give a diverse imputer (`istd` 0.001).

7. **Hard straight-through masks break training.** There is no soft-mask
   shortcut to remove: learned-mask G_m outputs are already binary
   (`m_soft` <= 0.001, since the sigmoid saturates). With straight-through
   masks the L_x gradient reaches individual mask bits. G_m collapses by step
   1,500 (`m_mae` 0.171), hides coordinates, and G_x drifts off the plane
   (`off` about 1.0) with 5 modes. Imputation accuracy peaks at 0.714 at step
   500 and ends at 0.440.

8. **Where it breaks.** At `mcar_p50` learned-mask MisGAN G_x is already
   degraded, and it is still improving at 7k steps: 3x the budget gives 3-sigma
   51% and `acc` 0.930. At `mcar_p80` every incomplete-data G_x fails:
   3-sigma 4 to 17% and 13 to 80 modes. Detaching G_m does not help there
   (`misgan_detach` 4.4%). The oracle G_x (99.8%) shows this is a
   missing-data failure, not a capacity limit. At p80, half the rows have at
   most one coordinate observed, and pairs of coordinates are observed jointly
   in only 4% of rows. `zerofill` fails at every rate, as expected: it learns
   the holes, with `off` about 1.

## Round 1 notes

- **Budget.** The default budget converges on the oracle arm: 100 modes and
  99.8% within 3 sigma by step 2,000 at `mcar_p50`, flat to step 7,000, and
  99.6 to 99.8% at the end for every mechanism.
  No recipe field is overridden. Pilots on the oracle arm tested three
  settings: no network LR horizon cap, no generator output noise, and a critic
  without Fourier features. None beat the default at 7k steps. Without output
  noise the sliced W1 drops (0.028 vs 0.040) at equal 3-sigma. The uncapped
  run was unstable.
- **tau.** Not varied. The fill is 0, the observed mean. With continuous data
  an observed value is never exactly 0, so D_x can always tell which
  coordinates are missing. A different tau would only move the fill point.
- **Sliced W1 and 3-sigma disagree.** The oracle G_x has the best 3-sigma and
  the worst sliced W1 (0.040 vs 0.023 for `misgan_detach`), which points to
  unequal mode weights or overly sharp modes, not missing modes. The ranking
  therefore puts sliced W1 last.
- **kNN is a strong baseline on `block`.** kNN (0.949) beats every GAN imputer
  (0.91) there. It is also better on the ambiguous rows under every mechanism.

## Round 1 recommended next experiments

1. **Detach G_m from L_x by default** (or put a much smaller weight on it), and
   rerun `mcar_p50` and `block` at 3x the budget. This is the largest
   single-switch gain found here.
2. **Mask-conditioned imputer critic.** Make D_i(x_hat, m) compare against
   G_x samples masked with the same m, so the critic scores conditionals per
   pattern instead of the pooled marginal. This targets `acc_lo` and `itv`
   directly.
3. **An imputer that cannot ignore its noise.** Use the anchor-repulsion or
   diversity term from the conditional particle-z study on G_i's particle
   input, and measure `istd` and `itv` against the Bayes floor at `mcar_p50`
   and `block`.
4. **Oversample ambiguous rows** in the D_i batch (rows with at most one
   observed coordinate), testing whether the imputer gap is only a matter of
   signal share.
5. **`mcar_p80` with G_m detached, a longer budget and nested anchor subsets**
   of complete rows (32 and 128), to find out whether a few full references
   rescue G_x when pairs are rarely observed together.

<!-- AUTO:start -->
## Leaderboards

Final EMA weights at the last update. Ranked by imputation mode accuracy, then generation 3-sigma %, then sliced W1. Italic rows are untrained references on the same test rows.

### mcar_p20

| arm | acc | gap | acc_lo | itv | istd | rmse | ms_1k | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppost:oracle | 1.000 | 0.000 | 0.062 | 0.000 | 0.002 | 0.021 | 0.8 | 99.5 | 100 | 99.8 | 0.047 | 0.010 | - | - | - | - |
| ppost:aegan_recon_w1 | 1.000 | 0.000 | 0.438 | 0.000 | 0.003 | 0.019 | 0.9 | 99.4 | 100 | 99.8 | 0.023 | 0.021 | - | - | - | - |
| ppost:aegan_recon_w0.1 | 1.000 | 0.000 | 0.062 | 0.000 | 0.003 | 0.026 | 0.9 | 99.4 | 100 | 99.7 | 0.023 | 0.028 | - | - | - | - |
| ppost:misgan_realmask | 1.000 | 0.000 | 0.125 | 0.000 | 0.003 | 0.025 | 0.9 | 99.5 | 100 | 99.7 | 0.029 | 0.020 | - | - | - | - |
| ppost:misgan_detach | 1.000 | 0.000 | 0.375 | 0.000 | 0.003 | 0.022 | 0.9 | 99.4 | 100 | 99.6 | 0.029 | 0.022 | - | - | - | - |
| ppost:aegan_ce | 1.000 | 0.000 | 0.062 | 0.000 | 0.005 | 0.030 | 1.3 | 99.2 | 100 | 96.9 | 0.022 | 0.037 | - | - | - | - |
| ppost:misgan | 1.000 | 0.000 | 0.062 | 0.000 | 0.004 | 0.029 | 0.9 | 99.2 | 100 | 95.9 | 0.021 | 0.032 | - | - | - | - |
| ppost:misgan_gauss | 1.000 | 0.000 | 0.062 | 0.000 | 0.017 | 0.048 | 0.9 | 87.9 | 20 | 5.8 | 0.034 | 0.070 | - | - | - | - |
| oracle | 0.996 | 0.004 | 1.000 | 0.004 | 0.000 | 0.066 | 0.5 | 92.2 | 100 | 99.8 | 0.047 | 0.010 | 0.020 | 0.053 | 0.001 | 0.996 |
| misgan_paired | 0.995 | 0.005 | 1.000 | 0.005 | 0.007 | 0.076 | 0.5 | 87.0 | 100 | 99.8 | 0.025 | 0.018 | 0.020 | 0.053 | 0.001 | 0.995 |
| misgan_realmask | 0.995 | 0.005 | 0.938 | 0.005 | 0.010 | 0.073 | 0.4 | 86.7 | 100 | 99.7 | 0.029 | 0.020 | 0.007 | 0.019 | 0.000 | 0.995 |
| aegan_recon_w1 | 0.993 | 0.007 | 0.938 | 0.007 | 0.007 | 0.072 | 0.9 | 87.9 | 100 | 99.8 | 0.023 | 0.021 | 0.015 | 0.034 | 0.001 | 0.993 |
| aegan_recon_w0.1 | 0.993 | 0.006 | 0.688 | 0.007 | 0.009 | 0.074 | 1.1 | 87.0 | 100 | 99.7 | 0.023 | 0.028 | 0.015 | 0.034 | 0.001 | 0.993 |
| misgan_detach | 0.991 | 0.009 | 1.000 | 0.009 | 0.013 | 0.105 | 0.5 | 84.6 | 100 | 99.6 | 0.029 | 0.022 | 0.020 | 0.053 | 0.001 | 0.991 |
| misgan | 0.991 | 0.009 | 1.000 | 0.010 | 0.006 | 0.101 | 0.5 | 76.5 | 100 | 95.9 | 0.021 | 0.032 | 0.042 | 0.119 | 0.000 | 0.991 |
| misgan_gauss | 0.975 | 0.024 | 1.000 | 0.025 | 0.008 | 0.170 | 0.7 | 45.5 | 20 | 5.8 | 0.034 | 0.070 | 0.018 | 0.049 | 0.002 | 0.979 |
| aegan_ce | 0.969 | 0.031 | 0.000 | 0.031 | 0.073 | 0.119 | 35.9 | 91.4 | 100 | 96.9 | 0.022 | 0.037 | 0.034 | 0.094 | 0.000 | 0.969 |
| zerofill | 0.498 | 0.501 | 1.000 | 0.502 | 0.010 | 0.922 | 1.0 | 25.3 | 5 | 7.6 | 0.202 | 0.904 | 0.020 | 0.053 | 0.001 | 0.586 |
| *bayes* | 1.000 | 0.000 | 0.312 | 0.000 | 0.001 | 0.010 | 83.5 | 98.9 | - | - | - | - | - | - | - | - |
| *knn* | 0.986 | 0.014 | 0.000 | 0.014 | 0.000 | 0.098 | 52.8 | 93.2 | - | - | - | - | - | - | - | - |
| *mean* | 0.490 | 0.510 | 1.000 | 0.510 | 0.000 | 0.994 | 0.0 | 25.7 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | - | 100 | 98.9 | 0.017 | 0.001 | - | - | - | - |

Bayes MAP accuracy 1.000. `ppost:<run>` rows impute with that run's G_x by particle posterior (M4096, no refinement). ms_1k = wall-clock ms per 1k rows for 16 draws. Mask TV column = TV of the observed-count histogram vs Binomial(8, 1-p).

Particle-posterior imputation by source G_x. Cells: acc / acc_lo / itv; `r` = with latent refinement; sigma and istd at M4096.

| source G_x | modes | hq | sigma | M256 | M256r | M4096 | M4096r | istd | ms_1k M4096 / r | own imputer |
|---|---|---|---|---|---|---|---|---|---|---|
| aegan_ce | 100 | 96.9 | 0.015 | 0.975 / 0.000 / 0.025 | 0.998 / 0.000 / 0.002 | 1.000 / 0.062 / 0.000 | 1.000 / 0.000 / 0.000 | 0.005 | 1.3 / 15.8 | 0.969 / 0.000 / 0.031 |
| aegan_recon_w0.1 | 100 | 99.7 | 0.012 | 0.980 / 0.125 / 0.020 | 0.997 / 0.000 / 0.003 | 1.000 / 0.062 / 0.000 | 1.000 / 0.062 / 0.000 | 0.003 | 0.9 / 16.0 | 0.993 / 0.688 / 0.007 |
| aegan_recon_w1 | 100 | 99.8 | 0.010 | 0.982 / 0.062 / 0.018 | 0.997 / 0.125 / 0.003 | 1.000 / 0.438 / 0.000 | 1.000 / 0.250 / 0.000 | 0.003 | 0.9 / 14.4 | 0.993 / 0.938 / 0.007 |
| misgan | 100 | 95.9 | 0.012 | 0.976 / 0.062 / 0.024 | 0.999 / 0.062 / 0.001 | 1.000 / 0.062 / 0.000 | 1.000 / 0.188 / 0.000 | 0.004 | 0.9 / 15.2 | 0.991 / 1.000 / 0.010 |
| misgan_detach | 100 | 99.6 | 0.010 | 0.989 / 0.375 / 0.011 | 0.999 / 0.250 / 0.001 | 1.000 / 0.375 / 0.000 | 1.000 / 0.375 / 0.000 | 0.003 | 0.9 / 15.4 | 0.991 / 1.000 / 0.009 |
| misgan_gauss | 20 | 5.8 | 0.029 | 0.993 / 0.312 / 0.007 | 0.999 / 0.250 / 0.001 | 1.000 / 0.062 / 0.000 | 1.000 / 0.062 / 0.000 | 0.017 | 0.9 / 15.7 | 0.975 / 1.000 / 0.025 |
| misgan_realmask | 100 | 99.7 | 0.010 | 0.985 / 0.188 / 0.015 | 0.999 / 0.250 / 0.001 | 1.000 / 0.125 / 0.000 | 1.000 / 0.000 / 0.000 | 0.003 | 0.9 / 15.8 | 0.995 / 0.938 / 0.005 |
| oracle | 100 | 99.8 | 0.010 | 0.988 / 0.062 / 0.012 | 0.999 / 0.125 / 0.000 | 1.000 / 0.062 / 0.000 | 1.000 / 0.188 / 0.000 | 0.002 | 0.8 / 15.4 | 0.996 / 1.000 / 0.004 |
| *bayes* | - | - | - | 1.000 / 0.312 / 0.000 | 1.000 / 0.312 / 0.000 | 1.000 / 0.312 / 0.000 | 1.000 / 0.312 / 0.000 | 0.001 | - | - |

### mcar_p50

| arm | acc | gap | acc_lo | itv | istd | rmse | ms_1k | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppost:oracle | 0.983 | 0.002 | 0.568 | 0.009 | 0.029 | 0.205 | 0.8 | 99.9 | 100 | 99.8 | 0.040 | 0.010 | - | - | - | - |
| ppost:misgan_realmask | 0.977 | 0.008 | 0.481 | 0.015 | 0.042 | 0.236 | 0.7 | 97.3 | 100 | 85.6 | 0.024 | 0.034 | - | - | - | - |
| ppost:misgan_detach | 0.976 | 0.009 | 0.446 | 0.016 | 0.044 | 0.242 | 0.6 | 95.9 | 100 | 78.3 | 0.023 | 0.049 | - | - | - | - |
| ppost:aegan_recon_w1 | 0.973 | 0.012 | 0.377 | 0.019 | 0.047 | 0.246 | 0.7 | 92.4 | 99 | 50.8 | 0.021 | 0.040 | - | - | - | - |
| ppost:aegan_ce | 0.969 | 0.015 | 0.342 | 0.023 | 0.052 | 0.260 | 0.7 | 89.9 | 96 | 57.0 | 0.017 | 0.060 | - | - | - | - |
| ppost:aegan_recon_w0.1 | 0.969 | 0.016 | 0.359 | 0.023 | 0.051 | 0.261 | 0.9 | 88.4 | 97 | 53.0 | 0.026 | 0.057 | - | - | - | - |
| ppost:misgan | 0.969 | 0.016 | 0.328 | 0.024 | 0.053 | 0.258 | 0.6 | 87.5 | 100 | 43.5 | 0.031 | 0.138 | - | - | - | - |
| ppost:misgan_gauss | 0.956 | 0.029 | 0.169 | 0.039 | 0.067 | 0.286 | 0.7 | 66.4 | 14 | 4.9 | 0.029 | 0.066 | - | - | - | - |
| misgan_long | 0.930 | 0.055 | 0.146 | 0.069 | 0.001 | 0.274 | 0.6 | 44.8 | 100 | 51.0 | 0.031 | 0.056 | 0.009 | 0.040 | 0.002 | 0.930 |
| misgan_detach | 0.925 | 0.059 | 0.145 | 0.075 | 0.001 | 0.267 | 0.4 | 57.2 | 100 | 78.3 | 0.023 | 0.049 | 0.006 | 0.020 | 0.002 | 0.925 |
| oracle | 0.921 | 0.064 | 0.112 | 0.079 | 0.001 | 0.327 | 0.5 | 70.2 | 100 | 99.8 | 0.040 | 0.010 | 0.006 | 0.020 | 0.002 | 0.921 |
| misgan_realmask | 0.918 | 0.066 | 0.128 | 0.081 | 0.011 | 0.270 | 0.5 | 54.2 | 100 | 85.6 | 0.024 | 0.034 | 0.006 | 0.013 | 0.000 | 0.918 |
| misgan_paired | 0.913 | 0.072 | 0.127 | 0.086 | 0.017 | 0.274 | 0.5 | 53.7 | 100 | 91.5 | 0.024 | 0.033 | 0.006 | 0.020 | 0.002 | 0.913 |
| aegan_recon_w0.1 | 0.910 | 0.075 | 0.140 | 0.089 | 0.008 | 0.279 | 0.9 | 54.6 | 97 | 53.0 | 0.026 | 0.057 | 0.012 | 0.025 | 0.001 | 0.910 |
| aegan_recon_w1 | 0.907 | 0.078 | 0.143 | 0.093 | 0.011 | 0.274 | 0.8 | 54.3 | 99 | 50.8 | 0.021 | 0.040 | 0.012 | 0.025 | 0.001 | 0.907 |
| misgan | 0.891 | 0.094 | 0.121 | 0.109 | 0.002 | 0.344 | 0.4 | 39.7 | 100 | 43.5 | 0.031 | 0.138 | 0.018 | 0.034 | 0.001 | 0.891 |
| misgan_gauss | 0.888 | 0.097 | 0.145 | 0.112 | 0.001 | 0.289 | 0.9 | 19.7 | 14 | 4.9 | 0.029 | 0.066 | 0.004 | 0.012 | 0.003 | 0.888 |
| aegan_ce | 0.598 | 0.387 | 0.052 | 0.400 | 0.288 | 0.438 | 25.0 | 38.1 | 96 | 57.0 | 0.017 | 0.060 | 0.004 | 0.015 | 0.000 | 0.753 |
| misgan_hard | 0.440 | 0.545 | 0.031 | 0.560 | 0.015 | 0.688 | 0.5 | 5.1 | 5 | 2.0 | 0.148 | 1.069 | 0.171 | 0.258 | 0.000 | 0.714 |
| zerofill | 0.151 | 0.834 | 0.014 | 0.849 | 0.014 | 0.931 | 0.6 | 4.4 | 4 | 3.6 | 0.347 | 1.344 | 0.006 | 0.020 | 0.002 | 0.159 |
| *bayes* | 0.985 | 0.000 | 0.555 | 0.006 | 0.026 | 0.202 | 48.9 | 98.9 | - | - | - | - | - | - | - | - |
| *knn* | 0.563 | 0.422 | 0.382 | 0.438 | 0.000 | 0.437 | 13.0 | 18.4 | - | - | - | - | - | - | - | - |
| *mean* | 0.126 | 0.859 | 0.015 | 0.874 | 0.000 | 0.992 | 0.0 | 4.7 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | - | 100 | 98.9 | 0.017 | 0.001 | - | - | - | - |

Bayes MAP accuracy 0.988. `ppost:<run>` rows impute with that run's G_x by particle posterior (M4096, no refinement). ms_1k = wall-clock ms per 1k rows for 16 draws. Mask TV column = TV of the observed-count histogram vs Binomial(8, 1-p).

Particle-posterior imputation by source G_x. Cells: acc / acc_lo / itv; `r` = with latent refinement; sigma and istd at M4096.

| source G_x | modes | hq | sigma | M256 | M256r | M4096 | M4096r | istd | ms_1k M4096 / r | own imputer |
|---|---|---|---|---|---|---|---|---|---|---|
| aegan_ce | 96 | 57.0 | 0.019 | 0.912 / 0.178 / 0.083 | 0.950 / 0.199 / 0.045 | 0.969 / 0.342 / 0.023 | 0.969 / 0.347 / 0.023 | 0.052 | 0.7 / 11.4 | 0.598 / 0.052 / 0.400 |
| aegan_recon_w0.1 | 97 | 53.0 | 0.019 | 0.908 / 0.189 / 0.087 | 0.945 / 0.185 / 0.050 | 0.969 / 0.359 / 0.023 | 0.969 / 0.358 / 0.023 | 0.051 | 0.9 / 11.3 | 0.910 / 0.140 / 0.089 |
| aegan_recon_w1 | 99 | 50.8 | 0.012 | 0.915 / 0.174 / 0.080 | 0.930 / 0.188 / 0.064 | 0.973 / 0.377 / 0.019 | 0.973 / 0.381 / 0.020 | 0.047 | 0.7 / 11.3 | 0.907 / 0.143 / 0.093 |
| misgan | 100 | 43.5 | 0.015 | 0.891 / 0.163 / 0.104 | 0.943 / 0.174 / 0.051 | 0.969 / 0.328 / 0.024 | 0.969 / 0.344 / 0.024 | 0.053 | 0.6 / 11.5 | 0.891 / 0.121 / 0.109 |
| misgan_detach | 100 | 78.3 | 0.012 | 0.927 / 0.224 / 0.067 | 0.954 / 0.226 / 0.041 | 0.976 / 0.446 / 0.016 | 0.976 / 0.450 / 0.016 | 0.044 | 0.6 / 11.6 | 0.925 / 0.145 / 0.075 |
| misgan_gauss | 14 | 4.9 | 0.023 | 0.901 / 0.131 / 0.095 | 0.943 / 0.156 / 0.052 | 0.956 / 0.169 / 0.039 | 0.958 / 0.174 / 0.037 | 0.067 | 0.7 / 11.5 | 0.888 / 0.145 / 0.112 |
| misgan_realmask | 100 | 85.6 | 0.012 | 0.915 / 0.239 / 0.079 | 0.946 / 0.241 / 0.048 | 0.977 / 0.481 / 0.015 | 0.977 / 0.467 / 0.015 | 0.042 | 0.7 / 11.3 | 0.918 / 0.128 / 0.081 |
| oracle | 100 | 99.8 | 0.010 | 0.904 / 0.202 / 0.090 | 0.947 / 0.215 / 0.047 | 0.983 / 0.568 / 0.009 | 0.983 / 0.569 / 0.010 | 0.029 | 0.8 / 16.3 | 0.921 / 0.112 / 0.079 |
| *bayes* | - | - | - | 0.985 / 0.555 / 0.006 | 0.985 / 0.555 / 0.006 | 0.985 / 0.555 / 0.006 | 0.985 / 0.555 / 0.006 | 0.026 | - | - |

### mcar_p80

| arm | acc | gap | acc_lo | itv | istd | rmse | ms_1k | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppost:oracle | 0.693 | 0.008 | 0.406 | 0.196 | 0.341 | 0.788 | 0.9 | 99.9 | 100 | 99.8 | 0.049 | 0.010 | - | - | - | - |
| ppost:misgan_realmask | 0.532 | 0.169 | 0.145 | 0.385 | 0.442 | 0.892 | 1.1 | 35.8 | 80 | 16.5 | 0.026 | 0.083 | - | - | - | - |
| ppost:aegan_recon_w1 | 0.466 | 0.235 | 0.097 | 0.469 | 0.473 | 0.931 | 1.2 | 14.4 | 21 | 5.3 | 0.044 | 0.115 | - | - | - | - |
| ppost:misgan | 0.462 | 0.239 | 0.099 | 0.470 | 0.465 | 0.913 | 1.3 | 11.2 | 20 | 5.9 | 0.041 | 0.216 | - | - | - | - |
| oracle | 0.461 | 0.240 | 0.091 | 0.480 | 0.318 | 0.979 | 0.5 | 41.7 | 100 | 99.8 | 0.049 | 0.010 | 0.039 | 0.114 | 0.002 | 0.461 |
| aegan_recon_w1 | 0.456 | 0.245 | 0.085 | 0.499 | 0.334 | 0.981 | 0.8 | 13.0 | 21 | 5.3 | 0.044 | 0.115 | 0.032 | 0.087 | 0.000 | 0.456 |
| ppost:misgan_gauss | 0.447 | 0.254 | 0.089 | 0.487 | 0.487 | 0.925 | 1.3 | 8.3 | 1 | 2.9 | 0.040 | 0.214 | - | - | - | - |
| misgan_realmask | 0.442 | 0.259 | 0.090 | 0.506 | 0.377 | 0.958 | 0.5 | 13.5 | 80 | 16.5 | 0.026 | 0.083 | 0.018 | 0.052 | 0.001 | 0.442 |
| misgan_paired | 0.430 | 0.270 | 0.090 | 0.517 | 0.310 | 0.960 | 0.6 | 9.1 | 58 | 13.6 | 0.031 | 0.075 | 0.039 | 0.114 | 0.002 | 0.430 |
| misgan | 0.427 | 0.274 | 0.085 | 0.534 | 0.304 | 0.963 | 0.5 | 6.4 | 20 | 5.9 | 0.041 | 0.216 | 0.021 | 0.064 | 0.000 | 0.431 |
| aegan_recon_w0.1 | 0.426 | 0.275 | 0.086 | 0.529 | 0.317 | 0.968 | 0.9 | 7.3 | 6 | 2.8 | 0.052 | 0.239 | 0.032 | 0.087 | 0.000 | 0.427 |
| misgan_gauss | 0.397 | 0.304 | 0.081 | 0.566 | 0.224 | 0.975 | 0.6 | 5.0 | 1 | 2.9 | 0.040 | 0.214 | 0.019 | 0.055 | 0.002 | 0.400 |
| ppost:misgan_detach | 0.390 | 0.311 | 0.079 | 0.548 | 0.504 | 0.945 | 0.9 | 8.9 | 13 | 4.4 | 0.052 | 0.265 | - | - | - | - |
| misgan_detach | 0.380 | 0.321 | 0.080 | 0.577 | 0.340 | 0.979 | 0.5 | 5.9 | 13 | 4.4 | 0.052 | 0.265 | 0.039 | 0.114 | 0.002 | 0.400 |
| ppost:aegan_recon_w0.1 | 0.369 | 0.332 | 0.076 | 0.572 | 0.510 | 0.945 | 1.3 | 4.9 | 6 | 2.8 | 0.052 | 0.239 | - | - | - | - |
| ppost:aegan_ce | 0.348 | 0.352 | 0.074 | 0.594 | 0.504 | 0.934 | 1.0 | 5.0 | 10 | 2.9 | 0.049 | 0.306 | - | - | - | - |
| aegan_ce | 0.285 | 0.416 | 0.062 | 0.664 | 0.545 | 0.949 | 34.7 | 4.6 | 10 | 2.9 | 0.049 | 0.306 | 0.015 | 0.046 | 0.001 | 0.300 |
| zerofill | 0.033 | 0.668 | 0.017 | 0.965 | 0.007 | 1.007 | 0.5 | 3.2 | 5 | 4.6 | 0.548 | 0.831 | 0.039 | 0.114 | 0.002 | 0.034 |
| *bayes* | 0.701 | 0.000 | 0.408 | 0.179 | 0.334 | 0.774 | 30.5 | 98.8 | - | - | - | - | - | - | - | - |
| *knn* | 0.282 | 0.419 | 0.217 | 0.719 | 0.000 | 0.653 | 12.9 | 8.6 | - | - | - | - | - | - | - | - |
| *mean* | 0.032 | 0.668 | 0.017 | 0.968 | 0.000 | 0.992 | 0.0 | 3.7 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | - | 100 | 98.9 | 0.017 | 0.001 | - | - | - | - |

Bayes MAP accuracy 0.722. `ppost:<run>` rows impute with that run's G_x by particle posterior (M4096, no refinement). ms_1k = wall-clock ms per 1k rows for 16 draws. Mask TV column = TV of the observed-count histogram vs Binomial(8, 1-p).

Particle-posterior imputation by source G_x. Cells: acc / acc_lo / itv; `r` = with latent refinement; sigma and istd at M4096.

| source G_x | modes | hq | sigma | M256 | M256r | M4096 | M4096r | istd | ms_1k M4096 / r | own imputer |
|---|---|---|---|---|---|---|---|---|---|---|
| aegan_ce | 10 | 2.9 | 0.110 | 0.316 / 0.071 / 0.629 | 0.389 / 0.086 / 0.551 | 0.348 / 0.074 / 0.594 | 0.404 / 0.087 / 0.535 | 0.504 | 1.0 / 15.2 | 0.285 / 0.062 / 0.664 |
| aegan_recon_w0.1 | 6 | 2.8 | 0.110 | 0.332 / 0.072 / 0.612 | 0.419 / 0.087 / 0.517 | 0.369 / 0.076 / 0.572 | 0.438 / 0.085 / 0.497 | 0.510 | 1.3 / 16.0 | 0.426 / 0.086 / 0.529 |
| aegan_recon_w1 | 21 | 5.3 | 0.057 | 0.339 / 0.074 / 0.602 | 0.421 / 0.087 / 0.514 | 0.466 / 0.097 / 0.469 | 0.487 / 0.097 / 0.446 | 0.473 | 1.2 / 15.2 | 0.456 / 0.085 / 0.499 |
| misgan | 20 | 5.9 | 0.057 | 0.379 / 0.081 / 0.561 | 0.448 / 0.095 / 0.485 | 0.462 / 0.099 / 0.470 | 0.477 / 0.101 / 0.453 | 0.465 | 1.3 / 16.1 | 0.427 / 0.085 / 0.534 |
| misgan_detach | 13 | 4.4 | 0.088 | 0.319 / 0.071 / 0.626 | 0.402 / 0.085 / 0.536 | 0.390 / 0.079 / 0.548 | 0.425 / 0.086 / 0.512 | 0.504 | 0.9 / 15.8 | 0.380 / 0.080 / 0.577 |
| misgan_gauss | 1 | 2.9 | 0.057 | 0.372 / 0.076 / 0.568 | 0.448 / 0.093 / 0.487 | 0.447 / 0.089 / 0.487 | 0.469 / 0.093 / 0.464 | 0.487 | 1.3 / 15.6 | 0.397 / 0.081 / 0.566 |
| misgan_realmask | 80 | 16.5 | 0.023 | 0.455 / 0.106 / 0.478 | 0.495 / 0.113 / 0.436 | 0.532 / 0.145 / 0.385 | 0.534 / 0.145 / 0.383 | 0.442 | 1.1 / 15.2 | 0.442 / 0.090 / 0.506 |
| oracle | 100 | 99.8 | 0.010 | 0.520 / 0.171 / 0.395 | 0.535 / 0.172 / 0.381 | 0.693 / 0.406 / 0.196 | 0.692 / 0.404 / 0.197 | 0.341 | 0.9 / 15.6 | 0.461 / 0.091 / 0.480 |
| *bayes* | - | - | - | 0.701 / 0.408 / 0.179 | 0.701 / 0.408 / 0.179 | 0.701 / 0.408 / 0.179 | 0.701 / 0.408 / 0.179 | 0.334 | - | - |

### block

| arm | acc | gap | acc_lo | itv | istd | rmse | ms_1k | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppost:oracle | 0.978 | -0.000 | 0.778 | 0.008 | 0.060 | 0.292 | 0.9 | 100.0 | 100 | 99.6 | 0.036 | 0.012 | - | - | - | - |
| ppost:aegan_recon_w1 | 0.976 | 0.002 | 0.759 | 0.011 | 0.066 | 0.300 | 0.9 | 99.9 | 100 | 99.8 | 0.022 | 0.014 | - | - | - | - |
| ppost:misgan | 0.975 | 0.003 | 0.748 | 0.011 | 0.067 | 0.306 | 0.8 | 99.9 | 100 | 99.8 | 0.031 | 0.016 | - | - | - | - |
| ppost:misgan_realmask | 0.974 | 0.004 | 0.739 | 0.013 | 0.073 | 0.308 | 1.1 | 99.9 | 100 | 100.0 | 0.027 | 0.017 | - | - | - | - |
| ppost:aegan_recon_w0.1 | 0.974 | 0.004 | 0.738 | 0.013 | 0.070 | 0.312 | 0.7 | 99.9 | 100 | 99.8 | 0.028 | 0.018 | - | - | - | - |
| ppost:misgan_detach | 0.973 | 0.005 | 0.731 | 0.013 | 0.073 | 0.315 | 0.9 | 99.8 | 100 | 99.6 | 0.050 | 0.017 | - | - | - | - |
| ppost:aegan_ce | 0.971 | 0.006 | 0.714 | 0.014 | 0.074 | 0.322 | 0.6 | 99.9 | 100 | 99.8 | 0.045 | 0.016 | - | - | - | - |
| ppost:misgan_gauss | 0.918 | 0.060 | 0.180 | 0.075 | 0.129 | 0.425 | 0.8 | 78.2 | 25 | 6.2 | 0.026 | 0.040 | - | - | - | - |
| aegan_recon_w1 | 0.912 | 0.066 | 0.115 | 0.084 | 0.094 | 0.380 | 0.7 | 85.6 | 100 | 99.8 | 0.022 | 0.014 | 0.006 | 0.009 | 0.000 | 0.912 |
| aegan_recon_w0.1 | 0.911 | 0.067 | 0.109 | 0.084 | 0.122 | 0.437 | 0.7 | 81.5 | 100 | 99.8 | 0.028 | 0.018 | 0.006 | 0.009 | 0.000 | 0.912 |
| oracle | 0.911 | 0.067 | 0.107 | 0.083 | 0.140 | 0.446 | 0.5 | 86.8 | 100 | 99.6 | 0.036 | 0.012 | 0.007 | 0.010 | 0.000 | 0.911 |
| misgan_realmask | 0.910 | 0.068 | 0.102 | 0.084 | 0.151 | 0.466 | 0.5 | 85.2 | 100 | 100.0 | 0.027 | 0.017 | 0.006 | 0.014 | 0.000 | 0.911 |
| misgan | 0.910 | 0.068 | 0.097 | 0.084 | 0.161 | 0.497 | 0.5 | 86.2 | 100 | 99.8 | 0.031 | 0.016 | 0.004 | 0.007 | 0.000 | 0.910 |
| misgan_paired | 0.910 | 0.068 | 0.095 | 0.085 | 0.162 | 0.494 | 0.5 | 86.5 | 100 | 99.8 | 0.032 | 0.016 | 0.007 | 0.010 | 0.000 | 0.911 |
| misgan_detach | 0.910 | 0.068 | 0.100 | 0.084 | 0.152 | 0.476 | 0.5 | 86.3 | 100 | 99.6 | 0.050 | 0.017 | 0.007 | 0.010 | 0.000 | 0.910 |
| misgan_gauss | 0.909 | 0.069 | 0.112 | 0.085 | 0.174 | 0.432 | 0.5 | 30.2 | 25 | 6.2 | 0.026 | 0.040 | 0.005 | 0.007 | 0.000 | 0.910 |
| aegan_ce | 0.897 | 0.080 | 0.103 | 0.097 | 0.160 | 0.437 | 27.3 | 95.3 | 100 | 99.8 | 0.045 | 0.016 | 0.002 | 0.004 | 0.000 | 0.897 |
| zerofill | 0.152 | 0.826 | 0.008 | 0.848 | 0.007 | 0.997 | 0.4 | 1.2 | 6 | 1.5 | 0.279 | 1.429 | 0.007 | 0.010 | 0.000 | 0.288 |
| *bayes* | 0.978 | 0.000 | 0.777 | 0.006 | 0.062 | 0.292 | 2.1 | 98.9 | - | - | - | - | - | - | - | - |
| *knn* | 0.949 | 0.028 | 0.494 | 0.050 | 0.000 | 0.227 | 18.5 | 93.8 | - | - | - | - | - | - | - | - |
| *mean* | 0.140 | 0.837 | 0.006 | 0.859 | 0.000 | 0.995 | 0.0 | 1.4 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | - | 100 | 99.0 | 0.011 | 0.001 | - | - | - | - |

Bayes MAP accuracy 0.984. `ppost:<run>` rows impute with that run's G_x by particle posterior (M4096, no refinement). ms_1k = wall-clock ms per 1k rows for 16 draws. Mask TV column = TV over the 4 block patterns (256-pattern space).

Particle-posterior imputation by source G_x. Cells: acc / acc_lo / itv; `r` = with latent refinement; sigma and istd at M4096.

| source G_x | modes | hq | sigma | M256 | M256r | M4096 | M4096r | istd | ms_1k M4096 / r | own imputer |
|---|---|---|---|---|---|---|---|---|---|---|
| aegan_ce | 100 | 99.8 | 0.010 | 0.875 / 0.200 / 0.115 | 0.918 / 0.209 / 0.073 | 0.971 / 0.714 / 0.014 | 0.971 / 0.712 / 0.015 | 0.074 | 0.6 / 11.1 | 0.897 / 0.103 / 0.097 |
| aegan_recon_w0.1 | 100 | 99.8 | 0.010 | 0.878 / 0.208 / 0.113 | 0.921 / 0.209 / 0.071 | 0.974 / 0.738 / 0.013 | 0.974 / 0.743 / 0.013 | 0.070 | 0.7 / 11.1 | 0.911 / 0.109 / 0.084 |
| aegan_recon_w1 | 100 | 99.8 | 0.010 | 0.885 / 0.200 / 0.106 | 0.905 / 0.204 / 0.086 | 0.976 / 0.759 / 0.011 | 0.976 / 0.762 / 0.010 | 0.066 | 0.9 / 15.1 | 0.912 / 0.115 / 0.084 |
| misgan | 100 | 99.8 | 0.010 | 0.888 / 0.201 / 0.103 | 0.920 / 0.201 / 0.072 | 0.975 / 0.748 / 0.011 | 0.976 / 0.756 / 0.011 | 0.067 | 0.8 / 15.1 | 0.910 / 0.097 / 0.084 |
| misgan_detach | 100 | 99.6 | 0.010 | 0.888 / 0.215 / 0.103 | 0.922 / 0.225 / 0.068 | 0.973 / 0.731 / 0.013 | 0.973 / 0.733 / 0.013 | 0.073 | 0.9 / 15.4 | 0.910 / 0.100 / 0.084 |
| misgan_gauss | 25 | 6.2 | 0.023 | 0.900 / 0.118 / 0.095 | 0.917 / 0.172 / 0.076 | 0.918 / 0.180 / 0.075 | 0.919 / 0.189 / 0.074 | 0.129 | 0.8 / 15.3 | 0.909 / 0.112 / 0.085 |
| misgan_realmask | 100 | 100.0 | 0.010 | 0.890 / 0.219 / 0.102 | 0.920 / 0.216 / 0.072 | 0.974 / 0.739 / 0.013 | 0.974 / 0.737 / 0.013 | 0.073 | 1.1 / 14.6 | 0.910 / 0.102 / 0.084 |
| oracle | 100 | 99.6 | 0.010 | 0.880 / 0.211 / 0.112 | 0.920 / 0.203 / 0.072 | 0.978 / 0.778 / 0.008 | 0.978 / 0.777 / 0.008 | 0.060 | 0.9 / 15.8 | 0.911 / 0.107 / 0.083 |
| *bayes* | - | - | - | 0.978 / 0.777 / 0.006 | 0.978 / 0.777 / 0.006 | 0.978 / 0.777 / 0.006 | 0.978 / 0.777 / 0.006 | 0.062 | - | - |

## Automatic comparisons (vs `misgan`)

- **mcar_p20**: misgan acc 0.991 vs Bayes 1.000 (gap +0.009); oracle dacc +0.005 ditv -0.005 dhq +3.9; zerofill dacc -0.492 ditv +0.492 dhq -88.2; misgan_realmask dacc +0.004 ditv -0.004 dhq +3.8; misgan_paired dacc +0.005 ditv -0.005 dhq +3.9; misgan_gauss dacc -0.015 ditv +0.015 dhq -90.1; misgan_detach dacc +0.000 ditv -0.000 dhq +3.7; aegan_recon_w0.1 dacc +0.003 ditv -0.003 dhq +3.8; aegan_recon_w1 dacc +0.003 ditv -0.003 dhq +3.9; aegan_ce dacc -0.022 ditv +0.022 dhq +1.0
- **mcar_p50**: misgan acc 0.891 vs Bayes 0.985 (gap +0.094); oracle dacc +0.030 ditv -0.030 dhq +56.3; zerofill dacc -0.740 ditv +0.740 dhq -40.0; misgan_realmask dacc +0.027 ditv -0.028 dhq +42.1; misgan_paired dacc +0.022 ditv -0.023 dhq +48.0; misgan_gauss dacc -0.003 ditv +0.003 dhq -38.6; misgan_hard dacc -0.451 ditv +0.451 dhq -41.5; misgan_detach dacc +0.035 ditv -0.034 dhq +34.8; misgan_long dacc +0.039 ditv -0.040 dhq +7.5; aegan_recon_w0.1 dacc +0.019 ditv -0.020 dhq +9.5; aegan_recon_w1 dacc +0.016 ditv -0.017 dhq +7.3; aegan_ce dacc -0.293 ditv +0.291 dhq +13.5
- **mcar_p80**: misgan acc 0.427 vs Bayes 0.701 (gap +0.274); oracle dacc +0.034 ditv -0.054 dhq +93.9; zerofill dacc -0.394 ditv +0.431 dhq -1.3; misgan_realmask dacc +0.015 ditv -0.028 dhq +10.6; misgan_paired dacc +0.003 ditv -0.017 dhq +7.7; misgan_gauss dacc -0.030 ditv +0.032 dhq -3.0; misgan_detach dacc -0.047 ditv +0.043 dhq -1.5; aegan_recon_w0.1 dacc -0.001 ditv -0.005 dhq -3.0; aegan_recon_w1 dacc +0.029 ditv -0.035 dhq -0.6; aegan_ce dacc -0.142 ditv +0.130 dhq -3.0
- **block**: misgan acc 0.910 vs Bayes 0.978 (gap +0.068); oracle dacc +0.001 ditv -0.001 dhq -0.2; zerofill dacc -0.758 ditv +0.764 dhq -98.3; misgan_realmask dacc +0.000 ditv -0.000 dhq +0.2; misgan_paired dacc -0.000 ditv +0.000 dhq -0.0; misgan_gauss dacc -0.001 ditv +0.001 dhq -93.6; misgan_detach dacc +0.000 ditv +0.000 dhq -0.1; aegan_recon_w0.1 dacc +0.001 ditv -0.000 dhq +0.0; aegan_recon_w1 dacc +0.002 ditv -0.000 dhq +0.0; aegan_ce dacc -0.012 ditv +0.013 dhq +0.0

## Prediction checks (numbers)

- mcar_p20 aegan_recon_w0.1: (a) acc_lo 0.688 vs Bayes 0.312; (b) itv/istd 0.007/0.009 vs ppost on its G_x 0.000/0.003; (c) G_x hq 99.7, modes 100
- mcar_p20 aegan_recon_w1: (a) acc_lo 0.938 vs Bayes 0.312; (b) itv/istd 0.007/0.007 vs ppost on its G_x 0.000/0.003; (c) G_x hq 99.8, modes 100
- mcar_p20 (d) ppost on oracle: acc_lo 0.062 (own G_i 1.000, Bayes 0.312; share of gap closed 1.36); itv 0.000 vs 0.004
- mcar_p20 (d) ppost on misgan_detach: acc_lo 0.375 (own G_i 1.000, Bayes 0.312; share of gap closed 0.91); itv 0.000 vs 0.009
- mcar_p20 (d) ppost on misgan_realmask: acc_lo 0.125 (own G_i 0.938, Bayes 0.312; share of gap closed 1.30); itv 0.000 vs 0.005
- mcar_p20 (d) ppost on misgan: acc_lo 0.062 (own G_i 1.000, Bayes 0.312; share of gap closed 1.36); itv 0.000 vs 0.010
- mcar_p20 (e) aegan_ce acc/acc_lo/itv 0.969/0.000/0.031 at 35.9 ms/1k vs ppost on its G_x 1.000/0.062/0.000 at 1.3 ms/1k
- mcar_p20 (f) misgan: ppost acc M256 0.976 -> M4096 1.000; acc_lo 0.062 -> 0.062
- mcar_p20 (f) misgan_gauss: ppost acc M256 0.993 -> M4096 1.000; acc_lo 0.312 -> 0.062
- mcar_p20 (f) oracle: ppost acc M256 0.988 -> M4096 1.000; acc_lo 0.062 -> 0.062
- mcar_p50 aegan_recon_w0.1: (a) acc_lo 0.140 vs Bayes 0.555; (b) itv/istd 0.089/0.008 vs ppost on its G_x 0.023/0.051; (c) G_x hq 53.0, modes 97
- mcar_p50 aegan_recon_w1: (a) acc_lo 0.143 vs Bayes 0.555; (b) itv/istd 0.093/0.011 vs ppost on its G_x 0.019/0.047; (c) G_x hq 50.8, modes 99
- mcar_p50 (d) ppost on oracle: acc_lo 0.568 (own G_i 0.112, Bayes 0.555; share of gap closed 1.03); itv 0.009 vs 0.079
- mcar_p50 (d) ppost on misgan_detach: acc_lo 0.446 (own G_i 0.145, Bayes 0.555; share of gap closed 0.73); itv 0.016 vs 0.075
- mcar_p50 (d) ppost on misgan_realmask: acc_lo 0.481 (own G_i 0.128, Bayes 0.555; share of gap closed 0.83); itv 0.015 vs 0.081
- mcar_p50 (d) ppost on misgan: acc_lo 0.328 (own G_i 0.121, Bayes 0.555; share of gap closed 0.48); itv 0.024 vs 0.109
- mcar_p50 (e) aegan_ce acc/acc_lo/itv 0.598/0.052/0.400 at 25.0 ms/1k vs ppost on its G_x 0.969/0.342/0.023 at 0.7 ms/1k
- mcar_p50 (f) misgan: ppost acc M256 0.891 -> M4096 0.969; acc_lo 0.163 -> 0.328
- mcar_p50 (f) misgan_gauss: ppost acc M256 0.901 -> M4096 0.956; acc_lo 0.131 -> 0.169
- mcar_p50 (f) oracle: ppost acc M256 0.904 -> M4096 0.983; acc_lo 0.202 -> 0.568
- mcar_p80 aegan_recon_w0.1: (a) acc_lo 0.086 vs Bayes 0.408; (b) itv/istd 0.529/0.317 vs ppost on its G_x 0.572/0.510; (c) G_x hq 2.8, modes 6
- mcar_p80 aegan_recon_w1: (a) acc_lo 0.085 vs Bayes 0.408; (b) itv/istd 0.499/0.334 vs ppost on its G_x 0.469/0.473; (c) G_x hq 5.3, modes 21
- mcar_p80 (d) ppost on oracle: acc_lo 0.406 (own G_i 0.091, Bayes 0.408; share of gap closed 0.99); itv 0.196 vs 0.480
- mcar_p80 (d) ppost on misgan_detach: acc_lo 0.079 (own G_i 0.080, Bayes 0.408; share of gap closed -0.00); itv 0.548 vs 0.577
- mcar_p80 (d) ppost on misgan_realmask: acc_lo 0.145 (own G_i 0.090, Bayes 0.408; share of gap closed 0.17); itv 0.385 vs 0.506
- mcar_p80 (d) ppost on misgan: acc_lo 0.099 (own G_i 0.085, Bayes 0.408; share of gap closed 0.04); itv 0.470 vs 0.534
- mcar_p80 (e) aegan_ce acc/acc_lo/itv 0.285/0.062/0.664 at 34.7 ms/1k vs ppost on its G_x 0.348/0.074/0.594 at 1.0 ms/1k
- mcar_p80 (f) misgan: ppost acc M256 0.379 -> M4096 0.462; acc_lo 0.081 -> 0.099
- mcar_p80 (f) misgan_gauss: ppost acc M256 0.372 -> M4096 0.447; acc_lo 0.076 -> 0.089
- mcar_p80 (f) oracle: ppost acc M256 0.520 -> M4096 0.693; acc_lo 0.171 -> 0.406
- block aegan_recon_w0.1: (a) acc_lo 0.109 vs Bayes 0.777; (b) itv/istd 0.084/0.122 vs ppost on its G_x 0.013/0.070; (c) G_x hq 99.8, modes 100
- block aegan_recon_w1: (a) acc_lo 0.115 vs Bayes 0.777; (b) itv/istd 0.084/0.094 vs ppost on its G_x 0.011/0.066; (c) G_x hq 99.8, modes 100
- block (d) ppost on oracle: acc_lo 0.778 (own G_i 0.107, Bayes 0.777; share of gap closed 1.00); itv 0.008 vs 0.083
- block (d) ppost on misgan_detach: acc_lo 0.731 (own G_i 0.100, Bayes 0.777; share of gap closed 0.93); itv 0.013 vs 0.084
- block (d) ppost on misgan_realmask: acc_lo 0.739 (own G_i 0.102, Bayes 0.777; share of gap closed 0.94); itv 0.013 vs 0.084
- block (d) ppost on misgan: acc_lo 0.748 (own G_i 0.097, Bayes 0.777; share of gap closed 0.96); itv 0.011 vs 0.084
- block (e) aegan_ce acc/acc_lo/itv 0.897/0.103/0.097 at 27.3 ms/1k vs ppost on its G_x 0.971/0.714/0.014 at 0.6 ms/1k
- block (f) misgan: ppost acc M256 0.888 -> M4096 0.975; acc_lo 0.201 -> 0.748
- block (f) misgan_gauss: ppost acc M256 0.900 -> M4096 0.918; acc_lo 0.118 -> 0.180
- block (f) oracle: ppost acc M256 0.880 -> M4096 0.978; acc_lo 0.211 -> 0.778
<!-- AUTO:end -->
