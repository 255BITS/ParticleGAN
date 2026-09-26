# MisGAN toy: findings

Protocol: [README.md](README.md). There are 25 runs, all on one seed, on the
branch `misgan-toy`. Every arm differs in substance. One run is deterministic:
`mcar_p50__misgan` gave the same numbers in the pilot and in both grid passes.
Treat differences of about 0.01 in `acc` as noise.

## Summary

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

## Notes

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

## Recommended next experiments

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

| arm | acc | gap | acc_lo | itv | istd | rmse | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.996 | 0.004 | 1.000 | 0.004 | 0.000 | 0.066 | 92.2 | 100 | 99.8 | 0.047 | 0.010 | 0.020 | 0.053 | 0.001 | 0.996 |
| misgan_paired | 0.995 | 0.005 | 1.000 | 0.005 | 0.007 | 0.076 | 87.0 | 100 | 99.8 | 0.025 | 0.018 | 0.020 | 0.053 | 0.001 | 0.995 |
| misgan_realmask | 0.995 | 0.005 | 0.938 | 0.005 | 0.010 | 0.073 | 86.7 | 100 | 99.7 | 0.029 | 0.020 | 0.007 | 0.019 | 0.000 | 0.995 |
| misgan | 0.991 | 0.009 | 1.000 | 0.010 | 0.006 | 0.101 | 76.5 | 100 | 95.9 | 0.021 | 0.032 | 0.042 | 0.119 | 0.000 | 0.991 |
| zerofill | 0.498 | 0.501 | 1.000 | 0.502 | 0.010 | 0.922 | 25.3 | 5 | 7.6 | 0.202 | 0.904 | 0.020 | 0.053 | 0.001 | 0.586 |
| *bayes* | 1.000 | 0.000 | 0.312 | 0.000 | 0.001 | 0.010 | 98.9 | - | - | - | - | - | - | - | - |
| *knn* | 0.986 | 0.014 | 0.000 | 0.014 | 0.000 | 0.098 | 93.2 | - | - | - | - | - | - | - | - |
| *mean* | 0.490 | 0.510 | 1.000 | 0.510 | 0.000 | 0.994 | 25.7 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | 100 | 98.9 | 0.017 | 0.001 | - | - | - | - |

Bayes MAP accuracy 1.000. Mask TV column = TV of the observed-count histogram vs Binomial(8, 1-p).

### mcar_p50

| arm | acc | gap | acc_lo | itv | istd | rmse | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| misgan_long | 0.930 | 0.055 | 0.146 | 0.069 | 0.001 | 0.274 | 44.8 | 100 | 51.0 | 0.031 | 0.056 | 0.009 | 0.040 | 0.002 | 0.930 |
| misgan_detach | 0.925 | 0.059 | 0.145 | 0.075 | 0.001 | 0.267 | 57.2 | 100 | 78.3 | 0.023 | 0.049 | 0.006 | 0.020 | 0.002 | 0.925 |
| oracle | 0.921 | 0.064 | 0.112 | 0.079 | 0.001 | 0.327 | 70.2 | 100 | 99.8 | 0.040 | 0.010 | 0.006 | 0.020 | 0.002 | 0.921 |
| misgan_realmask | 0.918 | 0.066 | 0.128 | 0.081 | 0.011 | 0.270 | 54.2 | 100 | 85.6 | 0.024 | 0.034 | 0.006 | 0.013 | 0.000 | 0.918 |
| misgan_paired | 0.913 | 0.072 | 0.127 | 0.086 | 0.017 | 0.274 | 53.7 | 100 | 91.5 | 0.024 | 0.033 | 0.006 | 0.020 | 0.002 | 0.913 |
| misgan | 0.891 | 0.094 | 0.121 | 0.109 | 0.002 | 0.344 | 39.7 | 100 | 43.5 | 0.031 | 0.138 | 0.018 | 0.034 | 0.001 | 0.891 |
| misgan_gauss | 0.888 | 0.097 | 0.145 | 0.112 | 0.001 | 0.289 | 19.7 | 14 | 4.9 | 0.029 | 0.066 | 0.004 | 0.012 | 0.003 | 0.888 |
| misgan_hard | 0.440 | 0.545 | 0.031 | 0.560 | 0.015 | 0.688 | 5.1 | 5 | 2.0 | 0.148 | 1.069 | 0.171 | 0.258 | 0.000 | 0.714 |
| zerofill | 0.151 | 0.834 | 0.014 | 0.849 | 0.014 | 0.931 | 4.4 | 4 | 3.6 | 0.347 | 1.344 | 0.006 | 0.020 | 0.002 | 0.159 |
| *bayes* | 0.985 | 0.000 | 0.555 | 0.006 | 0.026 | 0.202 | 98.9 | - | - | - | - | - | - | - | - |
| *knn* | 0.563 | 0.422 | 0.382 | 0.438 | 0.000 | 0.437 | 18.4 | - | - | - | - | - | - | - | - |
| *mean* | 0.126 | 0.859 | 0.015 | 0.874 | 0.000 | 0.992 | 4.7 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | 100 | 98.9 | 0.017 | 0.001 | - | - | - | - |

Bayes MAP accuracy 0.988. Mask TV column = TV of the observed-count histogram vs Binomial(8, 1-p).

### mcar_p80

| arm | acc | gap | acc_lo | itv | istd | rmse | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.461 | 0.240 | 0.091 | 0.480 | 0.318 | 0.979 | 41.7 | 100 | 99.8 | 0.049 | 0.010 | 0.039 | 0.114 | 0.002 | 0.461 |
| misgan_realmask | 0.442 | 0.259 | 0.090 | 0.506 | 0.377 | 0.958 | 13.5 | 80 | 16.5 | 0.026 | 0.083 | 0.018 | 0.052 | 0.001 | 0.442 |
| misgan_paired | 0.430 | 0.270 | 0.090 | 0.517 | 0.310 | 0.960 | 9.1 | 58 | 13.6 | 0.031 | 0.075 | 0.039 | 0.114 | 0.002 | 0.430 |
| misgan | 0.427 | 0.274 | 0.085 | 0.534 | 0.304 | 0.963 | 6.4 | 20 | 5.9 | 0.041 | 0.216 | 0.021 | 0.064 | 0.000 | 0.431 |
| misgan_detach | 0.380 | 0.321 | 0.080 | 0.577 | 0.340 | 0.979 | 5.9 | 13 | 4.4 | 0.052 | 0.265 | 0.039 | 0.114 | 0.002 | 0.400 |
| zerofill | 0.033 | 0.668 | 0.017 | 0.965 | 0.007 | 1.007 | 3.2 | 5 | 4.6 | 0.548 | 0.831 | 0.039 | 0.114 | 0.002 | 0.034 |
| *bayes* | 0.701 | 0.000 | 0.408 | 0.179 | 0.334 | 0.774 | 98.8 | - | - | - | - | - | - | - | - |
| *knn* | 0.282 | 0.419 | 0.217 | 0.719 | 0.000 | 0.653 | 8.6 | - | - | - | - | - | - | - | - |
| *mean* | 0.032 | 0.668 | 0.017 | 0.968 | 0.000 | 0.992 | 3.7 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | 100 | 98.9 | 0.017 | 0.001 | - | - | - | - |

Bayes MAP accuracy 0.722. Mask TV column = TV of the observed-count histogram vs Binomial(8, 1-p).

### block

| arm | acc | gap | acc_lo | itv | istd | rmse | ihq | modes | hq | swd | off | m_mae | m_tv | m_soft | acc_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.911 | 0.067 | 0.107 | 0.083 | 0.140 | 0.446 | 86.8 | 100 | 99.6 | 0.036 | 0.012 | 0.007 | 0.010 | 0.000 | 0.911 |
| misgan_realmask | 0.910 | 0.068 | 0.102 | 0.084 | 0.151 | 0.466 | 85.2 | 100 | 100.0 | 0.027 | 0.017 | 0.006 | 0.014 | 0.000 | 0.911 |
| misgan | 0.910 | 0.068 | 0.097 | 0.084 | 0.161 | 0.497 | 86.2 | 100 | 99.8 | 0.031 | 0.016 | 0.004 | 0.007 | 0.000 | 0.910 |
| misgan_paired | 0.910 | 0.068 | 0.095 | 0.085 | 0.162 | 0.494 | 86.5 | 100 | 99.8 | 0.032 | 0.016 | 0.007 | 0.010 | 0.000 | 0.911 |
| zerofill | 0.152 | 0.826 | 0.008 | 0.848 | 0.007 | 0.997 | 1.2 | 6 | 1.5 | 0.279 | 1.429 | 0.007 | 0.010 | 0.000 | 0.288 |
| *bayes* | 0.978 | 0.000 | 0.777 | 0.006 | 0.062 | 0.292 | 98.9 | - | - | - | - | - | - | - | - |
| *knn* | 0.949 | 0.028 | 0.494 | 0.050 | 0.000 | 0.227 | 93.8 | - | - | - | - | - | - | - | - |
| *mean* | 0.140 | 0.837 | 0.006 | 0.859 | 0.000 | 0.995 | 1.4 | - | - | - | - | - | - | - | - |
| *clean sample* | - | - | - | - | - | - | - | 100 | 99.0 | 0.011 | 0.001 | - | - | - | - |

Bayes MAP accuracy 0.984. Mask TV column = TV over the 4 block patterns (256-pattern space).

## Automatic comparisons (vs `misgan`)

- **mcar_p20**: misgan acc 0.991 vs Bayes 1.000 (gap +0.009); oracle dacc +0.005 ditv -0.005 dhq +3.9; zerofill dacc -0.492 ditv +0.492 dhq -88.2; misgan_realmask dacc +0.004 ditv -0.004 dhq +3.8; misgan_paired dacc +0.005 ditv -0.005 dhq +3.9
- **mcar_p50**: misgan acc 0.891 vs Bayes 0.985 (gap +0.094); oracle dacc +0.030 ditv -0.030 dhq +56.3; zerofill dacc -0.740 ditv +0.740 dhq -40.0; misgan_realmask dacc +0.027 ditv -0.028 dhq +42.1; misgan_paired dacc +0.022 ditv -0.023 dhq +48.0; misgan_gauss dacc -0.003 ditv +0.003 dhq -38.6; misgan_hard dacc -0.451 ditv +0.451 dhq -41.5; misgan_detach dacc +0.035 ditv -0.034 dhq +34.8; misgan_long dacc +0.039 ditv -0.040 dhq +7.5
- **mcar_p80**: misgan acc 0.427 vs Bayes 0.701 (gap +0.274); oracle dacc +0.034 ditv -0.054 dhq +93.9; zerofill dacc -0.394 ditv +0.431 dhq -1.3; misgan_realmask dacc +0.015 ditv -0.028 dhq +10.6; misgan_paired dacc +0.003 ditv -0.017 dhq +7.7; misgan_detach dacc -0.047 ditv +0.043 dhq -1.5
- **block**: misgan acc 0.910 vs Bayes 0.978 (gap +0.068); oracle dacc +0.001 ditv -0.001 dhq -0.2; zerofill dacc -0.758 ditv +0.764 dhq -98.3; misgan_realmask dacc +0.000 ditv -0.000 dhq +0.2; misgan_paired dacc -0.000 ditv +0.000 dhq -0.0
<!-- AUTO:end -->
