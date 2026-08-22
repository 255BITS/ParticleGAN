# Reconciling `per_mode_std_ratio` with `hq`

The audit reports mean within-mode std ratios of 2.1-5.4 (over-dispersed) next to `hq` values up to 0.992. A 2-D isotropic Gaussian cannot do both: at std ratio r its mass inside 3 sigma_true is `1 - exp(-(3/r)^2/2)`, which is **0.675 at r = 2** and 0.180 at r = 5. This file resolves the tension from the saved 100k EMA sample clouds (`final_samples.npy`) rather than by assumption.

## The sigma definitions

Three spread rules -- raw, trimmed, robust -- each computed per mode on the nearest-center assignment `per_mode_moments` uses, over modes holding >= 50 samples, then averaged over modes.

| estimator | definition | what it is sensitive to |
|---|---|---|
| `sigma_raw` | `sqrt(mean_d Var_d(x))` about the mode's sample mean -- the audit's own `per_mode_std` | the full second moment: one point at 10 sigma weighs as much as 100 at 1 sigma |
| `sigma_trim` | the same after dropping the 2% of each mode's points farthest from its center | the body of the blob |
| `sigma_trim_corrected` | `sigma_trim / 0.9593` | as above, with the bias the same trim inflicts on a *true* Gaussian divided out |
| `sigma_robust` | `median(||x - center||) / sqrt(2 ln 2)` (= median / 1.17741) | the core only; a Gaussian-consistent inversion that ignores everything past the median |

On a genuine Gaussian all four agree, so `sigma_raw / sigma_robust` reads directly as a tail-inflation factor: 1.0 means Gaussian, larger means the variance lives outside the core.

**The center matters as much as the spread rule.** A mode's *sample mean* is itself dragged by the far tail -- 3% of a mode's mass stranded at distance 2.5 moves the mean by 0.075, i.e. 2.5 sigma_true -- and every distance measured from a displaced center is then inflated. Each rule is therefore computed about all three centers that make sense: the sample **mean** (what the audit uses), the coordinate-wise **median** (robust and center-free), and the **true** grid center (the one `hq` measures from). The median- and true-center columns are the ones to read; the mean-centered ones are kept to show the size of the contamination.

One caveat on the trim: dropping 2% only helps when the tail is thinner than that. Where the far-tail mass exceeds the trim fraction the trimmed std stays close to the raw one, and the median-based column is the only estimator in the ladder that still reports the core.

**Units.** Samples, distances, sigmas and `center_bias` are all in absolute data coordinates. The data's within-mode sigma is **0.03** and the grid spacing between neighbouring centers is **1.0**, so 3 sigma_true = 0.09 (the `hq` ball), 10 sigma_true = 0.30, and the Voronoi boundary between two modes is at 0.5. A `center_bias` of 0.02 is 0.67 sigma_true and 2% of the way to the neighbouring mode.

## Nearest-center distance distribution

| group | q50 | q90 | q99 | q99.9 | max | frac <= 0.09 (`hq`) | frac > 0.30 | frac > 0.15 |
|---|---|---|---|---|---|---|---|---|
| `b_cap_c1p0_lr2p0` | 0.0324 | 0.0595 | 0.4673 | 2.0686 | 3.585 | 0.9867 ± 0.0031 | 0.00815 ± 0.00243 | 0.00839 ± 0.00247 |
| `a_r1r2_c1p0` | 0.0745 | 0.2044 | 0.5167 | 1.6132 | 2.232 | 0.5958 ± 0.0053 | 0.04677 ± 0.00190 | 0.18368 ± 0.00389 |
| `a_r1r2_c0p1` | 0.0323 | 0.0727 | 0.3729 | 2.0752 | 3.094 | 0.9427 ± 0.0051 | 0.01154 ± 0.00225 | 0.02133 ± 0.00216 |
| `c_eikonal_c0p1` | 0.0184 | 0.0490 | 2.0117 | 3.0573 | 4.538 | 0.9619 ± 0.0070 | 0.03151 ± 0.00390 | 0.03250 ± 0.00370 |

## Sigma-ratio decomposition

Ratios are `sigma / 0.03`, so 1.00 is a perfect match to the data.

| group | raw (audit) | trim, mean-ctr | trim, median-ctr | trim, median-ctr, bias-corr. | robust, mean-ctr | **robust, median-ctr** | **robust, true-ctr** | raw / robust |
|---|---|---|---|---|---|---|---|---|
| `b_cap_c1p0_lr2p0` | 2.73 ± 0.39 | 1.86 ± 0.34 | 1.86 ± 0.34 | 1.94 ± 0.36 | 1.22 ± 0.11 | **0.87 ± 0.01** | **0.92 ± 0.02** | 3.15x |
| `a_r1r2_c1p0` | 3.59 ± 0.08 | 2.90 ± 0.02 | 2.90 ± 0.02 | 3.03 ± 0.02 | 2.21 ± 0.02 | **2.16 ± 0.02** | **2.18 ± 0.02** | 1.66x |
| `a_r1r2_c0p1` | 2.53 ± 0.26 | 1.77 ± 0.24 | 1.77 ± 0.24 | 1.85 ± 0.25 | 1.18 ± 0.09 | **0.91 ± 0.02** | **0.96 ± 0.03** | 2.78x |
| `c_eikonal_c0p1` | 4.80 ± 0.51 | 4.04 ± 0.49 | 4.05 ± 0.49 | 4.22 ± 0.51 | 2.36 ± 0.38 | **0.55 ± 0.23** | **0.69 ± 0.27** | 8.73x |

The gap between the two robust columns and the mean-centered one is pure center contamination -- same points, same rule, only the center moved.

Which width actually explains `hq`? Feeding each ratio through `1 - exp(-(3/r)^2/2)` gives the hq a Gaussian of that width would score. The true-center column is the like-for-like one: `hq` is itself a distance-to-true-center statistic, so an offset blob is penalized in both.

| group | hq implied by raw | hq implied by core width (true-ctr) | measured hq | reading |
|---|---|---|---|---|
| `b_cap_c1p0_lr2p0` | 0.466 | 0.995 | 0.9867 | core width explains hq; raw std is tail-inflated |
| `a_r1r2_c1p0` | 0.294 | 0.614 | 0.5958 | core width explains hq; raw std is tail-inflated |
| `a_r1r2_c0p1` | 0.510 | 0.992 | 0.9427 | core width + a tail that leaves the 3-sigma ball |
| `c_eikonal_c0p1` | 0.182 | 0.992 | 0.9619 | core width + a tail that leaves the 3-sigma ball |

| group | center bias, mean (audit) | in sigma_true | center bias, median | in sigma_true | audited modes |
|---|---|---|---|---|---|
| `b_cap_c1p0_lr2p0` | 0.0248 ± 0.0046 | 0.83 ± 0.15 | 0.0105 ± 0.0007 | 0.35 ± 0.02 | 100.0 |
| `a_r1r2_c1p0` | 0.0195 ± 0.0008 | 0.65 ± 0.03 | 0.0149 ± 0.0011 | 0.50 ± 0.04 | 100.0 |
| `a_r1r2_c0p1` | 0.0185 ± 0.0037 | 0.62 ± 0.12 | 0.0095 ± 0.0010 | 0.32 ± 0.03 | 100.0 |
| `c_eikonal_c0p1` | 0.0783 ± 0.0116 | 2.61 ± 0.39 | 0.0124 ± 0.0030 | 0.41 ± 0.10 | 100.0 |

## Reconciled story, per group

- **`b_cap_c1p0_lr2p0`** -- a **near-true-width** core with a tail-inflated second moment: the raw ratio 2.73 would imply hq 0.466, while the core width 0.92 implies 0.995 against a measured 0.987. Core ratio 0.87 (about the per-mode median) vs raw 2.73 = 3.1x tail inflation, bought by the 0.82% of samples beyond 10 sigma_true.
- **`a_r1r2_c1p0`** -- a genuinely **over-dispersed** core (2.16x) with a tail-inflated second moment: the raw ratio 3.59 would imply hq 0.294, while the core width 2.18 implies 0.614 against a measured 0.596. Core ratio 2.16 (about the per-mode median) vs raw 3.59 = 1.7x tail inflation, bought by the 4.68% of samples beyond 10 sigma_true.
- **`a_r1r2_c0p1`** -- a **near-true-width** core with a tail-inflated second moment, and the tail is visible in hq too: raw ratio 2.53 implies hq 0.510 and the core width 0.96 implies 0.992, against 0.943 measured -- the shortfall against the core prediction is the tail leaving the 3-sigma ball. Core ratio 0.91 (about the per-mode median) vs raw 2.53 = 2.8x tail inflation, bought by the 1.15% of samples beyond 10 sigma_true.
- **`c_eikonal_c0p1`** -- an **under-dispersed** core with a tail-inflated second moment, and the tail is visible in hq too: raw ratio 4.80 implies hq 0.182 and the core width 0.69 implies 0.992, against 0.962 measured -- the shortfall against the core prediction is the tail leaving the 3-sigma ball. Core ratio 0.55 (about the per-mode median) vs raw 4.80 = 8.7x tail inflation, bought by the 3.15% of samples beyond 10 sigma_true.

## The substantive finding

**Is anyone matching the true variance, even robustly?** On the core estimator, yes -- and the misses run in *both* directions, which the raw ratio hides. The closest to the data is `a_r1r2_c0p1` at 0.91; the full range is 0.55 to 2.16 (`a_r1r2_c1p0`), against raw ratios of 2.5-4.8. That reverses the audit's headline reading in two places. First, `hq` and `per_mode_std_ratio` were never in conflict: a raw ratio of 2.4 implies a Gaussian hq of 0.542, but the raw ratio is a second moment of a tailed distribution, while the core width -- the thing `hq` actually reflects -- predicts measured hq to within 0.050 across these groups against 0.78 for the raw ratio, and its residual is one-signed (it always over-predicts, by exactly the tail that leaves the 3-sigma ball). Second, 'no variance collapse anywhere, all over-dispersed' does not survive the decomposition: `c_eikonal_c0p1` has a core ratio of 0.55 +- 0.23, i.e. its blobs are *under*-dispersed -- variance-compressed cores hidden behind a raw std that a 3.2% far tail inflates to 4.8. The `per_mode_std_ratio < 0.8` guard in `leaderboard.py` cannot see that, because it reads the tail-inflated number. The tail comparison separates the two arms cleanly. `c_eikonal_c0p1` puts 3.15% of its mass beyond 10 sigma_true and 3.25% beyond 0.15, against 0.82% and 0.84% for `b_cap_c1p0_lr2p0` -- 3.9x more far-tail mass, and the eikonal arm's 99th-percentile nearest-center distance is 2.01 against 0.47 -- i.e. its tail is not a wider blob but a population stranded whole grid cells away. Their cores go the other way (core ratio 0.55 for the eikonal arm vs 0.87 for the cap), so `c_eikonal`'s worse W1 and per-mode std are a tail phenomenon on top of an over-tight core, exactly the signature of a critic pinned to unit slope everywhere: it keeps a slope between the modes and leaves mass sitting there.

## Reproduction checks

`frac <= 0.09` recomputed here vs. the run's own `hq` (tolerance 0.005; `hq` is measured on a separate 20k draw from the same EMA model, so a ~1e-3 sampling wobble is expected), and `ratio raw` vs. the summary's `per_mode_std_ratio` (tolerance 0.001; same samples, so this one should be exact).

| group | max |hq dev| | hq tolerance | max |std-ratio dev| | pass |
|---|---|---|---|---|
| `b_cap_c1p0_lr2p0` | 0.00177 | 0.00775 | 9.62e-12 | yes |
| `a_r1r2_c1p0` | 0.00617 | 0.01542 | 2.73e-12 | yes |
| `a_r1r2_c0p1` | 0.00138 | 0.01029 | 5.31e-12 | yes |
| `c_eikonal_c0p1` | 0.00171 | 0.00958 | 9.28e-12 | yes |

