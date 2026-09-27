# Anisotropic core metric and SiLU critic screen

`python run.py` runs 21 CPU episodes (seed 0, 1 thread each, fixed cosine, same
recipe as the suite) in parallel in about 25 s. `run.log` has one line per
episode, and `results.jsonl` has one record per episode with final live metrics,
the sustained verdict and the failing thresholds at each observation.
Verdict = `run_episode` convergence: the whole schedule completed and the final
passing suffix is at least 5 of 24 observations.

## Problem and fix

`vector_anisotropic` gated on `component_covariance_error`. That metric assigns
each sample to its nearest mean and takes the relative Frobenius error of the
component's whole covariance. The target covariances are tiny (norm ≈0.09), so
1–8% of stray samples 1–2 units away dominate the error. The task was measuring
spill, not shape.

The task now gates on shape inside each component's 4σ core (squared Mahalanobis
≤16) and bounds stray mass separately:
`component_core_covariance_error ≤.5`, `component_core_min_eigen_ratio ≥.15`,
`max_component_spill ≤.05` (the largest per-component fraction of samples beyond
3σ; ≈1.1% for a true Gaussian). SW1, TV and HQ are unchanged. The old metrics
are still reported everywhere and still gate the other tasks.

## A. vector_anisotropic, 7 critic arms

| arm | old verdict (suffix) | new verdict (suffix) | sw1 | mass_tv | hq | old cov err | old min eig | core cov err | core min eig | max spill | new failing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| axis_silu | SUST (15) | **SUST (15)** | 0.041 | 0.019 | 0.987 | 0.18 | 0.31 | 0.17 | 0.31 | 0.023 | - |
| leaky_orig | SUST (15) | SUST (5) | 0.141 | 0.107 | 0.990 | 0.14 | 0.56 | 0.14 | 0.56 | 0.034 | - |
| axis_softplus50 | fail (0) | fail (0) | 0.192 | 0.191 | 0.986 | 0.60 | 0.74 | 0.14 | 0.69 | 0.021 | sw1, tv |
| axis_softplus5 | fail (0) | fail (0) | 0.184 | 0.179 | 0.987 | 1.65 | 0.37 | 0.17 | 0.24 | 0.037 | sw1, tv |
| axis_softplus1 | fail (0) | fail (0) | 0.175 | 0.136 | 0.846 | 7.08 | 0.72 | 0.26 | 0.07 | 0.201 | hq, core_eig, spill |
| axis_tanh | fail (0) | fail (0) | 0.219 | 0.193 | 0.976 | 1.17 | 0.45 | 0.16 | 0.07 | 0.075 | sw1, tv, core_eig, spill |
| axis_softplus20 | fail (0) | fail (0) | 0.328 | 0.333 | 0.988 | 0.51 | 0.00 | 0.50 | 0.00 | 1.000 | sw1, tv, all shape |

- No verdict flips. Core errors are ≤0.26 for every arm except softplus20, so
  shape is learned; the old covariance error (up to 7.08) was spill. The smooth
  arms still fail, but for different reasons. softplus5/50 fail on **mass
  allocation** (TV ≈0.18–0.19, SW1 just over 0.18). softplus1 and tanh fail on
  real spill (20% and 7.5%). softplus20 dropped a component (spill 1.0).
- LeakyReLU's passing suffix shrinks from 15 to 5 because one late observation
  (19/24) had spill above 0.05. It is now a marginal pass.

## B. SiLU screen on the other development tasks

| task | leaky verdict (suffix) | SiLU verdict (suffix) | sw1 leaky / SiLU | mass_tv leaky / SiLU | leaky failing | SiLU failing |
|---|---|---|---|---|---|---|
| vector_two_broad | SUST (19) | SUST (21) | 0.039 / 0.107 | 0.015 / 0.021 | - | - |
| vector_unequal_mass | fail (0) | fail (0) | 0.081 / 0.137 | 0.052 / 0.086 | cov | cov, eig |
| vector_unequal_width | fail (0) | fail (0) | 0.085 / 0.091 | 0.083 / 0.056 | cov (7.81) | cov (0.89) |
| vector_overlap | fail (4) | fail (4) | 0.092 / 0.136 | n/a | - (suffix) | - (suffix) |
| vector_spiral | SUST (23) | SUST (21) | 0.063 / 0.042 | n/a | - | - |
| vector_scale_drift (diag) | SUST (22) | SUST (11) | 0.070 / 0.056 | 0.020 / 0.002 | - | - |
| vector_narrow (diag) | fail (0) | fail (0) | 0.499 / 0.147 | 0.500 / 0.158 | sw1, tv, cov, eig | tv, cov, eig |

- There are no verdict changes. SiLU ties LeakyReLU 3/7, and on anisotropic it
  goes 1/1 with a much bigger margin (SW1 0.04 vs 0.14).
- The unequal_mass/width failures under the old metric also look like spill.
  LeakyReLU unequal_width has old cov 7.81 but core 0.18. SiLU unequal_width
  misses by 0.04 (0.89 vs 0.85), with core 0.16, spill 0.041 and core eig 0.57,
  so the core rule would pass it. SiLU unequal_mass collapses the rare 2%
  component to a point (core eig 0.00): real failure. LeakyReLU keeps its shape,
  but 42% of what lands near it is stray (spill 0.418).
- SiLU fixes narrow SW1 (0.50 → 0.15, no dropped components) but still fails
  covariance there.

## Recommendations

1. Keep the core/spill rule for anisotropic; its failures now name the real
   cause (mass or spill). Consider the same rule for unequal_width and
   unequal_mass in a separately versioned protocol change. Under the rule, SiLU
   unequal_width would pass and LeakyReLU unequal_width would fail on spill
   (0.055).
2. `axis_silu` is the best smooth critic: it is the only one that passes
   anisotropic, and it breaks even elsewhere. Its weak spot is the rare
   component (unequal_mass collapse); target that before promoting it.
3. softplus5/50 are mass-allocation failures, not shape failures. Further
   softplus-β sweeps on anisotropic are not worth running.
