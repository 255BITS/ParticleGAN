# Anisotropic core metric and SiLU critic screen

`python run.py` runs 21 CPU episodes (seed 0, 1 thread each, fixed cosine, same
recipe as the suite) in parallel in about 25 s. `run.log` has one line per
episode, and `results.jsonl` has one record per episode with final live metrics,
the sustained verdict and the failing thresholds at each observation.
Verdict = `run_episode` convergence: the whole schedule completed and the final
passing suffix is at least 5 of 24 observations.
`python run.py --v4` reruns the 4 protocol v4 episodes (about 10 s) into
`run_v4.log` and `results_v4.jsonl`; see the last section.

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

## Protocol v4: unequal width and unequal mass

`vector_unequal_width` and `vector_unequal_mass` now gate on the same six
core/spill bounds as anisotropic (unequal mass keeps `min_mass_ratio ≥.25`).
Same fixed cosine recipe, seed 0, sustained rule; the 4 episodes reproduce the
section B metrics exactly. Old = v2 whole-component bounds on the same run.

| task | critic | old verdict (suffix) | new verdict (suffix) | sw1 | mass_tv | old cov | core cov | core eig | spill (worst comp) | min mass ratio | new failing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unequal_width | axis_silu | fail (0) | **SUST (8)** | 0.091 | 0.056 | 0.89 | 0.16 | 0.57 | 0.041 | - | - |
| unequal_width | leaky_orig | fail (0) | fail (0) | 0.085 | 0.083 | 7.81 | 0.18 | 0.50 | 0.055 (σ=.07) | - | spill |
| unequal_mass | leaky_orig | fail (0) | fail (0) | 0.081 | 0.052 | 5.80 | 0.31 | 0.45 | 0.418 (2% comp) | 0.91 | spill |
| unequal_mass | axis_silu | fail (0) | fail (0) | 0.137 | 0.086 | 1.44 | 0.39 | 0.00 | 0.046 | 0.65 | core eig |

- SiLU unequal_width flips to a pass: its last 8 observations clear every bound.
  The old failure was spill-inflated covariance error (0.89 vs 0.85).
- LeakyReLU unequal_width fails only on spill in its last 5 observations; at
  the end the narrowest component (σ=.07) spills 5.5% beyond 3σ. The shape is
  fine (core 0.18).
- LeakyReLU unequal_mass: the rare 2% component gets enough mass (ratio 0.91)
  and its core shape is fine, but 42% of the samples assigned to it lie beyond
  3σ. These are strays from the big neighbors. In its last 4 observations spill
  is the only failing bound; before that core eig also flickers below 0.15.
- SiLU unequal_mass collapses the rare component to a point (core eig 0.00, core
  error 0.99), the same finding as B. This is a real shape failure. Core eig
  fails at every one of its last 6 observations, and spill fails at some.

### Whole-suite pass count (current protocol, 8 development tasks)

Sustained passes. Anisotropic and the other four tasks reuse the #208 runs.

| critic | ranking (6) | diagnostic (2) | total | fails |
|---|---|---|---|---|
| axis_silu | 4 | 1 | **5/8** | unequal_mass (core eig), overlap (suffix 4), narrow |
| leaky_orig | 3 | 1 | 4/8 | unequal_width (spill), unequal_mass (spill), overlap (suffix 4), narrow |

Under v3 both critics scored 4/8. The v4 change moves only SiLU unequal_width.

### Recommendations (v4)

1. Keep v4. Each remaining failure now names one cause (spill or a collapsed
   rare core) instead of a covariance number inflated 5–40×.
2. `axis_silu` now leads LeakyReLU 5/8 vs 4/8 and has a much larger margin on
   anisotropic. It is the better candidate default critic. Its one ranking gap
   is rare-mode collapse on unequal_mass. Test a fix aimed at that (for example
   more particles or a prior-side floor on the rare mode), not another
   activation sweep.
3. LeakyReLU's failures are all spill. A spill-reducing change (an output-side
   sharpness cap, or a critic with a sharper boundary between nearby modes)
   is the matching experiment for it. Its unequal_width miss is marginal
   (0.055 vs 0.05).
4. The unequal_mass spill metric is a ratio over a component with about 80
   samples, so a few dozen strays from 55%/30% neighbors dominate it. If that
   case keeps failing only on spill, consider normalizing spill by target mass.
   That would need a separate protocol change, not a tweak.

## Follow-up: SiLU rare-mode collapse

The unequal_mass collapse is diagnosed in [`../silu_rare_collapse/`](../silu_rare_collapse/README.md).
A smooth SiLU critic forms a bump wider than σ over the under-filled 2% mode, and it pulls the
mode's 3 particles together in the generator's row space. None of 14 SiLU arms keeps the rare
core (rare core eig ≤.08). No critic-side variant keeps SiLU's 5/8; both cross-checked
alternatives score 3/8.

## Follow-up: protocol v5

The unequal_mass core-eig and spill gates on the 2% component (~5 particles) mostly measured
particle count. Protocol v5 exempts components below 32 declared particles from the shape and spill
gates; their mass is still gated. See the [v5 section](../silu_rare_collapse/README.md#protocol-v5-particle-resolution-floor).
Pass counts are unchanged: axis_silu 5/8, LeakyReLU 4/8.
