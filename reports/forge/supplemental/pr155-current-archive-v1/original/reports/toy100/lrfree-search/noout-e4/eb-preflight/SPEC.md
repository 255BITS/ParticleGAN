# Prospective scalar empirical-Bayes gain preflight

Written before running `simulate.py`. This is a **feasibility probe**, not a
trainer candidate or an A2 calibration. It never reads a benchmark gradient,
model sample, real sample, task geometry, or gate result.

## Model and planned grid

Assume the gradient's true direction, marginal variance, and AR(1)
correlation are known. A row's standardized mean statistic is
`X = δ sqrt(I(m,ρ)) + Z`, where `I=(m(1−ρ)+2ρ)/(1+ρ)`. This idealizes away
unknown direction and covariance; implementation can only be harder. There
are `N=20,000` rows. A fraction `π` has persistent effects with random signs;
the rest has zero mean. Test `π∈{.01,.05}`, `|δ|∈{.5,1}`, touches
`m∈{128,700}`, and `ρ∈{0,.5,.9}`. Test standard normal and standardized
Student-t5 `Z` separately. The grid is a generic signal/noise stress family,
not a representation of a native task.

Also run pure-null `π=0` tables for each summary-noise family. In that case
`X=Z`, so `m` and `ρ` do not change the distribution. The t5 case changes
the distribution of the final standardized **summary**; it does not simulate
an AR(1) stream with t5 innovations.

Use three independent synthetic tables per grid point, with fixed NumPy
`default_rng` seeds `20260929`, `20260930`, and `20261001`. These are
replications of a synthetic calibration fixture, never alternate benchmark
seeds or a selection among trainer candidates. Apply every criterion to every
replication; no confidence bound or post-run tolerance is used.

Fit a two-component zero-mean Gaussian scale mixture to the `X` values:
`(1−π_hat) N(0,1) + π_hat N(0,1+a_hat)`. Initialize its parameters from the
second and fourth moments, then use EM to maximize likelihood. With
`u=max(mean(X²)−1,0)` and `v=max(mean(X⁴)/3−1−2u,0)`, initialize
`a=max(v/u,1e−9)` and `π_hat=clip(u/a,1e−9,1−1e−9)` when `u>1e−12`.
Otherwise return zero gains as a boundary fit. Each EM step clips `π_hat`
to `[1e−9,1−1e−9]` and `a` to `[1e−9,∞)`. Stop when both parameter
relative changes are below `1e−8`, or after 500 iterations. A fit still
unconverged at 500 iterations fails the preflight. Each row's
continuous gain is `w_i = P(slab|X_i) a_hat/(1+a_hat)`, between zero and one.
No FDR threshold, hold, exclusion, or per-row decision appears in this
preflight. The actual E4 controller is untouched.

## Falsifier fixed before execution

- For normal innovations, every scenario must give null-row mean gain ≤ `.05`.
- Both pure-null cases must give mean gain ≤ `.05`.
- At `ρ≤.5`, `m=128`, `|δ|=1`, drift-row mean gain must be ≥ `.5`.
- At `ρ≤.5`, `m=700`, `|δ|=.5`, drift-row mean gain must be ≥ `.5`.
- With t5 innovations, null-row mean gain must remain ≤ `.05` at every grid
  point. No power target is imposed under t5 or `ρ=.9`.

Record the null gain's median, 95th and 99th percentiles, and fraction above
`.5` for diagnosis. These tail summaries are not additional pass conditions.

This is deliberately a strong gate for an oracle simplification. If it fails,
do not port this Gaussian-mixture design to a native trainer. If it passes,
the unresolved work is substantial: infer direction/covariance/correlation
from online sparse touches, derive a state-based memory rule, validate its
false gain and stop-response on a separate synthetic fixture, and only then
freeze a new package for native runs. Passing here would not repair E4.
