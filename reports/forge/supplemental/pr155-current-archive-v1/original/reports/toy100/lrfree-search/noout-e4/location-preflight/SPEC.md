# Prospective signed-location mixture preflight

Written before `simulate.py` was run. This is synthetic-only feasibility
screening of an *idealized* row-gradient summary, not an E4 repair, candidate
trainer, or A2 calibration. It reads no benchmark gradient, output, gate, or
task geometry. Astra proposed this as a distinct model after the Gaussian
scale mixture failed its predeclared gate.

## Fixed fixture and acceptance rule

Use the same fixture family and criteria as the earlier
[Gaussian-mixture spec](../eb-preflight/SPEC.md): `N=20,000`,
`π∈{.01,.05}`, `|δ|∈{.5,1}`, touches `m∈{128,700}`,
`ρ∈{0,.5,.9}`, standard-normal or unit-variance Student-t5 *summary*
noise, plus pure-null tables. `X=δ sqrt(I(m,ρ))+Z` with random signs and
`I=(m(1−ρ)+2ρ)/(1+ρ)`. There are three independent fixed synthetic
replicates with NumPy `default_rng` seeds `20261002`, `20261003`, and
`20261004`. These are independent calibration tables, not alternate
benchmark seeds or trainer selection.

Every replicate must meet every applicable rule: mean gain on truly null rows
≤ `.05`; for normal noise and `ρ≤.5`, drift mean gain ≥ `.5` at `(m,|δ|)`
equal to `(128,1)` or `(700,.5)`; and the fit must converge. Pure-null
tables must have mean gain ≤ `.05`. Record null gain median, p95, p99, and
fraction above `.5` for diagnosis, without adding a post-run threshold.

## Fixed likelihood and numeric fit

For each table, use the *known* summary-noise family `f` and fit
`p(x)=(1−π_hat)f(x)+(π_hat/2)[f(x−μ)+f(x+μ)]`, `0≤π_hat≤1`, `μ≥0`.
The normal density has unit variance. Student-t5 uses scale `sqrt(3/5)`
so its variance is one. Minimize negative log likelihood (relative to the
all-null fit). This oracle knows the family; an online implementation would
have to infer it or be robust to misspecification.

Profile over `π_hat` at each `μ` using bounded golden-section minimization on
`[0,1]`, absolute tolerance `1e−6`, at most 100 evaluations; also compare
both `π_hat` endpoints. Scan `μ=0` and 32 equally spaced positive points
through `max(|X|)`. Refine every strict local optimum in that grid within
its two adjacent grid values using bounded golden-section minimization, absolute
tolerance `1e−5 max(1,max|X|)`, at most 100 evaluations. Compare all grid
and refinement points; take the highest likelihood, breaking exact ties
toward smaller `μ` and then smaller `π_hat`. A failed optimizer or nonfinite
likelihood fails that replicate; do not choose another fit after inspection.
The upper bound `max|X|` is determined by the row summaries, not a benchmark.

For the chosen fit, set continuous gain
`w(x)=P(signal|x) μ²/(1+μ²)`. At the `π_hat=0` or `μ=0` boundary, gain is
zero. No FDR decision, hold, exclusion, or benchmark-derived memory constant
appears. If this preflight fails, stop this mixture-gain route. If it passes,
online direction/covariance/dependence and memory selection still require an
independently specified detector and synthetic calibration before native use.

## Environment amendment before fixture execution

The first launch exited while importing SciPy: this environment's SciPy
requires NumPy `<1.25`, while installed NumPy is `2.0.2`. No fixture was
generated and `results.log` was empty. The bounded scalar searches above
are implemented directly with the golden-section rule in NumPy/Python, and
the Student-t normalization uses `math.lgamma`; the search intervals,
tolerances, evaluation cap, fixture, and acceptance rules are unchanged.
