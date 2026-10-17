# Source review of the fixed mean-law diagnostic

**PASS for the source-only plan, with the conditioning clarification below.**
Runnable source/input guards still precede its one numerical invocation.
No PT, array values, chart, model or forward was evaluated in this review.

## Pair provenance

The immutable harness creates `clean` once with indexed `_generate`, then
`noisy = clean + sigma * randn_like(clean)` (screen.py:940). `draw_holdout`
stores both results from that same call; `score_dir` writes those maps directly.
Thus the EMA clean/noisy holdout rows share sampled IDs and latent perturbations.
Headers show float32 arrays of shape (100000,2), matching completed7000 summaries
and offsets target1601/noise1602/latent1603. No IDs are archived separately;
pairing follows this construction and row order, not a label or nearest match.
Raw EMA anchors A and sampled clean C remain unpaired.

## Required interpretation

Write psi_g(x) for the fixed even-fitted transform, center, scale and clip frame.
For C→Y use **g(C) for both the bin and both psi evaluations**. Using psi_g(Y)
with g(Y) would mix group-frame reassignment into the noise increment. Report
natural g(Y) population energy and C→Y transition rates separately. State owner
has accepted this clarification.

The per-group identity mu_A=f_L*mu_L+f_U*mu_U is exact only with common group/frame
and fractions from A's actual counts. Empty cohorts remain explicit; do not
omit or renormalize them. The double-view legal cohort L has no identical real
sampling counterpart. Its real supported/inside target is a conditioning
contrast, not an equality null. Do not assign sampled C rows to latent-row L
without stored row IDs.

For fixed g(C), centered independent output noise has zero expected raw mean
increment; nonlinear D and radial clipping can produce a nonzero psi increment.
Let r_C=t−mu_C and delta=mu_Y−mu_C in that common frame. Then r_Y=r_C−delta.
Report weighted vectors, dot products and residual energies: direction alone
does not establish benefit, overshoot or historical causality. A→C additionally
contains latent perturbation through G and finite row sampling. Current charts
cannot reconstruct earlier reactions; their objective values are not one fixed
longitudinal loss.

The single prescribed chart/decomposition is sufficient. Support for the law
mismatch needs noise/anchor-to-clean increments aligned with the residual while
paired raw mean changes remain small. Support for cohort compensation needs an
opposed legal-versus-complement contribution. A negligible/opposed decomposition
rejects that explanation. Record clipping incidence when reporting bounded psi
moments; no alternate chart, cutoff or noise draw is needed.

## Smallest prospective alternative, after measurement only

Keep the existing learned groups, support/category checks and exact copy packet.
Replace the nonlinear critic moment with one **even-real-fitted global output
projection B of rank r≤8**. Stream products of flattened outputs with B, then
store only group centers/scales in r coordinates. Fit with bounded thin range/
power products; no output d×d covariance, group full Jacobian or all-N Jacobian.
Work is O(N*d*r), retained projection/group state O((d+G)*r).

Within a fixed group, ell_g(x)=(Bᵀx−c_g)/s_g is affine, so independent centered
output noise preserves its expected mean. A nonconstant affine map on unbounded
outputs cannot also be globally bounded. Retaining the prescribed radial cap
therefore makes the statistic clipped affine, not strictly linear: clip incidence
and noisy group reassignment remain limits. Nonlinear G's latent perturbation
can still shift even raw means. A first-moment objective alone cannot promise
covariance contraction or the original quality gates.

Any selected replacement must retain one mean hypothesis, common Q/(3K+3),
all original count tests, shared5% ordinary budget, inside/p>Q/same-category and
same-group legality in both views, unique protected sources, actual preview
progress and unchanged commit/reset/lease/noise laws. Its output rank must be
distinct from the critic chart rank in typed settings/state. No production law
or source change is qualified before the fixed diagnostic is measured.
