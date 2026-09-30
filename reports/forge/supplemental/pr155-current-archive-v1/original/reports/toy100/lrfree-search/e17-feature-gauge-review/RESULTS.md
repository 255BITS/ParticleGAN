# E17 learned-feature metric gauge audit

The fixed [specification](SPEC.md) was executed against unmodified E17
`birth_death.py` at SHA-256
`e68a0571db69df7813d09e4c3ef29d0c47373ab0bdec0aefa9c8ad2db703d3cf`.
Run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u probe.py`
from this directory to regenerate [results.json](results.json). This is a
synthetic, one-seed CPU counterexample, not a native100 score or a new trainer.

## Result

The critics return the **same scalar score for every input** in the fixture:
the largest logit difference is 0 in float64. Their extracted features equal
the declared transforms exactly. Yet the E17 isolation test and its executed
table moves change when either learned hidden channel is rescaled.

| Cloud | Hidden transform | BH isolation flags | Executed moves | Change from base |
| --- | --- | ---: | ---: | --- |
| Independent null | Base | 0 | 0 | — |
| Independent null | Either channel ×16 | 0 | 0 | No decision change; scores and p-values change |
| Shifted quarter of table | Base | 44 | 44 | — |
| Shifted quarter of table | First channel ×16 | 0 | 0 | 44 flags and moves lost |
| Shifted quarter of table | Second channel ×16 | 0 | 0 | 44 flags and moves lost |
| Shifted quarter of table | Both channels ×16 | 44 | 44 | Exact control match |
| Shifted quarter of table | 90° rotation | 44 | 44 | Exact control match |

The shifted base moves 44 particle rows to unflagged parents. Each anisotropic
transform moves none; the final latent table differs by up to 2.11 in one
coordinate. The reference kNN IDs change for all 1024 query rows under either
anisotropic transform. Isolation p-values change by as much as 0.986, while
the density-ratio statistic `x` changes by as much as 3.65. Uniform scaling and
rotation preserve `x`, scores, flags, parent choices, and final table rows.

The normal E17 float32 shortlist and an independent exhaustive float64 distance
calculation give identical flag sets and moves in every case. One fake-pool kNN
ID differs between these methods under the second-channel scaling because the
pool contains repeated draws; it does not change a distance or decision. The
sidecar independently reconstructed E17's isolation scores, conformal p-values,
BH flags, radii, pooled dimensions, and density-ratio statistic; its flags were
asserted equal to E17's `_isolated` output. `probe.py` then executed E17's real
`maybe_apply`, including its private RNG and row cloning.

## Interpretation and next check

This is a concrete parameterization sensitivity in the current learned-feature
Euclidean metric. The discriminator function, raw query/real/fake clouds,
particle positions, and private RNG are fixed; a change of hidden coordinates
alone changes the birth/death trajectory. It is therefore a portability risk
for the user's goal of adapting without domain-specific metric tuning. It does
**not** establish that every learned-feature metric is disallowed under A2, or
that this behavior explains any native100 failure. The null case does not
produce moves, so only the shifted case demonstrates trajectory divergence.

Astra independently reviewed the result and agrees it demonstrates different
decisions under the same critic function. Astra cautioned that the differing
moves come from isolation; this does not establish that the sequential
density-ratio reaction differs after multiple evaluations. Its conformal/BH
assumptions also need a separate audit.

## Covariance-normalized sidecar

Following Astra's recommendation, [REPAIR_SPEC.md](REPAIR_SPEC.md) froze a
second test before it ran. [whiten_probe.py](whiten_probe.py) fits a full-rank
covariance from **only the reference real half** and wraps E17's extracted
features with its inverse Cholesky factor. This implements the Mahalanobis
distance `(h_i-h_j)^T C^-1 (h_i-h_j)` using only learned critic features and
a generic reference statistic. The second test adds a non-diagonal invertible
hidden transform and again runs both E17's normal kNN and exhaustive float64.

All 20 transformed comparisons pass: maximum density-ratio `x` difference is
`1.43e-14`, isolation-score difference is `7.11e-15`, p-values and BH flags
match exactly, and chosen moves and final particle tables match exactly.
The reference covariance condition number is about 2.08 in the base critic
and at most 530 after scaling. Detailed measurements are in
[whiten_results.json](whiten_results.json).

**The sidecar loses the original fixture's detection:** the shifted base case
now has zero flags and zero moves, where E17's original metric had 44. Thus
this test establishes affine invariance of the implementation on a full-rank
fixture, but it does not show useful sensitivity or native100 performance.
Covariance whitening can change which deviations count as isolated. The
original 44 flags are not ground truth, so their disappearance alone does not
prove the new metric is worse. Astra recommends **holding native advancement**
until a predeclared generic audit measures null false discoveries and power
across fixed departure strengths, anisotropy, heavy tails, and legitimate
sparse components. Keep reference and calibration splits separate, include
rank-deficient cases, and record actual move decisions. Do not tune the metric
to recover these exact 44 flags. Then define behavior for ill-conditioned
feature covariance and test runtime at realistic feature width. An isotropic
ridge `C+lambda I` breaks full affine invariance; a pseudoinverse can ignore
query displacement outside the reference span. No E17 code or Claude run was
modified here.
