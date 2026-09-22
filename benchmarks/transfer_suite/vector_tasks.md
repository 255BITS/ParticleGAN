# Vector transfer task design

Eight development data tasks complement the original nine required regressions.
Six are ranking cases: broad separated components, unequal mixture masses,
unequal widths, covariance anisotropy, overlapping components, and a continuous
noisy spiral. Ranking failures affect comparison but do not veto a policy that
passes the required regressions. The deliberately narrow-density case and the
changing-unit case are diagnostic only, declared before reference runs. Neither
can change any ranking component. Each data family has one case and one weight.

The annulus family is reserved. Calibration never samples or evaluates it.
`run_episode` refuses reserved specs unless the caller explicitly passes
`allow_reserved=True` after policy selection is frozen. Unknown data families
also fail closed. The full numerical definitions, importance reasons, limitations
and fixed thresholds live in `TASKS` and `RESERVED` in `vector_tasks.py`.

Metrics are properties of the generated distribution:

- `sw1_normalized` is 32-projection sliced Wasserstein-1 divided by target RMS
  radius about its mean. Its bound is 0.18. Evaluation uses 4,096 samples and
  independent fixed generators; no evaluation draw changes a training stream.
- Identifiable separated mixtures additionally require target-mass TV ≤0.15,
  HQ ≥0.85, and mean relative component covariance error ≤0.85. HQ means squared
  Mahalanobis distance ≤9 under the assigned target component; a true 2D
  Gaussian has probability `1-exp(-9/2) ≈ 98.89%` in this ellipse. Gaussian
  widths and anisotropy therefore receive their own appropriate units.
- Component covariance error compares empirical covariance to the specified
  matrix using relative Frobenius norm. A center-only generator can have perfect
  HQ but covariance error 1, so it cannot pass by erasing all within-mode spread.
  Components with fewer than ten evaluation samples contribute error 1.
- The unequal-mass case compares occupancy to `[.55,.30,.13,.02]`; it never
  requests uniform occupancy. It also requires every component to receive at
  least 25% of its specified mass, preventing the rare component being silently
  erased by an absolute-TV tolerance.
- Every separated-mixture case also requires the minimum eigenvalue of every whitened
  empirical component covariance to be ≥0.15. A broad axis cannot conceal a
  collapsed narrow axis. This is a coarse covariance check, not a density proof.
- Overlapping mixtures and the spiral use only observable distribution metrics:
  SW1, normalized mean error ≤0.15, and relative covariance error ≤0.45. They
  have no component-label, occupancy or mode-count gate. Per-sample component
  membership is ambiguous for the overlapping task.

Bounds are predeclared regression tolerances, not statistical confidence levels
or post-search winner thresholds. Independent target draws must pass them, and
explicit wrong-mass/zero-spread controls must fail the corresponding metric.
Raw metrics remain available so that a marginal pass is visible.

Every episode trains the existing logistic RP loss, sample-point cap and learned
particle prior without particle L2. Defaults are 256 particles, G/D width 64,
two hidden layers, two critic Fourier bands, batch 128, Adam `(0,.99)`, LR .001,
D LR multiplier 1.5 and prior LR multiplier 10. Each spec can independently
declare architecture, update cadence (`d_every`/`g_every`), rates, regularizer and
budget. Seeds are fixed at 0; no seed search is performed.

Feedback receives current gradients after backward and before the optimizer
step. The last regularization action controls the next loss; evaluation metrics,
data-family names and thresholds never enter the controller. The fixed control
has no gradient feature extraction. All runs retain full actions and measured
controller overhead. Timing includes initialization and metrics and is not a
standalone claim of wall-clock speedup.

Exactly 24 live observations are spaced across the declared budget. Sustained
success requires the complete schedule and at least five passing final
observations; EMA is recorded separately and cannot rescue a live failure.
Changing-scale observations use the contemporaneous target, whose scale stops
changing at 60% of the budget. Updates between observation points are unmeasured.

Reference calibration evidence is recorded outside the repository under
`/tmp/pr36-transfer-vectors-*`, with the frozen task manifest, numerical source
hashes, all attempted controls, errors, raw observations and actions. Only fixed
cosine/constant reference controls or explicitly recorded simple numerical
reference settings may be calibrated; controller fitting is owned by the parent
comparison pipeline. No failing task is relabeled or removed after results.

Protocol v2 corrects a metric loophole found during review before controller
fitting: the mean covariance error alone allowed some components to collapse to
their centers while another component kept the average below its bound. The
minimum whitened eigenvalue bound originally applied only to the anisotropic
case; it now applies to every identifiable mixture. A partial-center-collapse
regression test demonstrates the loophole and its rejection. Task tiers, data,
training and all other bounds are unchanged. Original v1 sources and reference
results are retained exactly; v2 evidence explicitly re-scores their already
recorded per-component eigenvalue metrics without retraining.

## Fixed-reference evidence

The frozen seed-0 CPU reference uses the cosine schedule already implemented by
`FixedControl`. Under protocol v2 it passes 4/6 ranking cases at the last
checkpoint and sustains 3/6. Every episode completed all 24 observations. The
eight original episodes took 49.56 seconds combined, with individual costs
5.35–7.71 seconds including evaluation. These are local measurements, not
hardware-independent budgets.

| Development case | Tier | Final | Sustained | Passing final observations |
|---|---|---|---|---:|
| Broad separated | ranking | PASS | PASS | 19 |
| Unequal masses | ranking | FAIL | FAIL | 0 |
| Unequal widths | ranking | FAIL | FAIL | 0 |
| Anisotropic | ranking | PASS | PASS | 15 |
| Overlapping | ranking | PASS | FAIL | 4 |
| Narrow density | diagnostic | FAIL | FAIL | 0 |
| Changing scale | diagnostic | PASS | PASS | 22 |
| Noisy spiral | ranking | PASS | PASS | 23 |

The unequal-mass and unequal-width cases achieved final HQ of 0.950 and 0.971
but failed component covariance error (5.800 and 7.809, bound 0.85). Thus HQ
alone misses a substantial distribution error. Four additional permitted fixed
references—constant cap and cosine R1+R2 coefficient 0.1 on each of those two
cases—also failed; none was discarded. The overlap case illustrates why a
passing final checkpoint cannot substitute for five consecutive final passes.
These measured failures are retained as useful comparison cases. They are not
evidence that no formulation can solve them.

All twelve executions, source hashes, complete observations, EMA values, action
traces and errors are retained in `/tmp/pr36-transfer-vectors-20260922/`.
`v1_source/` preserves exact original source files; `v2_rescored/` contains the
corrected manifest, a source snapshot, target-oracle evidence and a readable
leaderboard. Every rescored row links its original JSON by SHA256 and records
separate execution/scoring protocols with `retrained: false`. All eight
independent target-sampler controls pass v2, and the partial-center-collapse
negative control fails. The reserved annulus remains unexamined.
