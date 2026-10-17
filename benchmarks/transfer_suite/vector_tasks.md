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
- The anisotropic (protocol v3), unequal-width and unequal-mass (protocol v4)
  cases score shape on each component's core instead: assigned
  samples with squared Mahalanobis distance ≤16 (within 4σ). It requires mean
  relative core covariance error ≤0.5, minimum whitened core eigenvalue ≥0.15,
  and `max_component_spill` ≤0.05, the largest per-component fraction of assigned
  samples beyond 3σ (≈1.1% for a true Gaussian). Components with fewer than ten
  (core) samples score core error 1, eigenvalue 0 and spill 1. Their SW1, TV, HQ
  (and unequal-mass `min_mass_ratio`) bounds are unchanged; the whole-component
  covariance bounds no longer gate them.
- Protocol v5 adds a particle-resolution floor to those three cases. A
  component is under-resolved when its declared share of the particle table,
  `masses[k] * particles`, is below `PARTICLE_FLOOR = 32`. The rule reads only
  the spec, never the run. The gated shape/spill statistics are
  `resolved_core_covariance_error` (mean), `resolved_core_min_eigen_ratio` (min)
  and `resolved_max_component_spill` (max), taken over resolved components only.
  Per-component values are still reported for every component, and
  `component_resolved` flags the exempt ones. The v4 all-component aggregates
  are also still reported. An exempt component stays gated by mass (`mass_tv`,
  `min_mass_ratio`) and the global `hq`. `resolve()` rejects a spec that gates
  on the resolved statistics when either no component is resolved or an exempt
  component has no `min_mass_ratio` bound. The empty-set values (0, 1, 0) can
  therefore appear only in reports, never in a verdict. With the default 256
  particles only the unequal-mass 2% component (≈5 particles) is exempt.
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

Protocol v3 (anisotropic core metric) changes only the anisotropic case. Its
target covariances have Frobenius norm ≈0.09, so a whole-component covariance
error let 1–8% of stray samples 1–2 units from a mean dominate the score: it
measured spill, not shape (for example 4.61 overall but 0.11 within 4σ). Shape is
now scored on the 4σ core and stray mass is bounded explicitly by
`max_component_spill`. The old whole-component metrics are still reported for
every identifiable mixture and still gate the other cases. Recorded historical
results keep the thresholds stored with them. Evidence:
[anisotropic core metric report](../../reports/transfer_suite/anisotropic_core_metric/README.md).

Protocol v4 (core metric for unequal widths and masses) applies the same six
core/spill bounds to the unequal-width and unequal-mass cases; unequal mass keeps
`min_mass_ratio` ≥0.25. Their v2 covariance failures were also spill: the
reference LeakyReLU critic scored 7.81 whole-component but 0.18 core error on
unequal width, and 5.80 vs 0.31 on unequal mass. Both cases still fail under v4,
now on the cause: 5.5% spill from the narrowest component and 42% spill around
the rare 2% component. Broad-separated, narrow and changing-scale cases keep the
v2 whole-component bounds. Recorded historical results keep the thresholds
stored with them; the v2 table below is not rescored. Evidence:
[v4 section of the core metric report](../../reports/transfer_suite/anisotropic_core_metric/README.md#protocol-v4-unequal-width-and-unequal-mass).

Protocol v5 (particle-resolution floor) adds the spec-only floor described
above to the three core/spill cases. The generator's output is one atom per
particle. At 256 particles the unequal-mass 2% component gets about 5 atoms, and
a perfect sampler with 5 atoms fails the core covariance bound 78% of the time
and the core eigenvalue bound 30% of the time. The v4 verdict on that component
therefore measured particle count. `PARTICLE_FLOOR = 32` is the smallest atom
count on a grid declared in advance at which a finite-atom oracle's
per-observation false-fail rate is ≤5% for every gated statistic and every
declared covariance shape. Core error is the binding statistic; eigenvalue
alone would allow 10. No GAN run was used to choose it. Anisotropic and
unequal-width values are unchanged (all their components have ≥64 particles).
Only the unequal-mass verdict inputs change, and its remaining failures under
the reference critics are marginal spill from the 55% component. Recorded
historical results, including the v4 tables, keep the thresholds stored with
them; `CORE_SPILL_BOUNDS_V4` preserves the v4 declaration. Evidence:
[protocol v5 section of the rare-collapse report](../../reports/transfer_suite/silu_rare_collapse/README.md#protocol-v5-particle-resolution-floor).

## Fixed-reference evidence

The frozen seed-0 CPU reference uses the cosine schedule already implemented by
`FixedControl`. The table below is protocol v2 evidence. Under protocol v2 it passes 4/6 ranking cases at the last
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
negative control fails. Calibration leaves the reserved annulus unexamined;
the later [frozen comparison](../../reports/transfer_suite/README.md) reports its transfer results.
