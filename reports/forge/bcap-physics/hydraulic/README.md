# Hydraulic output travel: bounded mechanism study

The travel bound nearly retains the Gaussian and improves native precision, but
**the candidate still fails the complete Gaussian and native gates**. Both
recipes retain the broad-vector pass, and both pass 2/4 matched executed tasks.
The native precision prediction is observed (.49586 versus .24072), while native
covariance trace bias worsens. Stop this exact revision as a global repair;
retain its positive-spacing capability for research. This explicitly scoped
diagnostic supplies no ordinary-tier qualification or public default change.

The candidate applies one global joint-output travel rule to the exact winning
BCAP recipe. A pressure analogy motivates measuring displacement after mechanical
leverage: useful forces can produce excessive output motion when normalized
layer updates add coherently. It does not imply an optimizer conservation law.

For parameters θ, sampled prior locations z and existing normalized proposal v,
write the first-order output displacement as
`u_i = J_θ G(z_i) v_θ + J_z G(z_i) v_z,i`.
Its mean is shared network/prior motion; the residual describes individual motion.
Coherent layer terms amplify `||sum_l J_l v_l||`, even when every layer has a
bounded parameter step.  Gradient direction and realized travel are distinct.

The proposed rule computes only from the consumed generator-side real batch:
`r = median_i min_(j != i) ||x_i - x_j||`, with global coefficient 1.
It proposes the original joint G/prior update, measures
`d(s) = sqrt(mean_i ||G_(θ+s vθ)(z_i+s vz,i+ε_i) - G_θ(z_i+ε_i)||²)`,
then uses `s=min(1,r/d(1))` and at most six further halvings until `d(s)<=r`.
If the bounded replay cannot satisfy the constraint, it rejects parameter motion.
Zero median spacing raises an explicit error before the G/prior step; D has
already stepped in the public trainer. This is not a successful zero-motion
update.  Repeated noiseless examples and finite-atom data can have zero spacing,
so this exact implementation requires positive-spacing batches and a stateless,
deterministic G. Image, conditional and finite-atom transfer are untested.

The same latent indices and Gaussian jitters are replayed; there are no extra RNG
draws, critic updates, evaluations, target centers or quality-aware decisions.
Unsampled prior rows stay fixed. Zero-momentum optimizer counters advance once.
The nominal constant G/E/D/prior steps remain .012/.012/.018/.030; loss,
BCAP cap/coefficient 1/every-update, smoothing .001, per-offset convolution,
no input/output noise, no EMA and task-owned settings remain the winner's.

The implementation observes proposed network-only and additional prior motion,
shared-mean energy fraction, accepted travel, backtracking scale and first-order
parameter work `-sum_p grad_p · accepted_delta_p`. These are training diagnostics,
not measures of objective improvement. The direction can still be wrong.

A training-batch output trust bound is related to function-space damping and
Gauss-Newton trust geometry discussed by [Martens (2020)](https://jmlr.org/papers/v21/17-678.html).
It is a scalar ray projection, without curvature inversion or natural-gradient
claim. [TRPO](https://proceedings.mlr.press/v37/schulman15.html) constrains policy
distribution change; this study bounds empirical sample displacement and inherits
none of its improvement guarantees. [TrasMuon (2026)](https://arxiv.org/abs/2602.13498)
restores magnitude control to orthogonalized optimizers through RMS calibration
and relative energy clipping. Here the measured quantity is joint generator/prior
output motion, including cross-layer coherence, and the radius comes from current
training-data spacing rather than gradient energy or an absolute spectral scale.

## Prior evidence, prediction and falsifier

The [original saved-state analysis](../../bcap-tier2-search/FAILURE_ANALYSIS.md)
finds native full-step median motion 7.26–7.92 target sigmas and 51–66% shared
motion energy. Half-step probes improve a geometric surrogate but do not select
an optimal rate. Gaussian width oscillation and distorted fitted-normal shape
also rule out a universal permanent-contraction account.
[Earlier pacing](../../dualnorm-pacing-v2/README.md) found task tradeoffs.
[PR319](https://github.com/255BITS/ParticleGAN/pull/319),
[PR320](https://github.com/255BITS/ParticleGAN/pull/320) and
[PR321](https://github.com/255BITS/ParticleGAN/pull/321) respectively test fixed
prior cap .001, network spectral cap .1 and their combination with several timing
laws. All fail strict Gaussian continuous learning; ring regresses. Those distinct
source/prior/loss/timing cohorts are motivation, never current control credit.
This candidate measures actual joint output displacement and changes neither the
winner's pressure field nor D, rather than repeating those fixed gradient caps.

Prespecified prediction: grid100 final precision at least .48 (approximately twice
the archived .24072 endpoint), while the matched vector_two_broad guardrail passes.
The complete native gate still requires precision >=.97, all 100 genuine modes,
full density/shape/accuracy bounds and five terminal passing checks.  All original
Gaussian gates and its own passing-smoke checkpoint prerequisite remain intact.
The quantitative grid falsifier is final precision <.48. Losing the broad-vector
pass also rejects the candidate as a global repair. Lower motion alone is not success.

Competing explanation: the batch-spacing radius is too small for timely initial
transport; limiting motion may leave wrong mass allocation, distorted shape and
critic oscillation untouched. A sample mean bound does not bound every individual
sample or unseen latent, and spacing depends on batch size/data dimension.

## Frozen scope and allowance

| Unchanged task | Updates | Full reservation per recipe | Role |
| --- | ---: | ---: | --- |
| two_pole |80|300 sec|Shared anchor; candidate explicitly BLOCKED on public_components |
| gaussian1d_smoke |1,000|120 sec|Shared acquisition anchor |
| gaussian1d_stability |6,000 total|600 sec|Own passing smoke prerequisite; retention/shift failure |
| grid100 |7,000|3,600 sec|Native overshoot failure |
| vector_two_broad |1,200|1,800 sec|Passing regression guardrail |

At most one substantive candidate plus one matched winner control: 12,840 sec
maximum full job reservation, within the 14,400 sec ceiling including software
checks.  The candidate's two-pole blocker is explicit: its caller-owned component
API cannot replay joint output travel. There is no substituted initializer or
silently inactive limiter. Independent diagnostic peers run; Gaussian continuation
runs only after its own full smoke passes.  Zero scientific retries, new seeds,
extra candidates, gate relaxation or promotion runs are allowed.

Both recipes execute in the same frozen source/runtime cohort, protocol seed 0,
public deterministic initializer and isolated/checkpointed constructor, data,
training-noise and evaluation streams. The saved original winner at revision
`dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`
/source digest `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`
remains archived original evidence, not a matched current control.

All submissions use the dedicated shared queue; the parent owns the only drain.
Tail `/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/logs/coordinator.log`.
Compact task tables will be published here; the parent maintains the one current
cross-track comparison. Bulk logs, checkpoints and event streams remain local.

The control study uses the winner's original non-saturating parent declaration
only as a supported admission reference. That parent is untrained and receives
no credit. The actual experiment contains exactly the hydraulic candidate and
the winning BCAP control, with the same frozen scientific source digest.

## Final numerical readout

[Compact final metrics](results.json), [provenance receipts](provenance.json) and
[actual-training GIF index](media/index.json) bind these measurements. No image
inspection participates in scoring.

| Unchanged task | Hydraulic | Matched winner | Exact main readout |
| --- | --- | --- | --- |
|two_pole|BLOCKED, no attempt|PASS|Component host cannot replay joint output travel|
|gaussian1d_smoke|PASS|PASS|First confirmed 834 vs 375; endpoint KS .0355133 vs .0718422|
|gaussian1d_stability|FAIL|FAIL|Stationary 71/72 vs 2/72; shifted hold 23/24 vs 0/24|
|grid100|FAIL|FAIL|100k holdout precision .49586 vs .24072; terminal accuracy 0/5 each|
|vector_two_broad|PASS|PASS|Passing terminal suffix 21/24 vs 22/24|

The four measured trainer comparisons have identical numerical pass counts
(2/4). Two-pole is an additional control-only anchor, not a candidate FAIL or
a matched superiority cell. Archived passes provide no current credit.

The candidate's Gaussian smoke first confirms at 834 versus 375 for the control,
so bounded output motion slows acquisition. It ends smoke with KS .0355133
versus.0718422. Stability restores each actual update 1000 endpoint exactly,
including optimizer and consumed-stream history; no best checkpoint substitutes
for that endpoint. The candidate starts from an instantaneous passing endpoint,
while the control's endpoint already violates the CDF bound.

The candidate passes 71/72 stationary checks and 23/24 shifted hold checks, versus
2/72 and 0/24 for the winner. Its deadline reacquisition passes, and the frozen
pre-shift copy passes 0/48 checks, supporting renewed learning after the mean
shift. Both complete-learner verdicts remain FAIL. Candidate failed hold states:

| Update/phase | KS | Margin above .05 | Width ratio | Mean error / sigma | Failed bound |
| --- | ---: | ---: | ---: | ---: | --- |
|3959 stationary|.0681017568|.0181017568|1.05168292|.16063972|CDF KS only|
|5125 shifted hold|.0540219022|.0040219022|1.01242655|.11926717|CDF KS only|

No rerun, larger evaluation draw, selected state or relaxed threshold replaces
these failures. Final candidate Gaussian KS .0151893 versus control.3206229 is
endpoint evidence, not a substitute for retained quality.

The broad-vector guardrail passes for both recipes. Candidate terminal passing
suffix 21/24, normalized sliced Wasserstein.118326, massTV .074463, quality.987793,
component covariance error .230946 and minimum eigen ratio .578556 all pass.
Control covariance error .385581 and minimum eigen ratio .400066 also pass.

The grid candidate's 100,000-draw holdout precision .49586 exceeds the prespecified
.48 prediction, while all five terminal accuracy checks fail. Center RMS .379340
exceeds.2, covariance trace bias+.894514 exceeds absolute.1, radialKS .333687
exceeds.04 and massTV .0735 exceeds.06. Full coverage requires precision .97 and
all 100 genuine modes; improved centering or aggregate precision cannot relax it.

Over 6000 Gaussian updates, the rule limits every proposal, with zero rejected
motions: mean proposed joint RMS .090524, accepted RMS .003602 and radius.005104.
Over 7000 native updates, those means are.245865, .008405 and.011909. Native
proposed network-only RMS averages.237872 and additional prior RMS .048817;
55.57% of proposed joint travel energy is shared mean displacement. These are
sequential network/prior contributions, including nonlinear interaction, not
independent causal ablations. All accepted replays meet their actual nonlinear
RMS bound (maximum native ratio .99999984, Gaussian.99999225). The limit reduces
travel; it supplies no promise of correct covariance or stable CDF shape.

The matched grid control exactly reproduces the original holdout precision
.24072 and center RMS 1.5203146. Candidate center RMS falls to.3793398 and radial
KS to.3336866 from .4908830; massTV falls to.0735 from .14718. Covariance trace
bias increases to+.894514 from+.371385. Thus the mechanism improves placement
and genuine precision while failing density shape and complete mode coverage.
The observed quantitative prediction is narrower than the unchanged full gate.

Higher served nearest-assigned covariance can reflect cross-cell truncation or
spill, so that metric alone cannot establish wider uncensored latent kernels.
A separate [saved-center Jacobian diagnostic](native-width .json) restores both
current trained G/prior states in CPU FP64 and evaluates all 20000 latent centers
without drawing samples or updating tensors. It computes local covariance
`.025² J(z) J(z)ᵀ`; median trace relative to the target total variance is
**4.618359 for hydraulic versus 2.976159 for the matched winner** (means 4.919280
versus 3.073333). The fraction above 1 is 98.665% versus 97.560%. This independently
supports excessive local generator leverage after the travel repair, beyond
center placement. It is a linearized diagnostic, excluding finite-jitter
activation crossings/nonlinearity and component-assignment effects; it neither
replaces served covariance nor proves the whole failure is caused by width.
Independent reverse-mode derivatives agree within 2.67e-15, restored checkpoint
digests and global RNG remain unchanged, and analysis costs 2.037 seconds.
The retained native oracle passes (precision .98914, center RMS .041738, radial
KS .001514), and frozen scorer/destructive-control software checks pass.

[Descriptive analysis of the two saved misses](gaussian-misses.json) exactly
reproduces their target KS using the existing 4,096 arrays. After fitting each
array's own mean/std, normal KS is .0182848 at 3959 and .0079143 at 5125. Passing
explicit moment bounds therefore does not establish nonnormal shape as the
cause of the target-law KS misses: permitted mean/width offsets also affect KS.
This analysis draws no new samples and leaves the original gates and arrays
unchanged; fitted-normal KS is not an alternative qualification metric.

## Disposition, provenance and reproduction

Keep the winning public recipe and all original qualification snapshots. Stop
this exact coefficient 1 revision as a global repair: it supplies no new full
gate pass, misses two strict Gaussian checks, worsens native covariance bias
and cannot execute two-pole. The improvement supports output-motion leverage
as a useful mechanism hypothesis, not a universal optimizer law. The saved-state diagnostics above separate the residual native width problem
from improved placement and the Gaussian moment drift from nonnormal shape.
The next research question should distinguish shared translation from spatial
Jacobian/differential spread control; simply increasing training or changing
the radius is not authorized by this readout. No image/conditional transfer,
seed robustness, scientific default adoption or extra candidate is claimed.

Both measured arms use source digest
`937cf7c08cea42796b95d33620092a309b179691c015dd49ac6a21f165354d0f`,
Python 3.12.13/Torch 2.14.0 on RTX A6000, seed 0, one Torch thread and deterministic
public initialization on the four matched learned tasks. The additional
two-pole control preserves its separately declared stored critic weights and
zero direct coordinates; it is a fixed-fixture task, never a substituted
initializer for a learned task. Candidate source-origin commit is `534b35f10be84500746a71e9707a6e0ead9844a7`;
control is `2957c118d2ba50ad8c0bf8138a020c4281a356d1`. Those commits differ
only in admission declarations outside the frozen scientific source. Four
matched-parity receipts verify initial G/D/prior hashes, recipes, priors, task
declarations, original Gaussian data-sequence digests, and every non-evaluation
consumed-stream seed/initial/final state. Vector/native training sequence parity
is bound by the identical task law, seed, deterministic draws/update count and
identical end-state data streams; no unrecorded batch digest is invented.
All consumed streams remain in local full provenance checkpoints. Clean live
sampling and the actual training prior law remain task-owned and unchanged.

Nine workers complete 28,480 new host updates with no scientific retries or
incomplete executions. The [Forge study readout](../../records/readout-bd78af3e5a0d3194f1692b77.json)
records the prediction as observed but the overall decision as INCOMPLETE,
because the declared two-pole candidate cell remains unsupported/unmeasured.
That capability blocker is distinct from the two measured numerical FAILs.
All authorized runnable work is finished; no additional experiment can fill
the missing cell without changing the mechanism/public host binding. Paid worker cost is **1661.774794 seconds**, versus the
**14,400-second ceiling**; full allowances of launched jobs sum to 12,540 seconds
and the prespecified two-arm maximum was 12,840. Final live reservation is zero.
Software checks and saved-array/GIF publication are separate; they add no
qualification credit. Shared GPU contention prevents speed ranking.

Validation: 154 focused public API/checkpoint/training/ownership/scorer checks
pass, including nonlinear replay bounds, exact checkpoint continuation, unchanged
consumed RNG states, malformed-checkpoint refusal and zero-spacing refusal.
Bulk stdout, per-update telemetry and checkpoints remain under the dedicated
queue; only compact results/receipts, reproduction sources and nine saved-training
GIFs are committed. The parent owns the one current cross-track comparison.

Inspect/render only the already completed experiment, without training:

```sh
.venv/bin/python reports/forge/bcap-physics/hydraulic/publish.py \
  --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue --media
.venv/bin/python reports/forge/bcap-physics/hydraulic/gaussian_miss_analysis.py
.venv/bin/python reports/forge/bcap-physics/hydraulic/native_width_analysis.py
.venv/bin/python -m experiments.forge compile --summaries-only
```

To reproduce training, use the frozen scientific source and both ready study
declarations with the same public Forge enqueue workflow. A shared coordinator
owns execution; every task keeps its full allowance and the Gaussian continuation
requires its own passing smoke. Logs are easy to tail:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/logs/coordinator.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/events.jsonl
```
