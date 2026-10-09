# Corrected constraint geometry: round two

Corrected schema2 nonascent projection adds no sustained pass: both arms finish
with 3 PASS and 3 FAIL in this six-task diagnostic. It fixes the inactive-update
confound and preserves both passing guardrails, but trajectory and residual
identity remain failures. Retain the winner and stop unchanged nonascent work.

This completed mechanism diagnostic tests corrected schema2 nonascent
projection against the exact saved BCAP winner. The [round-one report](../README.md)
and its measured schema1 identities remain intact: all round-one causal claims
are confounded by float32 recomposition of inactive updates. The correction
skips that copy when the projected displacement equals the already-applied
displacement. Round two uses new declarations and a fresh matched source cohort.

## Theory and admitted comparison

Let `d` be the actual rounded displacement after the ordinary full-DualNorm
generator/prior step, and `a_j = grad L_j` the gradient of each existing
protected objective. The candidate applies

`d* = argmin_v ||v - d||² / 2, subject to a_j · v <= 0`.

At most two constraints admit an explicit active-set solution in float64.
If every constraint is already satisfied, schema2 retains the updated
parameter tensors directly. Projection is after normalization and rate scaling;
there is no subsequent normalization. This guarantees first-order nonascent
up to numerical rounding. It guarantees neither strict progress nor finite-step
descent, and adversarial protection supplies no missing identity supervision.

The physical analogy is a feasible displacement against unilateral constraints.
The method is related to [PCGrad](https://arxiv.org/abs/2001.06782), which removes
conflicting gradient components, and
[multiobjective descent](https://arxiv.org/abs/1810.04650). This experiment uses
the nearest feasible *actual optimizer displacement*; it does not claim the
convergence results of another algorithm.

The saved winner diagnostics show coverage/paired gradients with network
cosines -0.899 for trajectory and -0.831 for residual student. The old source's
normalized trajectory step increased paired MSE to first order while its
adversarial-only step decreased it. Those archived diagnostics motivate the
mechanism; they are not matched round-two evidence. Round one's inactive
Gaussian smoke nevertheless changed its outcome, exposing the implementation
confound before any scientific interpretation could be made.

The only trainer delta is `Recipe.constraint_geometry_mode = nonascent`, with
schema2 inactive-step preservation. Both arms retain non-saturating loss,
zero-momentum full DualNorm, smoothing .001, per-offset convolution,
G/E .012, D .018, prior .030, constant learning rates, cap/coefficient 1 every
update, zero additive training noise, and live scoring without EMA. The control
resolves the saved declaration of
`bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`;
mutable preset defaults do not define the comparison.

Protected objectives are the existing adversarial loss for trajectory,
Gaussian and two-pole; adversarial plus the existing active paired residual
loss for residual student; adversarial plus the existing paired cover loss
for mid-scale identity. Trajectory has no paired-MSE training loss. No target
oracle, changed coverage coefficient, auxiliary supervision, alternate sampler,
or changed critic is introduced.

The six unchanged tasks are two-pole, Gaussian smoke and its own dependent
stability continuation, trajectory, residual student and mid-scale identity.
Task architecture, target/data law, seen batches, prior and widths, sampling,
update allowance, evaluation cadence and complete gates are frozen. Seed 0
uses the public deterministic initializer, with explicit constructor/data/
training-noise/evaluation streams checkpointed. The Forge Gaussian/vector
adapter reuses the same real tensor for D and G. The two-pole identity/zero/
stored-weight fixture remains an explicit separate task cohort.

## Forecasts and falsification, frozen before training

The study's numerical forecast is final trajectory `identity_mse <= .02`;
`identity_mse > .02` falsifies the proposed repair. Scientific success still
requires the full existing five-state sustained gate. Residual student must
satisfy all existing identity/landing conditions, and mid-scale must retain
its sustained passing guardrail. A zero-projection Gaussian or two-pole run
should match its control numerically; divergence would reopen the confound.

Competing explanations are an identity-ambiguous protected adversarial loss,
loss of useful motion through projection, and first-order feasibility without
finite-step progress. Saved prediction changes and identity-MSE changes will
be reported as intervals between scheduled evaluations, not per-step descent
certificates. Scorer controls will use target oracle and wrong-row permutation
solely to validate metric sensitivity.

Both schema-v3 candidates had ready studies with new round-two IDs before
training. Each reserves 6,420 worker seconds; campaign
`constraint_geometry-round2-v1` reserves 12,840 total against the authorized
14,400-second track ceiling. One candidate, one control, no seed repeats,
sweeps, second candidate, default adoption or ordinary Tier2 qualification.
Gaussian stability remains BLOCKED if its own smoke prerequisite fails.

## Software audit and execution

The software fixture in `tests/test_constraint_geometry.py` uses public
deterministic initialization, explicit sampled prior rows, and two optimizer
steps around a checkpoint round trip. CPU/CUDA × float32/float64 pass numerical
bitwise comparisons for protected-backward gradient parity, unchanged Python/
NumPy/CPU/CUDA RNG state, inactive parameter tensors, base optimizer state,
and restored schema2 state. The near-zero cancellation regression separately
proves that unconditional recomposition is observably wrong. These are software
fixtures, not trained task passes or alternate initialization cohorts.

One public Queue/drain worker executes both source-frozen arms on GPU 0 with
authorized sharing and no full-compile completion callback. Raw stdout,
per-update streams, checkpoints and tensors stay outside Git under
`/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/constraint_geometry`.
Easy-tail commands:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/constraint_geometry/logs/drain.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round2-20261009/constraint_geometry/queue/constraint_geometry-round2-v1/ATTEMPT_ID/run.log
```

## Complete numerical readout

All 12 attempts completed, with zero retries, BLOCKED, INCOMPLETE or INVALID
results. Both arms ran every permitted update and scheduled evaluation.
Gaussian stability continued its own smoke checkpoint, adding 5,000 updates
to the 1,000-update prefix. The pair consumed 15,360 training updates in total.

| Task | Corrected projection | Exact winner control | Complete gate |
| --- | --- | --- | --- |
| Trajectory | FAIL; MSE **.24373636**, 0/24 passes, suffix 0 | FAIL; MSE **.23986189**, 0/24, suffix 0 | MSE <= .02; five terminal passing checks |
| Residual student | FAIL; MSE **.06263989**, own landings **5/12**, wrong rate **7/12**, 0/24, suffix 0 | FAIL; MSE **.06103601**, own **6/12**, wrong **6/12**, 0/24, suffix 0 | MSE <= .02, success >= 1, wrong <= 0; five terminal passing checks |
| Mid-scale identity | PASS; suffix **20**, first pass 167, five-check confirmation 300, mid identity **.99295571** | PASS; suffix **22**, first pass 100, confirmation 234, mid identity **.98144031** | Both concept cosines >= .85, magnitudes [.75,1.25], identity at 0/mid >= .85; five terminal passing checks |
| Two-pole explicit fixture | PASS; suffix **17**, mean absolute position **.95850247**, median gradient **.95234573** | Identical PASS and metrics | Position >= .3, gradient <= 1; five terminal passing checks |
| Gaussian smoke | PASS; **3/24** scheduled passes, first confirmed 375; endpoint KS **.07184216** | Identical PASS and every numerical observation | Any scheduled complete pass with independent same-state confirmation |
| Gaussian stability | FAIL; stationary **2/72**, deadline FAIL, shifted hold **0/24**; endpoint KS **.32062289** | Identical FAIL and every numerical observation | All 72 stationary checks, five-terminal deadline reacquisition, all 24 shifted hold checks |

Both Gaussian gates require at least 4,096 samples, finite fraction 1, normalized
mean error <= .2, width ratio [.8,1.2], and KS <= .05 at each required check.
The smoke endpoint can fail while the acquisition gate passes; no endpoint
substitution is made. Stability ends at normalized mean error **.25523836** and
width ratio **.66233034** in both arms. Acquisition does not establish retention.
This diagnostic tally is not the ordinary Tier2 denominator or qualification.

## Mechanism diagnosis

| Candidate task | Projected / consumed steps | Maximum positive protected derivative before | After actual applied rounding |
| --- | --- | --- | --- |
| Trajectory | **44/400** | .02028830 | 9.67e-9 |
| Residual student | **84/400** | .01736269 | 1.14e-8 |
| Mid-scale identity | **50/800** | .00242274 | 9.27e-9 |
| Two-pole | **0/80** | 0 | 0 |
| Gaussian smoke | **0/1000** | 0 | 0 |
| Gaussian stability, including prefix | **0/6000** | 0 | 0 |

The implemented first-order protection operates where intended, with residual
positive derivatives below 1.2e-8 after float32 parameter rounding. Nevertheless
the trajectory forecast is falsified by more than an order of magnitude. Both
arms retain the exact wrong-row cycle **2 -> 5 -> 8 -> 11 -> 2**, with the other
eight rows closest to their own target. Residual ends with five own rows in the
candidate versus six in the control. Projection does not repair allocation.

Trajectory protects only the existing scalar adversarial objective, which
offers no per-row identity guarantee and no paired-MSE training term. Residual
does protect an existing paired residual objective that directly carries
identity information. Its failure shows that missing paired information alone
cannot explain both failures: nonascent of an informative gradient is still
weaker than useful finite-step progress. The archived winner critic diagnostics
suggested an identity-directed signal, but that old-source result is motivation
and does not certify the round-two critic's endpoint field.

Between the 24 scheduled conditional observations, mean prediction RMS movement
is **.05952680 versus .06089882** for trajectory and **.05381611 versus .05656722**
for residual (candidate versus control), reductions of approximately 2.3% and
4.9%. Trajectory MSE increases in **7/23 versus 11/23** intervals; residual in
**9/23 versus 8/23**. The candidate moves slightly less and still has worse final
identity. These multi-update intervals neither measure per-step retained norm
nor prove that projection removes useful progression on a particular step.
Strict-progress, finite-step curvature and identity ambiguity remain competing
explanations; this experiment does not distinguish them completely.

Target-informed scorer controls use only saved panels, with zero new draws or
updates. Both tasks' oracle has MSE **0** and passes; a one-row target permutation
has MSE **.12535641** and fails. Residual oracle success/wrong rates are **1/0**,
versus **0/1** for the permutation. Thus the metric rejects wrong identities;
these controls are not trained models or replacement initialization cohorts.

## Inactive parity, provenance and publication

[Saved parity audit](inactive-trained-parity.json) verifies exact equality of
final model tensors, role parameters, base optimizer state, every consumed named
RNG stream, all numerical observations and retained scored sample arrays for
two-pole and both Gaussian tasks. The Gaussian checkpoints additionally retain
ambient CPU/CUDA global RNG states that differ across worker processes. Both
states are unchanged from each run's certified initial checkpoint to its final
checkpoint: training binds the isolated named model stream inside `fork_rng`
and restores the ambient state. These unconsumed states are reported separately,
not silently reset. Original full-checkpoint and same-state-confirmation hashes
retain their original bytes; recipe/mode/statistics metadata makes cross-arm
whole-checkpoint hash equality inappropriate. Named streams also match for all
three active task pairs. All published task RNG audits report zero deviations.

The exact measured source is commit
`57aad8ea35fbd6989265db53af9ff39dff70b821`, digest
`b87904dcae3e2df70006c04b51d3fd65d89979a72c7f5f1e841ae684517f7685`.
Both arms bind this source and identical runtime/task contracts. The new
declarations, audit tests and theory were pushed before enqueue. Later commits
publish saved evidence only; no later optimizer implementation is being credited
with these measurements.

Candidate revision:
`118421df70cf5acb4d28c10994400067112be3825cefa42049bb20d93263a139`.
Control revision:
`b4957b01e002fcc51b3b52b1e327894d6d44b7cef03b4a2e4f59aa30e3101b64`.
Requests: candidate `a23561c3ed3914582a12aec0`, control
`5d7c25f3104e880885434fac`. Campaign charged **355.184536 worker seconds**,
with **zero outstanding reservations**, within its 12,840-second ceiling.
The remaining 1,560 seconds under the 14,400-second track ceiling cover bounded
software/checkpoint audits and any execution repair; no scientific retry or
additional candidate was run. Sharing and contention were authorized; cost is
accounting and is not an optimizer speed claim.

[Results](results.json), [compact certified receipts](receipts.json),
[provenance](provenance.json), [validation](validation.json), and
[GIF index](media/index.json) retain gates, complete source/runtime/recipe/init
bindings, checkpoint/stream hashes, original certificate identities and local
artifact paths. Reproduce the saved-only exports with
[publish.py](publish.py) and [audit_saved_parity.py](audit_saved_parity.py).
No publication step adds an optimizer update, sampler draw or new task grade.

| Task | Corrected projection actual training | Matched control actual training |
| --- | --- | --- |
| Trajectory | [GIF](media/candidate-trajectory.gif) | [GIF](media/control-trajectory.gif) |
| Residual student | [GIF](media/candidate-residual_student.gif) | [GIF](media/control-residual_student.gif) |
| Mid-scale identity | [GIF](media/candidate-mid_scale_identity.gif) | [GIF](media/control-mid_scale_identity.gif) |
| Two-pole explicit fixture | [GIF](media/candidate-two_pole.gif) | [GIF](media/control-two_pole.gif) |
| Gaussian smoke | [GIF](media/candidate-gaussian1d_smoke.gif) | [GIF](media/control-gaussian1d_smoke.gif) |
| Gaussian stability | [GIF](media/candidate-gaussian1d_stability.gif) | [GIF](media/control-gaussian1d_stability.gif) |

Every GIF uses actual saved training outputs with uniformly selected scheduled
states, paired with the task goal. Original observation curves remain complete;
no images are selected by grade. Bulk stdout, metric events, checkpoints and
tensors remain in the local queue. The compact audit/report/GIF sources are
committed, and the parent owns the one current cross-track goal leaderboard.

The focused software suite passes **205 checks**; Forge validates the new
declarations. Readouts conclude both frozen studies using summaries-only memory
compilation while preserving all archived qualification snapshots. Recommend
**no adoption, no Tier2 qualification, and no unchanged nonascent rerun**. A
future explicitly authorized mechanism should test useful strict progress on
existing identity-bearing losses without adding hidden oracle supervision.

Frozen study conclusions: [candidate readout](../../../records/readout-99157305f41ca0b884ae5c08.json)
and [control readout](../../../records/readout-f9ca0b7a952b308fb000ed29.json).
