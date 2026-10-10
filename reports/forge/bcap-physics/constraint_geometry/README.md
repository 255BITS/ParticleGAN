# Constraint geometry after full DualNorm

**The measured candidate has 2 PASS, 3 FAIL and 1 prerequisite BLOCKED; the matched
control has 3 PASS and 3 FAIL.** Eleven actual attempts cost 431.571 paid worker seconds
against the 12,840-second reservation ceiling; no reservation remains.

The conditional repair fails: trajectory and residual identity errors slightly
worsen against the matched winner, while both conditional guardrails pass.
Gaussian smoke also regresses without any projection activation; this introduces
a numerical confound, described below. This is an
explicit mechanism diagnostic, with unchanged numerical task gates and no
ordinary-tier qualification or default-promotion claim.

## Question and mechanism

Can useful descent directions survive competing set-coverage forces under full
DualNorm? The original winner's saved trajectory endpoint has coverage/identity
network-gradient cosine −0.899 and combined first-order identity-MSE change
+0.00579, despite adversarial-only change −0.02044. Residual student's cosine is
−0.831, with identity change −0.000426 versus −0.00805 without coverage. These
original-source endpoint derivatives motivate the hypothesis; they are not new
control passes or evidence about earlier training.

Let the unchanged host loss be `L = A + w_C C + w_R R + w_Z Z`, where A is
adversarial, C set coverage, R the existing paired loss when present, and Z prior
regularization. The full-DualNorm winner produces an actual displacement `d0`
across generator and learned-prior parameters, including its per-group rates and
sampled-row restrictions. The candidate computes existing protected gradients
`a_j = ∇ L_j` at the same pre-update state and solves

```
d* = argmin_d 1/2 ||d - d0||²
     subject to a_jᵀ d <= 0 for each existing protected objective.
```

The rule protects adversarial loss everywhere and the existing paired loss when
one is present: residual MSE in residual student and paired cover in mid-scale.
Trajectory's set coverage remains auxiliary because it does not bind identities.
All task coefficients, architectures, data, prior and training steps stay fixed.
For one active normal, `d* = d0 - max(aᵀd0,0) a / ||a||²`. For two, deterministic
active-set enumeration solves the tiny Gram system with a pseudoinverse. Zero
normals impose no constraint. Projection happens **after** the full DualNorm
transformation and includes joint prior/network displacement; no subsequent
normalization can undo it. Unsampled prior coordinates are excluded from both
displacement and constraint normals. The zero displacement is always feasible;
opposed gradients may remove all useful movement along their shared axis.

This is a parameter-space first-order non-ascent guarantee, subject to floating
point tolerance. The report will measure the actually applied displacement after
parameter recomposition. It does not guarantee finite-step loss reduction,
paired-identity descent from critic descent, feasible strict descent, or global
GAN convergence. Critic BCAP and update timing remain unchanged.

## Primary literature and competing explanation

[Yu et al., Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
project conflicting task gradients before aggregation. This candidate shares the
halfspace geometry but projects the actual normalized displacement, with fixed
protected objectives and no randomized order. It is not unchanged PCGrad.
[Sener and Koltun, Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)
frame conflicts through Pareto optimization. Here we seek the nearest feasible
non-ascent displacement rather than a minimum-norm convex gradient combination.
Neither paper establishes performance for BCAP or these conditional hosts.

A competing explanation is that the critic's local signal is insufficient to
escape a structured wrong permutation, or that coverage remains useful for
acquisition. Non-ascent constraints can stall at a Pareto boundary, and finite
normalized steps can still overshoot. Unconditional Gaussian instability and
finite MoG allocation do not supply an independent paired objective, so this
mechanism is expected to have little influence there. An unchanged Gaussian
failure is an anticipated scope limitation, not a reason to splice another rate.

The prior whole-rate [pacing search](../../dualnorm-pacing-v2/README.md) and
[saved Gaussian diagnosis](../../gaussian1d-diagnosis/README.md) do not test this
post-normalization directional constraint. Newer source-bound magnitude studies
[PR319](https://github.com/255BITS/ParticleGAN/pull/319),
[PR320](https://github.com/255BITS/ParticleGAN/pull/320), and
[PR321](https://github.com/255BITS/ParticleGAN/pull/321) fail continuous Gaussian
learning with fixed prior/network magnitude caps, including past extrapolation.
We retain those failures as context and do not repeat or reuse their cohorts.

## Frozen experiment

The protected losses are existing training objectives, not evaluation identity MSE.
In particular, trajectory protects **adversarial loss only**: it has no paired-MSE
training term. Residual student protects adversarial loss and its existing masked
residual MSE; mid-scale protects adversarial loss and its existing paired cover.

One global candidate `constraint_geometry-nonascent-v1` differs from
`constraint_geometry-control-v1` only by `Recipe.constraint_geometry_mode=nonascent`.
The control resolves the exact original winner recipe: non-saturating loss, full
DualNorm momentum0/smoothing .001/per-offset convolutions, constant G/E .012,
D .018 and prior .030, BCAP coeff1/cap1/every update, zero additive training noise
and no EMA. The original configuration identity is
`bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`;
original revision `dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`,
executed-source digest `2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`.
Original outcomes 7/21 Tier2 remain archived, not credited to this matched source.

Both declarations use the same frozen current executed source and Python3.12 /
RTX A6000 compute cohort. Protocol seed0, public deterministic initializer,
named constructor/data/training-noise/evaluation streams, sampling and full task
contracts remain unchanged. Every consumed stream and optimizer constraint
history is checkpointed. Unsupported component hosts fail preflight, and an
enabled optimizer refuses a step without the protected-loss hook.

The diagnostic includes unchanged two_pole (80 updates, 300-second reservation),
Gaussian smoke (1,000/120), Gaussian stability (own passing-smoke prerequisite,
through 6,000/600), trajectory (400/1,800), residual student (400/1,800), and the
original passing mid-scale identity guardrail (800/1,800). Reservations total
6,420 seconds per recipe, **12,840 seconds combined**, below the 14,400 allowance.
The shared parent coordinator is the sole GPU launcher. No scientific retry,
seed search, continuation after failure or second candidate is authorized.

Numerical prediction: trajectory final identity MSE ≤ .02 and at least five
passing terminal checks; residual student retains its full MSE/success/wrong-pad
gates; mid-scale retains all concept and identity bounds. Falsifier: trajectory
endpoint MSE > .02, or failure of its sustained gate. Partial endpoint gains and
first-order protection alone do not establish a repair. Software controls test
intersecting/opposed halfspaces, disabled-path parity, exact winner resolution,
sampled-row ownership, and checkpoint/missing-hook rejection. Existing frozen
scorer controls and actual observed training media will be published on readout.

## Reproduction and logs

The following declarations refer to the **measured f909ec56e commit and frozen
Python 3.12.13/Torch 2.14.0/RTX A6000 cohort**. Use that commit or the certified
source snapshot for scientific replay. The current corrected source is UNMEASURED
and cannot reuse the admitted study ID to launch a different source.

```
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue plan constraint_geometry-nonascent-v1 --study constraint_geometry-candidate-study-v1 --show-boundaries
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue constraint_geometry-nonascent-v1 --study constraint_geometry-candidate-study-v1
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue constraint_geometry-control-v1 --study constraint_geometry-control-study-v1
```

Local plans/tests/progress: `/tmp/bcap-physics-20261009/constraint_geometry/`.
Coordinator tail: `tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/logs/coordinator.log`.
Bulk logs, observation arrays and state tensors stay outside Git. Compact final
metrics/provenance and genuine training GIFs will be committed here. The parent
maintains the sole current cross-track leaderboard.

## Completed conditional comparison

Both recipes finish the full 400-update conditional budgets. Every trajectory
and residual observation fails the full numerical gate: 0/24 passing checks and
zero passing terminal suffix for each recipe. Endpoint MSEs slightly worsen, and
residual student loses one additional correct landing.

| Unchanged task | Matched winner control | Projected candidate | Full verdict |
| --- | ---: | ---: | --- |
| Trajectory identity MSE ≤ .02 | .239861891 | .244410008 | Both FAIL, suffix 0 |
| Residual identity MSE ≤ .02 | .061036013 | .062594503 | Both FAIL, suffix 0 |
| Residual success =1 / wrong-pad =0 | .500000 / .500000 | .416667 / .583333 | Both FAIL |
| Mid-scale identity at .5 ≥ .85 | .981440306 | .992974672 | Both PASS; terminal suffix 22 vs 20 |
| Two-pole mean-absolute ≥ .3 / critic slope ≤1 | .958502471 / .952345729 | .958502471 / .952345729 | Both PASS; suffix 17 |

Mid-scale's first passing check moves from update 100 to 167 and its five-check
confirmation from 234 to 300. Its endpoint identity improves slightly while
acquisition slows. This is a retained passing guardrail, not a repaired failing
task or justification to relax the conditional gates.

| Candidate host | Updates requiring projection | Maximum protected derivative before | Maximum actually applied derivative after |
| --- | ---: | ---: | ---: |
| Trajectory: adversarial only |39/400 (9.75%)| .018213862 |1.154e-8|
| Residual: adversarial and paired residual |50/400 (12.50%)| .017362749 |5.271e-9|
| Mid-scale: adversarial and paired cover |50/800 (6.25%)| .002422742 |8.047e-9|
| Two-pole: adversarial only |0/80|0|0|

These are parameter-space dot products, with the original gradient magnitudes;
rows with two protected objectives record the maximum across either one. They
are numerical mechanism measurements, not identity-MSE derivatives for
trajectory. Rounding leaves small positive residuals after recomposing float32
parameters; exact non-ascent is not claimed beyond this tolerance. The preserved
original-motion fractions are step counts, not accepted displacement norms.
Per-step accepted norms and constraint multipliers were not instrumented; the
compact results report actual RMS prediction movement between scored states.

Residual's existing training MSE applies to all 12 rows in this cohort. Yet its
identity MSE increases across 9/23 scored intervals, versus 8/23 for the control.
Trajectory increases across 12/23, versus 11/23. These are multi-update interval
measurements, not a per-step finite-loss audit. They show that the method does
not establish sustained paired progress. A constraint can remove ascent while
leaving tangent motion, insufficient correction, or finite-step curvature error.
The local geometric hypothesis is exercised but its proposed identity repair is
falsified.

Stop this exact revision. Before another trainer candidate, inspect when the
wrong permutation forms in saved training observations and distinguish weak
critic progress from finite-step loss increases. An explicit paired-identity
training term for trajectory would require a separately declared task variant,
with its original evidence identity retained; it cannot silently supply a
fixed-task trainer win. No further seed, coefficient, rate or budget search is
launched here.


## Unconditional limitation and numerical confound

The candidate Gaussian smoke completes all 1,000 updates and passes 0/24 paired
confirmation checks. Its final KS is .081634954, mean error .134887158 target
sigmas and width ratio 1.158333628. The matched winner smoke passes its acquisition
gate even though its endpoint KS .071842162 fails: acquisition is a confirmed
passing **scheduled state**, not a terminal-only gate. No candidate stability run
is eligible, and its cell stays BLOCKED instead of receiving a substitute state.

Gaussian projection activates 0/1,000 times. With no auxiliary prior objective,
the protected adversarial loss is already the total generator loss, so the
mathematical cone rule has no mechanism for repairing Gaussian instability.
The implemented enabled path nevertheless recomposes parameters as
`old + (updated - old)` after every step, including unchanged projections, and
performs an extra protected-gradient backward. It therefore lacks a proven
bitwise inactive-path identity. The [analytic arithmetic witness](roundtrip-witness.json)
shows a 2.55549e-9 float32 difference for one near-zero value, with zero updates or
samples. This is a demonstration that such drift is possible, not a causal audit
of every consumed training step. The observed smoke regression cannot be credited
to projected competing gradients when none were projected. A future implementation
would need exact inactive-path parity before a new source-bound comparison;
the software correction below adds no scientific training or qualification.


## Published software correction is UNMEASURED

Parent review requested correction of the verified inactive-step arithmetic bug.
The published optimizer now leaves already-applied tensors untouched whenever
projection is unchanged. Its checkpoint schema is 2, explicitly rejecting the
measured schema 1 state rather than silently continuing under different semantics.
A near-zero float32 optimizer fixture proves exact bitwise equality with plain
DualNorm and proves that the old recomposition loses a real value; approximate
`allclose` would miss the bug. These are software checks only.

**All scientific measurements in this report use the original f909ec56e source,
digest 43391f7415970f12e3818d76b95a4562bdc81570f5bd6c4e36fc54ad6d63ce8f.**
The corrected implementation has no trained quality measurements. The no-op
confound affects **all** attribution, including the conditional comparison:
those regressions compare the complete trained implementation, not projection
alone. Projection activation and actual finite-precision protected derivatives
remain valid observations of that source. No old receipt, verdict or numerical
qualification is regraded. Reproduction must use the pinned measured commit or
its certified Forge source snapshot; running the corrected source is a new cohort
and needs its own separately authorized study.


## Final artifacts and gates

[Complete compact metrics](results.json), [certificate projections](receipts.json)
and [provenance](provenance.json) bind every result to its original source/runtime,
recipe, initializer, sampling law, data digest and checkpointed streams. Matched
initialization and Gaussian data digests agree; every active recipe field agrees
except the declared mode. Two-pole retains its explicitly declared stored critic /
zero-particle fixture. It is a separate task cohort, not a substituted learned
initializer. Mid-scale retains its declared zero residual fixture under public
initialization. No comparison pools these with the learned Gaussian/trajectory
cohorts.

Control Gaussian smoke confirms at 375 (3/24 scheduled passing observations),
then completes all 1,000 updates. Its own stability continuation completes through
6,000, passes only 2/72 stationary checks, misses shifted reacquisition and passes
0/24 shifted hold checks. Final KS .320622894, mean error .255238362 sigmas and
width ratio .662330337 all fail. The candidate's prerequisite BLOCKED cell spends
zero stability updates. There are no retries, INCOMPLETE or INVALID attempts.
The finite campaign adds 10,360 total training updates across 11 actual attempts.

The original oracle and cyclic-permutation scorer controls are explicit
**target-informed scoring fixtures**. For both conditional scorers, oracle MSE 0
passes while wrong-identity MSE .125356406 fails. Residual oracle success 1 /
wrong-pad 0 passes; permutation success 0 / wrong-pad 1 fails. These controls do
not alter the learned baseline, initialization or declared gates. Saved endpoint
metrics recompute from the exact retained prediction panels.

Eleven [actual-training GIFs and renderer receipts](media/index.json) illustrate
both desired behavior and observed outputs. No image selection is based on grade,
and publication adds zero updates or sampling draws. Examples:

| Task | Candidate actual training | Matched control actual training |
| --- | --- | --- |
| Trajectory | [GIF](media/candidate-trajectory.gif) | [GIF](media/control-trajectory.gif) |
| Residual | [GIF](media/candidate-residual_student.gif) | [GIF](media/control-residual_student.gif) |
| Mid-scale guardrail | [GIF](media/candidate-mid_scale_identity.gif) | [GIF](media/control-mid_scale_identity.gif) |
| Two-pole anchor | [GIF](media/candidate-two_pole.gif) | [GIF](media/control-two_pole.gif) |
| Gaussian smoke | [GIF](media/candidate-gaussian1d_smoke.gif) | [GIF](media/control-gaussian1d_smoke.gif) |
| Gaussian stability | Own prerequisite BLOCKED | [GIF](media/control-gaussian1d_stability.gif) |

Bulk arrays, per-update events, stdout and checkpoints remain in
`/mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/constraint_geometry-round-v1/`.
The compact provenance retains exact certificate and artifact hashes and paths.
[Candidate readout](../../records/readout-35c64b46eedef0831b3a6aac.json) and
[control readout](../../records/readout-ac82943db25e6134866495ce.json) conclude the
frozen studies without default-promotion credit or historical regrading.

Validation: the measured source passes 200 focused checks; after the requested
inactive-path correction, 201 geometry, checkpoint, bitwise parity, convolution,
task-binding and Forge-study checks pass. Forge validates both declarations;
final memory compilation uses summaries-only and retains existing qualification
snapshots. Sharing/contention is explicit, and worker seconds are accounting,
not a speed claim. The parent keeps the single cross-track leaderboard.

## Corrected-source follow-up

[Round two](round2/README.md) measures corrected schema2 in a separate,
source-frozen matched comparison. It confirms inactive numerical parity and
finishes 3 PASS / 3 FAIL for both arms, with no new sustained identity pass.
The schema1 results and UNMEASURED status of the correction at this report's
original publication remain historical evidence; round-two receipts carry the
new measured identity and do not retroactively qualify schema1.

[Round three](round3/README.md) adds strict common descent with finite same-batch
acceptance as a substantive successor. Its matched schema2 comparison adds
sustained trajectory and residual passes, retains both guardrails and inactive
parity, and still fails Gaussian stability. Its new source/recipe receipts carry
that scoped result; earlier nonascent conclusions remain unchanged.
