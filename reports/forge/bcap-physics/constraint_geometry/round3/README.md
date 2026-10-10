# Strict common descent with finite realization

Strict common descent with finite realization adds **two sustained conditional
passes**: trajectory and residual student now pass every required endpoint bound
and their complete sustained gates. The candidate finishes **5 PASS / 1 FAIL**
against corrected schema2's **3 PASS / 3 FAIL**, with passing guardrails retained.
Gaussian continuous stability remains unresolved. Retain this as a scoped opt-in
repair; this diagnostic grants no default adoption or ordinary Tier2 qualification.

This completed round-three diagnostic tests one substantive successor to
[corrected schema2 nonascent](../round2/README.md): bounded common descent plus
same-batch Armijo acceptance on originally conflicting updates. Corrected
schema2 is the primary matched control. The exact BCAP winner and earlier
rounds remain archived context, not a third causal arm or independent replication.

## Evidence and hypothesis

Round two cleanly enforced protected rounded derivatives below 1.2e-8, yet
trajectory retained the four-row wrong-identity cycle and residual correct
landings worsened from six to five. Boundary feasibility did not establish
useful progression. Residual already has an informative paired loss, while
trajectory has only an adversarial protected loss and no paired-MSE training
term. These saved diagnostics motivate testing realized progress on the losses
that actually exist. No identity oracle is added.

The hypothesis is that a boundary projection can lose descent in the protected
normal direction, and that first-order descent can also fail at finite step
size. A bounded common-descent component addresses the first issue; evaluating
the same protected objectives on the same batch addresses the second.
Competing explanations are opposed protected gradients, loss of useful cover
motion, changing-critic overfitting, and inadequate identity information in the
trajectory objective. Finite minibatch descent need not solve any full task.

This construction draws on
[MGDA's common-descent criterion](https://mgda.inria.fr/mgda),
[Fliege and Svaiter's multicriteria steepest descent](https://research.birmingham.ac.uk/en/publications/steepest-descent-methods-of-multicriteria-optimization/),
and [Sener and Koltun's multiobjective formulation](https://arxiv.org/abs/1810.04650).
The constrained-motion analogy is a boundary retraction with an inward component
and an observed acceptance check. This stochastic, changing-critic experiment
does not inherit a deterministic optimization convergence guarantee.

## One global mechanism

Let `d` be the actual rounded full-DualNorm displacement and `a_j = grad L_j`
the already available protected loss gradients. Apply the successor only when
at least one `a_j · d > 0`, the same original conflict condition as schema2.
Otherwise retain the already-updated tensors bitwise and perform zero probes.

For nonzero normalized gradients `n_j = a_j / ||a_j||`, define

`c = mean_j n_j`, `q = -||d|| c / ||c||`,

`p = (project_nonascent(d, a) + q) / 2`.

With at most two unit normals, `c` is their minimum-norm convex combination.
If `||c|| <= 1e-12`, record a Pareto stall and retain the original parameters.
Otherwise `q` is a strict common-descent direction; the half blend preserves
some feasible tangential motion, with ideal Euclidean norm at most `||d||`.
No second normalization, changed objective coefficient or task-specific rate
is applied.

Try scales `alpha = 1, 1/2, ..., 1/256`, at most nine trials. After actual
parameter rounding, accept only if every nonzero protected gradient has
`a_j · applied < 0` and every protected objective satisfies

`L_j(theta + applied) <= L_j(theta) + 1e-4 * a_j · applied`.

The evaluator reuses the fixed critic, real logits, training inputs, row mask,
sampled latent IDs and within-component perturbation. It makes no sampler call,
new data draw or optimizer update. Replay must match the original loss, and
ambient RNG consumption fails closed. If all nine trials fail, restore the
original parameters. The base optimizer and training/data/RNG clocks advance
once, including rejected or stalled updates. Additional forwards are disclosed
as finite probes, not free optimization updates or new evaluations of the gate.

The public Recipe delta is `constraint_geometry_mode: nonascent -> strict_progress`.
The public optimizer is a subclass of corrected schema2; its checkpoint retains
the schema2 base state plus `strict_progress` schema1 counters and pending loss
values. A pending checkpoint requires rebinding its caller-owned deterministic
evaluator before stepping. Completed-update checkpoints include every consumed
stream and all optimizer/progress state. Forge rejects noisy/standardized replay
bindings; the scalar trainer rejects BatchNorm/Dropout and probes check ambient
RNG consumption. Caller-owned evaluators must remain pure. These deterministic
hosts, rather than arbitrary custom forwards, are the trained applicability scope.

Protected losses remain adversarial only for trajectory, Gaussian and two-pole;
adversarial plus the existing active paired residual for residual student;
adversarial plus existing paired cover for mid-scale. No trajectory paired loss,
new supervision, target retargeting or scorer-driven update is introduced.

All other settings retain the saved winner's non-saturating loss, full DualNorm,
momentum0, smoothing .001, per-offset convolution, constant G/E .012, D .018,
prior .030, cap/coefficient1 every update, zero additive train noise and live
scoring. The control adds the previously measured schema2 projection to that
exact saved recipe. Current preset defaults do not define either arm.

## Frozen question, gates and budget

The primary numerical forecast is final residual `identity_mse <= .02`;
`identity_mse > .02` falsifies the proposed repair. Full scientific success also
requires success rate 1, wrong-pad rate 0, and five terminal complete passes.
Trajectory's unchanged MSE <= .02 gate and five-check suffix test whether the
adversarial protected signal transfers useful identity progression. Mid-scale
must retain all concept/identity bounds with a five-check suffix. Inactive
Gaussian and fixed two-pole numeric outputs must match their controls.

The six tasks and [complete original numerical gates](../round2/README.md)
are unchanged: two-pole, Gaussian smoke and its own checkpoint-dependent
stability, trajectory, residual student and mid-scale identity. Acquisition is
not retention; smoke requires an independently confirmed same-state pass,
whereas stability requires all 72 stationary checks, deadline reacquisition
and all 24 shifted hold checks. Failed own smoke leaves stability BLOCKED.

Protocol seed0, public deterministic initializer, per-task architecture, data/
target, seen batches, priors/widths/weights, sampling, update allowance and
evaluation cadence are fixed. Constructor, data, noise and evaluation streams
are isolated and checkpointed. The frozen Forge Gaussian/vector adapter reuses
the same real tensor for D and G. Two-pole's identity/zero/stored-weight law is
an explicit separate fixture cohort. The numerical scorer, not image viewing,
determines every task result; saved oracle/permutation controls check identity
sensitivity and GIFs illustrate actual training.

New ready round-three studies admit one candidate and one matched schema2
control, reserving 6,420 seconds each in campaign `constraint_geometry-round3-v1`:
12,840 scientific worker seconds within the authorized 14,400-second track
ceiling, leaving 1,560 for bounded software/state audits or execution repair.
No sweep, seed repeat, extra candidate, ordinary Tier2 qualification or default
adoption is authorized by this diagnostic.

## Validation and execution

The focused software suite passes 213 checks: CPU/CUDA × float32/float64 gradient,
full RNG, inactive parameter/base optimizer state and checkpoint parity; strict
finite descent with nonlinear overshoot requiring backtracking; opposed-gradient
stall; pending-evaluator restore; near-zero cancellation; and fail-closed RNG/
replay restrictions. Software fixtures are separate from scientific task passes.

Both source-frozen arms were enqueued before a single public Queue/drain
worker ran on GPU0 with authorized sharing and no full-compile callback.
Bulk logs, events, checkpoints and tensors remain outside Git. Easy tail:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/constraint_geometry/logs/drain.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/constraint_geometry/queue/constraint_geometry-round3-v1/ATTEMPT_ID/run.log
```

## Complete numerical readout

All 12 attempts finish without retry, BLOCKED, INCOMPLETE or INVALID evidence.
Every admitted training step and evaluation completes. Both Gaussian stability
runs resume their own smoke state and add 5,000 steps to its 1,000-step prefix;
the pair consumes 15,360 paired training steps across all six tasks.

| Task | Strict progress candidate | Matched corrected schema2 control |
| --- | --- | --- |
| Trajectory | **PASS**, MSE **.00025847997**, 19/24 passing checks and terminal suffix **19**; first pass 100, confirmation 167 | FAIL, MSE **.2437363565**, 0/24, suffix 0 |
| Residual student | **PASS**, MSE **.00036959240**, success **1**, wrong-pad rate **0**, 21/24 passes and suffix **21**; first pass 67, confirmation 134 | FAIL, MSE **.0626398921**, success **5/12**, wrong **7/12**, 0/24, suffix 0 |
| Mid-scale identity | **PASS**, 21/24 checks, suffix **20**, five-check confirmation 300, mid identity **.98635834** | PASS, 20/24, suffix **20**, confirmation 300, mid identity **.99295571** |
| Two-pole explicit fixture | **PASS**, suffix **17**, mean absolute position **.95850247**, median gradient **.95234573** | Identical PASS, numerical curve and scored outputs |
| Gaussian smoke | **PASS**, 3/24 scheduled passes, first independent confirmation 375; endpoint KS **.07184216** | Identical PASS and observations |
| Gaussian stability | **FAIL**, stationary **2/72**, deadline FAIL, shifted hold **0/24**; endpoint KS **.32062289** | Identical FAIL and observations |

The residual forecast is observed and the complete original landing/MSE gate
passes, not just the forecast's scalar threshold. Trajectory also passes its
full gate without any new paired training objective. Every saved endpoint row
is closest to its own target in both conditional candidate tasks: **12/12**,
versus **8/12** and **5/12** in their controls. The trajectory control retains
the wrong cycle **2 -> 5 -> 8 -> 11 -> 2**; the candidate resolves it.

Mid-scale's early pass at 67 is interrupted before the sustained run begins at
167; sustained confirmation remains 300 in both arms. Its slightly lower mid
identity stays above .85, and all other existing concept/magnitude/identity
bounds pass. The fixture and smoke are useful guardrails, not substitutes for
continuous Gaussian learning. Both stability endpoints have normalized mean
error **.25523836** and width ratio **.66233034**, in addition to failing KS.
This six-task tally does not replace the ordinary Tier2 denominator.

## What realized progress establishes

| Candidate task | Conflicts / steps; accepted | Probe evaluations | Backtracks | Smallest accepted scale | Mean retained displacement norm / base norm |
| --- | --- | --- | --- | --- | --- |
| Trajectory | **12/400; 12** | **24** | **0** | **1** | **.70179** |
| Residual student | **186/400; 186** | **997** | **625** | **1/64** | **.11814** |
| Mid-scale identity | **45/800; 45** | **152** | **62** | **1/8** | **.46824** |
| Two-pole | **0/80** | **0** | **0** | Inactive | Inactive |
| Gaussian smoke | **0/1000** | **0** | **0** | Inactive | Inactive |
| Gaussian stability, including prefix | **0/6000** | **0** | **0** | Inactive | Inactive |

All **243** conflict steps accept a strict finite common descent update; trained
rejection and Pareto-stall counts are zero. The **1,173** probe evaluations include
243 baseline replay checks and 930 trial checks; they add no optimizer steps or
sampling draws. Maximum accepted Armijo violation is zero. Actual applied
rounded derivatives never have a positive maximum in the candidate; its
maximum retained norm ratios are **.70708 / .73533 / .93440** for trajectory,
residual and mid-scale, all below one. Each count comes from the final public
optimizer checkpoint, not inferred from images or endpoint metrics.

Same-batch protected loss decreases sum to **1.06403** adversarial for trajectory;
**.383408 adversarial / .122690 paired** for residual; and
**.189609 adversarial / .049317 paired cover** for mid-scale. These sum accepted
comparisons against each update's fixed critic/batch; they are not net objective
changes across the changing game or population guarantees.

Trajectory accepts all 12 conflict directions at full scale. Its improvement
therefore supports restoring strict protected descent beyond a boundary
projection; backtracking was never exercised there. Residual requires 625
reductions before accepting its informative paired/adversarial compromise,
showing that finite realization matters operationally in that task. This one
package comparison does not isolate the necessity of the direction blend and
acceptance rule separately. No extra ablation arm was admitted.

Scheduled prediction RMS motion is **.06773584 versus .05952680** for trajectory
and **.04314799 versus .05381611** for residual (candidate versus control).
Trajectory moves more over those intervals and solves identity; residual moves
less while preserving paired progress and solving landing. MSE increases in
**8/23 versus 7/23** trajectory intervals and **6/23 versus 9/23** residual
intervals. Interval counts include many inactive updates and changing critics;
they are not per-step acceptance certificates or a general oscillation metric.

Saved scorer controls independently reject wrong identities: oracle MSE **0**
passes, one-row permutation MSE **.12535641** fails; residual oracle success/wrong
rates **1/0** versus permutation **0/1**. These target-informed controls use only
saved panels and are neither trained models nor alternate initializers.

## Source, parity and retained artifacts

Measured source: commit **13450f4cae33748240ca5425814f3ff6fcc1bc41**, digest
`49084004acda1634f89ab66f027e521e42d54305e0722e39182c436488f85f7e`.
Implementation, declarations, theory and forecasts were pushed before enqueue.
Both arms have the same complete source/runtime/task/init/data bindings; only
the declared mode differs. Later commits publish saved evidence only.

[Inactive parity audit](inactive-trained-parity.json) proves bitwise final model
tensors, base optimizer state, consumed named streams, complete observations
and scored samples on two-pole and both Gaussian tasks. All six task pairs end
with identical consumed named stream states and zero unintended RNG deviations.
Unconsumed ambient CPU/CUDA globals differ across worker processes but remain
unchanged from each run's certified initial to final state. Those states and
original whole-checkpoint/confirmation hashes are preserved and reported
separately; declared recipe/progress metadata is not numeric parity evidence.

Campaign `constraint_geometry-round3-v1` charges **305.479838 worker seconds**,
with zero reservations remaining, within the 12,840-second scientific allowance
and 14,400-second authorized track ceiling including ancillary audit allowance.
All 12 workers finish and the bounded drain exits. GPU sharing/contended time
supplies accounting, not an optimizer speed claim.

Requests: candidate `32e39cb8205edc5e330a6826`, control
`b6d279fc2405feff3311c071`. Exact candidate/control revisions, all task gates,
progress counters and per-attempt identities are in [results.json](results.json).
[Compact receipts](receipts.json), [provenance](provenance.json) and
[validation](validation.json) retain original certificates, artifact hashes,
runtime/recipe/init/stream bindings and local checkpoint paths. The
[candidate readout](../../../records/readout-568ca0dd3d3cdf1221ae4605.json) and
[control readout](../../../records/readout-c31ec37dd154ae5d9b9bbf74.json) conclude
both frozen studies. Compilation is summaries-only; archived qualifications and
all prior forecasts/results remain unchanged.

## Actual-training media and recommendation

| Task | Strict progress actual training | Matched schema2 actual training |
| --- | --- | --- |
| Trajectory | [GIF](media/candidate-trajectory.gif) | [GIF](media/control-trajectory.gif) |
| Residual student | [GIF](media/candidate-residual_student.gif) | [GIF](media/control-residual_student.gif) |
| Mid-scale identity | [GIF](media/candidate-mid_scale_identity.gif) | [GIF](media/control-mid_scale_identity.gif) |
| Two-pole explicit fixture | [GIF](media/candidate-two_pole.gif) | [GIF](media/control-two_pole.gif) |
| Gaussian smoke | [GIF](media/candidate-gaussian1d_smoke.gif) | [GIF](media/control-gaussian1d_smoke.gif) |
| Gaussian stability | [GIF](media/candidate-gaussian1d_stability.gif) | [GIF](media/control-gaussian1d_stability.gif) |

The [media index](media/index.json) verifies 12 actual-training GIFs with uniformly
selected scored states and their desired target/behavior. Complete observations
are retained; grade determines no image selection. Export with
[publish.py](publish.py) and [audit_saved_parity.py](audit_saved_parity.py), using
only certified saved bytes. No publication update, sampling draw or task regrade
is added. Bulk logs/checkpoints/events stay outside Git. For the two new passes:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/constraint_geometry/queue/constraint_geometry-round3-v1/d3550f211f2a44238fc2000b58da93bf/run.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/constraint_geometry/queue/constraint_geometry-round3-v1/88479a53c6094731b36406059a7e01aa/run.log
```

Retain **strict_progress as a scoped opt-in conditional repair**, without a
default change or ordinary Tier2 qualification. The exact winner remains the
broader archived reference. Gaussian continuous learning is still a full FAIL;
no noisy/standardized/custom-forward, native-law, image or broad-vector transfer
qualification follows. Current screening profiles remain provisional. Future
transfer or component isolation needs a separately authorized, source-bound
comparison; do not silently combine this with other repairs or run another seed.
The parent owns the one current cross-track goal leaderboard.
