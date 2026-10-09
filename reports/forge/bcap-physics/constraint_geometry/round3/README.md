# Strict common descent with finite realization

This prospective round-three diagnostic tests one substantive successor to
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
stream and all optimizer/progress state. Unsupported noisy/standardized replay
bindings and stochastic/running-buffer scalar generators fail closed.

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

Both source-frozen arms will be enqueued before a single public Queue/drain
worker runs on GPU0 with authorized sharing and no full-compile callback.
Bulk logs, events, checkpoints and tensors remain outside Git. Easy tail:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/constraint_geometry/logs/drain.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/constraint_geometry/queue/constraint_geometry-round3-v1/ATTEMPT_ID/run.log
```

Final reports will retain exact measured source, complete verdicts, progress
counters, same-batch decrease totals, provenance and actual-training GIFs.
Publication will add zero optimizer updates or sampling draws. The parent owns
the one current cross-track goal leaderboard.
