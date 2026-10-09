# Corrected constraint geometry: round two

This prospective mechanism diagnostic tests corrected schema2 nonascent
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

The candidate and control studies are ready schema-v3 candidates with new
round-two study IDs. Each reserves 6,420 worker seconds; campaign
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

Final numerical readout, exact source receipts and actual-training GIFs will
be published from saved evidence without additional optimizer updates or
sampling draws. The parent owns the one current goal leaderboard.
