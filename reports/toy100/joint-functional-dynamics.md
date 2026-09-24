# Joint output geometry: two fixed-target optimizer experiments

Neither candidate qualifies. Both pass the 200-update warm filter, then fail
the unchanged 400-update cold trajectory gate. No cold ring, longer hold,
distribution shift or production test was run after those failures.

These methods already preserve the host's D-then-G order. The simultaneous
recorder issue identified in PR #82 does not explain their acquisition failures
or the earlier network-only functional metric's failure.

| Candidate | Warm checks / minimum HQ | Cold trajectory MSE | Passing cold checks |
| --- | --- | ---: | ---: |
| Joint G+prior functional displacement metric, output step .029 | 200/200 / .963379 | .0374133 | 0/24 |
| Output-coordinate RMSProp, output rate .00425, joint damped pullback | 200/200 / .901123 | .0419494 | 0/24 |

The cold limit remains MSE <= .02 with the original sustained suffix. The
first candidate plateaus around .036–.038 after update250; the second keeps
improving over much of the budget but has not acquired by400. Extra updates
would change the acquisition budget and cannot convert these into passes.

## Isolated question and methods

The earlier network-only metric passed warm hold but failed trajectory at
.037305. It left the prior's Adam displacement uncorrected. The first new
experiment therefore changes only the scope of the metric: differentiate
clean generated outputs jointly through network weights and learned particles.
For the current Adam metric P and joint output Jacobian J, solve

    (P^-1 + J.T J / (batch_size * .029)) delta = P^-1 delta_Adam.

The right side uses Adam's actual rounded proposal. Repeated particle draws
are deduplicated with exact square-root count weights. All G and prior
parameters move together; D keeps ordinary Adam. This structural correction
does not solve acquisition: its final error is almost unchanged.

The second experiment tests whether merely damping the parameter proposal
retains a poor output direction. It captures the actual training loss gradient
at each clean generator output, averages repeated observations of a particle,
and removes the batch averaging factor. Per-particle, per-coordinate second
moments use beta2=.999 and bias correction; the desired output step has fixed
rate .00425 and epsilon1e-8. It is lifted through the joint Jacobian as

    delta = P J.T (J P J.T + lambda I)^-1 desired,
    lambda = .001 * mean(diagonal(J P J.T)).

The implementation checks actual output displacement against the local linear
prediction and may halve the proposal up to12 times. This is a local model
check, not an evaluation-quality guard. In the warm run no halving was needed.
Mean accepted output RMS was .004916, compared with .004051 for the joint
metric. Neither controller reads mode centers, identity targets, quality
thresholds or elapsed training time. A zero output field yields a zero lifted
step. The output-moment state is new research state initialized when the
controller starts; it is not part of the inherited Adam state.

Both are small, deterministic, buffer-free generator experiments. Exact
particle input matching must be unambiguous; the RMSProp arm additionally
requires one conditioning row per particle. The output-only arm is evaluated
with the recipe's zero direct prior regularization. Exact Jacobians and dense
output solves are not a scalable implementation for native20k-particle tasks.

## Controls and evidence

Every warm experiment uses the same scheduled step1000 prefix, hash
`6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21`.
The unchanged child exactly matches the uninterrupted scheduled final state,
and each observation-only constant Adam child gets6/200 warm passes, ending
at6 modes / HQ .504639. No seed sweep was used.

G/D nominal rates stay .00425 and prior .0085; their Adam moments advance once
per update. The output-space optimizer adds one output second-moment update
per observed particle. Clean Jacobian calculations consume no training RNG.
Full cold trajectories take14.53 and14.05 seconds respectively on the pinned
single-thread CPU environment; those are host training times, not whole warm
driver runtimes.

Five focused tests check the joint solve against an independent primal solve,
observation-only parity over multiple updates, rejection of ambiguous latent
mapping before mutation, the damped lift's algebra and zero-field behavior,
and duplicate aggregation / optimizer moment / RNG accounting.

Raw JSON, declarations, exact source copies, host source snapshots and logs:

- [Joint metric warm](continuous-evidence/round3/joint-metric-warm/manifest.json)
- [Joint metric cold](continuous-evidence/round3/joint-metric-cold/manifest.json)
- [Output RMSProp warm](continuous-evidence/round3/output-rmsprop-warm/manifest.json)
- [Output RMSProp cold](continuous-evidence/round3/output-rmsprop-cold/manifest.json)

The manifests bind46 files with stored-byte and original-byte SHA256 hashes.
Drivers: `joint_functional_metric_probe.py`, `output_rmsprop_probe.py`.
Run warm with `--output NEW`, then cold with `--output NEW --cold-from WARM`.
Each driver refuses cold promotion unless the warm identity control and
candidate passed, then stops at the first failed full-budget host.

## Research context and recommendation

The literature was checked after inspecting the local failure. The updated
[Exact Gauss-Newton paper](https://arxiv.org/abs/2405.14402) motivates solving
small output-space systems; the
[Gauss-Newton geometry analysis](https://arxiv.org/abs/2412.14031) explains the
role of output geometry and damping. These are motivations, not convergence
guarantees for the present stochastic, nonlinear adversarial game. The
output-RMSProp/pullback combination here is an experimental construction,
not a reproduction of either published algorithm.

The May2026 [state-feedback GAN study](https://www.nature.com/articles/s44387-026-00120-3)
changes objectives and uses decaying regularization. It does not directly
supply the requested removal of LR decay while preserving this frozen recipe.

Do not run a gain grid or extend these failed acquisition budgets. The stronger
remaining lead is to preserve PR #82's alternating acquisition behavior and
verify proposed player steps against the same training batch.
