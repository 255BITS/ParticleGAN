# Round four: projection and local-v2 transport composition

**BOTH retains transport local-v2's sustained unequal-mass repair and broad guardrail,
but Gaussian retention worsens and its conditional repair remains unsupported.**
Projection alone reproduces its two sustained conditional repairs on this same
source. This completed four-arm diagnostic grants no ordinary qualification,
calibration acceptance or public-default promotion. [PR368](https://github.com/255BITS/ParticleGAN/pull/368)
is stacked on the original projection research branch and remains unmerged.

## Complete sustained results

| Original task | Exact winner | Strict projection | Local-v2 transport | BOTH |
| --- | --- | --- | --- | --- |
| Two-pole fixed fixture | PASS | PASS | BLOCKED | BLOCKED |
| Gaussian smoke | PASS | PASS | PASS | PASS |
| Gaussian own stability | FAIL | FAIL | FAIL | FAIL |
| Trajectory | FAIL | **PASS** | BLOCKED | BLOCKED |
| Residual student | FAIL | **PASS** | BLOCKED | BLOCKED |
| Mid-scale guardrail | PASS | PASS | BLOCKED | BLOCKED |
| Unequal mass | FAIL | FAIL | **PASS** | **PASS** |
| Two-broad guardrail | PASS | PASS | PASS | PASS |
| Declared subset totals | **4 PASS / 4 FAIL** | **6 PASS / 2 FAIL** | **3 PASS / 1 FAIL / 4 BLOCKED** | **3 PASS / 1 FAIL / 4 BLOCKED** |

[Final metrics and sustained graders](results.json), [compact certificates](receipts.json),
[matched scientific conditions](provenance.json), [freeze receipt](freeze.json),
and [saved singleton/inactive parity](saved-parity.json) retain exact identities.
These totals are a scoped arm comparison, with differing applicability. They do
not form a full-suite ranking or a pooled conditional/vector winner. The archived
winner retains its original 7/21 Tier2 result in the parent-owned current inventory.

**The rare-density repair survives composition.** Unequal-mass full covariance
error is 3.691653 for winner/projection, .522577 for local-v2 and **.350241** for
BOTH, under the unchanged <=.85 gate. Local-v2 passes 13/24 checks with terminal
suffix 5; BOTH passes **18/24 with suffix 9**. Minimum component eigen ratio improves
.366567→.581130, and rare spill falls .140351→.126984. Endpoint mass TV slightly
worsens .016426→.018867 and minimum mass ratio .899564→.877028: the gain is not
universal. Both broad arms pass 24/24, suffix24; full covariance .226564→.190535
improves while mass TV .000732→.005127 worsens within the passing bounds.

**The conditional singleton remains exact; conditional composition is unmeasured.**
Winner trajectory/residual MSEs .239862/.061036 become **.000258480/.000369592**
with projection; terminal suffixes are 19/21 versus 0/0, and every residual row is
correct with zero wrong-pad rate. Mid-scale and the fixed two-pole fixture remain
PASS. All six projection tasks reproduce their archived trained numeric states,
observation curves and outputs bitwise. Transport's four supported tasks likewise
reproduce the archived local-v2 states and curves. Five new winner/projection
inactive cohorts are bitwise equal after separating optimizer-class provenance
and unused ambient RNG from consumed numerical state. These are software/parity
proofs, not independent statistical replications or qualification reuse.

**Gaussian retention regresses relative to transport alone.** Stationary passing
checks fall **28/72→8/72**, shifted hold **11/24→1/24**, and final KS
**.060653→.241544**; deadline reacquisition FAILs for both. Winner and projection
remain exactly 2/72 stationary, 0/24 shifted, KS .320623. Smoke requires any
independently confirmed scheduled passing state while completing all 1,000 updates.
BOTH has only **2/24 confirmed states** (459,875), versus13/24 for local-v2; its
terminal KS .093780 also fails .05. A smoke PASS therefore does not certify the
endpoint or the complete retention gate. Every stability job restores its own
full 1,000-update smoke state and streams under the unchanged continuation law.

**Projection is active in BOTH.** Unequal mass has 66 conflicting proposals,
52 reductions and minimum accepted scale1/4; broad has 199 conflicts and 103
reductions. Gaussian's final cumulative 6,000-update counter has 1,407 conflicts
and 151 reductions, including 195 conflicts/14 reductions in its smoke prefix.
There are **1672 unique accepted conflicts**,306 reductions and3650 finite
probes across the combined runs, with zero rejections/stalls and zero accepted
Armijo violation. Do not count the restored smoke prefix again. These checks
protect the existing adversarial loss, not transport or the composite loss.
The result differs substantively from failed transport-v3's full-composite ray
backtracking; it does not isolate projection's direction blend from finite
acceptance or prove population density/retention convergence.

**Recommendation:** retain strict projection as the demonstrated conditional
repair and local-v2 as the better Gaussian/rare-density reference. Preserve BOTH
as a supported vector composition lead with a longer rare passing suffix and
worse Gaussian retention. Do not replace either singleton globally or claim that
BOTH preserves each repair: trajectory/residual/mid-scale lack a transport consumer
and remain BLOCKED for this exact recipe. Testing that union needs a separately
authorized host-consumer comparison. Unequal width, native grids, images and the
full ordinary suite are unmeasured here. Stop this campaign; no sweep, extra
candidate, seed run, continuation, merge or promotion follows.

## Motivation and mechanism

[Projection PR361](https://github.com/255BITS/ParticleGAN/pull/361) repairs trajectory
and residual under their full gates; the corrected nonascent primary control does
not. [Transport PR358 round2](https://github.com/255BITS/ParticleGAN/blob/f9f6419ac251033e9076f7ba4bab87e715d321bd/reports/forge/bcap-physics/kinetic_transport/round2/README.md)
repairs sustained unequal mass, whereas its round3 backtracking successor loses
that pass. These are distinct source/recipe cohorts. [Original saved evidence](prior-evidence.json)
records inspected checkpoint hashes and the original report identities; it grants
no matched new-source credit.

All arms retain nonsaturating loss, full DualNorm momentum0, smoothing .001,
per-offset convolution layout, G/E .012, D .018, prior .030, constant floors1,
BCAP coefficient1/cap1/every update, zero extra prior regularization, additive
noise and EMA. The historical bare BCAP preset is resolved with every winning
override rather than mistaken for the winning trained configuration.

The projection switch is exactly `constraint_geometry_mode=strict_progress`.
For a rounded proposal d conflicting with an existing protected gradient a,
it blends the nonascent projection with bounded common descent and backtracks
on those same protected losses using the existing finite Armijo check. It is
inactive bitwise when no original conflict occurs. Its implementation is unchanged
from trained source `13450f4cae33748240ca5425814f3ff6fcc1bc41`.

Transport uses the exact public module from trained source
`5c2a64682b8900c5b42de94a6c27502e41d500f2`: normalized empirical sliced W2 with
32 deterministic directions, plus the relative real-anchor feature residual,
each global weight1. Fourth-neighbor real widths and scales1/2/4 are detached.
For real-anchor features phi, L_local = mean[(E_fake phi - E_real phi)^2 /
(E_real phi)^2]. No labels, evaluator centers/sigma, new draws or serving changes
enter. The finite feature form has a positive semidefinite kernel interpretation;
it is not a characteristic population guarantee. See
[Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html) and
[Li et al.](https://proceedings.mlr.press/v37/li15.html) for MMD and differentiable
generator moment matching. [Sener and Koltun](https://arxiv.org/abs/1810.04650)
provide the multiobjective context for gradient conflict. These sources do not
prove this stochastic GAN composition's density or retention gates.

BOTH adds transport to the generator/prior total objective while retaining
projection's original protected losses. Scalar hosts protect adversarial loss
only; transport is not silently made a second protected loss. In adversarial-only
singletons, projection has no additional coverage objective. The G-phase uses the
same fake and real tensors already consumed by the host. Frozen vector/Gaussian
adapters reuse one actual real tensor for D/G, as before.

## Frozen scope and applicability

| Unchanged task | Winner | Projection | Transport local-v2 | BOTH |
| --- | --- | --- | --- | --- |
| Two-pole explicit fixed fixture | Supported | Supported | BLOCKED | BLOCKED |
| Gaussian smoke and own stability | Supported | Supported | Supported | Supported |
| Trajectory | Supported | Supported | BLOCKED | BLOCKED |
| Residual student | Supported | Supported | BLOCKED | BLOCKED |
| Mid-scale identity guardrail | Supported | Supported | BLOCKED | BLOCKED |
| Unequal mass | Supported | Supported | Supported | Supported |
| Two-broad guardrail | Supported | Supported | Supported | Supported |

The frozen conditional/fixed component hosts do not consume transport's
sample-space losses. Preserve these explicit BLOCKED cells with zero spend;
BOTH cannot claim preservation of the conditional repair in this comparison.
There is no task-specific routing or alternate initializer. Two-pole retains
its original identity/zero/stored-weight fixture cohort, separate from learned
initialization. The mid-scale task is parameter-only and its prior is not sampled.

Eight original task cards and numerical gates are unchanged. All arms use seed0,
public deterministic initialization, fixed architecture/data law, seen batches,
prior law, sampling, full committed-update budgets and scheduled evaluations.
Named constructor/data/training-noise/evaluation streams remain isolated and
checkpointed. Failed own smoke blocks its own stability continuation. Compatible
archived evidence is context only; new controls run on the same frozen source.

Four ready schema 3 candidates/studies were frozen before training and share campaign
`projection_transport-round4-v1`. Their total full task allowances are 40080
seconds (10020 per arm), within the 43200-second track ceiling. Unsupported cells
spend nothing. No sweep, seed experiment, second proposal, unchanged continuation
or failed transport-v3 controller is admitted. Software checks and read-only
saved-state analysis add no scientific qualification and are accounted separately.

The numerical preregistered signature is unequal-mass endpoint full component
covariance error <=.85; >.85 falsifies it. Complete sustained gates decide the
repair. The substantive composition question is whether BOTH retains local-v2's
complete rare-density PASS and the broad guardrail. A conditional union remains
unmeasured if the combined host is unsupported. Competing explanations include
adversarial protection suppressing useful transport, finite-anchor tail blindness,
changing critic objectives, and shared normalized map deformation. Stop after
this four-arm readout regardless of its outcome.

## Execution

Scientific source and ready declarations were pushed before enqueue. The independent
public Queue/drain runner disables the full-compile completion callback and uses
one shared GPU worker. Logs/checkpoints/JSONL remain outside Git. Publication
reconstructs metrics and actual-training GIFs from certified saved observations.
All 24 workers finished once, with 8 zero-spend BLOCKED cells and no invalid/error
worker or scientific retry. Paid worker wall time is **1593.852249 seconds**
(26.56 worker minutes); executed full allowances **28680 seconds**, planned
full allowances 40,080, frozen ceiling 43,200, remaining reservation 0. Contention
costs establish no speed ranking. The four arms add **36960 committed updates**;
stability restores each1000-update prefix and adds 5,000, so restored work is not
counted again. Software checks are a separate synthetic fixture cohort: 93 distinct
checks pass, with 5 public-trainer fixture updates across pretraining checks and
corrections. There are no capacity probes or additional scientific training
diagnostics. Publication exactly recomputes **768 saved scalar/vector metric sets**,
checks conditional scoring controls, and renders **24 nine-frame actual-training
GIFs** without new updates/draws. [Validation](validation.json) records unchanged
original task, qualification and telemetry hashes, Forge checks and original
public mechanism hashes. The runner and local log follower have stopped.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_transport/logs/driver.log
# Reproduce published media/metrics and parity without training:
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/projection_transport/round4/publish.py
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/projection_transport/round4/audit.py
```

Measured scientific commit: `63bad2205c36b792648168c869a55bac2e1f605d`;
all four arms execute source digest
`17cf8181e68a4dfaaab00c5f678c99dcf288f4cbc828a0d8f2ca4833c1c25e2f`.
Later publication changes reports/reproduction sources only.

## Actual-training media

| Task | Winner | Projection | Transport | BOTH |
| --- | --- | --- | --- | --- |
| two_pole | [GIF](media/winner-two_pole.gif) | [GIF](media/projection-two_pole.gif) | BLOCKED | BLOCKED |
| gaussian1d_smoke | [GIF](media/winner-gaussian1d_smoke.gif) | [GIF](media/projection-gaussian1d_smoke.gif) | [GIF](media/transport-gaussian1d_smoke.gif) | [GIF](media/both-gaussian1d_smoke.gif) |
| gaussian1d_stability | [GIF](media/winner-gaussian1d_stability.gif) | [GIF](media/projection-gaussian1d_stability.gif) | [GIF](media/transport-gaussian1d_stability.gif) | [GIF](media/both-gaussian1d_stability.gif) |
| trajectory | [GIF](media/winner-trajectory.gif) | [GIF](media/projection-trajectory.gif) | BLOCKED | BLOCKED |
| residual_student | [GIF](media/winner-residual_student.gif) | [GIF](media/projection-residual_student.gif) | BLOCKED | BLOCKED |
| mid_scale_identity | [GIF](media/winner-mid_scale_identity.gif) | [GIF](media/projection-mid_scale_identity.gif) | BLOCKED | BLOCKED |
| vector_unequal_mass | [GIF](media/winner-vector_unequal_mass.gif) | [GIF](media/projection-vector_unequal_mass.gif) | [GIF](media/transport-vector_unequal_mass.gif) | [GIF](media/both-vector_unequal_mass.gif) |
| vector_two_broad | [GIF](media/winner-vector_two_broad.gif) | [GIF](media/projection-vector_two_broad.gif) | [GIF](media/transport-vector_two_broad.gif) | [GIF](media/both-vector_two_broad.gif) |

[Media certificates](media/index.json) bind every frame to certified actual
observations, with identical fixed-index frame selection and unchanged full
numerical cadence. [Reused scorer controls](scorer-controls.json) verify original
oracle PASS and point-collapse FAIL identities and unchanged scorer/task laws.
