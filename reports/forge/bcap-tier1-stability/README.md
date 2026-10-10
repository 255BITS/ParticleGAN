# Two-pole stability diagnostics and repairs

Six separately admitted research diagnostics isolate projection, global transport,
and local transport on the original two-pole task. The original fixed critic and
zero-particle fixture, public component API, seed 0, target panel, 80 updates,
24 observations, five-check terminal suffix, movement bound 0.3 and critic slope
cap 1 are retained. These diagnostics grant no ordinary qualification.

Existing integration evidence has incumbent PASS (17/24 passing checks, suffix
17) and combined FAIL (6/24, suffix 1) despite a passing endpoint. The combined
optimizer recorded zero conflicts, blends, or stalls. Its 11 slope violations,
including four of the final five checks, motivate transport/critic response
ablations rather than blaming projection blocking.

The public global/local weights and conditional consumer are already independent.
No production algorithm change is required for the six ablations. Each arm is a
single global recipe, and all source/runtime/task bindings will be frozen before
admission. At most two repair screens can be proposed only after the six results;
their equations, predictions, falsifiers and structural/constant classification
must precede their additional 600-second reservation.

The six ablations completed on frozen source `d70c2c3d985c4c75f2dbe615c8494f2991add3b6`
(scientific digest `f8ab30ab879c32f9e53e8ca6294762574c43a4138b7fa45b6a288c88c0b149e4`).
They consumed 46.433 charged seconds across six attempts and no retries. Every
arm completed the original 80 updates and 24 checks. Exact metrics, certificates,
checkpoint identities, activation counters, and six actual-training GIFs are in
[results.json](results.json); raw records remain in the external archive.

| Trainer | Original gate | Passing / 24 | Terminal suffix | Final movement | Final slope |
|---|---|---:|---:|---:|---:|
| Incumbent | PASS | 17 | 17 | 0.958502 | 0.952346 |
| Projection only | PASS | 17 | 17 | 0.958502 | 0.952346 |
| Global + local, no projection | FAIL | 6 | 1 | 0.928788 | 0.985156 |
| Projection + global | FAIL | 6 | 1 | 0.959487 | 0.979061 |
| Projection + local | FAIL | 16 | 0 | 0.917709 | 1.009425 |
| Full combination | FAIL | 6 | 1 | 0.928788 | 0.985156 |

The full combination records zero conflicts, blends or stalls across 80 updates;
local-only records seven conflicts and seven blends, with zero stalls. All arms
pass endpoint movement. A passing endpoint critic slope does not repair a failed
terminal suffix. The global term reproduces the persistent critic-cap excursions;
the local term alone still causes a late excursion. These ablations implicate
transport's change to the critic's training distribution; they do not identify
projection as the cause of the full-combination failure.

[audit.json](audit.json), reproduced by `analyze_saved.py`, verifies all eight
certified saved states and both frozen source snapshots. Incumbent/projection
and transport/combination have exactly equal final critic and direct-particle
tensors as well as every scheduled observation. All eight arms share the exact
original task contract and all eleven consumed non-evaluation stream bindings
and states. The stored-weight/zero-particle initialization is bound to the exact
frozen construction source; a separate initial tensor dump was not retained.

The incumbent's original direct-particle cloud collapses to one pole. Saved
combined particles split six/six between the poles. The original question's
declared gate requires movement and a critic cap, so this distributional difference
is explanatory evidence rather than an extra qualification gate.

Two globally configured repairs are declared after this readout, under a separate
600-second reservation, preserving both transport terms and direction blend:

1. **Constant cap margin:** change only BCAP `reg_kappa` from 1 to 0.9. The
   worst combined observed slope is 1.095865; approximate proportional scaling
   predicts 0.986279 with this margin. This is one constant choice, not a sweep.
2. **Structural finite cap:** `critic_step_mode="finite_cap"` measures the maximum
   per-sample input-gradient norm M on every actual training real/fake penalty
   panel. From the proposed actual DualNorm displacement Δ, accept the largest
   α in {1, 1/2, …, 1/256} for which M(D+αΔ) ≤ max(κ, M(D)); retain D if none
   passes. The base optimizer advances exactly once. Actual rounded parameters
   are measured, and the critic LR record reflects the accepted fraction.

The structural delta adds one baseline and at most nine finite measurements,
each using two critic/input-gradient forwards per original penalty panel pair.
It consumes no new samples or evaluation panels. Counters, accepted scales and
zero steps are checkpointed; pending caller panels require a completed-update
boundary before checkpointing. The default `none` retains the original optimizer
class, omits new serialized fields, and requires no hooks. Active unsupported
noise/formulation modes fail validation. There is no global Lipschitz guarantee:
the following particle step can change the measured locations.

Both repairs predict final movement ≥ 0.3 and all five original terminal slopes
≤ 1; either an insufficient movement or suffix below five falsifies the repair.
The six-arm source and repair source are distinct and retain their true provenance.
These bounded screens use the native SVD backend. Phase 2 must compare fresh
incumbent and combined recipes under its common final source, including any
separately declared CPU SVD numerical option.

Both bounded repairs completed PASS on frozen source
`dea70adebf458a64827be0b4447abf1e167cb19a` (scientific digest
`4e7f0c8653a0439ae325fa82f4b8050fcd9343d564b0c4e261b3e700484b6d2d`).
Their two paid attempts consumed 15.236 seconds with no retries. Exact metrics,
final-five observations, receipt/checkpoint identities and both actual-training
GIFs are in [repairs/results.json](repairs/results.json).

| Repair of full combination | Original gate | Passing / 24 | Suffix | Final movement | Final slope | Worst slope |
|---|---|---:|---:|---:|---:|---:|
| Cap margin κ=0.9 | PASS | 16 | 9 | 0.927479 | 0.971363 | 1.000506 |
| Finite same-panel critic guard | PASS | 17 | 17 | 0.920403 | 0.919918 | 0.956873 |

The cap margin still has one early 0.000506 original-cap excursion, followed by
the required passing suffix. The finite guard has no scheduled cap excursions.
Both repairs retain a six/six split of the actual final particles across the two
poles, and both record zero projection conflicts/blends/stalls across 80 updates.
Neither success comes from returning to the incumbent's one-pole cloud.

The finite guard checks every original critic update: 540 attempted scales,
620 finite measurements including the 80 baselines, 31 accepted steps, 49 rejected
zero-displacement steps, and 62 steps damped below full scale. The sum of accepted
scales is 19.12890625, an average 0.239111 of the proposed full displacement per
update. The maximum measured accepted panel slope is 0.9999952912. All 80 critic
clocks and transport calls remain intact. Substantial critic rejection and added
gradient probes are material risks to learning and the word wall-clock budget.

The evidence supports finite critic response as the cause of this scoped cap
regression and shows two ways to repair it. Prefer the constant arm if the fresh
full Tier 1 screens tie, because it has no probe overhead or rejected steps. Keep
the structural arm as the stronger local cap-control comparison; it must earn
its extra complexity through full gates and acceptable runtime. No result here
repairs the full Tier 2 suite or establishes ordinary qualification.

Software checks pass 39 tests covering independent transport activation, finite
damping/rejection, exact checkpoint continuation, no-probe disabled behavior,
RNG consumption rejection, and existing mechanism/boundary contracts. Their
stdout is in `logs/critic-cap-software.log` in the external archive.

Reproduce declarations and execution using `workflow.py declare`, commit the
clean source, then `workflow.py run`. Raw logs are outside Git:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/stability/logs/driver.log
```

Repairs use the same actions with `--repairs`; publication reads certified saved
records and renders actual-training GIFs without training or sampling. Tail
`logs/repair-driver.log` for that campaign. No ordinary qualification is granted.
