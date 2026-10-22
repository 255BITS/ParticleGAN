# Settling result: no gate repair; the existing observer limits application

`bispf-settling-displacement-v1` finishes **4 PASS / 11 FAIL**, matching the
shared control's pass set. It repairs neither Gaussian stability nor overlap.
All four protected controls pass. Fourteen tasks have exactly equal retained
numerical trajectories and final training states; Grid100 changes, improves
precision and centering, and still fails with worsened covariance bias.

Keep the selected baseline. This candidate supplies no qualification or default
adoption. The [parent comparison](README.md) owns the four-arm readout and links
the goal's single current leaderboard; this mechanism report introduces no
separate ranking. [settling-results.json](settling-results.json) retains compact
final metrics, verdicts, hashes, costs and equality checks for all 15 pairs.

## Exact comparison and evidence

Both requests execute committed source
`2a2b662bdfa6249076b27dbea4c136e23d3e7a80`, digest
`8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb`,
in campaign `bispf-three-theme-repair-round1-v1`:

- Shared control request: `728323efa37b4c2345764a9d`.
- Settling request: `7a58ff2ed6de1ece217a77ce`.
- Both complete 15/15 tasks with 4 PASS and 11 FAIL. Repaired gates: 0;
  lost control passes: 0. The original 11 failure tasks all remain FAIL.
- Paid attempt seconds: control 1312.436881; settling 1401.374448. These
  instrumented, potentially contended measurements do not establish speed.

The [declared delta](settling.md) relocates existing thermal application to the
actual normalized/direction-blended G/prior movement and explicitly changes
SBNP gradient cooling. All task architectures, initial prior laws/capacities,
seen data sequence, sampling, seed-0 public initializer, update budgets and
evaluation cadence remain fixed. All 15 paired initialization/RNG receipts and
final checkpointed named-stream state hashes are equal. There is no new
training, sampling, gate change or continuation in
the analyses below.

For 14 pairs, **complete final metric dictionaries, every retained observation,
all final model state dictionaries and both final optimizer state dictionaries
are exact**. This includes all 121 stored Gaussian sample panels and 24 overlap
panels. Rotated100 and staggered100 additionally match in all 34 native live
observations and all 34 saved 4096-sample arrays per task. The hashes compare
trained tensor/optimizer contents, not serialized files whose recipes and
provenance naturally differ.

Per-update thermal, gradient and applied-movement histories were not serialized
by the frozen adapters. Their absence cannot be repaired from endpoint state.
Software checks establish the helper's scale law when active; they do not
supply an observed in-training scale history. Exact retained states show that
the recipe is ineffective on the settling tasks, without proving that every
unrecorded intermediate thermal query was one.

## Gaussian and overlap remain unchanged

The original Gaussian gate continues each arm's own passing 1000-update smoke
state, requires all 72 stationary checks, the last five recovery-window checks,
and all 24 shifted-hold checks. Both arms give:

| Phase | Passing observations | Longest passing run | Scientific result |
| --- | ---: | ---: | --- |
| Stationary, 1042–4000 | 6 / 72 | 2 | FAIL |
| Recovery, 4042–5000 | 2 / 24 | 1 | Last-five gate FAIL |
| Shifted hold, 5042–6000 | 2 / 24 | 1 | FAIL |

At update 6000, mean error .054736 sigma and width ratio .876510 pass while CDF
KS .070319 fails. Means and widths do not describe the complete output law.
The saved observer scale is exactly 1 at updates 1000/4000/6000; its residuals
are .159093/.107410/.175164, all above tau .020. Both arms' checkpoints have
the same earned values. Their shifted-target location response is also exact;
relocation did not change recovery or retention.

Overlap likewise retains 10/24 passing observations, longest run 3, terminal
suffix 0 versus required 5, and endpoint covariance error .522296 versus .45.
Its final thermal scale is 1 with residual .236079. Newly retained tensors
resolve the earlier signed-shape ambiguity: at step 1150 the observable
covariance is `[[.171444,.037335],[.037335,.240879]]`, contracted relative to
analytic target `diag(.425,.3025)`. At 1200 it is
`[[.441184,.184294],[.184294,.265234]]`: the x variance recovers while a large
spurious cross-covariance appears. Covariance-bias eigenvalues change from
`[-.260563,-.054615]` to `[-.196763,+.175680]`. This supports changing shape
and axis correlation, not a persistent uniform spread error. These analytic
signed diagnostics do not replace the original empirical-reference gate, and
overlap's hidden mixture labels remain unscored.

The median adjacent empirical Gaussian W1 movements are .369490 sigma during
stationarity, .226956 during recovery and .290698 during shifted hold. These
compare independent saved output distributions and include finite-sample
variation; they are not paired output or parameter displacements.

Actual-training illustrations come from this campaign's saved scored outputs:
[Gaussian](media/bispf-settling-displacement-v1/gaussian1d_stability.gif),
[overlap](media/bispf-settling-displacement-v1/vector_overlap.gif). Numerical
gates and artifacts determine the conclusions.

## Grid100 is the single changed task

Native live observations and saved output arrays first differ at update 4750,
after their last exact retained state at 4500. Every saved target array remains
exact across arms. This bounds the first observed divergence, not the exact
first thermal activation. The final earned scale .996701 and residual .019329
differ from control's scale 1 and residual .020752.

| Independent 100000-sample holdout metric | Control | Settling |
| --- | ---: | ---: |
| Precision | .304970 | .356680 |
| Center RMS, target sigmas | 1.497130 | 1.273105 |
| Covariance trace bias | +.325427 | +.463478 |
| Covariance Frobenius RMS | .610160 | .629606 |
| Radial KS | .423940 | .425754 |
| Mass TV | .170280 | .170870 |

All five terminal quality checks still fail. Higher precision and improved
centering coexist with worse covariance bias and small tail/allocation
regressions; this is an active finite-movement effect without a gate repair.
It does not establish a global density solution or explain Gaussian settling.

## Next action: retain the negative observer screen

The same-law nearest-neighbor floor is now measured: all 160 Gaussian and all
160 overlap disjoint 128-versus-128 panels from passing saved outputs exceed
the existing .020 cooling threshold. Their residual medians are .056271 and
.125827. A raw nearest-neighbor mismatch can remain large simply because the
two finite panels contain different draws from the same law.

One proposed energy-U replacement was evaluated without training or new draws,
using fixed approximate generated references. Its threshold-free separation is
good, but the **unchanged fast-attack EMA still reheats passing overlap states**,
and explicit separate analytical missing-mode fixtures cool before full
acquisition. Rare 2% population loss yields thermal scale .003408; losing one
of sixteen equal ring modes yields .027855. Complete collapse remains hot,
which does not establish safe partial acquisition. The [negative screen and
exact scope](settling.md#saved-state-signal-screen-reject-an-unchanged-energy-observer),
[compact screen receipt](settling-signal-summary.json) and
[reproduction source](probe_settling_signal.py) preserve these controls.

**Stop this exact repair revision and do not launch the energy-U drop-in
successor.** A future observer needs to address finite-sample noise and support
acquisition together, pass these saved-state/analytical controls, and declare a
new bounded source/study before training. The present result does not justify
another threshold sweep, seed experiment or unchanged replay for telemetry.

Reproduce the read-only full comparison and compact receipt:

```sh
OMP_NUM_THREADS=1 python reports/forge/three-theme-repairs-20261022/analyze_settling.py \
  --output /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/settling-final-analysis.json \
  --summary-output reports/forge/three-theme-repairs-20261022/settling-results.json
```

The script hashes every loaded sample/checkpoint/native-event/snapshot artifact,
checks both source bindings and complete 15-task requests, and consumes zero
sampling draws and zero optimizer updates. Full per-observation analyses and
raw training artifacts remain outside Git.
