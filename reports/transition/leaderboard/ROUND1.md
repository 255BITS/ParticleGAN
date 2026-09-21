# Conditioning challenge

The new best MoG score is **0.14179 joint SW1**, a **43.3% reduction** from the
established 0.24999 baseline. It uses three generators, one explicit-class joint
critic and generator class-input gain 4. G parameters, data and update budget
are unchanged. The reference-vs-reference floor is 0.03788.

We hold the 1,024-component MoG recipe, seed, 28k updates, data and evaluation
protocol fixed. Registered summaries and source archives are pinned in
configs/transition/leaderboard.json. These are development-benchmark results;
the fixed geometries have already informed our model choices.

1. **Explicit critic conditioning:** Feed class indicators to a scalar critic,
   removing UCD head selection and its classification loss. G is unchanged and
   D gains just 255 parameters. Result: **0.25182 SW1**, **0.01532 residual**;
   midpoint upper fractions **0.613 / 0.619** instead of targets **0.8 / 0.3**.
   This did not beat the leaderboard or fix conditioning, so UCD alone is not
   an adequate explanation of the failure.
2. **Stronger generator class input:** Keep the explicit-class critic and multiply
   G's class indicators by four. No new parameters or information. This tests
   whether stronger class inputs help the generators use the observed label.
   Result: **0.14179 SW1**, **0.01464 residual**; midpoint upper fractions
   **0.826 / 0.485**. This beats every established MoG entry, improves all three
   marginal distances, and restores substantial but incomplete class separation.

## Leaderboard outcome

| Setup | Joint SW1 | Mean residual | Upper class 0 / class 1 | Train seconds |
|---|---:|---:|---:|---:|
| Concat critic + G class gain 4 | **0.14179** | 0.01464 | 0.826 / 0.485 | 318.2 |
| Previous best: UCD + marginal critics | 0.24999 | 0.01486 | 0.577 / 0.571 | 609.6 |
| Concat critic + G class gain 1 | 0.25182 | 0.01532 | 0.613 / 0.619 | 319.5 |

The two concat runs have identical G/D parameter counts (65,286 / 135,169),
initial weights, MoG initialization, data and budgets. Scaling the observed class
input is the intended difference. This is evidence for the importance of generator
conditioning strength in this setup, not a claim that UCD cannot work or that the
new gain is optimal.

The winning run's interpolation/extrapolation SW1 is **0.12405 / 0.19503**.
State/action/next-state SW1 is **0.12387 / 0.15086 / 0.12743**, versus
**0.24065 / 0.23641 / 0.23913** for the previous leader. Coverage rises from .262
to .301, still far below the .954 reference floor. Shuffling the winning outputs
raises joint SW1 to .21541 and residual to .21396, demonstrating useful pairing.

The main remaining issues are class 1's upper-route bias (0.485 versus target
0.3), extrapolation, and physical consistency. Mean residual remains about 40%
of the mean reference step length. The score improvement is mostly distribution
fitting; it is not evidence that arbitrary state/action queries or rollouts work.

## Recommendation

Use `configs/transition/concat_class4.yaml` as the reference for the next round.
First test a stronger generator class gain, e.g. 8, with everything else fixed,
to see whether class 1's bias reduces. Then revisit marginal critics with working
class conditioning if per-block support errors remain. Keep the physics relation
as an evaluation diagnostic; do not silently enforce G3 = G1 + G2 to improve the
score. A later untouched geometry set should check generalization after selection
on this development leaderboard.

Validation: 19 transition/trajectory tests passed. Both challenger checkpoints
reproduced saved samples, their scores were recomputed from saved test arrays,
and MoG sigma survived checkpoint loading. Viewer JavaScript and plots were checked.
The registry verified the unchanged data/evaluation protocol and exact reference
samples for all five entries. No seed-only repeats were run.

See [the live leaderboard](README.md) for verified completed entries and viewers.
