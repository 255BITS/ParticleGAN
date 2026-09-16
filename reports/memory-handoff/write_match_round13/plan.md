# Generated-write continuation discrimination, round13

Five fresh 2k scouts, both GPUs, unchanged 10k schedule. Saved round12
match_shuffle25 at 2k is the primary matched-step control; its 5k checkpoint
is also reported. No seed repeats or new control retraining.

## Hypothesis and implementation

The winning mismatch ranking previously trained only on real-prefix memories.
Here the same G-facing point head ranks the correct next real point above an
other-episode real continuation after one generated replacement write.
Use exactly the existing circular shuffled donors, total mismatch weight .25,
and D GAN/default exact B-cap normalization by 1.25.

At target index t>=4, encode the real prefix through t-2, generate a detached
proposal for t-1 with the same particle and clock t-1, and write a blend of the
actual observed point t-1 and that proposal. Neither target t nor donor target
enters the memory construction. D ranks the original real target t and the
other-episode target under identical cached memory and target clock t.

Contexts: explored uses that resulting memory; mixed averages clean and explored
losses equally on every eligible example (not an interpolation of memories or a
random half-batch). Strength .25 or 1.0 ramps linearly over the first 500 steps.
The ramp is independent of existing point feedback. Prefix and replacement write
both receive mismatch gradients. The mixed_mild_headonly control detaches BOTH
mismatch contexts, disabling this loss's writer gradients only; the point head
and all existing writer training still operate.

| Run | Context | Strength | Mismatch writer gradients |
|---|---|---:|---|
| mixed_mild | mixed | .25 | yes |
| explored_mild | explored | .25 | yes |
| mixed_full | mixed | 1 | yes |
| explored_full | explored | 1 | yes |
| mixed_mild_headonly | mixed | .25 | no |

Existing point feedback (.5 probability/.25 strength), mixed point judging,
independent pair GAN (.25), proposal repair and six-band clock are unchanged.
G's objective is unchanged. D's new branch adds one detached G call (two internal
proposal-reader calls) and one write, independent of all other local branches.
Total D/G complete G calls: 5/4; reader calls: 10/8. Additional real-prefix
encoding remains comparable to old mismatch training but captures two positions.
No extra generated depth: at most two sequential outputs / one generated write.
Mixed judging adds another ranking and penalty evaluation; report actual costs.

Default clean configs and old checkpoints remain supported. No MSE objectives,
full generated training rollout, geometry labels/cursor, clipping, EMA, B-cap
overrides, or additional persistent state. D owns M. Runtime uses no expert.

## Fixed selection gates

Inspect completed runs only. Rank full warm1024 passes at prefixes8/32 first,
then minimum warm Q. Cold full passes and all256 results also reported.
At most two exact 2k->5k extensions, unchanged 10k schedule.
Qualify against saved match_shuffle25 2k if both warm pass fractions improve,
OR Q is >=20% better at BOTH prefixes, late Q no worse, radial RMSE <=5% worse,
and direction agreement <=2pp worse. Both routes require cold late stopping
<=1pp worse. Rank qualified models by minimum warm pass fraction then minimum Q.
Do not relax gates. Retain saved 5k as a descriptive comparison.

## Diagnostics

For every completed scout and the saved 2k/5k references, run the existing
matched-history radius/speed/direction interventions on CPU, recomputing original
and counterfactual paths on the same device. Add process-response windows after
0,1,8,32,128,512 generated writes: each measures the following32 generated points
(31 angular increments), not an instantaneous state measurement. Preserve the
existing last256 metric and report absolute fidelity as well as response.
Ideal normalized radius/speed response is1, not merely nonzero. These radii and
speeds lie within training support. This does not test unseen-domain transfer.
Also run completed-model point-head and late-restoration diagnostics. All long
trajectories and diagnostic regression measurements are evaluation-only.

Stable log: `tail -F runs/memory_path/core_round1/train.log`
Queue: runs/memory_path/write_match_round13

Additional pre-completion diagnostic: evaluate the same point-head ranking after
replacing the final prefix observation with a generated blend at strengths .25
and1. This checks whether the targeted continuation signal improves under one
generated write, separately from its long-run effect. Original targets and donor
selection are unchanged; no training or selection gate changes. Zero-strength
replacement reproduces clean ranking exactly in a focused test.
