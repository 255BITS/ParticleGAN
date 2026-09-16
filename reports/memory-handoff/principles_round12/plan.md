# First-principles local objectives, round12

Twelve fresh2k scouts, both GPUs, fixed existing data/particle streams and10k
schedule. Baseline is saved proposal_mixed_pair25 at2k. No seed sweeps. All
scouts retain six-band clock, proposal repair, mixed point judging, point feedback
strength.25/probability.5 and independent pair GAN weight.25.

## Hypotheses and controls

1. D may not distinguish incompatible continuations. Same point score head gets
additional paired GAN ranking: actual next sample versus another episode's real
next sample. Prefix>=4 only. Nearest donor chosen using last observed point,
never target coordinates, and excluding all same-episode candidates. This is a
cross-domain nearest-observation heuristic, not circle labels. Observation noise
means mismatched examples are not guaranteed impossible. Weights.1/.25/.5;
shuffle.25 control (deterministic circular donors among eligible examples).
D objective and its default exact B-cap mixture normalized by1+weight; G point
objective unchanged. Both clean-prefix score and writer trained in D phase.

2. Recovery from disturbed real-prefix history. Half of episodes get independent
Gaussian observation disturbance std.1 or.3 throughout the real prefix. Existing
pair branch reads that disturbed prefix and generates two samples separated by
one generated write. Judge always reads unperturbed observed prefix. Same noise
reused across D/G phases, dedicated reproducible RNG, no target corruption. No
extra generated depth; one additional prefix encoding per phase. Point branch
unchanged. Test each scale and nearest.25+noise.1.

3. Direct joint future GAN. G gains explicit relative query-offset Fourier inputs
(sin/cos-minus-one at four frequencies) that are exactly zero at runtime offset0.
Same reader/adapter predicts offsets(0,4,12) from identical prefix M and particle,
with target clock t+k; there are no writes between queries. Joint future D head
reads prefix M and all three predictions; real counterpart uses real future data.
Prefixes clamped to51 for this branch only. Future weights.1/.25 convexly mix
both D/G GAN losses and default exact B-cap with existing branches. Architecture
control has identical G query inputs but no future objective. Also nearest.25+
future.1 and nearest.25+noise.1+future.1. Separate D head adds capacity; G0 query
shares runtime parameters. This signal is adversarial, not regression.

All mechanisms optional/default-off; existing configs/checkpoints remain supported.
No MSE objectives, full generated training rollouts, geometry labels, moving
cursor, clipping, EMA or B-cap overrides. D owns persistent M, stateless G;
fixed particle per sequence. Independent branches never chain generated writes.
Maximum generated depth remains two outputs/one write. Future configurations
use7 full G calls/14 proposal-reader calls per phase; others4/8. Nearest matching
adds O((batch*samples)^2) distances and another real-prefix encoding in D phase.
Recovery/future add real-prefix computation; report actual time.

## Selection fixed before training

Inspect completed runs only. Full1024 cold and warm prefix8/32 passes remain
primary. Rank secondary progress by minimum warm Q across prefixes; preserve
lateQ, radial error, direction, phase and correct-arc diagnostics. Q is not a
success probability. At most two exact2k->5k continuations on same10k schedule.
Qualify if both warm pass fractions increase, OR Q>=20% better at BOTH prefixes,
lateQ no worse, radial<=5% worse and direction agreement<=2pp worse. Both routes
also require cold late stopping<=1pp worse. Rank qualifiers by minimum warm pass
fraction then minimum Q. No extension if none qualifies. Cold-only success is
reported separately. Never relax gates after inspecting results.

## Completed-model diagnostics

Held-out point-head ranking of correct next samples against nearest-history
mismatches, circular-shuffled mismatches, earlier points, and later points.
Report ranking/margins and donor endpoint distance; no geometry-trained labels.
Also measure G's local prediction error for evaluation only. Reuse fixed panel.
Late restoration compares autonomous M at clock288 with real-history M at the
same clock and recent32-observation real-history M; reset-clock control separates
clock effects. All starts end at the same reference point, same particle/device;
restoration jumps state and can change compatibility. Rescue is not proof of a
single causal explanation. Diagnose baseline and completed scouts; process probes
for promising models. All generated long paths remain evaluation-only.

Stable log: tail -F runs/memory_path/core_round1/train.log
Queue: runs/memory_path/principles_round12

## Additional diagnostic after initial completions

Mismatch scouts improved D ranking while retaining zero passes. Added an
evaluation-only sample-gradient probe: cosine of ascending D score with the
vector from generated point to the correct next target, at real-history M and
autonomous/restored late M. This adaptive diagnostic does not change training
or selection gates. It measures a local Euclidean direction, not a guaranteed
process-preserving repair field. No MSE objective or optimizer is introduced.
