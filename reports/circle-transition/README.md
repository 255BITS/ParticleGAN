# Circle transition leaderboard

**No completed runs.** This is a fresh benchmark for the
[circle transition-encoder toy](../../docs/circle-toy.md). There are no inherited
memory runs or Lunar Lander scores in this leaderboard.

## Proposed evaluation protocol

Freeze and record these settings with the first evaluator before comparing models:

- Start each episode from an independently sampled observed position and explicit
  circle geometry/signed speed. No observed trajectory prefix or recurrent state.
- Evaluate 128 held-out episodes, balanced across directions, at horizons 256 and
  1,024. Reuse the same panel for mechanism comparisons. Split by circle parameters
  to keep training/calibration, validation and final test panels separate.
- Main panel starts on the target circle. A separate recovery panel starts at
  normalized radii 0.8 and 1.2; report it separately, with a declared recovery window.
- Apply G2 actions through the displacement environment. No expert corrections,
  projection onto the circle, online optimization or replacement of observations
  by G3 predictions. Full rollouts are evaluation-only.
- Count full-trace success only when radial RMSE relative to the requested radius
  is below 0.1, mean absolute signed angular-step error is below 0.03 rad/step,
  direction agreement exceeds 95%, and accumulated progress in the requested
  direction completes at least one turn. Reject nonfinite trajectories. Also
  report maximum radial error and late-window errors so the average cannot hide
  long excursions or eventual failure.
- Report success by direction, signed-speed error, completed turns, radial RMSE,
  drift and stopping separately. Include held-out action error and G3 next-state
  error versus persistence, but do not rank the policy by prediction error alone.
- Rank first by worst-direction 1,024-step full-trace success, then by radial RMSE
  and signed-speed error. Report sample counts, update budget, parameter counts,
  examples seen and wall time. Compare learned arms at matched budgets.

The analytic expert, zero action and reversed expert are evaluator controls,
not learned entries. Definitions and numerical gates above are a proposed initial
protocol, not measured results; version any revisions before running comparisons.

## Results

| Run | Training recipe | Updates | Full-trace success 256 / 1024 | Radial RMSE | Signed-speed error | Wall time |
|---|---|---:|---:|---:|---:|---:|

Add rows only after a run and its frozen evaluation complete. Link the config,
source revision, checkpoint, metrics and assessment. Select mechanisms using
validation; reserve the final test panel for the selected comparison. Use metrics
rather than image inspection, and do not run seed-only repeats.

## Next recommendation

Implement the independent circle sampler and evaluator, validate the three
analytic controls, then train the first transition-encoder / paired-error model.
There is no current winner and no experiment is queued.
