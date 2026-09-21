# Circle transition leaderboard

**No completed learned runs yet.** This is a fresh benchmark for the
[circle transition-encoder toy](../../docs/circle-toy.md). There are no inherited
memory runs or Lunar Lander scores in this leaderboard.

## Evaluation protocol v1

Frozen before the first learned comparison. Thresholds are the handoff values.
The cell split, recovery window, and panel seeds are the concrete v1 choices
recorded in `lib/circle_transition.py` (`protocol()`).

- Geometry cells: 5×5 centers in [-0.75, 0.75]², 4 radii in [0.6, 1.4], 4 speed magnitudes in [0.12, 0.40]. Bucket `cell % 20`: train 0–13, validation 14–16, test 17–19.
- 128 episodes, both directions, horizons 256 and 1,024. Panel seeds: val/test main 51001/51002, recovery 51011/51012.
- Main panel starts on the circle. Recovery starts at normalized radii 0.8 and 1.2. Recovery-window radial metrics use the suffix after 64 steps and are not a substitute for full-trace success.
- Full-trace success: radial RMSE < 0.1, mean absolute signed-step error < 0.03 rad/step, direction agreement > 95%, at least one requested turn, finite trace.
- Rank by worst-direction 1,024-step full-trace success, then radial RMSE and signed-speed error. Local action / G3 error is reported and does not rank the policy.
- Analytic expert, zero action, and reversed expert are controls, not learned rows.

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
not learned entries. Protocol v1 above is the frozen comparison contract.

## Results

| Run | Training recipe | Updates | Full-trace success 256 / 1024 | Radial RMSE | Signed-speed error | Wall time |
|---|---|---:|---:|---:|---:|---:|

Add rows only after a run and its frozen evaluation complete. Link the config,
source revision, checkpoint, metrics and assessment. Select mechanisms using
validation; reserve the final test panel for the selected comparison. Use metrics
rather than image inspection, and do not run seed-only repeats.

## Next recommendation

Train `configs/circle/paired_error.yaml` (transition pretrain, then paired-error
RpGAN at `adv_weight=1` on `E_control`+G2) and fill this table from the frozen
test panel. There is no current winner. Do not add a seed repeat.
