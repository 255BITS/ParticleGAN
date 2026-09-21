# Circle transition leaderboard

First learned baseline is `paired_error` at `f8a5a4a`. This benchmark is the
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

Frozen evaluation after training. Rank key is worst-direction full-trace success
at 1,024 steps on the test main panel. Validation matched the test closely
(worst-direction success 0.203), and this recipe was fixed before that panel was
read. Metrics: [paired_error.json](paired_error.json). Config:
`configs/circle/paired_error.yaml`. Checkpoint and log:
`results/circle_transition/paired_error/` at revision `f8a5a4a`.

| Run | Training recipe | Updates | Success 256 / 1024 | Worst-dir 1024 | Radial RMSE | Signed-speed error | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|
| paired_error | Pretrain E_pair+G1/G2/G3 300 steps, then paired-error RpGAN `adv_weight=1` on E_control+G2 | 300+2000 | 0.336 / 0.328 | 0.203 | 0.248 | 0.043 | 49 s |

Controls on the same test main panel, 1,024 steps: analytic expert success 1.000,
radial RMSE 0, 47.7 turns; zero action success 0, 0 turns, stopping fraction 1;
reversed expert success 0, direction agreement 0, −47.7 turns. Learned direction
agreement is 1.000 and completed turns are 45.5. Counterclockwise success is
0.453 and clockwise success is 0.203. Late-window radial RMSE is 0.186. Maximum
radial error is 1.47. No episode was nonfinite. Recovery-panel worst-direction
success at 1,024 steps is 0.281.

Local test errors, separate from the rank: action L2 0.033, control-time G3 next
L2 0.782, persistence next L2 0.289, paired G3 (E_pair sees the expert action)
next L2 0.190. Budget: 76,800 pretrain row draws, 512,000 finetune row draws,
batch 128, 500 `b_cap` applications, 54,418 trainable controller parameters,
48.7 s wall on CPU. Diagnostic action MSE fell from about 1.3 to about 0.02 and
stayed outside the controller loss.

## Interpretation

The controller learned the requested direction and a near-expert step rate from
paired-error RpGAN alone. Worst-direction success is the reason it ranks above
zero and reversed motion: those controls score 0 because one never moves and the
other travels the wrong way. Radius is the remaining failure. Mean radial RMSE
0.248 and a late window of 0.186 sit above the 0.1 success bar, so only about a
third of episodes trace a tight circle for the full horizon. Local action error
is already small, and the logged MSE flattened after roughly 1,500 updates, so
the closed-loop radius walk is a compounding bias the current update is not
removing. Control-time G3 next-state error is 0.782 against a persistence baseline of
0.289. The paired head, which sees the expert action, reaches 0.190. The policy
rank remains closed-loop success.

## Next recommendation

One mechanism aimed at radial hold. Leave the paired-error objective,
`adv_weight=1`, the frozen scope, and the v1 panels in place. A longer run of
this same loss is unlikely to move RMSE 0.248 under 0.1 after the diagnostic
plateau. Compare that single change on the validation panel at the same update
budget, and touch the test panel only for the selected comparison. No seed repeat.
