# Lunar slow→fast evaluation

The `train_scope=control` run was scored on pop-os cuda:1. It is a failure.
No residual checkpoint has been rolled out in this checkout. Do not copy the
2D toy board into this file as a Lunar result.

Collect from YuE2 `#18` `particle_yue18_143320/best.pt`: 200/200 successful
landings, 63 fast / 62 slow, 60 pairs, 13495 rows, crashes_excluded=0.
Train: 2500 steps, `adv_weight=1`, `safe_fast_weight=0`. Shared-seed validation:

| Arm | Landings | Success steps | Crashes |
| --- | ---: | ---: | ---: |
| #18 baseline | 20/20 | 205.4 | 0 |
| slow→fast @250 | 12/20 | 329.0 | 3 |
| slow→fast @1000 | 0/20 | — | 19 |
| slow→fast @2500 | 0/20 | — | 18 |

Eval selected none. Longer training was worse. These pairs are strangers
(different episodes, nearest state). Do not train them. The trainer refuses
`slow_seed != fast_seed`. Collect has to be reworked to two teachers, the same
seed, a both-land gate, and progress alignment after
`examples/slow_fast_paired_2d.py` prints GATE PASS. Commands are in
[the experiment note](../../../docs/gym-slow-fast.md).
