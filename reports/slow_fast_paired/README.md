# Slow→fast paired finetune (CPU gate)

Gate **PASS**. Winner of the rank key: `connected`.
Gate PASS is the pairing check. Lunar collect now uses two teachers, the same seed, and progress alignment. Do not train the old stranger pairs. Commands are in `docs/gym-slow-fast.md`.
These numbers are a 2D pad. They are not Lunar landings.

One seed (`0`), fixed eval starts, no seed sweep. Rank is landings first,
then fewer steps among successes. An arm is ineligible when `adv_weight` is
not 1, `b_cap` did not run, landings fall below 0.90, crashes exceed 0.10,
or the return panel (steps 100, 200, 400) lost the pad. `connected`,
`overspeed`, and `stranger` use the same full lander, the same learning rate,
and the same RpGAN step. The pairs are the difference. `Early` is step 50.
`Return min` is the worst landing rate from step 100 on.

| Arm | Landings | Early (step 50) | Return min | Steps | Crash | Rank key | adv | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| zero | 1.000 | — | 1.000 | 37.12 | 0.000 | ineligible | — | False |
| slow_only | 1.000 | — | 1.000 | 36.49 | 0.000 | (1.000,36.49) | 1.0 | False |
| crash_fast | 0.455 | — | 0.455 | 13.66 | 0.545 | ineligible | 1.0 | False |
| unpaired | 0.105 | — | 0.105 | 32.81 | 0.895 | ineligible | 1.0 | False |
| supervised | 1.000 | — | 1.000 | 31.21 | 0.000 | ineligible | 0.0 | False |
| stranger | 0.205 | 1.000 | 0.205 | 39.18 | 0.795 | ineligible | 1.0 | False |
| overspeed | 0.762 | 0.608 | 0.562 | 27.61 | 0.237 | ineligible | 1.0 | False |
| connected | 1.000 | 0.873 | 1.000 | 28.31 | 0.000 | (1.000,28.31) | 1.0 | True |

Safe teacher mean steps 38.22. Fast teacher speed bias 0.02: update 2 lands in 23.41 steps (landings 1.000); update 3 lands in 20.04 steps; update 6 landings 0.442.
Collector: 80 same-seed both-land starts, 3058 progress-aligned rows (mean edit 0.081), 2905 stranger nearest-state rows (mean distance 0.201), fast failures excluded from the held set 0, crash episodes at the break 58. Mean fast steps on kept pairs 24.00.

## Why the controls lose

- `zero` is the safe teacher. No GAN step, so the rank key is ineligible. It lands, and it is slower than `connected`.
- `slow_only` is the same RpGAN step with the safe action as the target. It stays a successful slow landing and loses on steps.
- `crash_fast` trains on actions from fast-teacher episodes that missed the pad after the speed break. Contact can be sooner. Landings fall, so the rank key drops it.
- `unpaired` uses fast actions from other starts and trains every weight. The student leaves the pad.
- `supervised` matches the progress-aligned fast action with MSE and `adv_weight=0`. Landings may hold. The rank key rejects it because the #18 step did not run.
- `stranger` is the disabled Lunar collector: a different episode's fast action at the nearest state, plentiful rows, full-weight RpGAN at `adv_weight=1`. Landings fall and crashes rise.
- `overspeed` is the same progress alignment one speed update later. The teacher still lands. The student does not keep the pad.
- `connected` is the held speed: same seed, both land, progress `t/T`, full fast action, `adv_weight=1`. Landings hold on the return panel and success steps fall. Diagnostic MSE stays outside the loss.

## Lunar validation that this gate is built to catch

pop-os cuda:1 trained `train_scope=control` (E_control and G2) on nearest-state pairs from YuE2 #18 `particle_yue18_143320/best.pt`. Collect was 200/200 landings, 63 fast / 62 slow, 60 pairs, 13495 rows, crashes_excluded=0. Train was 2500 steps, `adv_weight=1`, `safe_fast_weight=0`. Shared-seed validation:

| Arm | Landings | Success steps | Crashes |
| --- | ---: | ---: | ---: |
| #18 baseline | 20/20 | 205.4 | 0 |
| slow→fast @250 | 12/20 | 329.0 | 3 |
| slow→fast @1000 | 0/20 | — | 19 |
| slow→fast @2500 | 0/20 | — | 18 |

Eval selected none. Longer training was worse. Those pairs are strangers: different episodes, matched by geometry, no shared landing. The trainer refuses `slow_seed != fast_seed` and any file named `pairs.npz`. Lunar collect now rolls two teachers on the same seed, keeps a pair only when both land, and aligns by progress. This board is not a new Lunar result.

```bash
python -u examples/slow_fast_paired_2d.py
```
