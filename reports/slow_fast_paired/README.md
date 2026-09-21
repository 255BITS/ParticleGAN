# Slow→fast paired finetune (CPU gate)

Gate **PASS**. Winner of the rank key: `connected`.
This gate must **PASS** before Lunar collect is reworked. Do not train the current stranger pairs. Commands are in `docs/gym-slow-fast.md`.
These numbers are a 2D pad. They are not Lunar landings.

One seed (`0`), fixed eval starts, no seed sweep. Rank is landings first,
then fewer steps among successes. An arm is ineligible when `adv_weight` is
not 1, `b_cap` did not run, landings fall below 0.90, crashes exceed 0.10,
or the return panel (steps 100, 200, 400) lost the pad. `stranger` and
`connected` use the same full lander, the same learning rate, and the same
RpGAN step. The first 50 updates kick both off the pad. `Early` is that
step. `Return min` is the worst landing rate from step 100 on.

| Arm | Landings | Early (step 50) | Return min | Steps | Crash | Rank key | adv | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| zero | 1.000 | — | 1.000 | 37.12 | 0.000 | ineligible | — | False |
| slow_only | 1.000 | — | 1.000 | 33.97 | 0.000 | (1.000,33.97) | 1.0 | False |
| crash_fast | 0.000 | — | 0.000 | 64.00 | 1.000 | ineligible | 1.0 | False |
| unpaired | 0.000 | — | 0.000 | 64.00 | 1.000 | ineligible | 1.0 | False |
| supervised | 1.000 | — | 1.000 | 12.03 | 0.000 | ineligible | 0.0 | False |
| stranger | 0.000 | 0.000 | 0.000 | 64.00 | 0.945 | ineligible | 1.0 | False |
| connected | 1.000 | 0.000 | 1.000 | 21.41 | 0.000 | (1.000,21.41) | 1.0 | True |

Collector: 80 matched starts, 3058 same-state rows, 2875 stranger nearest-state rows (mean distance 0.452), connected retime 0.25 (mean edit 0.060), fast failures excluded 0. Mean slow steps 38.22, mean fast steps 12.05.

## Why the controls lose

- `zero` is the competent slow lander. No GAN step, so the rank key is ineligible. It lands, and it is slower than `connected`.
- `slow_only` is the same RpGAN step with the slow member as the target. It stays a successful slow landing and loses on steps.
- `crash_fast` puts crash actions in the fast slot and trains every weight. Contact can be sooner. Landings collapse, so the rank key drops it.
- `unpaired` uses fast actions from other starts and trains every weight. The student leaves the pad. That is not the same world flown faster.
- `supervised` matches the fast member with MSE and `adv_weight=0`. Landings may be excellent. The rank key rejects it because the #18 step did not run.
- `stranger` is the Lunar collector: a different episode's fast action at the nearest state, plentiful rows, full-weight RpGAN at `adv_weight=1`. It leaves the pad and is still off at steps 100, 200, and 400.
- `connected` is the same update on a same-start retime (0.25 of the fast law at that state). It is off the pad at step 50 and back from step 100 through 400, with fewer success steps. Diagnostic MSE stays outside the loss.

## Lunar validation that this gate is built to catch

pop-os cuda:1 trained `train_scope=control` (E_control and G2) on nearest-state pairs from YuE2 #18 `particle_yue18_143320/best.pt`. Collect was 200/200 landings, 63 fast / 62 slow, 60 pairs, 13495 rows, crashes_excluded=0. Train was 2500 steps, `adv_weight=1`, `safe_fast_weight=0`. Shared-seed validation:

| Arm | Landings | Success steps | Crashes |
| --- | ---: | ---: | ---: |
| #18 baseline | 20/20 | 205.4 | 0 |
| slow→fast @250 | 12/20 | 329.0 | 3 |
| slow→fast @1000 | 0/20 | — | 19 |
| slow→fast @2500 | 0/20 | — | 18 |

Eval selected none. Longer training was worse. Those pairs are strangers: different episodes, matched by geometry, no shared landing. The trainer now refuses `slow_seed != fast_seed`. Rework collect to connected pairs before the next cuda:1 run. This board is not a new Lunar result.

```bash
python -u examples/slow_fast_paired_2d.py
```
