# Slow→fast paired finetune (CPU gate)

Gate **PASS**. Winner of the rank key: `anchored`.
This gate must **PASS** before the next Lunar retrain. Commands are in `docs/gym-slow-fast.md`.
These numbers are a 2D pad. They are not Lunar landings.

One seed (`0`), fixed eval starts, no seed sweep. Rank is landings first,
then fewer steps among successes. An arm is ineligible when `adv_weight` is
not 1, `b_cap` did not run, landings fall below 0.90, crashes exceed 0.10,
or a recorded checkpoint lost the pad. Steps count successful contacts only.
`contact` is any ground hit, so a crash can look fast there and still be ineligible.
`overwrite` and `anchored` record landings every 100 steps. `min landings` is
the worst of those checkpoints.

| Arm | Landings | Min landings | Steps | Crash | Contact | Rank key | adv | b_cap | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| zero | 1.000 | 1.000 | 37.12 | 0.000 | 37.12 | ineligible | — | 0 | False |
| slow_only | 1.000 | 1.000 | 33.97 | 0.000 | 33.97 | (1.000,33.97) | 1.0 | 100 | False |
| crash_fast | 0.000 | 0.000 | 64.00 | 1.000 | 6.77 | ineligible | 1.0 | 100 | False |
| unpaired | 0.000 | 0.000 | 64.00 | 1.000 | 10.40 | ineligible | 1.0 | 100 | False |
| supervised | 1.000 | 1.000 | 12.03 | 0.000 | 12.03 | ineligible | 0.0 | 0 | False |
| overwrite | 0.000 | 0.000 | 64.00 | 0.945 | 21.47 | ineligible | 1.0 | 100 | False |
| anchored | 1.000 | 1.000 | 23.83 | 0.000 | 23.83 | (1.000,23.83) | 1.0 | 100 | True |

Collector: 80 matched starts, 3058 same-state rows, 2875 nearest-state rows (mean distance 0.452), fast failures excluded 0. Mean slow steps 38.22, mean fast steps 12.05.

## Why the controls lose

- `zero` is the competent slow lander. No GAN step, so the rank key is ineligible. It lands, and it is slower than `anchored`.
- `slow_only` is the same RpGAN step with the slow member as the target. It stays a successful slow landing and loses on steps.
- `crash_fast` puts crash actions in the fast slot and trains every weight. Contact can be sooner. Landings collapse, so the rank key drops it.
- `unpaired` uses fast actions from other starts and trains every weight. The student leaves the pad. That is not the same world flown faster.
- `supervised` matches the fast member with MSE and `adv_weight=0`. Landings may be excellent. The rank key rejects it because the #18 step did not run.
- `overwrite` is the Lunar recipe: unfreeze the lander and fit nearest-state fast actions with paired-error RpGAN at `adv_weight=1`. Landings fall. A later checkpoint is worse. The rank key rejects it even if a leftover success is fast.
- `anchored` freezes that lander and trains a residual of at most 0.15 per channel on the same nearest-state rows, same RpGAN step, `b_cap` every fourth update. Diagnostic MSE stays outside the loss.

## Lunar validation that this gate is built to catch

pop-os cuda:1 trained `train_scope=control` (E_control and G2) on nearest-state pairs from YuE2 #18 `particle_yue18_143320/best.pt`. Collect was 200/200 landings, 63 fast / 62 slow, 60 pairs, 13495 rows, crashes_excluded=0. Train was 2500 steps, `adv_weight=1`, `safe_fast_weight=0`. Shared-seed validation:

| Arm | Landings | Success steps | Crashes |
| --- | ---: | ---: | ---: |
| #18 baseline | 20/20 | 205.4 | 0 |
| slow→fast @250 | 12/20 | 329.0 | 3 |
| slow→fast @1000 | 0/20 | — | 19 |
| slow→fast @2500 | 0/20 | — | 18 |

Eval selected none. Longer training was worse. The next gym recipe freezes #18 and trains only the bounded residual. Retrain only after this gate PASSes. This board is not a new Lunar result.

```bash
python -u examples/slow_fast_paired_2d.py
```
