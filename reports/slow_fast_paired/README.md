# Slow→fast paired finetune (CPU gate)

Gate **PASS**. Winner of the rank key: `paired`.
This gate must **PASS** before a Lunar speed claim. Lunar commands are in `docs/gym-slow-fast.md`.
These numbers are a 2D pad. They are not Lunar landings.

One seed (`0`), fixed eval starts, no seed sweep. Rank key is
`landings − 0.5 × (mean steps among successes / horizon)` and is
eligible only when `adv_weight=1` and sample-point `b_cap` ran.
Steps count successful pad contacts only. `contact` is any ground hit,
so a crash can look fast there and still lose the rank key.

| Arm | Landings | Steps | Crash | Contact | Score | Rank key | adv | b_cap | diag MSE | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| zero | 1.000 | 37.12 | 0.000 | 37.12 | 0.710 | -1.000 | — | 0 | — | False |
| slow_only | 1.000 | 33.97 | 0.000 | 33.97 | 0.735 | 0.735 | 1.0 | 100 | 0.0003 | False |
| crash_fast | 0.000 | 64.00 | 1.000 | 6.77 | -0.500 | -0.500 | 1.0 | 100 | 0.0374 | False |
| unpaired | 0.000 | 64.00 | 1.000 | 10.40 | -0.500 | -0.500 | 1.0 | 100 | 0.1082 | False |
| supervised | 1.000 | 12.03 | 0.000 | 12.03 | 0.906 | -1.000 | 0.0 | 0 | 0.0000 | False |
| paired | 0.955 | 11.25 | 0.045 | 11.24 | 0.867 | 0.867 | 1.0 | 100 | 0.0009 | True |

Collector: 80 matched starts, 3058 rows, fast failures excluded 0. Mean slow steps 38.22, mean fast steps 12.05.

## Why the controls lose

- `zero` is the slow initialization. No GAN step, so the rank key is ineligible. It lands, and it is slower than the paired arm.
- `slow_only` is the same RpGAN step with the slow member as the target. It stays a successful slow landing and does not take the rank key.
- `crash_fast` puts crash actions in the fast slot. Contact can be sooner. Success collapses, so it does not win.
- `unpaired` uses fast actions from other starts. The student leaves the pad or hits too hard. That is not the same world flown faster.
- `supervised` matches the fast member with MSE and `adv_weight=0`. Landings may be excellent. The rank key rejects it because the #18 step did not run.

## Lunar path

Keep this gate green. The gym collector, trainer, and shared-seed eval are documented in `docs/gym-slow-fast.md`. They use the same paired-error step (`adv_weight=1`, `b_cap` every fourth update, diagnostic MSE outside the loss) and do not use the safe-fast kinematic cost. This board is not a Lunar result.

```bash
python -u examples/slow_fast_paired_2d.py
```
