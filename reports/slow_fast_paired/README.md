# Slow→fast paired finetune (CPU gate)

Gate **PASS**. Winner of the rank key: `paired`.
Lunar real run is **NEXT and blocked** until this gate PASSes.
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

## Next Lunar steps (do not run yet)

1. Collect successful rollouts from the #18 paired-error controller (`adv_weight=1`), not from the safe-fast cost.
2. Split those successes into slow and fast by steps-to-land. Drop crashes and timeouts from the fast set.
3. Build pairs on the same or a nearby initial condition.
4. Finetune with `controller_objective`: paired-error RpGAN, `adv_weight=1`, sample-point `b_cap` every fourth update, diagnostic MSE outside the loss. Neutral is the slow action. Target is the fast action.
5. Leave `safe_fast_weight` at 0. Do not set `adv_weight=0`.

```bash
python -u examples/slow_fast_paired_2d.py
```
