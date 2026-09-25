# Safe-fast landing term

`particle.yaml` is the paired-error default. Every safe-fast weight there is
0, so the controller step is still YuE2 RpGAN at `adv_weight=1` and nothing
else. This note is the other arm:
`configs/gym/lunar_lander_particle_finetune/particle_safe_fast.yaml`.

Safe means the craft is on the pad with both speeds at or under the limit.
Fast means that happens in fewer steps. A hover that never descends is safe
and scores worst. A sink that hits the pad too hard scores with it. The
combined loss is the live GAN plus an explicit cost for time aloft, crashes,
and a bonus for a soft on-pad contact.

No Lunar landing number is claimed here. The gate is a CPU 2D plant.

## What the toy measures

State is lateral position, altitude, and both velocities. Altitude starts in
`[1.05, 2.35]`. The horizon is 48 steps. A landing is ground contact on the
pad (`|x| <= 0.35`) with `|vx|` and `|vy|` at most `0.62`. Anything faster, or
off the pad, is a crash. Still airborne at the horizon is a timeout.

The expert commands a sink of `0.10`. Paired-error RpGAN matches that expert,
so most starts are still in the air at step 48. The few that land take about
46 steps. The combined arm keeps `adv_weight=1` and adds

```text
safe_fast_weight * (time_weight * time_aloft + crash_weight * crash
                    - success_bonus * soft_on_pad_contact)
```

with `safe_fast_weight=1`, `time_weight=0.15`, `crash_weight=4`,
`success_bonus=2`. The score used only for the fixed references is
`landing_rate - 0.5 * (mean steps among successes / horizon)`. Timeouts use
the horizon as the step count, so a hover scores `-0.5`.

`adv_weight=0` is trained as an ablation. It lands, and the gate rejects it.
The combined sink (`0.430`) sits between the slow GAN (`0.101`) and that
rejected ablation (`0.527`), which is the GAN still pulling toward the expert.
`late_gan_grad` is the mean absolute GAN gradient on the sink parameter over
the last 40% of the 250 steps. It has to stay above `0.2` on the combined arm.

## Toy leaderboard

One seed (`0`), 200 fixed starts. Not a seed sweep. Not Lunar Lander.

| Arm | adv | safe-fast | Landings | Steps | Late GAN grad | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Hover (sink 0) | — | — | 0.000 | 48.00 | — | loses to a quick landing |
| Crash sink 0.90 | — | — | 0.000 | 48.00 | — | loses to a quick landing |
| Quick soft sink 0.55 | — | — | 1.000 | 19.30 | — | reference, score 0.799 |
| Baseline RpGAN | 1 | 0 | 0.015 | 45.67 | 0.00002 | slow expert match |
| Combined | 1 | 1 | 1.000 | 22.97 | 1.026 | pass |
| Supervised safe-fast | 0 | 1 | 1.000 | 19.89 | 0 | rejected |

Thresholds the combined arm has to clear: landing rate at least `0.95` and
at least `0.50` above the baseline; mean steps at most `28` and at least `12`
below the baseline. The baseline must land at most `0.25` of starts and take
at least `40` steps when it does. The sink gain trains with the recipe's
generator optimizer and the critic with the recipe's critic optimizer and
penalty, on the recipe's LR schedule. GAN-only must reach the slow expert's
sink (within `0.02`; nothing else acts on the gain), the combined arm needs a
late GAN gradient above `0.2`, and both GAN arms apply the critic penalty.

```bash
python -u examples/safe_fast_2d.py
```

`tests/test_safe_fast_2d.py` runs the same gate.

## Knob map

| Toy | Gym |
| --- | --- |
| `adv_weight=1` on paired-error RpGAN. Weight 0 is rejected. | `adv_weight`. `particle.yaml` and `particle_safe_fast.yaml` both set 1. |
| No safe-fast term. | `particle.yaml` omits the keys. Defaults are 0, and weight 0 does not attach the cost. |
| `safe_fast_weight=1` | `safe_fast_weight` |
| Time weight `0.15` | `safe_fast_time_weight` |
| Crash weight `4` | `safe_fast_crash_weight` |
| Success bonus `2` | `safe_fast_success_bonus` |
| Speed limit `0.62` | `safe_fast_speed_limit` |
| Pad half-width `0.35` | `safe_fast_pad_half` |
| Horizon `48` | `safe_fast_horizon` |

The gym cost is the same function on a kinematic unroll of `(x, y, vx, vy)`.
The physical action's first channel is the toy's vertical command and the
second is lateral. Angle and leg contacts stay at the batch values. This is
not Box2D, and the trainer still makes no simulator calls. L2 weights stay 0.
The slider arm is a different trainer and is unchanged.

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle_safe_fast.yaml
tail -F results/gym/lunar_lander_particle_finetune/safe_fast_live.log
```

The default path, still the paired-error recipe, is `particle.yaml`. Its log
line is `SAFE-FAST off`.
