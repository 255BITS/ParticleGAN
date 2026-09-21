# Safe-fast landing term

`particle.yaml` is the YuE2 paired-error default (`adv_weight` 1, safe-fast
weight 0). That is the #18 controller. This note is the other file,
`configs/gym/lunar_lander_particle_finetune/particle_safe_fast.yaml`.

The first version of that file (PR #21) was trained on Lunar and did not
transfer. The revision below is what the closed-loop toy now accepts. It has
not been retrained on Lunar.

## What #21 did on Lunar

Same protocol as #18, shared seeds, 20 validation worlds and 50 test worlds.
Checkpoints 250, 1,000, and 2,500 were scored. These numbers are from that
run, not from this checkout.

| Arm | Val | Test | Mean return |
| --- | ---: | ---: | ---: |
| #21 `particle_safe_fast.yaml` as shipped | 0/20 | 0/50 | about −407 |
| YuE2 paired-error RpGAN #18 | 20/20 | 50/50 | about 287.7 |

The toy that shipped with #21 PASSed. It trained and scored the same bipolar
plant, so a sink parameter that descended in that plant looked like a
landing. Lunar never flew that plant.

## Why the gym term erased landings

`gym_shaping_cost` unrolled `kinematic_step` on the physical action. That
step reads column 0 as lateral acceleration and column 1 as a bipolar
vertical acceleration (negative means thrust downward). A Lunar command is
`[main, side]`. Main is an up-only engine: −1 is off, not reverse thrust.
The #21 cost therefore treated the side engine as the descent command and
pushed it negative. On the real lander that is a lateral burn off the pad.

The same yaml set `safe_fast_speed_limit: 0.62`. The success bonus paid for
any impact slower than that. On the closed-loop pad, 0.62 is a crash. An
ablation with the throttle map corrected and the 0.62 limit left in place
still dives through the soft threshold. Both bugs are enough. `adv_weight`
was 1 the whole time; the extra term was large enough to leave the expert.

## What the toy scores now

The scored plant is a closed loop. The main engine produces thrust
`(main+1)/2` in `[0, 1]`. The side engine is lateral. A landing is on the
pad (`|x| <= 0.35`) with both speeds at most `0.18`, inside 40 steps.
Anything faster, or off the pad, is a crash. A hover (sink 0) and a sink of
`0.40` both score −0.5. A sink of `0.16` scores about `0.70`.

The frozen #21 loss is the old unpack plus speed limit `0.62`, with the same
RpGAN at `adv_weight=1`. The loose-limit arm uses the new throttle map and
keeps `0.62`. The fixed arm uses the throttle map and `0.18`. GAN-only is
the slow expert. `adv_weight=0` is trained and rejected.

## Toy board

One seed (`0`), 200 starts. `python -u examples/safe_fast_2d.py`.

| Arm | Closed-loop landings | Steps | Crash | What it is |
| --- | ---: | ---: | ---: | --- |
| Hover | 0.000 | 40 | 0.000 | never descends |
| Crash sink 0.40 | 0.000 | 40 | 1.000 | too fast |
| Quick soft 0.16 | 1.000 | 24.10 | 0.000 | reference, score 0.699 |
| GAN only | 0.755 | 32.57 | 0.000 | slow expert, `adv_weight` 1 |
| #21 shipped | 0.000 | 40 | 0.675 | bipolar unpack, limit 0.62, side bias −0.9 |
| Loose limit | 0.000 | 40 | 1.000 | throttle map, limit 0.62, sink 0.529 |
| Fixed | 1.000 | 24.82 | 0.000 | throttle map, limit 0.18, sink 0.153, late GAN grad 0.035 |
| Supervised | 1.000 | 24.58 | 0.000 | fixed shaping, `adv_weight` 0, rejected |

The gate PASSes only when the #21 arm and the loose-limit arm both fail the
closed-loop score, and the fixed arm beats GAN-only by at least 0.15 landing
rate and 5 steps, with `adv_weight` 1 and a late GAN gradient above 0.01.

## Knob map

| Toy | Gym |
| --- | --- |
| Up-only main, side stays lateral | `safe_fast_action_map: throttle_up`. `bipolar_swap` is rejected. |
| #21 unpack of `[main, side]` as `[lateral, vertical]` | Removed from `gym_shaping_cost`. Frozen in the toy as `v21`. |
| Soft speed `0.18` | `safe_fast_speed_limit`. Values above `0.25` are rejected. `0.62` was #21. |
| Time `0.15`, crash `4`, success `2`, weight `1` | Same keys as before. |
| Horizon `40` | `safe_fast_horizon` |
| `adv_weight` 1 | `adv_weight`. Weight 0 is rejected. |
| Term absent | `particle.yaml`. `safe_fast_weight` defaults to 0 and the cost is not attached. |

The gym unroll still uses this integrator on `(x, y, vx, vy)` only. It is
not Box2D. L2 weights stay 0. The slider trainer is unchanged.

```bash
python -u examples/safe_fast_2d.py
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle_safe_fast.yaml
tail -F results/gym/lunar_lander_particle_finetune/safe_fast_live.log
```

The default path remains `particle.yaml`. Its log line is `SAFE-FAST off`.
A Lunar retrain of the revised yaml has not been run.
