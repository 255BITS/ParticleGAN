# Slow→fast paired finetune (CPU toy)

Two teachers share one linear spine. The safe teacher is land-first: high
success, no speed term. The fast teacher is that same initialization trained
with an explicit altitude speed bias. Speed keeps rising until landings break.
The pass arm is the last speed a progress-aligned student can take. This file
does not claim Lunar landings.

`adv_weight` stays 1. There is no kinematic safe-fast cost and no hand-picked
fraction of the fast action. The student, the learning rate, and the RpGAN
step are the same on every full-weight arm. The pairs are the difference.

## What the toy measures

State is lateral position, altitude, and both velocities. A success is ground
contact on the pad (`|x| <= 0.35`) with `|vx|` and `|vy|` at most `0.62`.
Anything else on contact, or a leave from the box, is a crash. Still airborne
at step 64 is a timeout. Rank is landings first, then fewer steps among
successes.

The safe teacher is the frozen land-first law. It lands in about 37 steps.
The student starts there. The fast teacher copies those weights and then takes
Adam steps on

```text
landing penalty + 0.02 * leftover altitude
```

On this plant, one seed, the closed-loop curve is:

| Teacher update | Landings | Success steps | Crash |
| --- | ---: | ---: | ---: |
| 2 (held) | 1.000 | 23.41 | 0 |
| 3 (overspeed) | 1.000 | 20.04 | 0 |
| 6 (break) | 0.442 | 13.50 | 0.558 |

Update 2 is as fast as the progress-aligned student can learn. Update 3 is
faster and the student loses the pad. Update 6 is where the teacher itself
cannot land. Those missed episodes are the crash rows. They are not given a
separate downward-bias law.

Connected rows roll the **same start** with both teachers and keep the episode
only when both land and the fast landing is strictly sooner. Alignment is
progress `t/T` on that landing: the row state is the safe trajectory state,
and the target is the fast teacher's recorded action at the matching fraction
of its own landing. Stranger rows paste a different episode's fast action at
the nearest state. That cut still keeps thousands of rows.

## Controller step

```text
neutral = safe action at this state
connected target = fast action at progress t/T on the same landing
overspeed target = the same alignment, one speed update later
crash target = action from a fast-teacher episode that did not land
stranger target = fast action at another episode's nearest state
scale   = std(target - neutral), then gain so median row RMS is 1
real    = noise
fake    = noise + (student - target) / scale

D step: Rp logistic(real, fake) + sample-point b_cap every 4th update
G step: adv_weight * Rp logistic, adv_weight == 1, no action MSE
diag:   MSE(student, target) under no_grad, not in the loss
```

Learning rate is 0.02 for every full-weight arm.

`Early` is step 50. The return panel is steps 100, 200, and 400. `connected`
holds the pad on that panel (return min 1.000, final steps 28.31). `stranger`
is still on the pad at step 100 and off it by the end (landings 0.205, crash
0.795). `overspeed` ends at landings 0.762, crash 0.237.

## What has to lose

| Arm | What it is | Why it loses |
| --- | --- | --- |
| `zero` | Safe teacher, no update | Ineligible (`b_cap` did not run). Lands slowly. |
| `slow_only` | Full weights, target is the safe action | Stays a successful slow landing. Loses on steps. |
| `crash_fast` | Full weights, fast-teacher rows that missed | Landings 0.455, crash 0.545. Sooner contact does not count. |
| `unpaired` | Full weights, fast actions from other starts | Landings collapse. |
| `supervised` | MSE to the held progress target, `adv_weight=0` | Lands. Rejected because the GAN step is off. |
| `stranger` | Full weights, cross-episode nearest, `adv_weight=1` | Plentiful pairs. Landings fall, crashes rise. |
| `overspeed` | Full weights, progress pairs one update faster | Teacher still lands. Student does not keep the pad. |
| `connected` | Full weights, held-speed progress pairs | The only arm that may win. |

The rank key is `(landings, -mean success steps)`. It is ineligible when
`adv_weight` is not 1, `b_cap` did not run, landings fall below 0.90, crashes
exceed 0.10, or the return panel lost the pad.

## Run

```bash
python -u examples/slow_fast_paired_2d.py
```

Exit 0 is GATE PASS. The board is written to
[reports/slow_fast_paired/README.md](../reports/slow_fast_paired/README.md).
Lines are prefixed `[slow-fast]`.

```bash
python -m unittest tests.test_slow_fast_paired
```

## Lunar path

This gate must PASS before Lunar collect is reworked. The next collector
rolls a safe teacher and a speed-biased teacher on the **same seed**, keeps a
pair only when both land, and aligns by progress. Nearest-stranger pools stay
out. Push the speed term until landings break; do not shrink the fast action
by a fixed fraction, and do not train crashed fast-teacher rows.

The current gym pairs are the stranger arm. `train_gym_slow_fast.py` refuses
`slow_seed != fast_seed`. The failed cuda:1 numbers are in
[gym-slow-fast.md](gym-slow-fast.md). A PASS here is not a Lunar landing number.
