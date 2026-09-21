# Slow→fast paired finetune (CPU toy)

The working controller is a competent slow law that already lands. The Lunar
failure matched **different** successful episodes by nearest state. This gate
keeps that stranger pairing as a failing control and passes a same-start retime.
It does not claim Lunar landings.

`adv_weight` stays 1. There is no kinematic safe-fast cost. Both gate arms
use the same full lander and the same learning rate. The pairs are the difference.

## What the toy measures

State is lateral position, altitude, and both velocities. A success is ground
contact on the pad (`|x| <= 0.35`) with `|vx|` and `|vy|` at most `0.62`.
Anything else on contact, or a leave from the box, is a crash. Still airborne
at step 64 is a timeout. Rank is landings first, then fewer steps among
successes.

Two analytic laws label data. They are not the student.

- Slow: gentle sink, moderate lateral tracking. Lands in about 37 steps.
  The student starts here.
- Fast: stronger lateral tracking and an altitude flare. Lands in about 12
  steps. Used as a label from the **same start**.
- Crash: the fast lateral weights with a downward bias. It never enters the
  fast set.

`stranger` rows paste one other episode's fast-trajectory action onto a slow
state when the states are within 0.75. Thousands of rows survive that cut.
`connected` rows keep the slow state from a shared start and take 0.25 of the
step from the slow action to the fast law at that same state. Crashes stay
out of both sets.

## Controller step

```text
neutral = slow action at this state
stranger target = fast action at another episode's nearest state
connected target = slow action + 0.25 * (fast law(this state) - slow action)
scale   = std(target - neutral), then gain so median row RMS is 1
real    = noise
fake    = noise + (student - target) / scale

D step: Rp logistic(real, fake) + sample-point b_cap every 4th update
G step: adv_weight * Rp logistic, adv_weight == 1, no action MSE
diag:   MSE(student, target) under no_grad, not in the loss
```

Learning rate is 0.02 for every full-weight arm, including `stranger` and
`connected`.

The first 50 updates kick both of those arms off the pad. The return panel is
steps 100, 200, and 400. `stranger` is still off the pad there. `connected`
is back on it and faster. Do not keep a checkpoint from the first 50 updates.

## What has to lose

| Arm | What it is | Why it loses |
| --- | --- | --- |
| `zero` | Competent slow lander, no update | Ineligible (`b_cap` did not run). Lands slowly. |
| `slow_only` | Full weights, target is the slow member | Stays a successful slow landing. Loses on steps. |
| `crash_fast` | Full weights, crash actions | Landings collapse. |
| `unpaired` | Full weights, fast actions from other starts | Landings collapse. |
| `supervised` | MSE to the same-state fast member, `adv_weight=0` | May land quickly. Rejected because the GAN step is off. |
| `stranger` | Full weights, cross-episode nearest RpGAN, `adv_weight=1` | Plentiful pairs. Does not return to the pad. |
| `connected` | Full weights, same-start 0.25 retime, same step | The only arm that may win. |

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

This gate must PASS before Lunar collect is reworked. The current gym pairs
are the stranger arm. `train_gym_slow_fast.py` refuses `slow_seed != fast_seed`.
The failed cuda:1 numbers are in [gym-slow-fast.md](gym-slow-fast.md). A PASS
here is not a Lunar landing number.
