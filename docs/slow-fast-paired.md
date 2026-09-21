# Slow→fast paired finetune (CPU toy)

The working controller is a competent slow law that already lands. The Lunar
failure was a finetune of that lander on nearest-state (slow, fast) pairs.
This gate encapsulates that miss. It does not claim Lunar landings.

`adv_weight` stays 1. There is no kinematic safe-fast cost.

## What the toy measures

State is lateral position, altitude, and both velocities. A success is ground
contact on the pad (`|x| <= 0.35`) with `|vx|` and `|vy|` at most `0.62`.
Anything else on contact, or a leave from the box, is a crash. Still airborne
at step 64 is a timeout. Rank is landings first, then fewer steps among
successes. An arm that drops landings or adds crashes is ineligible even if
the successes that remain are quick.

Two analytic laws label data. They are not the student.

- Slow: gentle sink, moderate lateral tracking. Lands in about 37 steps.
  The student starts here.
- Fast: stronger lateral tracking and an altitude flare. Lands in about 12
  steps. Used as a label.
- Crash: the fast lateral weights with a downward bias. It never enters the
  fast set.

Same-state rows store the fast law at the slow trajectory's own state.
Nearest-state rows paste one other episode's fast-trajectory action onto the
slow state when the states are within 0.75. That transplant is what `overwrite`
and `anchored` train on. Crashes stay out of both sets.

## Controller step

```text
neutral = slow action
target  = fast action (nearest-state for overwrite and anchored)
scale   = std(target - neutral), then gain so median row RMS is 1
real    = noise
fake    = noise + (student - target) / scale

D step: Rp logistic(real, fake) + sample-point b_cap every 4th update
G step: adv_weight * Rp logistic, adv_weight == 1, no action MSE
diag:   MSE(student, target) under no_grad, not in the loss
```

`anchored` does not train the slow weights. Its action is

```text
clamp(slow_action(state) + 0.15 * tanh(delta(state)), -1, 1)
```

with the last layer of `delta` initialized at zero. Learning rate is 0.01.
`overwrite` trains the full linear lander at learning rate 0.02 on the same
nearest-state rows.

## What has to lose

| Arm | What it is | Why it loses |
| --- | --- | --- |
| `zero` | Competent slow lander, no update | Ineligible (`b_cap` did not run). Lands slowly. |
| `slow_only` | Full weights, target is the slow member | Stays a successful slow landing. Loses on steps. |
| `crash_fast` | Full weights, crash actions | Landings collapse. |
| `unpaired` | Full weights, fast actions from other starts | Landings collapse. |
| `supervised` | MSE to the same-state fast member, `adv_weight=0` | May land quickly. Rejected because the GAN step is off. |
| `overwrite` | Full weights, nearest-state RpGAN, `adv_weight=1` | The Lunar miss. Landings fall. Later checkpoints stay off the pad. |
| `anchored` | Frozen lander, residual scale 0.15, same rows | The only arm that may win. |

The rank key is `(landings, -mean success steps)`. It is ineligible when
`adv_weight` is not 1, `b_cap` did not run, landings fall below 0.90, crashes
exceed 0.10, or a recorded checkpoint (every 100 steps for `overwrite` and
`anchored`) lost the pad.

## Run

```bash
python -u examples/slow_fast_paired_2d.py
```

Exit 0 is GATE PASS. The board is written to
[reports/slow_fast_paired/README.md](../reports/slow_fast_paired/README.md).
Lines are prefixed `[slow-fast]` so they can be tailed.

```bash
python -m unittest tests.test_slow_fast_paired
```

## Lunar path

This gate must PASS before the next Lunar retrain. The failed cuda:1 numbers
and the residual commands are in [gym-slow-fast.md](gym-slow-fast.md). A PASS
here is not a Lunar landing number. If the residual run loses landings, stop.
Do not raise the scale to 0.20.
