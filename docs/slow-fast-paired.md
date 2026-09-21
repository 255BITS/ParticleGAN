# Slow→fast paired finetune (CPU toy)

The working Lunar controller is YuE2 paired-error RpGAN at `adv_weight=1`
([particle finetune](gym-particle-finetune.md)). The next mechanism is an
additional finetune from successful slow landings toward successful fast
ones, with that same controller step.

**Lunar real run is next and blocked until this gate PASSes.** This page does
not claim Lunar landings. Do not start the gym collection until that is
explicitly unblocked. `safe_fast_weight` is not this mechanism.

## What the toy measures

State is lateral position, altitude, and both velocities. A success is ground
contact on the pad (`|x| <= 0.35`) with `|vx|` and `|vy|` at most `0.62`.
Anything else on contact, or a leave from the box, is a crash. Still airborne
at step 64 is a timeout. Faster means fewer steps among successes.

Two analytic laws label data. They are not the student.

- Slow: gentle sink, moderate lateral tracking. Lands in about 37 steps.
- Fast: stronger lateral tracking and an altitude flare. Lands in about 12
  steps, still inside the speed limit.
- Crash: the fast lateral weights with a downward bias. Hits too hard. It is
  not a success, so it never enters the fast set.

The collector rolls both successful laws from the same starts. A pair is kept
only when both land and the fast rollout is strictly sooner. Training rows
are the slow trajectory's states, the slow action at that state, and the fast
action at that same state. That is the same world flown faster.

The student is a linear policy initialized at the slow law. Playback is the
student. The fast law is not copied into its weights.

## Controller step

Same formulation as the #18 particle finetune:

```text
neutral = slow action at this state
target  = fast action at this state
scale   = std(target - neutral), then gain so median row RMS is 1
real    = noise
fake    = noise + (student - target) / scale

D step: Rp logistic(real, fake) + sample-point b_cap every 4th update
G step: adv_weight * Rp logistic, adv_weight == 1, no action MSE
diag:   MSE(student, target) under no_grad, not in the loss
```

## What has to lose

| Arm | What it is | Why it loses the rank key |
| --- | --- | --- |
| `zero` | Slow initialization, no update | Ineligible (`b_cap` did not run). Lands slowly. |
| `slow_only` | Same GAN, target is the slow member | Stays a successful slow landing. |
| `crash_fast` | Fast slot filled with crash actions | Contact can be sooner. Success collapses. |
| `unpaired` | Fast actions from other starts | Not the same world. The student crashes. |
| `supervised` | MSE to the fast member, `adv_weight=0` | May land quickly. Rejected because the GAN step is off. |
| `paired` | Matched slow→fast RpGAN, `adv_weight=1` | The only arm that may win. |

The rank key is `landings − 0.5 × (mean steps among successes / horizon)`.
It is eligible only when `adv_weight` is 1 and `b_cap` was applied. A higher
raw score with `adv_weight=0` does not win.

## Run

CPU, one seed, well under a couple of minutes:

```bash
python -u examples/slow_fast_paired_2d.py
```

The process prints `[slow-fast]` lines as it goes (`python -u`, so they flush).
Exit 0 is GATE PASS. The board is written to
[reports/slow_fast_paired/README.md](../reports/slow_fast_paired/README.md).

```bash
python -m unittest tests.test_slow_fast_paired
```

## Lunar path

This toy must **GATE PASS** before a Lunar speed claim. The gym commands are
in [gym-slow-fast.md](gym-slow-fast.md): collect successful `#18` rollouts,
split by steps-to-land, drop crashes from the fast set, pair nearby starts,
and finetune with `controller_objective` (`adv_weight=1`, `safe_fast_weight=0`,
diagnostic MSE outside the loss). A PASS here is not a Lunar landing number.
