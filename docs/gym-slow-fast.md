# Lunar slow→fast paired finetune

The cuda:1 pairs were **strangers**: different successful episodes, matched by
nearest start, terrain, and state. Neutral was the slow action at `s`. Target
was the fast action at a nearby `s′`. There is no shared landing in that
edit. Fitting it destroyed a working controller. Do not train
`results/gym/lunar_lander_slow_fast/pairs.npz`. The trainer refuses
`slow_seed != fast_seed`. Collect has to be reworked to connected pairs
before the next cuda:1 run. This page does not claim a new Lunar landing number.

## What failed (pop-os cuda:1, `train_scope=control`)

Collect from YuE2 `#18` `particle_yue18_143320/best.pt`: **200/200** successful
landings, 63 fast / 62 slow, **60 pairs**, 13495 rows, `crashes_excluded=0`.
Train was 2500 steps, `adv_weight=1`, `safe_fast_weight=0`, about 59s, no late
diagnostic blow-up. Shared-seed validation versus the `#18` baseline:

| Arm | Landings | Success steps | Crashes |
| --- | ---: | ---: | ---: |
| **#18 baseline** | **20/20** | **205.4** | 0 |
| slow→fast @250 | 12/20 | 329.0 | 3 |
| slow→fast @1000 | **0/20** | — | 19 |
| slow→fast @2500 | **0/20** | — | 18 |

Eval selected none. Longer training was worse. Pair data looked healthy. The
failure was the update: nearest-state targets pasted a fast episode's action
onto a different state, and training `E_control` and G2 overwrote the lander.

## Toy gate

`python -u examples/slow_fast_paired_2d.py` must print `GATE PASS`. Both arms
use the same full lander, the same learning rate, and paired-error RpGAN at
`adv_weight=1`. The only difference is the pairs.

- `stranger` pastes another episode's fast action onto the nearest state.
  The rows are plentiful. Landings do not return at steps 100, 200, or 400.
- `connected` keeps the start. The target is a 0.25 retime of the fast law
  at that same state. Step 50 is off the pad. Steps 100, 200, and 400 are
  back on it, and success steps fall.

The first 50 updates kick either arm off the pad. Do not keep a checkpoint
from that window. `adv_weight` stays 1. `safe_fast_weight` stays 0. Diagnostic
MSE stays outside the loss. The kinematic plant cost is not the speed mechanism.

## Paths

| Role | Path |
| --- | --- |
| `#18` checkpoint | the file the failed run collected from |
| Failed stranger outputs | `results/gym/lunar_lander_slow_fast/particle/` (do not resume) |
| Stranger pairs, do not train | `results/gym/lunar_lander_slow_fast/pairs.npz` |
| Eval report | `reports/gym/lunar_lander_slow_fast/README.md` |

Collection seeds start at `591000`. Validation stays `391000–391019` and test
stays `491000–491049`. Those eval seeds are refused as collection seeds.

## Do not train the current pairs

```bash
python -u examples/slow_fast_paired_2d.py
```

That is the only command to run. The existing `pairs.npz` is stranger matching
(`slow_seed != fast_seed`). `train_gym_slow_fast.py` raises before it writes
a checkpoint. The next collector has to emit connected pairs: one successful
landing, re-timed, same seed, shared trajectory. A different episode's nearest
state is not that pair. After that collector exists, train with
`adv_weight=1` and `safe_fast_weight=0`, and stop if landings fall.

## Eval rule

Report landings and mean steps among successes, both on the shared seeds,
against the `#18` baseline. `mean_episode_steps` includes crashes and is not
the speed metric. A candidate is ineligible when landings fall, crashes rise,
or out-of-bounds rises. Validation selects an eligible checkpoint and writes
`best.pt`. Test is a separate `--split test` command after that selection.
A toy `GATE PASS` is not this table.

## CPU smoke

No simulator and no landing number:

```bash
python -u experiments/collect_slow_fast_lunar.py \
  --smoke \
  --out /tmp/slow_fast_smoke_pairs.npz \
  --live-log /tmp/slow_fast_collect.log
```

`tests/test_slow_fast_lunar.py` refuses a cross-seed pair file, then trains
4 CPU steps on a same-seed copy. That copy is only a plumbing check. It is
not a Lunar retime and it is not a landing number.
