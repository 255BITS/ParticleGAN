# Lunar slow→fast paired finetune

The cuda:1 pairs were **strangers**: different successful episodes, matched by
nearest start, terrain, and state. Neutral was the slow action at `s`. Target
was the fast action at a nearby `s′`. There is no shared landing in that
edit. Fitting it destroyed a working controller. Do not train
`results/gym/lunar_lander_slow_fast/pairs.npz`. The trainer refuses
`slow_seed != fast_seed`. Collect has to be reworked to two teachers, the same
seed, a both-land gate, and progress alignment before the next cuda:1 run.
This page does not claim a new Lunar landing number.

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

`python -u examples/slow_fast_paired_2d.py` must print `GATE PASS`. The pass
arm is two teachers on the same start.

- Safe teacher: land-first, no speed term.
- Fast teacher: the same spine, trained with an altitude speed bias. Speed
  rises until landings break. There is no separate crash policy and no
  hand-picked fraction of the fast action.
- Connected pairs: same seed, both land, aligned by progress `t/T`. Neutral
  is the safe action. Target is the fast action at that progress. On this
  plant the held teacher is update 2 (lands in 23.41 steps). The student
  keeps the pad and finishes in 28.31 steps.
- `overspeed` is update 3. The teacher still lands (20.04 steps). The same
  student ends at landings 0.762, crash 0.237.
- `crash_fast` is update 6, where the teacher itself lands 0.442. Those
  missed episodes stay out of the fast set. Training on them ends at
  landings 0.455, crash 0.545.
- `stranger` is cross-episode nearest state on the held teacher. The rows
  are plentiful (2905). The student ends at landings 0.205, crash 0.795.

`adv_weight` stays 1. `safe_fast_weight` stays 0. Diagnostic MSE stays
outside the loss. The kinematic plant cost is not the speed mechanism.

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
a checkpoint. The next collector has to roll two teachers on the same seed,
keep a pair only when both land, and align by progress. A different episode's
nearest state is not that pair. Push the fast teacher's speed term until
landings break, and leave crashed fast-teacher rows out of the file. After
that collector exists, train with `adv_weight=1` and `safe_fast_weight=0`,
and stop if landings fall.

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
not a two-teacher Lunar collect and it is not a landing number.
