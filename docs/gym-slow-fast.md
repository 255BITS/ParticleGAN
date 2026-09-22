# Lunar slow→fast paired finetune

The cuda:1 pairs were **strangers**: different successful episodes, matched by
nearest start, terrain, and state. Fitting that edit destroyed a working
controller. That collector is disabled. `build_slow_fast_pairs` raises.
`train_gym_slow_fast.py` refuses `pairs.npz`, any name containing
`do_not_train`, a manifest that is not `progress_same_seed` / `t/T`, and any
row with `slow_seed != fast_seed`.

The replacement matches the toy: a frozen safe teacher, a speed-biased fast
teacher from the same spine, and pairs kept only when the **same seed** lands
under both. Rows are aligned by progress `t/T`. Crashed fast-teacher episodes
stay out of the file. This page does not claim a new Lunar landing number.
Slider runs the cuda:1 commands below. This checkout does not.

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

Eval selected none. Longer training was worse. Nearest-state targets pasted a
fast episode's action onto a different state, and training `E_control` and G2
overwrote the lander. Do not resume
`results/gym/lunar_lander_slow_fast/particle/`.

## Toy gate

`python -u examples/slow_fast_paired_2d.py` prints `GATE PASS`. The pass arm
is two teachers on the same start.

- Safe teacher: land-first, no speed term.
- Fast teacher: the same spine, trained with an altitude speed bias. A crashy
  overall rate is allowed. There is no separate crash policy and no
  hand-picked fraction of the fast action.
- Connected pairs: same seed, both land, aligned by progress `t/T`. On this
  plant the passing student uses update 2 (teacher 23.41 steps, student 28.31).
  The same full-weight student does not fly update 6's both-land rows. Lunar
  does not copy update 2 as the hold.
- `overspeed` is update 3 (teacher 20.04 steps). The same student ends at
  landings 0.762, crash 0.237.
- `crash_fast` is update 6, where the teacher itself lands 0.442. Those
  missed episodes stay out of the fast set.
- `stranger` is cross-episode nearest state. The student ends at landings
  0.205, crash 0.795.

`adv_weight` stays 1. `safe_fast_weight` stays 0. Diagnostic MSE stays
outside the student loss. The kinematic plant cost is not the student's
speed mechanism. The fast teacher's own loss is a separate altitude term,
recorded as `speed_term`, not as `safe_fast_weight`.

The toy winner trains full policy weights. The gym student command below
still freezes `E_control` and G2 and trains an action residual of scale
0.15, which is what `train_gym_slow_fast.py` accepts. That residual is not
the measured toy winner. It is the locked student step, now fed same-seed
progress pairs instead of strangers. `adv_weight` stays 1.

## cuda:1 commands

GPU 0 is not used. Logs are prefixed `[fast-teacher]`, `[slow-fast-collect]`,
and `[slow-fast]`. Tail them while the job runs:

```bash
tail -f results/gym/lunar_lander_slow_fast/fast_teacher.log
tail -f results/gym/lunar_lander_slow_fast/collect.log
tail -f results/gym/lunar_lander_slow_fast/student.log
```

The yaml default `#18` path is
`results/gym/lunar_lander_particle_finetune/particle/best.pt`. The failed
cuda:1 collect used YuE2
`results/gym/lunar_lander_particle_finetune/particle_yue18_143320/best.pt`.
Pass that path, or whichever file is the frozen lander, on every `#18`
argument below. The fast teacher, the collector's safe teacher, the student
init, and the eval baseline must be the same checkpoint.

```bash
python -u experiments/train_gym_fast_teacher.py \
  --config configs/gym/lunar_lander_slow_fast/fast_teacher.yaml \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle_yue18_143320/best.pt

python -u experiments/collect_slow_fast_lunar.py \
  --config configs/gym/lunar_lander_slow_fast/collect.yaml \
  --safe-checkpoint results/gym/lunar_lander_particle_finetune/particle_yue18_143320/best.pt \
  --fast-checkpoint results/gym/lunar_lander_slow_fast/fast_teacher/held.pt

python -u experiments/train_gym_slow_fast.py \
  --config configs/gym/lunar_lander_particle_finetune/particle_slow_fast.yaml \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle_yue18_143320/best.pt

python -u experiments/evaluate_gym_slow_fast.py \
  --baseline results/gym/lunar_lander_particle_finetune/particle_yue18_143320/best.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/student/checkpoint_250.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/student/checkpoint_1000.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/student/checkpoint_2500.pt \
  --split validation \
  --device cuda:1
```

Omit the `--checkpoint` / `--safe-checkpoint` / `--baseline` overrides only
when `particle/best.pt` is that same `#18` file.

### Fast teacher

`configs/gym/lunar_lander_slow_fast/fast_teacher.yaml`. Starts from `#18`.
Delete `results/gym/lunar_lander_slow_fast/fast_teacher/` before rerunning.
That directory's stage-2 `held.pt` is the near-safe checkpoint and the trainer
refuses a nonempty output directory.
Trains `E_control` and G2 only. The loss on the safe teacher's successful
probe states is

`speed_bias * (main_engine * altitude) + anchor_weight * MSE(action, frozen safe action)`.

`speed_bias` starts at 0.05 and doubles each stage (`speed_growth: 2`), 40
steps per stage, 8 stages, Adam lr `1e-4`. The schedule does not stop when
landings fall. The anchor stops the first step from wiping the engine. It is
not a hand-picked fraction of the action. After each stage the script probes
seeds `581000–581019` (disjoint from validation, test, and collection). A
crashy probe is expected: a few landings and many crashes is a valid stage.

`held.pt` is chosen after every stage. It is the stage with the fewest mean
success steps among stages that landed at least `min_held_landings` times
(default 1). A later crashy stage beats an earlier 20/20. A stage that still
matches the safe landing count must also be at least 10% faster than the safe
probe, so a 20/20 result one step quicker is not written. If nothing qualifies,
the process exits 1 and does not write `held.pt`. Do not collect in that case.
Every stage is also saved as `stage_N.pt`. Crashed probe episodes are not a
training set.

### Collect

`configs/gym/lunar_lander_slow_fast/collect.yaml`. The fast teacher flies
every seed from `591000` (`episodes: 200`) first. The safe teacher flies a
seed only when that fast rollout's outcome is `successful_landing`. Seeds the
fast teacher misses are not flown by the safe teacher. A pair is still kept
only when both land and the fast landing is strictly sooner. Crashes,
timeouts, and flyaways never become targets. Every kept safe-trajectory state
is a row. The target is the fast action at the same fraction `t/T` on that
seed. The manifest is `pairing=progress_same_seed`, `alignment=t/T`. Output is
`results/gym/lunar_lander_slow_fast/pairs_progress.npz`.

If `pairs.npz` is still in that directory, collect renames it to
`pairs_stranger_do_not_train.npz` (and the `.json` sidecar) before writing.
It refuses to overwrite an existing archive. It also refuses to write a new
file named `pairs.npz`. The fast checkpoint must be `gym_fast_teacher_v1`.

### Student

`configs/gym/lunar_lander_particle_finetune/particle_slow_fast.yaml`.
`adv_weight: 1`, `safe_fast_weight: 0`, `train_scope: residual`,
`residual_scale: 0.15`. Pairs are `pairs_progress.npz`. Output is a fresh
`results/gym/lunar_lander_slow_fast/student/` directory. Do not resume
`particle/` or `residual/`. The trainer rejects a nonempty `out_dir`.

### Eval

Landings first, then mean steps among successes, on the shared seeds, against
the `#18` baseline. A candidate is ineligible when landings fall, crashes
rise, or out-of-bounds rises. `mean_episode_steps` includes crashes and is
not the speed metric. Validation writes `best.pt` only for an eligible
checkpoint. Test is a later `--split test` command. A toy `GATE PASS` is not
this table.

## Paths

| Role | Path |
| --- | --- |
| `#18` checkpoint | `particle/best.pt`, or `particle_yue18_143320/best.pt` when that is the lander |
| Fast teacher held file | `results/gym/lunar_lander_slow_fast/fast_teacher/held.pt` |
| Progress pairs | `results/gym/lunar_lander_slow_fast/pairs_progress.npz` |
| Student run | `results/gym/lunar_lander_slow_fast/student/` |
| Stranger pairs, do not train | `pairs.npz` (archived to `pairs_stranger_do_not_train.npz`) |
| Failed stranger outputs | `results/gym/lunar_lander_slow_fast/particle/` (do not resume) |
| Eval report | `reports/gym/lunar_lander_slow_fast/README.md` |

Collection seeds start at `591000`. Fast-teacher probes start at `581000`.
Validation stays `391000–391019` and test stays `491000–491049`.

## CPU smoke

No simulator and no landing number:

```bash
python -u experiments/collect_slow_fast_lunar.py \
  --smoke \
  --out /tmp/slow_fast_smoke_pairs.npz \
  --live-log /tmp/slow_fast_collect.log
```

`tests/test_slow_fast_lunar.py` checks progress pairs, the stranger-name
refusal, a 4-step residual student on those fake pairs, and a 2-step fast
teacher on random states. The fast-teacher smoke log says it is not a Lunar
landing.
