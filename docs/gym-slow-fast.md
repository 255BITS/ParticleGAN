# Lunar slow→fast paired finetune

The cuda:1 run that unfroze `#18` and fit nearest-state fast actions
**destroyed landings**. The next recipe freezes that controller and trains a
bounded action residual. Do not start that retrain until the CPU gate
PASSes. This page does not claim a new Lunar landing number.

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

## Fix, locked by the CPU gate

`python -u examples/slow_fast_paired_2d.py` must print `GATE PASS` before the
next gym train. On that plant the same nearest-state pairs do this:

- `overwrite` trains every lander weight with paired-error RpGAN at
  `adv_weight=1`. Landings go to 0. The rank key rejects it.
- `anchored` freezes the lander and trains a residual of at most **0.15** per
  channel, same rows, same RpGAN step, `b_cap` every fourth update. Every
  recorded checkpoint stays on the pad, and success steps fall. Diagnostic
  MSE stays outside the loss.

Scale 0.20 dipped landings on the toy, so the gym config locks `residual_scale`
at 0.15. `adv_weight` stays 1. `safe_fast_weight` stays 0. The kinematic plant
cost is not the speed mechanism. `train_scope=control` is rejected.

The gym residual is

```text
physical = clamp(tanh(G2).detach() + 0.15 * tanh(head(state, previous, terrain)), -1, 1)
loss     = controller_objective on scaler.action(physical)
```

`head` is zero-initialized, so step 0 is the loaded `#18` action. Learning
rates stay constant: residual `0.01`, critic `1e-3`. The world-model rate is
not used on the lander. Playback for eval adds the residual. A checkpoint
without a residual key plays `E_control -> G2` alone, which is how the failed
run should still be scored.

## Paths

| Role | Path |
| --- | --- |
| `#18` checkpoint | the file the pairs were collected from |
| Failed overwrite outputs | `results/gym/lunar_lander_slow_fast/particle/` (do not resume) |
| Pairs, reusable | `results/gym/lunar_lander_slow_fast/pairs.npz` |
| Next train outputs | `results/gym/lunar_lander_slow_fast/residual/` |
| Next train log | `results/gym/lunar_lander_slow_fast/residual.log` |
| Eval report | `reports/gym/lunar_lander_slow_fast/README.md` |

Collection seeds start at `591000`. Validation stays `391000–391019` and test
stays `491000–491049`. Those eval seeds are refused as collection seeds.

## Next cuda:1 command

Reuse the pairs. Fresh output directory. Point `--checkpoint` at the same
`#18` `best.pt` that collected the pairs (`particle_yue18_143320/best.pt` on
the machine that ran the failed job). If landings fall, stop. Do not raise
the residual scale and do not unfreeze `E_control` or G2.

```bash
python -u examples/slow_fast_paired_2d.py

python -u experiments/train_gym_slow_fast.py \
  --config configs/gym/lunar_lander_particle_finetune/particle_slow_fast.yaml \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle_yue18_143320/best.pt \
  --pairs results/gym/lunar_lander_slow_fast/pairs.npz \
  --out-dir results/gym/lunar_lander_slow_fast/residual \
  --device cuda:1
tail -F results/gym/lunar_lander_slow_fast/residual.log

python -u experiments/evaluate_gym_slow_fast.py \
  --baseline results/gym/lunar_lander_particle_finetune/particle_yue18_143320/best.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/residual/checkpoint_250.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/residual/checkpoint_1000.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/residual/checkpoint_2500.pt \
  --device cuda:1 \
  --split validation
```

The log line to look for is `RESIDUAL frozen=#18 scale=0.15`. `E_control and
G2 are not updated` must stay true. A nonempty `residual/` directory is
refused; pick another fresh directory instead of deleting a scored run.

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

`tests/test_slow_fast_lunar.py` trains 4 CPU steps on a tiny checkpoint and
checks that `E_control` and G2 match the init file, the playback edit is at
most 0.15, `adv_weight=1`, and one `b_cap` application ran.
