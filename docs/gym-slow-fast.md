# Lunar slow→fast paired finetune

This is the gym path that follows the CPU gate. The controller step is the
same one as `#18` / `particle.yaml`: paired-error RpGAN, `adv_weight=1`,
sample-point `b_cap` every fourth update. Diagnostic action MSE is logged
and is not in the loss. `safe_fast_weight` stays 0. The kinematic plant cost
is a different arm and is not used here.

The CPU toy must **GATE PASS** before a Lunar speed claim:

```bash
python -u examples/slow_fast_paired_2d.py
```

Exit 0 prints `GATE PASS`. That board is a 2D pad. It is not a Lunar landing
count. This checkout does not report Lunar landings.

## What is paired

Roll the frozen `#18` controller (`adv_weight=1`) on `LunarLander-v3`
continuous. Keep `successful_landing` only. Crashes, timeouts, and
out-of-bounds episodes are stored in the jsonl so they can be counted, and
they never enter the fast set. Successes are split by steps-to-land (default: at
or below the 30th percentile versus at or above the 70th). A pair is one slow
landing and one strictly faster landing with a nearby start and nearby
terrain. Rows use the slow trajectory: neutral is the slow action, target is
the fast action at the nearest state. Playback is still
`E_control(st, previous at) -> z -> G2`.

## Paths

| Role | Path |
| --- | --- |
| `#18` checkpoint | `results/gym/lunar_lander_particle_finetune/particle/best.pt` |
| Same weights if `best.pt` was not copied | `results/gym/lunar_lander_particle_finetune/particle/checkpoint_2500.pt` |
| Source episodes, overlap check only | `results/gym/lunar_lander/data/episodes.json` |
| Collected rollouts | `results/gym/lunar_lander_slow_fast/rollouts.jsonl` |
| Pairs | `results/gym/lunar_lander_slow_fast/pairs.npz` and `pairs.json` |
| Train log | `results/gym/lunar_lander_slow_fast/live.log` |
| Train outputs | `results/gym/lunar_lander_slow_fast/particle/` |
| Eval report | `reports/gym/lunar_lander_slow_fast/README.md` |

`best.pt` is the selected step-2500 `#18` controller. The trainer reads that
file; it does not read `particle_safe_fast.yaml`.

Collection seeds start at `591000`. Validation stays `391000–391019` and test
stays `491000–491049`, the same seeds as
`lib.gym_control_evaluation.freeze_protocol`. Those eval seeds are refused
as collection seeds.

## pop-os, GPU 1

```bash
python -u examples/slow_fast_paired_2d.py

python -u experiments/collect_slow_fast_lunar.py \
  --config configs/gym/lunar_lander_slow_fast/collect.yaml
tail -F results/gym/lunar_lander_slow_fast/live.log

python -u experiments/train_gym_slow_fast.py \
  --config configs/gym/lunar_lander_particle_finetune/particle_slow_fast.yaml
tail -F results/gym/lunar_lander_slow_fast/live.log

python -u experiments/evaluate_gym_slow_fast.py \
  --baseline results/gym/lunar_lander_particle_finetune/particle/best.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/particle/checkpoint_250.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/particle/checkpoint_1000.pt \
  --checkpoint results/gym/lunar_lander_slow_fast/particle/checkpoint_2500.pt \
  --device cuda:1 \
  --split validation
```

If `best.pt` is absent and `checkpoint_2500.pt` is the selected `#18` file,
pass that path as `--checkpoint` on the collector and as `--baseline` on the
eval. Point `particle_slow_fast.yaml` at the same file.

Pairing can be repeated without another rollout. The collector prints nearest
start and terrain distances. Raise a cap only after looking at those lines:

```bash
python -u experiments/collect_slow_fast_lunar.py \
  --from-jsonl \
  --rollouts results/gym/lunar_lander_slow_fast/rollouts.jsonl \
  --out results/gym/lunar_lander_slow_fast/pairs.npz \
  --max-start-distance 0.5 \
  --max-terrain-distance 0.5
```

## Eval rule

Report landings and mean steps among successes, both on the shared seeds,
against the `#18` baseline. `mean_episode_steps` includes crashes and is not
the speed metric. A candidate is ineligible when landings fall, crashes rise,
or out-of-bounds rises. Validation selects an eligible checkpoint and writes
`best.pt`. Test is a separate `--split test` command after that selection.
Do not treat a toy `GATE PASS` as this table.

## CPU smoke

No simulator and no landing number:

```bash
python -u experiments/collect_slow_fast_lunar.py \
  --smoke \
  --out /tmp/slow_fast_smoke_pairs.npz \
  --live-log /tmp/slow_fast_collect.log
```

Training still needs a real `#18` `best.pt` (or the unit test's tiny
checkpoint). `tests/test_slow_fast_lunar.py` builds that tiny checkpoint,
trains 4 steps on CPU, and checks `adv_weight=1`, one `b_cap` application,
and diagnostic MSE outside the loss.
