# ParticleGAN fine-tune (default)

**Default / recommended:** YuE2 paired-error RpGAN at `adv_weight=1`.
Weight 0 is rejected. On the shared Lunar control protocol this recipe scored
validation **20/20**, test **50/50**, mean return **287.7** at the selected
step 2500
([PR #18](https://github.com/255BITS/ParticleGAN/pull/18)).

| Arm | Val landings | Test landings | Test mean return |
| --- | ---: | ---: | ---: |
| **YuE2 paired-error RpGAN (`adv_weight=1`) — default** | **20/20** | **50/50** | **287.7** |
| L2 imitation (separate arm, same protocol) | 20/20 | 50/50 | 286.8 |
| Slider paired-error (separate arm) | — | 45/50 | — |
| Native #16 live `(record, z)` (not the default) | 1/20 | 4/50 | −66.8 |
| Collapsed pure RpGAN + `b_cap` (not the default) | 0/20 | 2/50 | −112.9 |

Those rows are cited from PR #18 and the earlier shared-protocol boards. This
checkout does not re-roll the simulator. L2
([L2 fine-tune](gym-l2-finetune.md)) and the slider fine-tune
([slider fine-tune](gym-slider-finetune.md)) stay separate arms. Their
configs are unchanged.

Playback is `E_control(st, previous at) -> z -> G2 -> at`. The controller
step is YuE2's paired-error relativistic logistic loss on the
edit-normalized G2 residual, with sample-point `b_cap` on that critic.
Imitation and reconstruction L2 stay out of the graph.

Landing selection is not `diag_action_mse` and is not inside the training
step. Score the EMA checkpoints with the shared control evaluator
(`lib/gym_control_evaluation.py`: validation seeds 391000–391019, test seeds
491000–491049; pick validation landing rate, then mean return, then the
earlier update). The command is below.

The collapsed four-path arm is why this recipe exists: diagnostic action MSE
exploded and landings fell to 0/20 and 2/50. That failure is the collapse toy,
not a score against the default.

The formulation is taken from the
[YuE2 concept-slider card](https://huggingface.co/ntc-ai/yue2-concept-sliders)
(`FORMULATION.md`, paired-error game, edit normalization, lazy `b_cap`, AR-only
training). Audio code does not run inside ParticleGAN. The 2D gate is
`python -u examples/yue2_particle_2d.py`.

## Controller step

```text
neutral = frozen init G2 action
target  = scaler.action(expert at)
scale   = std(target - neutral), then gain so median row RMS is 1
real    = noise
fake    = noise + (G2_action - target) / scale

D step: Rp logistic(real, fake) + sample-point b_cap every 4th update
G step: adv_weight * Rp logistic, adv_weight == 1, no action MSE
train:  E_control and G2
frozen: G1, G3, E_pair, prior, transition D
```

The critic is `GlobalMixErrorCritic` (`gmix_t8_w48_l1` on the card: 8 tokens,
width 48, 1 layer, 4 heads, score bound 8). Whitening must be
`paired_edit_per_coordinate_std_median_rms_gain`. Noise starts at
`max(edit_rms / 0.28, 0.03)` and holds at `1.3 * edit_rms`
(`FORMULATION.md`, normalize the paired edit; objectives, gradient cap).

`b_cap` is `GradientPenalty(arm="b_cap", coeff=1, kappa=1, lazy_k=4, norm="l2",
method="autograd")`. The every-fourth-step factor of 4 is the card's
compensation. The old four-path joint/action/state game remains in the library
and is not the controller step. Prior VIC is not applied because the prior is
frozen (`prior_loss=0` in the log).

Logged `diag_action_mse` is built under `no_grad`. It is not a landing rate
and it is not in the loss.

## What the 2D gate kept

| YuE2 card | 2D toy | Gym knob |
| --- | --- | --- |
| Paired-error RpGAN. A marginal critic is unchanged if rows exchange targets. `FORMULATION.md`, paired-error game. | Joint RpGAN + `b_cap` from a flipped sign stays flipped. Paired RpGAN + `b_cap` recovers the sign. | `controller_objective`, `adv_weight` 1 |
| Edit scale `std(target-neutral)`, median row RMS pinned to 1. | Neutral thrust is 0. | Neutral is the frozen init action. Absolute-target whitening is refused. |
| Lazy sample-point `b_cap` every 4th update, coeff 1, times 4. | Accepted arm requires applications > 0. | `edit_cap()`, logged `b_cap_applied` |
| Train AR QKVO only. Freeze NAR, MLP, embeddings, VAE. | Accepted arm trains the action sign only. | `train_scope: control` |
| Distillation rel-L2. The v2 teacher has no output MSE. `DISTILLATION.md`. | Supervised-only can land and is rejected because `adv_weight` is 0. | Not a knob. L2 weights stay 0. |
| Late-layer weighting is not in the card. | Not implemented. | Not a knob. |
| Strength sampling is a distillation detail. A linear action residual does not gain a second target from it. | Not required for the sign to recover. | Not a knob. Playback is full-strength G2. |
| UNI16 feature matching and end-margin logit MSE. v2 omits them. | Not implemented. | Not a knob. |

## CPU toys

`experiments/train_gym_particle_finetune.py` calls `controller_objective`.
It does not call the toys below. None of them is a Lunar landing count.

| Toy | Command | PASS / FAIL | Role |
| --- | --- | --- | --- |
| YuE2 paired-error | `python -u examples/yue2_particle_2d.py` | Exit 0 is **PASS**: joint RpGAN + `b_cap` misses the pad, supervised-only (`adv_weight=0`) is rejected, paired RpGAN + `b_cap` lands with weight 1. | Default CPU gate |
| Collapse repro (#17) | `python -u examples/particle_control_2d.py` | Exit 0 is `GATE COLLAPSE`: pure RpGAN + `b_cap` **FAILS** after imitation. A landing policy would exit 1. | Non-default. Baseline failure. [Note](particle-collapse-2d.md). |
| Native live pair (#16) | `python -u experiments/toy_particle_native_2d.py` | Old gate **PASS** is EMA action MSE ≤ 0.18 on the live `(record, z)` arm (recorded 0.1396). The observation-critic arm collapses (EMA ≥ 1, recorded 2.1065). Init paired error was 1.9997. The same recipe **failed Lunar** (val 1/20, test 4/50, mean return −66.8). | Non-default. Do not treat the old gate as the gym step. |
| Autopsy honest pad (#19) | `python -u experiments/toy_native16_autopsy.py` | Old gate: teacher-forced MSE ≤ 0.18. Honest gate: on-policy landing rate ≥ 0.80 with the learner's previous command. The #16 fixed arm passes the old gate and fails the honest one (landings 0.326). | Non-default. Why teacher-forced MSE lied. [Note](native16-autopsy.md). |

```bash
python -m unittest tests.test_yue2_particle_toy tests.test_particle_control_2d \
  tests.test_particle_native_2d tests.test_native16_autopsy
```

## Train, then select landings

Lines flush to `log.txt`, `metrics.jsonl`, and the shared live log. The run
directory's `live.log` is a symlink. Tail that file; `diag_action_mse` in it
is not a landing rate.

Full training uses GPU 1, a fresh directory, 2,500 updates, and batch 256.
The initialization checkpoint and expert episodes are the same files the L2
fine-tune used. Data seed stays `24002`. Defaults are
`configs/gym/lunar_lander_particle_finetune/particle.yaml` and `DEFAULTS` in
`experiments/train_gym_particle_finetune.py` (`adv_weight: 1.0`).

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml
tail -F results/gym/lunar_lander_particle_finetune/live.log
```

Checkpoints at 250, 1,000, and 2,500 are EMA weights, not a best-by-MSE pick.
The PR #18 board selected step 2500 by validation landings on the shared
control protocol. Recompute that selection with the control evaluator (same
seeds as `lib/gym_control_evaluation.py`: validation 391000–391019, test
491000–491049). The evaluator refuses `adv_weight` other than 1 and writes a
new report directory, so it does not overwrite the cited board:

```bash
python -u experiments/evaluate_gym_particle_finetune.py --freeze \
  --out reports/gym/lunar_lander_yue2 \
  --episodes results/gym/lunar_lander/data/episodes.json
python -u experiments/evaluate_gym_particle_finetune.py \
  --out reports/gym/lunar_lander_yue2 \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle/checkpoint_250.pt \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle/checkpoint_1000.pt \
  --checkpoint results/gym/lunar_lander_particle_finetune/particle/checkpoint_2500.pt \
  --final results/gym/lunar_lander_particle_finetune/particle/final.pt
```

CPU smoke does not produce a landing score:

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml \
  --steps 2 --device cpu \
  --out-dir results/gym/lunar_lander_particle_finetune/smoke
```

The cited board is
[here](../reports/gym/lunar_lander_particle_finetune/README.md).
