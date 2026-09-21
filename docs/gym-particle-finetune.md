# ParticleGAN fine-tune (Arm A)

Playback is unchanged: `E_control(st, previous at) -> z -> G2 -> at`. The
controller step is YuE2's paired-error relativistic logistic loss on the
edit-normalized G2 residual, with sample-point `b_cap` on that critic.
`adv_weight` is locked at 1. A configured GAN with weight 0 is rejected.
Imitation and reconstruction L2 stay out of the graph.

This is the continuation of the collapsed four-path particle finetune. On the
shared-protocol board that arm scored **0/20 validation** and **2/50 test**
landings while diagnostic action MSE exploded. Those numbers are the reason
for this change. They are not a score for the paired-error recipe, and this
checkout does not remeasure them.

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

## Logs and commands

Lines flush to `log.txt`, `metrics.jsonl`, and the shared live log. The run
directory's `live.log` is a symlink.

```bash
tail -F results/gym/lunar_lander_particle_finetune/live.log
python -u examples/yue2_particle_2d.py
```

Full training uses GPU 1, a fresh directory, 2,500 updates, and batch 256.
The initialization checkpoint and expert episodes are the same files the L2
fine-tune used. Data seed stays `24002`.

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml
```

CPU smoke does not produce a landing score:

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml \
  --steps 2 --device cpu \
  --out-dir results/gym/lunar_lander_particle_finetune/smoke
```

Checkpoints at 250, 1,000, and 2,500 are the last EMA weights, not a
best-by-MSE pick. Landing selection has not been run for this objective.
The short report is
[here](../reports/gym/lunar_lander_particle_finetune/README.md).
