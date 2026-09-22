# ParticleGAN fine-tune (Arm A)

Playback is unchanged: `E_control(st, previous at) -> z -> G2 -> at`. The
controller step is YuE2's paired-error relativistic logistic loss on the
edit-normalized G2 residual, with sample-point `b_cap` on that critic.
`adv_weight` is locked at 1. A configured GAN with weight 0 is rejected.
Imitation and reconstruction L2 stay out of the graph.
`configs/gym/lunar_lander_particle_finetune/particle.yaml` is this recipe:
the safe-fast weights default to 0, so they are not in the graph.
The optional safe-fast arm is a different file,
[`particle_safe_fast.yaml`](gym-safe-fast.md).
The locked-shared stamp is a third file,
[`locked_shared.yaml`](#locked-shared-arm). It does not replace this recipe.

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

## Locked shared arm

`configs/gym/lunar_lander_particle_finetune/locked_shared.yaml` trains with
the same script. `adv_posture: locked_shared` builds the Rp logistic loss and
the sample-point cap from
[`particlegan.locked_shared`](locked-shared.md) (`make_gan_loss`,
`make_b_cap` on `LOCKED_SHARED`). The yaml does not copy those pins.
`particle.yaml` stays YuE2 Arm A (`adv_posture: yue2`, the default when the
key is omitted).

| Knob | `locked_shared.yaml` | `particle.yaml` (YuE2) |
| --- | --- | --- |
| Rp logistic | `make_gan_loss()` | `rp_d_loss` / `rp_g_loss` (same logistic kernel) |
| `b_cap` coeff, kappa, norm, method, anneal | 1, 1, `l2`, autograd, none via `make_b_cap()` | same numbers via `edit_cap()` |
| `lazy_k` | stamp `1` (every step; coeff is not multiplied) | `EDIT_CAP_EVERY` `4`, and the penalty is multiplied by 4 |
| `adv_weight` | 1 | 1 |
| safe-fast | off; a nonzero weight is rejected | off here; the other file is `particle_safe_fast.yaml` on `yue2` |
| critic | host `GlobalMixErrorCritic` (`gmix_t8_w48_l1`) | same |
| prior | frozen checkpoint prior | same |
| FM, cover, 12-particle cloud, `particle_l2` | not in this step | not in this step |

`n_particles` 12, `particle_l2` 0.02, and `z_dim` 2 are the demo cloud. This
step does not build that cloud, so those fields are not applied. The prior
stays the frozen initialization prior. `cover_weight` 1.5 is the demo cover;
this controller has no cover term. `fm_weight` 0 matches, because feature
matching is not in the graph. `critic=host` matches: the critic stays the gym
edit critic, not a Music MLP. `pairing=live` matches the paired edit. That
is the gym host posture. It is not a second copy of the cap.

No Lunar landing score has been measured for this arm. A pop-os rollout is
follow-up.

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/locked_shared.yaml
tail -F results/gym/lunar_lander_particle_finetune/locked_shared_live.log
```

CPU smoke does not produce a landing score. It still needs the initialization
checkpoint and expert episodes:

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/locked_shared.yaml \
  --steps 2 --device cpu \
  --out-dir results/gym/lunar_lander_particle_finetune/locked_shared_smoke
```

## Kept example, not the gym step

`experiments/toy_particle_native_2d.py` and `tests/test_particle_native_2d.py`
are the earlier live `(record, z)` RpGAN gate. They stay as a CPU example.
`experiments/train_gym_particle_finetune.py` does not call that game. Its
controller step is `controller_objective`.

That toy's own readout is unchanged: observation-critic EMA action MSE 2.1065
(collapse, threshold ≥ 1) and live-pair EMA action MSE 0.1396 (pass, threshold
≤ 0.18). Init paired error was 1.9997. Those figures are not Lunar Lander
landings.

## Logs and commands

Lines flush to `log.txt`, `metrics.jsonl`, and the shared live log. The run
directory's `live.log` is a symlink. The locked-shared arm writes
`results/gym/lunar_lander_particle_finetune/locked_shared_live.log` instead,
so the two arms can be tailed separately.

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
