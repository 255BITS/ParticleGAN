# ParticleGAN fine-tune (Arm A)

Playback is still `E_control(st, previous at) -> z -> G2` on the three-generator
MoG graph. It is not a flat `(st, at) -> st+1` network and it does not use the
[Anima slider paired-error critic](gym-slider-gan.md).

The first version of this arm deleted paired imitation and reconstruction and
trained G, both encoders, the particle cloud, and the critics with Rp logistic
loss plus sample-point `b_cap`. On the shared fresh protocol that run selected
step 250 and landed 0/20 validation and 2/50 test (mean return about −113),
while L2 on the same protocol was 50/50. `diag_action_mse` grew from about 0.06
to above 1 during training. Those landing numbers were not remeasured here.

The continuation below is the model-glue winning recipe, checked on a CPU 2D
toy (`examples/particle_control_2d.py`) and wired into this trainer. It is not
a new Lunar landing result. GAN objects stay configured and are not applied.

```text
prior -> z -> G1 -> st
           -> G2 -> at
           -> G3 -> st+1

E_control(st, previous at, terrain) -> z -> G1 / G2 / G3
E_pair(st, current at, terrain) -> z -> G1 / G2 / G3
composed: prior (st, at) -> E_pair -> G3 -> st+1

D_joint(st, at, st+1)
D_action(at)
D_state(st, current role) and D_state(st+1, next role)  # shared weights

playback: E_control(st, previous at) -> z -> G2 -> at -> simulator.step(at)
```

Both encoders, all three generators, the MoG prior, and the three critics are
initialized from the validation-selected adversarial world-model checkpoint.
`E_control` starts as a copy of `E_pair`. The scaler is unchanged. Training
still uses the 9,297 heuristic-expert transitions from 47 training episodes.
Records stay shuffled. There is no trajectory loss and no simulator call inside
the update. The data-index seed matches the L2 fine-tune (`24002`); this is a
loss change, not a seed repeat.

## Which L2 terms changed

| Term | L2 fine-tune | This arm |
| --- | --- | --- |
| Action imitation MSE, `E_control -> G2` vs expert `at` | weight 1 | **removed** |
| Real reconstruction MSE + contact BCE, `E_pair(st, current at) -> G1/G2/G3` | weight 1 | **removed** |
| Synthetic reconstruction MSE + contact BCE, composed target detached | weight 1 | **removed** |
| Auxiliary / tiny paired L2 | none in the imitation arm; the joint arm uses the rows above | **none** (those three weights stay 0) |
| Model-glue action anchor | — | **0.1** normalized paired action error on G2 only |
| Model-glue functional match | — | **1.0** kinematic response, linear in the action |
| MoG variance/covariance regularizer on the raw particle table | kept | **configured, not applied** (cloud is frozen) |
| Logged action/state/next MSE and contact BCE | optimization terms | **diagnostics only**, built under `no_grad` |

The collapsed loop scored four fake paths with RpGAN. That loop is not the
training step anymore. The paths remain in `particle_game` for the configured
critic and are unused while `adv_weight` is 0:



- **control** replaces action imitation. `E_control` sees state, previous
  command, and terrain. It does not see the expert current action or successor.
- **encoded** replaces real reconstruction. `E_pair` still sees the expert
  current action, as that L2 term did. Playback does not use `E_pair`.
- **prior** keeps the unconditional particle draw through G1/G2/G3.
- **composed** replaces synthetic reconstruction: `E_pair` reads the sampled
  prior state and action and G3 emits the successor.

`particle_game` still knows those paths. The continuation does not call it.
Conditional diagnostics use probabilities and do not enter the loss.

## What trains

Only **G2** trains. G1, G3, `E_pair`, `E_control`, the MoG cloud, and the
critics stay frozen. `E_control` is still the playback encoder; it is not
updated. This matches model-glue `trainable_parts=heads` (output projection
only; stem and cloud fixed).

The step loss is `0.1 * paired action anchor + functional response`. The anchor
is per-coordinate variance-normalized MSE between G2 and the expert command.
The functional term is the model-glue response match through
`kinematic_response` on `(x, y, vx, vy)` and the physical action, including the
0.05 guided term at scale 4.5. That recipient is linear in the action, so the
functional gradient is a scaled action error. It is not Box2D, and it is not a
claim that the same weights will land.

Rp logistic `GANLoss` and sample-point `b_cap` (coeff 1, kappa 1, autograd L2,
every step) are still constructed and rejected if changed. `adv_weight` is 0,
so neither the discriminator nor the prior regularizer steps. Model-glue's
repaired `b_cap` screen used adversarial weight 0.001 and lost validation to
this supervised head continuation.

## Critics, learning rates, and b_cap

The critics remain in the checkpoint: **joint**, **action**, and the **shared
state** critic. They do not receive gradients in this continuation.

The loss objects come from the MoG recipe primitives:

- `GANLoss(loss_type="logistic", mode="rp")` — relativistic paired logistic.
  Discriminator steps sum the four roles. Generator steps use the joint term
  plus the mean of the action, current-state, and next-state terms
  (`marginal_weight` 1). Paths are averaged inside each role.
- `GradientPenalty(arm="b_cap", coeff=1, kappa=1, lazy_k=1, norm="l2", method="autograd", target_anneal="none")`.
  This is the **sample-point** cap, applied on real and fake observations.
  It is not `g_interp_cap` and not a finite-difference penalty.
- `MoGParticlePrior` with 1,024 components, z 32, and `ParticleRegularizer`
  once per generator step on the full raw center table.

Adam on G2 only, before the shared cosine schedule (full rate for 60% of
updates, then down to a 0.05 floor):

| Group | Modules | Learning rate | Betas |
| --- | --- | ---: | --- |
| Action head | G2 | 1e-5 | (0, 0.999) |
| Frozen | G1, G3, E_pair, E_control, prior, D | 0 | — |

Gradient clipping is 1.0 on G2. EMA decay is 0.98 on G (only G2 moves).
Checkpoints do not store optimizer state. Among checkpoints, the lowest proxy
action MSE on the first 256 training records is copied to `selected.pt`. That
proxy is not a landing rate.

## Logs and commands

Output goes to a fresh directory. The trainer refuses a nonempty one. Lines are
flushed to `log.txt`, `metrics.jsonl`, and the shared live log. The run
directory's `live.log` is a symlink to that shared file.

```bash
tail -F results/gym/lunar_lander_particle_finetune/live.log
```

Full training (GPU 1, fresh directory, 2,500 updates, batch 256). The
initialization checkpoint and expert episodes are the same files the L2
fine-tune used:

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml
```

CPU toy gate (no Lunar weights). This is the formulation check:

```bash
python -u examples/particle_control_2d.py
```

CPU smoke of the gym wiring (no landing score). Point `--checkpoint` and
`--episodes` at the frozen adversarial checkpoint and `episodes.json` when
they are present:

```bash
python -u experiments/train_gym_particle_finetune.py \
  --config configs/gym/lunar_lander_particle_finetune/particle.yaml \
  --steps 2 --device cpu \
  --out-dir results/gym/lunar_lander_particle_finetune/smoke
```

Checkpoints at 250, 1,000, and 2,500 are written for a later control rollout.
That evaluation has not been run. Do not read diagnostic MSE in the training
log as a landing rate, and do not treat this arm as better than the
[50/50 imitation fine-tune](../reports/gym/lunar_lander_control/README.md)
without that rollout. The short report is
[here](../reports/gym/lunar_lander_particle_finetune/README.md).
