# ParticleGAN fine-tune (Arm A)

This experiment takes the [L2 control fine-tune](gym-control.md) and replaces its
paired reconstruction and imitation losses with the classic ParticleGAN game.
The graph stays the three-generator MoG model. It does not become a flat
`(st, at) -> st+1` network, and it does not use the
[Anima slider paired-error critic](gym-slider-gan.md).

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
| Auxiliary / tiny paired L2 | none in the imitation arm; the joint arm uses the rows above | **none** (weights locked at 0) |
| MoG variance/covariance regularizer on the raw particle table | kept | **kept** (not a state or action reconstruction) |
| Logged action/state/next MSE and contact BCE | optimization terms | **diagnostics only**, built under `no_grad` |

The relativistic game is the fine-tune objective. Each discriminator step and
each generator step scores four fake paths against the expert triple:

- **control** replaces action imitation. `E_control` sees state, previous
  command, and terrain. It does not see the expert current action or successor.
- **encoded** replaces real reconstruction. `E_pair` still sees the expert
  current action, as that L2 term did. Playback does not use `E_pair`.
- **prior** keeps the unconditional particle draw through G1/G2/G3.
- **composed** replaces synthetic reconstruction: `E_pair` reads the sampled
  prior state and action and G3 emits the successor.

Fake contacts shown to the critics are Bernoulli bits. Generator steps use the
straight-through sigmoid. Conditional diagnostics use probabilities.

## Critics, learning rates, and b_cap

Critics that train: **joint**, **action**, and the **shared state** critic with
a current/next role bit. Terrain is critic context. The gradient penalty is
taken with respect to the observation, not the terrain or the role bit.

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

Adam groups, before the shared cosine schedule (full rate for 60% of updates,
then down to a 0.05 floor):

| Group | Modules | Learning rate | Betas |
| --- | --- | ---: | --- |
| Generator / encoders | G1, G2, G3, E_pair, E_control | 6e-4 | (0, 0.999) |
| Prior | MoG means | 6e-2 | (0.5, 0.999) |
| Discriminator | joint, action, shared state | 9e-4 | (0, 0.999) |

EMA decay is 0.995 on G, both encoders, and the prior. Discriminator weights
are the live weights. Checkpoints do not store optimizer state.

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

CPU smoke (no landing score). Point `--checkpoint` and `--episodes` at the
frozen adversarial checkpoint and `episodes.json` when they are present:

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
