# ParticleGAN fine-tune (Arm A)

This experiment takes the [L2 control fine-tune](gym-control.md) and replaces its
paired reconstruction and imitation losses with the classic ParticleGAN game.
The graph stays the three-generator MoG model. It does not become a flat
`(st, at) -> st+1` network, and it does not use the
[Anima slider paired-error critic](gym-slider-gan.md).

```text
prior -> z -> G1 -> st          # frozen
           -> G2 -> at          # trained
           -> G3 -> st+1        # frozen

E_control(st, previous at, terrain) -> z -> G   # trained
E_pair(st, current at, terrain) -> z -> G       # frozen

D(record, z)   # trained; terrain is context, not a penalty input
playback: E_control(st, previous at) -> z -> G2 -> at -> simulator.step(at)
```

G, `E_pair`, the MoG prior, and the observation critics are initialized from
the validation-selected adversarial world-model checkpoint. `E_control` starts
as a copy of `E_pair`. `D(record, z)` is new. The scaler is unchanged. Training
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
| MoG variance/covariance regularizer on the raw particle table | kept | **logged only** (the table stays frozen with the world model) |
| Logged action/state/next MSE and contact BCE | optimization terms | **diagnostics only**, built under `no_grad` |

Paired L2 stays at weight 0. The generator step that updates `E_control` and G2
is the relativistic loss alone (`adv_weight=1`, `supervised_only=false`). There
is no action-MSE anchor.

An observation critic D(record) only matches the marginal of transitions. It
does not make `E_control(st, previous at)` emit the latent of that expert
transition, so paired action error is free to rise. `examples/five_modes.py`
drops reconstruction and still inverts the generator because the critic scores
the joint pair `(x, z)` and the encoder stays on the live real side of the
RpGAN loss. Detaching that real pair puts the game back on observations only.
This arm uses that live pair.

Fake contacts shown to the critic are Bernoulli bits. Generator steps use the
straight-through sigmoid. Conditional diagnostics use probabilities.

- **control** is the live pair `(record, z)` with `z = E_control(st, previous at)`.
  The encoder does not see the expert current action or the successor.
- **prior** is `(G(z), z)` for an unconditional MoG draw. It is the other side
  of the same relativistic pair.

Encoded and composed reconstruction paths are not in this loss. They were
observation-space stand-ins for the L2 terms this arm removed.

## Critics, learning rates, and b_cap

The critic that trains is a fresh `D(record, z)`. Terrain is context and is not
part of the penalty input. Observation joint, action, and state critics stay
frozen and unused. G1, G3, `E_pair`, and the MoG table stay frozen, which is
the control-stack scope: a new critic cannot drag the world model. The table's
variance regularizer is logged and not stepped.

The loss objects are the locked MoG toy defaults:

- `GANLoss(loss_type="logistic", mode="rp")` — relativistic paired logistic.
  Discriminator and generator steps both use it. The generator loss is the
  controller update.
- `GradientPenalty(arm="b_cap", coeff=1, kappa=1, lazy_k=1, norm="l2", method="autograd", target_anneal="none")`.
  Sample-point cap on `(record, z)` for the control pair, every step. Not
  `g_interp_cap` and not a finite-difference penalty.

Adam groups, before the shared cosine schedule (full rate for 60% of updates,
then down to a 0.05 floor):

| Group | Modules | Learning rate | Betas |
| --- | --- | ---: | --- |
| Control | G2, E_control | 6e-4 | (0, 0.999) |
| Latent-joint critic | D(record, z) | 9e-4 | (0, 0.999) |
| Frozen | G1, G3, E_pair, MoG table, observation D | — | — |

EMA decay is 0.995 on G, both encoders, and the prior. The latent-joint critic
uses its live weights. Checkpoints do not store optimizer state. Playback reads
the EMA `E_control` and G2.

## CPU 2D gate

`experiments/toy_particle_native_2d.py` is the same pairing on a 2D record
`(state, action)` whose action is `tanh(-2.2 * previous)`. State does not set
the action, so an observation critic can match `p(state, action)` without
learning the pair. Both arms use Rp logistic and sample-point `b_cap` at
coefficient 1. The fixed arm's generator loss has adversarial weight 1 and L2
weight 0.

| Arm | EMA action MSE | Gate |
| --- | ---: | --- |
| Current observation game, every module trained | 2.1065 | collapse, threshold ≥ 1 |
| Fixed live `(record, z)` pair, E_control and G2 only | 0.1396 | pass, threshold ≤ 0.18 |

Init paired error was 1.9997. Elapsed time was about 6 seconds on CPU.
`tests/test_particle_native_2d.py` runs this gate.

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
