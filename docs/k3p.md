# K3P: the default ParticleGAN formulation

`get_recipe()` and `GANTrainer` train with **K3P**, the formulation that passed
all 22 declared toy gates plus the ring hold and its extension
([evidence](../reports/toy100/k3p-base/README.md)). It is a relativistic-paired
logistic GAN with a learned particle prior. Around the ordinary G/D/prior
updates it adds a critic penalty that changes with the critic's learning rate,
an EMA-critic anchor, a critic spike guard, sparse latent-row damping (A2), a
split learning-rate schedule and annealed input/output noise. All of these
are recipe hyperparameters; none is switched on or off by detecting the task.

## The penalty

With input dimension `d`, coefficient `c` and cap `κ` (both 1 by default):

```text
A = mean(||∇D(real)||² / d) + mean(relu(||∇D(fake)|| / √d − κ)²)
B = mean(relu(||∇D(real)|| − κ)²) + mean(relu(||∇D(fake)|| − κ)²)
P = mean(||∇D(real) − ∇D̄(real)||² / d)        D̄ = parameter EMA of D (decay .999)
s = max(0, min(1, 2r) − 2f) / (1 − 2f)          r = last critic LR / max critic LR
penalty = c/2 · (s·A + (1 − s)·(B + P))
```

`f` is `network_lr_floor` (.01). While the critic LR is at its peak, `s = 1`
and the penalty is exactly the RMS R1 + fake-cap form `A`. As the LR anneals,
it hands over to one-sided caps plus the anchor term `P`, which is zero for a
stationary critic, so it damps oscillation without flattening the critic at the
data. The anchor starts at the first blended call. With a constant LR, `s`
stays 1: same formulation, no separate code path, and no EMA forward.

`reg_every = k > 1` applies the same penalty every `k`-th step with
coefficient `k·c`.

## Defaults

| Field | Value | Meaning |
| --- | --- | --- |
| `reg_coeff`, `reg_kappa` | 1, 1 | penalty above |
| `reg_anchor_decay` | .999 | EMA critic decay per critic step |
| `betas`, `ema_decay` | (0, .999), .995 | Adam; G/prior EMA |
| `lr`, `d_lr_mult`, `prior_lr_mult` | .00425, 1, 2 | base rates |
| `lr_anneal_start`, `lr_floor` | .6, .05 | prior: hold 60%, cosine to 5% of the full budget |
| `network_lr_horizon_cap`, `network_lr_floor` | 1600, .01 | G and D: same cosine over `min(total, cap)` updates, then hold at 1% |
| `d_guard_ratio`, `d_guard_min_steps` | 5, 200 | clip a critic tensor whose grad RMS exceeds 5× its Adam RMS (0 disables) |
| `latent_damping_max_rate` | .5 | A2 on the particle table (0 disables) |
| `input_noise_std`, `input_noise_anneal_end` | .5, .1 | critic input noise, linear to 0 by 10% of training |
| `output_noise_std`, `output_noise_warmup` | .029, .2 | generator output noise, linear warmup over 20% |
| `prior_reg` | 0 | no particle spread penalty |
| `batch_size`, `z_dim`, `num_particles` | 2048, 2, 20000 | the qualified task shape |

`learning_rate_scales(step, recipe)` returns the `(network, prior)` LR
multipliers. `network_lr_horizon_cap=None` uses the full budget and
`network_lr_floor=None` reuses `lr_floor`.

## GANTrainer

`GANTrainer(recipe, G, D)` wires everything: it allocates the EMA critic
(`trainer.ema_D`, a frozen deep copy), builds the recipe's optimizers (which
allocate the A2 history buffer) and a noise stream, and saves and restores all
of them. Checkpoints use schema 3: the K3P state lives in the optimizer
states (`"regularizer"` entry) plus the noise stream. Schema-2 checkpoints
(separate `"k3p"` entry) are upgraded on load; schema-1 checkpoints (an older
formulation) are rejected with a clear error.

The trainer's EMA critic is robust: floating-point buffers are averaged,
integer buffers copied, and every anchor forward runs in the live critic's
train/eval mode on private buffer copies. BatchNorm statistics and
spectral-norm vectors of the EMA and of the live critic are never changed by
that forward, and no `.data` is swapped.

## Your own loop, and several critics

Do not instantiate K3P classes yourself. The recipe builds the
formulation into ordinary-looking PyTorch objects, the same ones the trainer
uses. Its step-time work runs inside the optimizers' `step()`, and all its
state is in their `state_dict()`:

```python
import copy
from particlegan import get_recipe, learning_rate_scales

recipe = get_recipe(total_steps=steps)
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty = recipe.make_critic_penalty(opt_d)   # reads EMA, LR record and step from opt_d
...
loss_d = adv + penalty(D, real, fake)
opt_d.zero_grad(); loss_d.backward(); opt_d.step()   # guard, Adam, anchor EMA + LR record
...
opt_g.zero_grad(); loss_g.backward(); opt_g.step()   # Adam with A2 latent damping
torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}, path)
```

`penalty.diagnostics()` returns host scalars such as the blend weight;
`make_critic_penalty(opt_d, collect_stats=True)` fills `penalty.last_stats`.
The penalty's step (for lazy application) is the critic optimizer's completed
step count + 1, so several calls per critic step share one step.

Extra arguments are conditioning, forwarded to the critic and its EMA; a
tuple/list output uses its first element (`output=` at construction selects
another layout):

```python
d_loss = d_loss + penalty(D, x_prev, fake.detach(), labels, xt=xt, t=t)
```

With one module that has several critic roles, pass the role's submodule; the
penalty evaluates the same-named EMA submodule. A module wrapping the critic
(e.g. `InputNoise(D, std, generator)`) works the same way:

```python
for role in D.roles():
    d_loss = d_loss + penalty(D.critic_for(role), xr, xf, ctx)
```

Critics with separate optimizers get separate pairs; their blend weights and
EMAs are independent. Nothing registers optimizer hooks or keeps module-level
state:

```python
opt_d2 = recipe.make_critic_optimizer(D2, ema_critic=copy.deepcopy(D2))
penalty2 = recipe.make_critic_penalty(opt_d2)
d2_loss = adv2 + penalty2(D2, real, fake)
opt_d2.zero_grad(); d2_loss.backward(); opt_d2.step()
```

The concrete classes (`K3PCriticAdam`, `K3PGeneratorAdam`, `CriticPenalty` and
the primitives `CriticAnchor`, `RobustCriticAnchor`, `CriticSpikeGuard`,
`LatentRowDamping`, `DirectParticleResponse`) stay importable from
`particlegan.k3p` for low-level tests and research, but they are not the public
API. Direct sample-particle groups use
`recipe.make_generator_optimizer(params, direct_particles=[...])`.
