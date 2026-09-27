# K3P: the default ParticleGAN formulation

`get_recipe()` and `GANTrainer` train with **K3P**: a relativistic-paired
(RpGAN) logistic GAN with a learned particle prior, a critic gradient penalty
without R1, an always-on EMA-critic anchor, a critic spike guard and sparse
latent-row damping (A2). Learning rates are constant, every optimizer is
AMSGrad, and there is no instance noise. All of these are recipe
hyperparameters; none is switched on or off by detecting the task.

## The penalty

With input dimension `d`, coefficient `c` (.3) and cap `κ` (1):

```text
x̂ = r + u·(f − r)            u ~ U(0, 1) per pair; reals and fakes paired by batch index
path = mean(relu(||∇D(x̂)|| / √d − κ)²)
fake = mean(relu(||∇D(f)|| / √d − κ)²)
prox = mean(||∇D(r) − ∇D̄(r)||² / d)        D̄ = parameter EMA of D (decay .999)
penalty = c/2 · (path + fake + prox)
```

Both caps are one-sided: a critic whose per-dimension slope stays under `κ`
pays nothing, so nothing pulls the slope at the data to zero (R1 does). The
path cap controls slope spikes anywhere between the data and the samples;
the anchor term is zero for a stationary critic and damps oscillation by tying
the critic's input gradient at the reals to that of its own parameter EMA. The
anchor starts at the first penalty call (the term is exactly zero then) and
the critic optimizer advances the EMA after every critic step.
`reg_anchor_weight` scales `prox` (0 removes it and needs no EMA critic).

`reg_every = k > 1` applies the same penalty every `k`-th step with
coefficient `k·c`. The path positions `u` come from the `generator` passed to
`make_critic_penalty` (`GANTrainer` passes its penalty stream), or the global
RNG.

## Defaults

| Field | Value | Meaning |
| --- | --- | --- |
| `reg_coeff`, `reg_kappa` | .3, 1 | penalty above |
| `reg_anchor_decay`, `reg_anchor_weight` | .999, 1 | EMA critic decay per critic step; weight of `prox` |
| `betas`, `amsgrad`, `ema_decay` | (0, .999), True, .995 | AMSGrad for G, prior and critic; G/prior EMA |
| `lr`, `d_lr_mult`, `prior_lr_mult` | .0085, .5, 2 | constant rates: G .0085, critic .00425, prior .017 |
| `d_guard_ratio`, `d_guard_min_steps` | 5, 200 | clip a critic tensor whose grad RMS exceeds 5× its Adam RMS (AMSGrad's max second moment; 0 disables) |
| `latent_damping_max_rate` | .5 | A2 on the particle table (0 disables) |
| `prior_reg` | 0 | no particle spread penalty |
| `batch_size`, `z_dim`, `num_particles` | 2048, 2, 20000 | the qualified task shape |

AMSGrad matters at a constant LR: with plain Adam the step creeps up during a
stationary hold, because the gradients shrink at equilibrium while the
second-moment estimate keeps decaying; AMSGrad's running maximum makes the step
shrink with the gradient instead.

### Optional schedule and noise

The recipe keeps an LR schedule and instance noise as knobs, off by default.
Lowering `lr_floor` (and optionally `network_lr_floor`,
`network_lr_horizon_cap`, `lr_anneal_start`) makes `GANTrainer` and
`learning_rate_scales(step, recipe)` anneal: G/D follow a cosine over
`min(total, cap)` updates down to `network_lr_floor`, the prior one over the
full budget down to `lr_floor`. `input_noise_std` / `output_noise_std` add
annealed critic input noise and warmed-up generator output noise in
`GANTrainer`. The penalty does not depend on the learning rate, so it is the
same formulation either way. In a caller-owned loop, `NetworkLRTransition`
starts the G/D decay when the caller's own validation metric plateaus; see
`scale_learning_rates` in the [API reference](api.md).

## GANTrainer

`GANTrainer(recipe, G, D)` wires everything: it allocates the EMA critic
(`trainer.ema_D`, a frozen deep copy), builds the recipe's optimizers (which
allocate the A2 history buffer) and its random streams (the penalty's path
positions among them), and saves and restores all of them. Checkpoints use
schema 3: the K3P state lives in the optimizer states (`"regularizer"` entry)
plus the streams. Schema-2 checkpoints
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
from particlegan import get_recipe, init

recipe = get_recipe(total_steps=steps)
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty = recipe.make_critic_penalty(opt_d)   # reads the EMA critic and step record from opt_d
...
loss_d = adv + penalty(D, real, fake)
opt_d.zero_grad(); loss_d.backward(); opt_d.step()   # guard, AMSGrad, anchor EMA
...
opt_g.zero_grad(); loss_g.backward(); opt_g.step()   # Adam with A2 latent damping
torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}, path)
```

`penalty.diagnostics()` returns host scalars such as the guard's clip count;
`make_critic_penalty(opt_d, collect_stats=True)` fills `penalty.last_stats`
(the path cap, fake cap and anchor terms).
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

Critics with separate optimizers get separate pairs; their EMAs are
independent. Nothing registers optimizer hooks or keeps module-level
state:

```python
init.deterministic_orthogonal_(D2, seed=3)   # optional: deterministic start for a fresh critic
opt_d2 = recipe.make_critic_optimizer(D2, ema_critic=copy.deepcopy(D2))
penalty2 = recipe.make_critic_penalty(opt_d2)
d2_loss = adv2 + penalty2(D2, real, fake)
opt_d2.zero_grad(); d2_loss.backward(); opt_d2.step()
```

The concrete classes (`K3PCriticAdam`, `K3PGeneratorAdam`, `CriticPenalty` and
the primitives `CriticAnchor`, `RobustCriticAnchor`, `CriticSpikeGuard`,
`LatentRowDamping`) stay importable from `particlegan.k3p` for low-level tests
and research, but they are not the public API.
