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

`reg_every = k > 1` applies the same K3P penalty every `k`-th step with
coefficient `k·c`. It never changes the technique.

## Defaults

| Field | Value | Meaning |
| --- | --- | --- |
| `reg_arm`, `reg_coeff`, `reg_kappa` | `k3p`, 1, 1 | penalty above |
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

`GANTrainer(recipe, G, D)` wires everything: it allocates the EMA critic, the
A2 history buffer and a noise stream, and saves and restores all of them.
Checkpoints use schema 2 (`"k3p": {"critic", "latent"}` plus the noise
stream). Schema-1 (GAN v3) checkpoints are rejected with a clear error.

The trainer's EMA critic is robust: floating-point buffers are averaged,
integer buffers copied, and every anchor forward runs in the live critic's
train/eval mode on private buffer copies. BatchNorm statistics and
spectral-norm vectors of the EMA and of the live critic are never changed by
that forward, and no `.data` is swapped.

## Your own loop, and several critics

Do not instantiate K3P classes yourself. The recipe builds the current best
formulation behind a formulation-agnostic interface, one call per optimizer;
the same objects the trainer uses:

```python
from particlegan import get_recipe, learning_rate_scales

recipe = get_recipe(total_steps=steps)
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
critic_reg = recipe.make_critic_regularizer(D, opt_d)   # EMA critic, penalty, spike guard
gen_reg = recipe.make_generator_regularizer(opt_g, latent_table=prior.z)  # A2 damping
...
loss_d = adv + critic_reg.penalty(D, real, fake, step)[0]
opt_d.zero_grad(); loss_d.backward()
critic_reg.step()                      # guard, opt_d.step(), anchor EMA + LR record
...
opt_g.zero_grad(); loss_g.backward()
gen_reg.step()                         # opt_g.step() with A2 latent damping
torch.save({"critic_reg": critic_reg.state_dict(), "gen_reg": gen_reg.state_dict(), ...}, path)
```

If you need to own `optimizer.step()` (e.g. under a `GradScaler`), call
`before_step()` and `after_step()` around it instead of `step()`.
`critic_reg.diagnostics()` returns host scalars such as the blend weight.

With one module that has several critic roles, keep one critic regularizer for
its optimizer and give each role its own view of the EMA critic:

```python
for role in D.roles():
    critic = D.critic_for(role)
    pen, _ = critic_reg.penalty(lambda x: critic(x, ctx), xr, xf, step,
                                ema_critic=critic_reg.ema_critic(lambda m, x: m.critic_for(role)(x, ctx)))
```

Critics with separate optimizers get separate regularizers (one
`recipe.make_critic_regularizer(D_k, opt_k)` call each); their blend weights
and EMAs are independent. Nothing registers optimizer hooks or keeps
module-level state.

The concrete classes (`K3PCritic`, `K3PGeneratorRegularizer` and the
primitives `CriticAnchor`, `RobustCriticAnchor`, `CriticSpikeGuard`,
`LatentRowDamping`, `DirectParticleResponse`) stay importable from
`particlegan.k3p` for low-level tests and research, but they are not the public
API. Direct sample-particle groups use
`recipe.make_generator_regularizer(opt, direct_particles=[...])`.

## Historical recipes

GAN v3 ([guide](gan-v3.md)) was the previous default. Benchmarks that replay
its archived receipts resolve their recipes through
`benchmarks.gan_v3.GAN_V3_FIELDS` / `legacy_recipe`, which set every K3P-era
field to its pre-K3P value.
