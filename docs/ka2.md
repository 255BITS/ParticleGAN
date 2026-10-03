# KA2: the default ParticleGAN formulation

`get_recipe()` and `GANTrainer` train with **KA2**. It is a relativistic-paired
logistic GAN with a learned particle prior, and the same recipe as the earlier
[K3P](k3p.md) formulation except for the critic: the critic penalty and its EMA
critic are driven by the critic's own Adam statistics instead of by its
learning rate. Around the ordinary G/D/prior updates the recipe adds that
penalty, an EMA-critic anchor, a critic spike guard, sparse latent-row damping
(A2), a split learning-rate schedule and annealed input/output noise. All of
these are recipe hyperparameters; none is switched on or off by detecting the
task.

## The penalty

With input dimension `d`, coefficient `c` and cap `κ` (both 1 by default):

```text
A = mean(||∇D(real)||² / d) + mean(relu(||∇D(fake)|| / √d − κ)²)
B = mean(relu(||∇D(real)|| − κ)²) + mean(relu(||∇D(fake)|| − κ)²)
P = mean(||∇D(real) − ∇D̄(real)||² / d)        D̄ = the EMA critic

calls 1..799:  penalty = c/2 · A
calls 800..:   penalty = c/2 · (.5·A + .5·(B + W·P))
```

The first 799 applied penalty calls are the RMS R1 + fake-cap form `A`. From
call 800 the penalty is a fixed half-and-half blend of `A` and the one-sided
caps `B` plus the anchor term `P`. `P` is zero for a stationary critic, so it
damps oscillation without flattening the critic at the data. The EMA critic
starts at the first blended call. Unlike K3P, the blend does not read the
learning rate: a constant LR blends the same way.

`critic_r1_real=False` drops the real-data term `mean(||∇D(real)||² / d)` from
`A` (caps and anchor stay). On the native 100-Gaussian gates the R1 term is
needed (see [E22](e22.md#ablation)).

### The anchor gate and EMA rate

After every critic Adam step the optimizer measures its *moment surprise*: for
each critic tensor, the RMS of the gradient over the RMS of Adam's
bias-corrected second moment, and the median over tensors. Each blended call
consumes the latest value:

- The reference is the median of the first 24 values; from the 25th value on,
  `ratio` = median of the last 24 / reference.
- **Gate `W`** (0 or 1): switches off when `ratio > 3` (the critic is moving
  much faster than it used to) and back on when `ratio < 1.75`.
- **EMA rate**: `D̄` updates with decay `1 − α·(1 − reg_anchor_min_decay)`,
  between 1 (frozen) and .9 (fastest, by default). `α` follows
  `clip((ratio − 1) / 2, 0, 1)`, rising at rate 1/60 per call and falling at
  rate .5. At `α = 0` the EMA holds still.
- **Reseed**: after 60 consecutive blended calls with the gate off and
  `ratio > 3`, `D̄` restarts from the live critic.

With a continuous LR controller (`continuous_policy`, e.g. `"dv12"`), `α` is
also scaled by the controller's game trust, and `W` stays 1 while its
data-drift signal is below .1.

`reg_every = k > 1` applies the same penalty every `k`-th step with
coefficient `k·c`. Skipped steps do not advance the call count.

## Defaults

| Field | Value | Meaning |
| --- | --- | --- |
| `reg_coeff`, `reg_kappa` | 1, 1 | penalty above |
| `reg_anchor_min_decay`, `reg_anchor_weight` | .9, 1 | fastest EMA-critic decay; `P`'s weight (0 removes the anchor and the need for an EMA critic) |
| `critic_r1_real` | True | keep the R1 term in `A` |
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

The remaining recipe fields (`continuous_policy`, `lr_control`,
`output_noise_mode`, `particle_birth_death`, `row_evidence_*`,
`table_release_rule`, `birth_death_*`, `serve_average`, `reopen_signal`,
`amsgrad`) default to off / the behaviour above. The [E22 configuration](e22.md)
switches them on.

`learning_rate_scales(step, recipe)` returns the `(network, prior)` LR
multipliers. `network_lr_horizon_cap=None` uses the full budget and
`network_lr_floor=None` reuses `lr_floor`.

For a caller-owned loop, the caller can start G/D decay when its own validation
metric plateaus. `NetworkLRTransition` keeps G/D at full LR until marked, then
cosine-decays them over the chosen duration to `network_lr_floor`. The particle
prior retains its normal full-budget schedule. The caller owns the plateau rule
and must save the transition state with its optimizer checkpoint:

```python
from particlegan import NetworkLRTransition, scale_learning_rates

transition = NetworkLRTransition(decay_steps=40_000)
# After validation at 120,000 completed updates meets a declared plateau rule:
transition.mark_plateau(120_000)
# Before the next optimizer update, using the number of completed updates:
network_scale, prior_scale = scale_learning_rates(
    120_000, recipe, (opt_g, opt_d), base_lrs, prior,
    network_transition=transition)
checkpoint["network_transition"] = transition.state_dict()
# On resume, recreate the same transition duration and restore its state:
transition.load_state_dict(checkpoint["network_transition"])
```

Passing no transition retains the recipe's fixed network horizon. The
caller-marked step remains fixed after marking; a different step raises an
error. This API does not inspect validation data or choose checkpoints.

## GANTrainer

`GANTrainer(recipe, G, D)` wires everything: it allocates the EMA critic
(`trainer.ema_D`, a frozen deep copy), builds the recipe's optimizers (which
allocate the A2 history buffer) and a noise stream, and saves and restores all
of them. Networks train from the weights they arrive with; initialize fresh
ones first (`particlegan.init.deterministic_orthogonal_`).

Checkpoints use schema 4: the KA2 controller state (call count, surprise
history, gate, EMA rate and counters) and the EMA critic live in the critic
optimizer's state (`"regularizer"` entry). Schema 1–3 checkpoints come from
older formulations and raise `ValueError`; resume them with the release that
wrote them (0.8.0 for K3P).

The trainer's EMA critic is robust: floating-point buffers are averaged,
integer buffers copied, and every anchor forward runs in the live critic's
train/eval mode on private buffer copies. BatchNorm statistics and
spectral-norm vectors of the EMA and of the live critic are never changed by
that forward, and no `.data` is swapped.

## Your own loop, and several critics

Do not instantiate KA2 classes yourself. The recipe builds the formulation into
ordinary-looking PyTorch objects, the same ones the trainer uses. Its step-time
work runs inside the optimizers' `step()`, and all its state is in their
`state_dict()`:

```python
import copy
from particlegan import get_recipe, init, learning_rate_scales

recipe = get_recipe(total_steps=steps)
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty = recipe.make_critic_penalty(opt_d)   # reads the EMA critic and controller from opt_d
...
loss_d = adv + penalty(D, real, fake)
opt_d.zero_grad(); loss_d.backward(); opt_d.step()   # guard, Adam, surprise, EMA update
...
opt_g.zero_grad(); loss_g.backward(); opt_g.step()   # Adam with A2 latent damping
torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}, path)
```

`penalty.diagnostics()` returns host scalars (anchor gate, EMA rate, surprise
ratio, EMA update/skip/reseed counts, guard clips);
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

Critics with separate optimizers get separate pairs; their controllers and
EMAs are independent. Nothing registers optimizer hooks or keeps module-level
state:

```python
init.deterministic_orthogonal_(D2, seed=3)   # optional: deterministic start for a fresh critic
opt_d2 = recipe.make_critic_optimizer(D2, ema_critic=copy.deepcopy(D2))
penalty2 = recipe.make_critic_penalty(opt_d2)
d2_loss = adv2 + penalty2(D2, real, fake)
opt_d2.zero_grad(); d2_loss.backward(); opt_d2.step()
```

The concrete classes (`KA2CriticAdam`, `KA2GradientPenalty`, `CriticPenalty`
in `particlegan.ka2`; `K3PGeneratorAdam` and the primitives `CriticAnchor`,
`RobustCriticAnchor`, `CriticSpikeGuard`, `LatentRowDamping`,
`DirectParticleResponse` in `particlegan.k3p`) stay importable for low-level
tests and research, but they are not the public API. Direct sample-particle
groups use `recipe.make_generator_optimizer(params, direct_particles=[...])`.
