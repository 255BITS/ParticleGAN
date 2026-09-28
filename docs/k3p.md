# The default ParticleGAN formulation: DV12 + KA2

`get_recipe()` and `GANTrainer` train with one formulation: a
relativistic-paired logistic GAN with a learned particle prior, the **KA2**
critic penalty, and the **DV12** learning-rate controller, which chooses every
learning rate from training signals. There is no LR schedule and no horizon:
`total_steps` defaults to `None`, so training can run indefinitely.
It is the winner of the LR-free search in PR #155
([report](../reports/toy100/lrfree-search/README.md) on that branch,
configuration `dv12-ams-rc3`). The file keeps its old name (`k3p.md`); K3P, the
0.8 formulation, is replayed only through `benchmarks.legacy`.

Around the ordinary G/D/prior updates the recipe's optimizers add the DV12
rates, AMSGrad, a critic spike guard, sparse latent-row damping (A2) and the
KA2 EMA critic; the prior adds support jitter to its draws and the generator
trains with constant output noise. All of these are recipe components; none
is switched on or off by detecting the task.

## The penalty (KA2)

With input dimension `d`, coefficient `c` (`reg_coeff`, 3) and cap `κ`
(`reg_kappa`, 1):

```text
A = mean(||∇D(real)||² / d) + mean(relu(||∇D(fake)|| / √d − κ)²)
B = mean(relu(||∇D(real)|| − κ)²) + mean(relu(||∇D(fake)|| − κ)²) + W·P
P = mean(||∇D(real) − ∇D̄(real)||² / d)        D̄ = parameter EMA of D
penalty = c/2 · A                     for the first 799 applied calls
penalty = c/2 · (.5·A + .5·B)         from call 800 on
```

- `W` (0 or 1) gates the anchor on the critic's Adam **moment surprise**: the
  median over tensors of `rms(grad) / sqrt(v̂)`, taken after each critic step.
  Its short-window median over a baseline frozen from the first 24 blended
  calls drops `W` to 0 above 3 and restores it below 1.75. While the data is
  not moving (DV12 `data_drive` < .1) `W` stays 1.
- D̄ tracks D with decay `1 − α·(1 − reg_anchor_min_decay)`: `α` rises with the
  surprise ratio and is scaled by DV12's game trust, so a calm critic keeps a
  slow anchor. A long run of high surprise reseeds D̄ from D.
- `reg_every = k > 1` applies the same penalty every `k`-th step with
  coefficient `k·c`.

Phase A is R1 at the reals. It is kept because the recorded runs use it; the
one-sided caps take over half of the weight after call 800.

## The learning rates (DV12)

Every update each role runs at a fraction of its peak rate (the group's `lr`):

```text
G     = lr         · (.01 + .99·m) · gt
prior = 2·lr       · (.05 + .95·m) · gt
D     = lr         · (.01 + .99·m) · gt / (1 + pe²)
```

- `pe`, the payoff error: EMA (.02) of `max(0, g_loss − d_loss) / log 2`, how
  far the generator is losing. The recipe's loss (`make_loss(opt_d)`) reports
  both values.
- `data_drive`: the RMS z-score of the drift between fast (.1) and slow (.01)
  EMA means of fixed random Fourier features of the real batches, mapped to
  [0, 1] as `clip((z − 3) / 3, 0, 1)`. The critic penalty shows each real batch
  to the controller. Real statistics set only this scalar; they never enter a
  loss, a sample or a parameter.
- `m`, mobility, moves toward `max(data_drive, min(1, pe²))`, rising at .05 and
  falling at .005 per update.
- `gt`, game trust, shrinks all rates when the critic's moment surprise rises
  above its baseline without data movement.

The rates stay high while the game is unresolved or the data moves, and relax
on their own. After a target change `data_drive` sends them back to full rate.
AMSGrad (every optimizer, `betas` (0, .999)) keeps the step from creeping up
as Adam's second moment decays over a long hold.

The feature normalization comes from the first real batch and the surprise
baseline from the first blended calls: the controller's references are fixed
at birth.

## Latent support jitter and output noise

`recipe.make_prior()` builds the particle table with `support_jitter`:
`prior.sample()` returns `z + support_width · ε`, shortened per row to at most
half the distance to the nearest other particle, so a draw stays in its own
cell. `support_width` is an EMA (.01 per generator step) of the table's
per-dimension spread times `N^(−1/d)`; the generator optimizer advances it.
The generator adds constant output noise `output_noise_std` (.029) in
training. `GANTrainer.sample()` returns clean samples: jittered latents, no
output noise.

## Defaults

| Field | Value | Meaning |
| --- | --- | --- |
| `reg_coeff`, `reg_kappa` | 3, 1 | KA2 penalty above |
| `reg_anchor_min_decay` | .9 | fastest EMA-critic decay (at full surprise gain) |
| `lr`, `d_lr_mult`, `prior_lr_mult` | .00425, 1, 2 | peak rates for DV12 |
| `betas`, `amsgrad`, `ema_decay` | (0, .999), True, .995 | Adam; G/prior EMA |
| `d_guard_ratio`, `d_guard_min_steps` | 5, 200 | clip a critic tensor whose grad RMS exceeds 5× its Adam RMS (AMSGrad: its running max); 0 disables |
| `latent_damping_max_rate` | .5 | A2 on the particle table (0 disables) |
| `output_noise_std`, `output_noise_warmup` | .029, 0 | constant generator output noise |
| `input_noise_std` | 0 | no critic input noise |
| `lr_floor`, `network_lr_floor`, `network_lr_horizon_cap` | 1, None, None | optional caller schedule, off |
| `prior_reg` | 0 | no particle spread penalty |
| `batch_size`, `z_dim`, `num_particles` | 2048, 2, 20000 | task shape |
| `total_steps` | None | optional loop / `GANTrainer` budget; `None` = no horizon (train indefinitely). Nothing in the formulation reads it; only the opt-in schedules (LR floor < 1, input noise, output-noise warmup) need an integer, and raise without one |

`learning_rate_scales(step, recipe)` and `scale_learning_rates` remain as an
optional caller schedule: they set each group's peak `lr`, and the optimizers
apply the DV12 fractions on top. At the default floors of 1 they return 1.
`NetworkLRTransition` works the same way.

## GANTrainer

`GANTrainer(recipe, G, D)` wires everything: it allocates the EMA critic
(`trainer.ema_D`, a frozen deep copy), builds the recipe's optimizers, loss
and penalty, draws latent jitter and output noise from a trainer stream, and
saves and restores all of them. Checkpoints use schema 3; the KA2 record and
the DV12 controller live in the critic optimizer's state (`"regularizer"`
entry) and the jitter width in the prior's buffers. Checkpoints written by
K3P (0.8) are rejected with a clear error; replay those through
`benchmarks.legacy.LegacyRecipe` or the release that wrote them.

The EMA critic is robust: floating-point buffers are averaged, integer
buffers copied, and every anchor forward runs in the live critic's train/eval
mode on private buffer copies. BatchNorm statistics and spectral-norm vectors
of the EMA and of the live critic are never changed by that forward.

## Your own loop, and several critics

Do not instantiate the classes yourself. The recipe builds the formulation
into ordinary-looking PyTorch objects, the same ones the trainer uses. Its
step-time work, learning rates included, runs inside the optimizers' `step()`,
and all its state is in their `state_dict()`:

```python
import copy
from particlegan import get_recipe, init

recipe = get_recipe()                         # no horizon: the loop decides how long to train
prior = init.deterministic_orthogonal_(recipe.make_prior())
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
loss = recipe.make_loss(opt_d)                # reports the payoff to the optimizers
penalty = recipe.make_critic_penalty(opt_d)   # shows the real batch to the optimizers
...
z, ids = prior.sample(batch)                  # jittered particle draws
x = G(z)
fake = x + recipe.output_noise_std * torch.randn_like(x)   # training-only output noise
d_loss = loss.d_loss(D(real), D(fake.detach())) + penalty(D, real, fake.detach())
opt_d.zero_grad(); d_loss.backward(); opt_d.step()   # DV12 rate, guard, Adam, KA2 EMA
g_loss = loss.g_loss(D(fake), D(real))
opt_g.zero_grad(); g_loss.backward(); opt_g.step()   # DV12 rates, Adam with A2, jitter width
torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}, path)
```

Call the penalty every critic step and use the bound loss for both losses:
`opt_d.step()` and `opt_g.step()` raise a clear error otherwise. With several
calls per step (roles, views), the controller observes the first real batch
and the last `d_loss`/`g_loss`. `group["lr"]` stays the peak you set; the
applied rates of the last step are in `opt.applied_lrs`, and
`penalty.diagnostics()` returns host scalars (phase weight, anchor gate,
guard clips, and the controller's mobility, trust, payoff error and scales).
`make_critic_penalty(opt_d, collect_stats=True)` fills `penalty.last_stats`.

Extra arguments are conditioning, forwarded to the critic and its EMA; a
tuple/list output uses its first element (`output=` at construction selects
another layout):

```python
d_loss = d_loss + penalty(D, x_prev, fake.detach(), labels, xt=xt, t=t)
```

With one module that has several critic roles, pass the role's submodule; the
penalty evaluates the same-named EMA submodule. A module wrapping the critic
(e.g. `InputNoise(D, std, generator)`) works the same way.

A critic with its own optimizer gets its own controller, record and EMA. A
generator optimizer follows one critic's controller:
`recipe.make_generator_optimizer(params, controller=opt_d.controller,
prior=prior)`. A critic optimizer with no generator optimizer attached sees no
payoff (`pe` stays 0). Nothing registers optimizer hooks or keeps
module-level state:

```python
init.deterministic_orthogonal_(D2, seed=3)   # optional: deterministic start for a fresh critic
opt_d2 = recipe.make_critic_optimizer(D2, ema_critic=copy.deepcopy(D2))
penalty2 = recipe.make_critic_penalty(opt_d2)
d2_loss = adv2 + penalty2(D2, real, fake)
opt_d2.zero_grad(); d2_loss.backward(); opt_d2.step()
```

The concrete classes (`K3PCriticAdam`, `K3PGeneratorAdam`, `CriticPenalty` and
the primitives `CriticAnchor`, `RobustCriticAnchor`, `CriticSpikeGuard`,
`LatentRowDamping` in `particlegan.k3p`, `DV12Controller` in
`particlegan.dv12`, the kernel in `particlegan.grad_regularizers`) stay
importable for low-level tests and research, but they are not the public API.

## Known limits

- `reg_coeff` sits in a narrow window: in PR #155 2.0, 2.5 and 3.5 each broke
  another gate. 3.0 is a working point, not a demonstrated optimum.
- The controller's references are fixed at birth (first real batch, first
  blended calls). On stationary data mobility's fixed .005 decay behaves
  somewhat like a clock.
- A one-field alternative, `output_noise_std=.018` (`t2-dv12q-ons018` in
  #155), passed all quick gates at the original budgets but holds `mode_hold`
  only thinly.
