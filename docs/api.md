# ParticleGAN API reference

ParticleGAN provides independent PyTorch priors, losses, and diffusion helpers.
You supply models and data; use an optional GAN trainer or compose your own loop. Install the package
with `python -m pip install particlegan`; the core dependency is PyTorch.

## A minimal training loop

This complete example learns a synthetic 2D distribution. Replace `real_batch`
with your pipeline and the MLPs with your networks. The helper applies one
shared recipe to optimizers, losses, regularization, decay and EMA. The recipe
contains configuration and component factories; the separately constructed
trainer owns the training lifecycle.

```python
import torch
from torch import nn
from particlegan import BatchDistanceDiscriminator, GANTrainer, get_recipe

torch.manual_seed(0)
device = torch.device("cpu")
recipe = get_recipe(total_steps=1000)
G = nn.Sequential(nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(.2),
                  nn.Linear(64, 64), nn.LeakyReLU(.2), nn.Linear(64, 2)).to(device)
D = BatchDistanceDiscriminator().to(device)
trainer = GANTrainer(recipe, G, D, seed=0)

def real_batch():
    return .2 * torch.randn(recipe.batch_size, 2, device=device) + 1

for step in range(recipe.total_steps):
    stats = trainer.step(real_batch(), generator_real=real_batch)
    if (step + 1) % 100 == 0:
        print(step + 1, stats["loss_d"].item(), stats["loss_g"].item(), flush=True)

samples = trainer.sample(256)            # Live weights by default.
ema_samples = trainer.sample(256, ema=True)
torch.save(trainer.state_dict(), "trainer.pt")
```

[The runnable example](../examples/quickstart_gan.py) adds CLI options, flushed
JSON logs and a data-RNG checkpoint. To verify continuation:

```bash
python -u examples/quickstart_gan.py --steps 1000 --stop-after 500 --output run.pt
python -u examples/quickstart_gan.py --steps 1000 --resume run.pt --output run.pt
```

The explicit [component-based loop](../examples/pytorch_loop.py) remains available
for applications that manage their own updates.

## GANTrainer

`GANTrainer(recipe, G, D, *, prior=None, seed=0, latent_generator=None,
penalty_generator=None, optimizer_options=None, penalty_options=None)` is an
explicitly imported helper, separate from `Recipe`. Move networks to the same
device and floating dtype first. When omitted, the helper constructs the prior
from the recipe; supply `prior=` to preserve an existing initialization.
`seed` controls owned sampling streams, while callers seed network/prior
initialization with `torch.manual_seed`.

The helper supports scalar, unconditional GANs with `ParticlePrior`. MoG,
encoders, conditional GANs and DDGAN use the component API. A step performs one
D update, then one G/prior update with fresh latent samples. During the G phase,
D is evaluated with frozen parameters; each original gradient flag is restored.
Small particle tables (at most 1,024 rows) are regularized in full; larger tables
use unique sampled rows. There is no particle L2 term.

- `step(real, generator_real=None, collect_stats=False)` returns detached scalar
  tensors `loss_d`, `loss_g`, `loss_gan`, `prior_regularization`, `penalty`, and
  integer `step`. The reported prior term is unweighted; `loss_g` includes its
  recipe weight. `generator_real` can supply a fresh tensor or zero-argument
  callback for RP/RA; otherwise the real batch is reused. RP requires equal
  batch sizes. `collect_stats=True` also returns penalty diagnostics.
- `sample(n, ema=False, generator=None)` defaults to live weights. Its separate
  RNG and temporary evaluation mode preserve training randomness and module
  modes. EMA averages G/prior parameters and copies their buffers, including
  integer counters. EMA never determines a live leaderboard pass.
- `state_dict()` includes G, D, prior, EMA, optimizers, initial learning rates,
  update count and RNG states. `load_state_dict(state)` restores them, including
  global PyTorch RNG. Recreate the same recipe, architecture, options, dtype
  and device, with the same parameter freezing, before loading. Save your data-loader position or separate data RNG
  alongside it. Loading on CPU first works for a compatible CUDA trainer:
  `trainer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))`.

The recipe's `total_steps` is the full schedule budget. Resume with the same
budget; further steps after it is exhausted raise an error. A failed user
callback can occur after D has updated, so restore a checkpoint before retrying
that interrupted update. AMP, distributed training and custom update ratios
require a caller-owned loop.

`get_recipe()` constructs **K3P** ([details](k3p.md)): Rp logistic, the K3P
critic penalty (coefficient 1, κ 1, EMA-critic anchor .999), critic spike guard
(ratio 5 after 200 steps), A2 latent-row damping, Adam (0,.999), G/D LR .00425
and particle LR .0085. G/D rates hold for 60% of a 1,600-update horizon, then
cosine to 1% (`network_lr_horizon_cap`, `network_lr_floor`); particle rates hold
for 60% of the budget, then cosine toward 5%. The critic sees annealed input
noise and the generator output carries warmed-up noise (also in `sample`). There
is no particle spread or L2 term. Live sampling is the default; EMA is explicit.

`GANTrainer` builds everything through the recipe: `trainer.opt_g, trainer.opt_d
= recipe.make_optimizers(G, D, prior, ema_critic=...)` (the trainer allocates
`trainer.ema_D`, a frozen deep copy) and `trainer.penalty =
recipe.make_critic_penalty(trainer.opt_d)`. Checkpoints use schema 3 (the K3P
state is inside the optimizer states, plus a noise stream); schema-2
checkpoints are upgraded on load and schema-1 (GAN v3) checkpoints raise
`ValueError`. Caller-owned loops use the same objects with an ordinary loop:
`penalty(D, real, fake)` in the critic loss, then `opt_d.step()` and
`opt_g.step()` as usual. See [regularization factories](#regularization-factories). `learning_rate_scales(step, recipe)` returns the
`(network, prior)` LR multipliers.

| Version | Live behavioral toys passed | Meaning |
| --- | ---: | --- |
| v1 (archived) | 5/19 | Original preset |
| v2 (archived) | 8/19 | Previous public preset |
| v3 (current) | **19/19** | Shared recipe with declared D choices |

The v3 reference D profile scores 15/19. Its 19/19 result keeps optimizer/loss
settings identical across tests and permits task-specific discriminator
architectures. Each host retains its frozen resources and update budget.
The generic API uses 20,000 particles and 7,000 updates; choose resources for
your application. [Illustrated guide, math and limits](gan-v3.md).

Historical rows preserve measured settings in benchmark receipts. They are not
public recipe selectors. Restore complete saved settings through
`Recipe(**saved_recipe)` and resume with the original networks and recipe.

### Optional vector discriminators

`BatchDistanceDiscriminator(in_dim=2, hidden_dim=96, n_hidden=3,
scales=(.1,.25,.5,1.), beta=6., eps=1e-5)` accepts nonempty flat
`[batch, in_dim]` inputs and returns one score per sample. Its 2D defaults exactly
reproduce the v3 unequal-mass witness with **19,013 parameters**: per-example
hidden feature centering, Softplus β6, and differentiable kernel-weighted
neighbor distances appended to the final head. Self-pairs are excluded.

Scores depend on other samples in the current input batch, including during
G updates and gradient-cap differentiation. Real/fake calls compute separate
features; there are no running statistics. Cost is quadratic in batch size,
and scales are in input-coordinate units. The measured witness uses 2D inputs
and batch size 128. You supply D explicitly; the class is not substituted for
every network in the 19/19 architecture profile.

`LinearSkipDiscriminator(in_dim=2, hidden_dim=96, n_hidden=2, fourier=2, beta=5.)`
remains the historical v2 witness: a smooth Fourier MLP plus a zero-initialized
raw linear branch, 10,467 parameters. It supports native cap double backward.
Both classes can be passed to `GANTrainer` or used in a custom loop.

## A minimal DDGAN + UCD loop

This standalone example learns two conditional 2D distributions. G predicts
clean data; DDGAN constructs a reverse transition; UCD selects the requested
class score without feeding the class label into D's network. Replace the
synthetic `real` and `labels` with batches from your pipeline.

```python
import copy
import torch
from torch import nn
from torch.nn import functional as F
from particlegan import DDGAN, UCD, get_recipe, scale_learning_rates, ucd_loss

device = torch.device("cpu")
recipe = get_recipe(model="ddgan", conditioning="ucd", num_classes=2)  # Add total_steps=5 for a smoke check.
process = DDGAN(recipe.alpha_bar).to(device)

class Generator(nn.Module):
    def __init__(self, z_dim, classes, steps):
        super().__init__()
        self.classes, self.steps = classes, steps
        self.net = nn.Sequential(nn.Linear(z_dim + classes + 3, 64),
                                 nn.LeakyReLU(0.2), nn.Linear(64, 2))

    def forward(self, z, labels, *, xt, t):
        c = F.one_hot(labels, self.classes).to(z)
        time = t[:, None].to(z) / self.steps
        return self.net(torch.cat((z, c, xt, time), dim=1))

class LogitNetwork(nn.Module):
    def __init__(self, classes, steps):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(nn.Linear(5, 64), nn.LeakyReLU(0.2),
                                 nn.Linear(64, classes))

    def forward(self, x, *, xt, t):
        return self.net(torch.cat((x, xt, t[:, None].to(x) / self.steps), dim=1))

G = Generator(recipe.z_dim, recipe.num_classes, process.steps).to(device)
D = UCD(LogitNetwork(recipe.num_classes, process.steps), recipe.num_classes).to(device)
prior = recipe.make_prior().to(device)
gan = recipe.make_loss()
spread = recipe.make_prior_regularizer()
# Adam optimizers whose step() runs the recipe's regularization (K3P today);
# the EMA critic is ours to allocate.
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
penalty = recipe.make_critic_penalty(opt_d)
ema_g = copy.deepcopy(G).eval().requires_grad_(False)
ema_prior = copy.deepcopy(prior).eval().requires_grad_(False)

for step in range(recipe.total_steps):
    # G/D follow the network schedule (K3P's blend floor), the prior its own.
    scale_learning_rates(step, recipe, (opt_g, opt_d), base_lrs, prior)

    labels = torch.randint(recipe.num_classes, (recipe.batch_size,), device=device)
    real = 0.2 * torch.randn(len(labels), 2, device=device) + (2 * labels[:, None] - 1)
    t = torch.randint(1, process.steps + 1, (len(real),), device=device)
    x_prev, xt = process.forward_pair(real, t)
    z, indices = prior.sample(len(real))
    clean = G(z, labels, xt=xt, t=t)
    fake = process.reverse(clean, xt, t, torch.randn_like(xt))

    opt_d.zero_grad(set_to_none=True)
    real_score, real_logits = D(x_prev, labels, xt=xt, t=t)
    fake_score, fake_logits = D(fake.detach(), labels, xt=xt, t=t)
    d_loss = gan.d_loss(real_score, fake_score)
    d_loss += ucd_loss(real_logits, fake_logits, labels, weight=recipe.ucd_weight)
    d_loss = d_loss + penalty(D, x_prev, fake.detach(), labels, xt=xt, t=t)
    d_loss.backward()
    opt_d.step()

    D.requires_grad_(False)
    opt_g.zero_grad(set_to_none=True)
    fake_score = D(fake, labels, xt=xt, t=t)[0]
    real_score = D(x_prev, labels, xt=xt, t=t)[0].detach()
    g_loss = gan.g_loss(fake_score, real_score) + spread(prior.z[indices.unique()])
    g_loss.backward()
    opt_g.step()
    D.requires_grad_(True)

    with torch.no_grad():
        for average, current in ((ema_g, G), (ema_prior, prior)):
            for target, source in zip(average.parameters(), current.parameters()):
                target.lerp_(source, 1 - recipe.ema_decay)
    if step % 100 == 0 or step + 1 == recipe.total_steps:
        print(f"step={step + 1} d={d_loss.item():.4f} g={g_loss.item():.4f}", flush=True)

# Conditional inference: fresh latent particles and Gaussian noise at each step.
with torch.inference_mode():
    labels = torch.arange(64, device=device) % recipe.num_classes
    samples = torch.randn(len(labels), 2, device=device)
    for step in range(process.steps, 0, -1):
        t = torch.full_like(labels, step)
        z, _ = ema_prior.sample(len(labels))
        clean = ema_g(z, labels, xt=samples, t=t)
        samples = process.reverse(clean, samples, t, torch.randn_like(samples))
```

The D update combines adversarial loss, class cross-entropy, and the candidate
gradient penalty. The G/prior update combines adversarial loss and particle
regularization. One random transition per example is used during training;
inference walks through every reverse step. Class-only UCD is the default;
[joint time/class heads](#ucd) are an independent option.

## Reference index

| Component | Purpose |
| --- | --- |
| [Priors](#priors) | Learnable particles, fixed-sigma Gaussian mixtures, or fresh Gaussian samples |
| [Losses](#losses-and-regularizers) | Adversarial objectives and additive regularizers |
| [DDGAN](#ddgan) | Forward corruption and reverse transitions |
| [UCD](#ucd) | Class-score selection and class supervision |
| [Recipes](#recipes-and-defaults) | Inspectable defaults and optional factories |
| [GANTrainer](#gantrainer) | Optional unconditional GAN updates, sampling and checkpoints |
| [Locked shared](#locked-shared) | Demo RpGAN + `b_cap` stamp (not `Recipe("gan")`) |
| [TOML](#toml-configuration) | Pass loaded dictionaries to constructors |
| [Other pipelines](#loss-augmentation-and-teacherstudent-pipelines) | Compose with existing objectives |
| [Inference](#inference-and-checkpoints) | Generate from saved G and prior states |

All names below are exported from `particlegan`. Modules use ordinary
`.to(device, dtype)`, `.parameters()`, and `.state_dict()` behavior. Move models
and priors before constructing optimizers. Stateless loss helpers need no device
setup. These primitives never step an optimizer. The optional `GANTrainer`
manages updates and restores RNG state when loading checkpoints.

## Priors

### `ParticlePrior`

```python
ParticlePrior(num_particles=20_000, z_dim=4, init_std=1.0,
              device=None, dtype=None, learnable=True, generator=None)
```

An `nn.Module` with table `prior.z` of shape `[num_particles, z_dim]`, initialized
from a zero-mean Gaussian with standard deviation `init_std`. By default the
table is a parameter. With `learnable=False`, it is a fixed buffer.

| Method / attribute | Result |
| --- | --- |
| `sample(batch_size, generator=None)` | `(z, indices)`: `[B, z_dim]` codes and `[B]` long indices, sampled uniformly with replacement |
| `sample(..., fixed_first_n=True, offset=0)` | Consecutive rows starting at `offset`; requires a block within the table |
| `sample_indices(batch_size, generator=None)` | Only the random indices, on the table's device |
| `prior(indices)` | Indexed codes with gradients to the selected rows |
| `num_particles`, `z_dim` | Dimensions of the table |

Sampling does not detach latent codes. Include the prior parameters in your
optimizer to learn them. Use `prior(indices)` through your distributed wrapper's
forward path when applicable. An explicit `torch.Generator` controls initialization
or sampling without consuming the global RNG; use a generator for the same device.

### `MoGParticlePrior`

```python
MoGParticlePrior(num_particles=400, z_dim=4, init_std=1.0,
                 device=None, dtype=None, learnable=True, generator=None,
                 *, sigma, standardize=True)
```

An equal-weight mixture: choose component `i` uniformly, then draw
`z = means()[i] + sigma * eps`, with standard-normal epsilon. The raw component
centers are the parameter `prior.z`. Sigma is a **required keyword argument**:
a finite, nonnegative real scalar stored as one shared isotropic buffer, fixed
during training. Construction draws the centers once and performs no calibration
or nearest-neighbor search. `learnable=False` freezes the table as a buffer.
Standardized reads require at least two components; raw reads allow one.
Coincident centers and `init_std=0` are valid with an explicit sigma.

With `standardize=True`, each read centers and divides the table by its
per-dimension sample standard deviation plus `1e-6`. This is differentiable;
all rows can receive gradients. With `standardize=False`, means are the raw
table, as in `ParticlePrior`. Learning and EMA updates do not recalibrate sigma.

| Method / attribute | Result |
| --- | --- |
| `sample(batch_size, generator=None, *, fixed_first_n=False, offset=0, eps=None)` | Noisy codes and selected component indices |
| `prior(indices, generator=None, *, eps=None)` | Noisy draws for supplied indices; use this forward path through DDP |
| `means()` | Differentiable read-space centers, with no noise |
| `z` | Raw learned table; the input to particle regularization |
| `sigma` | Fixed scalar buffer, moving with `.to(...)` |
| `d0`, `sigma_rel` | Legacy calibration metadata; zero for explicitly supplied sigma |
| `set_sigma(sigma)` | Explicitly replace the fixed scale and update zero-noise RNG handling |

`fixed_first_n=True` fixes component indices, **not epsilon**. For a stable scatter,
save a fixed epsilon tensor too:

```python
from particlegan import MoGParticlePrior

prior = MoGParticlePrior(num_particles=65536, z_dim=128, sigma=0.212616428732872)
eps = torch.randn(64, prior.z_dim, device=prior.z.device, dtype=prior.z.dtype)
z, indices = prior.sample(64, fixed_first_n=True, eps=eps)
# Reuse eps at every snapshot; use prior.means()[indices] only for a centers-only audit.
```

Explicit epsilon must match the sampled codes' shape, device and dtype. An explicit
generator controls both component selection and Gaussian draws without touching
global RNG. `sigma=0, standardize=False` preserves `ParticlePrior` outputs and
RNG consumption; zero sigma never draws noise. `eval()` keeps Gaussian noise on.

For DDP, sample indices from the unwrapped prior, then call the wrapped module:
`z = wrapped_prior(indices, generator=rng)`. Use `prior.z` for VICReg rather than
`prior(indices)` or sampled codes. The one-shot MoG benchmark regularizes the
full raw table at N ≤ 1024, otherwise `prior.z[indices.unique()]`. The denoising
trainer and loops above regularize sampled unique raw rows for either prior.

For EMA, deepcopy the prior and average its learned `z`; the fixed buffers retain
their fixed values. Standardization is computed from the EMA table itself.
State dicts include `z`, `sigma`, `d0`, and `_extra_state` containing `sigma_rel`
and `standardize`. Reconstruct with matching dimensions and an explicit placeholder `sigma=0`, then
`load_state_dict`:
read settings and noise are restored even if constructor defaults differ.
Legacy experimental checkpoints containing only z/sigma/d0 are accepted; supply
their original `standardize` setting when constructing the prior.

Optional spacing calibration is a standalone helper:

```python
from particlegan import MoGParticlePrior, calibrate_mog_sigma

prior = MoGParticlePrior(num_particles=400, z_dim=4, sigma=0)
sigma, d0 = calibrate_mog_sigma(prior.means(), sigma_rel=1/40)
prior.set_sigma(sigma)
prior.d0.copy_(d0)        # Optional historical metric/checkpoint metadata.
prior.sigma_rel = 1/40
```

This calibrates the **already initialized centers without redrawing them**.
The helper accepts supplied read-space centers; it does not standardize, mutate
centers, or consume RNG. It returns detached scalar tensors `(sigma, d0)` on the
centers' device and dtype. The exact median averages the two middle nearest-neighbor
distances for even component counts. As before, `d0` is rounded to the centers'
dtype before multiplication by `sigma_rel`. Calibration requires at least two
finite centers and a positive median spacing, even when `sigma_rel=0`.

**Calibration can be very expensive**, especially for 65,536 centers in 128
dimensions. It copies centers to CPU float64 and uses SciPy's exact CPU tree
when installed (`pip install 'particlegan[mog]'`), or memory-bounded, quadratic
Torch distances otherwise. Trees can also be slow in high dimensions. Choose
an explicit sigma to avoid this work. Sampling needs neither SciPy nor NumPy.

Migration from 0.5: replace constructor `sigma_rel=...` with `sigma=...`; the
`calibrate()` method is replaced by `calibrate_mog_sigma`. Historical MoG recipes
retain spacing calibration explicitly; `recipe.make_prior(sigma=...)` skips it.
For HyperGAN, pass `sigma=fixed_sigma` directly and remove the subsequent buffer
overwrite. Checkpoint loading must restore its saved sigma, even when it differs
from the constructor's value. Do not call the helper when loading a checkpoint.

The default experiment is [configs/mog/default.toml](../configs/mog/default.toml),
run via `python -u experiments/train_100gaussians.py --config configs/mog/default.toml`.
It retains the benchmark's networks, Fourier discriminator and full metric suite.
Use `get_recipe(prior_kind="mog", sigma_rel=.025)`; a recipe alone does not
reproduce benchmark quality with arbitrary networks or data.

### `GaussianPrior`

```python
GaussianPrior(z_dim=4, init_std=1.0, device=None, dtype=None)
```

An `nn.Module` that draws fresh Gaussian codes. `sample(batch_size,
generator=None)` returns `(z, None)`. Calling the module as
`prior(batch_size, generator=None)` returns only `z`. It has no trainable
parameters or particle table, so omit particle regularization. Its empty buffer
tracks device and dtype.

## Losses and regularizers

### `GANLoss`

```python
GANLoss(loss_type="logistic", mode="rp",
        label_smoothing=0.0, label_flip_prob=0.0)
```

`d_loss(real_logits, fake_logits)` and `g_loss(fake_logits, real_logits=None)`
return scalar tensors to minimize. Both preserve their input gradient paths;
the caller decides which scores to detach. Inputs are critic scores, without
a sigmoid; use matching shapes such as `[B]` or `[B, 1]`.

| Option | Values |
| --- | --- |
| `loss_type` | `logistic`, `hinge`, `wasserstein`, `lsgan` |
| `mode` | `vanilla`, `rp` (paired relativistic), `ra` (average relativistic) |

The default D loss is `softplus(fake - real).mean()`; the G loss reverses that
difference. Both relativistic modes require real scores for `g_loss`. Smoothing
and label flipping apply to the vanilla logistic D objective. Recompute D's
scores after updating D; freeze its weights during the G step while retaining
the gradient path through the fake input.

### `GradientPenalty`

```python
GradientPenalty(arm="k3p", coeff=1.0, kappa=1.0, lazy_k=1, norm="l2",
                target_anneal="none", total_steps=0,
                method="autograd", fd_eps=0.05, lr_floor=0.01, anchor=None)
```

Call `penalty(D, real, fake, step=1, generator=None)` to get a scalar loss.
`D` may be a module or callable returning one scalar score per example. The
default arm is K3P, which is stateful: call `penalty.after_critic_step(opt_d)`
after every critic `opt_d.step()` and give it an EMA critic (`anchor=` or
`ema_critic=`). `recipe.make_critic_penalty(opt_d)` with the recipe's
optimizers does this wiring for you; see [K3P](k3p.md). Other arms are stateless and ignore `after_critic_step`.

| `arm` | Penalty |
| --- | --- |
| `k3p` | Zero-centered real R1 + fake cap, blended by the critic LR ratio into a real/fake cap plus an EMA-critic gradient proximity term |
| `b_cap` | One-sided gradient cap on reals and fakes |
| `a_r1r2` | Zero-centered squared L2 gradients on reals and fakes |
| `c_eikonal` | Two-sided penalty around norm 1 |
| `d_asym` | Two-sided penalty with weaker pressure below norm 1 |
| `e_interp` | Two-sided norm-1 penalty on real/fake interpolates |
| `g_interp_cap` | One-sided gradient cap on interpolates |
| `f_none` | Zero penalty |

The helper recomputes D on detached candidate tensors with input gradients
enabled. It builds gradients for the critic's parameters; it does not train the
generator through the supplied fake batch. Keep conditioning tensors fixed in
a closure when regularizing a conditional critic.

- `lazy_k > 1` applies the penalty when `step % lazy_k == 0` at `lazy_k` times
  its weight. Supply the current D-update index; the default `step=1` never advances.
- Norm choices are `l2`, `l1`, and `linf`; `a_r1r2` requires `l2`.
- `target_anneal` accepts `none`, `linear`, or `delayed`; annealing requires
  positive `total_steps`.
- `method="finite_difference"` is an optional approximation supporting only
  L2 `b_cap`; `fd_eps` is its input displacement. The default uses exact autograd.
- `penalty.penalty(D, real, fake, step=1, generator=None, collect_stats=True)`
  returns `(loss, stats)`. The callable form disables stats to avoid scalar
  synchronization. A skipped penalty returns a detached zero.

### Locked shared

```python
from particlegan.locked_shared import LOCKED_SHARED, locked_adv_defaults, make_gan_loss, make_b_cap
```

`LOCKED_SHARED` is the demo stamp: RpGAN logistic, `b_cap` coeff 1, κ 1, L2,
`lazy_k` 1, feature matching off, cover 1.5, a 12-particle cloud at
`particle_l2` 0.02 when particles are built, and the host critic.
`make_gan_loss()` and `make_b_cap()` build those two objects and refuse any
other stamp. Music cover 1.0, a 128-particle hub cloud, FM-on, stranger
pairing, and a thinned κ are not this stamp. The selected `get_recipe()`
uses coefficient 6, κ=1.25 and prior regularization .05; this frozen stamp retains
its original values.
The full field table is in [locked shared](locked-shared.md). No repository trainer uses it.

### `ParticleRegularizer`

```python
ParticleRegularizer(target_std=1.0, eps=1e-4, weight=1.0)
```

Call with a floating tensor `[N, z_dim]`. Returns a weighted scalar combining
a hinge below `target_std` for each dimension's standard deviation and a
penalty on off-diagonal covariance. It does not force a Gaussian distribution.
Fewer than two rows produce a differentiable zero. It accepts arbitrary latent
rows and does not require a `ParticlePrior` object.

For selected sampled rows, use `spread(prior.z[indices.unique()])`. Deduplication
belongs to the caller; repeated row values are otherwise counted repeatedly.

## DDGAN

```text
DDGAN(alpha_bar=(1.0, 0.9, 0.5, 0.05, 0.0001), *,
      device=None, dtype=None, validate_args=True)
```

An `nn.Module` containing Gaussian diffusion coefficients as buffers.
`alpha_bar` starts at 1 and strictly decreases while remaining positive.
`process.steps == len(alpha_bar) - 1`.

| Method | Contract |
| --- | --- |
| `forward_pair(x0, t, rng=None, *, generator=None)` | Coupled real samples `(x_prev, xt)` from forward corruption; pass either RNG argument |
| `reverse(x0, xt, t, eta)` | Reverse transition from predicted clean `x0`, noisy `xt`, and caller-supplied noise `eta` |

Data has shape `[B, ...]`; `t` is a long tensor `[B]` with values `1..steps`.
Inputs and schedule share a device. Reverse inputs have identical shape and
dtype. `reverse` computes `A[t] * x0 + B[t] * xt + sqrt(posterior_var[t]) * eta`;
the final transition at `t=1` has zero noise variance. It preserves gradients
through its inputs and does not add a reconstruction or diffusion MSE loss.

For DDGAN training, G predicts clean data, then D judges the **reverse
transition**. Hold `xt`, labels, and time fixed for the candidate gradient penalty.
For instance, given caller-defined `G`, `critic`, `prior`, and a labeled batch:

```python
from particlegan import DDGAN

process = DDGAN().to(real.device)
t = torch.randint(1, process.steps + 1, (len(real),), device=real.device)
x_prev, xt = process.forward_pair(real, t)
z, indices = prior.sample(len(real))
fake_prev = process.reverse(G(z, labels, xt=xt, t=t), xt, t, torch.randn_like(xt))
d_penalty = penalty(lambda x: critic(x, labels, xt=xt, t=t)[0],
                    x_prev, fake_prev.detach())
```

Learned latent particles, forward-corruption noise, terminal `x_T`, and reverse
noise `eta` are separate sources of randomness. The selected denoising recipe
uses Gaussian corruption, Gaussian terminal state, and fresh Gaussian `eta`.

## UCD

```text
UCD(network, num_classes, *, target="class", num_steps=None, validate_args=True)
```

An `nn.Module` wrapping your logit network. Calling
`critic(x, labels, xt=None, t=None)` returns `(selected_score, logits)` with
shapes `[B]` and `[B, heads]`. Labels are long tensors `[B]` in `0..num_classes-1`.
Class labels select an output head; they are not inputs to `network`.

| Use | Network call | Logits | Selected head |
| --- | --- | --- | --- |
| One-shot, `target="class"` | `network(x)` | `[B, C]` | `labels` |
| DDGAN, `target="class"` | `network(x, xt=xt, t=t)` | `[B, C]` | `labels` |
| DDGAN, `target="time_class"` | `network(x, xt=xt)` | `[B, T*C]` | `(t-1)*C + labels` |

Joint time/class heads require `num_steps=T`. `critic.ucd_labels(labels, t=None)`
returns the head indices. The same operation is available without a wrapper:

```text
ucd_labels(labels, timestep=None, *, num_classes, target="class",
           num_steps=None, validate_args=True)
ucd_scores(logits, labels, timestep=None, *, num_classes, target="class",
           num_steps=None, validate_args=True)
ucd_loss(real_logits, fake_logits, targets, weight=0.02)
```

`ucd_loss` is `weight * (CE(real_logits, targets) + CE(fake_logits, targets))`.
The selected recipe adds it to D's loss. The function does not detach either
logit tensor, so detach fake samples before computing D's logits.

`ucd_scores` selects the adversarial scores directly from your model's existing
`[B, heads]` logits and returns `[B]`, preserving gradients. Use it when your
pipeline already computes logits or needs to keep its existing model and
checkpoint structure. `UCD` uses this same function internally. Joint time/class
selection requires `num_steps=T`.

DDGAN and UCD numeric bounds checks can synchronize CUDA. Set
`validate_args=False` only for already validated times/labels; shape, dtype,
and applicable device checks still run.

## Particle autoencoders

Set `prior_kind="mog"`, `sigma_rel=.025` and `encoder_mode="ae"` or `"hard"`
to add reconstruction encodings to caller-owned networks and loops. Set
`model="ddgan"` when supplying a diffusion generator. These component choices
share the winning optimizer/loss defaults. Hard VAE selects one particle with
prior-matching Gaussian noise and constant joint KL; reconstruction adds no KL.

| API | Contract |
| --- | --- |
| `recipe.encode(query, prior, offset=None, draws=2, generator=None)` | `ParticleEncoding`; AE requires offset and returns one draw; VAE rejects offset |
| `particle_ae(query, offset, prior, temperature=.25, distance_reduction="sum", offset_bound=3)` | Deterministic bounded-offset encoding |
| `particle_vae(query, prior, temperature=.25, distance_reduction="sum", draws=2, hard=True, generator=None)` | Default constant-KL posterior; `hard=False` opts into categorical sampling |
| `encoding.reconstruction_loss(prediction, target)` | MSE only, with score-gradient correction for categorical mode |
| `encoding.negative_elbo(prediction, target, observation_sigma=.03)` | Explicit Gaussian negative ELBO in nats, including joint KL; rejects AE |
| `recipe.make_optimizers(G, D, prior, encoder=E)` | Adds E at G's LR; deduplicates shared parameters |

Codes are `[B,S,latent_dim]`, predictions `[B,S,...]`, targets `[B,...]`.
`encoding.kl` is per-input joint KL; `log_probs` exposes the true posterior
(or None for AE). Optional categorical training needs at least two independent
draws. See the [full guide](particle-autoencoders.md) for defaults, runnable
examples, gradient caveats, DDGAN integration and measured evidence.

## Recipes and defaults

```python
get_recipe("gan", **overrides) # Named components, current shared hyperparameters.
recipe.replace(**overrides)   # A new immutable Recipe.
recipe.to_dict()              # Complete resolved fields.
Recipe(**resolved_dict)       # Restore explicit fields from a saved run.
```

`get_recipe(name="gan", **overrides)` selects components without constructing a
training loop. Explicit keyword fields override the selected configuration.
Every family uses the current shared optimizer, loss, penalty and schedule
defaults; historical hyperparameter versions are not selectable. Unknown names
and fields are rejected. Restore a complete saved configuration with
`Recipe(**saved_fields)`; use `recipe.replace(name="my-run")` to label a run.

| Name | Components and dimensions |
| --- | --- |
| `gan` (default) | Scalar GAN, 20,000 particles, latent dimension 4, no sampling noise |
| `mog` | GAN, 400 MoG components, latent dimension 4, relative sigma .025 |
| `ddgan` | DDGAN, UCD with 4 classes, discrete particles |
| `ddgan_mog` | DDGAN, UCD with 4 classes, 400 MoG components, relative sigma .025 |
| `ae_gan` | GAN, AE encoding, 400 MoG components, latent dimension 2 |
| `vae_gan` | GAN, hard VAE encoding, 400 MoG components, latent dimension 2 |
| `ae_ddgan` | DDGAN, AE encoding, 1,024 MoG components, latent dimension 64, batch 64 |

The encoder families use relative sigma .025. `ae_ddgan` uses mean routing
distance and temperature .125; the others use sum distance and temperature .25.
These component choices retain the original API's model-family structure with
the new shared hyperparameters. The GAN development-suite evidence does not
establish convergence of those hyperparameters for every AE/VAE/DDGAN setup.

### Components that change a loop

The recipe publishes settings and small operations; the caller composes them.
UCD settings describe score selection and class-loss weight. AE/VAE settings
describe encoding and reconstruction. Neither creates a training strategy:

```python
ucd_recipe = get_recipe("gan", conditioning="ucd", num_classes=4)
ae_recipe = get_recipe("ae_gan")
vae_recipe = get_recipe("vae_gan")
```

| Component | Recipe supplies | Caller controls |
| --- | --- | --- |
| UCD | Class count, target, auxiliary loss weight | Labels, discriminator heads, score selection, adding `ucd_loss` to the D objective |
| AE/VAE | Prior, routing settings, `encode`, reconstruction weight, optional encoder optimizer group | Encoder forward pass, reconstruction/adversarial loss composition, backward and optimizer steps |
| DDGAN | Model selection and diffusion schedule | Corruption, timestep sampling, reverse transitions and update order |

See the [DDGAN/UCD loop](#a-minimal-ddgan--ucd-loop) and
[AE/VAE example](../examples/particle_autoencoder.py). `GANTrainer(recipe, G, D)`
is a separate, optional helper for unconditional particle GANs; it explicitly
rejects UCD, encoders and DDGAN. `Recipe` has no trainer factory or training step.

Discrete learned particles are the zero-noise limit of a mixture of Gaussians.
The current `ParticlePrior` reads raw centers; `MoGParticlePrior` standardizes
centers by default, even when sigma is zero. For matching center reads use
`sigma_rel=0, standardize=False`. The default GAN retains `ParticlePrior`;
unifying implementations or promoting a noisy MoG default requires separate
validation.

### Explicit composition

Choose components explicitly, while inheriting the common training defaults:

```python
recipe = get_recipe(model="ddgan", conditioning="ucd", num_classes=4,
                    prior_kind="mog", sigma_rel=.025, num_particles=400)
prior = recipe.make_prior().to(device)
process = DDGAN(recipe.alpha_bar).to(device)
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
```

| Shared field | Default |
| --- | --- |
| `model`, `conditioning`, `num_classes` | `gan`, `scalar`, `None` |
| `z_dim`, `num_particles` | `2`, `20_000` |
| `prior_kind`, `sigma_rel`, `standardize` | `particles`, `0`, `True` (standardize applies only to MoG) |
| `loss_type`, `gan_mode` | `logistic`, `rp` |
| `lr`, `d_lr_mult`, `prior_lr_mult` | `.00425`, `1`, `2` |
| `betas`, `prior_betas` | `(0, .999)`, `None` (inherit betas) |
| `reg_arm`, `reg_coeff`, `reg_kappa` | `k3p`, `1`, `1` |
| `reg_every`, `reg_method` | `1` (K3P every k-th step at k× coefficient), `autograd` |
| `prior_reg`, `ema_decay` | `0`, `.995` |
| `lr_anneal_start`, `lr_floor` | `.6`, `.05` (prior schedule) |
| `network_lr_horizon_cap`, `network_lr_floor` | `1600`, `.01` (G/D schedule and K3P blend floor; `None` = full budget / `lr_floor`) |
| `reg_anchor_decay` | `.999` |
| `d_guard_ratio`, `d_guard_min_steps` | `5`, `200` (ratio 0 disables) |
| `latent_damping_max_rate` | `.5` (0 disables) |
| `direct_particle_betas` | `(0, .9)` (`make_generator_optimizer(direct_particles=...)`) |
| `input_noise_std`, `input_noise_anneal_end` | `.5`, `.1` |
| `output_noise_std`, `output_noise_warmup` | `.029`, `.2` |
| `batch_size`, `total_steps` | `2048`, `7_000` |
| `ucd_target`, `ucd_weight` | `class`, `.02` |
| `alpha_bar` | `(1, .9, .5, .05, .0001)` |

Architectures, model/prior choices, data and resource budgets belong to the
caller. Historical v1/v2 comparison receipts live in benchmark data, outside
the installable package; they are not alternate production defaults.

| Optional factory | Result |
| --- | --- |
| `recipe.make_prior(**kwargs)` | `ParticlePrior` or `MoGParticlePrior` selected by `prior_kind` |
| `recipe.make_loss(**kwargs)` | `GANLoss` using recipe loss and mode |
| `recipe.make_optimizers(G, D, prior=None, *, encoder=None, ema_critic=None, **adam_kwargs)` | `(opt_g, opt_d)` Adam optimizers doing the recipe's step-time work (see below) |
| `recipe.make_critic_optimizer(D, *, ema_critic=None, **adam_kwargs)` | Adam for one (additional) critic (see below) |
| `recipe.make_generator_optimizer(params, *, latent_table=None, direct_particles=None, **adam_kwargs)` | Adam for generator-side params (see below) |
| `recipe.make_critic_penalty(opt_d, *, output=None, generator=None, collect_stats=False, **penalty_kwargs)` | The critic penalty paired with a critic optimizer (see below) |
| `recipe.make_gradient_penalty(**kwargs)` | Bare `GradientPenalty` using recipe penalty settings (stateless arms) |
| `recipe.make_prior_regularizer(**kwargs)` | `ParticleRegularizer` with `weight=recipe.prior_reg` already applied |

### Regularization factories

The recipe, not the caller, chooses the regularization formulation, and your
loop stays plain PyTorch. Today the factories return K3P implementations
(`particlegan.k3p.K3PGeneratorAdam`, `K3PCriticAdam`, `CriticPenalty`); a
future formulation can replace them without changing caller code.

```python
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty = recipe.make_critic_penalty(opt_d)
d_loss = adv_d + penalty(D, real, fake)                  # or penalty(D, x, fake, labels, t=t)
opt_d.zero_grad(); d_loss.backward(); opt_d.step()       # guard, Adam, EMA + LR record
opt_g.zero_grad(); g_loss.backward(); opt_g.step()       # Adam with A2 latent damping

opt_d2 = recipe.make_critic_optimizer(D2, ema_critic=copy.deepcopy(D2))  # a second critic
penalty2 = recipe.make_critic_penalty(opt_d2)

torch.save({"G": G.state_dict(), "D": D.state_dict(), "D2": D2.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "opt_d2": opt_d2.state_dict()}, path)
```

- The optimizers are `torch.optim.Adam` subclasses: `param_groups`, LR
  schedulers, closures and `state_dict()`/`load_state_dict()` work as usual.
  Their `state_dict()` adds a `"regularizer"` entry holding the EMA critic,
  LR record, counters, guard count and A2/direct-particle histories, so the
  usual checkpoint above resumes bit-exactly.
- `ema_critic` is caller-allocated (e.g. `copy.deepcopy(D)`); the K3P penalty
  requires it. The optimizer freezes it and only writes it.
- `penalty(D, real, fake, *condition, **condition_kwargs)` returns a scalar.
  Conditioning is forwarded to the critic and its EMA. `D` may be the
  optimizer's critic, a submodule of it (one role of a shared module; the
  same-named EMA submodule is used) or a module wrapping one of those (e.g.
  `InputNoise(D, std, generator)`). A tuple/list output uses its first element
  unless `output=` selects the logits. The step for `reg_every` is the
  optimizer's completed step count + 1.
- `penalty.last_stats` holds the last call's stats when `collect_stats=True`;
  `penalty.diagnostics()` returns host scalars such as
  `{"blend_weight": ..., "clipped_tensors": ...}`; `penalty.ema_critic` is the
  paired EMA module.
- `make_optimizers` gives a learnable `ParticlePrior` table A2 damping (alone
  in its group with beta1 0) and the critic the spike guard. Set
  `latent_damping_max_rate=0` and `d_guard_ratio=0` for plain Adam steps.
- `make_generator_optimizer(..., direct_particles=[...])` applies the
  direct-particle response to that param group.

Factory keyword arguments override constructor values for that call, without
changing the recipe. Optimizers exclude frozen parameters; G and prior have
separate groups at `lr` and `lr * prior_lr_mult`, while D uses `lr * d_lr_mult`.
`prior_betas` optionally overrides Adam betas for the prior group only.
Set `prior_kind="mog"`, `sigma_rel` and `standardize` through the recipe, or
override them locally in `make_prior`. Nonzero sigma with the atoms kind is rejected.
If G contains the supplied prior, its parameters are included only once.
Additional Adam options such as `fused=True` or `eps=1e-8` are passed to both
optimizers; configure learning rates and betas through the recipe. You can
still construct optimizers yourself, including separate prior optimizers or
additional parameter groups for learned noise.

`learning_rate_scale(step, total_steps, start=.6, floor=.05)` returns a Python
float: hold 1, then cosine decay to `floor`. `step` counts completed updates
(zero before the first update). It changes no optimizer state and clamps after
the horizon. EMA, update ratios, and scheduling remain caller-owned.

## TOML configuration

Constructors accept ordinary dictionaries from `tomllib.load`, `toml.load`, or
any other parser. The library does not read configuration files.

```toml
[particlegan]
name = "ddgan"
z_dim = 16
num_particles = 4096
num_classes = 8
betas = [0.0, 0.999]

# Alternative: configure components independently.
[prior]
z_dim = 16
num_particles = 4096

[loss]
loss_type = "logistic"
mode = "rp"
```

```python
import tomllib  # Python 3.10: install tomli and import it as tomllib.
from particlegan import ParticlePrior, GANLoss, get_recipe

with open("model.toml", "rb") as file:
    config = tomllib.load(file)
recipe = get_recipe(**config["particlegan"])
prior = ParticlePrior(**config["prior"])
gan = GANLoss(**config["loss"])
recipe = get_recipe(**{**config["particlegan"], "lr": 1e-4})
```

Choose recipe-owned values or independent component sections for your pipeline;
they are not merged automatically. Recipe arrays become immutable tuples.
The repository's experiment CLI accepts flat TOML/YAML files using its existing
experiment field names; see the [runner guide](experiment-runner.md).

## Loss augmentation and teacher/student pipelines

Each loss stands alone. For example, using latent features from your own model:

```python
from particlegan import ParticleRegularizer

spread = ParticleRegularizer(weight=0.1)
loss = existing_loss + spread(latent_features)  # [batch, latent_dim]
```

In a teacher/student pipeline, the teacher can supply the real targets. Freeze
teacher outputs when that is your intended gradient policy, and compose the
student objective yourself:

```python
with torch.no_grad():
    targets = teacher(inputs)
z, indices = prior.sample(len(inputs))
fake = student(z, inputs)
# After your D update, with D parameters frozen:
loss = supervised_loss(fake, targets)
loss = loss + adversarial_weight * gan.g_loss(D(fake), D(targets).detach())
loss = loss + spread(prior.z[indices.unique()])
```

Your pipeline controls teacher mode, conditioning, weights, gradient paths,
backward, and optimizer steps. None of these helpers require a dataset class,
teacher interface, or fixed training loop.

## Inference and checkpoints

For one-shot inference, save the matched EMA generator and prior from the
one-shot loop:

```python
torch.save({"generator": ema_g.state_dict(), "prior": ema_prior.state_dict(),
            "recipe": recipe.to_dict()}, "model.pt")
```

In the receiving application, construct the same G architecture and prior size,
then restore them. The recipe stores hyperparameters, not the generator class:

```python
from particlegan import Recipe

state = torch.load("model.pt", map_location=device, weights_only=True)
recipe = Recipe(**state["recipe"])
G = nn.Sequential(nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(0.2),
                  nn.Linear(64, 2)).to(device)
prior = recipe.make_prior().to(device)
G.load_state_dict(state["generator"])
prior.load_state_dict(state["prior"])
G.eval()
prior.eval()
with torch.inference_mode():
    z, _ = prior.sample(64)
    samples = G(z)
```

For DDGAN inference, also restore the schedule (or reconstruct it from the saved
`alpha_bar`). Given your restored conditional G and labels, the reverse loop is:

```python
process = DDGAN(alpha_bar=recipe.alpha_bar).to(device)
with torch.inference_mode():
    x = torch.randn((len(labels), *data_shape), device=device)
    for step in range(process.steps, 0, -1):
        t = torch.full_like(labels, step)
        z, _ = prior.sample(len(labels))
        clean = G(z, labels, xt=x, t=t)
        x = process.reverse(clean, x, t, torch.randn_like(x))
```

Here `data_shape` excludes the batch dimension, for example `(3, 32, 32)`.
Use the model's dtype for `x`, and set G and prior to evaluation mode first.
Inference needs no discriminator or optimizer. Resuming training additionally
requires your D, optimizer, scheduler, EMA, update-counter, and RNG state.

The [implementation report](../reports/api.md) records the design, migration,
and validation. Legacy research names such as `GradRegularizer`,
`VICRegLikeLoss`, `DiffusionSchedule`, and `DrawSource` remain in package
submodules for repository compatibility; the APIs above are the public entry points.
