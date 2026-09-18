# ParticleGAN API reference

ParticleGAN provides independent PyTorch priors, losses, and diffusion helpers.
You own the models, data, training loop, and checkpoints. Install the package
with `python -m pip install particlegan`; the core dependency is PyTorch.

## A minimal training loop

This complete example learns a synthetic 2D distribution. Replace `real` with a
batch from your pipeline and replace the two MLPs with your networks. The
recommended defaults cover the optimizer, regularizers, schedule, and EMA;
override only what your application needs.

```python
import copy
import torch
from torch import nn
from particlegan import get_recipe, learning_rate_scale

device = torch.device("cpu")  # Change to your device.
recipe = get_recipe()  # Use total_steps=5 for a quick smoke check.

G = nn.Sequential(nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(0.2),
                  nn.Linear(64, 2)).to(device)
D = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2),
                  nn.Linear(64, 1)).to(device)
prior = recipe.make_prior().to(device)
gan = recipe.make_loss()
penalty = recipe.make_gradient_penalty()
spread = recipe.make_prior_regularizer()
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
ema_g = copy.deepcopy(G).eval().requires_grad_(False)
ema_prior = copy.deepcopy(prior).eval().requires_grad_(False)

for step in range(recipe.total_steps):
    scale = learning_rate_scale(step, recipe.total_steps,
                               recipe.lr_anneal_start, recipe.lr_floor)
    for opt, rates in zip((opt_g, opt_d), base_lrs):
        for group, rate in zip(opt.param_groups, rates):
            group["lr"] = rate * scale

    real = torch.randn(recipe.batch_size, 2, device=device)
    z, indices = prior.sample(len(real))
    fake = G(z)

    # Update D; detached fakes leave G and the prior untouched.
    opt_d.zero_grad(set_to_none=True)
    d_loss = gan.d_loss(D(real), D(fake.detach()))
    d_loss = d_loss + penalty(D, real, fake.detach(), step=step + 1)
    d_loss.backward()
    opt_d.step()

    # Update G and the prior through the updated, frozen D.
    D.requires_grad_(False)
    opt_g.zero_grad(set_to_none=True)
    g_loss = gan.g_loss(D(fake), D(real).detach())
    g_loss = g_loss + spread(prior.z[indices.unique()])
    g_loss.backward()
    opt_g.step()
    D.requires_grad_(True)

    with torch.no_grad():
        for average, current in ((ema_g, G), (ema_prior, prior)):
            for target, source in zip(average.parameters(), current.parameters()):
                target.lerp_(source, 1 - recipe.ema_decay)

    if step % 100 == 0 or step + 1 == recipe.total_steps:
        print(f"step={step + 1} d={d_loss.item():.4f} g={g_loss.item():.4f}", flush=True)

with torch.inference_mode():
    z, _ = ema_prior.sample(64)
    samples = ema_g(z)  # [64, 2]
```

The G optimizer includes the prior at its own learning rate. The regularizer
uses unique sampled rows and already includes `recipe.prior_reg`. EMA tracks
both G and the prior; these simple modules have no buffers to update. For models
with running statistics, choose how to update their EMA buffers too. If your
critic has intentionally frozen parameters, restore their original flags rather
than enabling every parameter after the G step.

For command-line arguments, TOML input, and flushed JSON logs, see
[the runnable loop](../examples/pytorch_loop.py). The example-first presentation
is inspired by [LeJEPA's minimal guide](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md).

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
from particlegan import DDGAN, UCD, get_recipe, learning_rate_scale, ucd_loss

device = torch.device("cpu")
recipe = get_recipe("ddgan", num_classes=2)  # Add total_steps=5 for a smoke check.
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
penalty = recipe.make_gradient_penalty()
spread = recipe.make_prior_regularizer()
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
ema_g = copy.deepcopy(G).eval().requires_grad_(False)
ema_prior = copy.deepcopy(prior).eval().requires_grad_(False)

for step in range(recipe.total_steps):
    scale = learning_rate_scale(step, recipe.total_steps,
                               recipe.lr_anneal_start, recipe.lr_floor)
    for opt, rates in zip((opt_g, opt_d), base_lrs):
        for group, rate in zip(opt.param_groups, rates):
            group["lr"] = rate * scale

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
    d_loss += penalty(lambda x: D(x, labels, xt=xt, t=t)[0],
                      x_prev, fake.detach(), step=step + 1)
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
| [TOML](#toml-configuration) | Pass loaded dictionaries to constructors |
| [Other pipelines](#loss-augmentation-and-teacherstudent-pipelines) | Compose with existing objectives |
| [Inference](#inference-and-checkpoints) | Generate from saved G and prior states |

All names below are exported from `particlegan`. Modules use ordinary
`.to(device, dtype)`, `.parameters()`, and `.state_dict()` behavior. Move models
and priors before constructing optimizers. Stateless loss helpers need no device
setup. No component selects a device, seeds global RNG, or steps an optimizer.

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
                 *, sigma_rel=1/40, standardize=True)
```

An equal-weight mixture: choose component `i` uniformly, then draw
`z = means()[i] + sigma * eps`, with standard-normal epsilon. The raw component
centers are the parameter `prior.z`. Sigma is a **shared fixed buffer**, calibrated
once as `sigma_rel * d0`, where `d0` is the median nearest-neighbor distance of
the initial read-space means. At least two components and positive initial
median spacing are required. `learnable=False` freezes the table as a buffer.

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
| `sigma`, `d0` | Saved scalar buffers, moving with `.to(...)` |
| `calibrate()` | Explicitly reset d0 and sigma from current means; not a training-step operation |

`fixed_first_n=True` fixes component indices, **not epsilon**. For a stable scatter,
save a fixed epsilon tensor too:

```python
from particlegan import MoGParticlePrior

prior = MoGParticlePrior()
eps = torch.randn(64, prior.z_dim, device=prior.z.device, dtype=prior.z.dtype)
z, indices = prior.sample(64, fixed_first_n=True, eps=eps)
# Reuse eps at every snapshot; use prior.means()[indices] only for a centers-only audit.
```

Explicit epsilon must match the sampled codes' shape, device and dtype. An explicit
generator controls both component selection and Gaussian draws without touching
global RNG. `sigma_rel=0, standardize=False` preserves `ParticlePrior` outputs and
RNG consumption; zero sigma never draws noise. `eval()` keeps Gaussian noise on.

For DDP, sample indices from the unwrapped prior, then call the wrapped module:
`z = wrapped_prior(indices, generator=rng)`. Use `prior.z` for VICReg rather than
`prior(indices)` or sampled codes. The one-shot MoG benchmark regularizes the
full raw table at N ≤ 1024, otherwise `prior.z[indices.unique()]`. The denoising
trainer and loops above regularize sampled unique raw rows for either prior.

For EMA, deepcopy the prior and average its learned `z`; the fixed buffers retain
their calibrated values. Standardization is computed from the EMA table itself.
State dicts include `z`, `sigma`, `d0`, and `_extra_state` containing `sigma_rel`
and `standardize`. Reconstruct with matching dimensions, then `load_state_dict`:
read settings and noise are restored even if constructor defaults differ.
Legacy experimental checkpoints containing only z/sigma/d0 are accepted; supply
their original `standardize` setting when constructing the prior.

Calibration uses SciPy's CPU tree when available. Install `particlegan[mog]` for
that optional acceleration. Without SciPy, exact Torch distances use bounded
temporary memory but quadratic work; large low-dimensional tables benefit from
the tree. No SciPy or NumPy is needed for sampling, gradients or the fallback.

The default experiment is [configs/mog/default.toml](../configs/mog/default.toml),
run via `python -u experiments/train_100gaussians.py --config configs/mog/default.toml`.
It retains the benchmark's networks, Fourier discriminator and full metric suite.
The corresponding API preset is `get_recipe("mog")`; a recipe alone does not
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
GradientPenalty(arm="b_cap", coeff=1.0, kappa=1.0, lazy_k=1, norm="l2",
                target_anneal="none", total_steps=0,
                method="autograd", fd_eps=0.05)
```

Call `penalty(D, real, fake, step=1, generator=None)` to get a scalar loss.
`D` may be a module or callable returning one scalar score per example. The
default is the mean squared excess of the input-gradient norm above `kappa`,
averaged over real and fake samples and scaled by `coeff`.

| `arm` | Penalty |
| --- | --- |
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

`get_recipe("ae_gan")`, `get_recipe("vae_gan")` and `get_recipe("ae_ddgan")`
add reconstruction encodings to caller-owned networks and loops. The default
VAE selects one particle with prior-matching Gaussian noise and constant joint
KL. No KL penalty is added by `reconstruction_loss`.

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
get_recipe(name="gan", **overrides)          # Returns a frozen Recipe.
recipe.replace(**overrides)                  # Returns a new Recipe.
recipe.to_dict()                             # All resolved fields.
Recipe(**resolved_dict)                      # Restore resolved fields.
```

`get_recipe("ddgan", num_classes=8)` starts with the DDGAN defaults and
overrides its class count. `Recipe` is the resolved data object: setting only
`Recipe(name="ddgan")` does **not** select those defaults. Use `get_recipe` to
resolve named presets. Unknown fields are rejected. Historical names
`100gaussians` and `denoising` remain accepted aliases for GAN and DDGAN.

`get_recipe("mog")` selects the compact MoG leader: 400 components, z_dim=4,
sigma_rel=1/40, standardized reads, 28,000 steps, prior LR multiplier 100
(relative to G, giving 0.06), and prior betas `(0.5, 0.999)`. G and D retain
betas `(0, 0.999)`. Other GAN recipe settings are unchanged. `get_recipe()`
and `get_recipe("gan")` still select the existing atoms recipe.

`get_recipe("ddgan_mog")` combines DDGAN/class-only UCD with the MoG settings
from the 100k study: 400 components, z_dim=4, sigma_rel=1/40, standardized reads,
100,000 updates, prior LR multiplier 100, prior betas `(0.5, 0.999)`, and
`lr_floor=1.0` for a constant learning rate. Other fields use DDGAN defaults.
The Gaussian diffusion schedule is unchanged; MoG supplies G's latent codes.

```python
recipe = get_recipe("ddgan_mog", num_classes=4)
prior = recipe.make_prior().to(device)
process = DDGAN(recipe.alpha_bar).to(device)
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
z, indices = prior.sample(batch_size)
prior_loss = recipe.make_prior_regularizer()(prior.z[indices.unique()])
```

All fields can be overridden, including `total_steps` and `lr_floor`. The recipe
supplies the package hyperparameters from the
[100k study](../reports/denoising-toy/mog_capacity_100k/READOUT.md), not its
architecture: that study used G width 32, D width 128 and depth 3. Callers own
the networks, loops and EMA. `get_recipe("mog")` continues to select one-shot
GAN settings; `get_recipe("ddgan")` continues to use atoms and 56,000 updates.

| Field | `get_recipe()` / `gan` | `ddgan` |
| --- | --- | --- |
| `model` | `gan` | `ddgan` |
| `conditioning`, `num_classes` | `scalar`, `None` | `ucd`, `4` |
| `z_dim`, `num_particles` | `4`, `20_000` | Same |
| `prior_kind`, `sigma_rel`, `standardize` | `particles`, `0`, `True` (standardize applies only to MoG) | Same |
| `loss_type`, `gan_mode` | `logistic`, `rp` | Same |
| `lr`, `d_lr_mult`, `prior_lr_mult` | `.0006`, `1.5`, `10` | Same |
| `betas` | `(0, .999)` | Same |
| `prior_betas` | `None` (inherit `betas`) | Same |
| `reg_arm`, `reg_coeff`, `reg_kappa` | `b_cap`, `1`, `1` | Same |
| `reg_every`, `reg_method` | `1`, `autograd` | Same |
| `prior_reg`, `ema_decay` | `1`, `.995` | Same |
| `lr_anneal_start`, `lr_floor` | `.6`, `.05` | Same |
| `batch_size`, `total_steps` | `256`, `7_000` | `256`, `56_000` |
| `ucd_target`, `ucd_weight` | `class`, `.02` (unused) | `class`, `.02` |
| `alpha_bar` | `(1, .9, .5, .05, .0001)` (unused) | `(1, .9, .5, .05, .0001)` |

Recipes describe recommended starting defaults; they do not construct
architectures or execute a training procedure. Networks, data, and training
budgets remain application choices.

| Optional factory | Result |
| --- | --- |
| `recipe.make_prior(**kwargs)` | `ParticlePrior` or `MoGParticlePrior` selected by `prior_kind` |
| `recipe.make_loss(**kwargs)` | `GANLoss` using recipe loss and mode |
| `recipe.make_gradient_penalty(**kwargs)` | `GradientPenalty` using recipe penalty settings |
| `recipe.make_prior_regularizer(**kwargs)` | `ParticleRegularizer` with `weight=recipe.prior_reg` already applied |
| `recipe.make_optimizers(G, D, prior=None, encoder=None, **adam_kwargs)` | `(opt_g, opt_d)`, ordinary Adam optimizers |

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
