# ParticleGAN PyTorch API

Status: implemented on branch `api`; validation results below.

For usage and constructor contracts, see the
[API reference and minimal training loop](../docs/api.md).

The follow-up [repository dogfooding report](api-dogfood.md) covers the
experiment/example migration, 797 existing configs, GPU smoke cases, and
before/after checkpoint comparisons.

Public defaults now use `get_recipe()` / `"gan"` and `get_recipe("ddgan")`.
The historical names below remain accepted for existing experiment configs;
their numerical settings are unchanged. The reference opens with standalone
GAN and DDGAN + UCD training loops.

## Goal and boundary

Install with pip, import from `particlegan`, and use the same primitives in this
repository and in another project's ordinary PyTorch training loop. Defaults
come from `examples/100gaussians.py` and `experiments/train_denoising.py` with
`configs/denoising/ddgan_ucd.yaml`, as selected by the user.

The caller owns networks, data, backward, optimizer steps, devices, precision,
distributed wrapping, logging, and checkpoints. No required trainer, `.fit()`,
dataset class, callbacks, implicit downloads, or automatic GPU selection.
Loss helpers return tensors and never call backward or step an optimizer.

### Independent use cases

1. **Loss augmentation:** add `weight * ParticleRegularizer()(latent_rows)` or
   `GradientPenalty()(critic, real, fake)` to an existing objective. None of the
   other package components are required. Callers choose where to detach;
   regularization must preserve gradients to the tensors being regularized.
   Critic gradient penalties deliberately recompute derivatives on detached
   candidate tensors, updating the critic rather than the data producer.
2. **Teacher/student:** the teacher may supply targets, samples, or features.
   The student can be an arbitrary module using a particle prior. The caller
   composes distillation, reconstruction, adversarial, and regularization losses
   and decides which networks receive gradients. No dataset-only interpretation
   of `real`, mandatory teacher interface, or fixed update ratio.
3. **Inference:** reconstruct the caller's G and prior, load their state dicts,
   call `.eval()`, and generate under `torch.no_grad()` or inference mode. Only
   DDGAN generation additionally needs its schedule and reverse loop. D,
   optimizers, training configs, and experiment dependencies are unnecessary.
   EMA inference saves/restores the matched EMA G and EMA prior together.

README examples must show these independent uses. A full loop is an optional
demonstration; it is never the only documented path into the package.

## Public building blocks

| API | Responsibility |
| --- | --- |
| `ParticlePrior(num_particles=20_000, z_dim=4)` | `nn.Module` with a learnable table; `sample(n)` returns `(z, indices)`; `forward(indices)` supports wrapped module calls |
| `GaussianPrior(z_dim=4)` | Fresh Gaussian draws with the same `(z, None)` sampling interface, without a large reference table |
| `GANLoss(loss_type="logistic", mode="rp")` | `d_loss(real_logits, fake_logits)` and `g_loss(fake_logits, real_logits)` |
| `GradientPenalty(arm="b_cap", coeff=1.0, ...)` | `penalty(D, real, fake, step=1)` returns `(tensor, stats)`; callable form returns just the tensor |
| `ParticleRegularizer()` | VICReg variance/covariance loss on supplied rows; zero for fewer than two rows |
| `DDGAN(alpha_bar=(1, .9, .5, .05, .0001))` | Gaussian forward pairs and reverse transitions; schedule is registered buffers |
| `UCD(network, num_classes, target="class", num_steps=None)` | Select a class score from logits without injecting labels into the network |
| `ucd_loss(real_logits, fake_logits, targets, weight=.02)` | Discriminator-only real/fake cross-entropy |
| `get_recipe("100gaussians" or "denoising", **overrides)` | Immutable, inspectable defaults and optional component/optimizer construction |
| `learning_rate_scale(step, total_steps, start=.6, floor=.05)` | Pure delayed-cosine schedule; caller sets learning rates |

Keep legacy names (`GradRegularizer`, `VICRegLikeLoss`, `DiffusionSchedule`,
`FreshGaussianPrior`, `make_prior`, and `DrawSource`) available in package
submodules for experiment migration. Repository `lib` compatibility imports
delegate to the package; the implementation has one home in `particlegan`.
The ambiguous historical `make_prior("gaussian")` frozen-table alias stays
confined to that compatibility API. New examples use explicit classes.

## Minimal integration

```python
from particlegan import ParticlePrior, GANLoss, GradientPenalty, ParticleRegularizer

prior = ParticlePrior(z_dim=4).to(device)
gan = GANLoss()
penalty = GradientPenalty()
spread = ParticleRegularizer()

# G and D are the caller's nn.Modules. Include prior.parameters() in an optimizer.
z, ids = prior.sample(len(real))
fake = G(z)
d_loss = gan.d_loss(D(real), D(fake.detach()))
d_loss = d_loss + penalty(D, real, fake.detach())
# Caller: zero D grads, d_loss.backward(), step D.

# Recompute scores after the D update; freeze D parameters for this G update.
g_loss = gan.g_loss(D(fake), D(real).detach())
g_loss = g_loss + spread(prior(ids.unique()))
# Caller: zero G/prior grads, g_loss.backward(), step G/prior, update EMA.
```

The README must include a complete executable loop with its optimizer setup,
freeze/restore behavior, schedule, and EMA, or link to an executable example.
The snippet above illustrates the primitive boundary, not a complete recipe.
Fresh Gaussian and frozen priors have no trainable table regularization.

## Optional recipe helpers

```python
recipe = get_recipe("100gaussians", z_dim=16, num_particles=4096, lr=3e-4)
prior = recipe.make_prior().to(device)
gan = recipe.make_loss()
penalty = recipe.make_gradient_penalty()
spread = recipe.make_prior_regularizer()
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
print(recipe.to_dict())
```

`make_optimizers` returns ordinary Adam optimizers: one with separate G and prior
parameter groups and one for D. Frozen parameters are excluded; prior groups
are omitted for nonlearnable priors. Optimizers are constructed after `.to()`.
The caller can supply their own optimizers or use the recipe values directly.
Factories must read the resolved overrides, and `recipe.replace(...)` returns a
new validated recipe. Unknown options fail immediately.

`learning_rate_scale` takes the number of completed updates (zero before the
first step), matching the reference trainers. It never changes an optimizer.

There is no all-in-one `ParticleGAN` training object in this first API. DDGAN and
UCD compose independently using the same losses, priors, and optimizers.

## TOML is ordinary constructor data

Public constructors accept keyword arguments from a loaded dictionary. No
configuration framework, file parser, or custom config object is required by
the primitives. This works with `tomllib.load`, `toml.load`, or another parser.

```toml
[particlegan]
name = "denoising"
z_dim = 16
num_particles = 4096
num_classes = 8
lr = 0.0003

[prior]
num_particles = 4096
z_dim = 16

[loss]
loss_type = "logistic"
mode = "rp"
```

```python
import tomllib  # Python 3.11+; tomli on 3.10
with open("model.toml", "rb") as f:
    config = tomllib.load(f)

recipe = get_recipe(**config["particlegan"])
# Or use independent sections directly, without any recipe:
prior = ParticlePrior(**config["prior"])
gan = GANLoss(**config["loss"])
```

Recipe configuration and per-component configuration are alternative entry
points; users choose which owns the values. Code overrides are explicit dict
merges (`get_recipe(**{**config["particlegan"], "lr": 1e-4})`). Unknown fields
raise errors. Normalize TOML list values such as `betas` and `alpha_bar` to
immutable tuples inside recipes. `Recipe(**full_resolved_dict)` reconstructs a
resolved recipe; `get_recipe(**overrides_dict)` resolves a named preset.

Repository experiment runners gain a shared TOML/YAML config reader, preserving
their existing flat schema and command-line flags. Ship TOML defaults/examples
for the two primary recipes; existing YAML configs remain supported. New config
files use TOML, while historical study configs and recorded artifacts retain
their original formats. Record effective config values for provenance regardless
of input format. Python 3.10 experiment installs use the conditional `tomli`
extra; the core package needs no TOML dependency.

## DDGAN and UCD contracts

```python
from particlegan import DDGAN, UCD, ucd_loss

process = DDGAN().to(device)
critic = UCD(logit_network, num_classes=4)
t = torch.randint(1, process.steps + 1, (len(real),), device=device)
x_prev, xt = process.forward_pair(real, t, rng)
z, ids = prior.sample(len(real), generator=rng)
x0_hat = G(z, labels, xt=xt, t=t)  # caller-defined generator
fake_prev = process.reverse(x0_hat, xt, t, torch.randn_like(xt))
real_score, real_logits = critic(x_prev, labels, xt=xt, t=t)
fake_score, fake_logits = critic(fake_prev.detach(), labels, xt=xt, t=t)
d_loss = gan.d_loss(real_score, fake_score)
d_loss += ucd_loss(real_logits, fake_logits, critic.ucd_labels(labels, t))
d_loss += penalty(lambda x: critic(x, labels, xt=xt, t=t)[0],
                  x_prev, fake_prev.detach())
```

- Data tensors have shape `[B, ...]`, latent codes `[B, z_dim]`, labels and times
  `[B]` with dtype `torch.long`. Times run from 1 through the number of transitions.
- A class-only UCD network receives `network(x, xt=xt, t=t)` and returns `[B, C]`.
  For one-shot generation it receives `network(x)`. The wrapper sees labels;
  the network does not.
- Joint UCD uses `target="time_class", num_steps=T`; its network receives
  `network(x, xt=xt)`, emits `[B, T*C]`, and selects `(t-1)*C + class`.
- The gradient penalty differentiates only the candidate, with `xt`, time, and
  labels fixed. The adversarial fake transition must retain gradients for G and
  prior in the G update. CE belongs to D, matching the current trainers.
- G predicts clean `x0`, then `reverse` constructs `x_(t-1)`. No diffusion MSE.
  Gaussian corruption, initial state, and reverse noise remain distinct from
  learned latent particles. Gaussian reverse noise is the denoising default.
- Sampling is a caller-owned loop over `T,...,1`, starting at Gaussian `x_T`,
  drawing fresh latent indices and reverse noise at each step. DDGAN has no
  optimizer or mandatory training-loop dependency.

## Exact default recipes

| Setting | `100gaussians` | `denoising` |
| --- | --- | --- |
| Process | One-shot GAN | Four-step DDGAN |
| Latent | 20,000 learned particles, dimension 4 | Same |
| Adversarial loss | Rp logistic | Same |
| D penalty | Exact L2 bcap, coefficient 1, cap 1, every update | Same |
| Particle regularizer | VICReg weight 1 on unique sampled rows | Same |
| Adam | betas `(0, .999)` | Same |
| Learning rates | G `.0006`, D `.0009`, prior `.006` | Same |
| EMA | G and prior, decay `.995` | Same |
| LR schedule | Hold 60%, cosine to 5% of initial LR | Same |
| Suggested batch size | 256 | 256 |
| Suggested horizon | 7,000 updates | 56,000 updates |
| UCD | Off | Class-only, 4 classes, CE weight `.02` |
| Cumulative signal variance | N/A | `[1, .9, .5, .05, .0001]` |

These are the user-selected defaults. Class count, data dimension, and network
architecture belong to the application and are easy to override. The reference
benchmarks also use MLPs and two Fourier frequencies in D; keep those in the
benchmark examples and document them. Supplying a different architecture does
not reproduce the benchmark by itself. Preserve the CIFAR, sparse, trajectory,
and historical experiment settings when migrating them.

## Package and migration

- Publish the `particlegan` namespace with Torch as the sole required runtime
  dependency. Retain setuptools; package core code without experiment assets.
- Put research dependencies in an `experiments` extra; retain `images` and `dev`.
  Document editable installation with the extras used by this repository.
- CPU operation, `.to(device/dtype)`, and module `state_dict` round trips must
  work. Explicit `torch.Generator` arguments must not consume global RNG streams.
- Public primitives do not choose seeds or alter global Torch settings. No
  promise of automatic DDP/compile/AMP support beyond the tested paths.
- DDGAN and UCD validate inputs by default. `validate_args=False` skips numeric
  bounds checks for already validated indices without skipping shape/dtype
  checks. Existing experiments use this path to avoid new CUDA synchronization.
- Refactor every example/experiment consumer of shared priors, losses, penalties,
  and DDGAN transitions to import the package. Move shared UCD loss/label logic
  to package helpers, and use selected recipe defaults in the primary trainers.
- Keep specialized experiment loops, metrics, logging, RNG streams, checkpoint
  keys, config aliases, and architectures. Low-level use of this API is intended.
- Include package sources in experiment source hashes/archives, so moving code
  cannot make resume/provenance checks overlook changes.
- Preserve legacy behavior where defaults were explicit. Handle singleton
  regularization safely and test any deliberate corrections.

## Implementation assignments and validation

Implementation was split across three subagents after this design was written:

1. **Package primitives:** finish extraction, public exports, recipe factories,
   DDGAN/UCD validation, small-cloud regularization, and focused unit checks.
2. **Repository migration:** migrate consumers and defaults, share UCD helpers,
   preserve experiment behavior and source provenance, add shared TOML/YAML
   loading and primary TOML configs. Own `lib`, existing examples, configs,
   and experiment scripts.
3. **Packaging and documentation:** package metadata, README, and a new executable
   user-owned training-loop example. Keep this design as the interface reference.

The coordinating agent checks integration and runs the regression suite, wheel
build/install/import outside the checkout, and CPU example smoke checks. Add
meaningful tests for gradient isolation, sampled-prior updates, recipe values,
DDGAN coefficients, UCD conditioning, RNG isolation, and checkpoint round trips.
Use fixed-input parity checks against existing primitives where feasible.
No training sweeps or seed experiments. Write validation output to a stable log
that can be tailed; report completed checks and any limitations here.

Framework references: ordinary module state/device behavior follows the
[PyTorch module conventions](https://docs.pytorch.org/docs/2.14/notes/modules.html).
Wheel and source-distribution validation follows the
[Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## Completion and validation

Implemented the primitives, optional recipe factories, direct TOML dictionary
construction, TOML/YAML experiment inputs, repository migration, and README
examples on `api`. Runtime packaging requires only Torch. Existing experiment
loops remain explicit consumers of the shared library. Repository `lib` imports
remain compatibility shims; the wheel exports only `particlegan`.

| Check | Result |
| --- | --- |
| Full CPU regression suite | 130 passed, 4 opt-in CUDA resume tests skipped; 22 subtests passed |
| Final TOML/config audit | 6 passed, including a subsequently added TOML manifest/hash test |
| Defaults against the original Git revision | All six trainer DEFAULTS mappings unchanged |
| Distribution build | Wheel and source archive built successfully |
| Installed wheel outside checkout | Import, recipe overrides, particle inference, DDGAN defaults passed; research imports blocked |
| Required wheel dependencies | Torch only; research/image/test dependencies are extras |
| Caller-owned loop | Default and TOML CPU smokes passed; installed-wheel TOML smoke produced finite EMA samples |

Validation log: `/tmp/particlegan-api-validation.log`.

```bash
tail -F /tmp/particlegan-api-validation.log
```

Build artifacts are in `/tmp/particlegan-api-dist/`. Version `0.2.0` is prepared
locally and has not been published to PyPI. CI now exercises Python 3.10–3.12,
the regression suite, and an installed-wheel smoke outside the checkout; the
local checks ran on Python 3.12. No new benchmark study or seed sweep was run,
so there is no new experiment leaderboard or claim of improved model quality.
The next release step is reviewing and publishing this API; additional model
performance experiments are unnecessary for validating this refactor itself.
