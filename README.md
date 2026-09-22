# ParticleGAN

**Learnable particle priors and GAN building blocks for PyTorch.**

[API reference](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md) · [Minimal GAN loop](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#a-minimal-training-loop) ·
[Minimal DDGAN + UCD loop](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#a-minimal-ddgan--ucd-loop)

[![Tests](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml/badge.svg)](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml)

![100 Gaussians with Particle Prior](https://raw.githubusercontent.com/255BITS/ParticleGAN/master/100gaussians.gif)

## Installation

Requires Python 3.10+ and PyTorch. Install from PyPI:

```bash
python -m pip install particlegan
```

For development and the repository's research experiments:

```bash
git clone https://github.com/255BITS/ParticleGAN.git
cd ParticleGAN
python -m pip install -e '.[experiments,dev]'
# Image experiments also need the images extra:
# python -m pip install -e '.[experiments,images,dev]'
```

CI tests Python 3.10–3.12 and builds installable distributions. See
[CI and PyPI releases](https://github.com/255BITS/ParticleGAN/blob/master/docs/releasing.md) for the automated publishing setup.

## Use in your PyTorch project

For an unconditional GAN, supply your networks and real batches; the optional
trainer applies the recipe's optimizer settings, particle regularization,
learning-rate decay and EMA:

```python
from particlegan import get_recipe

recipe = get_recipe("gan", total_steps=7000)
trainer = recipe.make_trainer(G, D)  # Move your networks to their device first.
for real in batches:               # Supply up to recipe.total_steps batches.
    stats = trainer.step(real)
samples = trainer.sample(256)       # Live weights; ema=True reports EMA separately.
```

Run `python -u examples/quickstart_gan.py --steps 1000` for a complete PyTorch-only
example with flushed logs and resumable checkpoints. See
[training and checkpoint contracts](docs/api.md#gantrainer).
The optional helper supports scalar, unconditional GANs with particle priors;
the independent primitives remain available for other training loops.

The [convergence leaderboard](reports/behavioral_baseline/convergence/README.md)
compares full behavioral passes, sustained coverage, speed and data-scale
sensitivity. `get_recipe("gan_behavioral")` exposes the measured toy candidate.
A [learned LR controller study](reports/learned_lr/README.md) also retains fitted
weights and held-out comparisons; its current policy does not beat cosine.
The broader [transfer leaderboard](reports/transfer_suite/README.md) separates
required regressions, practical ranking tests, and nonblocking stress diagnostics.
On the actual 100-Gaussian task the behavioral candidate ties stock's live convergence step, so
`"gan"` keeps the existing defaults. Neither preset guarantees transfer to new
architectures or data units.

See the [minimal GAN loop](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#a-minimal-training-loop),
[minimal DDGAN + UCD loop](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#a-minimal-ddgan--ucd-loop), and
[API reference](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#reference-index) for complete examples and contracts.

Our examples and experiment trainers consume these same public primitives and
recipe factories. See the [migration and compatibility checks](https://github.com/255BITS/ParticleGAN/blob/master/reports/api-dogfood.md)
for existing-config GPU smoke tests and checkpoint comparisons.

The [single-transition example](docs/transition-gan.md) defaults to the winning
MisGAN-inspired encoder: `G1 -> st`, `G2 -> at`, `G3 -> st+1`, plus
`E(st, at) -> z -> G3 -> st+1`. It uses 1,024 MoG components, bcap, joint/action
critics and a shared state critic. Run `python -u examples/transition_gan.py`.
See the [toy demo](reports/transition/demo/index.html),
[architecture and losses](docs/transition-gan-encoder.md), and
[leaderboard](reports/transition/leaderboard/README.md).

The [Lunar Lander world-model example](docs/gym-world-model.md) extends the three
generators to individual simulator transitions, with MoG1024, encoder routing,
terrain context, and joint/marginal critics. It includes replayed counterfactual
actions and direct/persistence/reconstruction comparisons on a finite dataset.
The [control experiment](docs/gym-control.md) compares expert action imitation
with joint three-generator training. Run `python -u examples/gym_lander_live.py`
to compare controllers in the live simulator with Play, Pause, and Reset.
The [state-only encoder experiment](docs/gym-state-control.md) trains from scratch
and tests whether G1/G3 auxiliary losses improve G2's control, compared with
detached diagnostic heads under the same training budget.
The [sparse-action experiment](docs/gym-sparse-action.md) keeps transitions from
47 expert episodes while revealing actions from only five, testing whether
auxiliary learning helps when explicit action supervision is scarce.
The [GAN control experiment](docs/gym-gan-control.md) trains all three generators
and the state encoder adversarially throughout, with a dedicated GAN-only
leaderboard comparing joint and marginal discriminators.
The [previous-action GAN experiment](docs/gym-previous-gan.md) restores
`E(st, at-1)` and trains the three-generator model from scratch with joint and
marginal GAN losses plus expert action MSE, using all 47 labeled episodes.
The [slider-error experiment](docs/gym-slider-gan.md) replaces paired MSE/BCE
supervision with an Anima-style critic on noisy prediction errors, while keeping
joint and marginal GAN training active.
The [ParticleGAN fine-tune](docs/gym-particle-finetune.md) keeps L2 weights at
0 and updates `E_control` and G2 with YuE2 paired-error RpGAN at `adv_weight=1`
plus sample-point b_cap on the edit critic. G1, G3, the paired encoder, the
prior, and the transition discriminators stay frozen. `particle.yaml` is that
default. The [safe-fast term](docs/gym-safe-fast.md) is a separate config,
`particle_safe_fast.yaml`, and does not replace it.
The [slider-error fine-tune](docs/gym-slider-finetune.md) uses the imitation
fine-tune's world-model initialization and replaces only action MSE with the
paired-error critic. G1, G3, the paired encoder, the prior, and the transition
discriminators stay frozen.

```python
from particlegan import ParticlePrior, GANLoss, GradientPenalty, ParticleRegularizer

prior = ParticlePrior().to(device)  # 20,000 learnable particles, z_dim=4
adversarial = GANLoss()            # relativistic-paired logistic loss
penalty = GradientPenalty()       # exact L2 cap penalty, weight=1, cap=1
spread = ParticleRegularizer()    # variance/covariance regularization, weight=1

# Customize with ordinary keyword arguments:
prior = ParticlePrior(num_particles=4096, z_dim=16).to(device)
adversarial = GANLoss(loss_type="hinge", mode="vanilla")
```

The [locked shared stamp](docs/locked-shared.md) is a separate, frozen demo
posture: RpGAN logistic, sample-point `b_cap` at coeff 1 and κ 1 every step,
feature matching off, cover weight 1.5, and a 12-particle cloud at
`particle_l2` 0.02 when you build one. The host critic stays yours.
`Recipe("gan")` is still 20_000 particles. This stamp is not a Music or Anima
transfer, and Lunar Lander does not use it yet.

```python
from particlegan.locked_shared import make_gan_loss, make_b_cap

loss = make_gan_loss()     # GANLoss("logistic", "rp")
penalty = make_b_cap()     # GradientPenalty b_cap, κ=1, lazy_k=1
```

[Measured conceptmod parity and leaderboard](reports/locked_shared/README.md):
three CPU training toys, numerical outcomes, and ten reference comparisons.
Configuration checks do not contribute to the score.

[Full live-weight behavioral baseline](reports/behavioral_baseline/README.md):
nine trained toys, 29 numerical bounds, separate EMA diagnostics, and ten shared
application checks. Includes a passing config and a reusable comparison runner.

`prior.sample(batch_size)` returns `(z, indices)`, with `z` shaped `[B, z_dim]`.
Include `prior.parameters()` in your generator optimizer to learn the particles.
Use `GaussianPrior(z_dim=16)` for fresh Gaussian samples with the same sampling
interface; its indices are `None` and it needs no particle regularization.

### Mixture-of-Gaussians particles (0.3.0)

`MoGParticlePrior` gives each learned particle a continuous Gaussian neighborhood.
Its defaults use the selected compact configuration: **400 components, z_dim=4,
sigma_rel=1/40, standardized means**. Sigma is shared and fixed after calibration.

```python
from particlegan import MoGParticlePrior, get_recipe

prior = MoGParticlePrior().to(device)
z, component_ids = prior.sample(256)

recipe = get_recipe("mog")  # 400 components, 28k steps, prior LR 0.06, beta1=0.5
prior = recipe.make_prior().to(device)
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
# Regularize raw means, not noisy draws. Use the full table at N <= 1024.
prior_loss = recipe.make_prior_regularizer()(prior.z)
```

Use `ParticlePrior` for atoms. Existing GAN/DDGAN presets and examples keep their
defaults; `mog` and `ddgan_mog` are dedicated opt-in recipes. The MoG recipe's `prior_lr_mult=100`
is relative to G's LR, giving 0.06. It is equivalent to the experiment's 10×
multiplier on the original prior LR of 0.006.

Run the default MoG benchmark with the existing experiment trainer:

```bash
python -u experiments/train_100gaussians.py --config configs/mog/default.toml
# During training:
tail -F results/mog/default/log.txt
```

The 400-component model passed the C0 acceptance envelope at 28k steps on the
100-Gaussian benchmark, using 50× fewer components and 4× the original updates.
This is a single-seed result; both MoG and atoms remain useful.
[Results and tradeoffs](results/mog/COMPONENT_SCALE.md) ·
[MoG API and checkpoint details](docs/api.md#mogparticleprior) ·
[Changelog](CHANGELOG.md).

Core MoG sampling and calibration work with PyTorch alone. For faster calibration
of large low-dimensional tables, install the optional extra with
`python -m pip install 'particlegan[mog]'` (or `python -m pip install -e '.[mog]'`
from this checkout).

### Particle AE-GAN, VAE-GAN and AE-DDGAN (0.5.0)

```python
from particlegan import get_recipe

recipe = get_recipe("vae_gan")  # also: ae_gan, ae_ddgan
prior = recipe.make_prior()
# query = E(x): [batch, recipe.z_dim], produced by your encoder
# encoded = recipe.encode(query, prior)
# x_hat = G(encoded.codes)  # [batch, draws, observed dimensions]
# loss = encoded.reconstruction_loss(x_hat, x)
```

`vae_gan` selects one particle and adds fixed-sigma Gaussian noise. Its joint
KL is constant, so it needs no KL regularizer in training. AE uses a deterministic
bounded offset instead. You own the networks, loop and loss composition;
reconstruction never silently adds KL. See the
[guide and runnable example](https://github.com/255BITS/ParticleGAN/blob/master/docs/particle-autoencoders.md)
for explicit ELBO reporting, optional categorical inference, AE-DDGAN integration
and measured limits. Genuine VAE results are toy-only; AE-GAN and AE-DDGAN have
matched CIFAR32 evidence.

### DDGAN with MoG particles (0.4.0)

Select the combined recipe with one name:

```python
from particlegan import DDGAN, get_recipe

recipe = get_recipe("ddgan_mog")
prior = recipe.make_prior().to(device)
process = DDGAN(recipe.alpha_bar).to(device)
opt_g, opt_d = recipe.make_optimizers(G, D, prior)

z, indices = prior.sample(batch_size)
prior_loss = recipe.make_prior_regularizer()(prior.z[indices.unique()])
```

Defaults: **400 components, z_dim=4, sigma_rel=0.025, standardized reads,
100k updates, constant LR, prior LR 0.06, prior betas (0.5, 0.999)**, and the
existing four-step DDGAN/class-only UCD settings. Override any recipe field with
keywords, for example `get_recipe("ddgan_mog", num_classes=8, z_dim=16)`.

These are the package hyperparameters from the
[100k study](reports/denoising-toy/mog_capacity_100k/READOUT.md), which used a
32-wide generator and 128-wide discriminator. You supply the networks, data,
training/sampling loop and EMA; the recipe does not select architectures.

### Add a loss to an existing pipeline

Components are independent. For example, add a critic penalty or a particle
spread term to losses your pipeline already computes:

```python
# In your discriminator update:
d_loss = existing_d_loss + penalty(D, real, fake.detach(), step=step)

# In your generator/prior update, when sampled particle indices are available:
g_loss = existing_g_loss + spread(prior.z[indices.unique()])
# Your code calls backward() and optimizer.step().
```

To use the adversarial objective itself, call
`adversarial.d_loss(D(real), D(fake.detach()))` for D and
`adversarial.g_loss(D(fake), D(real).detach())` for G. Freeze D's parameters
for the G update while retaining gradients through `D(fake)`.

### Selected defaults, with easy overrides

```python
from particlegan import get_recipe

recipe = get_recipe()  # Recommended GAN defaults.
recipe = recipe.replace(z_dim=16, num_particles=4096, lr=3e-4)
prior = recipe.make_prior().to(device)
adversarial = recipe.make_loss()
penalty = recipe.make_gradient_penalty()
spread = recipe.make_prior_regularizer()
opt_g, opt_d = recipe.make_optimizers(G, D, prior)  # after moving modules to device
print(recipe.to_dict())  # inspect every resolved value
```

The optional optimizer helper returns ordinary Adam optimizers. The G optimizer
has separate generator and particle groups. You can build your own optimizers
using the recipe's fields instead. Recipes are immutable; `.replace(...)`
returns a new one. Unknown options raise errors.

| Default | `get_recipe()` / `gan` | `ddgan` |
| --- | --- | --- |
| Generation | One-shot GAN | Four-step DDGAN |
| Prior | 20,000 learned particles, dimension 4 | Same |
| GAN loss / critic penalty | Rp logistic / exact L2 cap, weight 1 | Same |
| Particle regularizer | VICReg, weight 1, unique sampled rows | Same |
| Adam learning rates: G / D / prior | 0.0006 / 0.0009 / 0.006 | Same |
| Adam betas / EMA decay | (0, 0.999) / 0.995 | Same |
| Schedule | Hold 60%, cosine to 5% | Same |
| Batch size / training updates | 256 / 7,000 | 256 / 56,000 |
| Conditioning | Unconditional | Class-only UCD, 4 classes, CE weight 0.02 |

These defaults come from the selected 100-Gaussians and denoising experiments.
Networks remain application choices: the reference toy benchmarks use MLPs and
two Fourier frequencies in D. Changing the architecture or dataset changes the
experiment; the recipe alone does not establish convergence on a new problem.

[The executable PyTorch loop](https://github.com/255BITS/ParticleGAN/blob/master/examples/pytorch_loop.py) shows optimizer setup,
D freezing/restoration, unique-particle regularization, the learning-rate
schedule, and EMA for G and the prior. It uses small MLPs and synthetic data,
requires no research dependencies, and writes one flushed JSON record per log
line:

```bash
mkdir -p runs/api
python -u examples/pytorch_loop.py --steps 5 --batch-size 16 > runs/api/smoke.log 2>&1
# In another terminal while a longer run is active:
tail -f runs/api/smoke.log
```

### TOML is just constructor arguments

Load TOML with your preferred parser and unpack a section into a constructor.
The library does not require a parser or configuration framework:

```toml
[particlegan]
# Omit name for GAN defaults; use name = "ddgan" for DDGAN + UCD.
z_dim = 16
num_particles = 4096
lr = 0.0003

# Alternatively, configure independent primitives:
[prior]
z_dim = 16
num_particles = 4096

[loss]
loss_type = "logistic"
mode = "rp"
```

```python
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10: pip install tomli

with open("model.toml", "rb") as f:
    config = tomllib.load(f)

recipe = get_recipe(**config["particlegan"])
# Or construct components directly:
prior = ParticlePrior(**config["prior"])
adversarial = GANLoss(**config["loss"])
# Explicit code overrides are ordinary dictionary merges:
recipe = get_recipe(**{**config["particlegan"], "lr": 1e-4})
```

`toml.load(...)` dictionaries work too. Choose either recipe-owned values or
per-component sections for your application. Try the supplied configuration:

```bash
python -u examples/pytorch_loop.py --config examples/api.toml --steps 5
```

The primary research trainers also accept TOML or YAML using their existing flat
experiment schema (separate from the constructor sections above):

```bash
python experiments/train_100gaussians.py --config configs/100gaussians/default.toml
python experiments/train_denoising.py --config configs/denoising/default.toml
```

Running `python experiments/train_denoising.py` with no arguments loads
`configs/denoising/default.toml`. Install the `experiments` extra for these
trainers; the denoising trainer requires CUDA. See the
[experiment runner guide](https://github.com/255BITS/ParticleGAN/blob/master/docs/experiment-runner.md) for grids and recorded
effective configurations.

To use MoG latents with the DDGAN trainer:

```bash
mkdir -p results/denoising/mog
python -u experiments/train_denoising.py --config configs/denoising/mog.toml > results/denoising/mog/log.txt 2>&1
# From another terminal:
tail -F results/denoising/mog/log.txt
```

Set `prior = "mog"`; `sigma_rel` and `standardize` control its fixed noise and
read standardization. Optional `prior_betas` sets separate Adam betas for the
component means. The supplied config uses the one-shot MoG recipe's 400 components
and prior optimizer settings. The matched small-generator studies at
[14k updates](reports/denoising-toy/mog_capacity/READOUT.md) and
[100k updates](reports/denoising-toy/mog_capacity_100k/READOUT.md) report the
quality tradeoffs; the supplied full-width 56k configuration is not a selected
DDGAN benchmark winner. Set `generator_hidden` to vary generator width while
keeping discriminator width controlled by `hidden`.
The trainer regularizes raw means and preserves the calibrated noise in EMA and
checkpoints. Forward diffusion and the separately configured reverse `noise`
source retain their existing behavior.

### DDGAN and UCD compose independently

`DDGAN` supplies Gaussian forward pairs and reverse transitions. `UCD` selects
class scores from a logit network; it does not inject class labels into that
network. Neither owns your training loop. Here is the D-loss portion of a
conditional denoising pipeline, with caller-defined `G`, `logit_network`, data,
labels, and device:

```python
import torch
from particlegan import DDGAN, UCD, ucd_loss

recipe = get_recipe("ddgan", num_classes=4)
prior = recipe.make_prior().to(device)
adversarial = recipe.make_loss()
penalty = recipe.make_gradient_penalty()
process = DDGAN(alpha_bar=recipe.alpha_bar).to(device)
critic = UCD(logit_network, num_classes=recipe.num_classes).to(device)

t = torch.randint(1, process.steps + 1, (len(real),), device=device)
rng = torch.Generator(device=device).manual_seed(123)
x_prev, xt = process.forward_pair(real, t, rng)
z, indices = prior.sample(len(real), generator=rng)
x0_hat = G(z, labels, xt=xt, t=t)  # G predicts clean data
fake_prev = process.reverse(x0_hat, xt, t, torch.randn_like(xt))
real_score, real_logits = critic(x_prev, labels, xt=xt, t=t)
fake_score, fake_logits = critic(fake_prev.detach(), labels, xt=xt, t=t)
d_loss = adversarial.d_loss(real_score, fake_score)
d_loss += ucd_loss(real_logits, fake_logits, critic.ucd_labels(labels, t),
                   weight=recipe.ucd_weight)
d_loss += penalty(lambda x: critic(x, labels, xt=xt, t=t)[0],
                  x_prev, fake_prev.detach())
```

For class-only UCD, the network receives `network(x, xt=xt, t=t)` and returns
`[B, C]` logits. Labels and times are `[B]` long tensors; times run from 1 to T.
For joint time/class heads, use `UCD(network, num_classes=C,
target="time_class", num_steps=T)`; the network receives `network(x, xt=xt)`
and returns `[B, T*C]` logits. UCD also works without diffusion as
`critic(x, labels)` over a network that accepts only `x`.

Recompute critic scores after its update, freeze critic parameters, and keep
`fake_prev` attached for the G/prior adversarial loss. Class CE belongs to D.
The default schedule is `(1, .9, .5, .05, .0001)`; corruption and reverse noise
are Gaussian and separate from learned latent particles. See
[the API reference](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#ddgan) for the full composition rules.

### Teacher/student pipelines

A teacher can produce the target batch in your existing training pipeline.
Use matching conditioning for a paired supervised loss:

```python
teacher.eval()
with torch.no_grad():
    targets = teacher(inputs)
z, indices = prior.sample(len(inputs))
fake = student(z, inputs)
# Update your critic using targets as reals and fake.detach() as fakes.
# Then freeze the updated critic's parameters for this student/prior loss:
student_loss = supervised_weight * supervised_loss(fake, targets)
student_loss += adversarial_weight * adversarial.g_loss(
    D(fake), D(targets).detach(),
)
student_loss += spread(prior.z[indices.unique()])
# Your student/prior optimizer performs backward and step; restore D afterward.
```

The teacher, supervised objective, loss weights, and update order belong to your
application. You can also use only the prior or regularizers without an
adversarial objective.

### Inference without a critic or optimizer

Save G and the learned prior, ideally their EMA states. Keep the architecture
configuration needed to reconstruct G alongside the checkpoint:

```python
torch.save({"generator": G.state_dict(), "prior": prior.state_dict()}, "model.pt")

# In another application, reconstruct your generator architecture and prior:
G = build_generator(z_dim=16).to(device)
prior = ParticlePrior(num_particles=4096, z_dim=16).to(device)
state = torch.load("model.pt", map_location=device, weights_only=True)
G.load_state_dict(state["generator"])
prior.load_state_dict(state["prior"])
G.eval()
prior.eval()
with torch.inference_mode():
    z, _ = prior.sample(64)
    samples = G(z)
```

Use the same particle count and latent dimension as training. Conditional G
also receives labels or inputs. DDGAN inference additionally reconstructs its
schedule (or loads its `state_dict`) and starts from Gaussian `x_T`; loop over
`T, ..., 1`, drawing fresh latent samples and reverse noise at each step:
`x = process.reverse(G(z, labels, xt=x, t=t), x, t, noise)`. There is no critic
or optimizer in inference.


## The Problem

GANs can suffer from **mode collapse**: the generator produces only a subset of the data distribution. This project explores whether optimizing a finite latent particle cloud alongside the generator improves coverage on small, highly multimodal benchmarks.

## The Insight

**What if the prior could move too?**

We introduce learnable "particles" in latent space. Both the generator and these latent vectors are optimized during training. The experiments examine how that extra flexibility interacts with discriminator regularization, optimizer dynamics, and sample quality. The results are empirical observations on these benchmarks, not a guarantee against collapse.

### Historical Gaussian example

![100 Gaussians without Particle Prior](https://raw.githubusercontent.com/255BITS/ParticleGAN/master/100gaussians_no_particles.gif)

*Historical visualization from the older Gaussian example. Its architecture and training recipe differ from the particle example above, so these GIFs are not a matched prior comparison.*

## Evidence and controls

The historical [regularizer study](https://github.com/255BITS/ParticleGAN/blob/master/FINDINGS.md) compares discriminator penalties within the particle model. It does not establish that a fixed Gaussian prior necessarily collapses. The current examples share one training loop and matched defaults; the only training change for the Gaussian controls is removing the learned prior and its regularizer.

For a reproducible three-way comparison, run:

```bash
python experiments/compare_priors.py --study-dir runs/prior_comparison --run --device cuda:0
```

This runs learned particles, a frozen Gaussian table, and fresh Gaussian noise on paired seeds 23001–23003. It records configs, source revision, final samples, coverage, transport distances, and per-mode radial and covariance shape diagnostics. See [prior controls and interpretation](https://github.com/255BITS/ParticleGAN/blob/master/docs/prior-controls.md) and [reproducing the project](https://github.com/255BITS/ParticleGAN/blob/master/docs/reproducing.md).

The completed [nine-run matched comparison](https://github.com/255BITS/ParticleGAN/blob/master/reports/prior-comparison/README.md) reached 100/100 high-quality modes on every learned-prior seed, with a mean high-quality fraction of 98.6%, versus 8.1% for the frozen table and 6.4% for fresh Gaussian noise. This establishes a concentration advantage under this recipe. The report also shows remaining tail and covariance distortion, finite output support, and transport-metric tradeoffs; it does not establish complete Gaussian calibration or a general guarantee against collapse.

## How It Works

1. **Particle Prior**: Instead of sampling z ~ N(0, I), we maintain a set of learnable latent vectors (particles). During training, we sample from this discrete set.

2. **Joint Optimization**: Particles are optimized alongside G and D. Their positions can adapt to the data modes.

3. **VICReg Regularization**: We apply variance-covariance regularization to prevent particles from collapsing to a single point, while allowing arbitrary topology (clusters, gaps, etc.).

## Examples

### Five Modes (Text Generation)

A minimal example demonstrating the core idea. Five words ("apple", "grape", "lemon", "melon", "berry") are encoded into a 2D latent space. Each word gets one particle.

![Five Modes Training](https://raw.githubusercontent.com/255BITS/ParticleGAN/master/five_modes.gif)

```bash
python examples/five_modes.py
```

The visualization shows:
- **Left**: Loss curves for D and G/E/Prior
- **Center**: 2D latent space with particle positions (white stars) and encoded words (colored dots)
- **Right**: Reconstruction quality over training

### 100 Gaussians (2D Distribution)

The main benchmark. 100 Gaussian modes arranged on a 10×10 grid. This is a stress test for mode coverage.

```bash
python examples/100gaussians.py
```

The historical particle study reports runs with 100/100 modes and approximately 99% of samples within 3σ of a center after 7k steps. Coverage alone does not establish that the within-mode distribution is correct; the trainer also records shape and transport metrics.

The default recipe is RpGAN (relativistic, logistic) + a one-sided cap gradient penalty on D (`relu(‖∇ₓD‖ − 1)²` on reals and fakes, coeff 1.0), Fourier-feature D, EMA evaluation, Adam β1=0, base LR 6e-4 with a delayed cosine anneal. The cap won a 420-run bake-off against the zero-centered R1/R2 penalty, which is still available with `--reg_arm a_r1r2 --reg_coeff 0.02`. See [FINDINGS.md](https://github.com/255BITS/ParticleGAN/blob/master/FINDINGS.md) for the study and [docs/convergence-tips.md](https://github.com/255BITS/ParticleGAN/blob/master/docs/convergence-tips.md) for the transferable reasoning behind each ingredient.

**Without particle prior** (baseline):
```bash
python examples/100gaussians_no_particle_prior.py
```

This entrypoint uses the same architecture, losses, learning rates, schedule, and EMA as the particle example, with fresh Gaussian noise. Use `--prior frozen_gaussian` for a finite frozen-table control. The outcome depends on the recipe and seed; the baseline does not assume collapse.

## Project Structure

```
ParticleGAN/
├── particlegan/            # Installable PyTorch primitives and recipe helpers
│   ├── particle_prior.py   # Learnable particle cloud (nn.Module)
│   ├── gan_loss.py         # Flexible GAN losses (hinge, logistic, Wasserstein, LSGAN)
│   ├── grad_regularizers.py # D gradient penalties (cap, R1/R2, eikonal, ...)
│   ├── vicreg_loss.py      # Variance-covariance regularization
│   ├── diffusion.py        # DDGAN forward/reverse transitions
│   ├── conditioning.py     # UCD scores and class supervision
│   └── recipes.py          # Inspectable defaults and optional factories
├── lib/                    # Repository compatibility imports and research helpers
├── examples/
│   ├── pytorch_loop.py                  # Minimal caller-owned loop (Torch only)
│   ├── api.toml                         # Constructor/recipe configuration
│   ├── five_modes.py                    # Text generation toy problem
│   ├── 100gaussians.py                  # 100-mode benchmark (with particles)
│   └── 100gaussians_no_particle_prior.py # Baseline (without particles)
└── README.md
```

The grid-search infrastructure behind the study — config generation, the per-arm trainer, grid runner, and the analysis/leaderboard scripts — lives in `experiments/`, with the generated per-run configs in `configs/`.

The [CIFAR DDGAN experiment](https://github.com/255BITS/ParticleGAN/blob/master/reports/cifar-ddgan/README.md) scales the particle
recipe to images. Its [speed study](https://github.com/255BITS/ParticleGAN/blob/master/reports/cifar-ddgan/speed/READOUT.md) compares
exact/lazy/finite-difference bcap and backports the shared implementation to both
toy trainers. The faster CIFAR default retains exact derivatives; FD is optional.

## Notes

- The text experiments (`five_modes.py`) use the same recipe (RpGAN + one-sided cap penalty on the joint critic ∇₍ₓ,𝓏₎D, EMA, β1=0, cosine anneal)
- The 100-Gaussian experiments use the one-sided cap penalty (`--reg_arm`, default `b_cap`); a gradient penalty is what lets the sharp Fourier discriminator keep full mode coverage
- Particles use a higher learning rate (10×) than G/D for faster adaptation

## Changelog

Versions before 0.2 tracked the default recipe of `examples/100gaussians.py`.

### 0.5.0 — 2026-09-17

- Add particle AE-GAN, constant-KL VAE-GAN and AE-DDGAN recipes and public
  encoding/reconstruction helpers with caller-owned training loops.
- Add optional encoder optimizer integration, explicit ELBO reporting and
  categorical inference, numerical tests and installed-wheel coverage.
- Preserve queued toy/CIFAR studies and publish their leaderboards and limits.
  See the [full changelog](https://github.com/255BITS/ParticleGAN/blob/master/CHANGELOG.md).

### 0.4.0 — 2026-09-17

- Adds `get_recipe("ddgan_mog")`: four-step DDGAN with class-only UCD, 400 MoG
  components, z_dim 4, sigma_rel 0.025, standardized reads, 100k updates,
  constant LR, prior LR 0.06 and prior Adam betas (0.5, 0.999).
- Adds MoG support to `train_denoising` and checkpoint probes, raw-mean
  regularization, separate prior optimizer settings, and independent generator
  width through `generator_hidden`. Existing recipe defaults remain unchanged.
- Includes the matched 14k/100k capacity studies and frozen-noise interventions.
  At 100k, DDGAN+MoG reaches 79.57% joint HQ and all 100 modes; one-shot models
  retain higher HQ but cover 77 modes. These single-seed results use a small
  generator and do not isolate representational capacity. See the
  [100k readout](reports/denoising-toy/mog_capacity_100k/READOUT.md).

### 0.3.0 — 2026-09-17

- Adds the public `MoGParticlePrior`: a uniform mixture with learned means and a
  shared, fixed Gaussian sigma calibrated from initial nearest-neighbor spacing
  (defaults: 400 components, z_dim 4, sigma_rel 1/40, standardized reads).
- Adds `get_recipe("mog")`, the selected 400-component, 28k-step recipe (prior LR
  0.06, prior Adam betas (0.5, 0.999), existing GAN defaults), plus
  `configs/mog/default.toml` for the 100-Gaussian trainer. Recipe factories can
  select the prior and set prior betas separately.
- MoG supports explicit sampling generators, fixed epsilon snapshots, noisy module
  forward calls for DDP, raw-center regularization, EMA, and state-dict restoration
  of read configuration and calibrated noise. Legacy experimental checkpoints remain
  loadable.
- Core dependency stays PyTorch only; the optional `mog` extra installs SciPy for
  faster calibration of large tables, with an exact Torch fallback without it.
- Passed the frozen C0 envelope on 100 Gaussians at 28k steps (HQ/real 0.99953,
  width/real 0.92636, KL 0.02888) with 50× fewer components and 4× the steps of
  the original baseline. Single-seed result; see
  [results/mog/COMPONENT_SCALE.md](results/mog/COMPONENT_SCALE.md).

### 0.2.0 — 2026-09-16

- Adds the installable `particlegan` namespace, independent PyTorch primitives,
  immutable recipes, and direct use of loaded TOML dictionaries.
- Core runtime requires only Torch; research dependencies use the `experiments` extra.
- Repository trainers use the shared package while retaining their own loops.

### 0.1.2 — 2026-08-22

- Default gradient penalty switched to the one-sided cap (`b_cap`, `relu(‖∇ₓD‖ − 1)²`, coeff 1.0) via `lib/grad_regularizers.py`; base LR 3e-4 → 6e-4; run length 5k → 7k steps.
- Chosen by a 420-run controlled study ([FINDINGS.md](https://github.com/255BITS/ParticleGAN/blob/master/FINDINGS.md)): same game-damping as R1/R2, sharper modes (hq 0.986 vs a ~0.94 ceiling), honest per-mode core width (0.87), zero collapses. R1/R2 stays available via `--reg_arm a_r1r2`.
- Adds the `experiments/` study infrastructure and the deterministic video renderer.

### 0.1.1

- R3GAN-style defaults (previously undocumented; commit `b5529cb`): RpGAN logistic objective + zero-centered R1+R2 (γ=0.02) + Fourier-2 features on D + EMA(0.995) on G and prior + Adam β1=0 + delayed cosine LR anneal + z_dim 4.
- Full coverage with ≥90% hq in ~3.5k steps.

### 0.1.0

- Original example: vanilla/hinge GAN, no gradient regularizer, plain MLP D, z_dim 2, Adam β1=0.5, no EMA, no LR anneal.
- Never converged on the 100-Gaussians benchmark: ~86–92/100 modes, ~30% hq at 12k steps (baseline row in [docs/convergence-tips.md](https://github.com/255BITS/ParticleGAN/blob/master/docs/convergence-tips.md)).

## Citation

```bibtex
@software{particlegan2025,
  author = {Martyn Garcia},
  title = {ParticleGAN: Learnable Priors for Stable GANs},
  year = {2025},
  url = {https://github.com/255BITS/ParticleGAN}
}
```

## License

MIT
