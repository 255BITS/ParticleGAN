# ParticleGAN

**Learnable particle priors and GAN building blocks for PyTorch.**

**[K3P is the selected GPU research formulation](reports/toy100/k3p-base/README.md):**
**22/22 declared toy gates PASS**, with a **1,200-update ring hold and all 300
extension checks passing**. Target-shift recovery remains a measured failure
(28/81 deadline checks). [Leaderboard](reports/toy100/continuous-practical-leaderboard.md) ·
[Exact selected bundle](reports/toy100/current-research-base.json).

**K3P is the public API default** (`get_recipe()`, `GANTrainer`): the penalty,
EMA-critic anchor, critic spike guard, latent damping, LR schedule and noise are
package components with explicit hyperparameters.
[How K3P works and how to wire it into your own loop](docs/k3p.md).
The frozen research drivers remain for exact reproduction
([execution guide](reports/toy100/k3p-base/README.md#exact-selected-bundle-and-execution)).
The standard toy CLI and CPU CI gate still exercise the historical GAN v3 recipe.

**GAN v3 (previous default, history):** one shared recipe passed **19/19 live behavioral toys**
with declared discriminator choices (15/19 with the reference D profile).

| Recipe version | Live toys passed | Status |
| --- | ---: | --- |
| v1 (archived) | 5/19 | Original preset |
| v2 (archived) | 8/19 | Previous default |
| v3 (archived) | 19/19 | Previous default, with documented D choices; superseded by K3P |

[Illustrated guide and equations](docs/gan-v3.md) ·
[Full leaderboard](reports/transfer_suite/unadjusted/README.md) ·
[Installed-default verification](reports/transfer_suite/single_default_verification/README.md) ·
[Reproduce or compare a candidate](benchmarks/transfer_suite/UNADJUSTED_SEARCH.md).
EMA is separate; a PASS requires every metric at five consecutive final checks.

[API reference](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md) · [Minimal GAN loop](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#a-minimal-training-loop) ·
[Minimal DDGAN + UCD loop](https://github.com/255BITS/ParticleGAN/blob/master/docs/api.md#a-minimal-ddgan--ucd-loop)

[![Tests](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml/badge.svg)](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml)

![100 Gaussians: default GAN recipe converging with live weights](100gaussians.gif)

The package default, `GANTrainer(get_recipe("gan"), G, D)` with no overrides, live weights, seed 1234:
**100/100 modes, 98.9% within 3σ after 7,000 updates**, with all 100 modes first covered
at update 1,430. [Reproduce this animation](reports/readme-100gaussians/README.md#readme-hero-gif).

The [100-mode convergence gate](docs/toy100.md) tests square, rotated, and
staggered Gaussian grids together, with step-zero snapshots, live/EMA traces,
and explicit coverage, balance, and spread checks. The selected toy config
reaches **100/100 modes on all three**, sustaining the full live-weight gate
from updates 6,000–7,000. Train and gate them together with:

```bash
python -u -m benchmarks.toy100 run --output artifacts/toy100/gate
```

The command exits nonzero when any problem misses the numerical gate. Add
`--problem grid100` for an individual run. [Before/after GIFs, search results,
and the preserved default failures](reports/toy100/README.md) document the
explicit noise settings and staggered batch override.

## Installation

Requires Python 3.10+ and PyTorch. Install from PyPI:

```bash
python -m pip install particlegan==0.7.0
```

Version 0.7.0 includes GAN v3, named model-family recipes, the explicit
`GANTrainer`, and the 100-mode toy gate.

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

Train and fly a **fast Lunar lander** from scratch with one command:

```bash
uv run --extra lunar python -u examples/fast_lander.py
```

The command gathers expert flights, trains a world model and RpGAN controllers,
extracts successful slow/fast trajectories, selects a faster policy on validation
worlds, and exports real simulator GIFs plus a local demo page. It includes a
named Lunar variant with downward thrust. See the [fast-lander guide](docs/fast-lander.md)
for the measured results, artifacts, and success/speed gate.

For an unconditional GAN, supply your networks and real batches; the optional
trainer applies the recipe's optimizer settings, particle regularization,
learning-rate decay and EMA. `Recipe` holds hyperparameters and small component
factories; construct `GANTrainer` explicitly when you want it to own updates:

```python
from particlegan import GANTrainer, get_recipe

recipe = get_recipe(total_steps=7000)
trainer = GANTrainer(recipe, G, D)  # Move your networks to their device first.
for real in batches:               # Supply up to recipe.total_steps batches.
    stats = trainer.step(real)
samples = trainer.sample(256)       # Live weights; ema=True reports EMA separately.
```

Run `python -u examples/quickstart_gan.py --steps 1000` for a complete PyTorch-only
example with flushed logs and resumable checkpoints. See
[training and checkpoint contracts](docs/api.md#gantrainer).
The optional helper supports scalar, unconditional GANs with particle priors;
the independent primitives remain available for other training loops.

`get_recipe()` supplies the **K3P** defaults: Rp logistic, the K3P penalty
(coefficient 1, κ 1, EMA-critic anchor), critic spike guard, A2 latent damping,
Adam (0,.999), G/D LR .00425 with a 1,600-update horizon down to 1%, particle
LR .0085 over the full budget, annealed critic input noise and generator output
noise, batch 2048 and z_dim 2. With a single critic, the trainer's penalty
blend is

```text
penalty = ½ (s·A + (1 − s)·(B + P)),  s = max(0, min(1, 2r) − 2f) / (1 − 2f)
```

where A is RMS R1 plus a fake-gradient cap, B the one-sided caps, P the gap to
the EMA critic's input gradient, and r the critic's LR relative to its peak.
[Formula, defaults and multi-critic API](docs/k3p.md). The optional
`BatchDistanceDiscriminator` exposes local within-batch spread to D.
GAN v3 is documented as history in [the GAN v3 guide](docs/gan-v3.md).

Use `get_recipe("gan")`, `get_recipe("ae_gan")`, `get_recipe("vae_gan")`, or
another [model-family recipe](docs/api.md#recipes-and-defaults), with explicit
keyword overrides. Families select components and inherit the current shared
training hyperparameters. Historical optimizer versions are not selectable;
recipes never construct training loops. Restore an old run with its
complete saved `Recipe(**resolved_fields)` or the corresponding Git revision.
The [historical adjusted comparison](reports/transfer_suite/default_comparison/README.md)
used different optimizer settings per toy and remains separate. Earlier
[controller research](reports/learned_lr/README.md), [transfer studies](reports/transfer_suite/README.md)
and [individual solvability witnesses](reports/transfer_suite/solvability/README.md)
remain documented. The v3 result does not establish unseen-network or
100-Gaussian convergence improvements.

The [paired 2D transport extraction](reports/paired_error_2d/README.md) tests the
MSE-free paired-error game with movable/fixed clouds. All 12 matching application
runs reproduce exactly using this checkout's public primitives. The cap/schedule
helps affine fidelity and slightly worsens swirl fidelity at the fixed seed;
its behavior depends on the task and on live versus EMA evaluation.

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
import copy
from particlegan import GANLoss, ParticlePrior, get_recipe

recipe = get_recipe()
prior = recipe.make_prior().to(device)  # 20,000 particles, z_dim=2
adversarial = recipe.make_loss()       # relativistic-paired logistic
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))  # Adam
penalty = recipe.make_critic_penalty(opt_d)  # K3P today; reads its state from opt_d
spread = recipe.make_prior_regularizer()  # weight 0 in K3P (prior_reg)

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

### Mixture-of-Gaussians particles

`MoGParticlePrior` gives each learned particle a continuous Gaussian neighborhood.
Pass an explicit finite, nonnegative `sigma`; it is shared across all components
and fixed during training. Construction performs **no nearest-neighbor search**.
Historical recipes retain their spacing-based calibration as an explicit step.

```python
from particlegan import MoGParticlePrior, get_recipe

prior = MoGParticlePrior(num_particles=65536, z_dim=128, sigma=0.212616428732872).to(device)
z, component_ids = prior.sample(256)

recipe = get_recipe(prior_kind="mog", sigma_rel=.025, num_particles=400)
prior = recipe.make_prior().to(device)  # Explicit spacing calibration.
# Or recipe.make_prior(sigma=0.1) to bypass calibration.
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
# Regularize raw means, not noisy draws. Use the full table at N <= 1024.
prior_loss = recipe.make_prior_regularizer()(prior.z)
```

Use `ParticlePrior` for atoms. Prior type changes sampling, not the optimizer
recipe: the shared defaults still use G/D LR .00425 and particle LR .0085.
The historical MoG study below has explicit experiment settings in its config.

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

`calibrate_mog_sigma(centers, sigma_rel)` optionally returns `(sigma, d0)` from
supplied centers; see the [migration and checkpoint guide](docs/api.md#mogparticleprior).
Calibration can be very expensive for large, high-dimensional tables. It works
with PyTorch alone; for faster calibration of low-dimensional tables, install
`python -m pip install 'particlegan[mog]'` (or `python -m pip install -e '.[mog]'`
from this checkout).

### Particle AE-GAN, VAE-GAN and AE-DDGAN (0.5.0)

```python
from particlegan import get_recipe

recipe = get_recipe(prior_kind="mog", sigma_rel=.025, encoder_mode="hard")
prior = recipe.make_prior()
# query = E(x): [batch, recipe.z_dim], produced by your encoder
# encoded = recipe.encode(query, prior)
# x_hat = G(encoded.codes)  # [batch, draws, observed dimensions]
# loss = encoded.reconstruction_loss(x_hat, x)
```

`encoder_mode="hard"` selects one particle and adds fixed-sigma Gaussian noise. Its joint
KL is constant, so it needs no KL regularizer in training. AE uses a deterministic
bounded offset instead. You own the networks, loop and loss composition;
reconstruction never silently adds KL. See the
[guide and runnable example](https://github.com/255BITS/ParticleGAN/blob/master/docs/particle-autoencoders.md)
for explicit ELBO reporting, optional categorical inference, AE-DDGAN integration
and measured limits. Genuine VAE results are toy-only; AE-GAN and AE-DDGAN have
matched CIFAR32 evidence.

### DDGAN with MoG particles (0.4.0)

Choose the model and prior explicitly:

```python
from particlegan import DDGAN, get_recipe

recipe = get_recipe(model="ddgan", prior_kind="mog", sigma_rel=.025, conditioning="ucd", num_classes=4)
prior = recipe.make_prior().to(device)
process = DDGAN(recipe.alpha_bar).to(device)
opt_g, opt_d = recipe.make_optimizers(G, D, prior)

z, indices = prior.sample(batch_size)
prior_loss = recipe.make_prior_regularizer()(prior.z[indices.unique()])
```

The model and prior share the winning optimizer/loss defaults. Set resource
sizes, class count and budget explicitly for your application. The historical
[100k study](reports/denoising-toy/mog_capacity_100k/READOUT.md) used different
explicit settings and remains a recorded experiment, not a public preset.

### Add a loss to an existing pipeline

Components are independent. For example, add a critic penalty or a particle
spread term to losses your pipeline already computes:

```python
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty = recipe.make_critic_penalty(opt_d)  # one per critic optimizer
# In your discriminator update:
d_loss = existing_d_loss + penalty(D, real, fake.detach())
opt_d.zero_grad(); d_loss.backward(); opt_d.step()

# In your generator/prior update, when sampled particle indices are available:
g_loss = existing_g_loss + spread(prior.z[indices.unique()])
opt_g.zero_grad(); g_loss.backward(); opt_g.step()

# Checkpoint: the optimizers' state_dicts carry the EMA critic and all counters.
torch.save({"G": G.state_dict(), "D": D.state_dict(), "prior": prior.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}, path)
```

The recipe chooses the concrete formulation (K3P today: gradient penalty,
EMA-critic anchor, spike guard, A2 latent damping). Its step-time work runs
inside the optimizers' ordinary `step()`; the penalty is just a loss term.
Future formulations slot in without changing your loop. A second critic gets
its own `recipe.make_critic_optimizer(D2, ema_critic=copy.deepcopy(D2))` and
`recipe.make_critic_penalty(opt_d2)`.

To use the adversarial objective itself, call
`adversarial.d_loss(D(real), D(fake.detach()))` for D and
`adversarial.g_loss(D(fake), D(real).detach())` for G. Freeze D's parameters
for the G update while retaining gradients through `D(fake)`.

### Selected defaults, with easy overrides

```python
from particlegan import get_recipe, scale_learning_rates

recipe = get_recipe()  # Recommended GAN defaults.
recipe = recipe.replace(z_dim=16, num_particles=4096, lr=3e-4)
prior = recipe.make_prior().to(device)
adversarial = recipe.make_loss()
spread = recipe.make_prior_regularizer()
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))  # after .to(device)
base_lrs = [[g["lr"] for g in opt.param_groups] for opt in (opt_g, opt_d)]
penalty = recipe.make_critic_penalty(opt_d)  # one per critic optimizer
# Each update: scale_learning_rates(step - 1, recipe, (opt_g, opt_d), base_lrs, prior)
print(recipe.to_dict())  # inspect every resolved value
```

The optional optimizer helper returns Adam optimizers (subclasses whose
`step()` also runs the recipe's spike guard, EMA-critic update and A2 latent
damping; `state_dict()`, param groups and LR schedulers work as usual). The G
optimizer has separate generator and particle groups. You can build your own optimizers
using the recipe's fields instead. Recipes are immutable; `.replace(...)`
returns a new one. Unknown options raise errors.

| Shared default | Value |
| --- | --- |
| Model / conditioning | GAN / scalar |
| Prior | 20,000 learned particles, dimension 2, A2 row damping |
| Loss / penalty | Rp logistic / K3P (c 1, κ 1, EMA anchor .999) + critic spike guard |
| Particle regularizer | None (weight 0); no particle L2 |
| Adam G / D / particle LR | .00425 / .00425 / .0085 |
| Adam betas / EMA | (0, .999) / .995 |
| Schedule | G/D: hold 60% of 1,600 updates, cosine to 1%; particles: hold 60%, cosine to 5% |
| Noise | Critic input .5 → 0 by 10%; generator output 0 → .029 by 20% |
| Batch / updates | 2048 / 7,000 |

`model="ddgan"`, `prior_kind="mog"` and encoder options change components;
they do not silently select different learning rates or regularizers.
For flat 2D vectors, `BatchDistanceDiscriminator()` provides the final witness:
centered features, D96×3, Softplus β6 and four smooth local-distance features.
The older `LinearSkipDiscriminator` remains available. The new D depends on batch
composition and input units; see [the guide](docs/gan-v3.md) before changing those.

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
# Omit model for GAN; set model = "ddgan" for diffusion.
# For UCD also set conditioning = "ucd" and num_classes.
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
import copy
import torch
from particlegan import DDGAN, UCD, get_recipe, ucd_loss

recipe = get_recipe(model="ddgan", conditioning="ucd", num_classes=4)
prior = recipe.make_prior().to(device)
adversarial = recipe.make_loss()
process = DDGAN(alpha_bar=recipe.alpha_bar).to(device)
critic = UCD(logit_network, num_classes=recipe.num_classes).to(device)
opt_d = recipe.make_critic_optimizer(critic, ema_critic=copy.deepcopy(critic))
penalty = recipe.make_critic_penalty(opt_d)  # once, before the loop

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
d_loss = d_loss + penalty(critic, x_prev, fake_prev.detach(), labels, xt=xt, t=t)
opt_d.zero_grad(set_to_none=True)
d_loss.backward()
opt_d.step()
```

The conditioning after the two batches goes to the critic and its EMA; a
tuple output uses its first element (the score).

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

The default library recipe is GAN v3: Rp logistic, one-sided cap coefficient 6
and κ1.25, particle spread .05, no particle L2, Adam (0,.99), G/D LR .00425,
particle LR .0085, and delayed cosine decay. The `100gaussians` experiment alias
retains v2 settings. See the [versioned guide](docs/gan-v3.md) and
[original cap study](FINDINGS.md).

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

- The text experiments (`five_modes.py`) use the same recipe (RpGAN + K3P penalty on the joint critic ∇₍ₓ,𝓏₎D, EMA, β1=0, cosine anneal)
- The 100-Gaussian experiments use the one-sided cap penalty (`--reg_arm`, default `b_cap`); a gradient penalty is what lets the sharp Fourier discriminator keep full mode coverage
- GAN v3 particles use 2× the G learning rate; explicit experiment configurations can override that ratio.

## Changelog

Versions before 0.2 tracked the default recipe of `examples/100gaussians.py`.

### 0.7.0 — 2026-09-24

- Add the strict 100-mode toy gate (`python -m benchmarks.toy100 run`) with the
  simpler shared 22-toy recipe as its default; refresh the README animation.

- Promote the shared 19/19 recipe to GAN v3 as the single common default.
- Offer named model-family configurations with current shared hyperparameters
  and explicit overrides. Historical optimizer versions remain benchmark data.
- Keep training control flow in the separately constructed `GANTrainer` helper.
- Add `BatchDistanceDiscriminator`, the batch-aware final toy witness, and use
  it in the GAN quickstart. D architecture remains explicit in the leaderboard.
- Add the [illustrated configuration guide](docs/gan-v3.md), equations and
  versioned toy results. Live weights decide PASS; EMA remains separate.

### 0.6.0 — 2026-09-24

- Require explicit keyword `sigma` in `MoGParticlePrior`; construction no longer
  searches nearest neighbors. The shared isotropic noise remains fixed in training.
- Add optional `calibrate_mog_sigma(centers, sigma_rel)` returning `(sigma, d0)`;
  retain exact even-count median and historical dtype rounding. Recipes explicitly
  calibrate their initialized centers unless `make_prior(sigma=...)` overrides them.
- Preserve legacy checkpoint centers, sigma, d0, read settings, samples and RNG
  behavior. Load with matching dimensions and `sigma=0`, then `load_state_dict`.
- Migrate fixed-noise integrations to `MoGParticlePrior(..., sigma=fixed_sigma)`
  and remove post-construction sigma overwrites. Replace `calibrate()` with the
  standalone helper only when spacing-based calibration is intended.

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
