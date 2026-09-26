# ParticleGAN

**GANs with a learnable particle prior, for PyTorch.**

[![Tests](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml/badge.svg)](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml)

A GAN usually draws its latent code from a fixed Gaussian and leaves all of
the work of covering the data to the generator; when it cannot, modes go
missing. ParticleGAN replaces that noise with a table of learnable latent
vectors (*particles*) that are optimized together with the generator, so the
prior itself can move toward the data's modes. The package ships one training
configuration: a relativistic-pairing (RpGAN) logistic loss, a critic gradient
penalty that combines R1, capped gradients and an adaptive EMA-critic anchor,
and the optimizer settings and schedules that go
with them ([how it works](docs/ka2.md)). You write an ordinary PyTorch GAN
loop; the recipe builds the pieces.

This branch prepares **KA2 as the single default** for the API, trainer and
examples. It is an unreleased candidate: later recovery dropouts and an
unequal-mass toy failure prevent a qualified-winner claim.
[Measured results and remaining qualification](reports/ka2-default-candidate/README.md).

![100 Gaussians: default GAN recipe converging with live weights](100gaussians.gif)

The animation records the **released 0.8.0 K3P default**, live weights, seed 1234:
**100/100 modes, 98.9% within 3σ after 7,000 updates**, with all 100 modes first covered
at update 1,430. It is historical evidence, not a KA2 measurement.
[Reproduce this animation](reports/readme-100gaussians/README.md#readme-hero-gif).

## Install

Requires Python 3.10+ and PyTorch.

```bash
python -m pip install particlegan           # released library (0.8.0, K3P)
```

For the examples, experiments and tests, install from source:

```bash
git clone https://github.com/255BITS/ParticleGAN.git
cd ParticleGAN
python -m pip install -e '.[experiments,dev]'  # plus examples, experiments and tests
```

## Train a GAN

Everything comes from role-named factories on a recipe; the loop is yours.

```python
import copy
import torch
from torch import nn
from particlegan import get_recipe, scale_learning_rates

def real_batch(n):  # replace with your DataLoader: 8 Gaussians on a ring
    angle = torch.randint(8, (n, 1)) * torch.pi / 4
    return torch.cat([angle.cos(), angle.sin()], 1) + 0.05 * torch.randn(n, 2)

recipe = get_recipe(total_steps=2000)
G = nn.Sequential(nn.Linear(recipe.z_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 2))
D = nn.Sequential(nn.Linear(2, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))
prior = recipe.make_prior()                      # the learnable particle table
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty, gan = recipe.make_critic_penalty(opt_d), recipe.make_loss()
base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]

for step in range(recipe.total_steps):
    scale_learning_rates(step, recipe, (opt_g, opt_d), base_lrs, prior)
    real = real_batch(recipe.batch_size)
    z, _ = prior.sample(recipe.batch_size)
    fake = G(z)

    d_loss = gan.d_loss(D(real), D(fake.detach())) + penalty(D, real, fake.detach())
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()

    g_loss = gan.g_loss(D(fake), D(real))
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()
```

`opt_g` and `opt_d` are Adam optimizers whose `step()` also does the
formulation's step-time work, and their `state_dict()` holds all of its state,
so checkpoint them as usual. [`examples/pytorch_loop.py`](examples/pytorch_loop.py)
adds the remaining pieces of the default update (critic input noise, generator
output noise, EMA weights).

Or let `GANTrainer` run exactly that default update:

```python
from particlegan import GANTrainer

trainer = GANTrainer(get_recipe(), G, D)
for _ in range(trainer.recipe.total_steps):
    trainer.step(real_batch(trainer.recipe.batch_size))
samples = trainer.sample(1024)
```

## Model families

`get_recipe(name)` selects the model; every family trains the same way.

| Name | Model |
| --- | --- |
| `gan` (default) | GAN with a learnable particle prior |
| `mog` | GAN with a mixture-of-Gaussians particle prior |
| `ddgan` | Denoising-diffusion GAN with a UCD (class-conditional) critic |
| `ddgan_mog` | `ddgan` with a mixture-of-Gaussians prior |
| `ae_gan` | Autoencoder GAN: an encoder routes data to particles |
| `vae_gan` | Variational variant of `ae_gan` |
| `ae_ddgan` | Autoencoder denoising-diffusion GAN |

Any field can be overridden: `get_recipe("mog", total_steps=20_000)`.

## Learn more

- [How the training formulation works](docs/ka2.md), including several critics and conditional critics
- [API reference](docs/api.md) and a [minimal DDGAN + UCD loop](docs/api.md#a-minimal-ddgan--ucd-loop)
- Examples: [`quickstart_gan.py`](examples/quickstart_gan.py) (GANTrainer with checkpoints),
  [`pytorch_loop.py`](examples/pytorch_loop.py) (the full update in your own loop),
  [`100gaussians.py`](examples/100gaussians.py) (the benchmark above),
  [`particle_autoencoder.py`](examples/particle_autoencoder.py) (AE/VAE-GAN),
  [`fast_lander.py`](examples/fast_lander.py) (world model and controllers for Lunar Lander)
- [Changelog](CHANGELOG.md) · [Releasing](docs/releasing.md)

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
