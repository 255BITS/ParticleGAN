# ParticleGAN

**GANs with a learnable particle prior, for PyTorch.**

[![Tests](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml/badge.svg)](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml)

A GAN usually draws its latent code from a fixed Gaussian and leaves all of
the work of covering the data to the generator; when it cannot, modes go
missing. ParticleGAN replaces that noise with a table of learnable latent
vectors (*particles*) that are optimized together with the generator, so the
prior itself can move toward the data's modes. The package ships one training
configuration: a relativistic-pairing (RpGAN) logistic loss, a critic gradient
penalty that hands over from R1 to capped gradients plus an EMA-critic anchor
as the learning rate anneals, and the optimizer settings and schedules that go
with them ([how it works](docs/k3p.md)). You write an ordinary PyTorch GAN
loop; the recipe builds the pieces.

![100 Gaussians: default GAN recipe converging with live weights](100gaussians.gif)

Recorded with the 0.8.0 recipe and its original random initialization, live weights, seed 1234:
**100/100 modes, 98.9% within 3σ after 7,000 updates**, with all 100 modes first covered
at update 1,430. [Reproduce this animation](reports/readme-100gaussians/README.md#readme-hero-gif).

## Install

Requires Python 3.10+, PyTorch, and NumPy (installed as dependencies).

```bash
python -m pip install particlegan           # the library (0.8.0)
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
from particlegan import get_recipe, init

def real_batch(n):  # replace with your DataLoader: 8 Gaussians on a ring
    angle = torch.randint(8, (n, 1)) * torch.pi / 4
    return torch.cat([angle.cos(), angle.sin()], 1) + 0.05 * torch.randn(n, 2)

recipe = get_recipe(total_steps=2000)
G = nn.Sequential(nn.Linear(recipe.z_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 2))
D = nn.Sequential(nn.Linear(2, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))
init.deterministic_orthogonal_(G, seed=0)        # optional: repeatable weights
init.deterministic_orthogonal_(D, seed=1)
prior = init.deterministic_orthogonal_(recipe.make_prior())  # the learnable particle table
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty, gan = recipe.make_critic_penalty(opt_d), recipe.make_loss()

for step in range(recipe.total_steps):
    real = real_batch(recipe.batch_size)
    z, _ = prior.sample(recipe.batch_size)
    fake = G(z)

    d_loss = gan.d_loss(D(real), D(fake.detach())) + penalty(D, real, fake.detach())
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()

    g_loss = gan.g_loss(D(fake), D(real))
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()
```

`opt_g` and `opt_d` are Adam optimizers whose `step()` also applies the
recipe's LR schedule and the formulation's step-time work, and their
`state_dict()` holds all of that state, so checkpoint them as usual; the loop
never sets learning rates. [`examples/pytorch_loop.py`](examples/pytorch_loop.py)
adds the remaining pieces of the default update (critic input noise, generator
output noise, EMA weights).

Or let `GANTrainer` run exactly that default update:

```python
from particlegan import GANTrainer

recipe = get_recipe()
prior = init.deterministic_orthogonal_(recipe.make_prior())
trainer = GANTrainer(recipe, G, D, prior=prior)
for _ in range(trainer.recipe.total_steps):
    trainer.step(real_batch(trainer.recipe.batch_size))
samples = trainer.sample(1024)
```

Without `prior=`, the trainer builds a plain randomly drawn particle table.

## Repeatable initialization

The recipe never touches your weights: networks train from whatever they are
built with. `particlegan.init` is optional tooling in the spirit of
`torch.nn.init`, and the examples use it:

```python
from particlegan import init

init.deterministic_orthogonal_(G, seed=0)   # G, D and an encoder get different seeds
init.deterministic_orthogonal_(D, seed=1)
prior = init.deterministic_orthogonal_(recipe.make_prior())
```

It writes orthogonal matrices at PyTorch's default scale, patterned biases and
an evenly spread particle table, derived from `seed` alone: the same seed and
architecture give the same weights, and no random state is used. Call it on
fresh networks, before loading weights or building optimizers. A custom layer
it does not know raises an error until you declare it with `init.register`.
Values depend on each parameter's position in the module you pass, so
initialize whole networks: a submodule initialized on its own gets different
values. Projects with custom layers can guard this with a one-line test that
`init.declarations(net)` has no `None` entries
([example](docs/api.md#declarationsmodule)).
See the [API reference](docs/api.md#initialization) and the
[math and architecture guide](docs/initialization.md).

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

- [How the training formulation works](docs/k3p.md), including several critics and conditional critics
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
