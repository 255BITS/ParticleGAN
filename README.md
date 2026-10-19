# ParticleGAN

![ParticleGAN — conceptual artwork of luminous particles flowing into distinct clusters](docs/media/readme/particlegan-hero.png)

[Research attempt catalog](reports/forge/research-catalog/README.md): source-bound results, reusable lessons, failures and publication links.

**GANs with a learnable particle prior, for PyTorch.**

**Running experiments / proposing ideas:** start with
[ParticleGAN Forge — read before running ideas](EXPERIMENTATION.md).

**Review the research:** [experiments by tier, goals and training GIFs](reports/forge/EXPERIMENTS_BY_TIER.md)
and the [current solution leaderboard](reports/forge/technique-inventory.md).
The [experiment memory](reports/forge/EXPERIMENT_MEMORY.md) explains the hypotheses,
comparisons and recommendations behind the recorded results. These views are
regenerated as experiments and candidate solutions change; release selection
requires completed Forge qualification and review of the winner.

The [0.10 Forge requirements](reports/forge/convergence-policy-010/README.md)
make accurate acquisition, settling, endurance and recovery required again.
The [generated technique inventory](reports/forge/technique-inventory.md) shows
the expanded requirements alongside the original source-bound results.

[![Tests](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml/badge.svg)](https://github.com/255BITS/ParticleGAN/actions/workflows/tests.yml)

A GAN usually draws its latent code from a fixed Gaussian and leaves all of
the work of covering the data to the generator; when it cannot, modes go
missing. ParticleGAN replaces that noise with a table of learnable latent
vectors (*particles*) that are optimized together with the generator, so the
prior itself can move toward the data's modes. The default training recipe is
the selected **optimized BISPF** leader: non-saturating loss, a Born-Infeld
critic penalty, an SBNP prior regularizer, zero-momentum DualNorm and constant
rates. Batched convolution updates and BF16/channels-last training accelerate
supported CUDA image networks; weights, optimizer arithmetic and sampling stay
FP32. CPU and nonconvolutional networks retain their previous execution.
[Recipe settings and evidence](docs/api.md#recipes-and-defaults) explain the
configuration. You write an ordinary PyTorch GAN loop; the recipe builds the pieces.

Future promotions change one line in [particlegan/recipes.py](particlegan/recipes.py):
`DEFAULT_RECIPE = "bispf_optimized"`. All model-family factories inherit that trainer preset.
Explicit presets such as `get_recipe("ka2")` remain available. This is the
owner-selected research default; Forge calibration remains provisional.
The [throughput readout](reports/forge/bispf-throughput/README.md) preserves
19 passing and 11 failing task verdicts, while measuring 2.15× faster 256×256
DCGAN updates and about 2.3× faster small image updates versus 0.9.0. Across
all six DCGAN workloads, speedups range from 1.02× to 2.15×. These results
do not establish natural-image convergence or a universal architecture guarantee.

![Actual training of the 0.9.1 BISPF default on the stripes task, with numerical quality observations](reports/forge/bispf-throughput/media/bispf-throughput-v1/img_stripes2.gif)

The 0.9.1 default passes this fixed 600-update stripes task with all required
modes. Its matched mean update throughput improves 2.30× over 0.9.0;
[metrics and source-bound evidence](reports/forge/bispf-throughput/README.md)
include the unchanged bars failure and separate wide-channel controls.

![Archived 0.9.0 100-Gaussian training: particle outputs, target modes and numerical learning curves](100gaussians.gif)

Recorded with the **0.9.0 optimized BISPF default**, public deterministic initialization,
live weights and protocol seed 0: **100/100 modes, 42.1% within 3σ after 20,000 updates**.
The 95% precision goal is not met. This animation retains its 0.9.0 recipe;
[metrics and reproduction](reports/readme-100gaussians/README.md#current-default)
preserve the full result and the historical K3P animation.

**Without a learning-rate schedule.** The [E22 configuration](docs/e22.md) has no schedule or training
horizon, and no statistic of the raw data in its training control. It covers all 100 modes on three layouts,
ending at 98.3–98.6% of samples within 3σ after 7,000 updates:

![E22 on the grid, rotated and staggered 100-Gaussian problems](reports/e22-animation/e22-100gaussians.gif)

[How this animation was made](reports/e22-animation/README.md).

## Install

Requires Python 3.10+, PyTorch, and NumPy (installed as dependencies).

```bash
python -m pip install particlegan           # the library (0.9.2)
```

For the examples, experiments and tests, install from source:

```bash
git clone https://github.com/255BITS/ParticleGAN.git
cd ParticleGAN
python -m pip install -e '.[experiments,dev]'  # plus examples, experiments and tests
```

## Train a GAN

Version 0.9.1 makes the faster BISPF execution policy the default and retains
explicit historical settings and saved recipe restoration. See the [changelog](CHANGELOG.md)
for the release and its compatibility limits.

You supply ordinary PyTorch modules; `GANTrainer` binds the recipe's optimizers,
critic penalty and prior regularizer:

```python
import torch
from torch import nn
from particlegan import GANTrainer, get_recipe, init

def real_batch(n):  # replace with your DataLoader: 8 Gaussians on a ring
    angle = torch.randint(8, (n, 1)) * torch.pi / 4
    return torch.cat([angle.cos(), angle.sin()], 1) + 0.05 * torch.randn(n, 2)

recipe = get_recipe(total_steps=2000)
G = nn.Sequential(nn.Linear(recipe.z_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 2))
D = nn.Sequential(nn.Linear(2, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))
init.deterministic_orthogonal_(G, seed=0)
init.deterministic_orthogonal_(D, seed=1)
prior = init.deterministic_orthogonal_(recipe.make_prior())
trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)
for _ in range(recipe.total_steps):
    trainer.step(real_batch(recipe.batch_size))
samples = trainer.sample(1024)
```

Without `prior=`, the trainer builds a randomly drawn particle table. Networks
and device placement belong to the caller. The trainer does not reinitialize
supplied modules, so load pretrained weights before constructing it and
initialize only fresh parameters. Optimizers exclude frozen parameters while
retaining gradients through frozen modules to trainable inputs.

Caller-owned loops use the recipe's role-named factories, including
`make_optimizers`, `make_critic_penalty` and `make_prior_regularizer`.
Save the resolved `recipe.to_dict()` and the complete `trainer.state_dict()`
for continuation; changing the current default does not migrate an old run.
See the [API reference](docs/api.md#recipes-and-defaults) for component ownership
and optimizer contracts.

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

`get_recipe(name)` selects a model family or an E22 policy preset.

| Name | Model |
| --- | --- |
| `gan` (default) | GAN with a learnable particle prior |
| `e22` | Scalar particle GAN with stationarity control, row evidence, birth/death, learned noise and served averages |
| `e22_routed` | Conditional dense-bank adaptation with paired row evidence and guarded birth/death |
| `mog` | GAN with a mixture-of-Gaussians particle prior |
| `ddgan` | Denoising-diffusion GAN with a UCD (class-conditional) critic |
| `ddgan_mog` | `ddgan` with a mixture-of-Gaussians prior |
| `ae_gan` | Autoencoder GAN: an encoder routes data to particles |
| `vae_gan` | Variational variant of `ae_gan` |
| `ae_ddgan` | Autoencoder denoising-diffusion GAN |

Any field can be overridden: `get_recipe("mog", total_steps=20_000)`.
For E22, set the task's dimensions and output noise explicitly:
`get_recipe("e22", num_particles=20_000, z_dim=2, batch_size=2048, output_noise_std=.029)`.
[`E22Policy`](docs/e22.md) exposes the same controls used by `GANTrainer` for
caller-owned backward passes and optimizer steps, with checkpointing and served snapshots.
For conditional softmax blends, use `get_recipe("e22_routed", ...)` with the
explicit [`RoutedRows` contract](docs/e22_routed.md). This adaptation measures
the whole conditional forward and validates row moves on separate guard
contexts. Frozen BF16 modules can accompany FP32 trainable parameters and tables.
Multiple token routing sites can share one bank and controller through the
[full-model routing contract](docs/e22_routed_sites.md). Candidate checks rerun
the complete model, including downstream sites and the final paired output.
`RoutedRows(probe_interval=20)` schedules expensive row probes separately from
per-update gradient evidence and saves its clock in checkpoints. The optional
`output_error_guard=True` bounds clean paired-output MSE on guard contexts;
it does not establish repair of the noisy adversarial game. Split rows
transport their parent's Adam history at half mass.

## Learn more

- [Optimized BISPF settings and measured behavior](reports/forge/bispf-optimized/README.md)
- [Historical KA2 formulation](docs/ka2.md), including several critics and conditional critics
- [E22](docs/e22.md): a schedule-free configuration for the native 100-Gaussian problems with no data-space statistics
- [ParticleGAN Atlas](docs/atlas.md): illustrated E22 comparison, local population corrections and automatic backend selection
- [Routed paired E22](docs/e22_routed.md): conditional row evidence, guarded moves and clean serving
- [Shared-bank routing sites](docs/e22_routed_sites.md): sequential token routing and a matched spatial comparison
- [Whole-model checkpoint replay](docs/e22_routed_sites.md#activation-checkpointed-whole-model-replay): per-site DV12 without repeated training draws or diagnostics
- [Routed noisy-game qualification](docs/e22_routed_game.md): token penalty units, mass invariance and matched longer training
- [Many-site synchronization](docs/e22_routed_readbacks.md): deferred DV12 diagnostics and batched routing validation
- [API reference](docs/api.md) and a [minimal DDGAN + UCD loop](docs/api.md#a-minimal-ddgan--ucd-loop)
- Examples: [`quickstart_gan.py`](examples/quickstart_gan.py) (GANTrainer with checkpoints),
  [`e22_external_loop.py`](examples/e22_external_loop.py) (E22 in a caller-owned loop),
  [`e22_routed_paired.py`](examples/e22_routed_paired.py) (source/time-conditioned paired-error training),
  [`e22_routed_sites.py`](examples/e22_routed_sites.py) (sequential shared-bank routing and spatial ablations),
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
