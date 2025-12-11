# ParticleGAN

**GANs don't collapse when z can move too.**

![100 Gaussians with Particle Prior](100gaussians.gif)

## The Problem

Traditional GANs suffer from **mode collapse**: the generator learns to produce only a subset of the data distribution, ignoring other valid modes. This happens because G must warp a *fixed* prior (usually a Gaussian) to match the data. All geometric stress concentrates in G, causing the learned manifold to fold and tear.

## The Insight

**What if the prior could move too?**

Instead of forcing G to do all the work, we introduce learnable "particles" in latent space. These particles move during training to match the structure of the data, absorbing geometric stress alongside G. The result: stable convergence even on highly multimodal distributions.

### Without Particles: Mode Collapse

![100 Gaussians without Particle Prior](100gaussians_no_particles.gif)

*Same architecture, same hyperparameters, but with a fixed Gaussian prior — the generator collapses to a subset of modes.*

## Results

| Problem | Fixed Gaussian Prior | Particle Prior |
|---------|---------------------|----------------|
| 5 modes (text) | collapse | **converges** |
| 100 modes (2D grid) | collapse | **converges** |

## How It Works

1. **Particle Prior**: Instead of sampling z ~ N(0, I), we maintain a set of learnable latent vectors (particles). During training, we sample from this discrete set.

2. **Joint Optimization**: Particles are optimized alongside G and D. They naturally spread out to cover the data modes.

3. **VICReg Regularization**: We apply variance-covariance regularization to prevent particles from collapsing to a single point, while allowing arbitrary topology (clusters, gaps, etc.).

## Examples

### Five Modes (Text Generation)

A minimal example demonstrating the core idea. Five words ("apple", "grape", "lemon", "melon", "berry") are encoded into a 2D latent space. Each word gets one particle.

![Five Modes Training](five_modes.gif)

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

**With particle prior**: All 100 modes are captured.

**Without particle prior** (baseline):
```bash
python examples/100gaussians_no_particle_prior.py
```

The baseline demonstrates classic mode collapse — the generator covers only a fraction of the modes.

## Installation

```bash
git clone https://github.com/255BITS/ParticleGAN.git
cd ParticleGAN
pip install torch matplotlib numpy
```

## Project Structure

```
ParticleGAN/
├── lib/
│   ├── particle_prior.py   # Learnable particle cloud (nn.Module)
│   ├── gan_loss.py         # Flexible GAN losses (hinge, logistic, Wasserstein, LSGAN)
│   └── vicreg_loss.py      # Variance-covariance regularization
├── examples/
│   ├── five_modes.py                    # Text generation toy problem
│   ├── 100gaussians.py                  # 100-mode benchmark (with particles)
│   └── 100gaussians_no_particle_prior.py # Baseline (without particles)
└── README.md
```

## Key Components

### ParticlePrior (`lib/particle_prior.py`)

A simple nn.Module holding M learnable latent vectors of dimension D:

```python
from lib.particle_prior import ParticlePrior

prior = ParticlePrior(num_particles=1000, z_dim=2)
z, indices = prior.sample(batch_size=64)  # Sample 64 particles
```

### GANLoss (`lib/gan_loss.py`)

Supports multiple loss types and relativistic variants:

```python
from lib.gan_loss import GANLoss

loss_fn = GANLoss(loss_type='hinge', mode='vanilla')
d_loss = loss_fn.d_loss(d_real, d_fake)
g_loss = loss_fn.g_loss(d_real, d_fake)
```

### VICRegLikeLoss (`lib/vicreg_loss.py`)

Prevents particle collapse while allowing flexible topology:

```python
from lib.vicreg_loss import VICRegLikeLoss

reg = VICRegLikeLoss()
loss = reg(particle_positions)  # Encourages spread + decorrelation
```

## Notes

- The text experiments (`five_modes.py`) use R1 gradient penalty for stability
- The 100-Gaussian experiments work without gradient penalty
- Particles use a higher learning rate (10×) than G/D for faster adaptation

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
