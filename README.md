# ParticleGAN

**GANs don't collapse when z can move too.**

![Convergence](results/convergence.gif)

## The Problem

GANs mode collapse because the generator must warp a fixed prior distribution to match the data. All geometric error concentrates in G, causing the manifold to fold and tear.

## The Insight

What if the prior could move too?

Learnable "particles" in latent space absorb geometric stress alongside G.

![Failure without particles](results/failure.gif)

*Same setup, fixed Gaussian prior — collapse.*

## Results

| Setup | Fixed Gaussian | Particle Prior |
|-------|----------------|----------------|
| 5 modes | collapse | **converges** |
| 100 modes | collapse | **converges** |

## Quick Start

```bash
git clone https://github.com/255BITS/ParticleGAN.git
cd ParticleGAN
python examples/five_modes.py
```

## Note

The text experiments use R1 gradient penalty for stability. 
The 100-Gaussian experiments work without it.

# Citation
```
@software{particlegan2025,
  author = {Martyn Garcia},
  title = {ParticleGAN: Learnable Priors for Stable GANs},
  year = {2025},
  url = {https://github.com/255BITS/ParticleGAN}
}
```

# License

MIT
