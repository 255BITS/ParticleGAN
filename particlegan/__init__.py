"""Composable PyTorch primitives for GANs with learned particle priors.

Networks, training loops, devices, and optimizer steps belong to the caller.
"""
from .autoencoder import ParticleEncoding, particle_ae, particle_vae
from .conditioning import UCD, ucd_labels, ucd_loss, ucd_scores
from .diffusion import DDGAN
from .gan_loss import GANLoss
from .grad_regularizers import GradientPenalty
from .particle_prior import GaussianPrior, MoGParticlePrior, ParticlePrior
from .recipes import Recipe, get_recipe, learning_rate_scale
from .vicreg_loss import ParticleRegularizer

__all__ = [
    "ParticleEncoding", "particle_ae", "particle_vae",
    "ParticlePrior", "MoGParticlePrior", "GaussianPrior", "GANLoss", "GradientPenalty",
    "ParticleRegularizer", "DDGAN", "UCD", "ucd_labels", "ucd_loss", "ucd_scores",
    "Recipe", "get_recipe", "learning_rate_scale",
]
