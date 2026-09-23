"""Composable PyTorch primitives for GANs with learned particle priors.

Networks and devices belong to the caller; GANTrainer optionally owns updates.
"""
from .autoencoder import ParticleEncoding, particle_ae, particle_vae
from .conditioning import UCD, ucd_labels, ucd_loss, ucd_scores
from .diffusion import DDGAN
from .discriminators import BatchDistanceDiscriminator, LinearSkipDiscriminator
from .gan_loss import GANLoss
from .grad_regularizers import GradientPenalty
from .locked_shared import LOCKED_SHARED, locked_adv_defaults, make_b_cap, make_gan_loss
from .particle_prior import GaussianPrior, MoGParticlePrior, ParticlePrior
from .recipes import Recipe, get_recipe, learning_rate_scale
from .training import GANTrainer
from .vicreg_loss import ParticleRegularizer

__all__ = [
    "ParticleEncoding", "particle_ae", "particle_vae",
    "ParticlePrior", "MoGParticlePrior", "GaussianPrior", "GANLoss", "GradientPenalty",
    "ParticleRegularizer", "DDGAN", "UCD", "ucd_labels", "ucd_loss", "ucd_scores",
    "Recipe", "get_recipe", "learning_rate_scale", "GANTrainer",
    "BatchDistanceDiscriminator", "LinearSkipDiscriminator",
    "LOCKED_SHARED", "locked_adv_defaults", "make_gan_loss", "make_b_cap",
]
