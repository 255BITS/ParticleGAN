"""Composable PyTorch primitives for GANs with learned particle priors.

Networks and devices belong to the caller; GANTrainer optionally owns updates.
"""
from .autoencoder import ParticleEncoding, particle_ae, particle_vae
from .conditioning import UCD, ucd_labels, ucd_loss, ucd_scores
from .diffusion import DDGAN
from .discriminators import BatchDistanceDiscriminator, LinearSkipDiscriminator
from .gan_loss import GANLoss
from .grad_regularizers import GradientPenalty
from .k3p import CriticAnchor, CriticSpikeGuard, DirectParticleResponse, LatentRowDamping
from .locked_shared import LOCKED_SHARED, locked_adv_defaults, make_b_cap, make_gan_loss
from .particle_prior import GaussianPrior, MoGParticlePrior, ParticlePrior, calibrate_mog_sigma
from .recipes import Recipe, get_recipe, learning_rate_scale, learning_rate_scales
from .training import GANTrainer, K3PCritic
from .vicreg_loss import ParticleRegularizer

__all__ = [
    "ParticleEncoding", "particle_ae", "particle_vae",
    "ParticlePrior", "MoGParticlePrior", "calibrate_mog_sigma", "GaussianPrior", "GANLoss", "GradientPenalty",
    "CriticAnchor", "CriticSpikeGuard", "LatentRowDamping", "DirectParticleResponse",
    "ParticleRegularizer", "DDGAN", "UCD", "ucd_labels", "ucd_loss", "ucd_scores",
    "Recipe", "get_recipe", "learning_rate_scale", "learning_rate_scales", "GANTrainer", "K3PCritic",
    "BatchDistanceDiscriminator", "LinearSkipDiscriminator",
    "LOCKED_SHARED", "locked_adv_defaults", "make_gan_loss", "make_b_cap",
]
