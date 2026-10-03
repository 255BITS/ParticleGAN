"""Composable PyTorch primitives for GANs with learned particle priors.

Networks and devices belong to the caller; GANTrainer optionally owns updates.
"""
from .autoencoder import ParticleEncoding, particle_ae, particle_vae
from .capabilities import prior_capabilities, prior_mechanisms
from .birth_death import ParticleRows, ScalarHeadFeatures
from .conditioning import UCD, ucd_labels, ucd_loss, ucd_scores
from .continuous import DataDriftController
from .diffusion import DDGAN
from .discriminators import BatchDistanceDiscriminator, LinearSkipDiscriminator
from .gan_loss import GANLoss
from .particle_prior import GaussianPrior, MoGParticlePrior, ParticlePrior, calibrate_mog_sigma
from .policy import E22Policy, ServedModel, StepNoise, UpdatePolicy
from .recipes import (
    NetworkLRTransition,
    Recipe,
    get_recipe,
    learning_rate_scale,
    learning_rate_scales,
    scale_learning_rates,
)
from .routing import RoutedBatch, RoutedCandidate, RoutedExecution, RoutedRows
from .training import GANTrainer, InputNoise
from .vicreg_loss import ParticleRegularizer
from . import init

__all__ = [
    "init",
    "prior_capabilities", "prior_mechanisms",
    "ParticleEncoding", "particle_ae", "particle_vae",
    "ParticleRows", "ScalarHeadFeatures",
    "RoutedBatch", "RoutedCandidate", "RoutedExecution", "RoutedRows",
    "E22Policy", "UpdatePolicy", "ServedModel", "StepNoise",
    "ParticlePrior", "MoGParticlePrior", "calibrate_mog_sigma", "GaussianPrior", "GANLoss",
    "ParticleRegularizer", "DDGAN", "UCD", "ucd_labels", "ucd_loss", "ucd_scores",
    "Recipe", "NetworkLRTransition", "DataDriftController", "get_recipe", "learning_rate_scale", "learning_rate_scales", "scale_learning_rates", "GANTrainer", "InputNoise",
    "BatchDistanceDiscriminator", "LinearSkipDiscriminator",
]
