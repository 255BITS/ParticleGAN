"""Composable PyTorch primitives for GANs with learned particle priors.

Networks and devices belong to the caller; GANTrainer optionally owns updates.
Importing ParticleGAN disables multithreaded autograd scheduling on this thread.
"""
from .execution import disable_autograd_multithreading, serial_autograd

disable_autograd_multithreading()

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
    DEFAULT_RECIPE,
    NetworkLRTransition,
    Recipe,
    get_recipe,
    get_recipe_fields,
    learning_rate_scale,
    learning_rate_scales,
    scale_learning_rates,
)
from .routing import RoutedBatch, RoutedCandidate, RoutedExecution, RoutedRows
from .training import GANTrainer, InputNoise
from .tensorflow_adam import TensorFlowV1Adam
from .vicreg_loss import ParticleRegularizer
from .coulomb_prior import CoulombPriorRegularizer
from .sigreg import SIGReg
from .initial_mmd import InitialMMDRegularizer
from .bispf import (
    BISPFCriticAdam,
    BISPFCriticPenalty,
    BISPFEngine,
    BISPFLossAdapter,
    BISPFStepRecord,
    ContinuousThermalCooler,
    SBNPPriorRegularizer,
    SinkhornRelativisticLoss,
    StratifiedBlueNoisePrior,
)
from .priors import (
    GhostFreeDeMixingManipulator,
    ManipulatedPrior,
    PriorManipulator,
    SimplexRoutingManipulator,
    SobolevWavefrontManipulator,
    SplatCarrierManipulator,
    StratifiedSamplingManipulator,
)
from .wavefront import (
    DifferentiableSplatCarrierBasis,
    GhostFreeSimplexMixer,
    ResidualConvBlock,
    SimplexWavefrontGenerator,
    SobolevGradFilter,
    WavefrontManifoldPrior,
    sobolev_filter_2d,
)
from .tomographic import (
    CosineRepulsionManipulator,
    MarginalEntropyManipulator,
    MarginalEntropyRegularizer,
    NormalizedFeatureCosineRepulsion,
    TomographicConstraintTarget,
    evaluate_tomographic_constraints,
    symmetric_log_odds,
)
from . import init

__all__ = [
    "disable_autograd_multithreading", "serial_autograd",
    "init",
    "TensorFlowV1Adam",
    "prior_capabilities", "prior_mechanisms",
    "ParticleEncoding", "particle_ae", "particle_vae",
    "ParticleRows", "ScalarHeadFeatures",
    "RoutedBatch", "RoutedCandidate", "RoutedExecution", "RoutedRows",
    "E22Policy", "UpdatePolicy", "ServedModel", "StepNoise",
    "ParticlePrior", "MoGParticlePrior", "calibrate_mog_sigma", "GaussianPrior", "GANLoss",
    "CoulombPriorRegularizer", "ParticleRegularizer", "InitialMMDRegularizer", "SIGReg",
    "StratifiedBlueNoisePrior", "SBNPPriorRegularizer", "ContinuousThermalCooler", "BISPFEngine", "BISPFLossAdapter",
    "BISPFStepRecord", "BISPFCriticAdam", "BISPFCriticPenalty", "SinkhornRelativisticLoss",
    "DDGAN", "UCD", "ucd_labels", "ucd_loss", "ucd_scores",
    "DEFAULT_RECIPE", "Recipe", "NetworkLRTransition", "DataDriftController", "get_recipe", "get_recipe_fields", "learning_rate_scale", "learning_rate_scales", "scale_learning_rates", "GANTrainer", "InputNoise",
    "BatchDistanceDiscriminator", "LinearSkipDiscriminator",
    "PriorManipulator", "ManipulatedPrior", "StratifiedSamplingManipulator", "SimplexRoutingManipulator",
    "GhostFreeDeMixingManipulator", "SobolevWavefrontManipulator", "SplatCarrierManipulator",
    "sobolev_filter_2d", "SobolevGradFilter", "WavefrontManifoldPrior",
    "GhostFreeSimplexMixer", "DifferentiableSplatCarrierBasis", "ResidualConvBlock", "SimplexWavefrontGenerator",
    "symmetric_log_odds", "TomographicConstraintTarget",
    "MarginalEntropyRegularizer", "MarginalEntropyManipulator",
    "NormalizedFeatureCosineRepulsion", "CosineRepulsionManipulator",
    "evaluate_tomographic_constraints",
]

