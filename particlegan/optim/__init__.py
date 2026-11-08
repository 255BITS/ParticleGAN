"""Optimizer implementations selected by :class:`particlegan.Recipe`."""

from .dualnorm import NORMALIZED_FAMILIES, NormalizedOptimizer, polar_factor
from .gram_polar import GramGuardError, guarded_gram_polar
from .gram_svd_fallback import svd_fallback_polar

__all__ = ["NORMALIZED_FAMILIES", "NormalizedOptimizer", "polar_factor",
           "GramGuardError", "guarded_gram_polar", "svd_fallback_polar"]
