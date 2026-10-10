"""Optimizer implementations selected by :class:`particlegan.Recipe`."""

from .dualnorm import NORMALIZED_FAMILIES, NormalizedOptimizer, polar_factor

__all__ = ["NORMALIZED_FAMILIES", "NormalizedOptimizer", "polar_factor"]
