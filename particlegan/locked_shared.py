"""Demo locked_shared adversarial stamp.

This is the slider-lock / #94 posture other code can import. It is a frozen
definition plus two builders. It is not a trainer, a toy gate, or a gym config.

``lazy_k`` is 1, the ``GradientPenalty`` and ``Recipe`` default and the #94
``grad_lazy``. No repository trainer uses this stamp; the YuE2 gym controller
trains with the recipe's default critic penalty.

``get_recipe("gan")`` uses the shared GAN defaults (20_000 particles,
VICReg ``prior_reg`` .05). This stamp's 12-particle cloud and ``particle_l2``
apply only when the caller builds a cloud. The builders do not construct a
prior and do not add that term to the GAN loss.

The critic is the caller's. ``critic="host"`` refuses a Music MLP alias.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Mapping

from .gan_loss import GANLoss
from .grad_regularizers import GradientPenalty

# Slider / conceptmod names for the same pins. The public fields use this
# package's constructor words (``lazy_k``, ``critic``).
FIELD_ALIASES = MappingProxyType({
    "reg_coeff": ("b_cap", "adv_b_cap"),
    "reg_kappa": ("kappa", "adv_reg_kappa"),
    "reg_arm": ("grad_arm",),
    "reg_norm": ("grad_norm", "adv_norm"),
    "lazy_k": ("grad_lazy", "reg_lazy"),
    "critic": ("critic_arch",),
})


@dataclass(frozen=True)
class LockedShared:
    """Formulation pins. Budget (steps, seed, learning rate) is not here."""

    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    reg_norm: str = "l2"
    lazy_k: int = 1
    reg_method: str = "autograd"
    target_anneal: str = "none"
    fm_weight: float = 0.0
    cover_weight: float = 1.5
    cover_posture: str = "demo"
    n_particles: int = 12
    particle_l2: float = 0.02
    z_dim: int = 2
    pairing: str = "live"
    critic: str = "host"
    reg_impl: str = "grad_regularizer"

    def to_dict(self) -> dict:
        """Mutable copy. The frozen view is ``locked_adv_defaults()``."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


LOCKED_SHARED = LockedShared()

# Named drifts. Each one differs from LOCKED_SHARED. They are not recipes.
NAMED_DRIFTS = MappingProxyType({
    "music_cover_1": MappingProxyType({"cover_weight": 1.0, "cover_posture": "music"}),
    "hub128": MappingProxyType({"n_particles": 128}),
    "fm_on": MappingProxyType({"fm_weight": 0.1}),
    "stranger": MappingProxyType({"pairing": "stranger"}),
    "thinned_kappa": MappingProxyType({"reg_impl": "thinned_kappa"}),
})


def locked_adv_defaults() -> Mapping:
    """Frozen field map of ``LOCKED_SHARED``.

    ``particle_l2``, ``n_particles``, and ``z_dim`` describe the demo cloud.
    Apply them only when building particles. A host prior keeps its own width.
    """
    return MappingProxyType(LOCKED_SHARED.to_dict())


def drift(name: str) -> LockedShared:
    """Return one named non-stamp. Unknown names raise ``KeyError``."""
    return replace(LOCKED_SHARED, **NAMED_DRIFTS[name])


def _require_locked(stamp: LockedShared) -> LockedShared:
    if type(stamp) is not LockedShared:
        raise TypeError(f"stamp must be LockedShared, got {type(stamp).__name__}")
    if stamp != LOCKED_SHARED:
        drifted = [
            item.name for item in fields(LOCKED_SHARED)
            if getattr(stamp, item.name) != getattr(LOCKED_SHARED, item.name)
        ]
        raise ValueError(
            "locked_shared builders accept only LOCKED_SHARED; "
            f"drifted: {', '.join(drifted)}"
        )
    return stamp


def make_gan_loss(stamp: LockedShared = LOCKED_SHARED) -> GANLoss:
    """RpGAN logistic loss for the locked stamp. Drifts are refused."""
    _require_locked(stamp)
    return GANLoss(loss_type=stamp.loss_type, mode=stamp.gan_mode)


def make_b_cap(stamp: LockedShared = LOCKED_SHARED) -> GradientPenalty:
    """Sample-point ``b_cap`` for the locked stamp.

    The object is ``GradientPenalty`` itself (an alias of ``GradRegularizer``),
    so the cap center is ``kappa``. A thinned regularizer that stores kappa
    and hardcodes the center is not this builder.
    """
    _require_locked(stamp)
    return GradientPenalty(
        arm=stamp.reg_arm,
        coeff=stamp.reg_coeff,
        kappa=stamp.reg_kappa,
        norm=stamp.reg_norm,
        lazy_k=stamp.lazy_k,
        method=stamp.reg_method,
        target_anneal=stamp.target_anneal,
    )


__all__ = [
    "FIELD_ALIASES",
    "LOCKED_SHARED",
    "LockedShared",
    "NAMED_DRIFTS",
    "drift",
    "locked_adv_defaults",
    "make_b_cap",
    "make_gan_loss",
]
