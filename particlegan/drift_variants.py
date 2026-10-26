"""Feature flags of the continuous data-drift controller variants ``"dv1"`` ... ``"dv12"``.

Each variant name enables a fixed set of mechanisms. ``drift_features`` derives
them in one place, so modules test a named mechanism instead of repeating
tuples of variant names. Torch-free, so ``recipes`` can import it.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

DRIFT_VARIANTS = tuple(f"dv{level}" for level in range(1, 13))
_LEVEL = {name: level for level, name in enumerate(DRIFT_VARIANTS, 1)}


class DriftFeatures(NamedTuple):
    """Mechanisms enabled by one variant (all False for ``dv1``, ``None`` or an unknown name)."""

    data_memory: bool        # dv2+: slow memory of the data drive
    calm_anchor_hold: bool   # dv2: a calm KA2 anchor stops moving (release rate 0)
    fast_reference: bool     # dv3+: fast/slow EMA difference as the data score
    game_trust: bool         # dv4+: Adam-surprise game trust damps KA2 release
    payoff: bool             # dv5+: payoff error drives the mobility target
    drive_evidence: bool     # dv6+: data drive (not its memory) explains surprise
    critic_balance: bool     # dv7+: payoff-error critic learning-rate scale
    pair: bool               # dv8, dv9: paired real/fake feature drive
    pair_target: bool        # dv8: the pair drive raises the mobility target
    pair_prior: bool         # dv9: the pair drive raises the prior mobility
    latent_bandwidth: bool   # dv10-dv12: latent perturbation at the learned bandwidth
    support_trust: bool      # dv11: the probed support trust scales the perturbation
    local_support: bool      # dv12: the perturbation is clipped to half the nearest-row radius


@lru_cache(maxsize=None)
def drift_features(variant) -> DriftFeatures:
    """The ``DriftFeatures`` of a variant name."""
    level = _LEVEL.get(variant, 0)
    return DriftFeatures(
        data_memory=level >= 2,
        calm_anchor_hold=level == 2,
        fast_reference=level >= 3,
        game_trust=level >= 4,
        payoff=level >= 5,
        drive_evidence=level >= 6,
        critic_balance=level >= 7,
        pair=level in (8, 9),
        pair_target=level == 8,
        pair_prior=level == 9,
        latent_bandwidth=level >= 10,
        support_trust=level == 11,
        local_support=level == 12,
    )
