"""Born-Infeld Sinkhorn Particle Flow (BISPF) for schedule-free ParticleGAN training.

Implements the four time-invariant pillars of BISPF:
1. Scale-Free Sinkhorn Relativistic Pairing (SRpGAN)
2. Elastic Born-Infeld Transport-Chord Regularization (EBI-TCR) & Zero-Mean Logit Gauge
3. Stratified Blue-Noise Particle Prior (SBNP)
4. Residual-Driven Thermal Noise (RDTN)

This module is the stable import surface. The implementation lives in
``particlegan._bispf``:

* ``gates``     -- ``BISPFGates``, the declared shape and step gates;
* ``thermal``   -- the legacy thermal channel, ``bind_thermal_source`` and
  ``ContinuousThermalCooler``;
* ``prior``     -- the SBNP regularizer and ``StratifiedBlueNoisePrior``;
* ``transport`` -- ``sinkhorn_coupling`` (patch it there to reach every caller);
* ``engine``    -- ``BISPFEngine`` and ``BISPFLossAdapter``;
* ``critic``    -- step records, the critic forward hook and ``BISPFCriticAdam``;
* ``loss``      -- ``SinkhornRelativisticLoss``;
* ``penalty``   -- the EBI-TCR kernel and ``BISPFCriticPenalty``.
"""
from __future__ import annotations

# Private names below are compatibility re-exports (tests and archived tooling
# import them from ``particlegan.bispf``); they are deliberately absent from
# ``__all__``. ``noqa: F401`` keeps ``ruff --fix`` from deleting them.

from ._bispf.critic import (
    BISPFCriticAdam,
    BISPFStepRecord,
    _BISPFObserverStepRecord,  # noqa: F401
    _attach_critic_input_hook,  # noqa: F401
    _bispf_critic_forward_hook,  # noqa: F401
)
from ._bispf.engine import BISPFEngine, BISPFLossAdapter
from ._bispf.gates import BISPFGates, _DEFAULT_GATES  # noqa: F401
from ._bispf.loss import SinkhornRelativisticLoss
from ._bispf.penalty import _PHI1, _PHI2, BISPFCriticPenalty, _BISPFPenaltyKernel  # noqa: F401
from ._bispf.prior import SBNPPriorRegularizer, StratifiedBlueNoisePrior, sbnp_regularization_loss
from ._bispf.thermal import (
    FROZEN_THERMAL_SCALE,  # noqa: F401
    ContinuousThermalCooler,
    _LAST_THERMAL_SCALE,  # noqa: F401
    _LEGACY_THERMAL_CHANNEL,  # noqa: F401
    _publish_thermal,  # noqa: F401
    _read_thermal,  # noqa: F401
    bind_thermal_source,
)
from ._bispf.transport import _cloud_scale_factor, sinkhorn_coupling  # noqa: F401

__all__ = [
    "BISPFGates",
    "bind_thermal_source",
    "StratifiedBlueNoisePrior",
    "SBNPPriorRegularizer",
    "sbnp_regularization_loss",
    "ContinuousThermalCooler",
    "sinkhorn_coupling",
    "BISPFEngine",
    "BISPFLossAdapter",
    "BISPFStepRecord",
    "BISPFCriticAdam",
    "BISPFCriticPenalty",
    "SinkhornRelativisticLoss",
]
