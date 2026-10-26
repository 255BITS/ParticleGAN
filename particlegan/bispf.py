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

from ._bispf.critic import (
    BISPFCriticAdam,
    BISPFStepRecord,
    _BISPFObserverStepRecord,
    _attach_critic_input_hook,
    _bispf_critic_forward_hook,
)
from ._bispf.engine import BISPFEngine, BISPFLossAdapter
from ._bispf.gates import BISPFGates, _DEFAULT_GATES
from ._bispf.loss import SinkhornRelativisticLoss
from ._bispf.penalty import _PHI1, _PHI2, BISPFCriticPenalty, _BISPFPenaltyKernel
from ._bispf.prior import SBNPPriorRegularizer, StratifiedBlueNoisePrior, sbnp_regularization_loss
from ._bispf.thermal import (
    FROZEN_THERMAL_SCALE,
    ContinuousThermalCooler,
    _LAST_THERMAL_SCALE,
    _LEGACY_THERMAL_CHANNEL,
    _publish_thermal,
    _read_thermal,
    bind_thermal_source,
)
from ._bispf.transport import _cloud_scale_factor, sinkhorn_coupling

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
