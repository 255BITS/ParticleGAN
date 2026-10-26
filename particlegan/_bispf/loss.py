"""Scale-Free Sinkhorn Relativistic Pairing (SRpGAN) loss."""
from __future__ import annotations

import math
from typing import Any, Tuple

import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from . import transport as _transport
from .critic import BISPFStepRecord
from .gates import BISPFGates, _DEFAULT_GATES
from .thermal import _read_thermal


class SinkhornRelativisticLoss(GANLoss):
    """Scale-Free Sinkhorn Relativistic Pairing (SRpGAN) loss compatible with ``GANLoss``.

    ``g_loss`` applies the declared dense rescale (``BISPFGates.dense_g_rescale_*``).
    It reads the critic thermal scale from ``thermal_source`` (bound by
    ``bind_thermal_source``), or from the legacy process channel when unbound.
    """

    def __init__(
        self,
        loss: str = "sinkhorn_relativistic",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 5,
        record: BISPFStepRecord | None = None,
        gates: BISPFGates | None = None,
        thermal_source: Any = None,
    ) -> None:
        super().__init__("relativistic", labels=labels)
        if gates is not None and not isinstance(gates, BISPFGates):
            raise TypeError("gates must be a BISPFGates instance")
        self.loss = loss
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.record = record
        self.gates = _DEFAULT_GATES if gates is None else gates
        self.thermal_source = thermal_source

    def _coords(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_r = getattr(real_logits, "_bispf_input", None)
        x_f = getattr(fake_logits, "_bispf_input", None)
        if x_r is None or x_f is None or x_r.shape[0] != real_logits.shape[0] or x_f.shape[0] != fake_logits.shape[0]:
            x_r = real_logits.detach().reshape(real_logits.shape[0], -1)
            x_f = fake_logits.detach().reshape(fake_logits.shape[0], -1)
        return x_r, x_f

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        x_r, x_f = self._coords(real_logits, fake_logits)
        Pi, _, _, _ = _transport.sinkhorn_coupling(
            x_r, x_f, sinkhorn_eps=self.sinkhorn_eps, sinkhorn_iters=self.sinkhorn_iters
        )
        rec = self.record or getattr(real_logits, "_bispf_record", None)
        if rec is not None:
            rec._last_d_logits = (r, f)
        b_r = r.shape[0]
        delta = r.unsqueeze(1) - f.unsqueeze(0)
        return (b_r * Pi * F.softplus(-delta)).sum()

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        if real_logits is None:
            raise ValueError("SRpGAN requires real_logits in g_loss")
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        x_r, x_f = self._coords(real_logits, fake_logits)
        Pi, _, _, _ = _transport.sinkhorn_coupling(
            x_r, x_f, sinkhorn_eps=self.sinkhorn_eps, sinkhorn_iters=self.sinkhorn_iters
        )
        b_r = r.shape[0]
        delta = r.unsqueeze(1) - f.unsqueeze(0)
        loss_val = (b_r * Pi * F.softplus(delta)).sum()
        gates = self.gates
        s_dense = _read_thermal(self.thermal_source, fake_logits.device)
        b_f = fake_logits.shape[0]
        if (gates.active("dense_g_rescale") and b_f >= gates.dense_g_rescale_min_batch
                and s_dense < gates.dense_g_rescale_max_scale):
            linearized = s_dense / math.sqrt(max(1e-6, 1.0 - s_dense**2))
            s_v = max(1e-6, min(1.0, linearized * gates.dense_g_rescale_eps_smooth))
            loss_val = loss_val * s_v + loss_val.detach() * (1.0 - s_v)
        return loss_val

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        return self.g_loss(fake_logits, real_logits)
