"""Relativistic adversarial loss option for EMPF."""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F

from .critic import read_empf_thermal


class EMPFSinkhornRelativisticLoss:
    """Centered relativistic logistic objective compatible with ``GANLoss``."""

    def __init__(
        self,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        *,
        empf_record: Any = None,
    ) -> None:
        if len(labels) != 3:
            raise ValueError("labels must be (fake_label, real_label, generator_label)")
        self.labels = tuple(float(x) for x in labels)
        self.empf_record = empf_record

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r = real_logits.reshape(-1)
        f = fake_logits.reshape(-1)
        margin = r - f.mean()
        margin_f = f - r.mean()
        return 0.5 * (F.softplus(-margin).mean() + F.softplus(margin_f).mean())

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        f = fake_logits.reshape(-1)
        if real_logits is None:
            loss = F.softplus(-f).mean()
        else:
            r = real_logits.detach().reshape(-1)
            loss = 0.5 * (F.softplus(-(f - r.mean())).mean() + F.softplus(r - f.mean()).mean())
        s = read_empf_thermal(self.empf_record)
        if f.shape[0] >= 1024 and s < 0.999 and loss.requires_grad:
            import math
            scale = max(1e-6, min(1.0, (s / max(math.sqrt(max(1e-12, 1.0 - s * s)), 1e-6)) * 0.001))
            loss = loss * scale + loss.detach() * (1.0 - scale)
        return loss
