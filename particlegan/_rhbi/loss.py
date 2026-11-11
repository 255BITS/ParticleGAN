"""Relativistic Hellinger-Born-Infeld (RHBI) GAN loss adapter."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ..gan_loss import GANLoss
from .critic import _lookup_rhbi_capture_full
from .primitive import RHBILoss


class RHBIGANLoss(GANLoss):
    """``GANLoss`` subclass pairing calibrated log-odds critic training with ``RHBILoss``."""

    def __init__(
        self,
        loss: str = "non_saturating",
        *,
        labels: Tuple[float, float, float] = (0.0, 1.0, 1.0),
        weight: float = 1.0,
        rho: float = 2.0,
        tau: float = 2.0,
        eps: float = 0.1,
        iters: int = 8,
        c_max: float = 20.0,
        beta_ell: float = 0.1,
        nn_quantile: float = 0.25,
        critic_weight: float = 0.0,
        self_coupling: str = "hollow",
        relativistic_weight: float = 0.0,
        rapidity_screening: bool = True,
        scale_log_odds: bool = False,
        conditional_fibre: str = "joint",
        record: Any = None,
    ) -> None:
        super().__init__(loss, labels=labels)
        if (
            isinstance(critic_weight, bool)
            or not isinstance(critic_weight, (int, float))
            or not math.isfinite(critic_weight)
            or critic_weight < 0.0
        ):
            raise ValueError("critic_weight must be finite and nonnegative")
        if type(conditional_fibre) is not str or conditional_fibre not in ("joint", "slack"):
            raise ValueError("conditional_fibre must be 'joint' or 'slack'")
        self.critic_weight = float(critic_weight)
        self.conditional_fibre = conditional_fibre
        self.record = record
        self.rhbi = RHBILoss(
            weight=weight,
            rho=rho,
            tau=tau,
            eps=eps,
            iters=iters,
            c_max=c_max,
            beta_ell=beta_ell,
            nn_quantile=nn_quantile,
            rapidity_screening=rapidity_screening,
            self_coupling=self_coupling,
            relativistic_weight=relativistic_weight,
            scale_log_odds=scale_log_odds,
        )
        self.last_metrics: Dict[str, float] = {}

    def _resolve_pair_full(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, bool]:
        x_f, cond_f, rec_f, cc_f = _lookup_rhbi_capture_full(fake_logits, self.record)
        x_r, cond_r, rec_r, cc_r = _lookup_rhbi_capture_full(
            real_logits, rec_f if rec_f is not None else self.record
        )
        rec = rec_f if rec_f is not None else (rec_r if rec_r is not None else self.record)
        has_cont_cond = bool(cc_f or cc_r)
        if x_f is None or x_r is None:
            x_f = fake_logits.reshape(fake_logits.shape[0], -1)
            x_r = real_logits.reshape(real_logits.shape[0], -1)
            cond_f = None
            cond_r = None
            has_cont_cond = False
        return x_f, x_r, cond_f, cond_r, rec, has_cont_cond

    def _resolve_pair(
        self,
        fake_logits: torch.Tensor,
        real_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any]:
        x_f, x_r, cond_f, cond_r, rec, _ = self._resolve_pair_full(fake_logits, real_logits)
        return x_f, x_r, cond_f, cond_r, rec

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        base = super().d_loss(real_logits, fake_logits)
        if self.critic_weight == 0.0:
            return base
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond = self._resolve_pair_full(fake_logits, real_logits)
        if self.conditional_fibre == "slack" and has_cont_cond:
            return base
        slack_reg = self.rhbi.critic_slack_loss(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            critic_weight=self.critic_weight,
            record=rec,
            cond_fake=cond_f,
            cond_real=cond_r,
        )
        return base + slack_reg

    def g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor | None = None) -> torch.Tensor:
        if real_logits is None:
            if self.rhbi.weight == 0.0:
                return super().g_loss(fake_logits, None)
            raise ValueError("RHBIGANLoss requires real_logits in g_loss when rhbi_weight > 0")
        if self.rhbi.weight == 0.0:
            return super().g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, rec, has_cont_cond = self._resolve_pair_full(fake_logits, real_logits)
        if self.conditional_fibre == "slack" and has_cont_cond:
            b_f = fake_logits.shape[0]
            b_r = real_logits.shape[0]
            if b_f == b_r:
                f_flat = fake_logits.reshape(b_f, -1).mean(dim=1) if fake_logits.ndim > 1 else fake_logits.reshape(b_f)
                r_flat = (
                    real_logits.detach().reshape(b_r, -1).mean(dim=1)
                    if real_logits.ndim > 1
                    else real_logits.detach().reshape(b_r)
                )
                return self.rhbi.weight * F.softplus(r_flat - f_flat).mean()
            return self.rhbi.weight * super().g_loss(fake_logits, real_logits)
        allow_ell = torch.is_grad_enabled() and not getattr(rec, "_in_protected_probe", False)
        loss_val, _ = self.rhbi(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            record=rec,
            symmetric=False,
            allow_ell_update=allow_ell,
            cond_fake=cond_f,
            cond_real=cond_r,
            return_metrics=False,
        )
        return loss_val

    def joint_g_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        if not fake_logits.requires_grad or not real_logits.requires_grad:
            raise ValueError("joint_g_loss requires gradients on both generator and encoder streams")
        if self.rhbi.weight == 0.0:
            return super().joint_g_loss(fake_logits, real_logits)
        x_f, x_r, cond_f, cond_r, _rec, _has_cont_cond = self._resolve_pair_full(fake_logits, real_logits)
        allow_ell = torch.is_grad_enabled() and not getattr(_rec, "_in_protected_probe", False)
        loss_val, _ = self.rhbi.joint_relativistic_loss(
            x_f,
            x_r,
            fake_logits,
            real_logits,
            record=_rec,
            allow_ell_update=allow_ell,
            cond_fake=cond_f,
            cond_real=cond_r,
            return_metrics=False,
        )
        return loss_val
