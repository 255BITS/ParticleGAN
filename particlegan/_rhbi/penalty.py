"""RHBI critic penalty: smooth Born-Infeld endpoint Lipschitz cap."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ..grad_regularizers import CriticStepRecord, GradientPenalty, _score_scalar
from ..k3p import CriticPenalty as _K3PCriticPenalty, K3PCriticAdam, _first_output
from .critic import RHBICriticAdam, RHBIStepRecord, _attach_rhbi_critic_hook


class _RHBIPenaltyKernel:
    """Smooth C^infinity Born-Infeld endpoint gradient-norm cap with zero shape gates.

    Enforces ``||grad_x D(x)||_2 <= kappa`` via smooth squared softplus on active
    real and fake endpoints so the relativistic slack tunneling gain ``0.5 * tau``
    matches the unit linear-tail slope of the physical Born-Infeld transport cost.
    """

    _grad_norm = staticmethod(GradientPenalty._grad_norm)

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 0.0,
        ebi_beta: float = 30.0,
        real_weight: float = 1.0,
        fake_weight: float = 1.0,
        record: CriticStepRecord | None = None,
    ) -> None:
        for name, val, pos in (
            ("coeff", coeff, False),
            ("kappa", kappa, False),
            ("ebi_beta", ebi_beta, True),
            ("real_weight", real_weight, False),
            ("fake_weight", fake_weight, False),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or (
                val <= 0.0 if pos else val < 0.0
            ):
                raise ValueError(f"{name} must be finite and {'positive' if pos else 'nonnegative'}")
        if type(lazy_k) is not int or lazy_k < 1:
            raise ValueError("lazy_k must be a positive integer")
        if anchor_weight != 0.0:
            raise ValueError("RHBI does not use an EMA critic anchor (require reg_anchor_weight=0)")
        self.arm = "rhbi"
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.anchor_weight = 0.0
        self.ebi_beta = float(ebi_beta)
        self.real_weight = float(real_weight)
        self.fake_weight = float(fake_weight)
        self.record = RHBIStepRecord() if record is None else record

    def penalty(
        self,
        live,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int,
        collect_stats: bool = False,
        *,
        allow_concat: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if (
            self.coeff == 0.0
            or (self.real_weight == 0.0 and self.fake_weight == 0.0)
            or (self.lazy_k > 1 and step % self.lazy_k != 0)
        ):
            zero = x_real.new_zeros(())
            return zero, ({"applied": False, "pen": 0.0, "center": self.kappa} if collect_stats else {})

        self.record.calls += 1
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff

        self.record._in_penalty = True
        try:
            if (
                allow_concat
                and self.real_weight > 0.0
                and self.real_weight == self.fake_weight
                and x_real.shape == x_fake.shape
                and x_real.dtype == x_fake.dtype
            ):
                x_both = torch.cat((x_real.detach(), x_fake.detach()), dim=0).requires_grad_(True)
                d_out = live(x_both)
                grad = torch.autograd.grad(
                    outputs=_score_scalar(d_out),
                    inputs=x_both,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                norm_all = (grad.flatten(1).square().sum(dim=1) + 1e-12).sqrt()
                b_n = x_real.shape[0]
                cap_r = F.softplus(norm_all[:b_n] - self.kappa, beta=self.ebi_beta).square().mean()
                cap_f = F.softplus(norm_all[b_n:] - self.kappa, beta=self.ebi_beta).square().mean()
                total = self.real_weight * (cap_r + cap_f)
                norms = [norm_all.detach()] if collect_stats else []
            else:
                total = x_real.new_zeros(())
                norms = []
                for weight, inputs in ((self.real_weight, x_real), (self.fake_weight, x_fake)):
                    if weight == 0.0:
                        continue
                    x_req = inputs.detach().clone().requires_grad_(True)
                    d_out = live(x_req)
                    grad = torch.autograd.grad(
                        outputs=_score_scalar(d_out),
                        inputs=x_req,
                        create_graph=True,
                        only_inputs=True,
                    )[0]
                    norm = (grad.flatten(1).square().sum(dim=1) + 1e-12).sqrt()
                    cap = F.softplus(norm - self.kappa, beta=self.ebi_beta).square().mean()
                    total = total + weight * cap
                    if collect_stats:
                        norms.append(norm.detach())
        finally:
            self.record._in_penalty = False

        pen = 0.5 * coefficient * total
        if not collect_stats:
            return pen, {}
        ell_t = getattr(self.record, "_rhbi_ell_t", None)
        norm_mean_t = (norms[0].mean() if len(norms) == 1 else torch.cat(norms).mean()) if norms else pen.new_zeros(())
        if ell_t is not None and ell_t.device == pen.device:
            pen_v, mean_norm, ell_v = torch.stack(
                (pen.detach().float(), norm_mean_t.float(), ell_t.detach().float())
            ).tolist()
        else:
            pen_v, mean_norm = torch.stack((pen.detach().float(), norm_mean_t.float())).tolist()
            ell_v = float(getattr(self.record, "rhbi_ell_ema", 0.0))
        return pen, {
            "applied": True,
            "pen": float(pen_v),
            "center": self.kappa,
            "phase": self.arm,
            "mean_grad_norm": float(mean_norm),
            "rhbi_ell": float(ell_v),
        }


class RHBICriticPenalty(_K3PCriticPenalty):
    """Recipe-facing RHBI critic penalty paired with ``recipe.make_critic_optimizer``."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides) -> None:
        from ..optim.dualnorm import NormalizedOptimizer
        from ..tensorflow_adam import TensorFlowV1Adam

        if not (
            isinstance(optimizer, (RHBICriticAdam, K3PCriticAdam))
            or (
                type(optimizer) in (torch.optim.Adam, TensorFlowV1Adam)
                and getattr(optimizer, "recipe_optimizer_family", None) == "adam"
            )
            or (isinstance(optimizer, NormalizedOptimizer) and hasattr(optimizer, "record"))
        ):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")
        self.recipe, self.optimizer, self.critic = recipe, optimizer, optimizer.critic
        _attach_rhbi_critic_hook(self.critic, optimizer.record)
        options = recipe._penalty_options(**penalty_overrides)
        self.regularizer = _RHBIPenaltyKernel(record=optimizer.record, **options)
        self.initial_coeff = self.regularizer.coeff
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Any] = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}
        self._verified_critics: set[int] = set()
        self._has_batchnorm = any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.critic.modules()
        )

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        from ..recipe_schedules import apply_penalty_schedule

        cid = id(critic)
        if cid not in self._verified_critics:
            self._ema_view(critic)
            self._verified_critics.add(cid)
        if getattr(self, "recipe", None) is not None:
            apply_penalty_schedule(self.optimizer.record.observed_steps, self.recipe, self)
        output = self.output
        allow_concat = (
            not self._has_batchnorm
            and not condition
            and not condition_kwargs
            and getattr(critic, "std", 0.0) == 0.0
        )

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))

        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(
            live, x_real, x_fake, step, self.collect_stats, allow_concat=allow_concat
        )
        self.last_stats = stats
        if getattr(getattr(self, "recipe", None), "critic_step_mode", "none") == "finite_cap":
            real, fake = x_real.detach().clone(), x_fake.detach().clone()

            def finite_cap():
                return torch.maximum(
                    self.regularizer._grad_norm(live, real).max(),
                    self.regularizer._grad_norm(live, fake).max(),
                )

            self.optimizer.bind_cap_panel(finite_cap, self.regularizer.kappa)
        return penalty

    def diagnostics(self) -> Dict[str, Any]:
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(
            rhbi_ell=float(getattr(record, "rhbi_ell_ema", 0.0)),
            rhbi_ell_initialized=bool(getattr(record, "rhbi_ell_initialized", False)),
        )
        return out
