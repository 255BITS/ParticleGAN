"""Alternating G own-secant rank-one positive-curvature resolvent.

After the ordinary D update (bounded by PR #82's own-curvature rule), G makes
one ordinary Adam proposal against the realized D. A same-batch, same-noise
replay measures the own-field secant pair ``s=theta_proposal-theta_base`` and
``y=g_proposal-g_base``. When ``s dot y > 0``, the positive-semidefinite rank-one
curvature ``y y.T/(s dot y)`` is added to the inverse Adam metric. Its inverse
applied to the ordinary proposal is calculated by Sherman--Morrison, without
materializing any dense parameter matrix. Nonpositive or nonfinite curvature
falls back to PR #82's G scalar rho bound. Optimizer moments advance once.
"""

from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from reports.toy100 import alternating_curvature_scratch as adapter


METHOD = "alternating_positive_own_secant_resolvent"


def positive_secant_direction(s, y, metric, rho, fallback_bound):
    """Return contraction for a one-pair PSD own-curvature update."""
    s, y, metric = (x.flatten().double() for x in (s, y, metric))
    if s.shape != y.shape or s.shape != metric.shape or not bool((metric > 0).all()):
        raise ValueError("invalid secant/Adam metric shapes")
    if not all(bool(torch.isfinite(x).all()) for x in (s, y, metric)):
        raise FloatingPointError("nonfinite secant/Adam metric")
    curvature = float(torch.dot(s, y))
    py = metric * y
    ypy = float(torch.dot(y, py))
    if math.isfinite(curvature) and curvature > 0 and math.isfinite(ypy):
        denominator = curvature + ypy
        corrected = s - py * (curvature / denominator)
        if bool(torch.isfinite(corrected).all()):
            base_norm = float(torch.sqrt((s.square() / metric).sum()))
            final_norm = float(torch.sqrt((corrected.square() / metric).sum()))
            if final_norm > base_norm * (1 + 1e-10):
                raise RuntimeError("positive secant increased the Adam-metric step norm")
            return corrected, dict(rule="positive_rank_one", curvature=curvature,
                                   y_metric_y=ypy, metric_norm_ratio=(final_norm / base_norm
                                   if base_norm else 0.0), factor=1.0)
    factor = min(1., fallback_bound / rho) if rho > 0 else 1.
    return s * factor, dict(rule="scalar_curvature_fallback", curvature=curvature,
                            y_metric_y=ypy, metric_norm_ratio=factor, factor=factor)


class PositiveSecantRecorder(adapter.BothBoundRecorder):
    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if not self.passthrough and self.phase is not None and self.optimizers is not None:
            row = self.rows[optimizer]
            rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
            if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
                raise ValueError("invalid applied Adam rate")
            if "rates" in row and row["rates"] != rates:
                raise RuntimeError("nominal Adam rate changed during fixed-rate controller")
            row["rates"] = rates
            row["rate_observations"] = row.get("rate_observations", 0) + 1
        if self.phase != 2 or self.optimizers is None or optimizer is not self.optimizers[1]:
            return super().step(optimizer, ordinary_step, closure)
        if closure is not None:
            raise RuntimeError("closure outside declared game update")
        self.rows[optimizer]["calls"] += 1
        params = self._params(optimizer)
        g1 = [p.grad.detach().clone() for p in params]
        rho = adapter._rho(self.g_base, self.g1, self.gg0, g1, self.metric_g)
        sizes = [p.numel() for p in params]
        s = torch.cat([(new - base).flatten().double()
                       for base, new in zip(self.g_base, self.g1)])
        y = torch.cat([(later - g0).flatten().double()
                       for g0, later in zip(self.gg0, g1)])
        metric = torch.cat([value.flatten().double() for value in self.metric_g])
        direction, diagnostic = positive_secant_direction(
            s, y, metric, rho, self.curvature_bound)
        for parameter, base, block in zip(params, self.g_base, direction.split(sizes)):
            parameter.copy_((base.double() + block.reshape_as(parameter)).to(parameter.dtype))
        self.row["g"] = dict(rho=rho, **diagnostic)
        return None

    def receipt(self):
        value = super().receipt()
        optimizer_rows = []
        for optimizer in self.optimizers:
            row = self.rows[optimizer]
            steps = [int(optimizer.state[p]["step"])
                     for group in optimizer.param_groups for p in group["params"]]
            expected_step = self.start_step + self.outer_steps
            if (row["calls"] != 3 * self.outer_steps
                    or row.get("rate_observations") != row["calls"]
                    or not steps or any(step != expected_step for step in steps)):
                raise RuntimeError("gradient calls or Adam moments do not match outer updates")
            optimizer_rows.append(dict(role=row["role"], gradient_calls=row["calls"],
                                       nominal_rates=list(row["rates"]),
                                       rate_observations=row["rate_observations"],
                                       moment_step_min=min(steps), moment_step_max=max(steps)))
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     positive_secant_updates=sum(row["g"]["rule"] == "positive_rank_one"
                                                 for row in self.records),
                     fallback_updates=sum(row["g"]["rule"] == "scalar_curvature_fallback"
                                          for row in self.records),
                     g_positive_metric_contraction=True,
                     optimizers=optimizer_rows)
        return value


@contextmanager
def alternating_positive_secant(task="mode_hold", **options):
    with patch.object(adapter, "BothBoundRecorder", PositiveSecantRecorder):
        with adapter.alternating_curvature(task=task, bound_d=True,
                                           curvature_bound=.25,
                                           d_curvature_bound=2., **options) as value:
            yield value
