"""Scratch Optimistic Adam step adapter; never used by the production gate.

Algorithm 1 of Daskalakis et al., *Training GANs with Optimism* (2018),
https://arxiv.org/pdf/1711.00141, applies two current Adam directions minus
the previous one. Here ``alpha=1`` is that rule and ``alpha<1`` is an explicitly
experimental damped rule. The host's ordinary Adam first updates its moments
and parameters, then this adapter adds ``-alpha*lr_t*(u_t-u_{t-1})``. The saved
previous direction is unscaled, so *both* directions use the current scheduled
learning rate. No extra gradient evaluation or training RNG draw occurs.

The optional AMSGrad extension sets the maximum uncorrected second moment
before computing the bias-corrected direction, as in PyTorch's AMSGrad. This
adaptive preconditioner is independent of training horizon; constant nominal
LR does not mean a constant effective coordinate rate. The extension and its
optimistic composition are experimental; no convergence guarantee is claimed.

This patches only ``torch.optim.Adam.step`` within a context, covering G, D,
and prior parameter groups in native, vector, image, and custom host loops.
It intentionally rejects other variants absent from the declared CPU hosts.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from unittest.mock import patch

import torch


PAPER = "https://arxiv.org/pdf/1711.00141"
AMSGRAD_PAPER = "https://arxiv.org/abs/1904.09237"
ALGORITHM = "Daskalakis-Ilyas-Syrgkanis-Zeng Algorithm 1, damped alpha extension"
PREVIOUS_DIRECTION_KEY = "optimistic_prev_direction"
UPDATE_COUNT_KEY = "optimistic_update_count"


def _validate_group(group: dict) -> None:
    if (group.get("capturable", False)
            or group.get("differentiable", False) or group.get("maximize", False)
            or group.get("decoupled_weight_decay", False)):
        raise ValueError("scratch Optimistic Adam supports the declared ordinary Adam groups only")
    if group.get("weight_decay", 0) != 0:
        raise ValueError("scratch Optimistic Adam requires zero weight decay")
    rate = group["lr"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
        raise ValueError("scratch Optimistic Adam requires a finite scalar learning rate")
    if not 0 <= group["betas"][0] < 1 or not 0 <= group["betas"][1] < 1:
        raise ValueError("invalid Adam moment rates")


def _direction(state: dict, group: dict) -> torch.Tensor:
    """Compute the paper's bias-corrected, unscaled Adam direction."""
    t = int(state["step"])
    if t <= 0:
        raise ValueError("Adam state has no completed update")
    beta1, beta2 = group["betas"]
    mhat = state["exp_avg"] / (1.0 - beta1 ** t)
    second = state["max_exp_avg_sq"] if group.get("amsgrad", False) else state["exp_avg_sq"]
    vhat = second / (1.0 - beta2 ** t)
    return mhat / (vhat.sqrt() + group["eps"])


class OptimisticAdamRecorder:
    def __init__(self, alpha: float, *, amsgrad: bool = False, diagnostics: bool = False):
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("optimism alpha must be a finite fraction in [0, 1]")
        self.alpha = float(alpha)
        if type(amsgrad) is not bool or type(diagnostics) is not bool:
            raise ValueError("amsgrad and diagnostics must be booleans")
        self.amsgrad = amsgrad
        self.diagnostics = diagnostics
        self._active = {}

    def _record_for(self, optimizer: torch.optim.Adam) -> dict:
        key = id(optimizer)
        if key not in self._active:
            self._active[key] = {
                "optimizer": optimizer,
                "ordinal": len(self._active),
                "step_calls": 0,
                "group_lrs": [],
                "group_parameter_updates": [0 for _ in optimizer.param_groups],
                "group_diagnostics": [],
                "group_prior_markers": [group.get("_comparison_prior")
                                         for group in optimizer.param_groups],
            }
        return self._active[key]

    @torch.no_grad()
    def step(self, optimizer: torch.optim.Adam, ordinary_step, closure=None):
        for group in optimizer.param_groups:
            _validate_group(group)
            if group.get("amsgrad", False) and not self.amsgrad:
                raise ValueError("undeclared AMSGrad optimizer group")
            if self.amsgrad:
                if not group.get("amsgrad", False) and any(p in optimizer.state for p in group["params"]):
                    raise ValueError("AMSGrad must be declared before the first update")
                group["amsgrad"] = True
        receipt = self._record_for(optimizer)
        if self.diagnostics:
            if closure is not None:
                raise ValueError("diagnostic mode requires explicit frozen host gradients")
            before = {p: p.detach().clone() for group in optimizer.param_groups
                      for p in group["params"] if p.grad is not None}
        # Preserve Adam's ordinary moment update, closure semantics and exact
        # alpha=0 bit pattern. The correction below uses those current moments.
        loss = ordinary_step(optimizer, closure=closure)
        receipt["step_calls"] += 1
        receipt["group_lrs"].append([float(group["lr"]) for group in optimizer.param_groups])
        for index, group in enumerate(optimizer.param_groups):
            rate = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                receipt["group_parameter_updates"][index] += 1
                if self.alpha == 0:
                    continue
                state = optimizer.state[parameter]
                current = _direction(state, group)
                previous = state.get(PREVIOUS_DIRECTION_KEY)
                if previous is None:
                    previous = torch.zeros_like(current)
                elif previous.shape != current.shape or previous.dtype != current.dtype:
                    raise ValueError("saved optimistic direction is incompatible")
                parameter.add_(current - previous, alpha=-self.alpha * rate)
                state[PREVIOUS_DIRECTION_KEY] = current.detach().clone()
                state[UPDATE_COUNT_KEY] = int(state.get(UPDATE_COUNT_KEY, 0)) + 1
        if self.diagnostics:
            diagnostics = []
            for group in optimizer.param_groups:
                parameters = [p for p in group["params"] if p in before]
                n = sum(p.numel() for p in parameters)
                gradient_square = sum(float(p.grad.double().square().sum()) for p in parameters)
                update_square = sum(float((p - before[p]).double().square().sum()) for p in parameters)
                denominators = []
                for p in parameters:
                    state = optimizer.state[p]
                    second = state["max_exp_avg_sq"] if self.amsgrad else state["exp_avg_sq"]
                    denominator = (second / (1 - group["betas"][1] ** int(state["step"]))).sqrt() + group["eps"]
                    denominators.append((float(denominator.min()), float(denominator.max())))
                diagnostics.append(dict(
                    parameters=n, gradient_rms=math.sqrt(gradient_square / n),
                    update_rms=math.sqrt(update_square / n),
                    denominator_min=min(row[0] for row in denominators),
                    denominator_max=max(row[1] for row in denominators),
                ))
            receipt["group_diagnostics"].append(diagnostics)
        return loss

    def receipt(self) -> dict:
        """Serializable application proof, including every scheduled group LR."""
        records = []
        for item in self._active.values():
            optimizer = item["optimizer"]
            group_lrs = item["group_lrs"]
            previous = []
            previous_counts = []
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter, {})
                    value = state.get(PREVIOUS_DIRECTION_KEY)
                    if value is not None:
                        previous.append(hashlib.sha256(
                            value.detach().cpu().contiguous().numpy().tobytes(),
                        ).hexdigest())
                        previous_counts.append(int(state.get(UPDATE_COUNT_KEY, 0)))
            records.append({
                "ordinal": item["ordinal"],
                "class": type(optimizer).__name__,
                "step_calls": item["step_calls"],
                "group_parameter_counts": [sum(p.numel() for p in group["params"])
                                           for group in optimizer.param_groups],
                "group_prior_markers": item["group_prior_markers"],
                "group_amsgrad": [bool(group.get("amsgrad", False)) for group in optimizer.param_groups],
                "group_parameter_updates": item["group_parameter_updates"],
                "group_diagnostics": item["group_diagnostics"],
                "group_lrs": group_lrs,
                "lr_trace_sha256": hashlib.sha256(
                    json.dumps(group_lrs, separators=(",", ":")).encode(),
                ).hexdigest(),
                "previous_direction_state_count": len(previous),
                "previous_direction_state_sha256": hashlib.sha256(
                    "".join(previous).encode(),
                ).hexdigest(),
                "optimistic_parameter_update_count": sum(previous_counts),
            })
        return {
            "scratch_common_gate_eligible": False,
            "algorithm": ALGORITHM,
            "paper": PAPER,
            "alpha": self.alpha,
            "amsgrad": self.amsgrad,
            "amsgrad_paper": AMSGRAD_PAPER if self.amsgrad else None,
            "diagnostics": self.diagnostics,
            "constant_nominal_rate_is_not_constant_per_coordinate_preconditioner": True,
            "previous_direction_unscaled": True,
            "both_directions_use_current_scheduled_lr": True,
            "additional_gradient_evaluations": 0,
            "optimizer_count": len(records),
            "optimizer_step_calls": sum(row["step_calls"] for row in records),
            "parameter_updates": sum(sum(row["group_parameter_updates"]) for row in records),
            "optimizers": records,
        }


@contextmanager
def optimistic_adam(alpha: float, *, amsgrad: bool = False, diagnostics: bool = False):
    """Patch Adam updates only for one isolated scratch episode."""
    recorder = OptimisticAdamRecorder(alpha, amsgrad=amsgrad, diagnostics=diagnostics)
    ordinary_step = torch.optim.Adam.step

    def patched_step(optimizer, closure=None):
        return recorder.step(optimizer, ordinary_step, closure)

    with patch.object(torch.optim.Adam, "step", patched_step):
        yield recorder
