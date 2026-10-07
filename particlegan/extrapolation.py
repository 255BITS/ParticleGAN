"""Extrapolation from the past for a stateless, normalized game operator.

Gidel et al. (1802.10551), equations 20–21: look ahead with the previous
direction, evaluate once there, and apply the fresh direction from the base.
Here F is BCAP's dualnorm/row-normalized field, not the raw loss gradient.
There is no averaging or optimizer step during the temporary lookahead.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch

from .optim.dualnorm import polar_factor, spectral_capped_direction


@torch.no_grad()
def stateless_directions(optimizer):
    """Read the existing zero-momentum step rule without changing its state.

    Table ownership comes from the current generator draw; all other rows
    have zero direction. The cached direction retains that previous mask.
    """
    if optimizer.family not in ("dualnorm", "sgda") or optimizer.momentum:
        raise ValueError("past extrapolation requires stateless dualnorm or sgda")
    result = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            gradient, eps = parameter.grad, group["eps"]
            direction = torch.zeros_like(parameter)
            if gradient is not None:
                if gradient.is_sparse or not bool(torch.isfinite(gradient).all()):
                    raise ValueError("game gradients must be finite and dense")
                if group["algorithm"] == "sgda":
                    direction.copy_(gradient)
                elif group["algorithm"] == "rownorm":
                    rows = optimizer.sampled_rows_for(parameter)
                    if rows is None:
                        raise ValueError("game direction requires actual sampled rows")
                    selected = gradient[rows]
                    direction[rows] = selected / (selected.norm(dim=1, keepdim=True) + eps)
                elif group.get("network_update") == "spectral_capped":
                    direction.copy_(spectral_capped_direction(gradient, group["network_gradient_scale"]))
                elif parameter.ndim == 2:
                    if not bool(gradient.norm() < eps):
                        direction.copy_(polar_factor(gradient) * math.sqrt(max(1., parameter.shape[0] / parameter.shape[1])))
                else:
                    direction.copy_(gradient / (gradient.norm() + eps))
            result[parameter] = direction
    return result


class PastExtrapolation:
    """Checkpointed normalized directions; the first lookahead is zero."""

    def __init__(self, parameters, optimizers):
        self.parameters, self.optimizers = dict(parameters), tuple(optimizers)
        self.previous = None

    @contextmanager
    def lookahead(self):
        base = {name: value.detach().clone() for name, value in self.parameters.items()}
        try:
            if self.previous is not None:
                names = {parameter: name for name, parameter in self.parameters.items()}
                with torch.no_grad():
                    for optimizer in self.optimizers:
                        for group in optimizer.param_groups:
                            for parameter in group["params"]:
                                parameter.add_(self.previous[names[parameter]], alpha=-group["lr"])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(base[name])

    def fresh_directions(self):
        values = {}
        for optimizer in self.optimizers:
            values.update(stateless_directions(optimizer))
        return {name: values[parameter] for name, parameter in self.parameters.items()}

    def state_dict(self):
        return deepcopy({"schema": 1, "operator": "stateless_normalized", "previous": self.previous})

    def check_state(self, state, completed_steps):
        if (not isinstance(state, dict) or set(state) != {"schema", "operator", "previous"}
                or state["schema"] != 1 or state["operator"] != "stateless_normalized"):
            raise ValueError("invalid past extrapolation checkpoint")
        previous = state["previous"]
        if completed_steps == 0:
            if previous is not None:
                raise ValueError("initial past extrapolation cache must be empty")
        elif not isinstance(previous, dict) or previous.keys() != self.parameters.keys():
            raise ValueError("past extrapolation cache topology mismatch")
        else:
            for name, parameter in self.parameters.items():
                value = previous[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError("invalid past extrapolation direction")

    def load_state_dict(self, state):
        self.previous = (None if state["previous"] is None else
                         {name: value.detach().to(self.parameters[name]).clone()
                          for name, value in state["previous"].items()})
