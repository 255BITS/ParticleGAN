"""Apply BISPF's earned thermal scale to the finite generator/prior move.

The helpers have no controller, random stream, or persistent state. The caller
checkpoints the existing BISPF critic observer and underlying optimizer. Use
the context for the complete generator objective, then pass its yielded scale
to ``thermal_optimizer_step``. Critic training retains its original law.
"""
from contextlib import contextmanager
import math

import torch


def _scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("thermal displacement scale must be finite and in [0, 1]")
    return float(value)


@contextmanager
def unscaled_generator_gradients(record, device):
    """Yield the earned scale while disabling both G-side gradient coolers.

    BISPF's critic forward hook reads ``record._thermal_scale``; SBNP reads a
    device-local cache. Restore both exactly, including an absent cache entry,
    on normal exit, nested exit, or an exception. No residual observation or
    critic/optimizer counter changes here.
    """
    from .bispf import _LAST_THERMAL_SCALE

    earned = getattr(record, "_thermal_scale", None)
    scale = _scale(earned)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    missing = object()
    cached = _LAST_THERMAL_SCALE.get(device, missing)
    record._thermal_scale = 1.0
    _LAST_THERMAL_SCALE[device] = 1.0
    try:
        yield scale
    finally:
        record._thermal_scale = earned
        if cached is missing:
            _LAST_THERMAL_SCALE.pop(device, None)
        else:
            _LAST_THERMAL_SCALE[device] = cached


@torch.no_grad()
def thermal_optimizer_step(optimizer, scale, *, collect_stats=False):
    """Take one ordinary optimizer step and scale its actual finite movement.

    Direction blending and normalization finish first. Every optimized G/E and
    prior tensor then becomes ``old + scale * (base - old)``. Rates, gradients,
    optimizer clocks/history, sampled-row consumption, and prior policy remain
    those of the underlying optimizer. In particular this is a change in the
    applied prior update, including SBNP's relative force, rather than another
    gradient multiplier. A scale of one takes the exact original step.

    Diagnostics describe the realized tensors, after floating-point rounding,
    and are returned to the caller for its external trace. They do not alter
    continuation state. This helper intentionally has no closure interface.
    """
    scale = _scale(scale)
    if type(collect_stats) is not bool:
        raise TypeError("collect_stats must be a boolean")
    groups = optimizer.param_groups
    if any(group.get("role", group.get("forge_role")) == "critic" for group in groups):
        raise ValueError("thermal displacement applies only to generator, encoder and prior groups")
    result = {"thermal_scale": scale, "cooled": scale < 1.0}
    if scale == 1.0 and not collect_stats:
        optimizer.step()
        return result

    parameters = []
    seen = set()
    for group in groups:
        role = group.get("role", group.get("forge_role", "generator"))
        role = "prior" if role == "table" else role
        for parameter in group["params"]:
            if id(parameter) not in seen:
                parameters.append((role, parameter, parameter.detach().clone()))
                seen.add(id(parameter))
    gradients = {}
    if collect_stats:
        for role, parameter, _ in parameters:
            if parameter.grad is not None:
                gradients[role] = gradients.get(role, 0.0) + float(parameter.grad.double().square().sum())

    optimizer.step()
    roles = {}
    for role, parameter, original in parameters:
        if collect_stats:
            base_squared = float((parameter.double() - original.double()).square().sum())
        if scale < 1.0:
            parameter.copy_(torch.lerp(original, parameter, scale))
        if collect_stats:
            stats = roles.setdefault(role, {"gradient_squared_l2": gradients.get(role, 0.0),
                "base_squared_l2": 0.0, "applied_squared_l2": 0.0})
            stats["base_squared_l2"] += base_squared
            stats["applied_squared_l2"] += float((parameter.double() - original.double()).square().sum())
    if collect_stats:
        for stats in roles.values():
            base = stats["base_squared_l2"]
            stats["applied_to_base_l2"] = math.sqrt(stats["applied_squared_l2"] / base) if base else None
        result["roles"] = roles
    return result
