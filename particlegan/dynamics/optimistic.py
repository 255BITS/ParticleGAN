"""Optimistic Adam at the group's existing learning rate and betas.

Daskalakis, Ilyas, Syrgkanis, and Zeng, *Training GANs with Optimism*
(2018), Algorithm 1, with the paper's optimism coefficient 1. The Adam
direction at step t is the bias-corrected first moment divided by the
bias-corrected second-moment denominator::

    m_t = m̂_t / (sqrt(v̂_t) + ε)

The parameter update is ``-lr * (2 m_t - m_{t-1})``, and ``m_0 = 0``.
Ordinary Adam already applies ``-lr * m_t``, so the correction added
here is ``-lr * (m_t - m_{t-1})``. ``lr``, ``β``, and ``ε`` stay the
ones on the Adam group (K3P: network 0.00425, prior 0.0085, betas
``(0, 0.999)``, ``ε = 1e-8``). No extra forward, no new coefficient.

Hypothesis, written before the constant-LR screen: the neighbor hop in
#177 is a rotational game cycle, and this correction damps that cycle
while the learning rate stays at the pre-anneal value.

Installed on ``Adam.step`` before the probe captures it, so the critic,
the generator, and the particle prior all take the same correction.
``K3PCriticAdam`` and ``K3PGeneratorAdam`` call that step through
``_adam_step``. With the flag unset this module is not installed.
"""
from __future__ import annotations

import atexit
import json

import torch

receipt = {"mechanism": "optimistic", "steps": 0, "corrections": 0}
_ORIGINAL = None
_LOGGED = False
_ATEXIT = False
_PREV = "optimistic_prev_direction"


def uninstall() -> None:
    """Restore ``Adam.step``. Used by tests; training processes leave it installed."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    torch.optim.Adam.step = _ORIGINAL
    _ORIGINAL = None


def install() -> None:
    """Replace ``Adam.step`` until process exit. Call before anything captures it."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        return
    step = torch.optim.Adam.step
    if getattr(step, "hooked", False):
        step = step.__wrapped__
    _ORIGINAL = step
    torch.optim.Adam.step = _step
    _step._k3p_optimistic = True
    global _ATEXIT
    if not _ATEXIT:
        atexit.register(_emit)
        _ATEXIT = True
    print(json.dumps({
        "event": "dynamics",
        "name": "optimistic",
        "setting": "Daskalakis Algorithm 1, alpha=1, update -lr*(2*m_t - m_{t-1}) on D, G, and prior",
    }), flush=True)


def _emit() -> None:
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def _count(state) -> int:
    value = state["step"]
    return int(value.item() if torch.is_tensor(value) else value)


def _direction(state, group) -> torch.Tensor:
    """Bias-corrected Adam direction, the paper's ``m_t``."""
    t = _count(state)
    if t <= 0:
        raise RuntimeError("optimistic Adam saw a step count of 0")
    beta1, beta2 = group["betas"]
    # beta1 is 0 on this recipe. 0**t is 0 for t > 0, and 0**0 would be 1.
    b1 = 0.0 if float(beta1) == 0.0 else float(beta1) ** t
    b2 = 0.0 if float(beta2) == 0.0 else float(beta2) ** t
    mhat = state["exp_avg"] / (1.0 - b1)
    second = state["max_exp_avg_sq"] if group.get("amsgrad", False) else state["exp_avg_sq"]
    vhat = second / (1.0 - b2)
    return mhat / (vhat.sqrt() + group["eps"])


def _step(optimizer, closure=None):
    """Ordinary Adam, then the optimistic correction on every parameter that stepped."""
    global _LOGGED
    for group in optimizer.param_groups:
        decay = group.get("weight_decay", 0.0) or 0.0
        if decay != 0.0:
            raise RuntimeError("optimistic Adam expects zero weight decay")
    loss = _ORIGINAL(optimizer, closure)
    with torch.no_grad():
        for group in optimizer.param_groups:
            rate = float(group["lr"])
            sign = -1.0 if not group.get("maximize", False) else 1.0
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = optimizer.state[parameter]
                current = _direction(state, group)
                previous = state.get(_PREV)
                if previous is None:
                    previous = torch.zeros_like(current)
                # Adam already applied sign*lr*current. This adds the other half.
                parameter.add_(current - previous, alpha=sign * rate)
                state[_PREV] = current.detach().clone()
                receipt["corrections"] += 1
    receipt["steps"] += 1
    if not _LOGGED:
        _LOGGED = True
        print(json.dumps({
            "event": "dynamics_step",
            "name": "optimistic",
            "steps": 1,
            "group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
            "group_betas": [list(group["betas"]) for group in optimizer.param_groups],
        }), flush=True)
    return loss
