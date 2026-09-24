"""PR84 alternating update with Daskalakis optimistic Adam, alpha=1.

The smoothed generator critic and the fixed own-curvature bounds stay as in
PR84. The only change is the committed Adam displacement: after the ordinary
moment update, parameters move by ``-(1+alpha)*lr*u_t + alpha*lr*u_{t-1}``
with ``alpha=1`` (Algorithm 1, Training GANs with Optimism). The previous
direction is unscaled and starts at zero, so both terms use the current
group rate. No extra gradient, no loss term, no target, no clip ladder.
"""

from contextlib import contextmanager
import math

import torch

from reports.toy100.optimistic_adam_scratch import (
    PREVIOUS_DIRECTION_KEY, UPDATE_COUNT_KEY, _direction, _validate_group,
)
from reports.toy100.pr84_smoothed_candidate import METHOD as PR84_METHOD
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


METHOD = "pr84_alternating_optimistic_adam"
PAPER = "https://arxiv.org/pdf/1711.00141"
ALPHA = 1.0


@torch.no_grad()
def optimistic_adam_step(optimizer, ordinary_step, closure=None, *, alpha=ALPHA):
    """Ordinary Adam, then the paper's optimistic correction at the current rate."""
    if alpha != ALPHA:
        raise ValueError("this lane fixes optimism at the paper coefficient 1")
    for group in optimizer.param_groups:
        _validate_group(group)
        if group.get("amsgrad", False):
            raise ValueError("optimistic game step requires ordinary Adam")
    loss = ordinary_step(optimizer, closure=closure)
    updates = 0
    for group in optimizer.param_groups:
        rate = float(group["lr"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = optimizer.state[parameter]
            current = _direction(state, group)
            previous = state.get(PREVIOUS_DIRECTION_KEY)
            if previous is None:
                previous = torch.zeros_like(current)
            elif previous.shape != current.shape or previous.dtype != current.dtype:
                raise ValueError("saved optimistic direction is incompatible")
            parameter.add_(current - previous, alpha=-alpha * rate)
            state[PREVIOUS_DIRECTION_KEY] = current.detach().clone()
            state[UPDATE_COUNT_KEY] = int(state.get(UPDATE_COUNT_KEY, 0)) + 1
            updates += 1
    return loss, updates


@contextmanager
def pr84_optimistic_game(*, task="mode_hold", start_step=0, optimism=True):
    """PR84 host with optional optimistic Adam on each committed player step."""
    with pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
        recorder.optimism = bool(optimism)
        recorder.optimistic_updates = 0
        original_step = recorder.step
        original_receipt = recorder.receipt

        def step(optimizer, ordinary_step, closure=None):
            if not recorder.optimism or recorder.passthrough:
                return original_step(optimizer, ordinary_step, closure)

            def committed(opt, closure=None):
                loss, updates = optimistic_adam_step(opt, ordinary_step, closure, alpha=ALPHA)
                recorder.optimistic_updates += updates
                return loss

            return original_step(optimizer, committed, closure)

        def receipt():
            value = original_receipt()
            value.update(
                method=METHOD if recorder.optimism else PR84_METHOD,
                scratch_optimizer_policy=METHOD if recorder.optimism else PR84_METHOD,
                optimism=recorder.optimism,
                optimism_alpha=ALPHA if recorder.optimism else 0.0,
                optimism_paper=PAPER,
                additional_gradient_evaluations=0,
                optimistic_parameter_updates=recorder.optimistic_updates,
                purity="GAN dynamics only — no coverage/likelihood term",
            )
            return value

        recorder.step = step
        recorder.receipt = receipt
        yield recorder, source


def direction_gap(optimizer) -> float:
    """RMS of stored previous Adam directions; finite check for receipts."""
    total = 0.0
    count = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            previous = optimizer.state.get(parameter, {}).get(PREVIOUS_DIRECTION_KEY)
            if previous is None:
                continue
            total += float(previous.detach().double().square().sum())
            count += previous.numel()
    if not count:
        return 0.0
    gap = math.sqrt(total / count)
    if not math.isfinite(gap):
        raise FloatingPointError("nonfinite optimistic direction")
    return gap
