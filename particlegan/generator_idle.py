"""Opt-in generator idle gate. Off unless a caller sets the threshold.

The published trainer never consults this rule. A threshold of ``None`` returns
before any score is read, so the disabled step is the ordinary generator step.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

# Legacy hosts do not see the Recipe inside their update loop. GANTrainer reads
# the recipe field itself and does not consult this variable.
_CURRENT: ContextVar[float | None] = ContextVar("generator_idle_se", default=None)


def generator_idle_threshold() -> float | None:
    return _CURRENT.get()


@contextmanager
def generator_idle_scope(standard_errors: float | None):
    """Install the idle threshold for custom hosts. ``None`` leaves the rule off."""
    token = _CURRENT.set(standard_errors)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def generator_is_idle(real_logits: torch.Tensor, fake_logits: torch.Tensor,
                      standard_errors: float) -> bool:
    """Whether the critic's paired real−fake gap is unresolved at this threshold.

    Idle when the batch mean of ``real_logits - fake_logits`` is within
    ``standard_errors`` unbiased standard errors of zero. Fewer than two scores,
    or a non-finite mean or standard error, is not idle. A constant nonzero gap
    has standard error zero, so it is not idle. A constant zero gap is idle.
    """
    gap = (real_logits.detach() - fake_logits.detach()).reshape(-1).to(dtype=torch.float64)
    count = gap.numel()
    if count < 2:
        return False
    mean = gap.mean()
    variance = gap.var(unbiased=True)
    if not bool(torch.isfinite(mean)) or not bool(torch.isfinite(variance)):
        return False
    standard_error = torch.sqrt(variance / count)
    if not bool(torch.isfinite(standard_error)):
        return False
    return bool(mean.abs() <= float(standard_errors) * standard_error)


def _as_gap_pair(real_logits, fake_logits):
    if isinstance(real_logits, (list, tuple)):
        real_logits = torch.cat([tensor.detach().reshape(-1) for tensor in real_logits])
        fake_logits = torch.cat([tensor.detach().reshape(-1) for tensor in fake_logits])
    return real_logits, fake_logits


def gate_generator_update(optimizer, real_logits, fake_logits, standard_errors) -> bool:
    """Drop this generator step's gradients when ``standard_errors`` says it is idle.

    ``None`` returns immediately and does not read logits or gradients. An idle
    step clears gradients to ``None`` so Adam does not move parameters or moments.
    The caller still invokes ``optimizer.step()`` so learning-rate hooks and the
    trainer phase machine run.
    """
    if standard_errors is None:
        return False
    real_logits, fake_logits = _as_gap_pair(real_logits, fake_logits)
    if not generator_is_idle(real_logits, fake_logits, standard_errors):
        return False
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = None
    return True


def release_generator_step(optimizer, real_logits, fake_logits) -> bool:
    """Legacy-host gate. No-op, without reading logits, when the scope is unset."""
    return gate_generator_update(optimizer, real_logits, fake_logits, _CURRENT.get())
