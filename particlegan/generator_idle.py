"""Opt-in generator idle gate. Off unless a caller sets the threshold.

The trainer consults this rule only when its recipe enables it. A threshold of
``None`` returns before any score is read, preserving the ordinary update.

``generator_idle_mode`` selects the batch gap or a span-10 EMA of that gap.
The default is ``batch``, which is the original rule.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch

# Span 10. alpha = 2/(span+1) = 2/11. The asymptotic effective number of
# batch-means is (2-alpha)/alpha = 10. One decay, not a searched coefficient.
EMA_DECAY = 2.0 / 11.0


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

# Legacy hosts do not see the Recipe inside their update loop. GANTrainer reads
# the recipe field itself and does not consult this variable.
_CURRENT: ContextVar["IdleAccounting | None"] = ContextVar("generator_idle", default=None)


class IdleAccounting:
    """Idle decisions for one run, plus the EMA state when that mode is on."""

    def __init__(self, mode: str = "batch", standard_errors: float | None = None):
        if mode not in ("batch", "ema"):
            raise ValueError("generator_idle_mode must be 'batch' or 'ema'")
        if standard_errors is not None and (
                not _finite_number(standard_errors) or standard_errors < 0):
            raise ValueError("generator_idle_se must be None or a finite nonnegative number")
        self.mode = mode
        self.standard_errors = standard_errors
        self.decisions = 0
        self.idle_count = 0
        self._mean: float | None = None
        self._variance: float | None = None
        self._weight_sq = 0.0

    def state_dict(self) -> dict:
        return dict(schema_version=1, mode=self.mode, standard_errors=self.standard_errors,
                    ema_alpha=EMA_DECAY if self.mode == "ema" else None,
                    decisions=self.decisions, idle_count=self.idle_count,
                    mean=self._mean, variance=self._variance, weight_sq=self._weight_sq)

    def validate_state_dict(self, state) -> None:
        expected = self.state_dict()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or type(state["schema_version"]) is not int or state["schema_version"] != 1
                or state["mode"] != self.mode
                or state["standard_errors"] != self.standard_errors
                or (state["standard_errors"] is not None
                    and not _finite_number(state["standard_errors"]))
                or state["ema_alpha"] != expected["ema_alpha"]
                or any(type(state[key]) is not int or state[key] < 0
                       for key in ("decisions", "idle_count"))
                or state["idle_count"] > state["decisions"]
                or not _finite_number(state["weight_sq"])):
            raise ValueError("invalid generator idle checkpoint")
        empty = state["mean"] is None and state["variance"] is None and state["weight_sq"] == 0
        initialized = (self.mode == "ema" and state["decisions"] > 0
                       and _finite_number(state["mean"]) and _finite_number(state["variance"])
                       and state["variance"] >= 0 and 0 < state["weight_sq"] <= 1)
        if (not (empty or initialized) or (self.standard_errors is None
                and (state["decisions"] != 0 or state["idle_count"] != 0 or not empty))
                or (self.mode == "ema" and empty and state["idle_count"] != 0)):
            raise ValueError("invalid generator idle filter state")

    def load_state_dict(self, state) -> None:
        self.validate_state_dict(state)
        self.decisions, self.idle_count = state["decisions"], state["idle_count"]
        self._mean, self._variance, self._weight_sq = state["mean"], state["variance"], state["weight_sq"]

    def public(self) -> dict:
        fraction = None if self.decisions == 0 else self.idle_count / self.decisions
        return {
            "mode": self.mode,
            "standard_errors": self.standard_errors,
            "idle_count": self.idle_count,
            "decisions": self.decisions,
            "idle_fraction": fraction,
        }

    def decide(self, real_logits, fake_logits, standard_errors: float, mode: str) -> bool:
        if mode != self.mode or standard_errors != self.standard_errors:
            raise ValueError("generator idle decision differs from its bound recipe")
        if mode == "ema":
            idle = self._ema_idle(real_logits, fake_logits, standard_errors)
        elif mode == "batch":
            idle = generator_is_idle(real_logits, fake_logits, standard_errors)
        else:
            raise ValueError("generator_idle_mode must be 'batch' or 'ema'")
        self.decisions += 1
        if idle:
            self.idle_count += 1
        return idle

    def _ema_idle(self, real_logits, fake_logits, standard_errors: float) -> bool:
        """Idle when the smoothed gap is within ``standard_errors`` of its own SE.

        The filter is an EMA of the batch-mean gap and of the batch variance.
        The first step stores that batch, so its test matches the batch rule.
        Later steps use n_eff = 1 / sum(squared EMA weights). Fewer than two
        scores, or a non-finite batch, is not idle and does not move the filter.
        """
        gap = (real_logits.detach() - fake_logits.detach()).reshape(-1).to(dtype=torch.float64)
        count = int(gap.numel())
        if count < 2:
            return False
        mean = gap.mean()
        variance = gap.var(unbiased=True)
        if not bool(torch.isfinite(mean)) or not bool(torch.isfinite(variance)):
            return False
        mean_value = float(mean)
        variance_value = float(variance)
        if variance_value < 0.0 or not math.isfinite(mean_value) or not math.isfinite(variance_value):
            return False
        if self._mean is None:
            self._mean = mean_value
            self._variance = variance_value
            self._weight_sq = 1.0
        else:
            keep = 1.0 - EMA_DECAY
            self._mean = keep * self._mean + EMA_DECAY * mean_value
            self._variance = keep * self._variance + EMA_DECAY * variance_value
            self._weight_sq = keep * keep * self._weight_sq + EMA_DECAY * EMA_DECAY
        effective = 1.0 / self._weight_sq
        standard_error = math.sqrt(self._variance / (count * effective))
        if not math.isfinite(standard_error):
            return False
        return abs(self._mean) <= float(standard_errors) * standard_error


def generator_idle_threshold() -> float | None:
    accounting = _CURRENT.get()
    if accounting is None:
        return None
    return accounting.standard_errors


def generator_idle_accounting() -> IdleAccounting | None:
    return _CURRENT.get()


@contextmanager
def generator_idle_scope(standard_errors: float | None, mode: str = "batch"):
    """Install the idle rule for custom hosts. ``None`` leaves the rule off."""
    if mode not in ("batch", "ema"):
        raise ValueError("generator_idle_mode must be 'batch' or 'ema'")
    accounting = IdleAccounting(mode, standard_errors)
    token = _CURRENT.set(accounting)
    try:
        yield accounting
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


def gate_generator_update(optimizer, real_logits, fake_logits, standard_errors, *,
                          mode: str = "batch", accounting: IdleAccounting | None = None) -> bool:
    """Drop this generator step's gradients when ``standard_errors`` says it is idle.

    ``None`` returns immediately and does not read logits or gradients. An idle
    step clears gradients to ``None`` so Adam does not move parameters or moments.
    The caller still invokes ``optimizer.step()`` so learning-rate hooks and the
    trainer phase machine run.
    """
    if standard_errors is None:
        return False
    real_logits, fake_logits = _as_gap_pair(real_logits, fake_logits)
    if accounting is None:
        accounting = IdleAccounting(mode, standard_errors)
    if not accounting.decide(real_logits, fake_logits, standard_errors, mode):
        return False
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = None
    return True


def release_generator_step(optimizer, real_logits, fake_logits) -> bool:
    """Legacy-host gate. No-op, without reading logits, when the scope is unset."""
    accounting = _CURRENT.get()
    if accounting is None or accounting.standard_errors is None:
        return False
    return gate_generator_update(
        optimizer, real_logits, fake_logits, accounting.standard_errors,
        mode=accounting.mode, accounting=accounting,
    )
