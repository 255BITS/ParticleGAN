"""The critic gradient penalty kernel (K3P) and the per-critic step record it reads.

Users do not build these directly: ``recipe.make_critic_penalty(opt_d)``
pairs a ``GradientPenalty`` with the ``CriticStepRecord`` kept by the
recipe's critic optimizer. See ``docs/k3p.md`` for the formulation.
"""

from typing import Any, Dict, Optional, Tuple
import math

import torch


def _score_scalar(logits):
    """Sum of per-image logit means.

    One logit per image matches ``logits.sum()``: other batch rows do not
    change an image's gradient. Extra logits are averaged inside the image so
    a spatial map does not multiply that gradient by its number of locations.
    """
    if logits.ndim < 2:
        return logits.sum()
    return logits.flatten(1).mean(dim=1).sum()


class CriticStepRecord:
    """Per-critic step counters: penalty calls, critic steps and the applied LR (diagnostic).

    One record belongs to one critic optimizer, which calls ``record_step``
    after every step. The penalty counts its calls and reads the completed
    step count for lazy application.
    """

    KEYS = ("lr_max", "lr_last", "calls", "observed_steps")
    # Checkpoints from the EMA-anchor formulation also carry this key; it is dropped on load.
    _DROPPED_KEYS = ("anchor_started",)

    def __init__(self) -> None:
        self.lr_max = 0.0
        self.lr_last: Optional[float] = None
        self.calls = 0
        self.observed_steps = 0

    def record_step(self, optimizer_or_lr) -> None:
        if isinstance(optimizer_or_lr, (int, float, torch.Tensor)):
            lr = float(optimizer_or_lr)
        else:
            lr = max(float(g["lr"]) for g in optimizer_or_lr.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"lr_max": self.lr_max, "lr_last": self.lr_last, "calls": self.calls,
                "observed_steps": self.observed_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if isinstance(state, dict):
            state = {key: value for key, value in state.items() if key not in self._DROPPED_KEYS}
        if not isinstance(state, dict) or set(state) != set(self.KEYS):
            keys = sorted(state) if isinstance(state, dict) else state
            raise ValueError(f"k3p state keys {keys} != expected {sorted(self.KEYS)}")
        values = (float(state["lr_max"]), None if state["lr_last"] is None else float(state["lr_last"]),
                  int(state["calls"]), int(state["observed_steps"]))
        self.lr_max, self.lr_last, self.calls, self.observed_steps = values


class GradientPenalty:
    """K3P critic gradient penalty: zero-centred R1 on reals plus a one-sided cap on fakes.

    ``pen = coeff/2 * (r1 + fake_cap)`` with

    * ``r1 = mean ||g(r)||^2 / d``, which pulls the critic's slope at the
      reals to 0;
    * ``fake_cap = mean relu(||g(f)|| / sqrt(d) - kappa)^2``, which is one
      sided: a slope at the fakes below ``kappa`` (per dimension, RMS) is not
      penalized.

    ``g = grad_x D(x)`` and ``d`` is the per-sample input size. There is no
    EMA critic, no interpolation and no random draw.

    Args:
        coeff: penalty strength.
        kappa: the cap on the fakes' per-dimension (RMS) gradient norm.
        lazy_k: apply every k-th step with the coefficient multiplied by k.
        record: a ``CriticStepRecord`` shared with the critic optimizer that
            records its own steps (the recipe's critic optimizer does).
            Default: a private record.
    """

    def __init__(
        self,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        record: Optional[CriticStepRecord] = None,
    ) -> None:
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        for name in ("coeff", "kappa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lazy_k < 1:
            raise ValueError(f"lazy_k must be >= 1, got {lazy_k}")
        # Step counters; shared with the critic optimizer when the recipe builds the pair.
        self.record = CriticStepRecord() if record is None else record

    def after_critic_step(self, optimizer_or_lr) -> None:
        """Record one critic optimizer step (call right after ``optimizer.step()``).

        The recipe's critic optimizer does this itself.
        """
        self.record.record_step(optimizer_or_lr)

    def state_dict(self) -> Dict[str, Any]:
        return self.record.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.record.load_state_dict(state)

    def penalty(
        self,
        D: torch.nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int = 1,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """Return ``(penalty, stats)`` for one critic step.

        ``penalty`` is a scalar attached to D's graph, or a detached zero when
        the lazy schedule skips ``step``. ``stats`` is empty unless
        ``collect_stats`` (which costs a host sync); then it holds
        ``applied``, ``pen``, ``center``, ``r1`` and ``fake_cap``. Several
        calls per critic step (e.g. one per role of a shared module) are
        allowed.
        """
        if self.lazy_k > 1 and step % self.lazy_k != 0:
            zero = torch.zeros((), device=x_real.device, dtype=x_real.dtype)
            return zero, ({"applied": False, "pen": 0.0} if collect_stats else {})
        # Lazy regularization: fewer applications, proportionally bigger hits,
        # so the time-averaged pressure on D is unchanged.
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        return self._k3p_penalty(D, x_real, x_fake, coefficient, collect_stats)

    def __call__(self, D, x_real, x_fake, step=1):
        """Return only the penalty tensor, without collecting synchronized stats."""
        return self.penalty(D, x_real, x_fake, step, collect_stats=False)[0]

    def _k3p_penalty(self, D, x_real, x_fake, coefficient, collect_stats):
        dimension = x_real[0].numel()
        if dimension != x_fake[0].numel():
            raise ValueError("k3p needs reals and fakes of the same per-sample size")
        self.record.calls += 1
        r1 = (self._grad_norm(D, x_real, squared=True) / dimension).mean()
        fake_rms = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
        fake_cap = (fake_rms - self.kappa).relu().square().mean()
        pen = (coefficient / 2.0) * (r1 + fake_cap)
        if not collect_stats:
            return pen, {}
        return pen, {"applied": True, "pen": float(pen.detach()), "center": self.kappa,
                     "r1": float(r1.detach()), "fake_cap": float(fake_cap.detach())}

    @staticmethod
    def _grad_norm(D: torch.nn.Module, x: torch.Tensor, squared: bool = False) -> torch.Tensor:
        """Per-sample ``||grad_x D(x)||`` (or its square) with ``create_graph=True``.

        The differentiated scalar is the sum of per-image logit means. A critic
        that returns one logit per image is unchanged.
        """
        x = x.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
        sq = g.pow(2).flatten(1).sum(dim=1)
        if squared:
            return sq
        # Epsilon keeps the sqrt differentiable at g = 0.
        return torch.sqrt(sq + 1e-12)
