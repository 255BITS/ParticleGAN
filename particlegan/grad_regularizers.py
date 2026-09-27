"""The critic gradient penalty kernel (K3P) and the per-critic step record it reads.

Users do not build these directly: ``recipe.make_critic_penalty(opt_d)``
pairs a ``GradientPenalty`` with the ``CriticStepRecord`` and EMA critic kept
by the recipe's critic optimizer. See ``docs/k3p.md`` for the formulation.
"""

from typing import Any, Callable, Dict, Optional, Tuple
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
    """Per-critic step state that K3P's penalty reads: anchor start, counters, LR record.

    One record belongs to one critic optimizer. ``record_step`` (after every
    critic optimizer step) advances the anchor EMA once started, then records
    the applied LR (diagnostic only). The penalty starts the anchor and counts
    its calls.
    """

    KEYS = ("lr_max", "lr_last", "anchor_started", "calls", "observed_steps")

    def __init__(self, anchor: Optional[Any] = None) -> None:
        self.anchor = anchor
        self.lr_max = 0.0
        self.lr_last: Optional[float] = None
        self.anchor_started = False
        self.calls = 0
        self.observed_steps = 0

    def record_step(self, optimizer_or_lr) -> None:
        if self.anchor_started and self.anchor is not None:
            self.anchor.update_()
        if isinstance(optimizer_or_lr, (int, float, torch.Tensor)):
            lr = float(optimizer_or_lr)
        else:
            lr = max(float(g["lr"]) for g in optimizer_or_lr.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"lr_max": self.lr_max, "lr_last": self.lr_last, "anchor_started": self.anchor_started,
                "calls": self.calls, "observed_steps": self.observed_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.KEYS):
            keys = sorted(state) if isinstance(state, dict) else state
            raise ValueError(f"k3p state keys {keys} != expected {sorted(self.KEYS)}")
        values = (float(state["lr_max"]), None if state["lr_last"] is None else float(state["lr_last"]),
                  bool(state["anchor_started"]), int(state["calls"]), int(state["observed_steps"]))
        self.lr_max, self.lr_last, self.anchor_started, self.calls, self.observed_steps = values


def path_points(x_real: torch.Tensor, x_fake: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``x_hat = r + u (f - r)``, pairing reals and fakes by batch index (first ``min(len)`` rows)."""
    n = min(len(x_real), len(x_fake))
    r, f = x_real[:n].detach(), x_fake[:n].detach()
    u = u[:n].reshape(n, *([1] * (r.dim() - 1))).to(r.dtype)
    return r + u * (f - r)


class GradientPenalty:
    """K3P critic gradient penalty: a path cap, a fake cap and an EMA anchor. No R1.

    ``pen = coeff/2 * (path_cap + fake_cap + anchor_weight * prox)`` with

    * ``path_cap = mean relu(||g(x_hat)|| / sqrt(d) - kappa)^2`` at
      ``x_hat = r + u (f - r)``, ``u ~ U(0, 1)`` per pair, reals and fakes
      paired by batch index;
    * ``fake_cap = mean relu(||g(f)|| / sqrt(d) - kappa)^2``;
    * ``prox = mean ||g(r) - gbar(r)||^2 / d``, which ties the critic's input
      gradient at the reals to that of its parameter EMA Dbar. The anchor
      starts at the first call (``prox`` is exactly 0 then) and the critic
      optimizer's step record advances the EMA after every critic step.

    ``g = grad_x D(x)``, ``gbar = grad_x Dbar(x)`` and ``d`` is the per-sample
    input size. The caps are one-sided, so a critic whose slope stays below
    ``kappa`` is not penalized at all; nothing pulls the slope at the data to 0.

    Args:
        coeff: penalty strength.
        kappa: the cap on the per-dimension (RMS) gradient norm.
        lazy_k: apply every k-th step with the coefficient multiplied by k.
        anchor_weight: weight of the EMA-anchor term ``prox`` (0 removes it
            and needs no anchor).
        anchor: a ``particlegan.k3p.CriticAnchor`` (or anything with
            ``start_()``, ``update_()`` and ``__call__``).
        record: a ``CriticStepRecord`` shared with the critic optimizer that
            records its own steps (the recipe's critic optimizer does). Its
            anchor is used. Default: a private record.
    """

    def __init__(
        self,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 1.0,
        anchor: Optional[Any] = None,
        record: Optional[CriticStepRecord] = None,
    ) -> None:
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.anchor_weight = float(anchor_weight)
        for name in ("coeff", "kappa", "anchor_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lazy_k < 1:
            raise ValueError(f"lazy_k must be >= 1, got {lazy_k}")
        if record is None:
            record = CriticStepRecord(anchor)
        elif anchor is not None and anchor is not record.anchor:
            raise ValueError("pass either anchor= or a record= holding that anchor, not both")
        # Step state (anchor start, counters); shared with the critic
        # optimizer when the recipe builds the pair.
        self.record = record
        self.anchor = record.anchor
        # Identity of the critic served through the constructor anchor
        # (id only; not checkpointed). Guards against one instance silently
        # anchoring a second critic to the first critic's EMA.
        self._critic_id: Optional[int] = None

    def after_critic_step(self, optimizer_or_lr) -> None:
        """Record one critic optimizer step (call right after ``optimizer.step()``).

        Advances the anchor EMA (once started). The recipe's critic optimizer
        does this itself.
        """
        self.record.record_step(optimizer_or_lr)

    def state_dict(self) -> Dict[str, Any]:
        """Scalar step state. The anchor's EMA critic is saved separately."""
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
        *,
        ema_critic: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Return ``(penalty, stats)`` for one critic step.

        ``penalty`` is a scalar attached to D's graph, or a detached zero when
        the lazy schedule skips ``step``. ``stats`` is empty unless
        ``collect_stats`` (which costs a host sync); then it holds
        ``applied``, ``pen``, ``center``, ``path_cap``, ``fake_cap`` and
        ``prox``. ``ema_critic`` evaluates Dbar for this call (default: the
        anchor). ``generator`` draws the path positions ``u`` (default: the
        global RNG). Several calls per critic step (e.g. one per role of a
        shared module) are allowed.
        """
        if self.lazy_k > 1 and step % self.lazy_k != 0:
            zero = torch.zeros((), device=x_real.device, dtype=x_real.dtype)
            return zero, ({"applied": False, "pen": 0.0} if collect_stats else {})
        # Lazy regularization: fewer applications, proportionally bigger hits,
        # so the time-averaged pressure on D is unchanged.
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        return self._k3p_penalty(D, x_real, x_fake, step, coefficient, collect_stats, ema_critic, generator)

    def __call__(self, D, x_real, x_fake, step=1, *, ema_critic=None, generator=None):
        """Return only the penalty tensor, without collecting synchronized stats."""
        return self.penalty(D, x_real, x_fake, step, collect_stats=False, ema_critic=ema_critic,
                            generator=generator)[0]

    def _cap(self, D, x):
        """``mean relu(||grad_x D(x)|| / sqrt(d) - kappa)^2`` (create_graph)."""
        rms = self._grad_norm(D, x, squared=False) / x[0].numel() ** 0.5
        return (rms - self.kappa).relu().square().mean()

    def _k3p_penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic, generator):
        dimension = x_real[0].numel()
        if dimension != x_fake[0].numel():
            raise ValueError("k3p needs reals and fakes of the same per-sample size")
        use_anchor = self.anchor_weight != 0.0
        if use_anchor and step > self.lazy_k and self.record.observed_steps == 0:
            raise RuntimeError(
                "k3p: after_critic_step(critic_optimizer) was never called; "
                "without it the EMA anchor never moves"
            )
        if ema_critic is None and use_anchor:
            owner = getattr(self.anchor, "critic", None)
            if owner is not None and D is not owner:
                raise ValueError(
                    "k3p: D is not the critic this penalty's anchor tracks; "
                    "use one penalty per critic or pass ema_critic= explicitly"
                )
            if self._critic_id is None:
                self._critic_id = id(D)
            elif self._critic_id != id(D):
                raise ValueError(
                    "k3p: this penalty already serves a different critic; "
                    "use one penalty per critic or pass ema_critic= explicitly"
                )
        self.record.calls += 1
        u = torch.rand(len(x_real), device=x_real.device, generator=generator)
        path_cap = self._cap(D, path_points(x_real, x_fake, u))
        fake_cap = self._cap(D, x_fake.detach())
        total = path_cap + fake_cap
        prox = None
        if use_anchor:
            prox = self._prox(D, x_real, ema_critic, dimension)
            total = total + prox
        pen = (coefficient / 2.0) * total
        if not collect_stats:
            return pen, {}
        return pen, {"applied": True, "pen": float(pen.detach()), "center": self.kappa,
                     "path_cap": float(path_cap.detach()), "fake_cap": float(fake_cap.detach()),
                     "prox": 0.0 if prox is None else float(prox.detach())}

    def _prox(self, D, x_real, ema_critic, dimension):
        """``anchor_weight * mean ||g_r - gbar_r||^2 / d``; starts the anchor (prox 0) on the first call."""
        if not self.record.anchor_started:
            anchor = self.anchor
            if anchor is None and ema_critic is None:
                raise ValueError("k3p needs a CriticAnchor/ema_critic (or anchor_weight=0)")
            if anchor is not None:
                anchor.start_()
            self.record.anchor_started = True
            return x_real.new_zeros(())  # Dbar == D on the starting call
        anchor = ema_critic if ema_critic is not None else self.anchor
        if anchor is None:
            raise ValueError("k3p needs a CriticAnchor/ema_critic (or anchor_weight=0)")
        x = x_real.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
        xb = x_real.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            gb = torch.autograd.grad(_score_scalar(anchor(xb)), xb)[0].detach()
        return self.anchor_weight * (g - gb).pow(2).flatten(1).sum(dim=1).mean() / dimension

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
