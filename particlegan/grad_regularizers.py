"""The critic gradient penalty kernel (KA2) and the per-critic step record it reads.

Users do not build these directly: ``recipe.make_critic_penalty(opt_d)``
pairs a ``GradientPenalty`` with the ``CriticStepRecord``, EMA critic and LR
controller kept by the recipe's critic optimizer. See ``docs/k3p.md`` for the
formulation.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple
import math

import torch
import torch.nn.functional as F


# KA2 constants: pure phase A for the first WARMUP_CALLS - 1 applied calls,
# then a fixed 50/50 blend whose anchor is gated by the critic's Adam
# moment surprise (median of the last SHORT_WINDOW samples over the frozen
# median of the first BASE_WINDOW).
WARMUP_CALLS = 800
S_FIX = 0.5
SHORT_WINDOW = 24
GATE_MIN_SAMPLES = 25
BASE_WINDOW = 24
HIST_CAP = 400
REL_HI = 3.0
REL_LO = 1.75
GAIN_LO = 1.0
GAIN_HI = 3.0
K_ATK = 1.0 / 60.0
K_REL = 0.5
RESEED_STREAK = 60


def _score_scalar(logits):
    """Sum of per-image logit means.

    One logit per image matches ``logits.sum()``: other batch rows do not
    change an image's gradient. Extra logits are averaged inside the image so
    a spatial map does not multiply that gradient by its number of locations.
    """
    if logits.ndim < 2:
        return logits.sum()
    return logits.flatten(1).mean(dim=1).sum()


def _median(values):
    """Upper middle value for an even count (the frozen KA2 rule)."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


@torch.no_grad()
def _surprise_of(optimizer):
    """Median over tensors of ``rms(grad) / sqrt(v_hat)`` after a completed Adam step.

    A measurement, not a step emulation: it reads ``exp_avg_sq`` even when the
    group uses ``amsgrad``. Torch updates ``exp_avg_sq`` identically under
    AMSGrad, so the surprise is the same function of the gradient history in
    both modes; the never-decaying ``max_exp_avg_sq`` would depress the ratio
    for good after any early transient.
    """
    values = []
    for group in optimizer.param_groups:
        beta2 = group["betas"][1]
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if parameter.grad is None or not state or "exp_avg_sq" not in state:
                continue
            step = state["step"]
            if step < 1:
                continue
            vhat = state["exp_avg_sq"].mean() / (1.0 - beta2 ** step)
            surprise = parameter.grad.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
            values.append(float(surprise.detach()))
    return None if not values else _median(values)


class CriticStepRecord:
    """Per-critic KA2 state: penalty-call clock, moment-surprise gate and EMA rate.

    One record belongs to one critic optimizer. The penalty advances the call
    clock and the blend gate (``advance_blend``); ``record_step`` (after every
    critic optimizer step) samples the Adam moment surprise, reseeds or
    advances the EMA critic, and records the applied LR. Lazy skips do not
    advance the call clock; every optimizer step still updates the EMA.
    """

    _COUNTERS = ("calls", "observed_steps", "low_streak", "ema_updates", "ema_skips", "ema_reseeds")
    _OPTIONAL = ("lr_last", "last_sur", "sur_base", "last_ratio")
    KEYS = ("formulation", "anchor_min_decay", "lr_max", "lr_last", "anchor_started", "calls",
            "observed_steps", "last_sur", "sur_hist", "sur_base", "w", "low_streak", "alpha",
            "last_ratio", "ema_updates", "ema_skips", "ema_reseeds")

    def __init__(self, anchor: Optional[Any] = None, *, anchor_min_decay: float = 0.90) -> None:
        if (isinstance(anchor_min_decay, bool) or not math.isfinite(anchor_min_decay)
                or not 0.0 <= anchor_min_decay < 1.0):
            raise ValueError("anchor_min_decay must be finite and in [0, 1)")
        self.anchor = anchor
        self.anchor_min_decay = float(anchor_min_decay)
        self.lr_max = 0.0
        self.lr_last: Optional[float] = None
        self.anchor_started = False
        self.calls = 0
        self.observed_steps = 0
        self.last_sur = None
        self.sur_hist = []
        self.sur_base = None
        self.w = 1.0
        self.low_streak = 0
        self.alpha = 0.0
        self.last_ratio = None
        self.ema_updates = self.ema_skips = self.ema_reseeds = 0

    def advance_blend(self):
        """Consume one blended penalty call and return its anchor weight ``W``."""
        if self.last_sur is not None:
            self.sur_hist.append(float(self.last_sur))
            if len(self.sur_hist) > HIST_CAP:
                del self.sur_hist[:len(self.sur_hist) - HIST_CAP]
        ratio = None
        weight = 1.0
        if len(self.sur_hist) >= GATE_MIN_SAMPLES:
            if self.sur_base is None:
                self.sur_base = _median(self.sur_hist[:BASE_WINDOW])
            if self.sur_base and self.sur_base > 0.0:
                ratio = _median(self.sur_hist[-SHORT_WINDOW:]) / self.sur_base
                weight = self.w
                if weight >= 1.0 and ratio > REL_HI:
                    weight = 0.0
                elif weight <= 0.0 and ratio < REL_LO:
                    weight = 1.0
                self.w = weight
        self.last_ratio = ratio
        if ratio is None or ratio <= GAIN_LO:
            target = 0.0
        elif ratio >= GAIN_HI:
            target = 1.0
        else:
            target = (ratio - GAIN_LO) / (GAIN_HI - GAIN_LO)
        rate = K_ATK if target > self.alpha else K_REL
        self.alpha = max(0.0, min(1.0, self.alpha + (target - self.alpha) * rate))
        if weight < 0.5 and ratio is not None and ratio > REL_HI:
            self.low_streak += 1
        else:
            self.low_streak = 0
        return weight

    def record_step(self, optimizer) -> None:
        """After guard and Adam: sample surprise, reseed, then update the EMA critic."""
        surprise = _surprise_of(optimizer)
        if surprise is not None:
            self.last_sur = surprise
        if self.anchor_started and self.anchor is not None:
            if self.low_streak >= RESEED_STREAK:
                self.anchor.start_()
                self.ema_reseeds += 1
                self.low_streak = 0
            decay = 1.0 - self.alpha * (1.0 - self.anchor_min_decay)
            if decay >= 1.0:
                self.ema_skips += 1
            else:
                self.anchor.decay = decay
                self.anchor.update_()
                self.ema_updates += 1
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1

    def state_dict(self) -> Dict[str, Any]:
        state = {key: getattr(self, key) for key in self.KEYS if key != "formulation"}
        state["formulation"] = "ka2"
        state["sur_hist"] = list(self.sur_hist)
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("formulation") != "ka2":
            raise ValueError("this critic state comes from an older formulation (K3P) and cannot resume "
                             "under KA2; pin the release that wrote it, or start a new run")
        if set(state) != set(self.KEYS):
            raise ValueError("invalid KA2 critic state keys")
        if state["anchor_min_decay"] != self.anchor_min_decay:
            raise ValueError("checkpoint anchor_min_decay does not match this recipe")
        for key in self._COUNTERS:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid KA2 critic state {key}")
        for key in ("lr_max", "w", "alpha", *self._OPTIONAL):
            value = state[key]
            if value is None and key in self._OPTIONAL:
                continue
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"invalid KA2 critic state {key}")
        if state["w"] not in (0.0, 1.0) or state["alpha"] > 1.0 or type(state["anchor_started"]) is not bool:
            raise ValueError("invalid KA2 critic gate/alpha/anchor state")
        history = state["sur_hist"]
        if not isinstance(history, list) or len(history) > HIST_CAP or any(
                isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0 for value in history):
            raise ValueError("invalid KA2 critic surprise history")
        # Validate everything before mutation; never touch the anchor here.
        for key in self.KEYS:
            if key not in ("formulation", "anchor_min_decay"):
                setattr(self, key, deepcopy(state[key]))


class GradientPenalty:
    """KA2 critic gradient penalty: phase A, then a fixed blend with an EMA anchor.

    For the first 799 applied calls ``pen = coeff/2 * A``; from call 800 on
    ``pen = coeff/2 * (.5 A + .5 B)``, where

    * ``A`` is R1 on reals plus a one-sided cap on fakes, in RMS units:
      ``mean ||g_r||^2 / d + mean relu(||g_f|| / sqrt(d) - kappa)^2``;
    * ``B`` caps both gradient norms in L2 units and adds
      ``W * anchor_weight * prox``, ``prox = mean ||g_r - gbar_r||^2 / d``,
      tying the critic's input gradient to that of its parameter EMA Dbar.

    ``g = grad_x D(x)`` and ``d`` is the per-sample input size. ``W`` (0 or
    1) and the EMA tracking rate come from the record's moment-surprise gate;
    when a ``controller`` (the recipe's LR controller) is attached, its game
    trust scales the tracking rate and ``W`` stays 1 while the data is not
    moving.

    Args:
        coeff: penalty strength.
        kappa: the cap on the gradient norm.
        lazy_k: apply every k-th step with the coefficient multiplied by k.
        anchor_weight: weight of the EMA-anchor term (0 removes it and needs no anchor).
        anchor: a ``particlegan.k3p.CriticAnchor`` (or anything with
            ``start_()``, ``update_()``, ``decay`` and ``__call__``), started at
            the first blended call.
        record: a ``CriticStepRecord`` shared with the critic optimizer that
            records its own steps (the recipe's critic optimizer does). Its
            anchor is used. Default: a private record.
        controller: optional ``particlegan.dv12.DV12Controller``.
    """

    def __init__(
        self,
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        anchor_weight: float = 1.0,
        anchor: Optional[Any] = None,
        record: Optional[CriticStepRecord] = None,
        controller: Optional[Any] = None,
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
        if not isinstance(record, CriticStepRecord):
            raise TypeError("record must be a CriticStepRecord")
        self.record = record
        self.anchor = record.anchor
        self.controller = controller
        # Identity of the critic served through the constructor anchor (id
        # only; not checkpointed): one instance must not anchor a second
        # critic to the first critic's EMA.
        self._critic_id: Optional[int] = None

    def blend_weight(self) -> float:
        """Weight of phase A: 1 during the warmup calls, then the fixed .5."""
        return 1.0 if self.record.calls + 1 < WARMUP_CALLS else S_FIX

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
    ) -> Tuple[torch.Tensor, Dict]:
        """Return ``(penalty, stats)`` for one critic step.

        ``penalty`` is a scalar attached to D's graph, or a detached zero when
        the lazy schedule skips ``step``. ``stats`` is empty unless
        ``collect_stats`` (which costs a host sync). ``ema_critic`` evaluates
        Dbar for this call (default: the anchor).
        """
        if self.lazy_k > 1 and step % self.lazy_k != 0:
            zero = torch.zeros((), device=x_real.device, dtype=x_real.dtype)
            return zero, ({"applied": False, "pen": 0.0} if collect_stats else {})
        # Lazy regularization: fewer applications, proportionally bigger hits,
        # so the time-averaged pressure on D is unchanged.
        coefficient = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff
        return self._penalty(D, x_real, x_fake, step, coefficient, collect_stats, ema_critic)

    def __call__(self, D, x_real, x_fake, step=1, *, ema_critic=None):
        """Return only the penalty tensor, without collecting synchronized stats."""
        return self.penalty(D, x_real, x_fake, step, collect_stats=False, ema_critic=ema_critic)[0]

    def _penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic):
        """KA2, op for op the frozen PR #155 DV12 package's rule."""
        dimension = x_real[0].numel()
        if dimension != x_fake[0].numel():
            raise ValueError("the critic penalty needs reals and fakes of the same per-sample size")
        record = self.record
        if step > self.lazy_k and record.observed_steps == 0:
            raise RuntimeError("the critic penalty needs completed critic optimizer steps to observe "
                               "Adam moment surprise; step the recipe's critic optimizer")
        if ema_critic is None:
            owner = getattr(self.anchor, "critic", None)
            if owner is not None and D is not owner:
                raise ValueError("D is not the critic this penalty's anchor tracks; "
                                 "use one penalty per critic or pass ema_critic= explicitly")
            if self._critic_id is None:
                self._critic_id = id(D)
            elif self._critic_id != id(D):
                raise ValueError("this penalty already serves a different critic; "
                                 "use one penalty per critic or pass ema_critic= explicitly")
        record.calls += 1
        if record.calls + 1 <= WARMUP_CALLS:
            real_squared = self._grad_norm(D, x_real, squared=True) / dimension
            fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
            fake_cap = (fake_norm - self.kappa).relu().square()
            pen = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
            record.w = 1.0
            prox = None
            weight, phase = 1.0, "a"
        else:
            use_anchor = self.anchor_weight != 0.0
            anchor = ema_critic if ema_critic is not None else self.anchor
            if use_anchor and anchor is None:
                raise ValueError("the blended phase needs the critic's EMA")
            x = x_real.detach().clone().requires_grad_(True)
            g = torch.autograd.grad(_score_scalar(D(x)), x, create_graph=True)[0]
            sq_r = g.pow(2).flatten(1).sum(dim=1)
            n_r = torch.sqrt(sq_r + 1e-12)
            n_f = self._grad_norm(D, x_fake, squared=False)
            if not record.anchor_started:
                if self.anchor is not None:
                    self.anchor.start_()
                record.anchor_started = True
                prox = g.new_zeros(())
            elif not use_anchor:
                prox = g.new_zeros(())
            else:
                xb = x_real.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    gb = torch.autograd.grad(_score_scalar(anchor(xb)), xb)[0].detach()
                prox = (g - gb).pow(2).flatten(1).sum(dim=1).mean() / dimension
                if self.anchor_weight != 1.0:
                    prox = self.anchor_weight * prox
            weight = record.advance_blend()
            controller = self.controller
            if controller is not None:
                # Untrusted game states track the EMA more slowly; without
                # data movement the anchor stays engaged.
                record.alpha *= controller.game_trust
                if controller.data_drive < .1:
                    weight = record.w = 1.0
                    record.low_streak = 0
            a_term = (sq_r / dimension).mean() + (n_f / dimension ** 0.5 - self.kappa).relu().square().mean()
            b_term = F.relu(n_r - self.kappa).pow(2).mean() + F.relu(n_f - self.kappa).pow(2).mean() + weight * prox
            pen = (coefficient / 2.0) * (S_FIX * a_term + (1.0 - S_FIX) * b_term)
            phase = "blend"
        if not collect_stats:
            return pen, {}
        return pen, {"applied": True, "pen": float(pen.detach()), "center": self.kappa,
                     "s": 1.0 if phase == "a" else S_FIX, "w": weight, "alpha": record.alpha,
                     "prox": 0.0 if prox is None else float(prox.detach()), "phase": phase}

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
