"""KA2 critic regularization behind the recipe's ordinary PyTorch factories.

The asymmetric moment-surprise controller is local to one critic optimizer.
Its penalty-call clock, surprise history, EMA and all controller state travel
with ``optimizer.state_dict()``. The generator-side K3P primitives are shared;
no global optimizer hooks or parameter swapping are used here.

This implements the frozen KA2 mechanism, including its 799 pure-A calls
before the first blend at call 800. Learning-rate and noise schedules remain
recipe settings; only the critic's blend and release are independent of LR.
"""
from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from .grad_regularizers import CriticStepRecord, GradientPenalty, _score_scalar
from .k3p import CriticPenalty as _K3PCriticPenalty
from .k3p import K3PCriticAdam, _first_output


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

__all__ = ["KA2StepRecord", "KA2GradientPenalty", "KA2CriticAdam", "CriticPenalty"]


def _median(values):
    """The frozen rule selects the upper middle value for an even count."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


@torch.no_grad()
def _surprise_of(optimizer):
    """Read gradients and the *completed* Adam step's second moment."""
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


class KA2StepRecord(CriticStepRecord):
    """Per-critic controller; calls drive release, completed steps update EMA.

    The surprise sampled after Adam is consumed at the next applied blended
    penalty call. Multiple roles may therefore consume the same completed
    step's surprise, matching the frozen penalty-call clock. Lazy skips do
    not advance that clock; every actual optimizer step still updates EMA.
    """

    _COUNTERS = ("calls", "observed_steps", "low_streak", "ema_updates", "ema_skips", "ema_reseeds")
    _OPTIONAL = ("lr_last", "last_sur", "sur_base", "last_ratio")
    KEYS = ("formulation", "anchor_min_decay", "lr_max", "lr_last", "anchor_started", "calls",
            "observed_steps", "last_sur", "sur_hist", "sur_base", "w", "low_streak", "alpha",
            "last_ratio", "ema_updates", "ema_skips", "ema_reseeds")

    def __init__(self, anchor=None, *, anchor_min_decay=0.90):
        if (isinstance(anchor_min_decay, bool) or not math.isfinite(anchor_min_decay)
                or not 0.0 <= anchor_min_decay < 1.0):
            raise ValueError("anchor_min_decay must be finite and in [0, 1)")
        super().__init__(anchor)
        self.anchor_min_decay = float(anchor_min_decay)
        self.last_sur = None
        self.sur_hist = []
        self.sur_base = None
        self.w = 1.0
        self.low_streak = 0
        self.alpha = 0.0
        self.last_ratio = None
        self.ema_updates = self.ema_skips = self.ema_reseeds = 0

    def advance_blend(self):
        """Consume one blended penalty call and return its anchor weight."""
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

    def record_step(self, optimizer):
        """After guard and Adam: sample surprise, reseed, then update EMA."""
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

    def state_dict(self):
        state = {key: getattr(self, key) for key in self.KEYS if key != "formulation"}
        state["formulation"] = "ka2"
        state["sur_hist"] = list(self.sur_hist)
        return state

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("formulation") != "ka2":
            raise ValueError("KA2 cannot resume an older K3P critic state; use its original release")
        if set(state) != set(self.KEYS):
            raise ValueError("invalid KA2 controller state keys")
        if state["anchor_min_decay"] != self.anchor_min_decay:
            raise ValueError("KA2 checkpoint anchor_min_decay does not match this recipe")
        for key in self._COUNTERS:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid KA2 controller {key}")
        for key in ("lr_max", "w", "alpha", *self._OPTIONAL):
            value = state[key]
            if value is None and key in self._OPTIONAL:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid KA2 controller {key}")
        if state["w"] not in (0.0, 1.0) or state["alpha"] > 1.0 or type(state["anchor_started"]) is not bool:
            raise ValueError("invalid KA2 controller gate/alpha/anchor state")
        history = state["sur_hist"]
        if not isinstance(history, list) or len(history) > HIST_CAP or any(
                isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0 for value in history):
            raise ValueError("invalid KA2 controller surprise history")
        # Validate everything before mutation. In particular, the optimizer
        # preflights this on a shallow record copy, so never touch its anchor.
        for key in self.KEYS:
            if key not in ("formulation", "anchor_min_decay"):
                setattr(self, key, deepcopy(state[key]))


class KA2GradientPenalty(GradientPenalty):
    """Pure A for 799 applied calls, then .5 A + .5 (B + W * anchor).

    ``W`` and the EMA tracking rate come from the paired KA2 optimizer's
    moment-surprise controller. ``anchor_weight=0`` is the explicit anchor
    ablation. Constant LR does not disable the blend or its release signal.
    """

    def __init__(self, coeff=1.0, kappa=1.0, lazy_k=1, anchor_weight=1.0, *, record=None):
        if record is None:
            record = KA2StepRecord()
        if not isinstance(record, KA2StepRecord):
            raise TypeError("KA2GradientPenalty needs a KA2StepRecord")
        super().__init__(coeff=coeff, kappa=kappa, lazy_k=lazy_k, anchor_weight=anchor_weight, record=record)

    def blend_weight(self):
        """Frozen KA2 reports s=.5; its initial pure-A calls bypass the blend."""
        return S_FIX

    def _k3p_penalty(self, D, x_real, x_fake, step, coefficient, collect_stats, ema_critic):
        dimension = x_real[0].numel()
        if dimension != x_fake[0].numel():
            raise ValueError("ka2 needs reals and fakes of the same per-sample size")
        record = self.record
        if step > self.lazy_k and record.observed_steps == 0:
            raise RuntimeError("ka2 needs completed critic optimizer steps to observe Adam moment surprise")
        if ema_critic is None:
            owner = getattr(self.anchor, "critic", None)
            if owner is not None and D is not owner:
                raise ValueError("ka2: D is not the critic this penalty's anchor tracks")
            if self._critic_id is None:
                self._critic_id = id(D)
            elif self._critic_id != id(D):
                raise ValueError("ka2: use one penalty per critic or pass ema_critic= explicitly")
        record.calls += 1
        if record.calls + 1 <= WARMUP_CALLS:
            real_squared = self._grad_norm(D, x_real, squared=True) / dimension
            fake_norm = self._grad_norm(D, x_fake, squared=False) / dimension ** 0.5
            fake_cap = (fake_norm - self.kappa).relu().square()
            penalty = (coefficient / 2.0) * (real_squared.mean() + fake_cap.mean())
            record.w = 1.0
            prox = None
            weight, phase = 1.0, "a"
        else:
            use_anchor = self.anchor_weight != 0.0
            anchor = ema_critic if ema_critic is not None else self.anchor
            if use_anchor and anchor is None:
                raise ValueError("ka2 blended phase needs the critic's EMA")
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
            a_term = (sq_r / dimension).mean() + (n_f / dimension ** 0.5 - self.kappa).relu().square().mean()
            b_term = F.relu(n_r - self.kappa).pow(2).mean() + F.relu(n_f - self.kappa).pow(2).mean() + weight * prox
            penalty = (coefficient / 2.0) * (S_FIX * a_term + (1.0 - S_FIX) * b_term)
            phase = "blend"
        if not collect_stats:
            return penalty, {}
        return penalty, {"applied": True, "pen": float(penalty.detach()), "center": self.kappa,
                         "s": S_FIX, "w": weight, "alpha": record.alpha,
                         "prox": 0.0 if prox is None else float(prox.detach()), "phase": phase}


class KA2CriticAdam(K3PCriticAdam):
    """Adam with the shared spike guard and a private KA2 controller/EMA.

    The inherited step is guard -> Adam -> ``record_step``. KA2 replaces the
    latter's fixed EMA decay with moment-surprise tracking and guarded reseeds.
    Checkpoints from K3P are rejected before the live optimizer is changed.
    """

    def __init__(self, params, *, critic, ema_critic=None, anchor_min_decay=0.90,
                 guard_ratio=5.0, guard_min_steps=200, **adam_kwargs):
        record = KA2StepRecord(anchor_min_decay=anchor_min_decay)
        super().__init__(params, critic=critic, ema_critic=ema_critic, anchor_decay=anchor_min_decay,
                         guard_ratio=guard_ratio, guard_min_steps=guard_min_steps, **adam_kwargs)
        record.anchor = self.anchor
        self.record = record


class CriticPenalty(_K3PCriticPenalty):
    """Recipe-facing KA2 penalty with the shared conditioning and role support."""

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides):
        if not isinstance(optimizer, KA2CriticAdam):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")
        self.optimizer, self.critic = optimizer, optimizer.critic
        options = recipe._penalty_options(**penalty_overrides)
        if optimizer.anchor is None and options["anchor_weight"] != 0:
            raise ValueError("this penalty needs the critic's EMA: pass ema_critic=copy.deepcopy(critic) "
                             "to recipe.make_optimizers / recipe.make_critic_optimizer")
        self.regularizer = KA2GradientPenalty(record=optimizer.record, **options)
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}

    def diagnostics(self):
        out = super().diagnostics()
        record = self.optimizer.record
        out.update(anchor_weight=record.w, anchor_alpha=record.alpha, surprise_ratio=record.last_ratio,
                   ema_updates=record.ema_updates, ema_skips=record.ema_skips, ema_reseeds=record.ema_reseeds)
        return out
