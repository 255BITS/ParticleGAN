"""Historical K3P critic kernels and shared optimizer/EMA helpers.

Public recipe factories now select ``particlegan.ka2`` for the critic.
KA2 reuses this module's spike guard, robust EMA and conditional-critic
adapter, together with the unchanged ``K3PGeneratorAdam``, A2 sparse latent
damping and direct-particle response. The original ``K3PCriticAdam`` and
``CriticPenalty`` remain available for archived research and parity tests.

Users construct active components through the recipe's ``make_*`` factories.
Optimizer ``step()`` performs step-time work and ``state_dict()`` holds its
state. Nothing registers optimizer hooks or keeps module-level state.
"""
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .grad_regularizers import CriticStepRecord, GradientPenalty

__all__ = ["CriticAnchor", "RobustCriticAnchor", "CriticSpikeGuard", "LatentRowDamping",
           "DirectParticleResponse", "K3PCriticAdam", "K3PGeneratorAdam", "CriticPenalty"]

# Extra optimizer.state_dict() key holding the regularization state.
_STATE_KEY = "regularizer"


def _group_of(optimizer, param) -> Dict[str, Any]:
    for group in optimizer.param_groups:
        if any(p is param for p in group["params"]):
            return group
    raise ValueError("parameter is not in this optimizer")


class CriticAnchor:
    """Parameter EMA of ``critic`` held in the caller-allocated ``ema_critic``.

    ``ema_critic`` must be a structurally identical module (e.g.
    ``copy.deepcopy(critic).requires_grad_(False)``); only its parameters are
    written. ``start_()`` copies the live parameters in, ``update_()`` applies
    one EMA step (trainable parameters only; frozen ones are copied so no
    rounding drift accumulates), and calling the anchor evaluates the EMA
    critic. Checkpoint it by saving ``ema_critic.state_dict()``.
    """

    def __init__(self, critic: nn.Module, ema_critic: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"decay must satisfy 0 <= decay < 1, got {decay}")
        live = dict(critic.named_parameters())
        ema = dict(ema_critic.named_parameters())
        if live.keys() != ema.keys():
            raise ValueError("ema_critic parameter names differ from critic's")
        for name, p in live.items():
            if ema[name].shape != p.shape:
                raise ValueError(f"ema_critic parameter {name} shape {tuple(ema[name].shape)} != {tuple(p.shape)}")
            if ema[name] is p:
                raise ValueError("ema_critic shares parameters with critic; pass a separate copy")
        self.critic, self.ema_critic, self.decay = critic, ema_critic, float(decay)
        self._pairs = [(ema[name], p) for name, p in live.items()]

    @torch.no_grad()
    def start_(self) -> None:
        for e, p in self._pairs:
            e.copy_(p.detach())

    @torch.no_grad()
    def update_(self) -> None:
        for e, p in self._pairs:
            if p.requires_grad:
                e.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                e.copy_(p.detach())

    def __call__(self, x):
        return self.ema_critic(x)


class CriticSpikeGuard:
    """Scale down critic gradient spikes before an Adam step.

    Call ``apply_(optimizer)`` between ``backward()`` and ``optimizer.step()``.
    A tensor with at least ``min_steps`` prior Adam steps whose gradient RMS
    exceeds ``ratio * sqrt(mean bias-corrected v)`` is scaled to that ratio;
    everything else is multiplied by exactly 1.0. Tensors without Adam state
    yet are skipped. Returns the number of clipped tensors this step (a
    tensor, no host sync).
    """

    def __init__(self, ratio: float = 5.0, min_steps: int = 200) -> None:
        if not float(ratio) > 0.0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if int(min_steps) < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")
        self.ratio, self.min_steps = float(ratio), int(min_steps)
        self._clipped: Any = 0

    @torch.no_grad()
    def apply_(self, optimizer) -> Any:
        flags = []
        for group in optimizer.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                st = optimizer.state.get(p)
                if p.grad is None or not st or "exp_avg_sq" not in st:
                    continue
                t = st["step"]
                vhat = st["exp_avg_sq"].mean() / (1.0 - beta2 ** t)
                ratio = p.grad.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
                clip = (t >= self.min_steps) & (ratio > self.ratio)
                p.grad.mul_(torch.where(clip, self.ratio / ratio, torch.ones_like(ratio)))
                flags.append(clip)
        if not flags:
            return 0
        count = torch.stack(flags).sum()
        self._clipped = self._clipped + count
        return count

    @property
    def clipped_tensors(self) -> int:
        return int(self._clipped)

    def state_dict(self) -> Dict[str, int]:
        return {"clipped_tensors": int(self._clipped)}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        if set(state) != {"clipped_tensors"}:
            raise ValueError(f"guard state keys {sorted(state)} != ['clipped_tensors']")
        self._clipped = int(state["clipped_tensors"])


class LatentRowDamping:
    """A2: bounded coherence damping for a sparse latent table (e.g. ``ParticlePrior.z``).

    Row i's Adam response becomes ``rho_i * g_i / (sqrt(v_hat) + eps)`` with
    ``rho_i = .75 + .25 cos(g_i, h_i)`` in [0.5, 1], where ``h_i`` is row i's
    last observed gradient (``rho_i = 1`` without history). Applies only on
    steps where some row got no gradient AND the cumulative row-observation
    rate is below ``max_rate``; otherwise the parent Adam step is untouched.
    The table must be alone in an Adam group with beta1 == 0; v stays the
    parent's (raw g).

    ``history`` is caller-allocated (same shape/dtype/device as ``table``) and
    checkpointed by the caller alongside ``state_dict()``. Wrap the generator
    optimizer step: ``with damping.around(opt_g): opt_g.step()``
    (``K3PGeneratorAdam.step`` does this).
    """

    BETA1 = 0.5

    def __init__(self, table: nn.Parameter, history: torch.Tensor, max_rate: float = 0.5) -> None:
        if table.dim() != 2:
            raise ValueError("LatentRowDamping needs a 2-D (rows, dim) table")
        if history.shape != table.shape or history.dtype != table.dtype or history.device != table.device:
            raise ValueError("history must match table's shape, dtype and device")
        if not 0.0 < float(max_rate) <= 1.0:
            raise ValueError(f"max_rate must be in (0, 1], got {max_rate}")
        self.table, self.history, self.max_rate = table, history, float(max_rate)
        self.observed, self.total, self.started = 0, 0, False

    @torch.no_grad()
    def begin(self, optimizer) -> Optional[Tuple]:
        p = self.table
        if p.grad is None:
            return None
        group = _group_of(optimizer, p)
        if len(group["params"]) != 1:
            raise ValueError("LatentRowDamping needs the table alone in its param group")
        if group["betas"][0] != 0.0:
            raise ValueError("LatentRowDamping needs beta1 == 0 for the table's group")
        g = p.grad.detach()
        norm = g.square().sum(-1).sqrt()
        active = norm > 0
        count = int(active.sum())
        self.observed += count
        self.total += active.numel()
        rate = self.observed / self.total
        sparse = count < active.numel()
        if sparse and not self.started:
            self.history.zero_()
            self.started = True
        if not self.started:
            return None
        h = self.history
        token = None
        state = optimizer.state.get(p)
        # Without Adam state (first step) the parent beta1=0 step is used: with
        # no history rho == 1, which is the same update bit for bit.
        if sparse and rate < self.max_rate and state and "exp_avg" in state:
            hn = h.square().sum(-1).sqrt()
            has = active & (hn > 0)
            cos = (g * h).sum(-1) / (norm * hn).clamp_min(1e-30)
            rho = torch.where(has, .75 + .25 * cos, torch.ones_like(cos))
            bc1 = 1. - self.BETA1 ** (float(state["step"]) + 1.)
            # Adam's lerp with weight .5 gives m = rho*bc1*g, so m_hat = rho*g.
            state["exp_avg"].copy_(g * (2. * bc1 * rho - 1.).unsqueeze(-1))
            token = (group, group["betas"], state["exp_avg"], g.clone())
            group["betas"] = (self.BETA1, group["betas"][1])
        h[active] = g[active]
        return token

    @torch.no_grad()
    def end(self, token: Optional[Tuple]) -> None:
        if token is None:
            return
        group, betas, m, g = token
        group["betas"] = betas
        m.copy_(g)  # parent beta1=0 state holds the raw gradient

    @contextmanager
    def around(self, optimizer):
        token = self.begin(optimizer)
        try:
            yield
        finally:
            self.end(token)

    def state_dict(self) -> Dict[str, Any]:
        return {"observed": self.observed, "total": self.total, "started": self.started}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if set(state) != {"observed", "total", "started"}:
            raise ValueError(f"latent damping state keys {sorted(state)} != ['observed', 'started', 'total']")
        self.observed, self.total, self.started = int(state["observed"]), int(state["total"]), bool(state["started"])


class DirectParticleResponse:
    """LR gain for a param group of direct sample particles (not a latent table).

    For the step it sets the group's betas to ``betas`` and multiplies its LR
    by ``1 + clamp(cos(center(g_t), center(g_prev)), 0, 1)``, where
    ``center`` subtracts the per-column mean over particles and ``g_prev`` is
    the previous step's centered gradient kept in the caller-allocated flat
    ``history`` (numel = total numel of ``params``). ``params`` must be exactly
    one optimizer param group. ``gain=False`` keeps the LR unchanged (the
    betas still apply). Wrap the step: ``with resp.around(opt): opt.step()``
    (``K3PGeneratorAdam.step`` does this).
    """

    def __init__(self, params: Sequence[nn.Parameter], history: torch.Tensor,
                 betas: Tuple[float, float] = (0.0, 0.9), gain: bool = True) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("DirectParticleResponse needs at least one parameter")
        numel = sum(p.numel() for p in self.params)
        if history.dim() != 1 or history.numel() != numel:
            raise ValueError(f"history must be a flat tensor of {numel} elements")
        if history.dtype != self.params[0].dtype or history.device != self.params[0].device:
            raise ValueError("history must match the parameters' dtype and device")
        self.history, self.betas = history, (float(betas[0]), float(betas[1]))
        self.gain = bool(gain)
        self.started = False
        self.last_gain = 1.0

    def _group(self, optimizer):
        ids = {id(p) for p in self.params}
        for group in optimizer.param_groups:
            if {id(p) for p in group["params"]} == ids and len(group["params"]) == len(ids):
                return group
        raise ValueError("params must be exactly one param group of this optimizer")

    @torch.no_grad()
    def begin(self, optimizer) -> Optional[Tuple]:
        group = self._group(optimizer)
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return None
        if len(grads) != len(self.params):
            raise ValueError("DirectParticleResponse needs gradients for all or none of its params")
        current = torch.cat([(g.detach() - g.detach().mean(dim=0, keepdim=True)).flatten() for g in grads])
        gain = 1.0
        if self.started and self.gain:
            cosine = float(F.cosine_similarity(current, self.history, dim=0, eps=1e-12))
            gain = 1.0 + max(0.0, min(1.0, cosine))
        self.history.copy_(current)
        self.started = True
        self.last_gain = gain
        token = (group, group["lr"], group["betas"])
        group["betas"] = self.betas
        if gain != 1.0:
            group["lr"] *= gain
        return token

    def end(self, token: Optional[Tuple]) -> None:
        if token is None:
            return
        group, lr, betas = token
        group["lr"], group["betas"] = lr, betas

    @contextmanager
    def around(self, optimizer):
        token = self.begin(optimizer)
        try:
            yield
        finally:
            self.end(token)

    def state_dict(self) -> Dict[str, Any]:
        return {"started": self.started}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if set(state) != {"started"}:
            raise ValueError(f"direct response state keys {sorted(state)} != ['started']")
        self.started = bool(state["started"])


def _buffer_pairs(ema, live):
    live_b, ema_b = dict(live.named_buffers()), dict(ema.named_buffers())
    if live_b.keys() != ema_b.keys():
        raise ValueError("ema_critic buffer names differ from critic's")
    for name, b in live_b.items():
        if ema_b[name].shape != b.shape or ema_b[name].dtype != b.dtype:
            raise ValueError(f"ema_critic buffer {name} differs in shape or dtype")
    return [(ema_b[name], b) for name, b in live_b.items()]


class RobustCriticAnchor(CriticAnchor):
    """``CriticAnchor`` that also averages buffers and never mutates state on forward.

    Floating-point buffers (e.g. BatchNorm running statistics) are averaged
    with the parameters; integer buffers are copied. Every EMA forward runs
    in the live critic's per-module train/eval mode and restores any EMA
    buffer it changed (BatchNorm statistics, spectral-norm ``u``/``v``), so
    evaluating the anchor never changes the EMA or the live critic. No
    ``.data`` swapping is involved.
    """

    def __init__(self, critic, ema_critic, decay=0.999):
        super().__init__(critic, ema_critic, decay)
        self._buffer_pairs = _buffer_pairs(ema_critic, critic)
        ema_modules, live_modules = list(ema_critic.modules()), list(critic.modules())
        if len(ema_modules) != len(live_modules):
            raise ValueError("ema_critic module structure differs from critic's")
        self._module_pairs = list(zip(ema_modules, live_modules))
        # (owning module, buffer name) for every EMA buffer.
        self._owned = [(module, name) for module in ema_modules
                       for name, buffer in module._buffers.items() if buffer is not None]

    @torch.no_grad()
    def start_(self):
        super().start_()
        for e, b in self._buffer_pairs:
            e.copy_(b)

    @torch.no_grad()
    def update_(self):
        super().update_()
        for e, b in self._buffer_pairs:
            if e.is_floating_point():
                e.mul_(self.decay).add_(b, alpha=1.0 - self.decay)
            else:
                e.copy_(b)

    def forward(self, fn, x):
        """Evaluate ``fn(ema_critic, x)`` without side effects on any module state."""
        modes = [(e, e.training) for e, _ in self._module_pairs]
        for e, live in self._module_pairs:
            e.training = live.training
        # The forward runs on private copies of every EMA buffer: a train-mode
        # BatchNorm or spectral norm updates (and may save for backward) the
        # copies, and the EMA's own buffers are put back untouched afterwards.
        # No in-place restore, so the caller's later input-gradient is valid.
        originals = [(module, name, module._buffers[name]) for module, name in self._owned]
        try:
            for module, name, buffer in originals:
                module._buffers[name] = buffer.clone()
            return fn(self.ema_critic, x)
        finally:
            for module, name, buffer in originals:
                module._buffers[name] = buffer
            for e, flag in modes:
                e.training = flag

    def __call__(self, x):
        return self.forward(lambda module, inputs: module(inputs), x)


def _adam_step(optimizer):
    """Adam's own update for ``optimizer`` without re-running optimizer step hooks.

    ``torch.optim`` wraps each class's ``step`` once to run step hooks; our
    subclasses' ``step`` is wrapped too, so call the unwrapped Adam update.
    """
    step = Adam.step
    if getattr(step, "hooked", False):
        step = step.__wrapped__
    return step(optimizer)


def _closure_loss(closure):
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()


class K3PCriticAdam(Adam):
    """Adam for one critic whose ``step()`` also does K3P's critic-side work.

    Built by ``recipe.make_critic_optimizer(critic, ema_critic=...)`` (and by
    ``recipe.make_optimizers``). ``step()`` = spike guard on the gradients,
    the Adam update, then the anchor EMA update and the LR record that the
    paired penalty (``recipe.make_critic_penalty(optimizer)``) reads.
    ``ema_critic`` is the caller-allocated EMA module (e.g.
    ``copy.deepcopy(critic)``); None keeps no anchor. ``state_dict()`` holds
    everything (Adam state, EMA critic, LR record, counters) under the extra
    ``"regularizer"`` key, so the usual optimizer checkpoint resumes exactly.
    """

    _OWN_ATTRS = ("critic", "ema_critic", "anchor", "record", "guard")

    def __init__(self, params, *, critic, ema_critic=None, anchor_decay=0.999,
                 guard_ratio=5.0, guard_min_steps=200, **adam_kwargs):
        super().__init__(params, **adam_kwargs)
        if not isinstance(critic, nn.Module):
            raise TypeError("critic must be an nn.Module")
        self.critic, self.ema_critic = critic, ema_critic
        self.anchor = None
        if ema_critic is not None:
            ema_critic.requires_grad_(False)
            self.anchor = RobustCriticAnchor(critic, ema_critic, decay=anchor_decay)
        self.record = CriticStepRecord(self.anchor)
        self.guard = None if guard_ratio == 0 else CriticSpikeGuard(ratio=guard_ratio, min_steps=guard_min_steps)

    def __getstate__(self):
        # Optimizer pickles/deep-copies only defaults/state/param_groups; keep ours.
        state = super().__getstate__()
        state.update({key: self.__dict__[key] for key in self._OWN_ATTRS})
        return state

    def step(self, closure=None):
        loss = _closure_loss(closure)
        if self.guard is not None:
            self.guard.apply_(self)
        _adam_step(self)
        self.record.record_step(self)
        return loss

    def state_dict(self):
        state = super().state_dict()
        state[_STATE_KEY] = {
            "record": self.record.state_dict(),
            "ema": None if self.ema_critic is None else self.ema_critic.state_dict(),
            "guard": None if self.guard is None else self.guard.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop(_STATE_KEY, None)
        if not isinstance(extra, dict) or set(extra) != {"record", "ema", "guard"}:
            raise ValueError("critic optimizer state has no valid 'regularizer' entry")
        if (extra["ema"] is None) != (self.ema_critic is None) or (extra["guard"] is None) != (self.guard is None):
            raise ValueError("critic optimizer state does not match this recipe/EMA critic")
        # Validate every part before mutating anything.
        copy(self.record).load_state_dict(extra["record"])
        if self.guard is not None:
            copy(self.guard).load_state_dict(dict(extra["guard"]))
        if self.ema_critic is not None:
            deepcopy(self.ema_critic).load_state_dict(extra["ema"])
        super().load_state_dict(state_dict)
        self.record.load_state_dict(extra["record"])
        if self.guard is not None:
            self.guard.load_state_dict(dict(extra["guard"]))
        if self.ema_critic is not None:
            self.ema_critic.load_state_dict(extra["ema"])


class K3PGeneratorAdam(Adam):
    """Adam for the generator side whose ``step()`` applies K3P's update modifications.

    Built by ``recipe.make_generator_optimizer(params, latent_table=...,
    direct_particles=...)`` (and by ``recipe.make_optimizers``, which passes a
    particle prior's table). A2 ``LatentRowDamping`` acts on a sparse latent
    table (alone in its param group with beta1 == 0) and
    ``DirectParticleResponse`` on one param group of direct sample particles;
    their history buffers are allocated here. With neither, ``step()`` is
    exactly ``Adam.step()``. ``state_dict()`` carries the histories and
    counters under the extra ``"regularizer"`` key.
    """

    _OWN_ATTRS = ("latent_damping", "latent_history", "direct_response", "direct_history")

    def __init__(self, params, *, latent_table=None, direct_particles=None, latent_max_rate=0.5,
                 direct_betas=(0.0, 0.9), direct_gain=True, **adam_kwargs):
        super().__init__(params, **adam_kwargs)
        self.latent_damping = self.latent_history = None
        self.direct_response = self.direct_history = None
        if latent_table is not None and latent_table.requires_grad and latent_max_rate > 0:
            group = _group_of(self, latent_table)
            if len(group["params"]) != 1 or group["betas"][0] != 0.0:
                raise ValueError("A2 latent damping needs the prior table alone with beta1 == 0; "
                                 "set latent_damping_max_rate=0 to train without it")
            self.latent_history = torch.zeros_like(latent_table, requires_grad=False)
            self.latent_damping = LatentRowDamping(latent_table, self.latent_history, max_rate=latent_max_rate)
        if direct_particles is not None:
            particles = list(direct_particles)
            if not particles:
                raise ValueError("direct_particles must contain at least one parameter")
            self.direct_history = torch.zeros(sum(p.numel() for p in particles),
                                              dtype=particles[0].dtype, device=particles[0].device)
            self.direct_response = DirectParticleResponse(particles, self.direct_history, betas=direct_betas,
                                                          gain=direct_gain)
            self.direct_response._group(self)  # validate: exactly one param group

    def __getstate__(self):
        # Optimizer pickles/deep-copies only defaults/state/param_groups; keep ours.
        state = super().__getstate__()
        state.update({key: self.__dict__[key] for key in self._OWN_ATTRS})
        return state

    def step(self, closure=None):
        loss = _closure_loss(closure)
        latent = None if self.latent_damping is None else self.latent_damping.begin(self)
        try:
            direct = None if self.direct_response is None else self.direct_response.begin(self)
            try:
                _adam_step(self)
            finally:
                if self.direct_response is not None:
                    self.direct_response.end(direct)
        finally:
            if self.latent_damping is not None:
                self.latent_damping.end(latent)
        return loss

    def state_dict(self):
        state = super().state_dict()
        state[_STATE_KEY] = {
            "latent": None if self.latent_damping is None else
            {"state": self.latent_damping.state_dict(), "history": self.latent_history},
            "direct": None if self.direct_response is None else
            {"state": self.direct_response.state_dict(), "history": self.direct_history},
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop(_STATE_KEY, None)
        if not isinstance(extra, dict) or set(extra) != {"latent", "direct"}:
            raise ValueError("generator optimizer state has no valid 'regularizer' entry")
        parts = ((extra["latent"], self.latent_damping, self.latent_history),
                 (extra["direct"], self.direct_response, self.direct_history))
        for part, owner, history in parts:
            if (part is None) != (owner is None):
                raise ValueError("generator optimizer state does not match this recipe")
            if part is not None:
                if (not isinstance(part, dict) or set(part) != {"state", "history"}
                        or not isinstance(part["history"], torch.Tensor)
                        or part["history"].shape != history.shape or part["history"].dtype != history.dtype):
                    raise ValueError("generator optimizer history does not match")
                copy(owner).load_state_dict(dict(part["state"]))  # validate before mutating
        super().load_state_dict(state_dict)
        for part, owner, history in parts:
            if part is not None:
                owner.load_state_dict(dict(part["state"]))
                with torch.no_grad():
                    history.copy_(part["history"])


def _no_anchor(x):
    raise RuntimeError("this critic optimizer has no EMA critic")


def _first_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


class CriticPenalty:
    """The recipe's critic penalty, paired with one critic optimizer.

    Built by ``recipe.make_critic_penalty(opt_d)``; call it like a loss::

        d_loss = adv + penalty(D, real, fake)                      # plain critic
        d_loss = adv + penalty(D, x, fake, labels, xt=xt, t=t)     # conditional critic

    Extra positional/keyword arguments are forwarded as conditioning to the
    critic and to the paired EMA critic. ``D`` is the optimizer's critic, one of
    its submodules (a role of a shared module; the same-named EMA submodule is
    used), or a module wrapping one of those (e.g. ``InputNoise(D)``; the EMA
    is evaluated through a shallow copy of the wrapper). A tuple/list output
    uses its first element unless ``output=`` selects the logits. The step
    used for lazy application is the optimizer's completed step count + 1, so
    several calls per critic step (roles, views) share one step. Returns the
    scalar penalty; ``last_stats`` holds the stats of the last call when
    ``collect_stats`` is set, ``diagnostics()`` host scalars.
    """

    def __init__(self, recipe, optimizer, *, output=None, collect_stats=False, **penalty_overrides):
        if not isinstance(optimizer, K3PCriticAdam):
            raise TypeError("optimizer must come from recipe.make_critic_optimizer or recipe.make_optimizers")
        self.optimizer, self.critic = optimizer, optimizer.critic
        options = recipe._penalty_options(**penalty_overrides)
        if optimizer.anchor is None and options["anchor_weight"] != 0:
            raise ValueError("this penalty needs the critic's EMA: pass ema_critic=copy.deepcopy(critic) "
                             "to recipe.make_optimizers / recipe.make_critic_optimizer")
        self.regularizer = GradientPenalty(record=optimizer.record, **options)
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats = {}
        self._names = {id(module): name for name, module in self.critic.named_modules()}

    @property
    def ema_critic(self):
        """The paired optimizer's EMA critic module (None without one)."""
        return self.optimizer.ema_critic

    def _ema_view(self, critic):
        """``m -> module`` mapping the EMA root to the EMA counterpart of ``critic``."""
        if isinstance(critic, nn.Module):
            name = self._names.get(id(critic))
            if name is not None:
                return lambda m: m.get_submodule(name)
            for key, child in critic._modules.items():
                inner = None if child is None else self._names.get(id(child))
                if inner is not None:
                    def view(m, key=key, inner=inner):
                        clone = copy(critic)
                        clone._modules = dict(critic._modules)
                        clone._modules[key] = m.get_submodule(inner)
                        return clone
                    return view
        raise TypeError("pass the critic paired with this penalty's optimizer, one of its submodules, "
                        "or a module wrapping one of those")

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        output = self.output

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))
        options = {}
        anchor = self.optimizer.anchor
        if anchor is not None:
            view = self._ema_view(critic)
            options["ema_critic"] = lambda x: anchor.forward(
                lambda m, inputs: output(view(m)(inputs, *condition, **condition_kwargs)), x)
        else:  # reg_anchor_weight == 0: the kernel never evaluates an anchor
            options["ema_critic"] = _no_anchor
        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(live, x_real, x_fake, step, self.collect_stats, **options)
        self.last_stats = stats
        return penalty

    def diagnostics(self):
        """Host-side scalars for logging (blend weight; guard clip count)."""
        out = {"blend_weight": float(self.regularizer.blend_weight())}
        if self.optimizer.guard is not None:
            out["clipped_tensors"] = self.optimizer.guard.clipped_tensors
        return out
