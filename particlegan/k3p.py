"""K3P: the current best critic/generator regularization formulation.

Users do not instantiate these classes: ``recipe.make_critic_regularizer(D,
opt_d)`` and ``recipe.make_generator_regularizer(opt_g, latent_table=prior.z)``
return the recipe's current formulation behind a formulation-agnostic
interface (``penalty``/``before_step``/``after_step``/``step``/``state_dict``).
The classes stay importable from this module for low-level tests and research.

The K3P penalty itself is ``GradRegularizer(arm="k3p")``. Nothing here
registers optimizer hooks or keeps module-level state.

* ``CriticAnchor``      -- parameter EMA Dbar of one critic (K3P's prox anchor);
  ``RobustCriticAnchor`` also averages buffers with side-effect-free forwards.
* ``CriticSpikeGuard``  -- per-tensor gradient-spike clip before a critic Adam step.
* ``LatentRowDamping``  -- A2: bounded coherence damping of sparse latent-table rows.
* ``DirectParticleResponse`` -- LR gain for direct sample-particle groups.
* ``K3PCritic`` / ``K3PGeneratorRegularizer`` -- the per-optimizer bundles the
  recipe factories return.
"""
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CriticAnchor", "RobustCriticAnchor", "CriticSpikeGuard", "LatentRowDamping",
           "DirectParticleResponse", "K3PCritic", "K3PGeneratorRegularizer"]


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
    optimizer step: ``with damping.around(opt_g): opt_g.step()``.
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
    one optimizer param group. Wrap the step: ``with resp.around(opt): opt.step()``.
    """

    def __init__(self, params: Sequence[nn.Parameter], history: torch.Tensor,
                 betas: Tuple[float, float] = (0.0, 0.9)) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("DirectParticleResponse needs at least one parameter")
        numel = sum(p.numel() for p in self.params)
        if history.dim() != 1 or history.numel() != numel:
            raise ValueError(f"history must be a flat tensor of {numel} elements")
        if history.dtype != self.params[0].dtype or history.device != self.params[0].device:
            raise ValueError("history must match the parameters' dtype and device")
        self.history, self.betas = history, (float(betas[0]), float(betas[1]))
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
        if self.started:
            cosine = float(F.cosine_similarity(current, self.history, dim=0, eps=1e-12))
            gain = 1.0 + max(0.0, min(1.0, cosine))
        self.history.copy_(current)
        self.started = True
        self.last_gain = gain
        token = (group, group["lr"], group["betas"])
        group["betas"] = self.betas
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


class K3PCritic:
    """K3P implementation of the critic regularizer for one critic optimizer.

    Build it with ``recipe.make_critic_regularizer(critic, optimizer)``; the
    recipe picks this class while K3P is the current best formulation.
    Allocates the EMA critic (``deepcopy(critic)``, frozen), a
    ``RobustCriticAnchor`` over it, the recipe's gradient penalty and critic
    spike guard. One module used in several roles shares one object and passes
    an ``ema_critic`` per role::

        reg = recipe.make_critic_regularizer(D, opt_d)
        loss = adv + reg.penalty(D, real, fake, step)[0]
        # shared module: reg.penalty(lambda x: D.role(x), xr, xf, step,
        #                            ema_critic=reg.ema_critic(lambda m, x: m.role(x)))
        opt_d.zero_grad(); loss.backward(); reg.step()

    Formulation-agnostic interface: ``penalty``, ``ema_critic``,
    ``before_step``/``after_step`` (or ``step()`` = before, ``optimizer.step()``,
    after), ``diagnostics``, ``state_dict``/``load_state_dict``. ``optimizer``
    may be None for penalty-only use (the step methods then raise).
    """

    def __init__(self, recipe, critic, optimizer, **penalty_overrides):
        from .recipes import Recipe
        if not isinstance(recipe, Recipe):
            raise TypeError("recipe must be a Recipe")
        self.critic, self.optimizer = critic, optimizer
        arm = penalty_overrides.get("arm", recipe.reg_arm)
        self.ema = self.anchor = None
        if arm == "k3p":
            self.ema = deepcopy(critic).requires_grad_(False)
            self.anchor = RobustCriticAnchor(critic, self.ema, decay=recipe.reg_anchor_decay)
        self.regularizer = recipe.make_gradient_penalty(anchor=self.anchor, **penalty_overrides)
        self.guard = (None if recipe.d_guard_ratio == 0
                      else CriticSpikeGuard(ratio=recipe.d_guard_ratio, min_steps=recipe.d_guard_min_steps))

    def ema_critic(self, fn=None):
        """Callable ``x -> fn(ema_module, x)`` (default ``ema_module(x)``), side-effect free.

        None when the formulation keeps no EMA critic.
        """
        if self.anchor is None:
            return None
        if fn is None:
            return self.anchor
        return lambda x: self.anchor.forward(fn, x)

    def penalty(self, D, x_real, x_fake, step, *, generator=None, collect_stats=False, ema_critic=None):
        """``(penalty, stats)`` for one critic (or one role of it)."""
        options = {} if ema_critic is None else {"ema_critic": ema_critic}
        return self.regularizer.penalty(D, x_real, x_fake, step, generator, collect_stats, **options)

    def _require_optimizer(self):
        if self.optimizer is None:
            raise RuntimeError("critic regularizer was built without an optimizer")
        return self.optimizer

    def before_step(self):
        """Call between ``backward()`` and ``optimizer.step()`` (spike guard)."""
        optimizer = self._require_optimizer()
        if self.guard is not None:
            self.guard.apply_(optimizer)

    def after_step(self):
        """Call right after ``optimizer.step()`` (anchor EMA, then the LR record)."""
        self.regularizer.after_critic_step(self._require_optimizer())

    def step(self):
        """``before_step()``, ``optimizer.step()``, ``after_step()``."""
        self.before_step()
        self.optimizer.step()
        self.after_step()

    def diagnostics(self):
        """Host-side scalars for logging (K3P: blend weight; guard clip count)."""
        out = {}
        if self.regularizer.arm == "k3p":
            out["blend_weight"] = float(self.regularizer.blend_weight())
        if self.guard is not None:
            out["clipped_tensors"] = self.guard.clipped_tensors
        return out

    def state_dict(self):
        return {"penalty": self.regularizer.state_dict(),
                "ema": None if self.ema is None else self.ema.state_dict(),
                "guard": None if self.guard is None else self.guard.state_dict()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"penalty", "ema", "guard"}:
            raise ValueError("invalid critic regularizer state")
        if (state["ema"] is None) != (self.ema is None) or (state["guard"] is None) != (self.guard is None):
            raise ValueError("critic regularizer state does not match this recipe")
        self.regularizer.load_state_dict(state["penalty"])
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        if self.guard is not None:
            self.guard.load_state_dict(state["guard"])


class K3PGeneratorRegularizer:
    """K3P implementation of the generator-side update for one generator optimizer.

    Build it with ``recipe.make_generator_regularizer(optimizer, latent_table=...,
    direct_particles=...)``. It allocates its own history buffers and applies
    A2 ``LatentRowDamping`` to a sparse latent table (e.g. ``prior.z``, alone in
    its Adam group with beta1 == 0) and/or ``DirectParticleResponse`` to one
    param group of direct sample particles. When neither applies (or damping
    is disabled by the recipe), ``step()`` is exactly ``optimizer.step()``.

    Formulation-agnostic interface: ``before_step()``/``after_step()`` around
    ``optimizer.step()`` (or ``step()``, or ``with reg.around(): opt.step()``),
    ``state_dict``/``load_state_dict`` (history buffers included).
    """

    def __init__(self, recipe, optimizer, *, latent_table=None, direct_particles=None):
        self.optimizer = optimizer
        self.latent_damping = self.latent_history = None
        self.direct_response = self.direct_history = None
        if latent_table is not None and latent_table.requires_grad and recipe.latent_damping_max_rate > 0:
            group = _group_of(optimizer, latent_table)
            if len(group["params"]) != 1 or group["betas"][0] != 0.0:
                raise ValueError("A2 latent damping needs the prior table alone with beta1 == 0; "
                                 "set latent_damping_max_rate=0 to train without it")
            self.latent_history = torch.zeros_like(latent_table, requires_grad=False)
            self.latent_damping = LatentRowDamping(latent_table, self.latent_history,
                                                   max_rate=recipe.latent_damping_max_rate)
        if direct_particles is not None:
            params = list(direct_particles)
            if not params:
                raise ValueError("direct_particles must contain at least one parameter")
            self.direct_history = torch.zeros(sum(p.numel() for p in params),
                                              dtype=params[0].dtype, device=params[0].device)
            self.direct_response = DirectParticleResponse(params, self.direct_history,
                                                          betas=recipe.direct_particle_betas)
        self._tokens = None

    def before_step(self):
        """Call between ``backward()`` and ``optimizer.step()``."""
        if self._tokens is not None:
            raise RuntimeError("before_step() called twice without after_step()")
        latent = None if self.latent_damping is None else self.latent_damping.begin(self.optimizer)
        try:
            direct = None if self.direct_response is None else self.direct_response.begin(self.optimizer)
        except BaseException:
            if self.latent_damping is not None:
                self.latent_damping.end(latent)
            raise
        self._tokens = (latent, direct)

    def after_step(self):
        """Call right after ``optimizer.step()``."""
        if self._tokens is None:
            raise RuntimeError("after_step() without before_step()")
        latent, direct = self._tokens
        self._tokens = None
        if self.direct_response is not None:
            self.direct_response.end(direct)
        if self.latent_damping is not None:
            self.latent_damping.end(latent)

    @contextmanager
    def around(self):
        self.before_step()
        try:
            yield
        finally:
            self.after_step()

    def step(self):
        """``optimizer.step()`` with the generator-side modifications."""
        with self.around():
            self.optimizer.step()

    def state_dict(self):
        return {
            "latent": None if self.latent_damping is None else
            {"state": self.latent_damping.state_dict(), "history": self.latent_history},
            "direct": None if self.direct_response is None else
            {"state": self.direct_response.state_dict(), "history": self.direct_history},
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"latent", "direct"}:
            raise ValueError("invalid generator regularizer state")
        parts = ((state["latent"], self.latent_damping, self.latent_history),
                 (state["direct"], self.direct_response, self.direct_history))
        for part, owner, history in parts:
            if (part is None) != (owner is None):
                raise ValueError("generator regularizer state does not match this recipe")
            if part is not None:
                if (not isinstance(part, dict) or set(part) != {"state", "history"}
                        or not isinstance(part["history"], torch.Tensor)
                        or part["history"].shape != history.shape or part["history"].dtype != history.dtype):
                    raise ValueError("generator regularizer history does not match")
                copy(owner).load_state_dict(dict(part["state"]))  # validate before mutating
        for part, owner, history in parts:
            if part is not None:
                owner.load_state_dict(dict(part["state"]))
                with torch.no_grad():
                    history.copy_(part["history"])
