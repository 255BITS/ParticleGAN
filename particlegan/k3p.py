"""K3P: the critic/generator regularization ParticleGAN trains with.

Users do not instantiate these classes. The recipe builds them behind
formulation-agnostic factories and a plain PyTorch loop::

    opt_g, opt_d = recipe.make_optimizers(G, D, prior)
    penalty = recipe.make_critic_penalty(opt_d)
    d_loss = adv_d + penalty(D, real, fake)
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()

The optimizers' ``step()`` does all step-time work and their ``state_dict()``
holds all state. The classes stay importable here for tests and research.
Nothing registers optimizer hooks or keeps module-level state.

* ``CriticSpikeGuard``  -- per-tensor gradient-spike clip before a critic Adam step.
* ``LatentRowDamping``  -- A2: bounded coherence damping of sparse latent-table rows.
* ``K3PCriticAdam`` / ``K3PGeneratorAdam`` -- the Adam subclasses the recipe's
  optimizer factories return; ``CriticPenalty`` -- the penalty paired with a
  ``K3PCriticAdam`` (``recipe.make_critic_penalty``).
"""
from contextlib import contextmanager
from copy import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from .grad_regularizers import CriticStepRecord, GradientPenalty

__all__ = ["CriticSpikeGuard", "LatentRowDamping", "K3PCriticAdam", "K3PGeneratorAdam", "CriticPenalty"]

# Extra optimizer.state_dict() key holding the regularization state.
_STATE_KEY = "regularizer"


def _group_of(optimizer, param) -> Dict[str, Any]:
    for group in optimizer.param_groups:
        if any(p is param for p in group["params"]):
            return group
    raise ValueError("parameter is not in this optimizer")


class CriticSpikeGuard:
    """Scale down critic gradient spikes before an Adam step.

    Call ``apply_(optimizer)`` between ``backward()`` and ``optimizer.step()``.
    A tensor with at least ``min_steps`` prior Adam steps whose gradient RMS
    exceeds ``ratio * sqrt(mean bias-corrected v)`` is scaled to that ratio;
    ``v`` is Adam's running second moment ``exp_avg_sq`` (also under AMSGrad,
    whose update uses the running max instead); everything else is
    multiplied by exactly 1.0. Tensors without Adam state yet are skipped.
    Returns the number of clipped tensors this step (a tensor, no host sync).
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

    Built by ``recipe.make_critic_optimizer(critic)`` (and by
    ``recipe.make_optimizers``). ``step()`` = spike guard on the gradients,
    the Adam update, then the step record that the paired penalty
    (``recipe.make_critic_penalty(optimizer)``) reads. ``state_dict()`` holds
    everything (Adam state, step record, guard counter) under the extra
    ``"regularizer"`` key, so the usual optimizer checkpoint resumes exactly.
    """

    _OWN_ATTRS = ("critic", "record", "guard")

    def __init__(self, params, *, critic, guard_ratio=5.0, guard_min_steps=200, **adam_kwargs):
        super().__init__(params, **adam_kwargs)
        if not isinstance(critic, nn.Module):
            raise TypeError("critic must be an nn.Module")
        self.critic = critic
        self.record = CriticStepRecord()
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
            "guard": None if self.guard is None else self.guard.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop(_STATE_KEY, None)
        if isinstance(extra, dict) and "ema" in extra:
            # Checkpoints from the EMA-anchor formulation also carry the EMA
            # critic; this formulation has no anchor, so it is dropped.
            extra = {key: value for key, value in extra.items() if key != "ema"}
        if not isinstance(extra, dict) or set(extra) != {"record", "guard"}:
            raise ValueError("critic optimizer state has no valid 'regularizer' entry")
        if (extra["guard"] is None) != (self.guard is None):
            raise ValueError("critic optimizer state does not match this recipe")
        # Validate every part before mutating anything.
        copy(self.record).load_state_dict(extra["record"])
        if self.guard is not None:
            copy(self.guard).load_state_dict(dict(extra["guard"]))
        super().load_state_dict(state_dict)
        self.record.load_state_dict(extra["record"])
        if self.guard is not None:
            self.guard.load_state_dict(dict(extra["guard"]))


class K3PGeneratorAdam(Adam):
    """Adam for the generator side whose ``step()`` applies K3P's A2 latent damping.

    Built by ``recipe.make_generator_optimizer(params, latent_table=...)`` (and
    by ``recipe.make_optimizers``, which passes a particle prior's table). A2
    ``LatentRowDamping`` acts on a sparse latent table (alone in its param
    group with beta1 == 0); its history buffer is allocated here. Without a
    table, ``step()`` is exactly ``Adam.step()``. ``state_dict()`` carries the
    history and counters under the extra ``"regularizer"`` key.
    """

    _OWN_ATTRS = ("latent_damping", "latent_history")

    def __init__(self, params, *, latent_table=None, latent_max_rate=0.5, **adam_kwargs):
        super().__init__(params, **adam_kwargs)
        self.latent_damping = self.latent_history = None
        if latent_table is not None and latent_table.requires_grad and latent_max_rate > 0:
            group = _group_of(self, latent_table)
            if len(group["params"]) != 1 or group["betas"][0] != 0.0:
                raise ValueError("A2 latent damping needs the prior table alone with beta1 == 0; "
                                 "set latent_damping_max_rate=0 to train without it")
            self.latent_history = torch.zeros_like(latent_table, requires_grad=False)
            self.latent_damping = LatentRowDamping(latent_table, self.latent_history, max_rate=latent_max_rate)

    def __getstate__(self):
        # Optimizer pickles/deep-copies only defaults/state/param_groups; keep ours.
        state = super().__getstate__()
        state.update({key: self.__dict__[key] for key in self._OWN_ATTRS})
        return state

    def step(self, closure=None):
        loss = _closure_loss(closure)
        latent = None if self.latent_damping is None else self.latent_damping.begin(self)
        try:
            _adam_step(self)
        finally:
            if self.latent_damping is not None:
                self.latent_damping.end(latent)
        return loss

    def state_dict(self):
        state = super().state_dict()
        state[_STATE_KEY] = {
            "latent": None if self.latent_damping is None else
            {"state": self.latent_damping.state_dict(), "history": self.latent_history},
        }
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop(_STATE_KEY, None)
        if isinstance(extra, dict) and "direct" in extra and extra["direct"] is None:
            extra = {k: v for k, v in extra.items() if k != "direct"}  # the removed direct response, unused
        if not isinstance(extra, dict) or set(extra) != {"latent"}:
            raise ValueError("generator optimizer state has no valid 'regularizer' entry")
        part, owner, history = extra["latent"], self.latent_damping, self.latent_history
        if (part is None) != (owner is None):
            raise ValueError("generator optimizer state does not match this recipe")
        if part is not None:
            if (not isinstance(part, dict) or set(part) != {"state", "history"}
                    or not isinstance(part["history"], torch.Tensor)
                    or part["history"].shape != history.shape or part["history"].dtype != history.dtype):
                raise ValueError("generator optimizer history does not match")
            copy(owner).load_state_dict(dict(part["state"]))  # validate before mutating
        super().load_state_dict(state_dict)
        if part is not None:
            owner.load_state_dict(dict(part["state"]))
            with torch.no_grad():
                history.copy_(part["history"])


def _first_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


class CriticPenalty:
    """The recipe's critic penalty, paired with one critic optimizer.

    Built by ``recipe.make_critic_penalty(opt_d)``; call it like a loss::

        d_loss = adv + penalty(D, real, fake)                      # plain critic
        d_loss = adv + penalty(D, x, fake, labels, xt=xt, t=t)     # conditional critic

    Extra positional/keyword arguments are forwarded as conditioning to the
    critic. ``D`` is the critic (or a role or wrapper of it, e.g.
    ``InputNoise(D)``) whose input gradient is penalized. A tuple/list output
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
        self.regularizer = GradientPenalty(record=optimizer.record, **recipe._penalty_options(**penalty_overrides))
        self.output = _first_output if output is None else output
        self.collect_stats = bool(collect_stats)
        self.last_stats = {}

    def __call__(self, critic, x_real, x_fake, *condition, **condition_kwargs):
        output = self.output

        def live(x):
            return output(critic(x, *condition, **condition_kwargs))
        step = self.optimizer.record.observed_steps + 1
        penalty, stats = self.regularizer.penalty(live, x_real, x_fake, step, self.collect_stats)
        self.last_stats = stats
        return penalty

    def diagnostics(self):
        """Host-side scalars for logging (guard clip count)."""
        out = {}
        if self.optimizer.guard is not None:
            out["clipped_tensors"] = self.optimizer.guard.clipped_tensors
        return out
