"""K3P components: explicit, per-critic, hook-free.

The K3P penalty itself is ``GradRegularizer(arm="k3p")``; this module holds the
stateful pieces around it. Nothing here registers optimizer hooks, keeps
module-level state or allocates networks/buffers on its own: the caller passes
in the EMA critic and history buffers, and saves/restores them together with
each object's scalar ``state_dict()``. ``GANTrainer`` does that wiring for you.

* ``CriticAnchor``      -- parameter EMA Dbar of one critic (K3P's prox anchor).
* ``CriticSpikeGuard``  -- per-tensor gradient-spike clip before a critic Adam step.
* ``LatentRowDamping``  -- A2: bounded coherence damping of sparse latent-table rows.
* ``DirectParticleResponse`` -- LR gain for direct sample-particle groups.

Custom loop, one critic (repeat per critic optimizer)::

    ema = copy.deepcopy(D).requires_grad_(False)
    anchor = CriticAnchor(D, ema)
    reg = GradRegularizer(arm="k3p", anchor=anchor)
    guard = CriticSpikeGuard()
    ...
    loss_d = adv + reg(D, x_real, x_fake, step)
    loss_d.backward()
    guard.apply_(opt_d)
    opt_d.step()
    reg.after_critic_step(opt_d)
"""
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CriticAnchor", "CriticSpikeGuard", "LatentRowDamping", "DirectParticleResponse"]


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
