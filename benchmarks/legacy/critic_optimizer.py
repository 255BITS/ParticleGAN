"""The EMA-critic anchor and the critic optimizer that carries it (legacy K3P).

ParticleGAN's shipped penalty has no EMA anchor, so the package critic
optimizer keeps no EMA critic. The archived K3P formulation replayed through
``LegacyRecipe`` (the phase-A/B blend of ``grad_regularizers.GradientPenalty``)
evaluates a parameter-EMA critic; ``LegacyCriticAdam`` is the package critic
optimizer plus that EMA, exactly as it was before the anchor was removed.
Benchmarks only.
"""
from copy import copy, deepcopy
from typing import Any

import torch
import torch.nn as nn

from particlegan.k3p import K3PCriticAdam

from .grad_regularizers import CriticStepRecord

__all__ = ["CriticAnchor", "RobustCriticAnchor", "LegacyCriticAdam"]

_STATE_KEY = "regularizer"


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


class LegacyCriticAdam(K3PCriticAdam):
    """``K3PCriticAdam`` with the caller-allocated EMA critic (``ema_critic``) the legacy K3P anchor reads.

    ``step()`` = spike guard, Adam update, then the step record, which also
    advances the anchor EMA once the paired legacy penalty has started it.
    ``state_dict()`` carries the EMA critic under ``"regularizer"``.
    """

    _OWN_ATTRS = ("critic", "ema_critic", "anchor", "record", "guard")

    def __init__(self, params, *, critic, ema_critic=None, anchor_decay=0.999, **kwargs):
        super().__init__(params, critic=critic, **kwargs)
        self.ema_critic, self.anchor = ema_critic, None
        if ema_critic is not None:
            ema_critic.requires_grad_(False)
            self.anchor = RobustCriticAnchor(critic, ema_critic, decay=anchor_decay)
        self.record = CriticStepRecord(self.anchor)

    def state_dict(self):
        state = super().state_dict()
        ema = None if self.ema_critic is None else self.ema_critic.state_dict()
        state[_STATE_KEY] = {**state[_STATE_KEY], "ema": ema}
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.get(_STATE_KEY)
        if not isinstance(extra, dict) or set(extra) != {"record", "ema", "guard"}:
            raise ValueError("critic optimizer state has no valid 'regularizer' entry")
        if (extra["ema"] is None) != (self.ema_critic is None) or (extra["guard"] is None) != (self.guard is None):
            raise ValueError("critic optimizer state does not match this recipe/EMA critic")
        # Validate every part before mutating anything.
        copy(self.record).load_state_dict(extra["record"])
        if self.ema_critic is not None:
            deepcopy(self.ema_critic).load_state_dict(extra["ema"])
        state_dict[_STATE_KEY] = {key: value for key, value in extra.items() if key != "ema"}
        super().load_state_dict(state_dict)
        if self.ema_critic is not None:
            self.ema_critic.load_state_dict(extra["ema"])
