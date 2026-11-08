"""BISPF critic state: step records, the critic forward hook and the critic Adam optimizer."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam
from .thermal import _LEGACY_THERMAL_CHANNEL, _publish_thermal

# Historical BISPF controller packets without ``thermal_scale`` reconstruct it
# as clamp(ema_residual / _LEGACY_THERMAL_TAU, _LEGACY_THERMAL_FLOOR, 1).
_LEGACY_THERMAL_FLOOR = 0.05
_LEGACY_THERMAL_TAU = 0.020


class _BISPFObserverStepRecord(CriticStepRecord):
    """The existing passive observer law with complete BISPF continuation.

    Keep ``CriticStepRecord.record_step`` and the penalty's original observer
    fallback unchanged. Older passive checkpoints omitted earned transport
    moments; they cannot reconstruct an exact continuation from step counts.
    """

    def __init__(self):
        super().__init__()
        self.ema_residual = 1.0
        self.ema_data_scale = 1.0
        self.initialized = False
        self._thermal_scale = 1.0
        self._last_d_logits = None

    def state_dict(self):
        return {**super().state_dict(), "bispf": {
            "schema": 1, "ema_residual": self.ema_residual,
            "ema_data_scale": self.ema_data_scale, "initialized": self.initialized,
            "thermal_scale": self._thermal_scale,
        }}

    @staticmethod
    def base_state(state):
        if not isinstance(state, dict) or "bispf" not in state:
            raise ValueError("BISPF observer checkpoint is missing transport observer state; "
                             "resume the historical checkpoint with its original package source")
        if set(state) != {*CriticStepRecord.KEYS, "bispf"}:
            raise ValueError("invalid BISPF observer record fields")
        observer = state["bispf"]
        if (not isinstance(observer, dict)
                or set(observer) != {"schema", "ema_residual", "ema_data_scale", "initialized", "thermal_scale"}
                or type(observer["schema"]) is not int or observer["schema"] != 1
                or type(observer["initialized"]) is not bool):
            raise ValueError("invalid BISPF observer metadata")
        for key in ("ema_residual", "ema_data_scale", "thermal_scale"):
            value = observer[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid BISPF observer value")
        if observer["thermal_scale"] > 1:
            raise ValueError("invalid BISPF observer thermal scale")
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state):
        super().load_state_dict(self.base_state(state))
        observer = state["bispf"]
        self.ema_residual = float(observer["ema_residual"])
        self.ema_data_scale = float(observer["ema_data_scale"])
        self.initialized = observer["initialized"]
        self._thermal_scale = float(observer["thermal_scale"])
        self._last_d_logits = None

    def restore_device_state(self, device):
        _publish_thermal(self, device, self._thermal_scale)


class BISPFStepRecord(CriticStepRecord):
    """Per-critic BISPF state traveling with ``optimizer.state_dict()``."""

    _COUNTERS = ("calls", "observed_steps")
    _OPTIONAL = ("lr_last",)
    KEYS = (
        "formulation",
        "lr_max",
        "lr_last",
        "anchor_started",
        "calls",
        "observed_steps",
        "ema_residual",
        "ema_data_scale",
        "initialized",
    )

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self.ema_residual: float = 1.0
        self.ema_data_scale: float = 1.0
        self.initialized: bool = False
        self._last_d_logits: Tuple[torch.Tensor, torch.Tensor] | None = None

    def observe_sinkhorn(self, res: float, data_scale: float, ema_decay: float = 0.95) -> None:
        if not self.initialized:
            self.ema_residual = float(res)
            self.ema_data_scale = float(data_scale)
            self.initialized = True
        elif float(res) > float(self.ema_residual):
            self.ema_residual = float(res)
            self.ema_data_scale = float(data_scale)
        else:
            self.ema_residual = ema_decay * self.ema_residual + (1.0 - ema_decay) * float(res)
            self.ema_data_scale = ema_decay * self.ema_data_scale + (1.0 - ema_decay) * float(data_scale)

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._last_d_logits = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "formulation": "bispf",
            "lr_max": float(self.lr_max),
            "lr_last": None if self.lr_last is None else float(self.lr_last),
            "anchor_started": bool(self.anchor_started),
            "calls": int(self.calls),
            "observed_steps": int(self.observed_steps),
            "ema_residual": float(self.ema_residual),
            "ema_data_scale": float(self.ema_data_scale),
            "initialized": bool(self.initialized),
            "thermal_scale": float(getattr(self, "_thermal_scale", 1.0)),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("formulation") != "bispf":
            raise ValueError("BISPF critic optimizer requires a BISPF controller checkpoint")
        if set(state) not in (set(self.KEYS), {*self.KEYS, "thermal_scale"}):
            raise ValueError("invalid BISPF controller state keys")
        if "thermal_scale" in state:
            value = state["thermal_scale"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid BISPF controller thermal scale")
        for key in self._COUNTERS:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid BISPF controller {key}")
        for key in ("lr_max", "ema_residual", "ema_data_scale", *self._OPTIONAL):
            value = state[key]
            if value is None and key in self._OPTIONAL:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid BISPF controller {key}")
        if type(state["anchor_started"]) is not bool or type(state["initialized"]) is not bool:
            raise ValueError("invalid BISPF controller boolean flags")
        for key in self.KEYS:
            if key != "formulation":
                setattr(self, key, state[key])
        self._last_d_logits = None
        # Historical formulation packets retain their documented reconstruction;
        # newly written packets preserve the exact earned scale.
        self._thermal_scale = float(state.get("thermal_scale", max(_LEGACY_THERMAL_FLOOR, min(1.0, float(self.ema_residual) / _LEGACY_THERMAL_TAU))))

def _bispf_critic_forward_hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> Any:
    rec = getattr(module, "_bispf_record", None)
    if args and isinstance(args[0], torch.Tensor):
        target = output[0] if isinstance(output, (tuple, list)) and output else output
        if isinstance(target, torch.Tensor):
            if not hasattr(target, "_bispf_input"):
                target._bispf_input = args[0]
            if rec is not None and not hasattr(target, "_bispf_record"):
                target._bispf_record = rec
    if rec is not None and not getattr(rec, "_in_penalty", False) and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and isinstance(output, torch.Tensor) and output.requires_grad:
            out_scaled = output * s + output.detach() * (1.0 - s)
            if hasattr(output, "_bispf_input"):
                out_scaled._bispf_input = output._bispf_input
            if hasattr(output, "_bispf_record"):
                out_scaled._bispf_record = output._bispf_record
            return out_scaled
    return None


def _attach_critic_input_hook(critic: nn.Module, record: CriticStepRecord) -> None:
    """Attach a lightweight forward hook so SinkhornRelativisticLoss can read input coordinates."""
    record._thermal_scale = 1.0
    record._in_penalty = False
    if getattr(record, "legacy_thermal_channel", True):
        _LEGACY_THERMAL_CHANNEL.clear()
    critic._bispf_record = record
    if getattr(critic, "_bispf_hook_registered", False):
        return
    critic.register_forward_hook(_bispf_critic_forward_hook)
    critic._bispf_hook_registered = True


class BISPFCriticAdam(K3PCriticAdam):
    """Critic Adam optimizer for ``critic_formulation='bispf'`` with no EMA critic copy."""

    def __init__(
        self,
        params,
        *,
        critic: nn.Module,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs,
    ) -> None:
        record = BISPFStepRecord()
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = record
        _attach_critic_input_hook(critic, record)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        device = self.param_groups[0]["params"][0].device
        _publish_thermal(self.record, device, self.record._thermal_scale)
