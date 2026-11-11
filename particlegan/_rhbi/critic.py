"""RHBI critic state: step records, input/conditioning capture hook, and critic Adam optimizer."""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam

_MAX_RING_ENTRIES = 16
_RECENT_CAPTURES: "OrderedDict[Tuple[int, int, int, int], Tuple[torch.Tensor, Optional[torch.Tensor], Any]]" = OrderedDict()


def _storage_key(tensor: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.numel() == 0:
        return None
    try:
        ptr = int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return None
    if ptr == 0:
        return None
    return (ptr, int(tensor.storage_offset()), int(tensor.numel()), int(tensor.shape[0]))


def _extract_critic_inputs(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract concatenated floating batch inputs and optional discrete batch conditions."""
    if batch_size <= 0:
        return None, None
    float_parts = []
    disc_parts = []
    items = list(args)
    if kwargs:
        for key in sorted(kwargs):
            items.append(kwargs[key])
    for item in items:
        if not isinstance(item, torch.Tensor) or item.ndim < 1 or item.shape[0] != batch_size or item.numel() == 0:
            continue
        if item.is_floating_point():
            float_parts.append(item.reshape(batch_size, -1))
        elif not item.is_complex():
            disc_parts.append(item.detach().to(dtype=torch.int64).reshape(batch_size, -1))
    if not float_parts:
        return None, None
    x_in = float_parts[0] if len(float_parts) == 1 else torch.cat(float_parts, dim=-1)
    cond_disc = None if not disc_parts else (
        disc_parts[0] if len(disc_parts) == 1 else torch.cat(disc_parts, dim=-1)
    )
    return x_in, cond_disc


def _record_rhbi_capture(
    output: Any,
    x_input: Optional[torch.Tensor],
    cond_disc: Optional[torch.Tensor],
    record: Any,
    *,
    overwrite: bool = False,
) -> None:
    if x_input is None:
        return
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(target, torch.Tensor) or target.ndim < 1 or target.shape[0] != x_input.shape[0]:
        return
    if overwrite or not hasattr(target, "_rhbi_input"):
        target._rhbi_input = x_input
        target._rhbi_cond = cond_disc
    if record is not None and (overwrite or not hasattr(target, "_rhbi_record")):
        target._rhbi_record = record
    key = _storage_key(target)
    if key is None:
        return
    entry = (x_input, cond_disc, record)
    if record is not None and hasattr(record, "_capture_ring"):
        ring: OrderedDict = record._capture_ring
        if overwrite or key not in ring:
            ring[key] = entry
            ring.move_to_end(key)
            while len(ring) > _MAX_RING_ENTRIES:
                ring.popitem(last=False)
    if overwrite or key not in _RECENT_CAPTURES:
        _RECENT_CAPTURES[key] = entry
        _RECENT_CAPTURES.move_to_end(key)
        while len(_RECENT_CAPTURES) > _MAX_RING_ENTRIES:
            _RECENT_CAPTURES.popitem(last=False)


def _finalize_x_for_logits(logits: torch.Tensor, x_val: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x_val is None:
        return None
    if not logits.requires_grad and x_val.requires_grad:
        return x_val.detach()
    return x_val


def _lookup_rhbi_capture(
    logits: torch.Tensor,
    fallback_record: Any = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Any]:
    b = logits.shape[0] if logits.ndim >= 1 else 0
    x_attr = getattr(logits, "_rhbi_input", None)
    c_attr = getattr(logits, "_rhbi_cond", None)
    r_attr = getattr(logits, "_rhbi_record", None)
    if isinstance(x_attr, torch.Tensor) and x_attr.shape[0] == b:
        return _finalize_x_for_logits(logits, x_attr), c_attr, (r_attr if r_attr is not None else fallback_record)

    key = _storage_key(logits)
    if key is not None:
        for rec_candidate in (r_attr, fallback_record):
            ring = getattr(rec_candidate, "_capture_ring", None)
            if ring is not None and key in ring:
                x_c, c_c, r_c = ring[key]
                if x_c.shape[0] == b:
                    return _finalize_x_for_logits(logits, x_c), c_c, (r_c if r_c is not None else fallback_record)
        if key in _RECENT_CAPTURES:
            x_c, c_c, r_c = _RECENT_CAPTURES[key]
            if x_c.shape[0] == b:
                return _finalize_x_for_logits(logits, x_c), c_c, (r_c if r_c is not None else fallback_record)
    return None, None, (r_attr if r_attr is not None else fallback_record)


class _RHBIObserverStepRecord(CriticStepRecord):
    """Passive critic step record with complete RHBI horizon continuation."""

    def __init__(self) -> None:
        super().__init__()
        self._rhbi_ell_scalar: float = 0.0
        self._rhbi_ell_t: Optional[torch.Tensor] = None
        self.rhbi_ell_initialized: bool = False
        self._in_penalty: bool = False
        self._in_protected_probe: bool = False
        self._capture_ring: "OrderedDict[Tuple[int, int, int, int], Tuple[torch.Tensor, Optional[torch.Tensor], Any]]" = OrderedDict()

    @property
    def rhbi_ell_ema(self) -> float:
        if self._rhbi_ell_t is not None:
            return float(self._rhbi_ell_t.item())
        return self._rhbi_ell_scalar

    @rhbi_ell_ema.setter
    def rhbi_ell_ema(self, value: float) -> None:
        self._rhbi_ell_scalar = float(value)
        self._rhbi_ell_t = None

    def record_step(self, optimizer_or_lr) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            **super().state_dict(),
            "rhbi": {
                "schema": 1,
                "ell_ema": float(self.rhbi_ell_ema),
                "ell_initialized": bool(self.rhbi_ell_initialized),
            },
        }

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "rhbi" not in state:
            raise ValueError("RHBI observer checkpoint is missing transport horizon state")
        if set(state) != {*CriticStepRecord.KEYS, "rhbi"}:
            raise ValueError("invalid RHBI observer record fields")
        observer = state["rhbi"]
        if (
            not isinstance(observer, dict)
            or set(observer) != {"schema", "ell_ema", "ell_initialized"}
            or type(observer["schema"]) is not int
            or observer["schema"] != 1
            or type(observer["ell_initialized"]) is not bool
        ):
            raise ValueError("invalid RHBI observer metadata")
        fval = observer["ell_ema"]
        if isinstance(fval, bool) or not isinstance(fval, (int, float)) or not math.isfinite(fval) or fval < 0.0:
            raise ValueError("invalid RHBI observer ell_ema")
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        observer = state["rhbi"]
        self._rhbi_ell_scalar = float(observer["ell_ema"])
        self._rhbi_ell_t = None
        self.rhbi_ell_initialized = bool(observer["ell_initialized"])
        self._capture_ring = OrderedDict()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_RHBIObserverStepRecord":
        dup = _RHBIObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        dup._rhbi_ell_scalar = self._rhbi_ell_scalar
        dup._rhbi_ell_t = None if self._rhbi_ell_t is None else self._rhbi_ell_t.clone()
        dup.rhbi_ell_initialized = self.rhbi_ell_initialized
        dup._in_penalty = self._in_penalty
        dup._in_protected_probe = self._in_protected_probe
        return dup


class RHBIStepRecord(CriticStepRecord):
    """Per-critic RHBI state traveling with ``RHBICriticAdam.state_dict()``."""

    _COUNTERS = ("calls", "observed_steps")
    _OPTIONAL = ("lr_last",)
    KEYS = (
        "formulation",
        "lr_max",
        "lr_last",
        "anchor_started",
        "calls",
        "observed_steps",
        "ell_ema",
        "ell_initialized",
    )

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._rhbi_ell_scalar: float = 0.0
        self._rhbi_ell_t: Optional[torch.Tensor] = None
        self.rhbi_ell_initialized: bool = False
        self._in_penalty: bool = False
        self._in_protected_probe: bool = False
        self._capture_ring: "OrderedDict[Tuple[int, int, int, int], Tuple[torch.Tensor, Optional[torch.Tensor], Any]]" = OrderedDict()

    @property
    def rhbi_ell_ema(self) -> float:
        if self._rhbi_ell_t is not None:
            return float(self._rhbi_ell_t.item())
        return self._rhbi_ell_scalar

    @rhbi_ell_ema.setter
    def rhbi_ell_ema(self, value: float) -> None:
        self._rhbi_ell_scalar = float(value)
        self._rhbi_ell_t = None

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "formulation": "rhbi",
            "lr_max": float(self.lr_max),
            "lr_last": None if self.lr_last is None else float(self.lr_last),
            "anchor_started": bool(self.anchor_started),
            "calls": int(self.calls),
            "observed_steps": int(self.observed_steps),
            "ell_ema": float(self.rhbi_ell_ema),
            "ell_initialized": bool(self.rhbi_ell_initialized),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("formulation") != "rhbi":
            raise ValueError("RHBI critic optimizer requires an RHBI controller checkpoint")
        if set(state) != set(self.KEYS):
            raise ValueError("invalid RHBI controller state keys")
        for key in self._COUNTERS:
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid RHBI controller {key}")
        for key in ("lr_max", "ell_ema", *self._OPTIONAL):
            value = state[key]
            if value is None and key in self._OPTIONAL:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid RHBI controller {key}")
        if type(state["anchor_started"]) is not bool or type(state["ell_initialized"]) is not bool:
            raise ValueError("invalid RHBI controller boolean flags")
        self.lr_max = float(state["lr_max"])
        self.lr_last = None if state["lr_last"] is None else float(state["lr_last"])
        self.anchor_started = bool(state["anchor_started"])
        self.calls = int(state["calls"])
        self.observed_steps = int(state["observed_steps"])
        self._rhbi_ell_scalar = float(state["ell_ema"])
        self._rhbi_ell_t = None
        self.rhbi_ell_initialized = bool(state["ell_initialized"])
        self._capture_ring = OrderedDict()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "RHBIStepRecord":
        dup = RHBIStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        dup._rhbi_ell_scalar = self._rhbi_ell_scalar
        dup._rhbi_ell_t = None if self._rhbi_ell_t is None else self._rhbi_ell_t.clone()
        dup.rhbi_ell_initialized = self.rhbi_ell_initialized
        dup._in_penalty = self._in_penalty
        dup._in_protected_probe = self._in_protected_probe
        return dup


def _rhbi_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> None:
    rec = getattr(module, "_rhbi_record", None)
    if rec is not None and getattr(rec, "_in_penalty", False):
        return
    target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(target, torch.Tensor) or target.ndim < 1:
        return
    x_in, cond_disc = _extract_critic_inputs(args, kwargs, int(target.shape[0]))
    if x_in is not None:
        _record_rhbi_capture(output, x_in, cond_disc, rec, overwrite=True)


def _attach_rhbi_critic_hook(critic: nn.Module, record: CriticStepRecord) -> None:
    """Attach a forward hook so ``RHBIGANLoss`` can read critic input coordinates."""
    if not hasattr(record, "rhbi_ell_initialized"):
        record.rhbi_ell_ema = 0.0
        record.rhbi_ell_initialized = False
    if not hasattr(record, "_in_penalty"):
        record._in_penalty = False
    if not hasattr(record, "_in_protected_probe"):
        record._in_protected_probe = False
    if not hasattr(record, "_capture_ring"):
        record._capture_ring = OrderedDict()
    critic._rhbi_record = record
    if getattr(critic, "_rhbi_hook_registered", False):
        return
    critic.register_forward_hook(_rhbi_critic_forward_hook, with_kwargs=True)
    critic._rhbi_hook_registered = True


def bind_rhbi_record(record: Any, *consumers: Any) -> None:
    """Bind a critic step record to ``RHBIGANLoss`` instances."""
    for consumer in consumers:
        if consumer is not None and hasattr(consumer, "record"):
            consumer.record = record


class RHBICriticAdam(K3PCriticAdam):
    """Critic Adam optimizer for ``critic_formulation='rhbi'`` with no EMA critic copy."""

    def __init__(
        self,
        params,
        *,
        critic: nn.Module,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs,
    ) -> None:
        record = RHBIStepRecord()
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
        _attach_rhbi_critic_hook(critic, record)
