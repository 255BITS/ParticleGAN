"""BISPF thermal state: the legacy process channel, thermal binding and the residual-driven cooler."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch

# A thermal scale at or below this value is the frozen state: the critic
# penalty and the SBNP regularizer pass their loss through with gradient
# scale s (see their docstrings). Not a shape gate.
FROZEN_THERMAL_SCALE = 1e-4


# Legacy process-wide thermal channel, keyed by device. A BISPF critic penalty
# publishes its thermal scale here only while its record allows it
# (``record.legacy_thermal_channel``, default True). Readers that are not bound
# to a thermal source (an SBNP regularizer, SinkhornRelativisticLoss or
# StratifiedBlueNoisePrior built outside GANTrainer) read the channel. This
# keeps caller-owned loops on their historical trajectories. GANTrainer binds
# its readers to its own critic record with ``bind_thermal_source`` and
# switches publishing off, so trainers in one process do not interact.
_LEGACY_THERMAL_CHANNEL: Dict[torch.device, float] = {}
# Historical name of the same dict. policy.py, whose bytes are bound by
# qualified Forge task sources, snapshots it around validation probes.
_LAST_THERMAL_SCALE = _LEGACY_THERMAL_CHANNEL


def _publish_thermal(record: Any, device: torch.device, scale: float) -> None:
    record._thermal_scale = scale
    if getattr(record, "legacy_thermal_channel", True):
        _LEGACY_THERMAL_CHANNEL[device] = scale


def _read_thermal(source: Any, device: torch.device, default: float = 1.0) -> float:
    if source is not None:
        return float(getattr(source, "_thermal_scale", 1.0))
    return float(_LEGACY_THERMAL_CHANNEL.get(device, default))


def bind_thermal_source(record: Any, *readers: Any) -> None:
    """Bind thermal readers to one critic record and take that record off the legacy channel.

    ``record`` is the critic optimizer's ``record``, whose ``_thermal_scale``
    the BISPF critic penalty updates and the optimizer checkpoints. Each reader
    with a ``thermal_source`` attribute (SBNP regularizer, Sinkhorn loss,
    StratifiedBlueNoisePrior) then reads that record instead of the
    process-wide legacy channel.
    """
    record.legacy_thermal_channel = False
    for reader in readers:
        if reader is not None and hasattr(reader, "thermal_source"):
            reader.thermal_source = record

class ContinuousThermalCooler:
    """Continuous residual-driven thermal cooler for basin settling and autonomous re-heating.

    ``update`` tracks ``ema_residual``, the scale-free nearest-neighbor
    Sinkhorn residual ``delta_nn``. The EMA uses fast attack: a residual above
    the current EMA replaces it immediately. ``scale_for(batch_size, dim)``
    returns a smoothstep of the cooling ratio::

        tau = effective_tau(batch_size, dim)
        u   = clamp(ema_residual / tau, 0, 1)
        s   = s_min + (1 - s_min) * (3 u^2 - 2 u^3)        # in [s_min, 1]

    ``effective_tau`` is ``tau_cooling``, raised for small batches when
    ``tau_batch_scaling`` is on: for ``b = max(min_batch, batch_size)`` below
    ``reference_batch`` it multiplies by ``(batch_numerator / b) ** (1 / d)``
    with ``d = clamp(dim, 1, max_dim)``. ``scale`` queries
    ``scale_for(default_batch_size, default_dim)``. The defaults (128, 2)
    equal ``reference_batch``, so ``scale`` uses ``tau_cooling`` unscaled.

    The rownorm linearization ``s / sqrt(1 - s^2) * eps_smooth`` is not part of
    the cooler. It is the generator-loss dense rescale declared by
    ``BISPFGates.dense_g_rescale_*``.
    """

    def __init__(
        self,
        tau_cooling: float = 0.020,
        s_min: float = 1e-5,
        ema_decay: float = 0.95,
        initial_residual: float = 1.0,
        *,
        tau_batch_scaling: bool = True,
        reference_batch: int = 128,
        min_batch: int = 16,
        batch_numerator: float = 1024.0,
        max_dim: int = 4,
        default_batch_size: int = 128,
        default_dim: int = 2,
    ) -> None:
        for name, val in (("tau_cooling", tau_cooling), ("s_min", s_min), ("ema_decay", ema_decay),
                          ("batch_numerator", batch_numerator)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if s_min > 1.0:
            raise ValueError("s_min must be <= 1.0")
        if ema_decay >= 1.0:
            raise ValueError("ema_decay must be < 1.0")
        if type(tau_batch_scaling) is not bool:
            raise TypeError("tau_batch_scaling must be a boolean")
        for name, val in (("reference_batch", reference_batch), ("min_batch", min_batch), ("max_dim", max_dim),
                          ("default_batch_size", default_batch_size), ("default_dim", default_dim)):
            if type(val) is not int or val < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.tau_cooling = float(tau_cooling)
        self.s_min = float(s_min)
        self.ema_decay = float(ema_decay)
        self.tau_batch_scaling = tau_batch_scaling
        self.reference_batch = reference_batch
        self.min_batch = min_batch
        self.batch_numerator = float(batch_numerator)
        self.max_dim = max_dim
        self.default_batch_size = default_batch_size
        self.default_dim = default_dim
        self.ema_residual = float(initial_residual)
        self.initialized = False

    def update(self, batch_residual: float | torch.Tensor) -> float:
        """Update EMA residual with a new batch residual observation and return ``scale``."""
        res = float(batch_residual.item()) if isinstance(batch_residual, torch.Tensor) else float(batch_residual)
        if not self.initialized:
            self.ema_residual = res
            self.initialized = True
        elif res > self.ema_residual:
            # Fast-attack peak detection: instant re-heating upon shift shock
            self.ema_residual = res
        else:
            self.ema_residual = self.ema_decay * self.ema_residual + (1.0 - self.ema_decay) * res
        return self.scale

    def effective_tau(self, batch_size: int = 128, dim: int = 2) -> float:
        """Cooling threshold for a batch size and dimension (see the class docstring)."""
        if not self.tau_batch_scaling:
            return self.tau_cooling
        d = max(1, min(dim, self.max_dim))
        b = max(self.min_batch, batch_size)
        ratio = self.batch_numerator / b
        scale = 1.0 if b >= self.reference_batch else (ratio if d == 1 else ratio ** (1.0 / d))
        return self.tau_cooling * scale

    def scale_for(self, batch_size: int = 128, dim: int = 2) -> float:
        """Smoothstep thermal scale in ``[s_min, 1]`` (see the class docstring)."""
        tau = self.effective_tau(batch_size=batch_size, dim=dim)
        if tau <= 0.0:
            return 1.0
        u = min(1.0, max(0.0, self.ema_residual / tau))
        if u >= 1.0:
            return 1.0
        if u <= 0.0:
            return self.s_min
        s = self.s_min + (1.0 - self.s_min) * (3.0 * u * u - 2.0 * u * u * u)
        return max(self.s_min, min(1.0, s))

    @property
    def scale(self) -> float:
        """Thermal scale at the declared default query ``(default_batch_size, default_dim)``."""
        return self.scale_for(batch_size=self.default_batch_size, dim=self.default_dim)

    def apply_straight_through(self, tensor: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        """Apply straight-through gradient scaling: backward gradient is scaled by scale (or self.scale) while forward pass is identity."""
        s = self.scale if scale is None else float(scale)
        if s < 1.0 and tensor.requires_grad:
            return tensor * s + tensor.detach() * (1.0 - s)
        return tensor

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply straight-through gradient scaling with current cooler scale."""
        return self.apply_straight_through(tensor)

    _DECLARED = ("tau_batch_scaling", "reference_batch", "min_batch", "batch_numerator", "max_dim",
                 "default_batch_size", "default_dim")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tau_cooling": self.tau_cooling,
            "s_min": self.s_min,
            "ema_decay": self.ema_decay,
            "ema_residual": self.ema_residual,
            "initialized": self.initialized,
            **{name: getattr(self, name) for name in self._DECLARED},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore cooler state. The declared gate fields are constructor-owned: a state that
        records a different value is rejected, and older states may omit them. Older states may
        carry the removed, never-used ``eps_smooth``."""
        for name in self._DECLARED:
            if name in state and state[name] != getattr(self, name):
                raise ValueError(f"cooler state {name}={state[name]!r} does not match the declared "
                                 f"{getattr(self, name)!r}")
        self.tau_cooling = float(state["tau_cooling"])
        self.s_min = float(state["s_min"])
        self.ema_decay = float(state["ema_decay"])
        self.ema_residual = float(state["ema_residual"])
        self.initialized = bool(state["initialized"])
