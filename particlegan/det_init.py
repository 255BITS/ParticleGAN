"""Deterministic initial parameters for K3P, independent of the torch seed.

``install(name)`` replaces random Linear / Conv / particle-prior draws with a
closed-form function of tensor shape. Sample and noise generators are still
advanced by the original ``uniform_`` / ``normal_`` calls, then the parameter
is overwritten, so later sample streams stay on the seed.

Variants (all scales match ``kaiming_uniform_(a=sqrt(5))`` element RMS):

* ``eye`` — tiled identity (eye repeated so every row and column is used).
  Square hidden layers are ``gain * I``, which is orthogonal. Convolutions
  are delta-identity: that matrix sits on the center tap and every other tap
  is zero. Biases are zero.
* ``eye_bias`` — same weights, biases a Weyl sequence inside the default
  ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]`` box.
* ``eye_pad`` — identity in the leading ``min(rows, cols)`` block, zeros
  elsewhere, same RMS match, plus the bias pattern. The padding is the
  rectangular-linear form of a scaled identity.
* ``hid`` — square hidden weights are a fixed 3-Householder orthogonal matrix
  times the Kaiming gain (not a QR of a random or hashed matrix). Rectangular
  input and output layers stay tiled identity at the Kaiming scale. Convs
  put that matrix on the center tap only (delta-orthogonal). Biases are zero.
* ``hid_bias`` — ``hid`` weights with the Weyl bias pattern.
* ``hid_edge`` — ``hid`` weights, bias pattern only on rectangular layers.
* ``hid_q`` — ``hid`` weights, bias pattern at a quarter of the default bound.
* ``eye_sign`` — tiled identity with alternating row signs and zero bias.
* ``sign_q`` — ``eye_sign`` weights, bias at a quarter of the bound.
* ``hid_h`` — ``hid`` weights, bias at half the default bound.
* ``pad_in`` — padded identity on expanding layers, tiled identity on
  contracting layers, bias pattern on every layer.
* ``tile_in`` — tiled identity on expanding layers, padded identity on
  contracting layers, bias pattern on every layer.
* ``tile_read`` — tiled identity everywhere except a generator-shaped map
  (``1 < rows < cols``), which stays eye-padded so a 2-wide output reads
  one hidden block. A 1-wide critic readout stays tiled and sees every
  unit. Full Weyl bias.
* ``tile_read_q`` — ``tile_read`` weights, bias at a quarter of the bound.
* ``hid_read`` — square hidden layers are the Householder orthogonal times
  the Kaiming gain; rectangular layers follow ``tile_read``. Quarter bias.

The particle table is a Kronecker-Weyl sequence mapped through a rational
inverse-normal (or into the requested uniform box). It is not a whitened QR
cloud and not a frozen Philox draw. Explicit host writes (``copy_(eye)``,
``zeros_``) still run after this and win.
"""
from __future__ import annotations

import contextvars
import json
import math

import torch
from torch import nn

VARIANTS = ("eye", "eye_bias", "eye_pad", "hid", "hid_bias",
            "hid_edge", "hid_q", "hid_h", "eye_sign", "sign_q", "pad_in", "tile_in",
            "tile_read", "tile_read_q", "hid_read")
_BIAS_VARIANTS = frozenset(("eye_bias", "eye_pad", "hid_bias", "hid_edge", "hid_q", "hid_h",
                            "sign_q", "pad_in", "tile_in", "tile_read", "tile_read_q", "hid_read"))
_ORTHO_VARIANTS = frozenset(("hid", "hid_bias", "hid_edge", "hid_q", "hid_h", "hid_read"))
_READ_VARIANTS = frozenset(("tile_read", "tile_read_q", "hid_read"))
_BIAS_SCALE = {"hid_q": 0.25, "tile_read_q": 0.25, "hid_read": 0.25,
               "sign_q": 0.25, "hid_h": 0.5}
_GAIN = math.sqrt(2.0 / (1.0 + 5.0))  # calculate_gain("leaky_relu", sqrt(5))
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
_STATE = {"name": None}
_PATCHED = False
_ORIG = {}
_Q_CACHE: dict[int, torch.Tensor] = {}
_IN_PRIOR = contextvars.ContextVar("k3p_det_prior", default=False)

# Acklam's inverse-normal rational approximation. Pure float64 arithmetic so
# the same bits land on CPU and CUDA after the float32 cast.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def install(name: str) -> str:
    """Activate one variant. Default training never calls this."""
    if name not in VARIANTS:
        raise ValueError(f"init must be one of {', '.join(VARIANTS)}")
    _ensure_patched()
    _STATE["name"] = name
    print(json.dumps({"event": "det_init", "name": name}), flush=True)
    return name


def uninstall() -> None:
    """Leave the patches in place but inert, so the default init returns."""
    _STATE["name"] = None


def active() -> str | None:
    return _STATE["name"]


def ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse standard normal CDF. ``p`` in (0, 1), float64 in and out."""
    p = p.to(dtype=torch.float64)
    low = p < _P_LOW
    high = p > 1.0 - _P_LOW
    central = ~low & ~high
    out = torch.empty_like(p)
    if bool(central.any()):
        q = p - 0.5
        r = q * q
        num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        den = (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        out = torch.where(central, num / den, out)
    if bool(low.any()):
        out = torch.where(low, _tail(p), out)
    if bool(high.any()):
        out = torch.where(high, -_tail(1.0 - p), out)
    return out


def _tail(p: torch.Tensor) -> torch.Tensor:
    q = torch.sqrt(-2.0 * torch.log(p))
    num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
    den = ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    return num / den


def _alpha(index: int) -> float:
    if index < len(_PRIMES):
        return math.sqrt(_PRIMES[index])
    return math.sqrt(_PRIMES[index % len(_PRIMES)] * (index // len(_PRIMES) + 2))


def _weyl(n: int, d: int) -> torch.Tensor:
    """Kronecker sequence in ``(1e-12, 1-1e-12)`` with shape ``(n, d)``."""
    if n < 0 or d < 0:
        raise ValueError("weyl size must be nonnegative")
    if n == 0 or d == 0:
        return torch.zeros(n, d, dtype=torch.float64)
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    cols = []
    for j in range(d):
        x = idx * _alpha(j)
        cols.append(x - torch.floor(x))
    out = torch.stack(cols, 1) if d > 1 else cols[0].unsqueeze(1)
    return out.clamp(1e-12, 1.0 - 1e-12)


def _householder(n: int) -> torch.Tensor:
    """Deterministic orthogonal ``n x n`` from three rational Householder reflectors."""
    hit = _Q_CACHE.get(n)
    if hit is not None:
        return hit
    if n <= 0:
        q = torch.zeros(0, 0, dtype=torch.float64)
    elif n == 1:
        q = torch.ones(1, 1, dtype=torch.float64)
    else:
        rows = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for prime in (2, 3, 5):
            v = [(((i + 1) * prime) % 997) / 997.0 - 0.5 for i in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            inv = 1.0 / norm
            v = [x * inv for x in v]
            qv = [sum(rows[i][j] * v[j] for j in range(n)) for i in range(n)]
            for i in range(n):
                scale = 2.0 * qv[i]
                row = rows[i]
                for j in range(n):
                    row[j] -= scale * v[j]
        q = torch.tensor(rows, dtype=torch.float64)
    _Q_CACHE[n] = q
    return q


def _fan(weight: torch.Tensor) -> tuple[float, float]:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    fan_in = float(fan_in)
    std = _GAIN / math.sqrt(fan_in) if fan_in > 0 else 0.0
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    return std, bound


def _how(rows: int, cols: int) -> str:
    name = _STATE["name"]
    if name in _ORTHO_VARIANTS and rows == cols and rows > 0:
        return "ortho"
    if name in _READ_VARIANTS and 1 < rows < cols:
        return "pad"
    if name in ("eye_sign", "sign_q"):
        return "sign"
    if name == "eye_pad":
        return "pad"
    if name == "pad_in":
        return "pad" if rows >= cols else "tile"
    if name == "tile_in":
        return "pad" if rows < cols else "tile"
    return "tile"


def _unit(rows: int, cols: int, padded: bool) -> torch.Tensor:
    block = torch.zeros(rows, cols, dtype=torch.float64)
    if rows == 0 or cols == 0:
        return block
    if padded:
        n = min(rows, cols)
        block[:n, :n] = torch.eye(n, dtype=torch.float64)
        return block
    if rows >= cols:
        reps, rem = divmod(rows, cols)
        eye = torch.eye(cols, dtype=torch.float64)
        for r in range(reps):
            block[r * cols:(r + 1) * cols] = eye
        if rem:
            block[reps * cols:, :rem] = torch.eye(rem, dtype=torch.float64)
    else:
        reps, rem = divmod(cols, rows)
        eye = torch.eye(rows, dtype=torch.float64)
        for r in range(reps):
            block[:, r * rows:(r + 1) * rows] = eye
        if rem:
            block[:rem, reps * rows:] = torch.eye(rem, dtype=torch.float64)
    return block


def _matrix(rows: int, cols: int, std: float, how: str, numel: int) -> torch.Tensor:
    if how == "ortho":
        return _householder(rows) * _GAIN
    unit = _unit(rows, cols, padded=(how == "pad"))
    if how == "sign" and rows:
        signs = torch.where((torch.arange(rows) % 2).to(dtype=torch.bool),
                            -torch.ones(rows, dtype=torch.float64),
                            torch.ones(rows, dtype=torch.float64))
        unit = unit * signs[:, None]
    nnz = float(unit.abs().sum())
    if nnz == 0 or numel == 0:
        return unit
    return unit * (std * math.sqrt(numel / nnz))


def _place(tensor: torch.Tensor, values: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.copy_(values.to(dtype=tensor.dtype, device=tensor.device))


def _write_weight(weight: torch.Tensor) -> None:
    shape = tuple(weight.shape)
    if len(shape) < 2:
        return
    rows, cols = int(shape[0]), int(shape[1])
    std, _ = _fan(weight)
    block = _matrix(rows, cols, std, _how(rows, cols), weight.numel())
    if len(shape) == 2:
        out = block
    else:
        out = torch.zeros(shape, dtype=torch.float64)
        center = tuple(s // 2 for s in shape[2:])
        out[(slice(None), slice(None), *center)] = block
    _place(weight, out)


def _write_bias(bias: torch.Tensor | None, weight: torch.Tensor) -> None:
    if bias is None:
        return
    name = _STATE["name"]
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    rectangular = rows != cols
    use_bias = name in _BIAS_VARIANTS and not (name == "hid_edge" and not rectangular)
    if not use_bias:
        _place(bias, torch.zeros(bias.shape, dtype=torch.float64))
        return
    _, bound = _fan(weight)
    bound *= _BIAS_SCALE.get(name, 1.0)
    n = bias.numel()
    vals = (2.0 * _weyl(n, 1).reshape(-1) - 1.0) * bound
    _place(bias, vals.reshape(bias.shape))


def _unit_interval(tensor: torch.Tensor) -> torch.Tensor:
    shape = tuple(tensor.shape)
    if len(shape) == 2:
        return _weyl(shape[0], shape[1])
    if len(shape) == 1:
        return _weyl(shape[0], 1).reshape(shape)
    return _weyl(tensor.numel(), 1).reshape(shape)


def _fill_normal(tensor: torch.Tensor, mean: float, std: float) -> None:
    if tensor.numel() == 0:
        return
    _place(tensor, float(mean) + float(std) * ndtri(_unit_interval(tensor)))


def _fill_uniform(tensor: torch.Tensor, lo: float, hi: float) -> None:
    if tensor.numel() == 0:
        return
    u = _unit_interval(tensor)
    _place(tensor, float(lo) + (float(hi) - float(lo)) * u)


def _mark(weight: torch.Tensor, kind: str) -> None:
    weight._det_role = "weight"
    weight._det_kind = kind


def _bounds(args, kwargs) -> tuple[float, float]:
    lo = args[0] if len(args) > 0 else kwargs.get("from", 0.0)
    hi = args[1] if len(args) > 1 else kwargs.get("to", 1.0)
    return float(lo), float(hi)


def _normal_stats(args, kwargs) -> tuple[float, float]:
    mean = args[0] if len(args) > 0 else kwargs.get("mean", 0.0)
    std = args[1] if len(args) > 1 else kwargs.get("std", 1.0)
    return float(mean), float(std)


def _after_uniform(tensor, args, kwargs) -> None:
    if not _STATE["name"] or not isinstance(tensor, nn.Parameter):
        return
    role = getattr(tensor, "_det_role", None)
    if role == "prior":
        lo, hi = _bounds(args, kwargs)
        _fill_uniform(tensor, lo, hi)
    elif role == "weight":
        _write_weight(tensor)


def _after_normal(tensor, args, kwargs) -> None:
    if not _STATE["name"] or not _IN_PRIOR.get():
        return
    mean, std = _normal_stats(args, kwargs)
    _fill_normal(tensor, mean, std)
    if isinstance(tensor, nn.Parameter):
        tensor._det_role = "prior"


def _ensure_patched() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from particlegan.particle_prior import ParticlePrior

    _ORIG["linear"] = nn.Linear.reset_parameters
    _ORIG["conv"] = nn.modules.conv._ConvNd.reset_parameters
    _ORIG["uniform"] = torch.Tensor.uniform_
    _ORIG["normal"] = torch.Tensor.normal_
    _ORIG["prior"] = ParticlePrior.__init__

    def linear_reset(self) -> None:
        _mark(self.weight, "linear")
        _ORIG["linear"](self)
        if _STATE["name"]:
            _write_weight(self.weight)
            _write_bias(self.bias, self.weight)

    def conv_reset(self) -> None:
        _mark(self.weight, "conv")
        _ORIG["conv"](self)
        if _STATE["name"]:
            _write_weight(self.weight)
            _write_bias(self.bias, self.weight)

    def uniform_(self, *args, **kwargs):
        out = _ORIG["uniform"](self, *args, **kwargs)
        _after_uniform(self, args, kwargs)
        return out

    def normal_(self, *args, **kwargs):
        out = _ORIG["normal"](self, *args, **kwargs)
        _after_normal(self, args, kwargs)
        return out

    def prior_init(self, *args, **kwargs):
        token = _IN_PRIOR.set(True)
        try:
            _ORIG["prior"](self, *args, **kwargs)
        finally:
            _IN_PRIOR.reset(token)
        if _STATE["name"] and isinstance(getattr(self, "z", None), nn.Parameter):
            self.z._det_role = "prior"

    nn.Linear.reset_parameters = linear_reset
    nn.modules.conv._ConvNd.reset_parameters = conv_reset
    torch.Tensor.uniform_ = uniform_
    torch.Tensor.normal_ = normal_
    ParticlePrior.__init__ = prior_init
    _PATCHED = True
