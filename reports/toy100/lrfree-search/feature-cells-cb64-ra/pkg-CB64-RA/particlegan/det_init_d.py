"""Deterministic K3P initializers. Family D hybrids of hid_q and qr_pb_pq.

``install(name)`` overwrites random Linear, Conv, and particle-prior draws
with a closed-form function of tensor shape and construction order. The
original ``uniform_`` / ``normal_`` still runs first, so later sample streams
stay on the seed. Host writes after that draw (``copy_(eye)``, ``zero_()``)
are left alone.

Nothing here changes learning rates, schedules, penalties, or loss weights.
Omit ``install`` and the PyTorch init is unchanged.

Weight rules
  house_hid  Square matrices: 3-reflector Householder times the Kaiming gain.
             Rectangles: tiled scaled identity at the Kaiming RMS. Convolutions
             put that matrix on the center tap (hid_q weights).
  house_all  Same Householder, semi-orthogonal on every matrix including
             rectangles and full convolution kernels, scaled to the RMS of the
             distribution that actually filled the tensor.
  qr         Float64 QR of a splitmix64 normal matrix, sign-fixed, scaled to
             that same declared RMS (qr_pb_pq weights).
  mix        Householder on square hidden maps, QR on rectangular input/output
             maps. Square-channel convolutions use the center-tap Householder;
             other convolutions use QR.

Bias rules
  weyl_q   Kronecker-Weyl sequence in a quarter of the declared uniform box.
  pattern  Hash pattern standardized to the declared mean and std.
  zero     All-zero, when the host did not already rewrite the bias.

Prior rules
  weyl  Kronecker-Weyl mapped by Acklam's inverse normal, or into the box.
  r2    Roberts R2 sequence, one point per row, through ``torch.special.ndtri``
        or into the box.

``hid_q`` and ``qr_pb_pq`` are the two parents, included so a CPU screen can
compare hybrids to them on the same build.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

_SPECS = {
    "hid_q": dict(w="house_hid", b="weyl_q", p="weyl"),
    "qr_pb_pq": dict(w="qr", b="pattern", p="r2"),
    "hq_pb": dict(w="house_hid", b="pattern", p="weyl"),
    "hq_pq": dict(w="house_hid", b="weyl_q", p="r2"),
    "hq_pb_pq": dict(w="house_hid", b="pattern", p="r2"),
    "hq_zb_pq": dict(w="house_hid", b="zero", p="r2"),
    "mix_pb_pq": dict(w="mix", b="pattern", p="r2"),
    "mix_wq_pq": dict(w="mix", b="weyl_q", p="r2"),
    "mix_pb_weyl": dict(w="mix", b="pattern", p="weyl"),
    "mix_zb_pq": dict(w="mix", b="zero", p="r2"),
    "qr_wq_pq": dict(w="qr", b="weyl_q", p="r2"),
    "hh_pb_pq": dict(w="house_all", b="pattern", p="r2"),
    "hh_wq_pq": dict(w="house_all", b="weyl_q", p="r2"),
    "hh_zb_pq": dict(w="house_all", b="zero", p="r2"),
}
VARIANTS = tuple(_SPECS)
_GAIN = math.sqrt(2.0 / (1.0 + 5.0))  # calculate_gain("leaky_relu", sqrt(5))
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
_STATE = {"name": None}
_PATCHED = False
_ORIG = {}
_Q_CACHE: dict[int, torch.Tensor] = {}
_SEQ = [0]
_IN_PRIOR = contextvars.ContextVar("k3p_det_prior", default=False)
LOG: list[dict] = []

# Acklam inverse-normal. Float64 in and out, then the parameter cast.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425
_CPU = torch.device("cpu")


def install(name: str) -> str:
    """Activate one variant. Construction order restarts at zero."""
    if name not in _SPECS:
        raise ValueError(f"init must be one of {', '.join(VARIANTS)}")
    _ensure_patched()
    _STATE["name"] = name
    _SEQ[0] = 0
    LOG.clear()
    print(json.dumps({"event": "det_init", "name": name, "spec": _SPECS[name]}), flush=True)
    return name


def uninstall() -> None:
    """Leave the patches installed but inert."""
    _STATE["name"] = None
    _SEQ[0] = 0


def active() -> str | None:
    return _STATE["name"]


def spec(name: str | None = None) -> dict:
    return dict(_SPECS[name or _STATE["name"]])


def ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse standard normal CDF. ``p`` in (0, 1), float64 in and out."""
    p = p.to(dtype=torch.float64, device=_CPU)
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
        return torch.zeros(n, d, dtype=torch.float64, device=_CPU)
    idx = torch.arange(1, n + 1, dtype=torch.float64, device=_CPU)
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
        q = torch.zeros(0, 0, dtype=torch.float64, device=_CPU)
    elif n == 1:
        q = torch.ones(1, 1, dtype=torch.float64, device=_CPU)
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
        q = torch.tensor(rows, dtype=torch.float64, device=_CPU)
    _Q_CACHE[n] = q
    return q


def _house_semi(rows: int, cols: int) -> torch.Tensor:
    """Leading block of a Householder orthogonal. Columns or rows are orthonormal."""
    if rows <= 0 or cols <= 0:
        return torch.zeros(rows, cols, dtype=torch.float64, device=_CPU)
    if rows >= cols:
        return _householder(rows)[:, :cols]
    return _householder(cols)[:rows, :]


def _hash_u01(key: int, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    with np.errstate(over="ignore"):
        x = (np.arange(n, dtype=np.uint64) + np.uint64(key)) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)


def _key(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") >> 1


def _alloc_key(role: str, shape) -> int:
    key = _key(_SEQ[0], role, tuple(shape))
    _SEQ[0] += 1
    return key


def _semi_orthogonal(key: int, rows: int, cols: int) -> torch.Tensor:
    """Float64 QR with a positive diagonal. Orthonormal columns, or rows if ``rows < cols``."""
    flip = rows < cols
    r, c = (cols, rows) if flip else (rows, cols)
    u = torch.from_numpy(np.ascontiguousarray(_hash_u01(key, r * c))).reshape(r, c)
    source = torch.special.ndtri(u)
    q, rr = torch.linalg.qr(source)
    d = torch.sign(torch.diagonal(rr))
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q * d
    return q.T.contiguous() if flip else q


def _r2(n: int, d: int) -> np.ndarray:
    """Roberts R2 in ``[0, 1)^d``. Row ``i`` is one point. No RNG."""
    phi = 2.0
    for _ in range(64):
        phi = (1.0 + phi) ** (1.0 / (d + 1))
    alpha = np.array([(1.0 / phi) ** (j + 1) for j in range(d)])
    return np.mod(0.5 + np.arange(1, n + 1)[:, None] * alpha[None, :], 1.0)


def _declared(tag):
    kind, a, b = tag
    if kind == "uniform":
        rms = math.sqrt((a * a + a * b + b * b) / 3.0)
        return rms, (a + b) / 2.0, (b - a) / math.sqrt(12.0)
    return math.sqrt(a * a + b * b), a, b


def _kaiming_std(weight: torch.Tensor) -> float:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    return _GAIN / math.sqrt(fan_in) if fan_in > 0 else 0.0


def _unit(rows: int, cols: int) -> torch.Tensor:
    block = torch.zeros(rows, cols, dtype=torch.float64, device=_CPU)
    if rows == 0 or cols == 0:
        return block
    if rows >= cols:
        reps, rem = divmod(rows, cols)
        eye = torch.eye(cols, dtype=torch.float64, device=_CPU)
        for r in range(reps):
            block[r * cols:(r + 1) * cols] = eye
        if rem:
            block[reps * cols:, :rem] = torch.eye(rem, dtype=torch.float64, device=_CPU)
    else:
        reps, rem = divmod(cols, rows)
        eye = torch.eye(rows, dtype=torch.float64, device=_CPU)
        for r in range(reps):
            block[:, r * rows:(r + 1) * rows] = eye
        if rem:
            block[:rem, reps * rows:] = torch.eye(rem, dtype=torch.float64, device=_CPU)
    return block


def _place(tensor: torch.Tensor, values: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.copy_(values.to(dtype=tensor.dtype, device=tensor.device))


def _record(tensor, role, rule, key) -> None:
    raw = tensor.detach().cpu().contiguous()
    LOG.append({
        "role": role, "rule": rule, "key": key, "shape": list(tensor.shape),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    })


def _write_house_hid(weight: torch.Tensor) -> None:
    shape = tuple(weight.shape)
    rows, cols = int(shape[0]), int(shape[1])
    std = _kaiming_std(weight)
    if rows == cols and rows > 0:
        block = _householder(rows) * _GAIN
    else:
        unit = _unit(rows, cols)
        nnz = float(unit.abs().sum())
        block = unit if nnz == 0 or weight.numel() == 0 else unit * (std * math.sqrt(weight.numel() / nnz))
    if len(shape) == 2:
        out = block
    else:
        out = torch.zeros(shape, dtype=torch.float64, device=_CPU)
        center = tuple(s // 2 for s in shape[2:])
        out[(slice(None), slice(None), *center)] = block
    _place(weight, out)


def _write_scaled(weight: torch.Tensor, q: torch.Tensor, tag, out_shape) -> None:
    rows, cols = q.shape
    rms = _declared(tag)[0]
    _place(weight, (q * rms * math.sqrt(max(rows, cols))).reshape(out_shape))


def _write_qr(weight: torch.Tensor, tag, key: int) -> None:
    shape = tuple(weight.shape)
    rows = int(shape[0])
    cols = int(math.prod(shape[1:])) if len(shape) > 1 else 1
    _write_scaled(weight, _semi_orthogonal(key, rows, cols), tag, shape)


def _write_house_all(weight: torch.Tensor, tag) -> None:
    shape = tuple(weight.shape)
    rows = int(shape[0])
    cols = int(math.prod(shape[1:])) if len(shape) > 1 else 1
    _write_scaled(weight, _house_semi(rows, cols), tag, shape)


def _channel_square(weight: torch.Tensor) -> bool:
    shape = tuple(weight.shape)
    return len(shape) >= 2 and int(shape[0]) == int(shape[1]) and int(shape[0]) > 1


def _write_weight(weight: torch.Tensor, tag, key: int) -> str:
    which = _SPECS[_STATE["name"]]["w"]
    if which == "house_hid":
        _write_house_hid(weight)
        return "house_hid"
    if which == "house_all":
        _write_house_all(weight, tag)
        return "house_all"
    if which == "qr":
        _write_qr(weight, tag, key)
        return "qr"
    if _channel_square(weight):
        _write_house_hid(weight)
        return "mix_house"
    _write_qr(weight, tag, key)
    return "mix_qr"


def _pattern(key: int, shape, tag) -> torch.Tensor:
    n = int(math.prod(shape)) if len(shape) else 1
    _, mean, std = _declared(tag)
    u = torch.from_numpy(np.ascontiguousarray(_hash_u01(key, n))) * 2 - 1
    if n > 1:
        u = (u - u.mean()) / u.std(unbiased=False)
    return (mean + std * u).reshape(shape)


def _weyl_quarter(shape, tag) -> torch.Tensor:
    kind, a, b = tag
    n = int(math.prod(shape)) if len(shape) else 1
    signed = 2.0 * _weyl(n, 1).reshape(-1) - 1.0
    if kind == "uniform":
        center = 0.5 * (a + b)
        half = 0.5 * (b - a)
        values = center + signed * half * 0.25
    else:
        values = a + signed * b * 0.25
    return values.reshape(shape)


def _write_bias(bias: torch.Tensor, tag, key: int) -> str:
    rule = _SPECS[_STATE["name"]]["b"]
    if rule == "zero":
        _place(bias, torch.zeros(bias.shape, dtype=torch.float64, device=_CPU))
    elif rule == "pattern":
        _place(bias, _pattern(key, tuple(bias.shape), tag))
    else:
        _place(bias, _weyl_quarter(tuple(bias.shape), tag))
    return "bias_" + rule


def _unit_interval(tensor: torch.Tensor) -> torch.Tensor:
    shape = tuple(tensor.shape)
    if len(shape) == 2:
        return _weyl(shape[0], shape[1])
    if len(shape) == 1:
        return _weyl(shape[0], 1).reshape(shape)
    return _weyl(tensor.numel(), 1).reshape(shape)


def _write_prior(tensor: torch.Tensor, tag) -> str:
    kind, a, b = tag
    rule = _SPECS[_STATE["name"]]["p"]
    if rule == "weyl":
        u = _unit_interval(tensor)
        values = a + (b - a) * u if kind == "uniform" else a + b * ndtri(u)
        name = "prior_weyl"
    else:
        shape = tuple(tensor.shape)
        if len(shape) >= 2:
            u_np = _r2(shape[0], int(math.prod(shape[1:])))
        else:
            u_np = _r2(max(tensor.numel(), 1), 1)
        u = torch.from_numpy(np.ascontiguousarray(u_np)).reshape(shape)
        values = a + (b - a) * u if kind == "uniform" else a + b * torch.special.ndtri(u)
        name = "prior_r2"
    _place(tensor, values)
    return name


def _apply_fill(tensor: torch.Tensor, tag) -> None:
    role = "prior" if _IN_PRIOR.get() else getattr(tensor, "_det_role", None)
    if role is None:
        return
    tensor._det_role = role
    key = _alloc_key(role, tensor.shape)
    if role == "prior":
        rule = _write_prior(tensor, tag)
    elif role == "bias":
        rule = _write_bias(tensor, tag, key)
    else:
        rule = _write_weight(tensor, tag, key)
    _record(tensor, role, rule, key)


def _bounds(args, kwargs):
    lo = args[0] if len(args) > 0 else kwargs.get("from", 0.0)
    hi = args[1] if len(args) > 1 else kwargs.get("to", 1.0)
    return float(lo), float(hi)


def _normal_stats(args, kwargs):
    mean = args[0] if len(args) > 0 else kwargs.get("mean", 0.0)
    std = args[1] if len(args) > 1 else kwargs.get("std", 1.0)
    return float(mean), float(std)


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
        self.weight._det_role = "weight"
        if self.bias is not None:
            self.bias._det_role = "bias"
        _ORIG["linear"](self)

    def conv_reset(self) -> None:
        self.weight._det_role = "weight"
        if self.bias is not None:
            self.bias._det_role = "bias"
        _ORIG["conv"](self)

    def uniform_(self, *args, **kwargs):
        out = _ORIG["uniform"](self, *args, **kwargs)
        if _STATE["name"] and isinstance(self, nn.Parameter):
            lo, hi = _bounds(args, kwargs)
            _apply_fill(self, ("uniform", lo, hi))
        return out

    def normal_(self, *args, **kwargs):
        out = _ORIG["normal"](self, *args, **kwargs)
        if _STATE["name"] and isinstance(self, nn.Parameter):
            mean, std = _normal_stats(args, kwargs)
            _apply_fill(self, ("normal", mean, std))
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
