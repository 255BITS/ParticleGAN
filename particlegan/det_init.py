"""Deterministic K3P inits. Family F changes only the particle prior.

Two weight arms stay fixed:

* ``hid_q`` — Householder orthogonal square hidden layers, tiled scaled
  identity on rectangular layers, Weyl bias at a quarter of the Kaiming bound.
* ``qr_pb_pq`` — QR orthogonal weights at the declared element RMS, hash-uniform
  bias at the declared mean and standard deviation.

The particle table is the only thing that varies. Sequences are R2, Sobol,
Halton, a Fibonacci golden-angle lattice, a Korobov rank-1 lattice, a
stratified cell-center grid, a Latin hypercube, and the Kronecker-Weyl
sequence already used by ``hid_q``. Maps ``g`` / ``x`` / ``b`` / ``s`` match
the declared prior mean and standard deviation (inverse-normal, exact
marginals, variance-matched box, equal-norm sphere).

Sample generators still run, then the parameter is overwritten, so a seed
offset moves data, particle indices, and noise only. Omit ``install`` and the
recipe init is unchanged.

``hid_q`` and ``qr_pb_pq`` reproduce those published inits, including the
inverse-normal each one used. Other names share that arm's weights and biases.
"""
from __future__ import annotations

import contextvars
import gc
import hashlib
import json
import math
import warnings

import numpy as np
import torch
from torch import nn

SEQUENCES = ("r2", "sobol", "halton", "fib", "r1", "strat", "lhs", "weyl")
MAPS = ("g", "x", "b", "s")
ARMS = ("hid_q", "qr_pb_pq")
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
_GAIN = math.sqrt(2.0 / (1.0 + 5.0))  # calculate_gain("leaky_relu", sqrt(5))
_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_TAG = "_ortho_tag"
_STATE = {"name": None, "arm": None, "seq": None, "mp": None}
_PATCHED = False
_ORIG = {}
_Q_CACHE: dict[int, torch.Tensor] = {}
_IN_PRIOR = contextvars.ContextVar("k3p_det_prior", default=False)
_OPT_COUNTER = [0]

# Acklam inverse-normal. float64 in and out, matching the hid_q prior.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def _format_name(arm: str, seq: str, mp: str) -> str:
    if arm == "hid_q" and seq == "weyl" and mp == "g":
        return "hid_q"
    if arm == "qr_pb_pq" and seq == "r2" and mp == "g":
        return "qr_pb_pq"
    if mp == "g":
        return f"{arm}_{seq}"
    return f"{arm}_{seq}_{mp}"


VARIANTS = tuple(_format_name(arm, seq, mp) for arm in ARMS for seq in SEQUENCES for mp in MAPS)


def parse(name: str) -> tuple[str, str, str]:
    """Return ``(arm, sequence, map)`` for one ``--init`` name."""
    if name == "hid_q":
        return "hid_q", "weyl", "g"
    if name == "qr_pb_pq":
        return "qr_pb_pq", "r2", "g"
    arm = None
    rest = None
    for candidate in ARMS:
        prefix = candidate + "_"
        if name.startswith(prefix):
            arm = candidate
            rest = name[len(prefix):]
            break
    if arm is None or not rest:
        raise ValueError(f"init must be one of {', '.join(VARIANTS)}")
    parts = rest.split("_")
    if len(parts) >= 2 and parts[-1] in MAPS:
        mp = parts[-1]
        seq = "_".join(parts[:-1])
    else:
        mp = "g"
        seq = rest
    if seq not in SEQUENCES or mp not in MAPS:
        raise ValueError(f"init must be one of {', '.join(VARIANTS)}")
    return arm, seq, mp


def install(name: str) -> str:
    """Activate one variant. Default training never calls this."""
    arm, seq, mp = parse(name)
    _ensure_patched()
    _OPT_COUNTER[0] = 0
    _STATE.update(name=name, arm=arm, seq=seq, mp=mp)
    print(json.dumps({"event": "det_init", "name": name, "arm": arm, "seq": seq, "map": mp}), flush=True)
    return name


def uninstall() -> None:
    """Leave the patches in place but inert, so the default init returns."""
    _STATE.update(name=None, arm=None, seq=None, mp=None)


def active() -> str | None:
    return _STATE["name"]


def torch_build() -> dict:
    return {
        "torch": torch.__version__,
        "torch_git": getattr(torch.version, "git_version", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_file": torch.__file__,
        "numpy": np.__version__,
    }


def acklam(p: torch.Tensor) -> torch.Tensor:
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


def _r2_numpy(n: int, d: int) -> np.ndarray:
    """Roberts R2 in ``[0, 1)``. Same recurrence the qr_pb_pq prior used."""
    phi = 2.0
    for _ in range(64):
        phi = (1.0 + phi) ** (1.0 / (d + 1))
    alpha = np.array([(1.0 / phi) ** (j + 1) for j in range(d)])
    return np.mod(0.5 + np.arange(1, n + 1)[:, None] * alpha[None, :], 1.0)


def _cube(seq: str, n: int, d: int) -> torch.Tensor:
    """Unit-cube points, shape ``(n, d)``, in ``(0, 1)``."""
    if n == 0 or d == 0:
        return torch.zeros(n, d, dtype=torch.float64)
    if seq == "weyl":
        return _weyl(n, d)
    if seq == "r2":
        return torch.from_numpy(np.ascontiguousarray(_r2_numpy(n, d))).clamp(1e-12, 1.0 - 1e-12)
    if seq == "sobol":
        engine = torch.quasirandom.SobolEngine(d, scramble=False)
        drawn = engine.draw(n + 1)[1:].to(dtype=torch.float64)
        return drawn.clamp(1e-12, 1.0 - 1e-12)
    if seq == "halton":
        cols = []
        for j in range(d):
            base = _PRIMES[j]
            col = torch.empty(n, dtype=torch.float64)
            for i in range(1, n + 1):
                value = 0.0
                scale = 1.0 / base
                index = i
                while index:
                    index, digit = divmod(index, base)
                    value += digit * scale
                    scale /= base
                col[i - 1] = value
            cols.append(col)
        return torch.stack(cols, 1).clamp(1e-12, 1.0 - 1e-12)
    if seq == "r1":
        target = int(round(n / _PHI)) % n or 1
        generator = target
        while math.gcd(generator, n) != 1:
            generator += 1
            if generator >= n:
                generator = 1
                break
        powers = [1]
        for _ in range(1, d):
            powers.append((powers[-1] * generator) % n)
        index = torch.arange(n, dtype=torch.float64)
        cols = [((index * power) % n + 0.5) / n for power in powers]
        return torch.stack(cols, 1)
    if seq == "strat":
        m = 1
        while m ** d < n:
            m += 1
        total = m ** d
        if total == n:
            indices = torch.arange(n)
        else:
            indices = torch.floor(torch.arange(n, dtype=torch.float64) * total / n).to(dtype=torch.long)
        cols = []
        for j in range(d):
            digit = (indices // (m ** j)) % m
            cols.append((digit.to(dtype=torch.float64) + 0.5) / m)
        return torch.stack(cols, 1)
    if seq == "lhs":
        base = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        cols = []
        for j in range(d):
            keys = torch.frac((torch.arange(n, dtype=torch.float64) + 1.0) * math.sqrt(_PRIMES[j]))
            cols.append(base[torch.argsort(keys, stable=True)])
        return torch.stack(cols, 1)
    raise ValueError(seq)


def _fib_disk(n: int, d: int) -> torch.Tensor:
    """Golden-angle sunflower in successive coordinate planes. Unit disk."""
    out = torch.zeros(n, d, dtype=torch.float64)
    if n == 0 or d == 0:
        return out
    pair = 0
    dim = 0
    index = torch.arange(n, dtype=torch.float64)
    while dim + 1 < d:
        radius = torch.sqrt((index + 0.5) / n)
        theta = index * _GOLDEN_ANGLE + pair * _GOLDEN_ANGLE / 2.0
        out[:, dim] = radius * torch.cos(theta)
        out[:, dim + 1] = radius * torch.sin(theta)
        dim += 2
        pair += 1
    if dim < d:
        frac = torch.frac((index + 1.0) * (1.0 / _PHI))
        out[:, dim] = 2.0 * frac - 1.0
    return out


def _declared(kind: str, a: float, b: float) -> tuple[float, float]:
    if kind == "uniform":
        return (a + b) / 2.0, (b - a) / math.sqrt(12.0)
    return a, b


def _exact(values: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    if values.numel() == 0 or std == 0:
        return torch.full_like(values, mean)
    centered = values - values.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True)
    flat = scale < 1e-12
    if bool(flat.any()):
        index = torch.arange(values.shape[0], dtype=torch.float64)
        alternate = index - index.mean()
        alternate = alternate / alternate.std(unbiased=False).clamp_min(1e-12)
        for j in range(values.shape[1]):
            if bool(flat[0, j]):
                centered[:, j] = alternate
        scale = centered.std(dim=0, unbiased=False, keepdim=True)
    return mean + std * centered / scale.clamp_min(1e-12)


def _sphere(direction: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    if direction.numel() == 0 or std == 0:
        return torch.full_like(direction, mean)
    centered = direction - direction.mean(dim=0, keepdim=True)
    norm = centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radius = float(std) * math.sqrt(centered.shape[1])
    return mean + centered / norm * radius


def _inverse(arm: str):
    if arm == "hid_q":
        return acklam
    return torch.special.ndtri


def mapped_cloud(shape: tuple[int, ...], kind: str, a: float, b: float,
                 seq: str, mp: str, arm: str) -> torch.Tensor:
    """Prior values with the declared mean and standard deviation."""
    if len(shape) != 2:
        raise ValueError(f"particle prior must be rank 2, got {shape}")
    n, d = shape
    mean, std = _declared(kind, a, b)
    inverse = _inverse(arm)
    if seq == "fib":
        disk = _fib_disk(n, d)
        if mp == "s":
            out = torch.zeros_like(disk)
            dim = 0
            while dim + 1 < d:
                plane = disk[:, dim:dim + 2]
                radius = plane.norm(dim=1, keepdim=True).clamp_min(1e-12)
                out[:, dim:dim + 2] = plane / radius * (std * math.sqrt(2.0))
                dim += 2
            if dim < d:
                sign = torch.where(disk[:, dim] >= 0, torch.ones(n, dtype=torch.float64),
                                   -torch.ones(n, dtype=torch.float64))
                out[:, dim] = sign * std
            return mean + out
        # Unit-disk coordinate second moment is 1/4. U[-1, 1] second moment is 1/3.
        scaled = torch.zeros_like(disk)
        dim = 0
        while dim + 1 < d:
            scaled[:, dim:dim + 2] = disk[:, dim:dim + 2] * (2.0 * std)
            dim += 2
        if dim < d:
            scaled[:, dim] = disk[:, dim] * (std * math.sqrt(3.0))
        if mp == "g":
            return mean + scaled
        if mp == "b":
            # Same angles, radius clipped to the variance-matched box.
            half = std * math.sqrt(3.0)
            reach = disk.abs().amax(dim=0, keepdim=True).clamp_min(1e-12)
            boxed = disk / reach * half
            return mean + boxed
        if mp == "x":
            return _exact(mean + scaled, mean, std)
        raise ValueError(mp)
    cube = _cube(seq, n, d)
    if mp == "g":
        if kind == "uniform":
            return a + (b - a) * cube
        return mean + std * inverse(cube)
    if mp == "x":
        if kind == "uniform":
            drawn = a + (b - a) * cube
        else:
            drawn = mean + std * inverse(cube)
        return _exact(drawn, mean, std)
    if mp == "b":
        half = std * math.sqrt(3.0)
        drawn = mean + (cube - 0.5) * 2.0 * half
        return _exact(drawn, mean, std)
    if mp == "s":
        direction = (cube - 0.5) if kind == "uniform" else inverse(cube)
        return _sphere(direction, mean, std)
    raise ValueError(mp)


def _published_r2(shape: tuple[int, ...], kind: str, a: float, b: float) -> torch.Tensor:
    """Bit-match of the qr_pb_pq R2 particle draw (no clamp, torch.special.ndtri)."""
    n, d = shape
    raw = _r2_numpy(n, int(np.prod(shape[1:])))
    u = torch.from_numpy(np.ascontiguousarray(raw)).reshape(shape)
    if kind == "uniform":
        return a + (b - a) * u
    return a + b * torch.special.ndtri(u)


def _fill_prior(tensor: torch.Tensor, kind: str, a: float, b: float) -> None:
    arm, seq, mp = _STATE["arm"], _STATE["seq"], _STATE["mp"]
    shape = tuple(tensor.shape)
    if arm == "qr_pb_pq" and seq == "r2" and mp == "g" and len(shape) == 2:
        values = _published_r2(shape, kind, float(a), float(b))
    elif arm == "hid_q" and seq == "weyl" and mp == "g":
        if len(shape) == 2:
            cube = _weyl(shape[0], shape[1])
        elif len(shape) == 1:
            cube = _weyl(shape[0], 1).reshape(shape)
        else:
            cube = _weyl(tensor.numel(), 1).reshape(shape)
        if kind == "uniform":
            values = float(a) + (float(b) - float(a)) * cube
        else:
            values = float(a) + float(b) * acklam(cube)
    else:
        values = mapped_cloud(shape, kind, float(a), float(b), seq, mp, arm)
    _place(tensor, values)


def _place(tensor: torch.Tensor, values: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.copy_(values.to(dtype=tensor.dtype, device=tensor.device))


def _householder(n: int) -> torch.Tensor:
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


def _unit(rows: int, cols: int) -> torch.Tensor:
    block = torch.zeros(rows, cols, dtype=torch.float64)
    if rows == 0 or cols == 0:
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


def _hid_matrix(rows: int, cols: int, std: float, numel: int) -> torch.Tensor:
    if rows == cols and rows > 0:
        return _householder(rows) * _GAIN
    unit = _unit(rows, cols)
    nnz = float(unit.abs().sum())
    if nnz == 0 or numel == 0:
        return unit
    return unit * (std * math.sqrt(numel / nnz))


def _write_hid_weight(weight: torch.Tensor) -> None:
    shape = tuple(weight.shape)
    if len(shape) < 2:
        return
    rows, cols = int(shape[0]), int(shape[1])
    std, _ = _fan(weight)
    block = _hid_matrix(rows, cols, std, weight.numel())
    if len(shape) == 2:
        out = block
    else:
        out = torch.zeros(shape, dtype=torch.float64)
        center = tuple(s // 2 for s in shape[2:])
        out[(slice(None), slice(None), *center)] = block
    _place(weight, out)


def _write_hid_bias(bias: torch.Tensor | None, weight: torch.Tensor) -> None:
    if bias is None:
        return
    _, bound = _fan(weight)
    bound *= 0.25
    n = bias.numel()
    vals = (2.0 * _weyl(n, 1).reshape(-1) - 1.0) * bound
    _place(bias, vals.reshape(bias.shape))


def _bounds(args, kwargs) -> tuple[float, float]:
    lo = args[0] if len(args) > 0 else kwargs.get("from", 0.0)
    hi = args[1] if len(args) > 1 else kwargs.get("to", 1.0)
    return float(lo), float(hi)


def _normal_stats(args, kwargs) -> tuple[float, float]:
    mean = args[0] if len(args) > 0 else kwargs.get("mean", 0.0)
    std = args[1] if len(args) > 1 else kwargs.get("std", 1.0)
    return float(mean), float(std)


def _hash_u01(key: int, n: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = (np.arange(n, dtype=np.uint64) + np.uint64(key)) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)


def _key(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") >> 1


def _source(key: int, rows: int, cols: int) -> torch.Tensor:
    u = torch.from_numpy(_hash_u01(key, rows * cols)).reshape(rows, cols)
    return torch.special.ndtri(u)


def _semi_orthogonal(key: int, rows: int, cols: int) -> torch.Tensor:
    flip = rows < cols
    r, c = (cols, rows) if flip else (rows, cols)
    q, rr = torch.linalg.qr(_source(key, r, c))
    diag = torch.sign(torch.diagonal(rr))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    q = q * diag
    return q.T.contiguous() if flip else q


def _declared_rms_mean_std(tag) -> tuple[float, float, float]:
    kind, a, b = tag
    if kind == "uniform":
        return math.sqrt((a * a + a * b + b * b) / 3.0), (a + b) / 2.0, (b - a) / math.sqrt(12.0)
    return math.sqrt(a * a + b * b), a, b


def _pattern_bias(key: int, shape, tag) -> torch.Tensor:
    n = int(np.prod(shape)) if len(shape) else 1
    _, mean, std = _declared_rms_mean_std(tag)
    u = torch.from_numpy(_hash_u01(key, n)) * 2 - 1
    if n > 1:
        u = (u - u.mean()) / u.std(unbiased=False)
    return (mean + std * u).reshape(tuple(shape))


def _owners() -> dict:
    own = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for obj in gc.get_objects():
            try:
                if isinstance(obj, nn.Module):
                    for pname, param in obj.named_parameters(recurse=False):
                        own.setdefault(id(param), (obj, pname))
            except Exception:
                pass
    return own


def _is_identity_or_const(value: torch.Tensor) -> bool:
    flat = value.detach().cpu().flatten()
    if flat.numel() == 0:
        return True
    if bool((flat == flat[0]).all()):
        return True
    if value.ndim == 2 and value.shape[0] == value.shape[1]:
        return bool(torch.equal(value.detach().cpu(), torch.eye(value.shape[0], dtype=value.dtype, device="cpu")))
    return False


def _apply_qr(opt) -> None:
    from particlegan.particle_prior import ParticlePrior

    oi = _OPT_COUNTER[0]
    _OPT_COUNTER[0] += 1
    own = _owners()
    params = [p for group in opt.param_groups for p in group["params"]]
    for pi, param in enumerate(params):
        owner, pname = own.get(id(param), (None, None))
        tagged = getattr(param, _TAG, None)
        fresh = tagged is not None and tagged[1] == param._version
        key = _key(oi, pi, tuple(param.shape))
        if tagged is None or not fresh:
            continue
        is_prior = getattr(param, "_det_role", None) == "prior" or (
            isinstance(owner, ParticlePrior) and pname == "z")
        if is_prior and param.ndim == 2:
            kind, a, b = tagged[0]
            _fill_prior(param, kind, a, b)
            continue
        if pname == "bias" and param.ndim == 1:
            _place(param, _pattern_bias(key, param.shape, tagged[0]))
            continue
        if param.ndim < 1:
            continue
        rms = _declared_rms_mean_std(tagged[0])[0]
        rows = param.shape[0]
        cols = int(np.prod(param.shape[1:])) if param.ndim >= 2 else param.shape[0]
        if param.ndim == 1:
            rows = 1
        q = _semi_orthogonal(key, rows, cols)
        _place(param, (q * rms * math.sqrt(max(rows, cols))).reshape(param.shape))


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
    _ORIG["adam"] = torch.optim.Adam.__init__

    def linear_reset(self) -> None:
        self.weight._det_role = "weight"
        _ORIG["linear"](self)
        if _STATE["arm"] == "hid_q":
            _write_hid_weight(self.weight)
            _write_hid_bias(self.bias, self.weight)

    def conv_reset(self) -> None:
        self.weight._det_role = "weight"
        _ORIG["conv"](self)
        if _STATE["arm"] == "hid_q":
            _write_hid_weight(self.weight)
            _write_hid_bias(self.bias, self.weight)

    def uniform_(self, *args, **kwargs):
        out = _ORIG["uniform"](self, *args, **kwargs)
        if isinstance(self, nn.Parameter) and _STATE["arm"]:
            lo, hi = _bounds(args, kwargs)
            if _STATE["arm"] == "hid_q" and getattr(self, "_det_role", None) == "prior":
                _fill_prior(self, "uniform", lo, hi)
            setattr(self, _TAG, (("uniform", lo, hi), self._version))
        return out

    def normal_(self, *args, **kwargs):
        out = _ORIG["normal"](self, *args, **kwargs)
        if isinstance(self, nn.Parameter) and _STATE["arm"] and _IN_PRIOR.get():
            mean, std = _normal_stats(args, kwargs)
            self._det_role = "prior"
            if _STATE["arm"] == "hid_q":
                _fill_prior(self, "normal", mean, std)
            setattr(self, _TAG, (("normal", mean, std), self._version))
        return out

    def prior_init(self, *args, **kwargs):
        token = _IN_PRIOR.set(True)
        try:
            _ORIG["prior"](self, *args, **kwargs)
        finally:
            _IN_PRIOR.reset(token)
        if isinstance(getattr(self, "z", None), nn.Parameter):
            self.z._det_role = "prior"

    def adam_init(self, params, *args, **kwargs):
        _ORIG["adam"](self, params, *args, **kwargs)
        if _STATE["arm"] == "qr_pb_pq":
            _apply_qr(self)

    nn.Linear.reset_parameters = linear_reset
    nn.modules.conv._ConvNd.reset_parameters = conv_reset
    torch.Tensor.uniform_ = uniform_
    torch.Tensor.normal_ = normal_
    ParticlePrior.__init__ = prior_init
    torch.optim.Adam.__init__ = adam_init
    _PATCHED = True
