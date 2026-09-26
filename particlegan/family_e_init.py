"""Family E: deterministic structured orthogonal initializers for K3P.

Every value is a closed-form function of tensor shape and parameter ordinal.
The torch RNG is advanced by the original ``uniform_`` / ``normal_`` call and
then overwritten, so later sample streams stay on the seed. Host writes that
land after that draw (``zeros_``, ``copy_`` of an identity) are kept.

Scale ``std`` matches the declared init's element RMS. Scale ``frob`` matches
the expected Frobenius norm of Kaiming ``uniform_(a=sqrt(5))`` onto the same
orthonormal frame. For a Linear or a convolution those two scalars are the
same number (fan-in equals the flattened column count).

The particle table is the Roberts R2 sequence in the declared distribution.
That prior is the one ring passes have used. Bias schemes are the ones already
on the board: zero, a standardized golden pattern at the declared RMS, and a
Weyl sequence at a quarter of the default bound.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import torch
from torch import nn

_PHI = (math.sqrt(5.0) - 1.0) / 2.0
_SQRT2 = math.sqrt(2.0)
FAMILIES = ("giv", "giw", "cay", "rft", "cir", "haar", "walsh", "sob", "lat", "but", "exp")
BIASES = ("bz", "pb", "wq")
_SCALES = ("std", "frob")
# Identity of a single-variant patch. ``--init default`` installs this name.
# ``--init <name>`` overrides it. Every name in ``names()`` stays available.
PATCH_DEFAULT = "giv_bz_pq"
TAG = "_family_e_tag"
_ORIG: dict = {}
_PATCHED = False
_OWNERS: dict[int, tuple[nn.Module, str]] = {}
_STATE = {"name": None, "family": None, "bias": None, "scale": "std", "counter": 0}
LOG: list = []


def names() -> tuple[str, ...]:
    """Every ``--init`` name this module accepts."""
    out = [f"{fam}_{bias}_pq" for fam in FAMILIES for bias in BIASES]
    out += [f"{fam}_{bias}_pq_frob" for fam in FAMILIES for bias in BIASES]
    return tuple(out)


def parse(name: str) -> tuple[str, str, str]:
    scale = "frob" if name.endswith("_frob") else "std"
    body = name[: -len("_frob")] if scale == "frob" else name
    if not body.endswith("_pq"):
        raise ValueError(f"unknown init {name!r}; choose from {', '.join(names())}")
    family, bias = body[: -len("_pq")].rsplit("_", 1)
    if family not in FAMILIES or bias not in BIASES or scale not in _SCALES:
        raise ValueError(f"unknown init {name!r}; choose from {', '.join(names())}")
    return family, bias, scale


def install(name: str) -> str:
    """Activate one variant. Default training never calls this."""
    family, bias, scale = parse(name)
    _ensure_patched()
    _STATE.update(name=name, family=family, bias=bias, scale=scale, counter=0)
    LOG.clear()
    print(json.dumps({"event": "family_e_init", "name": name,
                      "family": family, "bias": bias, "scale": scale, "prior": "r2"}), flush=True)
    return name


def uninstall() -> None:
    """Restore the wrapped methods. Used by tests."""
    global _PATCHED
    if not _PATCHED:
        _STATE["name"] = None
        return
    torch.Tensor.uniform_ = _ORIG["uniform_"]
    torch.Tensor.normal_ = _ORIG["normal_"]
    nn.Module.register_parameter = _ORIG["register_parameter"]
    torch.optim.Adam.__init__ = _ORIG["adam"]
    _PATCHED = False
    _STATE["name"] = None
    _OWNERS.clear()


def active() -> str | None:
    return _STATE["name"]


def dump(path: str) -> str:
    blob = "".join(row["sha256"] for row in LOG).encode()
    all_sha = hashlib.sha256(blob).hexdigest()
    payload = {
        "name": _STATE["name"], "family": _STATE["family"], "bias": _STATE["bias"],
        "scale": _STATE["scale"], "prior": "r2", "all_params_sha256": all_sha,
        "n": len(LOG), "params": LOG,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return all_sha


def _ensure_patched() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _ORIG["uniform_"] = torch.Tensor.uniform_
    _ORIG["normal_"] = torch.Tensor.normal_
    _ORIG["register_parameter"] = nn.Module.register_parameter
    _ORIG["adam"] = torch.optim.Adam.__init__

    def uniform_(self, *args, **kwargs):
        out = _ORIG["uniform_"](self, *args, **kwargs)
        if isinstance(self, nn.Parameter) and _STATE["name"]:
            lo = float(args[0] if len(args) > 0 else kwargs.get("from", 0.0))
            hi = float(args[1] if len(args) > 1 else kwargs.get("to", 1.0))
            _tag(self, ("uniform", lo, hi))
        return out

    def normal_(self, *args, **kwargs):
        out = _ORIG["normal_"](self, *args, **kwargs)
        if isinstance(self, nn.Parameter) and _STATE["name"]:
            mean = float(args[0] if len(args) > 0 else kwargs.get("mean", 0.0))
            std = float(args[1] if len(args) > 1 else kwargs.get("std", 1.0))
            _tag(self, ("normal", mean, std))
        return out

    def register_parameter(self, name, param):
        _ORIG["register_parameter"](self, name, param)
        if isinstance(param, nn.Parameter):
            _OWNERS[id(param)] = (self, name)

    def adam_init(opt, params, *args, **kwargs):
        _ORIG["adam"](opt, params, *args, **kwargs)
        if _STATE["name"]:
            _apply(opt)

    torch.Tensor.uniform_ = uniform_
    torch.Tensor.normal_ = normal_
    nn.Module.register_parameter = register_parameter
    torch.optim.Adam.__init__ = adam_init
    _PATCHED = True


def _alloc_key() -> int:
    key = int(_STATE["counter"])
    _STATE["counter"] = key + 1
    return key


def _tag(param: nn.Parameter, dist: tuple) -> None:
    prev = getattr(param, TAG, None)
    key = prev[2] if prev is not None else _alloc_key()
    owner = _OWNERS.get(id(param))
    oname = type(owner[0]).__name__ if owner is not None else ""
    pname = owner[1] if owner is not None else ""
    setattr(param, TAG, (dist, param._version, key, oname, pname))


def _declared(dist: tuple) -> tuple[float, float, float]:
    """Return ``(rms, mean, half_width)`` of the declared box or normal."""
    kind, a, b = dist
    if kind == "uniform":
        rms = math.sqrt((a * a + a * b + b * b) / 3.0)
        return rms, (a + b) / 2.0, (b - a) / 2.0
    rms = math.sqrt(a * a + b * b)
    return rms, a, abs(b) * math.sqrt(3.0)


def _kaiming_sigma(fan_in: int) -> float:
    return 1.0 / math.sqrt(3.0 * max(fan_in, 1))


def _scale(rows: int, cols: int, dist: tuple, weight: torch.Tensor) -> float:
    rms = _declared(dist)[0]
    rank = min(rows, cols)
    if _STATE["scale"] == "frob":
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        sigma = _kaiming_sigma(int(fan_in))
        target = sigma * math.sqrt(rows * cols)
        return target / math.sqrt(rank)
    return rms * math.sqrt(max(rows, cols))


def _frac(value: float) -> float:
    return value - math.floor(value)


def _thue(index: int) -> float:
    return 1.0 if (int(index).bit_count() & 1) == 0 else -1.0


def _sign_columns(q: torch.Tensor, key: int) -> torch.Tensor:
    signs = torch.tensor([_thue(j + key * 17) for j in range(q.shape[1])], dtype=torch.float64)
    return q * signs


def _givens_rows(q: torch.Tensor, i: int, j: int, theta: float) -> None:
    c, s = math.cos(theta), math.sin(theta)
    qi = q[i].clone()
    qj = q[j].clone()
    q[i] = c * qi + s * qj
    q[j] = -s * qi + c * qj


def _square_givens(n: int, key: int, alpha: float) -> torch.Tensor:
    q = torch.eye(n, dtype=torch.float64)
    step = 0
    for i in range(n):
        for j in range(i + 1, n):
            theta = math.pi * _frac((step + 1 + key) * alpha)
            _givens_rows(q, i, j, theta)
            step += 1
    return q


def _real_fourier(n: int) -> torch.Tensor:
    index = torch.arange(n, dtype=torch.float64)
    cols = [torch.full((n,), 1.0 / math.sqrt(n), dtype=torch.float64)]
    limit = n // 2 - 1 if n % 2 == 0 else (n - 1) // 2
    for k in range(1, limit + 1):
        angle = 2.0 * math.pi * index * k / n
        cols.append(math.sqrt(2.0 / n) * torch.cos(angle))
        cols.append(math.sqrt(2.0 / n) * torch.sin(angle))
    if n % 2 == 0 and n > 1:
        cols.append((1.0 - 2.0 * (index % 2)) / math.sqrt(n))
    return torch.stack(cols, dim=1)


def _circulant(n: int, key: int) -> torch.Tensor:
    spec = torch.empty(n, dtype=torch.float64)
    spec[0] = _thue(key)
    for k in range(1, n // 2 + 1):
        sign = _thue(k + key)
        if 2 * k == n:
            spec[k] = sign
        else:
            spec[k] = sign
            spec[n - k] = sign
    column = torch.fft.ifft(spec.to(torch.complex128)).real
    return torch.stack([torch.roll(column, i) for i in range(n)], dim=0)


def _haar(n: int) -> torch.Tensor:
    """Orthonormal Haar butterfly. Pairs that fit are rotated by pi/4; any n works."""
    q = torch.eye(n, dtype=torch.float64)
    span = 1
    s = math.sqrt(0.5)
    while span < n:
        start = 0
        while start < n:
            for k in range(span):
                i = start + k
                j = i + span
                if j >= n or j >= start + 2 * span:
                    break
                qi = q[i].clone()
                qj = q[j].clone()
                q[i] = s * (qi + qj)
                q[j] = s * (qi - qj)
            start += 2 * span
        span *= 2
    return q


def _sylvester(n: int) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h


def _walsh(n: int) -> torch.Tensor:
    size = 1 << (n - 1).bit_length()
    had = _sylvester(size)
    changes = (had[:, 1:] * had[:, :-1] < 0).sum(dim=1)
    order = torch.argsort(changes, stable=True)
    sequency = had[order] / math.sqrt(size)
    if size == n:
        return sequency
    return _qr_square(sequency[:n, :n])


def _butterfly(n: int, key: int) -> torch.Tensor:
    q = torch.eye(n, dtype=torch.float64)
    stride = 1
    stage = 0
    while stride < n:
        start = 0
        k = 0
        while start < n:
            for offset in range(stride):
                i = start + offset
                j = i + stride
                if j >= n or j >= start + 2 * stride:
                    break
                theta = math.pi * _frac((stage + 1) * (k + 1) * _PHI + key * _PHI)
                _givens_rows(q, i, j, theta)
                k += 1
            start += 2 * stride
        stride *= 2
        stage += 1
    return q


def _skew_dense(n: int, key: int) -> torch.Tensor:
    a = torch.zeros((n, n), dtype=torch.float64)
    scale = 1.0 / math.sqrt(n)
    for i in range(n):
        for j in range(i + 1, n):
            value = math.sin(2.0 * math.pi * _frac((i + 1) * (j + 1) * _PHI + key * _PHI)) * scale
            a[i, j] = value
            a[j, i] = -value
    return a


def _cayley(n: int, key: int) -> torch.Tensor:
    a = _skew_dense(n, key)
    eye = torch.eye(n, dtype=torch.float64)
    solved = torch.linalg.solve(eye + a, eye)
    return (eye - a) @ solved


def _expm_skew(n: int, key: int) -> torch.Tensor:
    """exp of a tridiagonal skew matrix via scaling-and-squaring. Orthogonal in exact arithmetic."""
    a = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n - 1):
        value = math.sin(2.0 * math.pi * _frac((i + 1) * _PHI + (key + 1) * _SQRT2))
        a[i, i + 1] = value
        a[i + 1, i] = -value
    norm = float(torch.linalg.matrix_norm(a, ord=1))
    steps = max(0, int(math.ceil(math.log2(max(norm, 1.0)))) + 1)
    b = a / (2.0 ** steps)
    term = torch.eye(n, dtype=torch.float64)
    out = term.clone()
    for k in range(1, 18):
        term = term @ (b / k)
        out = out + term
    for _ in range(steps):
        out = out @ out
    return out


def _primitive(mask: int, degree: int) -> bool:
    if degree < 1 or (mask & 1) == 0:
        return False
    poly = mask | (1 << degree)
    state = 1
    for period in range(1, 1 << degree):
        bit = state & 1
        state >>= 1
        if bit:
            state ^= poly >> 1
        if state == 1:
            return period == (1 << degree) - 1
    return False


_POLY: list[int] = []


def _polys(count: int) -> list[int]:
    """Primitive polynomials, bit k = coefficient of x^k, including x^degree and 1."""
    while len(_POLY) < count:
        degree = 1
        found = False
        while not found:
            for mask in range(1, 1 << degree, 2):
                if _primitive(mask, degree):
                    poly = mask | (1 << degree)
                    if poly not in _POLY:
                        _POLY.append(poly)
                        found = True
                        break
            if not found:
                degree += 1
    return _POLY[:count]


def _sobol_directions(dim: int, bits: int = 32) -> list[list[int]]:
    directions = [[1 << (bits - 1 - i) for i in range(bits)]]
    polys = _polys(dim - 1) if dim > 1 else []
    for poly in polys:
        degree = poly.bit_length() - 1
        m = [0] + [1] * degree
        for i in range(degree + 1, bits + 1):
            acc = (m[i - degree] << degree) ^ m[i - degree]
            for j in range(1, degree):
                if (poly >> (degree - j)) & 1:
                    acc ^= m[i - j] << j
            m.append(acc)
        directions.append([m[i] << (bits - i) for i in range(1, bits + 1)])
    return directions


def _sobol(n: int, d: int, key: int) -> torch.Tensor:
    directions = _sobol_directions(d)
    skip = 1 + key
    total = skip + n
    x = [0] * d
    rows = []
    for i in range(total):
        if i:
            bit = (i & -i).bit_length() - 1
            for axis in range(d):
                x[axis] ^= directions[axis][bit]
        if i >= skip:
            rows.append([value / float(1 << 32) for value in x])
    unit = torch.tensor(rows, dtype=torch.float64).clamp(1e-12, 1.0 - 1e-12)
    return torch.special.ndtri(unit)


def _lattice(n: int, d: int, key: int) -> torch.Tensor:
    primes = []
    p = 2
    while len(primes) < d:
        if all(p % q != 0 for q in primes):
            primes.append(p)
        p += 1 if p == 2 else 2
    alpha = torch.tensor([math.sqrt(p) for p in primes], dtype=torch.float64)
    index = torch.arange(1, n + 1, dtype=torch.float64)[:, None]
    unit = torch.remainder(index * alpha + (key + 1) * _PHI, 1.0).clamp(1e-12, 1.0 - 1e-12)
    return torch.special.ndtri(unit)


def _qr_square(block: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(block)
    sign = torch.where(torch.diagonal(r) < 0, -torch.ones(r.shape[0], dtype=torch.float64),
                       torch.ones(r.shape[0], dtype=torch.float64))
    return q * sign


def _semi_from_square(q: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    size = max(rows, cols)
    if q.shape[0] != size:
        q = _qr_square(q[:size, :size]) if q.shape[0] > size else q
    if rows >= cols:
        return q[:, :cols]
    return q[:rows, :]


def _semi_from_qr(source: torch.Tensor) -> torch.Tensor:
    rows, cols = source.shape
    if rows >= cols:
        q, r = torch.linalg.qr(source)
        sign = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
        return q * sign.to(dtype=torch.float64)
    q, r = torch.linalg.qr(source.T)
    sign = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    return (q * sign.to(dtype=torch.float64)).T


def basis(family: str, rows: int, cols: int, key: int = 0) -> torch.Tensor:
    """Semi-orthogonal ``[rows, cols]`` float64 matrix. Columns (or rows) are orthonormal."""
    if rows < 1 or cols < 1:
        return torch.zeros((rows, cols), dtype=torch.float64)
    size = max(rows, cols)
    if family == "giv":
        square = _square_givens(size, key, _PHI)
    elif family == "giw":
        square = _square_givens(size, key, _SQRT2)
    elif family == "cay":
        square = _cayley(size, key)
    elif family == "rft":
        square = _real_fourier(size)
    elif family == "cir":
        square = _circulant(size, key)
    elif family == "haar":
        square = _haar(size)
    elif family == "walsh":
        square = _walsh(size)
    elif family == "sob":
        square = None
    elif family == "lat":
        square = None
    elif family == "but":
        square = _butterfly(size, key)
    elif family == "exp":
        square = _expm_skew(size, key)
    else:
        raise ValueError(family)
    if family == "sob":
        frame = _semi_from_qr(_sobol(rows, cols, key))
    elif family == "lat":
        frame = _semi_from_qr(_lattice(rows, cols, key))
    else:
        frame = _semi_from_square(square, rows, cols)
    return _sign_columns(frame, key)


def _r2(n: int, d: int) -> torch.Tensor:
    """Roberts R2 in ``[0, 1)``. Same recurrence the earlier screens used."""
    phi = 2.0
    for _ in range(64):
        phi = (1.0 + phi) ** (1.0 / (d + 1))
    alpha = torch.tensor([phi ** (-(j + 1)) for j in range(d)], dtype=torch.float64)
    index = torch.arange(1, n + 1, dtype=torch.float64)[:, None]
    return torch.remainder(0.5 + index * alpha, 1.0).clamp(1e-12, 1.0 - 1e-12)


def _map_unit(unit: torch.Tensor, dist: tuple) -> torch.Tensor:
    kind, a, b = dist
    if kind == "uniform":
        return a + (b - a) * unit
    return a + b * torch.special.ndtri(unit)


def _pattern(n: int, dist: tuple, key: int) -> torch.Tensor:
    _rms, mean, _half = _declared(dist)
    std = _rms if dist[0] == "normal" else _declared(dist)[0]
    # ``_declared`` rms is the element std for a centered uniform and for a normal.
    if dist[0] == "uniform":
        std = (dist[2] - dist[1]) / math.sqrt(12.0)
        mean = (dist[1] + dist[2]) / 2.0
    else:
        mean, std = dist[1], abs(dist[2])
    if n == 0:
        return torch.zeros(0, dtype=torch.float64)
    unit = torch.tensor([_frac((i + 1) * _PHI + key * _PHI) for i in range(n)], dtype=torch.float64)
    unit = unit * 2.0 - 1.0
    if n > 1:
        centered = unit - unit.mean()
        denom = torch.sqrt(torch.mean(centered * centered)).clamp_min(1e-12)
        unit = centered / denom
    return mean + std * unit


def _weyl_quarter(n: int, dist: tuple, key: int) -> torch.Tensor:
    _rms, mean, half = _declared(dist)
    if n == 0:
        return torch.zeros(0, dtype=torch.float64)
    index = torch.arange(1, n + 1, dtype=torch.float64)
    unit = torch.remainder(index * _SQRT2 + (key + 1) * _PHI, 1.0)
    return mean + (2.0 * unit - 1.0) * (half / 4.0)


def _rewrite(param: nn.Parameter, dist: tuple, key: int, oname: str, pname: str) -> tuple[torch.Tensor, str]:
    if oname == "ParticlePrior" and pname == "z" and param.ndim == 2:
        unit = _r2(int(param.shape[0]), int(param.shape[1]))
        return _map_unit(unit, dist).reshape(param.shape), "prior_r2"
    if param.ndim >= 2 and param.numel() > 0:
        rows = int(param.shape[0])
        cols = int(param.numel() // rows)
        frame = basis(_STATE["family"], rows, cols, key)
        value = frame * _scale(rows, cols, dist, param)
        return value.reshape(param.shape), f"weight_{_STATE['family']}_{_STATE['scale']}"
    if param.ndim == 1:
        bias = _STATE["bias"]
        if bias == "bz":
            return torch.zeros(param.shape, dtype=torch.float64), "bias_zero"
        if bias == "wq":
            return _weyl_quarter(param.numel(), dist, key).reshape(param.shape), "bias_weyl_quarter"
        return _pattern(param.numel(), dist, key).reshape(param.shape), "bias_pattern"
    return _pattern(param.numel(), dist, key).reshape(param.shape), "pattern_other"


@torch.no_grad()
def _apply(opt) -> None:
    params = [p for group in opt.param_groups for p in group["params"]]
    for param in params:
        tagged = getattr(param, TAG, None)
        if tagged is None:
            rule = "keep_untagged"
            fresh = False
        else:
            dist, version, key, oname, pname = tagged
            fresh = version == param._version
            rule = None
            if fresh:
                value, rule = _rewrite(param, dist, key, oname, pname)
                param.copy_(value.to(dtype=param.dtype, device=param.device))
            else:
                rule = "keep_host"
        flat = param.detach().cpu().contiguous()
        LOG.append({
            "owner": tagged[3] if tagged else "",
            "name": tagged[4] if tagged else "",
            "shape": list(param.shape),
            "rule": rule,
            "fresh": fresh if tagged is not None else False,
            "sha256": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
        })


def _dump_if_requested() -> None:
    path = os.environ.get("K3P_INIT_DUMP")
    if not path or not _STATE["name"]:
        return
    os.makedirs(path, exist_ok=True)
    dump(os.path.join(path, "family-e-init.json"))


import atexit

atexit.register(_dump_if_requested)
