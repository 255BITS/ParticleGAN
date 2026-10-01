"""Deterministic structured-orthogonal initializers for K3P.

Activated only by ``configure(name)``. The default PyTorch path is untouched
when this module is never configured.

Every rewritten value is a closed-form function of construction order and
shape, computed in float64 on CPU and then cast into the parameter. The torch
RNG is still consumed by the original ``uniform_`` / ``normal_`` call so
sample and noise streams keep the recipe seed; the stored values do not.

Bases (all orthonormal before the scalar scale):

* ``hadamard`` — Sylvester Hadamard, bit-reversed rows and columns, Thue-Morse
  signs, thin QR so a truncated block stays orthonormal.
* ``dct`` / ``dst`` — orthonormal DCT-II / DST-I columns (a frequency band
  chosen by the layer index).
* ``householder`` — product of Householder reflectors from fixed sine vectors,
  then a sign-fixed QR.

Weight scale matches ``kaiming_uniform_(a=sqrt(5))`` for that shape, either
by Frobenius norm or by the Marchenko-Pastur spectral edge
``sigma * (sqrt(rows) + sqrt(cols))``. Biases are zeros or a DCT vector at
the Kaiming bias RMS. A learned particle table keeps its declared law:
a Weyl sequence in a uniform box, or a trig-whitened / Weyl-normal cloud.

This is not the earlier hash-signed Hadamard, not QR of a splitmix Gaussian,
and not a frozen torch-seed prior.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn

NAMES = (
    "had_tm_frob",
    "had_tm_spec",
    "had_tm_frob_weyl",
    "had_tm_frob_bias",
    "dct_frob",
    "dct_spec",
    "dct_frob_bias",
    "dct_frob_weyl",
    "dst_frob",
    "dst_spec",
    "house_frob",
    "house_spec",
)

_VARIANTS = {
    "had_tm_frob": ("hadamard", "frob", "zero", "whiten"),
    "had_tm_spec": ("hadamard", "spec", "zero", "whiten"),
    "had_tm_frob_weyl": ("hadamard", "frob", "zero", "weyl"),
    "had_tm_frob_bias": ("hadamard", "frob", "dct", "whiten"),
    "dct_frob": ("dct", "frob", "zero", "whiten"),
    "dct_spec": ("dct", "spec", "zero", "whiten"),
    "dct_frob_bias": ("dct", "frob", "dct", "whiten"),
    "dct_frob_weyl": ("dct", "frob", "zero", "weyl"),
    "dst_frob": ("dst", "frob", "zero", "whiten"),
    "dst_spec": ("dst", "spec", "zero", "whiten"),
    "house_frob": ("householder", "frob", "zero", "whiten"),
    "house_spec": ("householder", "spec", "zero", "whiten"),
}

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_state = {
    "variant": None,
    "frozen": False,
    "count": 0,
    "owners": {},
    "writes": [],
    "installed": False,
}


def configure(name: str) -> None:
    """Install hooks and select one variant. Construction order restarts at 0."""
    if name not in _VARIANTS:
        raise ValueError(f"unknown init {name!r}; choose from {', '.join(NAMES)}")
    _state["variant"] = _VARIANTS[name]
    _state["name"] = name
    _state["frozen"] = False
    _state["count"] = 0
    _state["writes"] = []
    _install()


def deactivate() -> None:
    """Stop rewriting. Hooks stay installed and become pass-through."""
    _state["variant"] = None
    _state["frozen"] = True


def freeze() -> None:
    """Stop rewriting once optimizers exist so training noise cannot be overwritten."""
    if _state["frozen"] or _state["variant"] is None:
        return
    _state["frozen"] = True
    parts = [row["sha256"] for row in _state["writes"]]
    digest = hashlib.sha256("".join(parts).encode()).hexdigest() if parts else None
    print(json.dumps({
        "event": "STRUCTURED_INIT",
        "variant": _state["name"],
        "writes": len(_state["writes"]),
        "sha256": digest,
        "rules": sorted({row["rule"] for row in _state["writes"]}),
    }), flush=True)


def install_seed_offset(offset: int) -> None:
    """Shift every torch seed by ``offset``. Offset 0 is a no-op.

    Sample and noise streams move. Structured init does not read the seed.
    """
    offset = int(offset)
    if offset == 0 or _state.get("seed_offset"):
        return
    _state["seed_offset"] = offset
    modulus = 2 ** 63

    def shift(seed):
        return (int(seed) + offset) % modulus

    def manual_seed(seed):
        shifted = shift(seed)
        if torch.cuda.is_available() and not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(shifted)
        return torch.default_generator.manual_seed(shifted)

    torch.manual_seed = manual_seed
    torch.random.manual_seed = manual_seed
    if torch.cuda.is_available():
        cuda_seed, cuda_seed_all = torch.cuda.manual_seed, torch.cuda.manual_seed_all
        torch.cuda.manual_seed = lambda seed: cuda_seed(shift(seed))
        torch.cuda.manual_seed_all = lambda seed: cuda_seed_all(shift(seed))
    base = torch.Generator

    class _OffsetGenerator(base):
        def manual_seed(self, seed):
            return super().manual_seed(shift(seed))

    torch.Generator = _OffsetGenerator


def digest_params(params) -> str:
    hasher = hashlib.sha256()
    for param in params:
        value = param.detach().cpu().contiguous()
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _install() -> None:
    if _state["installed"]:
        return
    _state["installed"] = True
    uniform, normal = torch.Tensor.uniform_, torch.Tensor.normal_
    register = nn.Module.register_parameter

    def uniform_(self, *args, **kwargs):
        out = uniform(self, *args, **kwargs)
        if _rewriting(self):
            lo = float(args[0] if args else kwargs.get("from", 0.0))
            hi = float(args[1] if len(args) > 1 else kwargs.get("to", 1.0))
            _rewrite(self, ("uniform", lo, hi))
        return out

    def normal_(self, *args, **kwargs):
        out = normal(self, *args, **kwargs)
        if _rewriting(self):
            mean = float(args[0] if args else kwargs.get("mean", 0.0))
            std = float(args[1] if len(args) > 1 else kwargs.get("std", 1.0))
            _rewrite(self, ("normal", mean, std))
        return out

    def register_parameter(self, name, param):
        register(self, name, param)
        if isinstance(param, nn.Parameter):
            _state["owners"][id(param)] = (self, name)

    torch.Tensor.uniform_ = uniform_
    torch.Tensor.normal_ = normal_
    nn.Module.register_parameter = register_parameter
    adam = torch.optim.Adam.__init__

    def adam_init(self, params, *args, **kwargs):
        # The K3P drivers froze rewrites at the first Adam so later noise draws
        # stay random. Values written during construction are already in place.
        freeze()
        return adam(self, params, *args, **kwargs)

    torch.optim.Adam.__init__ = adam_init


def _rewriting(tensor) -> bool:
    return (_state["variant"] is not None and not _state["frozen"]
            and isinstance(tensor, nn.Parameter))


def _orthonormalize(block: torch.Tensor) -> torch.Tensor:
    """Thin QR with positive R diagonal. ``block`` is tall or square, float64 CPU."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        q, r = torch.linalg.qr(block)
    finally:
        torch.set_num_threads(threads)
    diag = torch.sign(torch.diagonal(r))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    return q * diag


def _arange(n: int, dtype=torch.int64) -> torch.Tensor:
    return torch.arange(n, dtype=dtype, device="cpu")


def _thue_morse(idx: torch.Tensor) -> torch.Tensor:
    x = idx.to(device="cpu", dtype=torch.int64)
    x = x ^ (x >> 32)
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    ones = torch.ones(x.shape, dtype=torch.float64, device="cpu")
    return torch.where((x & 1) == 0, ones, -ones)


def _bit_reverse(n: int, bits: int) -> torch.Tensor:
    idx = _arange(n)
    rev = torch.zeros(n, dtype=torch.int64, device="cpu")
    for bit in range(bits):
        rev = (rev << 1) | ((idx >> bit) & 1)
    return rev


def _sylvester(n: int) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float64, device="cpu")
    while matrix.shape[0] < n:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix


def _basis(rows: int, cols: int, key: int) -> torch.Tensor:
    """Orthonormal columns if ``rows >= cols``, else orthonormal rows. float64 CPU."""
    basis, _, _, _ = _state["variant"]
    flip = rows < cols
    tall_rows, tall_cols = (cols, rows) if flip else (rows, cols)
    if basis == "hadamard":
        block = _hadamard_block(tall_rows, tall_cols, key)
    elif basis == "dct":
        block = _dct_block(tall_rows, tall_cols, key)
    elif basis == "dst":
        block = _dst_block(tall_rows, tall_cols, key)
    elif basis == "householder":
        block = _householder_block(tall_rows, tall_cols, key)
    else:
        raise RuntimeError(basis)
    return block.T.contiguous() if flip else block


def _hadamard_block(rows: int, cols: int, key: int) -> torch.Tensor:
    width = 1 << (max(rows, cols) - 1).bit_length()
    bits = width.bit_length() - 1
    matrix = _sylvester(width)
    order = _bit_reverse(width, bits)
    row_order = torch.roll(order, shifts=int(key % width))
    col_order = torch.roll(order, shifts=int((key * 3 + 1) % width))
    signed = _thue_morse(_arange(width) + (key % 4096) * 17)
    col_signs = _thue_morse(_arange(width) * 2 + 1 + (key % 4096))
    block = signed[:, None] * matrix[row_order][:, col_order] * col_signs[None, :]
    block = block[:rows, :cols] / math.sqrt(width)
    return _orthonormalize(block)


def _dct_block(rows: int, cols: int, key: int) -> torch.Tensor:
    start = int(key % rows)
    freq = (_arange(cols) + start) % rows
    position = torch.arange(rows, dtype=torch.float64, device="cpu")[:, None]
    alpha = torch.where(
        freq == 0,
        torch.full((cols,), 1.0 / math.sqrt(rows), dtype=torch.float64, device="cpu"),
        torch.full((cols,), math.sqrt(2.0 / rows), dtype=torch.float64, device="cpu"),
    )
    return alpha[None, :] * torch.cos(math.pi * (position + 0.5) * freq.to(torch.float64) / rows)


def _dst_block(rows: int, cols: int, key: int) -> torch.Tensor:
    start = int(key % rows)
    freq = (_arange(cols) + start) % rows + 1
    position = torch.arange(1, rows + 1, dtype=torch.float64, device="cpu")[:, None]
    scale = math.sqrt(2.0 / (rows + 1))
    return scale * torch.sin(math.pi * position * freq.to(torch.float64) / (rows + 1))


def _householder_block(rows: int, cols: int, key: int) -> torch.Tensor:
    reflectors = torch.eye(rows, dtype=torch.float64, device="cpu")
    for index in range(rows - 1):
        length = rows - index
        grid = torch.arange(length, dtype=torch.float64, device="cpu")
        vector = torch.sin(math.pi * (grid + 1.0) * ((index + 1 + (key % 997)) / (length + 1.0)))
        pivot = vector[0]
        vector = vector.clone()
        vector[0] = pivot + (1.0 if float(pivot) >= 0.0 else -1.0) * torch.linalg.vector_norm(vector)
        norm = torch.linalg.vector_norm(vector)
        if float(norm) == 0.0:
            continue
        vector = vector / norm
        block = reflectors[index:, :]
        reflectors[index:, :] = block - (2.0 * vector[:, None]) * (vector @ block)
    return _orthonormalize(reflectors[:, :cols])


def _kaiming_sigma(fan_in: int) -> float:
    # leaky_relu gain at a=sqrt(5) is 1/sqrt(3); uniform draws use that std.
    return 1.0 / math.sqrt(3.0 * fan_in)


def _weight_scale(rows: int, cols: int, fan_in: int) -> float:
    _, scale, _, _ = _state["variant"]
    sigma = _kaiming_sigma(fan_in)
    rank = min(rows, cols)
    if scale == "frob":
        target = sigma * math.sqrt(rows * cols)
        return target / math.sqrt(rank)
    if scale == "spec":
        return sigma * (math.sqrt(rows) + math.sqrt(cols))
    raise RuntimeError(scale)


def _weyl(n: int, d: int, key: int) -> torch.Tensor:
    index = torch.arange(n, dtype=torch.float64, device="cpu")[:, None]
    alphas = torch.tensor([_PRIMES[(key + j) % len(_PRIMES)] for j in range(d)],
                          dtype=torch.float64, device="cpu").sqrt()
    return torch.remainder((index + 0.5) * alphas, 1.0)


def _prior(n: int, d: int, kind: str, a: float, b: float, key: int) -> torch.Tensor:
    _, _, _, prior = _state["variant"]
    if kind == "uniform":
        return a + (b - a) * _weyl(n, d, key)
    if n >= d:
        if prior == "weyl":
            unit = _weyl(n, d, key).clamp(1e-12, 1.0 - 1e-12)
            return a + b * torch.special.ndtri(unit)
        source_key = key
        index = torch.arange(1, n + 1, dtype=torch.float64, device="cpu")[:, None]
        alpha = torch.tensor([math.sqrt(_PRIMES[(source_key + j) % len(_PRIMES)]) for j in range(d)],
                             dtype=torch.float64, device="cpu")
        beta = torch.tensor([math.sqrt(_PRIMES[(source_key + d + j) % len(_PRIMES)]) for j in range(d)],
                            dtype=torch.float64, device="cpu")
        source = torch.sin(2.0 * math.pi * index * alpha) + torch.cos(2.0 * math.pi * index * beta)
        return a + b * math.sqrt(n) * _orthonormalize(source)
    unit = _weyl(n, d, key).clamp(1e-12, 1.0 - 1e-12)
    return a + b * torch.special.ndtri(unit)


def _bias(n: int, fan_in: int, key: int) -> torch.Tensor:
    _, _, bias, _ = _state["variant"]
    if bias == "zero" or n == 0:
        return torch.zeros(n, dtype=torch.float64, device="cpu")
    position = torch.arange(n, dtype=torch.float64, device="cpu")
    freq = 1 + (key % max(n, 1))
    vector = torch.cos(math.pi * (position + 0.5) * freq / n)
    rms = torch.sqrt(torch.mean(vector * vector)).clamp_min(1e-12)
    target = _kaiming_sigma(max(fan_in, 1))
    return vector * (target / rms)


def _fan_in(weight: torch.Tensor) -> int:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    return int(fan_in)


def _rewrite(param: nn.Parameter, tag) -> None:
    key = _state["count"]
    _state["count"] += 1
    owner, name = _state["owners"].get(id(param), (None, None))
    kind, a, b = tag
    oname = type(owner).__name__ if owner is not None else ""
    rule = None
    value = None
    if oname == "ParticlePrior" and name == "z" and param.ndim == 2:
        value = _prior(param.shape[0], param.shape[1], kind, a, b, key)
        rule = f"prior_{kind}_{_state['variant'][3]}"
    elif param.ndim >= 2 and param.numel() > 0:
        rows = int(param.shape[0])
        cols = int(param.numel() // rows)
        fan_in = _fan_in(param) if param.ndim >= 2 else cols
        scale = _weight_scale(rows, cols, max(fan_in, 1))
        value = _basis(rows, cols, key) * scale
        rule = f"weight_{_state['variant'][0]}_{_state['variant'][1]}"
    elif param.ndim == 1:
        fan_in = _sibling_fan_in(owner, name) or (1.0 / (b * b) if kind == "uniform" and b else param.numel())
        if name == "bias" or (name or "").endswith(".bias"):
            value = _bias(param.numel(), int(round(fan_in)), key)
            rule = f"bias_{_state['variant'][2]}"
        else:
            rms = _declared_rms(kind, a, b)
            position = torch.arange(param.numel(), dtype=torch.float64, device="cpu")
            vector = torch.cos(math.pi * (position + 0.5) * (1 + key % max(param.numel(), 1)) / max(param.numel(), 1))
            center = _declared_mean(kind, a, b)
            value = center + vector * (rms / torch.sqrt(torch.mean(vector * vector)).clamp_min(1e-12))
            rule = "vector_dct"
    elif param.ndim == 0:
        value = torch.tensor(_declared_mean(kind, a, b), dtype=torch.float64, device="cpu")
        rule = "scalar_mean"
    if value is None:
        return
    with torch.no_grad():
        param.copy_(value.reshape(param.shape).to(dtype=param.dtype, device=param.device))
    raw = param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    _state["writes"].append({
        "key": key, "rule": rule, "shape": list(param.shape), "name": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    })


def _sibling_fan_in(owner, name):
    if owner is None or not name:
        return None
    weight_name = name[:-4] + "weight" if name.endswith("bias") else None
    if weight_name is None:
        return None
    weight = dict(owner.named_parameters()).get(weight_name)
    if weight is None or weight.ndim < 2:
        return None
    return _fan_in(weight)


def _declared_mean(kind, a, b) -> float:
    return (a + b) / 2.0 if kind == "uniform" else a


def _declared_rms(kind, a, b) -> float:
    if kind == "uniform":
        return math.sqrt((a * a + a * b + b * b) / 3.0)
    return math.sqrt(a * a + b * b)
