"""Scrambled-free quasi-random replacements for K3P training sample streams.

``rng`` (the default) leaves every draw on the torch RNG that the call site
already used. ``sobol`` and ``r2`` are deterministic low-discrepancy sequences.
Each ``(stream, dimension)`` engine starts at the first point and advances
across updates; it is not restarted per update and it is not scrambled.

Points map onto the same distributions as the torch draws they replace:
uniform indices via ``floor(u * n)``, non-uniform categoricals via the inverse
CDF (so unequal masses stay unequal), and Gaussians via the standard-normal
inverse CDF. There is no per-mode quota, stratification, or batch balancing.

Streams, advanced only inside :func:`update` (evaluation and initialization
stay on the torch RNG):

* ``data`` — real-batch index / uniform / Gaussian draws
* ``critic_index`` — the first particle-index or latent draw of an update
* ``generator_latent`` — later particle-index, latent, and prior-jitter draws
* ``output`` — generator output noise

Critic input noise and evaluation draws are not replaced. With a deterministic
initialization the decisive training streams no longer depend on the seed, but
input noise and eval can still move weights or reported metrics with the seed.
Repeat bit-identity is the same command twice.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os

import torch

KINDS = ("rng", "sobol", "r2")

_kind = "rng"
_engines: dict[tuple[str, int], object] = {}
_depth = 0
_suspended = 0
_prior_slot = 0
_last_role = "critic_index"


def kind() -> str:
    return _kind


def replacing() -> bool:
    """True only while a non-rng stream is inside an unsuspended update."""
    return _kind != "rng" and _depth > 0 and _suspended <= 0


def configure(kind_name: str) -> str:
    """Select the stream and drop any sequences already drawn."""
    global _kind, _depth, _suspended, _prior_slot, _last_role
    kind_name = str(kind_name).lower()
    if kind_name not in KINDS:
        raise ValueError(f"sample stream must be one of {KINDS}, got {kind_name!r}")
    _kind = kind_name
    _engines.clear()
    _depth = 0
    _suspended = 0
    _prior_slot = 0
    _last_role = "critic_index"
    print(json.dumps({"event": "sample_stream", "kind": _kind}), flush=True)
    return _kind


def add_argument(parser) -> None:
    """Add ``--sample-stream`` without changing the default (torch RNG)."""
    parser.add_argument(
        "--sample-stream",
        choices=KINDS,
        default=None,
        help="training sample stream (default: torch RNG, or $K3P_SAMPLE_STREAM)",
    )


def apply(value: str | None) -> str:
    """Apply a CLI value. Omitted values read ``K3P_SAMPLE_STREAM`` or stay ``rng``."""
    if value is None:
        value = os.environ.get("K3P_SAMPLE_STREAM", "rng")
    return configure(value)


@contextmanager
def update():
    """One training update. Re-entrant; only the outermost entry resets prior roles.

    A no-op when the kind is ``rng``, so the default path does not touch sequences
    or the torch RNG.
    """
    global _depth, _prior_slot
    if _kind == "rng":
        yield
        return
    if _depth == 0:
        _prior_slot = 0
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1


@contextmanager
def suspend():
    """Keep evaluation draws on the torch RNG even if they run inside an update."""
    global _suspended
    if _kind == "rng":
        yield
        return
    _suspended += 1
    try:
        yield
    finally:
        _suspended -= 1


def take_prior_role() -> str:
    """First prior draw of an update is the critic index; the rest are generator latent."""
    global _prior_slot, _last_role
    role = "critic_index" if _prior_slot == 0 else "generator_latent"
    _prior_slot += 1
    _last_role = role
    return role


def last_prior_role() -> str:
    """Role of the prior draw that just ran (MoG noise and particle jitter do not take another)."""
    return _last_role


def points_drawn(stream: str, dim: int) -> int:
    engine = _engines.get((stream, int(dim)))
    return 0 if engine is None else int(engine.drawn)  # type: ignore[attr-defined]


class _Sobol:
    def __init__(self, dim: int):
        self.engine = torch.quasirandom.SobolEngine(dim, scramble=False)
        self.drawn = 0

    def draw(self, count: int) -> torch.Tensor:
        points = self.engine.draw(count, dtype=torch.float64)
        self.drawn += count
        return points


class _R2:
    """Roberts R2 in ``[0, 1)^d``. Same alpha recurrence as the qr_pb_pq init."""

    def __init__(self, dim: int):
        phi = 2.0
        for _ in range(64):
            phi = (1.0 + phi) ** (1.0 / (dim + 1))
        self.alpha = torch.tensor([(1.0 / phi) ** (j + 1) for j in range(dim)], dtype=torch.float64)
        self.n = 1
        self.drawn = 0

    def draw(self, count: int) -> torch.Tensor:
        index = torch.arange(self.n, self.n + count, dtype=torch.float64).unsqueeze(1)
        self.n += count
        self.drawn += count
        return torch.remainder(0.5 + index * self.alpha, 1.0)


def _engine(stream: str, dim: int):
    key = (stream, int(dim))
    engine = _engines.get(key)
    if engine is None:
        engine = _Sobol(dim) if _kind == "sobol" else _R2(dim)
        _engines[key] = engine
    return engine


def _uniforms(stream: str, count: int, dim: int) -> torch.Tensor:
    if count <= 0 or dim <= 0:
        raise ValueError("quasi-random draws need a positive count and dimension")
    return _engine(stream, dim).draw(int(count))


def _open_unit(u: torch.Tensor) -> torch.Tensor:
    # Unscrambled Sobol starts at 0; ndtri(0) is -inf.
    eps = torch.finfo(torch.float64).eps
    return u.clamp(eps, 1.0 - eps)


def _device(device):
    if device is None:
        return torch.empty(0).device
    return torch.device(device)


def _place(tensor: torch.Tensor, device, dtype) -> torch.Tensor:
    if dtype is None:
        return tensor.to(_device(device))
    return tensor.to(device=_device(device), dtype=dtype)


def uniforms(stream: str, count: int, dim: int, *, device=None, dtype=None) -> torch.Tensor:
    """``count`` points in ``[0, 1)^dim`` on the requested device and dtype."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    return _place(_uniforms(stream, count, dim), device, dtype)


def indices(stream: str, count: int, n_choices: int, *, device=None) -> torch.Tensor:
    """Uniform integer indices in ``[0, n_choices)`` via ``floor(u * n)``."""
    if n_choices <= 0:
        raise ValueError("n_choices must be positive")
    unit = _uniforms(stream, count, 1).reshape(-1)
    idx = torch.floor(unit * int(n_choices)).to(torch.long).clamp(max=int(n_choices) - 1)
    return idx.to(_device(device))


def categorical(stream: str, count: int, weights, *, device=None) -> torch.Tensor:
    """Inverse-CDF sample of a categorical. Not a uniform floor and not a quota."""
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    cdf = torch.cumsum(weights, 0)
    cdf = cdf / cdf[-1]
    unit = torch.minimum(_uniforms(stream, count, 1).reshape(-1), cdf[-1])
    idx = torch.searchsorted(cdf.contiguous(), unit, right=False).to(torch.long)
    idx = idx.clamp(max=weights.numel() - 1)
    return idx.to(_device(device))


def normal(stream: str, shape, *, device=None, dtype=None) -> torch.Tensor:
    """Independent standard normals. One low-discrepancy point per row."""
    shape = tuple(int(size) for size in shape)
    if not shape or shape[0] <= 0:
        raise ValueError("normal shape must start with a positive batch")
    dim = 1
    for size in shape[1:]:
        dim *= size
    gauss = torch.special.ndtri(_open_unit(_uniforms(stream, shape[0], dim))).reshape(shape)
    return _place(gauss, device, dtype)


def index_and_normal(stream, count, n_choices, noise_dim, *, device=None, dtype=None):
    """One point per sample: coordinate 0 is a uniform index, the rest are Gaussian."""
    unit = _uniforms(stream, count, 1 + int(noise_dim))
    idx = torch.floor(unit[:, 0] * int(n_choices)).to(torch.long).clamp(max=int(n_choices) - 1)
    gauss = torch.special.ndtri(_open_unit(unit[:, 1:]))
    dev = _device(device)
    return idx.to(dev), _place(gauss, dev, dtype)


def categorical_and_normal(stream, count, weights, noise_dim, *, device=None, dtype=None):
    """One point per sample: coordinate 0 is an inverse-CDF category, the rest are Gaussian."""
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    cdf = torch.cumsum(weights, 0)
    cdf = cdf / cdf[-1]
    unit = _uniforms(stream, count, 1 + int(noise_dim))
    probe = torch.minimum(unit[:, 0].contiguous(), cdf[-1])
    idx = torch.searchsorted(cdf.contiguous(), probe, right=False).to(torch.long)
    idx = idx.clamp(max=weights.numel() - 1)
    gauss = torch.special.ndtri(_open_unit(unit[:, 1:]))
    dev = _device(device)
    return idx.to(dev), _place(gauss, dev, dtype)


def uniform_and_normal(stream, count, n_uniform, noise_dim, *, device=None, dtype=None):
    """Leading uniform coordinates (shared transforms stay shared) plus Gaussian tail."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    unit = _uniforms(stream, count, int(n_uniform) + int(noise_dim))
    dev = _device(device)
    uni = unit[:, : int(n_uniform)].to(device=dev, dtype=dtype)
    gauss = torch.special.ndtri(_open_unit(unit[:, int(n_uniform) :])).to(device=dev, dtype=dtype)
    return uni, gauss


def image_batch(centers: torch.Tensor, count: int, noise_std: float) -> torch.Tensor:
    """Template index plus isotropic jitter, clamped to ``[0, 1]``.

    The torch path is ``randint`` then ``randn_like``, matching the historical
    image hosts. The quasi-random path uses one joint point per image.
    """
    if not replacing():
        real = centers[torch.randint(len(centers), (count,))]
        return (real + noise_std * torch.randn_like(real)).clamp(0.0, 1.0)
    idx, noise = index_and_normal(
        "data", count, len(centers), int(centers[0].numel()),
        device=centers.device, dtype=centers.dtype,
    )
    real = centers[idx]
    return (real + float(noise_std) * noise.reshape(real.shape)).clamp(0.0, 1.0)
