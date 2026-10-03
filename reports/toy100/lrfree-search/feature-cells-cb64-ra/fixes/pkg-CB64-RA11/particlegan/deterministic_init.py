"""Deterministic initial parameters that do not read the global torch seed.

``install`` replaces Parameter ``uniform_`` / ``normal_`` draws. The caller's
RNG is still advanced by the same count, so later sample and noise draws keep
following the seed. Written values come from a seed-independent source.

Kinds:

* ``ortho_lsuv`` — orthogonal Linear/Conv weights (float64 QR of a Sobol
  Gaussian), then one scalar per layer so a fixed Sobol probe has activation
  variance 1. Biases and the learned particle prior are Sobol fills of the
  declared uniform/normal range. They are not orthogonal.
* ``sobol``, ``halton`` — NOT orthogonal. Each tensor is a low-discrepancy
  sequence mapped into the bounds of the ``uniform_`` / ``normal_`` call
  (``kaiming_uniform_(a=sqrt(5))`` is that call for ``nn.Linear``).
* ``fixedgen`` — NOT orthogonal. One ``torch.Generator`` seeded with
  ``FIXED_GENERATOR_SEED`` draws the declared uniform/normal distribution.

The default training path does not call ``install``.
"""
from __future__ import annotations

import json
import math
import os
import warnings

import torch
from torch import nn

KINDS = ("ortho_lsuv", "sobol", "halton", "fixedgen")
# sobol, halton and fixedgen are low-discrepancy or fixed-generator fills.
# They are not orthogonal. ortho_lsuv orthogonalizes Linear/Conv weights only.
NOT_ORTHOGONAL = ("sobol", "halton", "fixedgen")
FIXED_GENERATOR_SEED = 123456789
LSUV_TARGET_VARIANCE = 1.0
LSUV_PROBE = 64
_WRAPS = {
    "OutputNoise", "IsolatedOutputNoise", "StatefulInputNoise", "InputNoise",
    "_InputAdapter", "_OutputAdapter",
}
_LAYER = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)

_kind = None
_orig_uniform = None
_orig_normal = None
_orig_register = None
_owners: dict[int, tuple[nn.Module, str]] = {}
_qmc_skip = 1
_fixed_gen = None
_prepared: set[int] = set()
_busy = False


def active_kind():
    return _kind


def install(kind: str) -> str:
    """Install one deterministic init. Raises if a different kind is already installed."""
    global _kind, _orig_uniform, _orig_normal, _orig_register, _qmc_skip, _fixed_gen
    if kind not in KINDS:
        raise ValueError(f"init must be one of {', '.join(KINDS)}")
    if _kind is not None:
        if _kind != kind:
            raise RuntimeError(f"deterministic init already installed as {_kind}")
        return _kind
    _kind = kind
    _qmc_skip = 1
    _fixed_gen = torch.Generator(device="cpu")
    _fixed_gen.manual_seed(FIXED_GENERATOR_SEED)
    _orig_uniform = torch.Tensor.uniform_
    _orig_normal = torch.Tensor.normal_
    _orig_register = nn.Module.register_parameter
    torch.Tensor.uniform_ = _uniform
    torch.Tensor.normal_ = _normal
    nn.Module.register_parameter = _register
    if kind == "ortho_lsuv":
        _wrap_construction()
    orthogonal = kind == "ortho_lsuv"
    print(json.dumps({
        "event": "deterministic_init",
        "kind": kind,
        "orthogonal": orthogonal,
        "weights": "orthogonal+lsuv" if orthogonal else "not orthogonal",
        "biases_and_particle_prior": "sobol" if orthogonal else kind,
        "fixed_generator_seed": FIXED_GENERATOR_SEED if kind == "fixedgen" else None,
    }), flush=True)
    return kind


def use_init(name: str | None) -> str | None:
    """Install ``name``, or ``K3P_INIT`` when ``name`` is empty. ``None`` leaves init unchanged."""
    chosen = name or os.environ.get("K3P_INIT") or None
    if not chosen:
        return None
    return install(chosen)


def add_init_argument(parser):
    parser.add_argument(
        "--init",
        default=None,
        choices=KINDS,
        help="Deterministic init, independent of the torch seed. "
             "Default: unchanged PyTorch init. "
             "sobol, halton and fixedgen are NOT orthogonal. "
             "ortho_lsuv orthogonalizes Linear/Conv weights and LSUV-rescales them "
             "on a fixed Sobol probe. Biases and the learned particle prior are "
             "deterministic for every option.",
    )


def _register(self, name, param):
    _orig_register(self, name, param)
    if isinstance(param, nn.Parameter):
        _owners[id(param)] = (self, name)


def _owner(param):
    return _owners.get(id(param), (None, None))


def _burn(param, op, args, generator):
    """Advance the RNG the original draw would have consumed, and discard it."""
    scratch = torch.empty(param.shape, dtype=param.dtype, device=param.device)
    if op == "uniform":
        _orig_uniform(scratch, args[0], args[1], generator=generator)
    else:
        _orig_normal(scratch, args[0], args[1], generator=generator)


def _engine(d, kind):
    from scipy.stats import qmc
    d = max(1, int(d))
    if d > 21201:
        raise ValueError(f"quasi-random dimension {d} exceeds 21201")
    if kind == "halton":
        return qmc.Halton(d=d, scramble=False)
    return qmc.Sobol(d=d, scramble=False)


def _draw(engine, n):
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return np.clip(engine.random(int(n)), 1e-12, 1.0 - 1e-12)


def _unit(n, d):
    """Next disjoint Sobol/Halton block. Halton only when that kind is installed."""
    global _qmc_skip
    engine = _engine(d, "halton" if _kind == "halton" else "sobol")
    engine.fast_forward(_qmc_skip)
    block = _draw(engine, n)
    _qmc_skip += int(n)
    return block


def _fixed_sobol(n, d):
    """Probe points: always the same Sobol block, independent of init order."""
    engine = _engine(d, "sobol")
    engine.fast_forward(1)
    return _draw(engine, n)


def _mapped(shape, dist, a, b):
    import numpy as np
    from scipy.special import ndtri
    shape = tuple(int(v) for v in shape)
    fan = int(np.prod(shape[1:])) if len(shape) >= 2 else 1
    if len(shape) >= 2 and fan <= 21201:
        unit = _unit(shape[0], fan).reshape(shape)
    else:
        unit = _unit(int(np.prod(shape)) or 1, 1).reshape(shape)
    if dist == "uniform":
        values = a + (b - a) * unit
    else:
        values = a + b * ndtri(unit)
    return values


def _write(param, values):
    import numpy as np
    cpu = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float64))
    param.copy_(cpu.to(dtype=param.dtype))


def _dot(a, b):
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _householder_q(source):
    """Thin Q with orthonormal columns and a nonnegative R diagonal.

    Pure Python float64. Householder reflections stay orthonormal even when
    the Sobol Gaussian is rank-deficient, and the bits do not depend on BLAS.
    """
    rows = source.tolist()
    height, width = len(rows), len(rows[0])
    cols = [[rows[r][c] for r in range(height)] for c in range(width)]
    reflectors = []
    signs = []
    for k in range(width):
        x = cols[k][k:]
        norm = math.sqrt(_dot(x, x))
        if norm == 0.0:
            reflectors.append(None)
            signs.append(1.0)
            continue
        sign = 1.0 if x[0] >= 0.0 else -1.0
        u = [x[0] + sign * norm]
        u.extend(x[1:])
        inv = 1.0 / math.sqrt(_dot(u, u))
        u = [value * inv for value in u]
        reflectors.append(u)
        for j in range(k, width):
            segment = cols[j][k:]
            coeff = 2.0 * _dot(u, segment)
            for i, value in enumerate(u):
                cols[j][k + i] -= coeff * value
        signs.append(-1.0 if cols[k][k] < 0.0 else 1.0)
    qcols = [[1.0 if r == c else 0.0 for r in range(height)] for c in range(width)]
    for k in range(width - 1, -1, -1):
        u = reflectors[k]
        if u is None:
            continue
        length = len(u)
        for j in range(width):
            segment = qcols[j][k:k + length]
            coeff = 2.0 * _dot(u, segment)
            for i, value in enumerate(u):
                qcols[j][k + i] -= coeff * value
    for c, sign in enumerate(signs):
        if sign < 0.0:
            for r in range(height):
                qcols[c][r] = -qcols[c][r]
    return [[qcols[c][r] for c in range(width)] for r in range(height)]


def _semi_orthogonal(rows, cols):
    import numpy as np
    from scipy.special import ndtri
    flip = rows < cols
    tall, wide = (cols, rows) if flip else (rows, cols)
    # Elementwise 1D Sobol. A d-dimensional Sobol cloud of d points is
    # rank-deficient after ndtri (its first point is the cube center).
    source = ndtri(_unit(tall * wide, 1)).reshape(tall, wide)
    q = np.asarray(_householder_q(source), dtype=np.float64)
    return np.ascontiguousarray(q.T if flip else q)


def _orthogonal_weight(param, module):
    import numpy as np
    if isinstance(module, nn.Linear):
        out, inn = param.shape
        matrix = _semi_orthogonal(out, inn)
    elif isinstance(module, nn.Conv2d) and module.groups == 1:
        out = param.shape[0]
        matrix = _semi_orthogonal(out, param[0].numel()).reshape(param.shape)
    elif isinstance(module, nn.ConvTranspose2d) and module.groups == 1:
        inn, out = param.shape[:2]
        matrix = _semi_orthogonal(out, inn * int(param[0, 0].numel()))
        matrix = np.transpose(matrix.reshape(out, inn, *param.shape[2:]), (1, 0, *range(2, param.ndim)))
    else:
        return False
    _write(param, np.ascontiguousarray(matrix))
    return True


def _fill(param, dist, a, b, generator):
    global _busy
    _busy = True
    try:
        _burn(param, dist, (a, b), generator)
        module, name = _owner(param)
        if (_kind == "ortho_lsuv" and name == "weight" and isinstance(module, _LAYER)
                and _orthogonal_weight(param, module)):
            pass
        elif _kind == "fixedgen":
            cpu = torch.empty(param.shape, dtype=param.dtype, device="cpu")
            if dist == "uniform":
                _orig_uniform(cpu, a, b, generator=_fixed_gen)
            else:
                _orig_normal(cpu, a, b, generator=_fixed_gen)
            param.copy_(cpu)
        else:
            _write(param, _mapped(param.shape, dist, a, b))
        param._det_tag = {"version": param._version, "dist": dist, "a": a, "b": b, "name": name}
    finally:
        _busy = False
    return param


def _uniform(self, *args, **kwargs):
    if _busy or _kind is None or not isinstance(self, nn.Parameter):
        return _orig_uniform(self, *args, **kwargs)
    lo = args[0] if args else kwargs.get("from", 0.0)
    hi = args[1] if len(args) > 1 else kwargs.get("to", 1.0)
    return _fill(self, "uniform", float(lo), float(hi), kwargs.get("generator"))


def _normal(self, *args, **kwargs):
    if _busy or _kind is None or not isinstance(self, nn.Parameter):
        return _orig_normal(self, *args, **kwargs)
    mean = args[0] if args else kwargs.get("mean", 0.0)
    std = args[1] if len(args) > 1 else kwargs.get("std", 1.0)
    return _fill(self, "normal", float(mean), float(std), kwargs.get("generator"))


def _fresh(param):
    tag = getattr(param, "_det_tag", None)
    return tag is not None and tag["version"] == param._version


def _unwrap(module):
    seen = set()
    while module is not None and id(module) not in seen:
        seen.add(id(module))
        if type(module).__name__ not in _WRAPS:
            return module
        inner = getattr(module, "model", None)
        if not isinstance(inner, nn.Module):
            return module
        module = inner
    return module


def _probe(module):
    layers = [m for m in module.modules() if isinstance(m, _LAYER)]
    if not layers:
        return None
    first, last = layers[0], layers[-1]
    if isinstance(first, nn.Linear):
        din = first.in_features
        fourier = int(getattr(module, "fourier", 0) or 0)
        if fourier:
            din = din // (1 + 2 * fourier)
        critic = (isinstance(last, nn.Linear) and last.out_features == 1
                  and not any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in layers))
        if critic or hasattr(module, "in_dim"):
            width = int(getattr(module, "in_dim", din))
            return torch.from_numpy(_fixed_sobol(LSUV_PROBE, width) * 10.0 - 5.0)
        from scipy.special import ndtri
        return torch.from_numpy(ndtri(_fixed_sobol(LSUV_PROBE, din)))
    channels = first.in_channels
    unit = _fixed_sobol(LSUV_PROBE, channels * 8 * 8).reshape(LSUV_PROBE, channels, 8, 8)
    return torch.from_numpy(unit)


def prepare_modules(*modules) -> None:
    """LSUV-rescale orthogonal weights. No-op unless ``ortho_lsuv`` is installed."""
    if _kind != "ortho_lsuv":
        return
    for module in modules:
        if isinstance(module, nn.Module):
            _lsuv(_unwrap(module))


def _measure_at(root, layer):
    """Post-activation when the next sibling is not another Linear/Conv."""
    for parent in root.modules():
        children = list(parent.children())
        for index, child in enumerate(children):
            if child is layer and index + 1 < len(children) and not isinstance(children[index + 1], _LAYER):
                return children[index + 1]
    return layer


def _lsuv(module):
    if module is None or id(module) in _prepared:
        return
    _prepared.add(id(module))
    originals = [m for m in module.modules() if isinstance(m, _LAYER) and _fresh(m.weight)]
    if not originals:
        return
    probe = _probe(module)
    if probe is None:
        return
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        work = _clone_float64(module).eval()
        original_layers = [m for m in module.modules() if isinstance(m, _LAYER)]
        work_all = [m for m in work.modules() if isinstance(m, _LAYER)]
        fresh = [m for m in original_layers if _fresh(m.weight)]
        # Positional lockstep: deepcopy preserves module order.
        work_layers = [layer for layer, orig in zip(work_all, original_layers) if orig in fresh]
        probe = probe.to(dtype=torch.float64)
        for orig, layer in zip(fresh, work_layers):
            captured = {}
            site = _measure_at(work, layer)

            def _hook(mod, inputs, output, box=captured):
                box["y"] = output.detach()

            handle = site.register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    work(probe)
            except Exception as exc:
                print(json.dumps({"event": "lsuv_skip", "module": type(module).__name__,
                                  "error": repr(exc)}), flush=True)
                return
            finally:
                handle.remove()
            values = captured.get("y")
            if values is None or values.numel() < 2:
                continue
            flat = values.detach().double().reshape(-1)
            mean = float(flat.mean())
            var = float((flat - mean).pow(2).mean())
            if not math.isfinite(var) or var < 1e-12:
                continue
            scale = math.sqrt(LSUV_TARGET_VARIANCE / var)
            with torch.no_grad():
                layer.weight.mul_(scale)
                _scale_param(orig.weight, scale)
                if orig.bias is not None and _fresh(orig.bias):
                    layer.bias.mul_(scale)
                    _scale_param(orig.bias, scale)
            if os.environ.get("K3P_INIT_LOG") == "1":
                print(json.dumps({"event": "lsuv", "module": type(module).__name__,
                                  "layer": type(orig).__name__, "var": var, "scale": scale}), flush=True)
    finally:
        torch.set_num_threads(threads)


def _clone_float64(module):
    import copy
    return copy.deepcopy(module).cpu().double()


def _scale_param(param, scale):
    updated = param.detach().cpu().double().mul_(scale).to(dtype=param.dtype)
    param.copy_(updated)
    if getattr(param, "_det_tag", None) is not None:
        param._det_tag = {**param._det_tag, "version": param._version}


def _wrap_construction():
    from particlegan.training import GANTrainer
    from particlegan.recipes import Recipe
    if getattr(GANTrainer.__init__, "_det_init", False):
        return
    orig_init = GANTrainer.__init__

    def init(self, recipe, generator, discriminator, *args, **kwargs):
        prepare_modules(generator, discriminator)
        return orig_init(self, recipe, generator, discriminator, *args, **kwargs)

    init._det_init = True
    GANTrainer.__init__ = init
    orig_make = Recipe.make_optimizers

    def make(self, generator, discriminator=None, *args, **kwargs):
        prepare_modules(generator, discriminator, kwargs.get("encoder"))
        return orig_make(self, generator, discriminator, *args, **kwargs)

    Recipe.make_optimizers = make
