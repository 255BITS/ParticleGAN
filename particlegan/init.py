"""Deterministic network initialization, in the style of ``torch.nn.init``.

``deterministic_orthogonal_(module, seed=0)`` rewrites a module's trainable
parameters in place. Each supported layer declares the distribution PyTorch's
own constructor draws from (``Uniform``/``Normal``); a matrix becomes an
orthogonal matrix at that distribution's RMS, a vector becomes a fixed pattern
with its mean and std, and a particle table becomes R2 low-discrepancy points
(``R2Normal``). Values come from hashing ``(seed, parameter index, shape)`` in
float64 on the CPU, so no RNG state is read or consumed and results do not
depend on device.

Layers declare their parameters with :func:`register`. Parameters no
declaration covers raise by default (``strict=True``), so a custom layer is
never silently left on a different init.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Union

import torch
from torch import nn

from . import _qr

__all__ = ["Uniform", "Normal", "R2Normal", "KEEP", "register", "declarations",
           "deterministic_orthogonal_"]


@dataclass(frozen=True)
class Uniform:
    """The constructor's U(low, high): matrices orthogonal at its RMS, vectors a pattern."""
    low: float
    high: float


@dataclass(frozen=True)
class Normal:
    """The constructor's N(mean, std**2): matrices orthogonal at its RMS, vectors a pattern."""
    mean: float = 0.0
    std: float = 1.0


@dataclass(frozen=True)
class R2Normal:
    """Particle table: each row is one R2 point mapped through the N(mean, std**2) quantile."""
    mean: float = 0.0
    std: float = 1.0


class _Keep:
    def __repr__(self):
        return "KEEP"


KEEP = _Keep()
"""Declare a parameter whose constructor value is deliberate (norm scales, temperatures)."""

Spec = Union[Uniform, Normal, R2Normal, _Keep]
Declarations = Union[Mapping[str, Spec], Callable[[nn.Module], Mapping[str, Spec]]]
_REGISTRY: dict[type, tuple[Declarations, Callable[[nn.Module], None] | None]] = {}


def register(cls: type, declarations: Declarations = None, *,
             finalize: Callable[[nn.Module], None] | None = None) -> None:
    """Declare how ``deterministic_orthogonal_`` treats ``cls``'s own parameters.

    ``declarations`` maps each direct parameter name (not a submodule's) to a
    spec: ``Uniform``, ``Normal``, ``R2Normal`` or ``KEEP``. Pass a callable
    ``module -> mapping`` when the spec depends on the instance, e.g. its fan-in.
    Declarations merge along the class hierarchy, so a subclass declares only
    the parameters it adds (or overrides). ``finalize(module)`` runs without
    grad after the call writes any parameter inside the module, for layout
    fix-ups such as zeroing an embedding padding row. Re-registering a class
    replaces its entry.
    """
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError("register expects an nn.Module subclass")
    if declarations is None:
        declarations = {}
    if not callable(declarations) and not isinstance(declarations, Mapping):
        raise TypeError("declarations must be a mapping or a callable returning one")
    _REGISTRY[cls] = (declarations, finalize)


def _resolve(module):
    merged, finalizers = {}, []
    for cls in reversed(type(module).__mro__):
        if cls not in _REGISTRY:
            continue
        entry, finalize = _REGISTRY[cls]
        found = entry(module) if callable(entry) else entry
        for name, spec in found.items():
            if not isinstance(spec, (Uniform, Normal, R2Normal, _Keep)):
                raise TypeError(f"{cls.__name__} declares {name!r} as {spec!r}; "
                                "use Uniform, Normal, R2Normal or KEEP")
            if not hasattr(module, name):
                raise ValueError(f"{cls.__name__} declares {name!r}, which "
                                 f"{type(module).__name__} does not have")
        merged.update(found)
        if finalize is not None:
            finalizers.append(finalize)
    return merged, finalizers


def _plan(module):
    if not isinstance(module, nn.Module):
        raise TypeError("expected an nn.Module")
    owners, finalizers, holders = {}, [], {}  # holders: parameter id -> owning class name
    for child in module.modules():
        if any(isinstance(p, nn.parameter.UninitializedParameter)
               for p in child.parameters(recurse=False)):
            raise ValueError("materialize lazy layers before initializing them")
        declared, finalize = _resolve(child)
        if finalize:
            finalizers.append((child, finalize))
        for name, parameter in child.named_parameters(recurse=False):
            holders.setdefault(id(parameter), type(child).__name__)
            if name in declared:
                owners.setdefault(id(parameter), declared[name])
    return owners, finalizers, holders


def declarations(module: nn.Module) -> dict[str, Spec | None]:
    """Resolved spec for every trainable parameter; ``None`` marks an undeclared one.

    Changes nothing. Use it to check a custom network before initializing it.
    """
    owners, _, _ = _plan(module)
    return {path: owners.get(id(parameter))
            for path, parameter in module.named_parameters() if parameter.requires_grad}


def _keep_constant(parameter):
    # Deliberate constructor values: zero vectors, constant or identity matrices.
    if parameter.ndim < 2:
        return bool(torch.all(parameter == 0))
    if (parameter.numel() > 1 or bool(torch.all(parameter == 0))) and bool(
            torch.all(parameter == parameter.flatten()[0])):
        return True
    return (parameter.ndim == 2 and parameter.shape[0] == parameter.shape[1]
            and torch.equal(parameter, torch.eye(parameter.shape[0],
                                                device=parameter.device, dtype=parameter.dtype)))


def _moments(spec):
    if isinstance(spec, Uniform):
        a, b = spec.low, spec.high
        return math.sqrt((a * a + a * b + b * b) / 3), (a + b) / 2, (b - a) / math.sqrt(12)
    return math.sqrt(spec.mean ** 2 + spec.std ** 2), spec.mean, spec.std


def _draw(spec, key, shape):
    if isinstance(spec, R2Normal):
        if len(shape) != 2:
            raise ValueError("R2Normal declares a 2-D particle table")
        return spec.mean + spec.std * torch.special.ndtri(_qr.r2_points(*shape))
    rms, mean, std = _moments(spec)
    if len(shape) < 2:
        return _qr.pattern(key, shape, mean, std)
    rows, cols = shape[0], math.prod(shape[1:])
    return (_qr.semi_orthogonal(key, rows, cols) * rms * math.sqrt(max(rows, cols))).reshape(shape)


@torch.no_grad()
def deterministic_orthogonal_(module: nn.Module, *, seed: int = 0, strict: bool = True) -> nn.Module:
    """Initialize ``module``'s trainable parameters in place and return it.

    Matrices become orthogonal at the RMS of the distribution their layer
    declares (PyTorch's default scale); vectors become a deterministic pattern
    with that distribution's mean and std; particle tables become R2 points.
    ``seed`` selects the values: the same seed and architecture give the same
    weights, so give networks with matching shapes different seeds (the
    examples use G=0, D=1, E=2). R2 tables do not depend on the seed.

    Kept as-is: frozen parameters, buffers, ``KEEP`` declarations, zero
    vectors and constant or identity matrices the constructor set on purpose.
    With ``strict=True`` any other trainable parameter without a declaration
    raises before anything is written; declare it with :func:`register`, or
    pass ``strict=False`` to leave it on its constructor's init.

    Call it on a fresh network, before loading weights or building optimizers.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    owners, finalizers, holders = _plan(module)
    trainable = [(i, path, p) for i, (path, p) in enumerate(module.named_parameters())
                 if p.requires_grad and p.numel()]
    missing = [f"{path!r} ({holders[id(p)]})" for _, path, p in trainable if id(p) not in owners]
    if strict and missing:
        shown = ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "")
        raise ValueError(
            f"deterministic_orthogonal_ has no declaration for {len(missing)} trainable "
            f"parameter(s): {shown}. Declare them with particlegan.init.register(<layer class>, "
            f"{{name: Uniform/Normal/R2Normal/KEEP}}), or pass strict=False to leave them as-is.")
    changed = set()
    for index, _, parameter in trainable:
        spec = owners.get(id(parameter))
        if spec is None or spec is KEEP or (not isinstance(spec, R2Normal) and _keep_constant(parameter)):
            continue
        value = _draw(spec, _qr.key(seed, index, tuple(parameter.shape)), tuple(parameter.shape))
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        changed.add(id(parameter))
    for child, finalize in finalizers:
        if any(id(p) in changed for p in child.parameters()):
            for step in finalize:
                step(child)
    return module


# Built-in declarations: PyTorch's own constructor distributions.
def _fan_in_uniform(module):
    fan_in = math.prod(module.weight.shape[1:])
    bound = 1 / math.sqrt(fan_in) if fan_in else 0.
    return {"weight": Uniform(-bound, bound), "bias": Uniform(-bound, bound)}


for _cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
             nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d):
    register(_cls, _fan_in_uniform)


def _zero_padding_row(module):
    if module.padding_idx is not None:
        module.weight[module.padding_idx].zero_()


register(nn.Embedding, {"weight": Normal(0., 1.)}, finalize=_zero_padding_row)


def _attention(module):
    result = {"in_proj_bias": KEEP}
    for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
        weight = getattr(module, name)
        if weight is not None:
            bound = math.sqrt(6 / sum(weight.shape))
            result[name] = Uniform(-bound, bound)
    for name in ("bias_k", "bias_v"):
        weight = getattr(module, name)
        if weight is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
            result[name] = Normal(0., math.sqrt(2 / (fan_in + fan_out)))
    return result


register(nn.MultiheadAttention, _attention)
for _cls in (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm,
             nn.modules.instancenorm._InstanceNorm, nn.RMSNorm):
    register(_cls, {"weight": KEEP, "bias": KEEP})
register(nn.PReLU, {"weight": KEEP})


def _register_particlegan():
    from .diffusion import DrawSource
    from .discriminators import BatchDistanceDiscriminator
    from .particle_prior import MoGParticlePrior, ParticlePrior, calibrate_mog_sigma

    def zero_batch_readout(module):
        count = module.scales.numel()
        if module.head.in_features != module.layers[-1].out_features + count:
            raise ValueError("unsupported batch-distance feature layout")
        module.head.weight[:, -count:].zero_()

    def recalibrate(prior):
        # Calibrated spacing (d0 > 0, even at sigma_rel=0) follows the new means;
        # an explicit sigma is kept.
        if prior.sigma_rel > 0 or prior.d0 > 0:
            sigma, d0 = calibrate_mog_sigma(prior.means(), prior.sigma_rel)
            prior.set_sigma(sigma)
            prior.d0.copy_(d0)

    register(ParticlePrior, lambda prior: {"z": R2Normal(0., prior.init_std)})
    register(MoGParticlePrior, finalize=recalibrate)
    register(DrawSource, {"table": R2Normal(0., 1.)})
    register(BatchDistanceDiscriminator, finalize=zero_batch_readout)


_register_particlegan()
