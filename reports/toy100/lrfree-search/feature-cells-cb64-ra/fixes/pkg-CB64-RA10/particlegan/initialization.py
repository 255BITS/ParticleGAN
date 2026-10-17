"""Explicit deterministic initialization, without process-wide PyTorch hooks."""
from __future__ import annotations

import math

import torch
from torch import nn

from . import qr_bz_pq_init as _qr

_MARK = "_particlegan_initialized"
_external_init = None  # Explicit research-registry hooks take precedence.
_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d,
          nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def _declarations(module):
    """Standard PyTorch scales; custom parameters are deliberately left alone."""
    if isinstance(module, (nn.Linear, *_CONVS)):
        weight = module.weight
        if isinstance(weight, nn.parameter.UninitializedParameter):
            raise ValueError("materialize lazy layers before initialize_")
        fan_in = math.prod(weight.shape[1:])
        bound = 1 / math.sqrt(fan_in) if fan_in else 0.
        return {"weight": ("uniform", -bound, bound),
                "bias": ("uniform", -bound, bound)}
    if isinstance(module, nn.Embedding):
        return {"weight": ("normal", 0., 1.)}
    if isinstance(module, nn.MultiheadAttention):
        result = {}
        for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
            weight = getattr(module, name, None)
            if weight is not None:
                bound = math.sqrt(6 / sum(weight.shape))
                result[name] = ("uniform", -bound, bound)
        # Packed QKV retains the host's packed-matrix scale and layout.
        for name in ("bias_k", "bias_v"):
            weight = getattr(module, name, None)
            if weight is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
                result[name] = ("normal", 0., math.sqrt(2 / (fan_in + fan_out)))
        return result
    return {}


def _keep_constant(parameter):
    if parameter.ndim < 2:
        return bool(torch.all(parameter == 0))
    if (parameter.numel() > 1 or bool(torch.all(parameter == 0))) and bool(
            torch.all(parameter == parameter.flatten()[0])):
        return True
    return (parameter.ndim == 2 and parameter.shape[0] == parameter.shape[1]
            and torch.equal(parameter, torch.eye(parameter.shape[0],
                                                device=parameter.device, dtype=parameter.dtype)))


@torch.no_grad()
def _initialize(module, *, key=0, only_new=False):
    if not isinstance(module, nn.Module):
        raise TypeError("initialize_ expects an nn.Module")
    if type(key) is not int or key < 0:
        raise ValueError("key must be a nonnegative integer")
    owners = {}
    for child in module.modules():
        declarations = _declarations(child)
        for name, parameter in child.named_parameters(recurse=False):
            if name in declarations:
                owners.setdefault(id(parameter), (child, name, declarations[name]))
    changed = set()
    for index, parameter in enumerate(module.parameters()):
        if (not parameter.requires_grad or not parameter.numel()
                or id(parameter) not in owners
                or (only_new and getattr(parameter, _MARK, False))):
            continue
        child, name, tag = owners[id(parameter)]
        setattr(parameter, _MARK, True)
        if _keep_constant(parameter):
            continue
        tensor_key = _qr._key(key, index, tuple(parameter.shape))
        if name == "bias":
            value = _qr.pattern_bias(tensor_key, parameter.shape, tag)
        else:
            rows, cols = parameter.shape[0], math.prod(parameter.shape[1:])
            value = (_qr.semi_orthogonal(tensor_key, rows, cols, variant="qr")
                     * _qr._declared_rms_mean_std(tag)[0] * math.sqrt(max(rows, cols)))
            value = value.reshape(parameter.shape)
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        if isinstance(child, nn.Embedding) and child.padding_idx is not None:
            parameter[child.padding_idx].zero_()
        changed.add(id(parameter))
    from .discriminators import BatchDistanceDiscriminator
    for child in module.modules():
        if isinstance(child, BatchDistanceDiscriminator) and id(child.head.weight) in changed:
            count = child.scales.numel()
            if child.head.in_features != child.layers[-1].out_features + count:
                raise ValueError("unsupported batch-distance feature layout")
            child.head.weight[:, -count:].zero_()
    return bool(changed)


def initialize_(module: nn.Module, *, key: int = 0) -> nn.Module:
    """Initialize a fresh network in place and return it (``batch_feature_zero``).

    Linear/convolution/embedding/attention weights use deterministic float64
    CPU QR at standard PyTorch RMS scales; random linear/conv biases use a
    deterministic pattern. BatchDistanceDiscriminator's batch readout starts
    at zero. Frozen parameters, constant/identity matrices, zero biases,
    normalization parameters, buffers, and unknown custom parameters are kept.
    No RNG state is consumed and no optimizer or global hooks are involved.

    ``key`` distinguishes networks without using a random seed. The recipe
    uses keys 0/1/2 for generator/critic/encoder. Initialize before constructing
    optimizers or loading trained weights. Recipe factories recognize already
    initialized parameters and do not reset them. Custom scales/layouts should
    be initialized by the caller with ``initialization=None`` on the recipe.
    """
    _initialize(module, key=key)
    return module


@torch.no_grad()
def _initialize_prior(prior, std):
    # Recipe-created priors only. A caller-supplied prior is never redrawn.
    if not isinstance(prior.z, nn.Parameter) or not prior.z.requires_grad:
        return prior
    value = _qr.qmc_draw(0, prior.z.shape, ("normal", 0., std), rows_as_points=True)
    prior.z.copy_(value.to(device=prior.z.device, dtype=prior.z.dtype))
    return prior
