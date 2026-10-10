"""The measured QR/pattern/R2 initializer with neutral batch-distance readout.

Call ``install`` before constructing fresh models and their Adam optimizers.
The ordinary random draws still occur; replacement values use the original
optimizer-index/parameter-index/shape keys. Parameters are replaced only at
Adam construction, never during training. This is opt-in and process-wide.

The numerical primitives are shared with the original orthogonal-search port;
its zero-bias variant and global installation state are not modified here.
"""
from __future__ import annotations

import gc
import json
import math
import warnings
import weakref

import torch
from torch import nn

from . import qr_bz_pq_init as _qr
from particlegan.discriminators import BatchDistanceDiscriminator
from particlegan.particle_prior import ParticlePrior

VARIANT = "batch_feature_zero"
_TAG = "_batch_feature_init_declaration"
_originals: dict = {}
_wrappers: dict = {}
_optimizer_index = 0
_zeroed_heads: dict[int, weakref.ReferenceType] = {}


def install(name: str = VARIANT) -> str:
    """Activate the initializer and restart construction order for fresh models.

    Reinstalling in the same process restarts the keys without stacking hooks.
    Use one initialization family per process. Existing trained/pretrained
    models must not be reinitialized with this construction hook.
    """
    if name != VARIANT:
        raise ValueError(f"init must be {VARIANT!r}")
    global _optimizer_index
    _optimizer_index = 0
    _zeroed_heads.clear()
    if not _originals:
        for operation in ("uniform_", "normal_"):
            original = getattr(torch.Tensor, operation)
            _originals[operation] = original

            def draw(self, *args, _original=original, _operation=operation, **kwargs):
                value = _original(self, *args, **kwargs)
                if isinstance(self, nn.Parameter):
                    if _operation == "uniform_":
                        a = args[0] if args else kwargs.get("from", 0.)
                        b = args[1] if len(args) > 1 else kwargs.get("to", 1.)
                        declaration = ("uniform", float(a), float(b))
                    else:
                        a = args[0] if args else kwargs.get("mean", 0.)
                        b = args[1] if len(args) > 1 else kwargs.get("std", 1.)
                        declaration = ("normal", float(a), float(b))
                    setattr(self, _TAG, (declaration, self._version))
                return value

            _wrappers[operation] = draw
            setattr(torch.Tensor, operation, draw)
        original_adam = torch.optim.Adam.__init__
        _originals["adam"] = original_adam

        def adam_init(opt, params, *args, **kwargs):
            original_adam(opt, params, *args, **kwargs)
            _apply(opt)

        _wrappers["adam"] = adam_init
        torch.optim.Adam.__init__ = adam_init
    print(json.dumps(dict(event="det_init", name=name,
                          weights="qr_declared_rms", bias="pattern", prior="r2",
                          batch_feature_readout="zero")), flush=True)
    return name


def uninstall() -> None:
    """Restore the hooks captured by install; leave all parameter values intact."""
    if not _originals:
        return
    targets = [(torch.Tensor, "uniform_", "uniform_"),
               (torch.Tensor, "normal_", "normal_"),
               (torch.optim.Adam, "__init__", "adam")]
    if any(getattr(owner, attr) is not _wrappers[key] for owner, attr, key in targets):
        raise RuntimeError("another initializer changed the hooks; use one family per process")
    for owner, attr, key in targets:
        setattr(owner, attr, _originals[key])
    _originals.clear()
    _wrappers.clear()
    _zeroed_heads.clear()


@torch.no_grad()
def _apply(opt) -> None:
    global _optimizer_index
    oi = _optimizer_index
    _optimizer_index += 1
    params = [p for group in opt.param_groups for p in group["params"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        owners = _qr._owners()
    has_prior = any(isinstance(owners.get(id(p), (None,))[0], ParticlePrior) for p in params)
    for pi, parameter in enumerate(params):
        declaration = getattr(parameter, _TAG, None)
        # Keep constants and explicit post-draw host assignments, including
        # identity generators. Empty tensors need no initialization.
        if (not parameter.numel() or declaration is None or
                declaration[1] != parameter._version):
            continue
        tag = declaration[0]
        owner, role = owners.get(id(parameter), (None, None))
        key = _qr._key(oi, pi, tuple(parameter.shape))
        if isinstance(owner, ParticlePrior) and role == "z":
            value = _qr.qmc_draw(key, parameter.shape, tag, rows_as_points=True)
        elif role == "bias" and parameter.ndim == 1:
            value = _qr.pattern_bias(key, parameter.shape, tag)
        else:
            rows = parameter.shape[0] if parameter.ndim >= 2 else 1
            cols = math.prod(parameter.shape[1:]) if parameter.ndim >= 2 else parameter.numel()
            q = _qr.semi_orthogonal(key, rows, cols, variant="qr")
            rms = _qr._declared_rms_mean_std(tag)[0]
            value = (q * rms * math.sqrt(max(rows, cols))).reshape(parameter.shape)
        parameter.copy_(value.to(dtype=parameter.dtype, device=parameter.device))

    if has_prior:
        return
    parameter_ids = {id(p) for p in params}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for module in gc.get_objects():
            if not isinstance(module, BatchDistanceDiscriminator):
                continue
            head = module.head
            if id(head.weight) not in parameter_ids:
                continue
            previous = _zeroed_heads.get(id(head))
            if previous is not None and previous() is head:
                continue
            features = module.scales.numel()
            if head.in_features != module.layers[-1].out_features + features:
                raise ValueError("unsupported batch-distance feature layout")
            head.weight[:, -features:].zero_()
            _zeroed_heads[id(head)] = weakref.ref(head)
