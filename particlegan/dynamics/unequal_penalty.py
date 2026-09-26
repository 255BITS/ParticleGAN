"""Let the probe's frozen penalty accept the current ``ema_critic`` argument.

``vector_unequal_mass`` goes through ``CriticPenalty``, which calls
``GradientPenalty.penalty(..., ema_critic=...)``. The probe then replaces that
method with ``mechanism.scaled_penalty``, whose signature does not take
``ema_critic`` and whose body reads ``self.arm``. The current penalty has no
``arm``, so the call raises ``TypeError`` before any update.

This hook runs after that assignment. On the current penalty (no ``arm``) it
forwards to the method ``mechanism`` saved as ``_original_penalty``, including
``ema_critic``. The frozen body is left for an arm-based instance. Ring, hold,
and stay call the legacy penalty and do not enter this function.
"""
from __future__ import annotations

import sys

_FINDER = None


def uninstall() -> None:
    global _FINDER
    if _FINDER is None:
        return
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
    _FINDER = None


def install() -> None:
    """Install the import hook. Safe before ``mechanism`` is imported."""
    global _FINDER
    if _FINDER is not None:
        return
    _FINDER = _Finder()
    sys.meta_path.insert(0, _FINDER)
    module = sys.modules.get("mechanism")
    if module is not None:
        _patch(module)


def _patch(module) -> None:
    frozen = getattr(module, "scaled_penalty", None)
    original = getattr(module, "_original_penalty", None)
    if frozen is None or original is None or getattr(frozen, "_unequal_fix", False):
        return

    def scaled_penalty(self, D, x_real, x_fake, step=1, collect_stats=True, *, ema_critic=None, generator=None):
        if not hasattr(self, "arm"):
            return original(self, D, x_real, x_fake, step, collect_stats, ema_critic=ema_critic)
        return frozen(self, D, x_real, x_fake, step, generator, collect_stats)

    scaled_penalty._unequal_fix = True
    module.scaled_penalty = scaled_penalty
    regularizers = sys.modules.get("particlegan.grad_regularizers")
    if regularizers is not None and regularizers.GradientPenalty.penalty is frozen:
        regularizers.GradientPenalty.penalty = scaled_penalty


class _Loader:
    def __init__(self, inner, name):
        self.inner = inner
        self.name = name

    def create_module(self, spec):
        create = getattr(self.inner, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        if self.name == "mechanism":
            _patch(module)


class _Finder:
    def find_spec(self, fullname, path, target=None):
        if fullname != "mechanism" or fullname in sys.modules:
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, fullname)
        return spec
