"""Scoped policy variants of the scalar Gaussian smoke and stability questions.

``gaussian_tasks.py`` is an evaluator source pinned by the ordinary Gaussian
task declarations, so its bytes stay unchanged. Its entry points validate the
original live/MoG constants. A registered policy variant differs from its
parent only by its declared transformation (state-selected scoring and an
explicit particle cloud), so this module executes the unchanged
``gaussian_tasks`` code objects with two substitutions: the entry validator
first checks the variant's transformation identity and then every original
constant on its unchanged parent declaration; the shared constructor binds the
cohort's declared measurement law to every trainer it builds. The first
trainer of a run is the measured model. A later one (the stability frozen
control) records its draws in the same ledger, so both arms are measured under
one law (clean, or as served with the learned output noise).
"""
import contextvars
import types

from . import gaussian_tasks as original
from .tier1_policy import is_policy_task, validate as validate_policy


def validate_task(task):
    if not is_policy_task(task):
        raise ValueError("policy Gaussian execution requires a registered scoped variant")
    original.validate_task(validate_policy(task))


def _rebind(names=("build", "grade", "run_gaussian")):
    scope = dict(vars(original))
    scope["validate_task"] = validate_task
    for name in names:
        function = getattr(original, name)
        bound = types.FunctionType(function.__code__, scope, function.__name__,
                                   function.__defaults__, function.__closure__)
        bound.__kwdefaults__ = function.__kwdefaults__
        bound.__doc__ = function.__doc__
        scope[name] = bound
    return scope


_SCOPE = _rebind()
_ORIGINAL_BUILD = _SCOPE["build"]
_RUN_LAWS = contextvars.ContextVar("policy_gaussian_run_laws", default=None)


def build(request, task, device, *, max_steps=1000):
    """The unchanged constructor plus the cohort's declared measurement law."""
    from .adapters import bind_measurement_law
    context, trainer, spec = _ORIGINAL_BUILD(request, task, device, max_steps=max_steps)
    laws = _RUN_LAWS.get()
    primary = laws[0] if laws else None
    law = bind_measurement_law(trainer, task, draws=None if primary is None else primary.draws,
                               role="measured" if primary is None else "frozen_control")
    if laws is not None:
        laws.append(law)
    return context, trainer, spec


_SCOPE["build"] = build
grade = _SCOPE["grade"]
_ORIGINAL_RUN = _SCOPE["run_gaussian"]


def run_gaussian(request, task, output, device, *, prerequisites=None, diagnostic=False):
    """Run the unchanged evaluator code; its trainers share one measurement ledger."""
    token = _RUN_LAWS.set([])
    try:
        return _ORIGINAL_RUN(request, task, output, device, prerequisites=prerequisites, diagnostic=diagnostic)
    finally:
        _RUN_LAWS.reset(token)
