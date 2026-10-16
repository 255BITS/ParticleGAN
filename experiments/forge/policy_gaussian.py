"""Scoped policy variants of the scalar Gaussian smoke and stability questions.

``gaussian_tasks.py`` is an evaluator source pinned by the ordinary Gaussian
task declarations, so its bytes stay unchanged. Its entry points validate the
original live/MoG constants. A registered policy variant differs from its
parent only by its declared transformation (state-selected scoring and an
explicit particle cloud), so this module executes the unchanged
``gaussian_tasks`` code objects with exactly one substitution: the entry
validator first checks the variant's transformation identity and then every
original constant on its unchanged parent declaration.
"""
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
build = _SCOPE["build"]
grade = _SCOPE["grade"]
run_gaussian = _SCOPE["run_gaussian"]
