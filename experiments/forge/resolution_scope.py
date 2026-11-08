"""Context-local binding of one explicit read-only resolution cache.

``planning.resolution_cache(root)`` creates a ``planning.ResolutionCache`` and
binds it here for the duration of a ``with`` block. The few catalogue readers
that memoize while a cache is bound (``views.load_tasks``,
``tier1_policy.validate`` and ``decision_contracts._bindings``) ask
``active(root)`` for it explicitly. No module rebinds another module's globals.

The binding lives in a ``contextvars.ContextVar``: each thread (and each
asyncio task) sees only the cache bound in its own context, so concurrent
resolutions never observe one another's memo tables. A cache object shared on
purpose (for example through ``contextvars.copy_context``) guards its memo
tables with its own lock.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_ACTIVE: ContextVar = ContextVar("forge_resolution_cache", default=None)


def active(root=None):
    """Return the cache bound in this context, if it serves ``root``.

    ``root=None`` returns any bound cache; callers that know their checkout
    pass it so a cache for one repository never answers for another.
    """
    cache = _ACTIVE.get()
    if cache is None or (root is not None and Path(root).resolve() != cache.root):
        return None
    return cache


@contextmanager
def bind(cache):
    """Bind ``cache`` for this context only; the previous binding is restored on exit."""
    token = _ACTIVE.set(cache)
    try:
        yield cache
    finally:
        _ACTIVE.reset(token)
