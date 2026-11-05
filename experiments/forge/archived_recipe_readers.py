"""Software compatibility for archived analyzers reading saved recipe fields.

The original diagnostic and its CLI retain their frozen source contract. This
software compatibility entrypoint adapts only that diagnostic's saved-recipe
restoration call; it neither selects a training preset nor changes public API
objects. Results from a current runtime do not replace the archived diagnosis.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys

from particlegan import Recipe


_ROOT = Path(__file__).resolve().parents[2]
_OPTIMISTIC_SOURCE = "reports/toy100/optimistic_regrade.py"
_OPTIMISTIC_SHA256 = "bf46e609482e10cd2808868a5ed7a09d61f7621823b40986b5035c3037ca20e6"


def _restore_recipe(**saved_fields):
    return Recipe.from_dict(saved_fields)


def bind_optimizer_groups(record, receipt, protocol):
    """Run the immutable scratch rate checker with explicit recipe restoration.

    Load a fresh private module namespace and change only its saved-recipe
    factory. Archived callers, source receipts and the public class stay intact.
    This software check confers no experiment or qualification credit.
    """
    path = _ROOT / _OPTIMISTIC_SOURCE
    if hashlib.sha256(path.read_bytes()).hexdigest() != _OPTIMISTIC_SHA256:
        raise ValueError("archived optimistic regrader source differs")
    spec = importlib.util.spec_from_file_location("_archived_optimistic_regrader", path)
    reader = importlib.util.module_from_spec(spec)
    search_path = sys.path[:]
    try:
        spec.loader.exec_module(reader)
    finally:
        # The archived script prepends its repository root for direct execution.
        sys.path[:] = search_path
    reader.Recipe = _restore_recipe
    return reader._bind_optimizer_groups(record, receipt, protocol)


class _SavedRecipeAPI:
    def __init__(self, api):
        self._api = api

    def __getattr__(self, name):
        return getattr(self._api, name)

    def Recipe(self, **saved_fields):
        """The archived analyzer passes checkpoint fields at this boundary."""
        return self._api.Recipe.from_dict(saved_fields)


def analyze_saved_state(diagnostic, api, named_streams, state, receipt, *,
                        n=4096, context_count=8):
    """Use explicit restoration while preserving the archived numerical code.

    ``diagnostic`` is the original, independently source-checked module. The
    caller owns artifact and runtime admission; this helper adds no such claim.
    No public constructor or module global is patched, even temporarily.
    """
    if not callable(getattr(api.Recipe, "from_dict", None)):
        raise TypeError("current saved-state analysis requires Recipe.from_dict")
    return diagnostic.analyze(_SavedRecipeAPI(api), named_streams, state, receipt,
                              n=n, context_count=context_count)
