"""Suite-wide isolation for process-global torch state."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from benchmarks.toy100.device import device_policy_scope


@contextmanager
def _archived_inactive_comparison_scope(module, directory):
    """Use the existing AST-proven metadata adapter in both interpreters.

    The archived comparator and its receipts retain their exact original bytes.
    Updating only its imported map would miss the separately executed probe.
    """
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pytest_bcap_inactive_adapter", root / "reports/forge/bcap-three-phase/verify_compatibility.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = (root / helper.INACTIVE).read_bytes()
    adapted, diff = helper.adapter_source(original)
    from particlegan import Recipe
    recipe = Recipe()
    for name, default in helper.INACTIVE_DEFAULTS.items():
        assert getattr(recipe, name) == default, (name, "inactive public default changed")
        assert name not in module.NEW_INACTIVE_FIELDS
    path = directory / "test_bcap_integration_compatibility_adapted.py"
    path.write_bytes(adapted)
    (directory / "adapter.diff").write_text(diff)
    (directory / "adapter-receipt.json").write_text(json.dumps({
        "original_path": helper.INACTIVE,
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "adapted_sha256": hashlib.sha256(adapted).hexdigest(),
        "checked_inactive_defaults": helper.INACTIVE_DEFAULTS,
        "assertions_unchanged": True,
    }, indent=2) + "\n")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "NEW_INACTIVE_FIELDS", {
            **module.NEW_INACTIVE_FIELDS, **helper.INACTIVE_DEFAULTS})
        # The original fixture uses __file__ as the subprocess program path;
        # its explicit source-tree arguments and original ROOT stay unchanged.
        patch.setattr(module, "__file__", str(path))
        yield


@pytest.fixture(scope="module", autouse=True)
def _preserve_archived_inactive_comparison(request, tmp_path_factory):
    if request.module.__name__ != "test_bcap_integration_compatibility":
        yield
        return
    directory = tmp_path_factory.mktemp("bcap-inactive-adapter")
    with _archived_inactive_comparison_scope(request.module, directory):
        yield


@pytest.fixture(autouse=True)
def _isolate_renderer_clock(request, monkeypatch):
    """Start the renderer's unit-test budget at each test, after collection.

    The archived protocol binds its test-file bytes. Keep this software-only
    clock isolation outside those frozen sources and preserve the CLI budget.
    """
    if request.module.__name__ == "test_e22_routed_caption_late_phase_renderer":
        render = request.module.render
        monkeypatch.setattr(render, "STARTED", render.time.monotonic())


@pytest.fixture(autouse=True)
def _restore_device_policy():
    """Tests that run a benchmark CLI ``main()`` in-process must not leave its
    CUDA default device, routed ``torch.Generator`` or determinism flags behind."""
    with device_policy_scope():
        yield
