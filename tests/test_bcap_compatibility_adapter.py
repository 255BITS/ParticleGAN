"""Check the transparent metadata-only adaptation without running training."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helper():
    spec = importlib.util.spec_from_file_location("compatibility_adapter_test", ROOT / "reports/forge/bcap-three-phase/verify_compatibility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_preserves_every_assertion_and_adds_only_checked_inactive_defaults():
    module = helper()
    old = (ROOT / module.INACTIVE).read_bytes()
    new, diff = module.adapter_source(old)
    additions = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removals = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert additions == ['+    "critic_step_mode": "none",', '+    "optimizer_svd_backend": "native",']
    assert not removals
    assert new.count(b"assert ") == old.count(b"assert ")
    assert (ROOT / module.INACTIVE).read_bytes() == old


def test_adapter_refuses_modified_original_assertions():
    module = helper()
    old = (ROOT / module.INACTIVE).read_bytes().replace(b"assert torch.equal(left, right)", b"assert True")
    with pytest.raises(ValueError, match="immutable original comparator changed"):
        module.adapter_source(old)
