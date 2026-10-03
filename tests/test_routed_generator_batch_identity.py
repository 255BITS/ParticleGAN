"""Publication portability without a new scientific execution or Git commit."""
import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "examples" / "routed_generator_batch.py"


def test_archived_protocol_keeps_its_original_source_boundary():
    api = runpy.run_path(str(DRIVER))
    protocol = json.loads(DRIVER.with_name("routed_generator_batch_protocol.json").read_text())
    # Later development can legitimately change scientific package bytes. The
    # archived declaration must still reject them, rather than being repinned.
    matches = all((ROOT / path).is_file()
                  and hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
                  for path, expected in protocol["source_hashes"].items())
    assert api["source_readback"](protocol) is matches
    if matches:
        actual = api["execution_identity"](protocol)
        assert actual["verified_source_files"] == len(protocol["source_hashes"])
        assert actual["reference_package_git_sha"] == api["BASE_SHA"]
    else:
        with pytest.raises(RuntimeError, match="source identity differs"):
            api["execution_identity"](protocol)


def test_identity_accepts_new_checkout_and_prose_but_rejects_source_drift(tmp_path, monkeypatch):
    api = runpy.run_path(str(DRIVER))

    # Only metadata changes when publication adds a commit or unbound report.
    scientific = tmp_path / "particlegan" / "policy.py"
    scientific.parent.mkdir()
    scientific.write_bytes(b"frozen scientific package bytes\n")
    fixture = {"package_git_sha": api["BASE_SHA"], "source_hashes": {
        "particlegan/policy.py": hashlib.sha256(scientific.read_bytes()).hexdigest()}}
    new_checkout = "f" * 40
    monkeypatch.setattr(api["execution_identity"].__globals__["subprocess"],
                        "check_output", lambda *args, **kwargs: new_checkout + "\n")
    first = api["execution_identity"](fixture, root=tmp_path)
    (tmp_path / "new_results.md").write_text("Publication report only.\n")
    assert api["execution_identity"](fixture, root=tmp_path) == first
    assert first["checkout_git_sha"] == new_checkout != api["BASE_SHA"]
    scientific.write_bytes(scientific.read_bytes() + b"scientific drift\n")
    assert not api["source_readback"](fixture, root=tmp_path)
    with pytest.raises(RuntimeError, match="source identity differs"):
        api["execution_identity"](fixture, root=tmp_path)
    scientific.unlink()
    assert not api["source_readback"](fixture, root=tmp_path)
