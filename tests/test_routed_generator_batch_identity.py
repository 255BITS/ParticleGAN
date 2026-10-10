"""Publication portability without a new scientific execution or Git commit."""
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "examples" / "routed_generator_batch.py"


def test_identity_accepts_new_checkout_and_prose_but_rejects_source_drift(tmp_path, monkeypatch):
    api = runpy.run_path(str(DRIVER))
    protocol = json.loads(DRIVER.with_name("routed_generator_batch_protocol.json").read_text())
    # Current objectives have changed; they cannot enter the historical cohort.
    assert hashlib.sha256((ROOT / "particlegan/gan_loss.py").read_bytes()).hexdigest() != protocol["source_hashes"]["particlegan/gan_loss.py"]
    with pytest.raises(RuntimeError, match="source identity differs"):
        api["execution_identity"](protocol)
    # This real publication commit contains every declared frozen source blob.
    publication = "e71265fd7b77ae2ab70ccbf8b7f0049c3d539b84"
    historical = tmp_path / "historical-checkout"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(ROOT), str(historical)], check=True)
    subprocess.run(["git", "update-ref", "--no-deref", "HEAD", publication], cwd=historical, check=True)
    subprocess.run(["git", "checkout", publication, "--", *protocol["source_hashes"]], cwd=historical, check=True)
    actual = api["execution_identity"](protocol, root=historical)
    assert actual["verified_source_files"] == len(protocol["source_hashes"])
    assert actual["reference_package_git_sha"] == api["BASE_SHA"]
    assert actual["checkout_git_sha"] == publication

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
