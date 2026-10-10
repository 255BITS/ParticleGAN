"""Saved-byte archive integrity; fixtures contain no neural state or execution."""
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest


PATH = Path(__file__).resolve().parents[1] / "reports/forge/gaussian-smoke-inventory/archive_final.py"
SPEC = importlib.util.spec_from_file_location("gaussian_final_archive", PATH)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive.encoded(value))


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path / "worktree"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / ".gitignore").write_text("artifacts/\n")
    source_files = {"particlegan/saved.json": hashlib.sha256(b"frozen source\n").hexdigest()}
    source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(archive, "SOURCE_DIGEST", source_digest)
    source = {"origin_commit": archive.SOURCE, "digest": source_digest}
    queue = root / "runs/forge" / archive.ROUND
    queue.mkdir(parents=True)
    for name in ("queue.lock", "coordinator.lock"):
        (queue / name).touch()
    (queue / "drain.log").write_text("original setup error\n")
    (queue / "controller.log").write_text("actual controller output\n")
    worker = queue / archive.ROUND / "attempt-1"
    worker.mkdir(parents=True)
    (worker / "execution.lock").touch()
    (worker / "run.log").write_text("original blocker log\n")
    (worker / "provenance-state.pt").write_bytes(b"saved byte fixture, never a model")
    snapshot = queue / "snapshots" / source_digest
    write(snapshot / "forge-source.json", {**source, "files": source_files})
    (snapshot / "particlegan").mkdir()
    (snapshot / "particlegan/saved.json").write_bytes(b"frozen source\n")
    campaign = {"reserved_seconds": 0, "spent_seconds": 2.5}
    state = {"campaigns": {archive.ROUND: campaign}, "submissions": {
        "request-1": {"status": "blocked", "request": {"source": source}}},
        "jobs": {"job-1": {"status": "terminal", "subscribers": ["request-1"],
            "attempts": [{"attempt_id": "attempt-1"}]}},
        "charges": [{"attempt_id": "attempt-1", "seconds": 2.5, "owner": {"campaign": archive.ROUND}}]}
    write(queue / "queue/state.json", state)
    write(queue / "launch-receipt.json", {"source_origin_commit": archive.SOURCE,
        "source_digest": source_digest, "previous_paid_seconds": 3708.623330772156})
    original = root / "reports/forge/attempts/attempt-1"
    write(original / "request.json", {"request": {"source": source}, "worker": {"directory": str(worker)}})
    write(original / "evidence.json", {"original": "blocker evidence"})
    write(original / "result.json", {"attempt_id": "attempt-1", "raw": {"attempt_status": "error"}})
    studies = [f"gaussian-smoke-inventory-fixture-{index}-v4" for index in range(12)]
    write(root / "configs/forge/rounds" / (archive.ROUND + ".json"), {
        "campaign": "configs/forge/campaigns/" + archive.ROUND + ".json",
        "studies": studies, "study_refusals": {}})
    write(root / "configs/forge/campaigns" / (archive.ROUND + ".json"), campaign)
    for study in studies:
        write(root / "configs/forge/searches" / (study + ".json"), {"id": study})
        write(root / "reports/forge/configuration-search" / (study + ".json"), {
            "source_digest": source_digest, "queue_root": str(queue), "campaign": {"id": archive.ROUND},
            "campaign_accounting": campaign, "stage": "enqueued",
            "trials": [{"request_id": "request-1", "status": "BLOCKED", "attempt_ids": ["attempt-1"]}]})
    return root, queue, root / "artifacts/final.tar.gz", root / "reports/final-archive.json"


def test_byte_exact_deterministic_final_archive_keeps_blockers_and_both_logs(cohort):
    root, queue, output, receipt_path = cohort
    receipt = archive.create(root, output, receipt_path)
    inventory = archive.verify(output)
    assert receipt["qualification_input"] is False and receipt["finalized"] is True
    assert receipt["attempts"] == ["attempt-1"] and receipt["attempt_count"] == 1
    assert receipt["paid_seconds"] == 2.5 and receipt["reserved_seconds"] == 0
    assert len(receipt["search_report_hashes"]) == 12
    assert receipt["members_digest"] == hashlib.sha256(archive.encoded(inventory)).hexdigest()
    for name, identity in inventory["files"].items():
        assert archive.digest(root / name) == identity["sha256"]
    assert all((queue / name).relative_to(root).as_posix() in inventory["files"] for name in ("drain.log", "controller.log"))
    assert any(name.endswith("provenance-state.pt") for name in inventory["files"])
    other = root / "artifacts/identical.tar.gz"
    second = archive.create(root, other, root / "reports/second-archive.json")
    assert output.read_bytes() == other.read_bytes() and receipt["sha256"] == second["sha256"]
    with pytest.raises(ValueError, match="immutable"):
        archive.create(root, output, receipt_path)


@pytest.mark.parametrize("change,reason", [
    (lambda s: s["jobs"]["job-1"].update(status="running"), "active workers"),
    (lambda s: s["submissions"]["request-1"].update(status="queued"), "active campaign"),
    (lambda s: s["campaigns"][archive.ROUND].update(reserved_seconds=1), "reserves"),
    (lambda s: s["charges"][0].update(seconds=3), "accounting"),
    (lambda s: s["charges"].append(s["charges"][0].copy()), "accounting"),
])
def test_refuses_active_or_unfinalized_accounting(cohort, change, reason):
    root, queue, output, receipt = cohort
    state = archive.read(queue / "queue/state.json")
    change(state)
    write(queue / "queue/state.json", state)
    with pytest.raises(ValueError, match=reason):
        archive.create(root, output, receipt)
    assert not output.exists()


def test_refuses_live_coordinator_and_nonignored_bulk_destination(cohort):
    root, queue, output, receipt = cohort
    with (queue / "coordinator.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active coordinator"):
            archive.create(root, output, receipt)
    with pytest.raises(ValueError, match="ignored worktree artifacts"):
        archive.create(root, root / "reports/raw.tar.gz", receipt)


@pytest.mark.parametrize("mutation,reason", [
    ("stale_search", "refresh final"), ("source", "frozen source file changed"),
    ("symlink", "symlink"), ("missing_original", "No such file"),
])
def test_refuses_stale_missing_or_changed_originals(cohort, mutation, reason):
    root, queue, output, receipt = cohort
    if mutation == "stale_search":
        path = next((root / "reports/forge/configuration-search").glob("*.json"))
        value = archive.read(path); value["campaign_accounting"]["spent_seconds"] = 0
        write(path, value)
    elif mutation == "source":
        (queue / "snapshots" / archive.SOURCE_DIGEST / "particlegan/saved.json").write_bytes(b"changed")
    elif mutation == "symlink":
        (queue / "aliased.log").symlink_to(queue / "controller.log")
    else:
        (root / "reports/forge/attempts/attempt-1/evidence.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=reason):
        archive.create(root, output, receipt)
    assert not output.exists()


def test_saved_archive_rejects_corruption_and_duplicate_members(cohort):
    root, queue, output, receipt = cohort
    archive.create(root, output, receipt)
    with tarfile.open(output, "r:gz") as original:
        entries = [(member, original.extractfile(member).read()) for member in original.getmembers()]
    for mutation in ("bytes", "duplicate"):
        broken = root / "artifacts" / (mutation + ".tar.gz")
        with tarfile.open(broken, "w:gz") as target:
            for index, (member, payload) in enumerate(entries):
                if mutation == "bytes" and index == 0:
                    payload = b"X" * len(payload)
                target.addfile(member, io.BytesIO(payload))
            if mutation == "duplicate":
                target.addfile(entries[0][0], io.BytesIO(entries[0][1]))
        with pytest.raises(ValueError, match="bytes differ|duplicate archive"):
            archive.verify(broken)
