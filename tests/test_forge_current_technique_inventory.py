"""One current table can be rebuilt from committed, source-bound evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("forge_current_inventory", ROOT / "reports/forge/regenerate_technique_inventory.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def _register(root, manifest, name, source, *, finished=None, passed=0):
    attempt = "attempt-" + source if finished else None
    row = {"candidate_id": name, "candidate_revision": "revision-" + source,
           "cohort": "cohort-" + source, "technique": name,
           "bindings": {"source_digest": source * 64, "prior": {"kind": "mog", "sigma": .025}},
           "runtime_cohort": {"execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "gpu-" + source}}},
           "tiers": {tier: {"passed": passed if tier == "1" else 0, "total": len(tasks)}
                     for tier, tasks in manifest["tier_requirements"].items()},
           "qualified_tier": 1 if passed == 3 else 0, "attempt_ids": [attempt] if attempt else [],
           "tasks": [{"task_id": "two_pole", "status": "FAIL" if attempt else "UNKNOWN"}],
           "cost": {"wall_seconds": 5.25 if attempt else None}}
    report = {key: deepcopy(manifest[key]) for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements")}
    report.update(schema_version=1, publication_scope="frozen_source", execution_backend="cuda",
                  frozen_source={"commit": "commit-" + source}, rows=[row], provenance={"qualified_receipts": {}})
    if attempt:
        proof = {"canonical_result_hash": "result-" + source, "source_digest": source * 64,
                 "original_file_sha256": {name: name + "-hash" for name in ("request", "evidence", "result")}}
        summary = {"candidate_id": name, "candidate_revision": row["candidate_revision"],
                   "certificate_validated": True, "qualification_input": False, "qualification_reuse": False,
                   "provenance": {"canonical_result_hash": proof["canonical_result_hash"], "source_digest": source * 64,
                                  "original_files": {name: {"sha256": digest} for name, digest in proof["original_file_sha256"].items()}}}
        atomic_json(root / f"reports/forge/technique-receipts/{attempt}.json", summary)
        report["provenance"]["qualified_receipts"][attempt] = proof
    report["provenance"]["input_digest"] = stable_hash(report)
    path = root / f"reports/forge/technique-evidence/{source}.json"
    atomic_json(path, report)
    entry = {"snapshot": path.relative_to(root).as_posix(), "json_sha256": file_hash(path),
             "source_commit": "commit-" + source, "candidates": {name: finished}}
    manifest["cohorts"].append(entry)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    atomic_json(root / f"configs/forge/ideas/{name}.json", {"id": name})
    return entry, report


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    policy = {"id": "discriminator_stability", "revision": 2}
    atomic_json(tmp_path / "configs/forge/views/discriminator_stability.json", policy)
    manifest = {"schema_version": 1, "view": policy["id"], "view_revision": 2,
                "policy_fingerprint": stable_hash(policy), "tier_requirements": {
                    "1": ["two_pole", "token", "ae"], "2": [f"q-{i}" for i in range(19)], "3": ["hold", "extension"]},
                "cohorts": [], "retired_publications": {}}
    _register(tmp_path, manifest, "bcap", "a", finished="2026-10-01T01:00:00+00:00", passed=3)
    _register(tmp_path, manifest, "atlas", "b")
    def unexpected(*args, **kwargs):
        pytest.fail("cached publication must not resolve live receipts or train")
    monkeypatch.setattr(publication, "write_report", unexpected)
    monkeypatch.setattr(publication, "regenerate", unexpected)
    return tmp_path, manifest


def _outputs(root):
    return {path: path.read_bytes() for path in (root / "reports/forge").glob("technique-inventory.*")}


def test_current_publication_needs_no_originals_and_is_idempotent(evidence):
    root, _ = evidence
    result = publication.publish_current(root)
    report = read_json(result["json"])
    assert result["report"] == str(root / "reports/forge/technique-inventory.md")
    assert len(report["rows"]) == 2 and report["publication_scope"] == "current_technique_inventory"
    assert report["rows"][0]["tiers"]["1"] == {"passed": 3, "total": 3}
    assert report["qualification_input"] is False and report["qualification_reuse"] is False
    assert not (root / "reports/forge/attempts").exists()
    assert list((root / "reports/forge").rglob("*.md")) == [Path(result["report"])]
    outputs = _outputs(root)
    times = {path: path.stat().st_mtime_ns for path in outputs}
    assert result == publication.publish_current(root)
    assert outputs == _outputs(root)
    assert times == {path: path.stat().st_mtime_ns for path in outputs}


def test_newer_measured_revision_updates_one_row_without_pooling(evidence):
    root, manifest = evidence
    publication.publish_current(root)
    old = (root / manifest["cohorts"][0]["snapshot"]).read_bytes()
    _, newer = _register(root, manifest, "bcap", "c", finished="2026-10-02T01:00:00+00:00")
    result = read_json(publication.publish_current(root)["json"])
    rows = [row for row in result["rows"] if row["candidate_id"] == "bcap"]
    assert len(rows) == 1 and rows[0]["candidate_revision"] == "revision-c"
    assert rows[0]["tiers"]["1"]["passed"] == 0
    assert rows[0]["runtime_cohort"] == newer["rows"][0]["runtime_cohort"]
    assert rows[0]["bindings"] == newer["rows"][0]["bindings"]
    assert old == (root / manifest["cohorts"][0]["snapshot"]).read_bytes()
    assert len(list((root / "reports/forge").rglob("*.md"))) == 1


@pytest.mark.parametrize("field", ["cohort", "runtime_cohort"])
def test_equal_time_different_cohorts_cannot_silently_choose_a_winner(evidence, field):
    root, manifest = evidence
    publication.publish_current(root)
    before = _outputs(root)
    entry = deepcopy(manifest["cohorts"][0])
    report = read_json(root / entry["snapshot"])
    report["rows"][0][field] = "different-cohort" if field == "cohort" else {
        "execution_backend": "cuda", "compute_profiles": {"cuda": {"model": "different-gpu"}}}
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    entry["snapshot"] = "reports/forge/technique-evidence/other-runtime.json"
    atomic_json(root / entry["snapshot"], report)
    entry["json_sha256"] = file_hash(root / entry["snapshot"])
    manifest["cohorts"].append(entry)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    with pytest.raises(ValueError, match="ambiguous latest recorded technique cohort"):
        publication.publish_current(root)
    assert _outputs(root) == before


@pytest.mark.parametrize("unresolved", [False, True])
def test_new_card_is_discovered_with_truthful_unknown_or_blocked_status(evidence, monkeypatch, unresolved):
    root, manifest = evidence
    _, report = _register(root, deepcopy(manifest), "new-technique", "c")
    if unresolved:
        report["rows"][0].update(candidate_revision=None, cohort=None, bindings={})
        report["rows"][0]["tasks"][0]["status"] = "BLOCKED"
    # Leave the new card declared, but not registered as measured evidence.
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    def live(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json"))}
    monkeypatch.setattr(publication, "write_report", live)
    result = read_json(publication.publish_current(root)["json"])
    assert len(result["rows"]) == 3
    new = next(row for row in result["rows"] if row["candidate_id"] == "new-technique")
    assert new["tasks"][0]["status"] == ("BLOCKED" if unresolved else "UNKNOWN")
    assert new["tiers"]["1"] == {"passed": 0, "total": 3}
    assert not new["attempt_ids"] and "publication_key" not in new
    if unresolved:
        assert "unresolved / unresolved" in (root / "reports/forge/technique-inventory.md").read_text()


def test_explicit_frozen_regrade_registers_evidence_and_updates_canonical(evidence, monkeypatch):
    root, manifest = evidence
    publication.publish_current(root)
    _, report = _register(root, deepcopy(manifest), "bcap", "c", finished="2026-10-03T01:00:00+00:00")
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    atomic_json(root / "reports/forge/attempts/attempt-c/result.json", {"raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    def regrade(*args, source_commit, output_prefix, **kwargs):
        assert source_commit == "commit-c"
        assert root not in output_prefix.parents
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    result = publication.publish_current(root, source_commit="commit-c")
    current = read_json(result["json"])
    assert len(current["rows"]) == 2
    assert next(row for row in current["rows"] if row["candidate_id"] == "bcap")["candidate_revision"] == "revision-c"
    updated = read_json(root / publication.EVIDENCE_MANIFEST)
    assert len(updated["cohorts"]) == 3
    assert (root / updated["cohorts"][-1]["snapshot"]).is_file()
    assert len(list((root / "reports/forge").rglob("*.md"))) == 1
    assert publication.publish_current(root) == result


def test_same_source_cpu_registration_retains_cuda_evidence(evidence, monkeypatch):
    root, manifest = evidence
    original = read_json(root / manifest["cohorts"][0]["snapshot"])
    report = deepcopy(original)
    row = report["rows"][0]
    row["runtime_cohort"] = {"execution_backend": "cpu", "compute_profiles": {"cpu": {"model": "cpu-a"}}}
    row["candidate_revision"], row["cohort"] = "revision-cpu", "cohort-cpu"
    row["attempt_ids"] = ["attempt-cpu"]
    report["execution_backend"] = "cpu"
    proof = deepcopy(report["provenance"]["qualified_receipts"]["attempt-a"])
    proof["canonical_result_hash"] = "result-cpu"
    report["provenance"]["qualified_receipts"] = {"attempt-cpu": proof}
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    summary = read_json(root / "reports/forge/technique-receipts/attempt-a.json")
    summary["candidate_revision"] = "revision-cpu"
    summary["provenance"]["canonical_result_hash"] = "result-cpu"
    atomic_json(root / "reports/forge/technique-receipts/attempt-cpu.json", summary)
    atomic_json(root / "reports/forge/attempts/attempt-cpu/result.json", {"raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    # Restrict this fixture to its measured technique; Atlas has no CPU entry.
    (root / "configs/forge/ideas/atlas.json").unlink()
    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-a"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    cpu = publication.publish_current(root, execution_backend="cpu", source_commit="commit-a")
    assert read_json(cpu["json"])["rows"][0]["candidate_revision"] == "revision-cpu"
    cuda = read_json(publication.publish_current(root)["json"])
    assert cuda["rows"][0]["candidate_revision"] == "revision-a"
    both = read_json(publication.publish_current(root, execution_backend=None)["json"])
    assert {row["runtime_cohort"]["execution_backend"] for row in both["rows"]} == {"cpu", "cuda"}
    assert len(read_json(root / publication.EVIDENCE_MANIFEST)["cohorts"]) == 3


@pytest.mark.parametrize("tamper,message", [
    ("bytes", "snapshot hash mismatch"), ("digest", "input digest mismatch"),
    ("proof", "candidate/source cohort differs"), ("denominator", "tier denominator"),
    ("protocol", "incompatible tier_requirements"), ("policy", "current view policy differs"),
    ("contract", "invalid recipe_contracts identity"),
])
def test_invalid_evidence_cannot_overwrite_current_table(evidence, tamper, message):
    root, manifest = evidence
    publication.publish_current(root)
    before = _outputs(root)
    entry = manifest["cohorts"][0]
    path = root / entry["snapshot"]
    report = read_json(path)
    if tamper == "proof":
        summary_path = root / "reports/forge/technique-receipts/attempt-a.json"
        summary = read_json(summary_path)
        summary["candidate_revision"] = "wrong-revision"
        atomic_json(summary_path, summary)
    elif tamper == "policy":
        atomic_json(root / "configs/forge/views/discriminator_stability.json", {"id": "discriminator_stability", "revision": 3})
    else:
        if tamper in {"bytes", "digest", "denominator"}:
            report["rows"][0]["tiers"]["1"]["total"] = 2
        elif tamper == "protocol":
            report["tier_requirements"]["1"] = ["two_pole"]
        elif tamper == "contract":
            report["recipe_contracts"] = {"invalid-hash": {"reg_arm": "other"}}
        if tamper not in {"bytes", "digest"}:
            report["provenance"].pop("input_digest")
            report["provenance"]["input_digest"] = stable_hash(report)
        atomic_json(path, report)
        if tamper != "bytes":
            entry["json_sha256"] = file_hash(path)
            atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    with pytest.raises(ValueError, match=message):
        publication.publish_current(root)
    assert _outputs(root) == before


def test_cli_has_one_default_destination(evidence, capsys):
    root, _ = evidence
    publication.main(["--root", str(root)])
    assert '"rows":2' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        publication.main(["--root", str(root), "--output-prefix", "another-board"])


def test_raw_technique_export_cannot_replace_registered_current_evidence(evidence):
    from experiments.forge.technique_board import write_report
    root, _ = evidence
    publication.publish_current(root)
    before = _outputs(root)
    with pytest.raises(ValueError, match="regenerate_technique_inventory.py"):
        write_report(root)
    assert _outputs(root) == before


def test_committed_cohorts_rebuild_every_scientific_row_in_a_checkout_without_raw_logs(tmp_path):
    for relative in (publication.EVIDENCE_MANIFEST.parent, Path("reports/forge/technique-receipts"),
                     Path("configs/forge/ideas"), Path("configs/forge/views")):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    manifest = read_json(tmp_path / publication.EVIDENCE_MANIFEST)
    expected = {}
    for entry in manifest["cohorts"]:
        snapshot = read_json(tmp_path / entry["snapshot"])
        expected.update({row["candidate_id"]: row for row in snapshot["rows"] if row["candidate_id"] in entry["candidates"]})
    # This fixture isolates cached reconstruction; separate tests exercise new
    # cards' live declaration resolution without needing a full Git checkout.
    for card in (tmp_path / "configs/forge/ideas").glob("*.json"):
        if card.stem not in expected:
            card.unlink()
    result = read_json(publication.publish_current(tmp_path)["json"])
    display_fields = {"technique", "publication_key", "qualification_input", "qualification_reuse"}
    science = lambda row: {key: value for key, value in row.items() if key not in display_fields}
    assert {row["candidate_id"]: science(row) for row in result["rows"]} == {name: science(row) for name, row in expected.items()}
    assert len(result["rows"]) == len(expected)
    assert not (tmp_path / "reports/forge/attempts").exists()
    assert len(list((tmp_path / "reports/forge").rglob("*.md"))) == 1
