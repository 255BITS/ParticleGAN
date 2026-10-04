"""One current table can be rebuilt from committed, source-bound evidence."""
from copy import deepcopy
from collections import Counter
import importlib.util
from pathlib import Path
import shutil

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin

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
    policy = {"id": "discriminator_stability", "revision": 2,
              "assignments": [{"task": name, "importance": "required", "qualification_tier": 1}
                              for name in ("two_pole", "token", "ae")]}
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
    # These cached-evidence fixtures intentionally have no executable toy
    # declarations; only the complete committed-checkout test uses real tasks.
    monkeypatch.setattr("experiments.forge.views.load_view", lambda root, view_id:
                        read_json(Path(root) / f"configs/forge/views/{view_id}.json"))
    return tmp_path, manifest


def _outputs(root):
    return {path: path.read_bytes() for path in (root / "reports/forge").glob("technique-inventory.*")}


def _family_pin(root, manifest, row):
    card = {"schema_version": 1, "scope": "whole_candidate_family_current", "default_adoption": False,
            "view": manifest["view"], "policy_fingerprint": manifest["policy_fingerprint"],
            "selections": [family_row_pin(row, selection_kind="historical_incumbent", reason="Recorded incumbent.")]}
    atomic_json(root / CURRENT_SELECTION, card)
    return card


def test_current_pin_retains_exact_incumbent_after_new_source_and_never_pools(evidence):
    root, manifest = evidence
    incumbent = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)
    _register(root, manifest, "bcap", "c", finished="2026-10-03T01:00:00+00:00", passed=1)
    result = read_json(publication.publish_current(root)["json"])
    selected = next(row for row in result["rows"] if row["candidate_id"] == "bcap")
    assert selected["candidate_revision"] == "revision-a"
    assert selected["tasks"] == incumbent["tasks"] and selected["tiers"] == incumbent["tiers"]
    assert len([row for row in result["rows"] if row["trainer_family"] == "bcap"]) == 1
    assert {row["candidate_revision"] for row in result["configuration_rows"] if row["trainer_family"] == "bcap"} == {"revision-a", "revision-c"}
    assert result["provenance"]["family_current_selection_sha256"] == file_hash(root / CURRENT_SELECTION)
    before = _outputs(root)
    assert publication.publish_current(root)["input_digest"] == result["provenance"]["input_digest"]
    assert _outputs(root) == before


@pytest.mark.parametrize("tamper", ["duplicate", "policy", "source", "recipe", "runtime", "whole_row", "standard"])
def test_invalid_current_family_pin_cannot_change_publication(evidence, tamper):
    root, manifest = evidence
    row = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    row["trainer_family"] = "bcap"
    card = _family_pin(root, manifest, row)
    publication.publish_current(root)
    before = _outputs(root)
    if tamper == "duplicate":
        card["selections"].append(deepcopy(card["selections"][0]))
    elif tamper == "policy":
        card["policy_fingerprint"] = "edited"
    elif tamper == "standard":
        card["selections"][0]["selection_kind"] = "configured_standard"
    else:
        key = {"source": "source_digest", "recipe": "recipe_sha256", "runtime": "runtime_cohort_sha256",
               "whole_row": "scientific_row_sha256"}[tamper]
        card["selections"][0][key] = "edited"
    atomic_json(root / CURRENT_SELECTION, card)
    with pytest.raises(ValueError):
        publication.publish_current(root)
    assert _outputs(root) == before


def _copy_word_diagnostics(root):
    selection = read_json(ROOT / Path("configs/forge/selections/word-joint-task-v1.json"))
    paths = [Path("configs/forge/selections/word-joint-task-v1.json")]
    paths += [Path(row[key]["path"]) for row in selection["recipes"] for key in ("receipt", "parent")]
    for relative in paths:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, root / relative)
    return selection


def _copy_completed_studies(root):
    from experiments.forge.completed_studies import REGISTRY
    registry = read_json(ROOT / REGISTRY)
    paths = [REGISTRY]
    for study in registry["studies"]:
        paths += [study[role]["path"] for role in ("report", "archive", "readout", "archive_readout")]
        paths += [pin["path"] for pin in study["media"]]
    for relative in paths:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, root / relative)
    return registry


def test_completed_studies_cannot_replace_selected_family_configs_or_history(evidence):
    root, manifest = evidence
    incumbent = read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0]
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)
    baseline = read_json(publication.publish_current(root)["json"])
    assert "completed_api_studies" not in baseline
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    _copy_completed_studies(root)
    published = publication.publish_current(root)
    current = read_json(published["json"])
    for key, value in baseline.items():
        if key != "provenance":
            assert current[key] == value
    assert set(current) - set(baseline) == {"completed_api_studies"}
    assert current["provenance"]["selected_rows_sha256"] == baseline["provenance"]["selected_rows_sha256"]
    assert current["provenance"]["input_digest"] != baseline["provenance"]["input_digest"]
    rows = current["completed_api_studies"]["rows"]
    assert rows[0]["counts"] == {"PASS": 19}
    assert rows[1]["counts"] == {"FAIL": 2}
    assert all(row["qualification_input"] is row["reuse"] is False for row in rows)
    assert all(row["counts"]["execution"] == {"FAIL": 2, "UNKNOWN": 14} for row in rows[2:])
    markdown = Path(published["report"]).read_text()
    assert "task_diagnostics" not in current
    assert "Five-word joint task diagnostics" not in markdown
    assert "<summary>Selected configurations and provenance</summary>" in markdown
    assert "19/19 original PASS" in markdown and "2/2 new hold FAIL" in markdown
    assert "19/19 original PASS" in markdown.split("| Trainer family / runtime", 1)[0]
    outputs = _outputs(root)
    mtimes = {path: path.stat().st_mtime_ns for path in outputs}
    assert publication.publish_current(root) == published
    assert outputs == _outputs(root)
    assert mtimes == {path: path.stat().st_mtime_ns for path in outputs}
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest


@pytest.mark.parametrize("role", ["report", "archive_readout", "gif"])
def test_completed_study_input_drift_preserves_public_outputs_and_evidence(evidence, role):
    root, _ = evidence
    registry = _copy_completed_studies(root)
    publication.publish_current(root)
    before = _outputs(root)
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    study = registry["studies"][0]
    pin = study["media"][0] if role == "gif" else study[role]
    path = root / pin["path"]
    path.write_bytes(path.read_bytes() + b"changed input")
    with pytest.raises(ValueError, match="pinned input drift"):
        publication.publish_current(root)
    assert _outputs(root) == before
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest


def test_baseline_diagnosis_is_additive_bound_navigation(evidence):
    root, _ = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    relative = Path("reports/forge/c6-baseline-debug-20261003")
    shutil.copytree(ROOT / relative, root / relative)
    current = read_json(publication.publish_current(root)["json"])
    for key, value in baseline.items():
        if key != "provenance":
            assert current[key] == value
    assert set(current) - set(baseline) == {"baseline_debugging"}
    debug = current["baseline_debugging"]
    assert debug["qualification_input"] is debug["qualification_reuse"] is False
    assert debug["files_sha256"] == {
        path.relative_to(root).as_posix(): file_hash(path) for path in (root / relative).iterdir()
    }
    before = _outputs(root)
    (root / relative / "BASELINE_SELECTION.json").unlink()
    with pytest.raises(FileNotFoundError):
        publication.publish_current(root)
    assert _outputs(root) == before


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


def test_recorded_policy_preserves_old_rows_after_view_and_declarations_change(evidence, capsys):
    root, manifest = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    archive = Path("configs/forge/view-history/discriminator_stability-v2.json")
    atomic_json(root / archive, read_json(root / "configs/forge/views/discriminator_stability.json"))
    atomic_json(root / "configs/forge/views/discriminator_stability.json", {
        "id": "discriminator_stability", "revision": 3,
        "assignments": [{"task": "new-acquisition", "qualification_tier": 1, "importance": "required"}]})
    atomic_json(root / "configs/forge/ideas/new-word-candidate.json", {"id": "new-word-candidate"})
    protected = {path: path.read_bytes() for path in (root / publication.EVIDENCE_MANIFEST.parent).glob("*.json")}
    before = _outputs(root)
    with pytest.raises(ValueError, match="current view policy differs"):
        publication.publish_current(root)
    assert _outputs(root) == before
    publication.main(["--root", str(root), "--recorded-policy", str(archive)])
    assert '"rows":2' in capsys.readouterr().out
    result = publication.publish_current(root, recorded_policy=archive)
    report = read_json(result["json"])
    assert report["recorded_policy"] == archive.as_posix()
    assert report["view_revision"] == 2
    assert report["tier_requirements"] == manifest["tier_requirements"]
    for key in ("rows", "configuration_rows", "evidence_rows"):
        assert report[key] == baseline[key]
    assert all(row["candidate_id"] != "new-word-candidate" for row in report["configuration_rows"])
    markdown = Path(result["report"]).read_text()
    assert "Recorded Forge trainer-family leaderboard" in markdown
    assert "discriminator_stability revision 2" in markdown
    assert "Tier 1: 3, Tier 2: 19, Tier 3: 2" in markdown
    assert "--recorded-policy " + archive.as_posix() in markdown
    assert not (root / "reports/forge/attempts").exists()
    assert all(path.read_bytes() == original for path, original in protected.items())
    outputs = _outputs(root)
    times = {path: path.stat().st_mtime_ns for path in outputs}
    assert result == publication.publish_current(root, recorded_policy=archive)
    assert _outputs(root) == outputs
    assert times == {path: path.stat().st_mtime_ns for path in outputs}


@pytest.mark.parametrize("field", ["id", "revision", "assignments"])
def test_recorded_policy_must_match_registered_identity(evidence, field):
    root, _ = evidence
    publication.publish_current(root)
    before = _outputs(root)
    policy = read_json(root / "configs/forge/views/discriminator_stability.json")
    policy[field] = "changed"
    archive = "configs/forge/view-history/changed.json"
    atomic_json(root / archive, policy)
    with pytest.raises(ValueError, match="recorded view policy differs"):
        publication.publish_current(root, recorded_policy=archive)
    assert _outputs(root) == before


def test_recorded_policy_cannot_register_source_evidence(evidence):
    root, _ = evidence
    publication.publish_current(root)
    before = _outputs(root)
    with pytest.raises(ValueError, match="cannot register new source evidence"):
        publication.publish_current(root, recorded_policy="unused.json", source_commit="commit-a")
    assert _outputs(root) == before


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


def _later_policy_evidence(root, manifest, monkeypatch, *, revision=3):
    policy = {"id": "discriminator_stability", "revision": revision}
    atomic_json(root / "configs/forge/views/discriminator_stability.json", policy)
    current = {**deepcopy(manifest), "view_revision": revision, "policy_fingerprint": stable_hash(policy),
               "tier_requirements": {**deepcopy(manifest["tier_requirements"]),
                                     "1": ["two_pole", "token", "ae", "ring", "word"]}, "cohorts": []}
    _, report = _register(root, current, "bcap", "c", finished="2026-10-03T01:00:00+00:00")
    _, unknown = _register(root, deepcopy(current), "atlas", "d")
    unknown["rows"][0]["status"] = "BLOCKED"
    unknown["rows"][0]["tasks"][0]["status"] = "BLOCKED"
    report["rows"].extend(unknown["rows"])
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    atomic_json(root / "reports/forge/attempts/attempt-c/result.json", {
        "raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    monkeypatch.setattr(publication, "write_report", regrade)
    return report


def test_explicit_policy_advance_preserves_original_cohorts_without_new_credit(evidence, monkeypatch, capsys):
    root, manifest = evidence
    baseline = read_json(publication.publish_current(root)["json"])
    protected = {entry["snapshot"]: (root / entry["snapshot"]).read_bytes() for entry in manifest["cohorts"]}
    _later_policy_evidence(root, manifest, monkeypatch)
    publication.main(["--root", str(root), "--source-commit", "commit-c", "--device", "cuda", "--advance-policy"])
    assert '"rows":2' in capsys.readouterr().out
    updated = read_json(root / publication.EVIDENCE_MANIFEST)
    assert updated["view_revision"] == 3
    assert len(updated["cohorts"]) == 1 and len(updated["archived_policies"]) == 1
    assert updated["cohorts"][0]["candidates"] == {
        "bcap": "2026-10-03T01:00:00+00:00", "atlas": None}
    archive = updated["archived_policies"][0]
    assert archive == {key: deepcopy(manifest[key]) for key in (*publication.POLICY_FIELDS, "cohorts")}
    def unexpected(*args, **kwargs):
        pytest.fail("frozen blockers must not resolve against live source")
    monkeypatch.setattr(publication, "write_report", unexpected)
    result = publication.publish_current(root)
    current = read_json(result["json"])
    bcap = next(row for row in current["rows"] if row["candidate_id"] == "bcap")
    assert bcap["candidate_revision"] == "revision-c"
    assert bcap["tiers"]["1"] == {"passed": 0, "total": 5}
    assert bcap["qualified_tier"] == 0
    atlas = next(row for row in current["rows"] if row["candidate_id"] == "atlas")
    assert atlas["attempt_ids"] == [] and atlas["cost"]["wall_seconds"] is None
    assert atlas["status"] == "BLOCKED" and atlas["candidate_revision"] == "revision-d"
    assert atlas["runtime_cohort"]["execution_backend"] == "cuda"
    assert len(current["configuration_rows"]) == 2
    assert all(row["candidate_revision"] != "revision-a" for row in current["configuration_rows"])
    original = next(row for row in current["archived_evidence_rows"] if row["candidate_id"] == "bcap")
    assert original["evidence_policy"]["view_revision"] == 2
    assert original["tiers"]["1"] == {"passed": 3, "total": 3}
    assert original["qualified_tier"] == 1
    original.pop("evidence_policy")
    assert original in baseline["evidence_rows"]
    assert all((root / relative).read_bytes() == content for relative, content in protected.items())
    assert "recorded denominators 3/19/2" in Path(result["report"]).read_text()
    assert len(list((root / "reports/forge").rglob("*.md"))) == 1
    outputs = _outputs(root)
    assert publication.publish_current(root) == result
    assert _outputs(root) == outputs
    assert publication.publish_current(root, source_commit="commit-c", execution_backend="cuda", advance_policy=True)
    assert publication.publish_current(root) == result
    assert read_json(root / publication.EVIDENCE_MANIFEST) == updated


def test_policy_advance_refuses_old_source_and_retains_current_files(evidence, monkeypatch):
    root, manifest = evidence
    publication.publish_current(root)
    report = _later_policy_evidence(root, manifest, monkeypatch)
    report["policy_fingerprint"] = manifest["policy_fingerprint"]
    before = _outputs(root)
    manifest_bytes = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    with pytest.raises(ValueError, match="current view policy differs"):
        publication.publish_current(root, source_commit="commit-c")
    with pytest.raises(ValueError, match="new source evidence differs from the current view policy"):
        publication.publish_current(root, source_commit="commit-c", advance_policy=True)
    assert _outputs(root) == before
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest_bytes


def test_policy_advance_requires_later_revision_and_source(evidence, monkeypatch):
    root, manifest = evidence
    publication.publish_current(root)
    before = _outputs(root)
    with pytest.raises(ValueError, match="requires --source-commit"):
        publication.publish_current(root, advance_policy=True)
    _later_policy_evidence(root, manifest, monkeypatch, revision=1)
    with pytest.raises(ValueError, match="later revision of the same view"):
        publication.publish_current(root, source_commit="commit-c", advance_policy=True)
    assert _outputs(root) == before


def test_archived_policy_corruption_cannot_overwrite_current_publication(evidence, monkeypatch):
    root, manifest = evidence
    _later_policy_evidence(root, manifest, monkeypatch)
    publication.publish_current(root, source_commit="commit-c", advance_policy=True)
    before = _outputs(root)
    updated = read_json(root / publication.EVIDENCE_MANIFEST)
    path = root / updated["archived_policies"][0]["cohorts"][0]["snapshot"]
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        publication.publish_current(root)
    assert _outputs(root) == before


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
    cuda = read_json(publication.publish_current(root, execution_backend="cuda")["json"])
    assert cuda["rows"][0]["candidate_revision"] == "revision-a"
    with pytest.raises(ValueError, match="explicit whole-row family selection"):
        publication.publish_current(root)
    both = read_json(publication.publish_current(
        root, recorded_policy="configs/forge/views/discriminator_stability.json")["json"])
    assert {row["runtime_cohort"]["execution_backend"] for row in both["rows"]} == {"cpu", "cuda"}
    assert len(read_json(root / publication.EVIDENCE_MANIFEST)["cohorts"]) == 3


def test_unmeasured_other_backend_cannot_overwrite_measured_registration(evidence, monkeypatch):
    root, manifest = evidence
    _, report = _register(root, deepcopy(manifest), "bcap", "c", finished="2026-10-03T01:00:00+00:00")
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    unmeasured = deepcopy(report["rows"][0])
    unmeasured.update(attempt_ids=[], candidate_revision="unmeasured-cpu", cohort="unmeasured-cpu",
                      runtime_cohort={"execution_backend": "cpu"},
                      tasks=[{"task_id": "two_pole", "status": "UNKNOWN"}])
    report["rows"].append(unmeasured)
    report["provenance"].pop("input_digest")
    report["provenance"]["input_digest"] = stable_hash(report)
    atomic_json(root / "reports/forge/attempts/attempt-c/result.json", {
        "raw": {"finished_at": "2026-10-03T01:00:00+00:00"}})
    def regrade(*args, output_prefix, **kwargs):
        atomic_json(output_prefix.with_suffix(".json"), report)
        return {"json": str(output_prefix.with_suffix(".json")), "source_commit": "commit-c"}
    monkeypatch.setattr(publication, "regenerate", regrade)
    result = publication.publish_current(root, source_commit="commit-c")
    current = read_json(result["json"])
    selected = next(row for row in current["rows"] if row["candidate_id"] == "bcap")
    assert selected["candidate_revision"] == "revision-c"
    assert selected["attempt_ids"] == ["attempt-c"]
    assert selected["runtime_cohort"]["execution_backend"] == "cuda"
    assert publication.publish_current(root) == result


def test_family_new_hardware_keeps_its_own_canonical_fallback(evidence):
    root, manifest = evidence
    (root / "configs/forge/ideas/atlas.json").unlink()
    atomic_json(root / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [
        {"id": "bcap", "label": "BCap", "canonical_candidate": "bcap", "candidates": ["bcap"]}]})
    entry, snapshot = _register(root, manifest, "bcap-trial", "c", finished="2026-10-03T01:00:00+00:00")
    atomic_json(root / "configs/forge/ideas/bcap-trial.json", {"id": "bcap-trial", "trainer_family": "bcap"})
    canonical = deepcopy(snapshot["rows"][0])
    canonical.update(candidate_id="bcap", candidate_revision="revision-canonical-c", cohort="canonical-c", attempt_ids=[],
                     tasks=[{"task_id": "two_pole", "status": "UNKNOWN"}], cost={"wall_seconds": None})
    snapshot["rows"].append(canonical)
    snapshot["provenance"].pop("input_digest")
    snapshot["provenance"]["input_digest"] = stable_hash(snapshot)
    atomic_json(root / entry["snapshot"], snapshot)
    entry["json_sha256"] = file_hash(root / entry["snapshot"])
    atomic_json(root / publication.EVIDENCE_MANIFEST, manifest)
    with pytest.raises(ValueError, match="explicit whole-row family selection"):
        publication.publish_current(root)
    recorded = read_json(publication.publish_current(
        root, recorded_policy="configs/forge/views/discriminator_stability.json")["json"])
    assert len(recorded["rows"]) == 2 and all(row["candidate_id"] == "bcap" for row in recorded["rows"])
    assert len(recorded["configuration_rows"]) == 3
    assert next(row for row in recorded["rows"] if row["cohort"] == "canonical-c")["attempt_ids"] == []
    incumbent = deepcopy(read_json(root / manifest["cohorts"][0]["snapshot"])["rows"][0])
    incumbent["trainer_family"] = "bcap"
    _family_pin(root, manifest, incumbent)
    result = read_json(publication.publish_current(root)["json"])
    assert len(result["rows"]) == 1 and result["rows"][0]["cohort"] == "cohort-a"
    assert len(result["configuration_rows"]) == 3
    assert {row["alternative_scope"] for row in result["configuration_rows"]} == {"selected", "archived_alternative"}


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


def _tier_display_fixture(tier1, later="UNKNOWN"):
    required = {"1": [f"smoke-{i}" for i in range(5)],
                "2": [f"quality-{i}" for i in range(19)],
                "3": [f"hold-{i}" for i in range(2)]}
    statuses = {"1": tier1, "2": [later] * 19, "3": [later] * 2}
    row = {"candidate_id": "atlas", "trainer_family": "atlas", "technique": "Atlas",
           "runtime_cohort": {"execution_backend": "cpu"}, "qualified_tier": 0,
           "tasks": [{"task_id": task, "status": status, "gate_status": status}
                     for tier, names in required.items() for task, status in zip(names, statuses[tier])],
           "tiers": {tier: {"passed": values.count("PASS"), "total": len(values),
                            "counts": dict(Counter(values))} for tier, values in statuses.items()}}
    return {"view": "discriminator_stability", "tier_requirements": required, "rows": [row], "evidence_sources": {},
            "provenance": {"input_digest": "synthetic-display-only"}}


@pytest.mark.parametrize("status,label", [
    ("BLOCKED", "BLOCKED"), ("UNKNOWN", "UNKNOWN"), ("NOT_RUN", "NOT RUN"),
    ("INCOMPLETE", "INCOMPLETE"), ("INVALID", "INVALID"),
])
def test_current_tier_cells_keep_blocked_and_unmeasured_denominators(tmp_path, status, label):
    result = _tier_display_fixture([status] * 5, later=status)
    before = deepcopy(result)
    markdown = publication._current_markdown(result, tmp_path, tmp_path / "reports/forge/table.md")
    row = next(line for line in markdown.splitlines() if line.startswith("| Atlas<br>"))
    for total in (5, 19, 2):
        assert f"{label} ({total} required)" in row
    assert "FAIL" not in row
    assert result == before


@pytest.mark.parametrize("statuses,expected", [
    (["PASS", "FAIL", "BLOCKED", "UNKNOWN", "NOT_RUN"],
     "1/5<br>BLOCKED 1 · FAIL 1 · NOT RUN 1 · UNKNOWN 1"),
    (["FAIL"] * 5, "0/5<br>FAIL 5"),
    (["FAIL"] + ["UNKNOWN"] * 4, "0/5<br>FAIL 1 · UNKNOWN 4"),
    (["NOT_RUN", "UNKNOWN", "NOT_RUN", "UNKNOWN", "UNKNOWN"],
     "NOT RUN/UNKNOWN (5 required)"),
    (["PASS"] * 5, "5/5 PASS"),
])
def test_current_tier_cells_show_actual_mixed_gate_counts(tmp_path, statuses, expected):
    result = _tier_display_fixture(statuses)
    before = deepcopy(result)
    markdown = publication._current_markdown(result, tmp_path, tmp_path / "reports/forge/table.md")
    row = next(line for line in markdown.splitlines() if line.startswith("| Atlas<br>"))
    assert expected in row
    assert "UNKNOWN (19 required)" in row and "UNKNOWN (2 required)" in row
    assert result == before


def test_legacy_tier_display_does_not_invent_an_all_failed_breakdown(tmp_path):
    result = _tier_display_fixture(["BLOCKED"] * 5)
    row = result["rows"][0]
    row["tiers"]["1"].pop("counts")
    assert publication._current_tier_cell(row, "1", result["tier_requirements"]["1"], "table.json") == "BLOCKED (5 required)"
    # A legacy aggregate can lack enough task detail to reconstruct its counts.
    row["tiers"]["1"]["passed"] = 1
    before = deepcopy(result)
    rendered = publication._current_tier_cell(row, "1", result["tier_requirements"]["1"], "table.json")
    assert rendered == "1/5<br>[5 required; task statuses](table.json)"
    assert "FAIL" not in rendered and result == before


def test_main_policy_rows_link_separate_history_and_failed_hold_without_credit():
    result = read_json(ROOT / "reports/forge/technique-inventory.json")
    result["rows"] = [row for row in result["rows"] if row["trainer_family"] in {"atlas", "e22"}]
    assert {row["trainer_family"] for row in result["rows"]} == {"atlas", "e22"}
    before = deepcopy(result)
    markdown = publication._current_markdown(result, ROOT, ROOT / "reports/forge/technique-inventory.md")
    table_rows = [line for line in markdown.splitlines() if line.startswith(("| Atlas<br>", "| E22<br>"))]
    assert len(table_rows) == 2
    for line, label in zip(table_rows, ("Atlas", "E22")):
        assert "[Atlas history: 19/19 PASS](continuous-baseline-20261003/README.md)" in line
        assert f"[C6 {label} hold FAIL](c6-baseline-debug-20261003/README.md)" in line
        assert all(f"BLOCKED ({total} required)" in line for total in (5, 19, 2))
        assert line.endswith("| 0 |")
    assert "E22 history: 19/19" not in markdown
    assert "no current tier credit" in markdown and result == before
    without_history = deepcopy(result)
    without_history.pop("completed_api_studies")
    assert publication._separate_baseline_links(without_history, result["rows"][0], ROOT,
                                                ROOT / "reports/forge/technique-inventory.md") == []


def _copy_atlas_progress(root):
    for relative, _, _ in publication.ATLAS_PROGRESS_INPUTS.values():
        path = Path("reports/forge") / relative
        for name in (path, path.with_name("README.md")):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, root / name)


def test_atlas_progress_is_source_bound_additive_display_and_preserves_every_ordinary_row(evidence):
    root, _ = evidence
    before = read_json(publication.publish_current(root)["json"])
    manifest = (root / publication.EVIDENCE_MANIFEST).read_bytes()
    assert "atlas_unblocking_progress" not in before
    _copy_atlas_progress(root)
    published = publication.publish_current(root)
    after = read_json(published["json"])
    for key, value in before.items():
        if key != "provenance":
            assert after[key] == value
    assert set(after) - set(before) == {"atlas_unblocking_progress"}
    assert (root / publication.EVIDENCE_MANIFEST).read_bytes() == manifest
    assert after["provenance"]["selected_rows_sha256"] == before["provenance"]["selected_rows_sha256"]
    progress = after["atlas_unblocking_progress"]
    assert progress["baseline"]["counts"] == {"PASS": 7, "FAIL": 11, "BLOCKED": 8}
    assert (progress["passed"], progress["required"]) == (7, 8)
    assert [row["passed"] for row in progress["adaptations"]] == [4, 1, 1, 1, 0]
    assert len({row["source"]["digest"] for row in progress["adaptations"]}) == 2
    assert progress["word"]["status"] == "INVALID"
    assert progress["word"]["numerical_gate"] == "UNAVAILABLE"
    assert progress["word"]["raw_completed_steps"] == 20001
    assert progress["word"]["recorded_endpoint"]["mass_tv"] == .6
    assert all(progress[key] is False for key in (
        "qualification_input", "qualification_reuse", "ordinary_tier_credit", "cross_cohort_pooling",
        "default_adoption", "speed_ranking"))
    markdown = Path(published["report"]).read_text()
    assert markdown.index("## Atlas unblocking progress") < markdown.index("## Ordinary MoG qualification")
    assert "**7 PASS / 11 FAIL / 8 BLOCKED**" in markdown and "**7/8 previously blocked question gates**" in markdown
    assert "INVALID; 20,001 updates; raw goals miss; numerical gate UNAVAILABLE" in markdown
    assert "one unchanged family/configuration tuple must satisfy all 26 required" in markdown
    atlas_row = next(line for line in markdown.splitlines() if line.lower().startswith("| atlas<br>"))
    assert "[Adaptation progress](#atlas-unblocking-progress)" in atlas_row
    assert after["rows"] == before["rows"]
    outputs = _outputs(root)
    assert publication.publish_current(root) == published and _outputs(root) == outputs


@pytest.mark.parametrize("view,recorded_policy", [
    ("discriminator_stability", "configs/forge/view-history/discriminator_stability-v2.json"),
    ("quality_coverage", None),
])
def test_ordinary_atlas_progress_does_not_change_recorded_or_other_view_rendering(view, recorded_policy):
    result = _tier_display_fixture(["UNKNOWN"] * 5)
    result.update(view=view, view_revision=2)
    if recorded_policy:
        result["recorded_policy"] = recorded_policy
    path = ROOT / "reports/forge/another-view.md"
    original = publication._current_markdown(result, ROOT, path)
    result["atlas_unblocking_progress"] = publication._atlas_unblocking_progress(ROOT)
    before = deepcopy(result)
    rendered = publication._current_markdown(result, ROOT, path)
    assert rendered == original
    assert "## Ordinary MoG qualification" not in rendered
    assert "## Atlas unblocking progress" not in rendered
    assert "[Adaptation progress]" not in rendered
    assert result == before


@pytest.mark.parametrize("tamper", ["partial", "changed_count", "word_pass", "source", "credit"])
def test_changed_progress_evidence_cannot_rewrite_ordinary_outputs(evidence, tamper):
    root, _ = evidence
    _copy_atlas_progress(root)
    publication.publish_current(root)
    before = _outputs(root)
    if tamper == "partial":
        (root / "reports/forge" / publication.ATLAS_PROGRESS_INPUTS["word_context"][0]).unlink()
    else:
        key = "word_context" if tamper == "word_pass" else "baseline"
        path = root / "reports/forge" / publication.ATLAS_PROGRESS_INPUTS[key][0]
        value = read_json(path)
        if tamper == "changed_count":
            value["counts"]["PASS"] = 26
        elif tamper == "word_pass":
            value["execution_status"] = "PASS"
        elif tamper == "source":
            value["source"]["origin_commit"] = "0" * 40
        else:
            value["qualification_input"] = True
        atomic_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        publication.publish_current(root)
    assert _outputs(root) == before


def test_committed_atlas_progress_display_matches_its_generator_without_raw_evidence():
    result = read_json(ROOT / "reports/forge/technique-inventory.json")
    expected = publication._atlas_unblocking_progress(ROOT)
    assert result["atlas_unblocking_progress"] == expected
    markdown = publication._current_markdown(result, ROOT, ROOT / "reports/forge/technique-inventory.md")
    assert (ROOT / "reports/forge/technique-inventory.md").read_text() == markdown
    assert "**7/8 previously blocked question gates**" in markdown


def test_committed_cohorts_rebuild_every_scientific_row_in_a_checkout_without_raw_logs(tmp_path):
    for relative in (publication.EVIDENCE_MANIFEST.parent, Path("reports/forge/technique-receipts"),
                     Path("configs/forge/ideas"), Path("configs/forge/configurations"),
                     Path("configs/forge/searches"), Path("configs/forge/views"),
                     Path("configs/forge/view-history"),
                     Path("configs/forge/tasks"), Path("configs/forge/protocols"),
                     Path("reports/forge/configuration-search")):
        if not (ROOT / relative).is_dir():
            continue
        shutil.copytree(ROOT / relative, tmp_path / relative)
    shutil.copyfile(ROOT / "configs/forge/trainer-families.json", tmp_path / "configs/forge/trainer-families.json")
    shutil.copyfile(ROOT / "configs/forge/defaults.json", tmp_path / "configs/forge/defaults.json")
    manifest = read_json(tmp_path / publication.EVIDENCE_MANIFEST)
    expected, all_snapshots, registered_rows, unregistered_shadows = {}, [], [], []
    snapshot_bytes = {}
    for entry in manifest["cohorts"]:
        snapshot_bytes[entry["snapshot"]] = (tmp_path / entry["snapshot"]).read_bytes()
        snapshot = read_json(tmp_path / entry["snapshot"])
        all_snapshots.extend(snapshot["rows"])
        for row in snapshot["rows"]:
            if row["candidate_id"] not in entry["candidates"]:
                continue
            recorded_at = entry["candidates"][row["candidate_id"]]
            # Completion-time registrations attest measured runtime rows. A
            # frozen all-backend roster also contains unmeasured CPU shadows
            # of the same CUDA config IDs; they are not registered evidence.
            if isinstance(recorded_at, str) and not row.get("attempt_ids"):
                unregistered_shadows.append(row)
                continue
            registered_rows.append(row)
            key = row["candidate_id"], row["runtime_cohort"]["execution_backend"]
            rank = recorded_at or ""
            if key not in expected or rank > expected[key][0]:
                expected[key] = rank, row
    # This fixture isolates cached reconstruction; separate tests exercise new
    # cards' live declaration resolution without needing a full Git checkout.
    for card in list((tmp_path / "configs/forge/ideas").glob("*.json")) + list((tmp_path / "configs/forge/configurations").glob("*.json")):
        if card.stem not in {key[0] for key in expected}:
            card.unlink()
    recorded = ("configs/forge/view-history/discriminator_stability-v2.json"
                if manifest["view_revision"] == 2 else None)
    result = read_json(publication.publish_current(tmp_path, recorded_policy=recorded)["json"])
    display_fields = {"technique", "publication_key", "qualification_input", "qualification_reuse",
                      "trainer_family", "configuration_id", "comparison_cohort", "selected_configuration", "alternative_scope"}
    science = lambda row: {key: value for key, value in row.items() if key not in display_fields}
    scientific_variants = [science(row) for row in result["configuration_rows"]]
    assert all(science(row) in scientific_variants for _, row in expected.values())
    assert all(science(row) not in scientific_variants for row in unregistered_shadows)
    assert all(row in [science(original) for original in all_snapshots] for row in scientific_variants)
    assert Counter(stable_hash(science(row)) for row in result["evidence_rows"]) == Counter(stable_hash(science(row)) for row in registered_rows)
    assert len(result["rows"]) == len({(row["trainer_family"], stable_hash(row["runtime_cohort"])) for row in result["configuration_rows"]})
    assert len(result["configuration_rows"]) >= len(expected)
    assert all((tmp_path / relative).read_bytes() == original for relative, original in snapshot_bytes.items())
    assert not (tmp_path / "reports/forge/attempts").exists()
    assert len(list((tmp_path / "reports/forge").rglob("*.md"))) == 1
    selection = _copy_word_diagnostics(tmp_path)
    updated = read_json(publication.publish_current(tmp_path)["json"])
    for key in ("rows", "configuration_rows", "evidence_rows", "archived_evidence_rows", "tier_requirements"):
        assert updated.get(key) == result.get(key)
    assert "task_diagnostics" not in updated
    assert "Five-word joint task diagnostics" not in (tmp_path / "reports/forge/technique-inventory.md").read_text()
    before = _outputs(tmp_path)
    times = {path: path.stat().st_mtime_ns for path in before}
    publication.publish_current(tmp_path)
    assert _outputs(tmp_path) == before
    assert times == {path: path.stat().st_mtime_ns for path in before}
    # A historical direct-task export is no longer an active solution input.
    (tmp_path / selection["recipes"][0]["receipt"]["path"]).write_text("{}\n")
    publication.publish_current(tmp_path)
    assert _outputs(tmp_path) == before
