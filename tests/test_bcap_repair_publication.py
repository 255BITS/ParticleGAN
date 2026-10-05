"""Bounded publication verification with durable receipts, never training."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.views import view_fingerprint

REPOSITORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repair_publication", REPOSITORY / "reports/forge/bcap-tier1-repair/publish_evidence.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.fixture(params=[False, True], ids=["rates", "rates-and-moments"])
def publication(tmp_path, monkeypatch, request):
    root = tmp_path
    generator = root / "reports/forge/regenerate_technique_inventory.py"
    generator.parent.mkdir(parents=True)
    shutil.copyfile(REPOSITORY / "reports/forge/regenerate_technique_inventory.py", generator)
    publisher = repair._publisher(root)
    monkeypatch.setattr(repair, "_publisher", lambda _: publisher)
    names = [f"task{n}" for n in range(7)]
    view = {"id": "discriminator_stability", "revision": 6, "assignments": [
        {"task": name, "importance": "required", "qualification_tier": 1, "order": n}
        for n, name in enumerate(names)]}
    state = {"submissions": {}, "jobs": {}, "campaigns": {}}
    results = {"status": "COMPLETE", "qualification_input": False, "candidates": [], "campaign_accounting": {}}
    frozen = {}

    def add(candidate, campaign, source_character, task_names, passing, source_file_character=None):
        source_files = {"particlegan/fixture.py": (source_file_character or source_character) * 64}
        source = {"files": source_files, "digest": stable_hash(source_files), "origin_commit": source_character * 40}
        job = {"task_id": task_names[0], "task_ids": task_names, "compatibility_key": candidate}
        declared = {"candidate": {"id": candidate}, "candidate_revision": candidate + "-revision",
                    "campaign_id": campaign, "source": source, "runtime": {"backend": "cpu"},
                    "view": view, "through_tier": 1, "jobs": [job]}
        attempt = "attempt-" + candidate
        tasks = [{"task_id": name, "gate_status": "PASS" if n < passing else "FAIL",
                  "compatibility_key": candidate, "metrics": {"score": float(n), "component_counts": [1, 2, 3]}}
                 for n, name in enumerate(task_names)]
        result = {"candidate_revision": declared["candidate_revision"], "attempt_id": attempt, "task_results": tasks}
        original = root / "reports/forge/attempts" / attempt
        atomic_json(original / "request.json", declared)
        atomic_json(original / "result.json", result)
        atomic_json(original / "evidence.json", {"result_hash": stable_hash(result), "source": source, "runtime": declared["runtime"]})
        state["submissions"][candidate] = {"request": declared, "status": "completed"}
        state["jobs"][candidate] = {"status": "completed"}
        readout = {"candidate": candidate, "candidate_revision": declared["candidate_revision"], "campaign": campaign,
                   "submission_status": "completed", "source_digest": source["digest"], "executed_commit": source["origin_commit"],
                   "tasks": [{"task_id": task["task_id"], "status": task["gate_status"], "metrics": task["metrics"],
                              "attempt_id": attempt, "canonical_result_sha256": stable_hash(result)} for task in tasks]}
        results["candidates"].append(readout)
        return source, declared, {"candidate_id": candidate, "candidate_revision": declared["candidate_revision"],
            "attempt_ids": [attempt], "tasks": [{"task_id": t["task_id"], "status": t["gate_status"]} for t in tasks],
            "tiers": {"1": {"passed": passing, "total": len(task_names)}},
            "bindings": {"source_digest": source["digest"], "configuration_id": candidate}}

    selected = [repair.STUDY] + (["bcap-tier1-repair-moments-v1"] if request.param else [])
    for group, study in enumerate(selected):
        filename, count = repair.STUDIES[study]
        state["campaigns"][study] = {"reserved_seconds": 0}
        results["campaign_accounting"][study] = {"reserved_seconds": 0}
        trials, rows = [], []
        for index in range(count):
            candidate = f"recipe-{group}-{index}"
            source, _, row = add(candidate, study, "ab"[group], names, index % 4 + 2)
            rows.append(row)
            path = root / f"configs/forge/configurations/{candidate}.json"
            atomic_json(path, {"id": candidate})
            trials.append({"candidate": candidate, "declaration": str(path.relative_to(root)),
                           "declaration_sha256": file_hash(path), "tasks": names})
        spec_path = root / f"configs/forge/searches/{study}.json"
        atomic_json(spec_path, {"id": study})
        atomic_json(root / repair.DIRECTORY / filename, {"study": study, "trials": trials, "source_digest": source["digest"],
            "policy_fingerprint": view_fingerprint(view), "spec": str(spec_path.relative_to(root)), "spec_sha256": file_hash(spec_path)})
        frozen[source["origin_commit"]] = {"view": view["id"], "view_revision": 6,
            "policy_fingerprint": view_fingerprint(view), "tier_requirements": {"1": names}, "rows": rows,
            "task_contracts": {source["digest"]: {"fixture": group}}}
    # Both cohorts execute identical scientific bytes at different commits.
    # The ordinary rows must retain their exact attempt commit even though
    # digest-level publication provenance has two recorded origins.
    add("incumbent", repair.DIAGNOSTIC, "c", ["gaussian-longer", "ring-longer"], 0,
        source_file_character="b" if request.param else "a")
    results["campaign_accounting"][repair.DIAGNOSTIC] = {"reserved_seconds": 0}
    atomic_json(root / repair.DIRECTORY / "results.json", results)
    atomic_json(root / "runs/forge/bcap-tier1-repair/queue/queue/state.json", state)
    readout = root / repair.DIRECTORY / "README.md"
    readout.write_text("All complete recipes are reported; the incumbent is retained.\n")
    selection = root / "configs/forge/selections/family-current-v1.json"
    atomic_json(selection, {"recorded_policy": "v5", "selected": "incumbent"})
    calls = []

    def frozen_report(_, commit, **kwargs):
        calls.append(commit)
        measured = deepcopy(frozen[commit])
        return {}, measured, commit, sorted({row["bindings"]["source_digest"] for row in measured["rows"]})

    monkeypatch.setattr(publisher, "_frozen_report", frozen_report)
    monkeypatch.setattr(publisher, "_resolve_commit", lambda _, commit: commit)
    return root, results, state, calls, selection


def test_freeze_regrades_each_source_once_and_preserves_whole_rows_and_selection(publication):
    root, _, _, calls, selection = publication
    before = selection.read_bytes()
    outcome = repair.freeze(root)
    report = repair._verified_snapshot(root)
    assert len(calls) == len(report["repair_studies"])
    assert len(report["rows"]) == sum(repair.STUDIES[study][1] for study in report["repair_studies"])
    assert report["duration_diagnostics"]["qualification_input"] is False
    assert outcome["all_required_pass_candidates"] == []
    assert all(best["required_passes"] == 5 for best in outcome["best_observed_by_study"].values())
    assert selection.read_bytes() == before
    assert not (root / repair.SNAPSHOT.with_suffix(".md")).exists()
    repair.register(root)
    rendered = repair.display_section(root, root / "reports/forge/technique-inventory.md")
    assert "5/7 PASS" in rendered and "the selected incumbent is retained" in rendered
    assert "view revision 6" in rendered and "task6` FAIL" in rendered
    assert selection.read_bytes() == before


@pytest.mark.parametrize("mutation", ["in_progress", "pending_submission", "reserved", "draining_cancelled"])
def test_publication_rejects_unfinished_or_draining_work(publication, mutation):
    root, results, state, calls, _ = publication
    candidate = next(iter(state["submissions"]))
    if mutation == "in_progress":
        results["status"] = "IN_PROGRESS"
    elif mutation == "pending_submission":
        state["submissions"][candidate]["status"] = "queued"
    elif mutation == "reserved":
        state["campaigns"][repair.STUDY]["reserved_seconds"] = 1
    else:
        state["submissions"][candidate]["status"] = "cancelled"
        state["jobs"][candidate]["status"] = "running"
    atomic_json(root / repair.DIRECTORY / "results.json", results)
    atomic_json(root / "runs/forge/bcap-tier1-repair/queue/queue/state.json", state)
    with pytest.raises(ValueError, match="COMPLETE|finish or cancel|draining"):
        repair.freeze(root)
    assert calls == [] and not (root / repair.SNAPSHOT).exists()


@pytest.mark.parametrize("filename", ["results.json", "rates-evidence.json", "README.md"])
def test_registered_navigation_rejects_changed_artifacts(publication, filename):
    root, *_ = publication
    repair.freeze(root)
    repair.register(root)
    with (root / repair.DIRECTORY / filename).open("a") as stream:
        stream.write(" \n")
    with pytest.raises(ValueError, match="navigation hash mismatch"):
        repair.display_section(root, root / "reports/forge/technique-inventory.md")


def test_numerical_receipt_and_source_binding_fail_closed(publication):
    root, results, _, _, _ = publication
    repair.freeze(root)
    results["candidates"][0]["tasks"][0]["metrics"]["score"] = 999
    atomic_json(root / repair.DIRECTORY / "results.json", results)
    with pytest.raises(ValueError, match="certified numerical receipt"):
        repair.register(root)
    assert not (root / repair.REGISTRY).exists()


def test_expected_studies_prevent_partial_publication(publication):
    root, _, _, calls, _ = publication
    with pytest.raises(ValueError, match="complete expected admitted study set"):
        repair.freeze(root, expected_studies=[])
    assert calls == []


def test_stale_complete_readout_cannot_hide_new_admission(publication):
    root, _, state, _, _ = publication
    repair.freeze(root)
    old = next(iter(state["submissions"].values()))
    new = deepcopy(old)
    new["request"]["campaign_id"] = "bcap-tier1-repair-moments-v1"
    new["status"] = "running"
    state["submissions"]["new-admission"] = new
    atomic_json(root / "runs/forge/bcap-tier1-repair/queue/queue/state.json", state)
    with pytest.raises(ValueError, match="omits an admitted study|active or draining"):
        repair.register(root)


def test_no_registration_yields_no_display_section(tmp_path):
    assert repair.display_section(tmp_path, tmp_path / "board.md") == ""


def test_identical_source_bytes_accept_distinct_recorded_origins(publication):
    root, results, _, _, _ = publication
    repair.freeze(root)
    report = repair._verified_snapshot(root)
    shared_origin = "b" * 40 if len(report["repair_studies"]) == 2 else "a" * 40
    shared_rows = [row for row in report["rows"] if shared_origin in row["bindings"]["recorded_source_origin_commits"]]
    assert shared_rows
    assert all(row["bindings"]["source_origin_commit"] is None for row in shared_rows)
    assert all(set(row["bindings"]["recorded_source_origin_commits"]) == {shared_origin, "c" * 40}
               for row in shared_rows)
    assert any(row["executed_commit"] == shared_origin for row in results["candidates"])
    assert repair._verified_readout(root, report) == results
    repair.register(root)


def test_shared_digest_cannot_substitute_another_recorded_execution_commit(publication):
    root, results, _, _, _ = publication
    repair.freeze(root)
    report = repair._verified_snapshot(root)
    shared_origin = "b" * 40 if len(report["repair_studies"]) == 2 else "a" * 40
    ordinary = next(row for row in results["candidates"] if row["executed_commit"] == shared_origin)
    # This forged commit really occurs in the digest-level origin list. Only
    # the later exact certified attempt check detects the substitution.
    ordinary["executed_commit"] = "c" * 40
    atomic_json(root / repair.DIRECTORY / "results.json", results)
    with pytest.raises(ValueError, match="certified numerical receipt"):
        repair._verified_readout(root, report)
    with pytest.raises(ValueError, match="certified numerical receipt"):
        repair.register(root)
    assert not (root / repair.REGISTRY).exists()
