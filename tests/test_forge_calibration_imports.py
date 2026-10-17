"""Cross-profile reuse of certified diagnostics, with no training or new receipts."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge import calibration as c, calibration_lane as lane, views
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue
from experiments.forge.sampling import executed_receipt, PUBLIC_PRIOR_CLEAN
from test_forge_promotion import setup as promotion_setup, save_attempt
from test_forge_queue import SLOTS
from forge_legacy_fixtures import pin_legacy


@pytest.fixture
def imports_setup(promotion_setup):
    root, _, _, _ = promotion_setup
    # Both screens were already in the task catalog used for the original run.
    task = read_json(root / "configs/forge/tasks/cheap.json")
    task["id"] = task["execution"]["fixture_id"] = "alternate"
    atomic_json(root / "configs/forge/tasks/alternate.json", task)
    path = root / "configs/forge/views/stability.json"
    view = read_json(path)
    view["assignments"].append({"task": "alternate", "qualification_tier": 2, "importance": "required", "order": 1})
    atomic_json(path, view)
    profile_path = root / "configs/forge/calibration/current-fixture.json"
    contract = {"schema_version": 1, "id": "original-lane", "profile": "current-fixture",
        "profile_sha256": file_hash(profile_path), "view": "stability",
        "selections": [{"lineage_id": "negative1", "tasks": ["quality"], "reason": "Independent reference diagnostic."}],
        "budgets": {"task_seconds": {"quality": 10}, "candidate_seconds": 10, "campaign_seconds": 10},
        "execution_backend": "cpu", "cuda_model": None, "purpose": "Compare declared screens.",
        "failure_policy": "continue_registered_diagnostics", "qualification_reuse": False}
    contract_path = root / "original-lane.json"
    atomic_json(contract_path, contract)
    lane.register(root, contract_path)
    request = lane.plan_calibration(root, "original-lane", root / "queue", freeze_source=True)[0]
    queue = Queue(root / "queue", report_root=root / "reports/forge")
    queue.submit(request, request["calibration_campaign"])
    claim = queue.claim(SLOTS)
    assert claim["job"]["task_id"] == "quality"
    raw = {"evidence": {"observations": [{"step": i, "score": 0.} for i in range(1, 25)],
                        "live": {"score": 0.}, "scoring_weights": "live",
                        **executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")}}
    grade = views.grade_result(request["tasks"]["quality"], raw)
    atomic_json(Path(claim["worker"]["directory"]) / "terminal.json", {
        "token": claim["worker"]["token"], "attempt_status": "completed", "result": raw, "elapsed_seconds": 1.,
        "grading": {"raw_hash": stable_hash(raw), "source_digest": request["source"]["digest"], "grades": {"quality": grade}}})
    queue.collect()
    assert queue.claim(SLOTS) is None
    # Replace software-fixture receipts, retaining only the original lane's
    # measurement of negative1/quality. Counterfeit PASS stamps are regraded.
    for name, filename, score in (("finished", "qualification", 1.), ("negative1", "calibration-negative1", 0.),
                                  ("negative2", "calibration-negative2", 0.)):
        ordinary = resolve_idea(root, name, through_tier=3, execution_backend="cpu")
        save_attempt(root, ordinary, filename, score=score, cost={"cheap": .05, "alternate": .05, "quality": 1.},
                     tasks=["cheap", "alternate"] + ([] if name == "negative1" else ["quality"]))
    target = read_json(profile_path)
    target.update(id="alternate", smoke_tasks=["alternate"],
                  cohort=c.calibration_cohort(request, ["alternate", "quality"]),
                  diagnostic_imports=c.diagnostic_imports(root, "original-lane", {"negative1": ["quality"]}))
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    return root, queue, request, target, contract


def negative(report):
    return next(row for row in report["matrix"] if row["id"] == "negative1")


def test_queue_dedup_reuses_original_diagnostic_across_explicit_profile_import(imports_setup):
    root, queue, request, target, contract = imports_setup
    original = target["diagnostic_imports"][0]
    before = {str(p): p.read_bytes() for p in (root / "reports/forge/attempts").rglob("*") if p.is_file()}
    contract.update(id="target-lane", profile="alternate",
                    profile_sha256=file_hash(root / "configs/forge/calibration/alternate.json"))
    path = root / "target-lane.json"
    atomic_json(path, contract)
    lane.register(root, path)
    target_request = lane.plan_calibration(root, "target-lane", root / "queue", freeze_source=True)[0]
    assert target_request["calibration_lane"]["cohort_sha256"] != request["calibration_lane"]["cohort_sha256"]
    assert target_request["jobs"] == request["jobs"]
    submission = queue.submit(target_request, target_request["calibration_campaign"])
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert state["submissions"][submission["request"]["request_id"]]["status"] == "completed"
    assert state["campaigns"]["calibration-target-lane"]["spent_seconds"] == 0
    assert sum(len(job["attempts"]) for job in state["jobs"].values()) == 1
    report = c.calibrate(root, "alternate")
    assert report["adoption"] == "PASS"
    row = negative(report)
    assert row["reference"]["decision"] == "FAIL" and row["reference"]["cost"]["wall_seconds"] == 1.
    assert sum(i["attempt_id"] == original["attempt_id"] for i in row["inputs"]) == 1
    assert not report["current_qualification_reuse"]
    assert before == {str(p): p.read_bytes() for p in (root / "reports/forge/attempts").rglob("*") if p.is_file()}
    # Same selected task still requires separate ordinary evidence.
    pin_legacy(root)
    ordinary = resolve_idea(root, "negative1", through_tier=3, freeze_source=True,
                            execution_backend="cpu", queue_root=root / "queue")
    queue.submit(ordinary, {"id": "ordinary", "budget_seconds": 30, "candidate_budget_seconds": 30})
    assert queue.claim(SLOTS)["job"]["task_id"] == "cheap"
    normal = next(j for j in ordinary["jobs"] if j["task_id"] == "quality")
    assert queue.inspect()["jobs"][normal["compatibility_key"]]["result"] is None


def test_unlisted_foreign_diagnostic_neither_contributes_nor_poisons(imports_setup):
    root, _, request, target, _ = imports_setup
    original_id = target["diagnostic_imports"][0]["attempt_id"]
    target.pop("diagnostic_imports")
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    report = c.calibrate(root, "alternate")
    row = negative(report)
    assert row["reference"]["decision"] == "UNKNOWN" and not row["conflicts"]
    assert original_id not in {i["attempt_id"] for i in row["inputs"]}
    # Even an incomplete unselected foreign request cannot poison acceptance.
    save_attempt(root, resolve_idea(root, "negative1", through_tier=3, execution_backend="cpu"),
                 "normal-reference", tasks=["quality"], score=0.)
    atomic_json(root / "reports/forge/attempts/unlisted-incomplete/request.json", {"request": request})
    assert c.calibrate(root, "alternate")["adoption"] == "PASS"


@pytest.mark.parametrize("mutation", ["source", "seed", "task", "candidate", "registration", "result", "certificate",
                                      "missing", "diagnostic_key", "qualification_key", "request_hash", "criteria"])
def test_import_mismatch_cannot_be_hidden_by_other_compatible_evidence(imports_setup, mutation):
    root, _, _, target, _ = imports_setup
    binding = target["diagnostic_imports"][0]
    directory = root / "reports/forge/attempts" / binding["attempt_id"]
    if mutation in {"source", "seed", "task", "candidate", "registration"}:
        path = directory / "request.json"
        saved = read_json(path)
        request = saved["request"]
        if mutation == "source":
            request["source"]["digest"] = "0" * 64
        elif mutation == "seed":
            request["protocol"]["seed"] = 1729
        elif mutation == "task":
            request["tasks"]["quality"]["execution"]["steps"] += 1
        elif mutation == "candidate":
            request["candidate_revision"] = "0" * 64
        else:
            request["calibration_lane"]["registration_sha256"] = "0" * 64
        atomic_json(path, saved)
    elif mutation == "missing":
        (directory / "result.json").unlink()
    elif mutation in {"diagnostic_key", "qualification_key"}:
        binding["tasks"]["quality"][mutation] = "0" * 64
    elif mutation == "request_hash":
        binding["files"]["request.json"] = "0" * 64
    elif mutation == "criteria":
        path = root / "configs/forge/calibration/changed-criteria.json"
        criteria = read_json(root / "configs/forge/calibration/criteria.json")
        criteria["maximum_false_accept_fraction"] = .2
        atomic_json(path, criteria)
        target.update(criteria=path.name, criteria_sha256=file_hash(path))
    else:
        path = directory / ("result.json" if mutation == "result" else "evidence.json")
        value = read_json(path)
        value["tampered"] = True
        atomic_json(path, value)
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    normal = resolve_idea(root, "negative1", through_tier=3, execution_backend="cpu")
    save_attempt(root, normal, "other-reference", tasks=["quality"], score=0.)
    report = c.calibrate(root, "alternate")
    assert report["adoption"] == "BLOCKED"
    assert negative(report)["conflicts"]


def test_import_binding_cannot_omit_an_observed_failure_or_its_cost(imports_setup):
    root, _, request, target, _ = imports_setup
    save_attempt(root, request, "another-paid-outcome", tasks=["quality"], score=1., cost=2.)
    report = c.calibrate(root, "alternate")
    assert report["adoption"] == "BLOCKED"
    assert negative(report)["reference"]["cost"]["wall_seconds"] == 3.
    bindings = c.diagnostic_imports(root, "original-lane", {"negative1": ["quality"]})
    assert len(bindings) == 2
    target["diagnostic_imports"] = bindings
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    report = c.calibrate(root, "alternate")
    assert negative(report)["reference"]["decision"] == "UNKNOWN"  # conflicting original outcomes
    assert negative(report)["reference"]["cost"]["wall_seconds"] == 3.


def test_imports_preserve_certified_retry_history_and_both_paid_attempts(imports_setup):
    from test_forge_calibration import current_attempt
    root, _, request, target, _ = imports_setup
    identity = target["diagnostic_imports"][0]["attempt_id"]
    previous = current_attempt(root, request, identity, status="INCOMPLETE", raw_status="timeout",
                               costs={"quality": 2.}, omit=("cheap", "alternate"))
    retry = {"attempt_id": identity, "result_hash": stable_hash(previous), "reason": "Repair fixture worker",
             "authorized_at": "2026-09-28T00:00:00Z"}
    current_attempt(root, request, "repaired-import", score=0., retry=retry, costs={"quality": 1.},
                    omit=("cheap", "alternate"))
    target["diagnostic_imports"] = c.diagnostic_imports(root, "original-lane", {"negative1": ["quality"]})
    assert len(target["diagnostic_imports"]) == 2
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    report = c.calibrate(root, "alternate")
    assert report["adoption"] == "PASS"
    assert negative(report)["reference"]["cost"]["wall_seconds"] == 3.
    assert negative(report)["reference"]["decision"] == "FAIL"
    target["diagnostic_imports"] = [row for row in target["diagnostic_imports"] if row["attempt_id"] == "repaired-import"]
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    report = c.calibrate(root, "alternate")
    assert report["adoption"] == "BLOCKED" and negative(report)["conflicts"]
    assert negative(report)["reference"]["cost"]["wall_seconds"] == 3.


def test_import_requires_full_available_target_cohort_not_only_reference_task(imports_setup):
    root, _, _, target, _ = imports_setup
    target["cohort"]["identity"]["tasks"]["alternate"]["execution"] = "0" * 64
    target["cohort"]["sha256"] = stable_hash(target["cohort"]["identity"])
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    report = c.calibrate(root, "alternate")
    assert report["adoption"] == "BLOCKED"
    assert any("complete scientific cohort" in problem for problem in negative(report)["conflicts"])


def test_import_builder_is_read_only_and_refuses_unmeasured_or_unregistered_cells(imports_setup):
    root, _, _, target, _ = imports_setup
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert c.diagnostic_imports(root, "original-lane", {"negative1": ["quality"]}) == target["diagnostic_imports"]
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="original registered"):
        c.diagnostic_imports(root, "original-lane", {"negative1": ["alternate"]})
    (root / "reports/forge/attempts" / target["diagnostic_imports"][0]["attempt_id"] / "result.json").unlink()
    with pytest.raises(ValueError, match="already-recorded"):
        c.diagnostic_imports(root, "original-lane", {"negative1": ["quality"]})


@pytest.mark.parametrize("mutation", ["duplicate", "candidate", "task", "qualification", "missing_hash"])
def test_import_schema_rejects_ambiguous_or_undeclared_authority(imports_setup, mutation):
    root, _, _, target, _ = imports_setup
    binding = target["diagnostic_imports"][0]
    if mutation == "duplicate":
        target["diagnostic_imports"].append(deepcopy(binding))
    elif mutation == "candidate":
        binding["candidate_revision"] = "0" * 64
    elif mutation == "task":
        binding["tasks"]["undeclared"] = binding["tasks"].pop("quality")
    elif mutation == "qualification":
        binding["qualification_reuse"] = True
    else:
        binding["files"].pop("evidence.json")
    atomic_json(root / "configs/forge/calibration/alternate.json", target)
    with pytest.raises(ValueError, match="diagnostic import"):
        c.calibrate(root, "alternate")
