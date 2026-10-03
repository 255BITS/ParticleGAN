"""Bounded diagnostic registrations and forged-lane rejection; no training."""
from copy import deepcopy

import pytest

from experiments.forge import calibration_lane as lane
from experiments.forge import calibration, views
from experiments.forge.api import CapabilityError
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.planning import resolve_idea
from experiments.forge.promotion import validate_screening_submission
from test_forge_promotion import setup as promotion_setup, save_attempt
from forge_legacy_fixtures import pin_legacy


@pytest.fixture
def setup(promotion_setup):
    root, _, _, _ = promotion_setup
    profile = root / "configs/forge/calibration/current-fixture.json"
    contract = {"schema_version": 1, "id": "diagnose-negative", "profile": "current-fixture",
        "profile_sha256": file_hash(profile), "view": "stability",
        "selections": [{"lineage_id": "negative1", "tasks": ["quality"],
                        "reason": "Measure independent reference after this screen rejection."}],
        "budgets": {"task_seconds": {"quality": 10}, "candidate_seconds": 10, "campaign_seconds": 10},
        "execution_backend": "cpu", "cuda_model": None,
        "purpose": "Estimate whether the cheap screen rejected a useful candidate.",
        "failure_policy": "continue_registered_diagnostics", "qualification_reuse": False}
    path = root / "lane-contract.json"
    atomic_json(path, contract)
    return root, path, contract


def registered(setup, freeze=False):
    root, path, _ = setup
    artifact = lane.register(root, path)
    requests = lane.plan_calibration(root, artifact["registration_id"], root / "queue", freeze_source=freeze)
    return artifact, requests


def test_lane_registers_only_explicit_bounded_diagnostics_without_queue_side_effects(setup):
    root, path, _ = setup
    artifact, requests = registered(setup)
    assert artifact == lane.register(root, path)
    assert artifact["maximum_reserved_seconds"] == 10
    assert len(requests) == 1 and requests[0]["candidate"]["id"] == "negative1"
    assert requests[0]["view"]["assignments"] == [
        {"task": "quality", "qualification_tier": 1, "importance": "diagnostic", "order": 0}]
    assert requests[0]["protocol"]["seed"] == 0
    assert not (root / "queue").exists()
    verdict = views.qualify(requests[0]["view"], requests[0]["tasks"], [], candidate=requests[0]["candidate"])
    assert verdict["status"] == "DIAGNOSTIC" and not verdict["eligible"] and verdict["qualified_tier"] == 0


def test_diagnostic_keys_preserve_measurement_cohort_but_never_ordinary_qualification_reuse(setup):
    root, _, _ = setup
    _, requests = registered(setup)
    request = requests[0]
    ordinary = resolve_idea(root, "negative1", through_tier=3, execution_backend="cpu")
    assert calibration.calibration_cohort(request, ["cheap", "quality"]) == calibration.calibration_cohort(
        ordinary, ["cheap", "quality"])
    for diagnostic, normal in zip(request["jobs"], ordinary["jobs"]):
        assert diagnostic["compatibility_key"] != normal["compatibility_key"]
        assert diagnostic["qualification_compatibility_key"] == normal["compatibility_key"]
        assert diagnostic["science"] == {**normal["science"], "evidence_use": "calibration_diagnostic"}


def test_read_only_preview_and_explicit_snapshot_freezing(setup):
    root, _, _ = setup
    artifact, requests = registered(setup)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert requests == lane.plan_calibration(root, artifact["registration_id"], root / "queue")
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    lane.verify_request(root, requests[0])
    with pytest.raises(CapabilityError, match="snapshot"):
        lane.validate_submission(root, requests[0])
    frozen = lane.plan_calibration(root, artifact["registration_id"], root / "queue", freeze_source=True)[0]
    assert lane.validate_submission(root, frozen) == frozen["calibration_campaign"]


def test_origin_only_advance_keeps_lane_current_and_registration_idempotent(setup, monkeypatch):
    root, path, _ = setup
    artifact, requests = registered(setup, freeze=True)
    request = requests[0]
    registration = root / "reports/forge/calibration-lanes" / artifact["registration_id"] / "registration.json"
    original_bytes = registration.read_bytes()
    inspect = lane.planning.inspect_source
    monkeypatch.setattr(lane.planning, "inspect_source", lambda *a, **kw:
                        {**inspect(*a, **kw), "origin_commit": "a" * 40})
    (root / "DOCS_ONLY.md").write_text("Documentation changed; execution inputs did not.\n")
    fresh = resolve_idea(root, "negative1", through_tier=3, execution_backend="cpu")
    assert fresh["source"]["origin_commit"] != request["source"]["origin_commit"]
    assert fresh["source"]["digest"] == request["source"]["digest"]
    assert fresh["source"]["files"] == request["source"]["files"]
    assert lane.plan_calibration(root, artifact["registration_id"], root / "queue", freeze_source=True) == requests
    assert lane.validate_submission(root, request) == request["calibration_campaign"]
    assert lane.register(root, path) == artifact
    assert registration.read_bytes() == original_bytes

    # The exception is for comparing fresh inputs, never for rewriting archived
    # provenance or authorizing changed science under the previous registration.
    forged = deepcopy(request)
    forged["source"]["origin_commit"] = fresh["source"]["origin_commit"]
    with pytest.raises(CapabilityError, match="exact frozen"):
        lane.verify_request(root, forged)
    (root / "particlegan/mechanism.py").write_text("fixture = 2\n")
    for action in (lambda: lane.plan_calibration(root, artifact["registration_id"], root / "queue"),
                   lambda: lane.validate_submission(root, request), lambda: lane.register(root, path)):
        with pytest.raises(CapabilityError, match="pinned profile revision|frozen cohort"):
            action()
    assert registration.read_bytes() == original_bytes


def test_origin_exception_preserves_registration_hash_and_snapshot_checks(setup, monkeypatch):
    from pathlib import Path
    root, _, _ = setup
    artifact, requests = registered(setup, freeze=True)
    request = requests[0]
    inspect = lane.planning.inspect_source
    monkeypatch.setattr(lane.planning, "inspect_source", lambda *a, **kw:
                        {**inspect(*a, **kw), "origin_commit": "b" * 40})
    member = Path(request["source"]["snapshot_path"]) / "particlegan/mechanism.py"
    member.write_text("fixture = 'tampered snapshot'\n")
    with pytest.raises(CapabilityError, match="snapshot changed"):
        lane.validate_submission(root, request)
    registration = root / "reports/forge/calibration-lanes" / artifact["registration_id"] / "registration.json"
    tampered = read_json(registration)
    tampered["subjects"]["negative1"]["base_request"]["source"]["origin_commit"] = "c" * 40
    atomic_json(registration, tampered)
    with pytest.raises(CapabilityError, match="content hash"):
        lane.plan_calibration(root, artifact["registration_id"], root / "queue")


@pytest.mark.parametrize("mutation", ["seed", "lineage", "task", "task_budget", "campaign_budget",
                                      "candidate_budget", "qualify", "implicit_continue", "profile"])
def test_contract_rejects_sweeps_undeclared_cells_unbounded_work_and_qualification(setup, mutation):
    root, path, contract = setup
    if mutation == "seed":
        contract["seed"] = 1729
    elif mutation == "lineage":
        contract["selections"][0]["lineage_id"] = "new-unregistered-candidate"
    elif mutation == "task":
        contract["selections"][0]["tasks"] = ["unregistered"]
        contract["budgets"]["task_seconds"] = {"unregistered": 10}
    elif mutation == "task_budget":
        contract["budgets"]["task_seconds"]["quality"] = 1
    elif mutation == "campaign_budget":
        contract["budgets"]["campaign_seconds"] = 1
    elif mutation == "candidate_budget":
        contract["budgets"]["candidate_seconds"] = 1
    elif mutation == "qualify":
        contract["qualification_reuse"] = True
    elif mutation == "implicit_continue":
        contract["failure_policy"] = "auto_expand_after_every_rejection"
    else:
        contract["profile_sha256"] = "0" * 64
    atomic_json(path, contract)
    with pytest.raises((CapabilityError, ValueError)):
        lane.register(root, path)
    assert not (root / "reports/forge/calibration-lanes").exists()


@pytest.mark.parametrize("mutation", ["source", "source_schema", "source_manifest", "runtime", "recipe",
                                      "criteria", "resource", "task", "view"])
def test_submit_rejects_in_stage_tuning_even_with_unchanged_saved_request(setup, mutation, monkeypatch):
    root, _, _ = setup
    _, requests = registered(setup, freeze=True)
    if mutation == "source":
        (root / "particlegan/mechanism.py").write_text("fixture = 2\n")
    elif mutation in {"source_schema", "source_manifest"}:
        inspect = lane.planning.inspect_source
        def changed_source(*args, **kwargs):
            value = inspect(*args, **kwargs)
            if mutation == "source_schema":
                value["schema_version"] += 1
            else:
                value["files"]["particlegan/mechanism.py"] = "0" * 64
            return value
        monkeypatch.setattr(lane.planning, "inspect_source", changed_source)
    elif mutation == "runtime":
        manifest = lane.planning.runtime_manifest
        monkeypatch.setattr(lane.planning, "runtime_manifest", lambda: {**manifest(), "python": "changed"})
    else:
        paths = {"recipe": "configs/forge/ideas/negative1.json", "criteria": "configs/forge/calibration/criteria.json",
                 "resource": "configs/forge/tasks/quality.json", "task": "configs/forge/tasks/quality.json",
                 "view": "configs/forge/views/stability.json"}
        path = root / paths[mutation]
        value = read_json(path)
        if mutation == "recipe":
            value["recipe_overrides"]["reg_anchor_weight"] = 19.
        elif mutation == "criteria":
            value["maximum_false_accept_fraction"] = 1.
        elif mutation == "task":
            value["evaluation"]["thresholds"][0][2] = .5
        elif mutation == "view":
            value["revision"] += 1
        else:
            value["resources"]["timeout_seconds"] = 20
        atomic_json(path, value)
    # Archived exact evidence remains identifiable, but no new execution can
    # hide a current source/recipe/criteria/resource change.
    assert lane.verify_request(root, requests[0])["qualification_reuse"] is False
    with pytest.raises((CapabilityError, ValueError)):
        lane.validate_submission(root, requests[0])


@pytest.mark.parametrize("mutation", ["seed", "recipe", "scoring", "budget", "selection", "registration", "namespace"])
def test_forged_diagnostic_request_is_rejected(setup, mutation):
    root, _, _ = setup
    _, requests = registered(setup, freeze=True)
    request = deepcopy(requests[0])
    if mutation == "seed":
        request["protocol"]["seed"] = 1729
    elif mutation == "recipe":
        request["candidate"]["recipe_overrides"]["reg_anchor_weight"] = 19.
    elif mutation == "scoring":
        request["tasks"]["quality"]["evaluation"]["scoring_weights"] = "ema"
    elif mutation == "budget":
        request["calibration_campaign"]["budget_seconds"] *= 2
    elif mutation == "selection":
        request["view"]["assignments"].append({"task": "cheap", "qualification_tier": 1,
                                               "importance": "diagnostic", "order": 1})
    elif mutation == "registration":
        request["calibration_lane"]["registration_sha256"] = "0" * 64
    else:
        request["jobs"][0]["compatibility_key"] = request["jobs"][0]["qualification_compatibility_key"]
    with pytest.raises((CapabilityError, ValueError)):
        lane.validate_submission(root, request)


def test_stripping_registration_does_not_turn_diagnostics_into_ordinary_screening(setup):
    _, requests = registered(setup)
    request = requests[0]
    request.pop("calibration_lane")
    with pytest.raises(CapabilityError):
        validate_screening_submission(request)
    request.pop("calibration_campaign")
    request["view"].pop("evidence_scope")
    with pytest.raises(CapabilityError):
        validate_screening_submission(request)


def test_registration_cannot_be_rewritten_or_expand_tasks_after_execution(setup):
    root, path, contract = setup
    registered(setup)
    contract["selections"][0]["tasks"].append("cheap")
    contract["budgets"].update(task_seconds={"cheap": 10, "quality": 10}, candidate_seconds=20, campaign_seconds=20)
    atomic_json(path, contract)
    with pytest.raises(CapabilityError, match="immutable"):
        lane.register(root, path)


def test_selected_diagnostics_cannot_omit_declared_dependency(setup):
    root, path, contract = setup
    task_path = root / "configs/forge/tasks/quality.json"
    task = read_json(task_path)
    task["dependencies"] = [{"task": "cheap", "kind": "gate"}]
    atomic_json(task_path, task)
    request = resolve_idea(root, "negative1", through_tier=3, execution_backend="cpu")
    profile_path = root / "configs/forge/calibration/current-fixture.json"
    profile = read_json(profile_path)
    profile["cohort"] = calibration.calibration_cohort(request, ["cheap", "quality"])
    atomic_json(profile_path, profile)
    contract["profile_sha256"] = file_hash(profile_path)
    atomic_json(path, contract)
    with pytest.raises(CapabilityError, match="dependency"):
        lane.register(root, path)
    contract["selections"][0]["tasks"] = ["cheap", "quality"]
    contract["budgets"].update(task_seconds={"cheap": 10, "quality": 10}, candidate_seconds=20, campaign_seconds=20)
    atomic_json(path, contract)
    _, requests = registered(setup)
    jobs = {job["task_id"]: job for job in requests[0]["jobs"]}
    assert jobs["quality"]["science"]["prerequisites"]["cheap"] == jobs["cheap"]["compatibility_key"]
    assert jobs["quality"]["science"]["prerequisites"]["cheap"] != jobs["cheap"]["qualification_compatibility_key"]


def test_calibrator_ingests_registered_diagnostics_and_rejects_forged_lane(setup):
    root, _, _ = setup
    _, requests = registered(setup)
    request = requests[0]
    # Replace the selected reference receipt in this software fixture with its
    # registered diagnostic equivalent; keep the ordinary smoke rejection.
    normal = resolve_idea(root, "negative1", through_tier=3, execution_backend="cpu")
    save_attempt(root, normal, "calibration-negative1", score=0., cost=.05, tasks=["cheap"])
    save_attempt(root, request, "diagnostic-reference", score=0., cost=1., tasks=["quality"])
    report = calibration.calibrate(root, "current-fixture")
    assert report["adoption"] == "PASS"
    path = root / "reports/forge/attempts/diagnostic-reference/request.json"
    saved = read_json(path)
    saved["request"]["calibration_lane"]["registration_sha256"] = "0" * 64
    atomic_json(path, saved)
    report = calibration.calibrate(root, "current-fixture")
    assert report["adoption"] == "BLOCKED"
    negative = next(row for row in report["matrix"] if row["id"] == "negative1")
    assert negative["reference"]["task_statuses"]["quality"] == "INVALID"
    assert negative["reference"]["cost"]["wall_seconds"] == 1.


def test_queue_runs_only_registered_diagnostics_after_smoke_failure_and_retains_budget(setup):
    from pathlib import Path
    from experiments.forge.queue import Queue
    from test_forge_queue import SLOTS
    root, path, contract = setup
    contract["selections"][0]["tasks"] = ["cheap", "quality"]
    contract["budgets"].update(task_seconds={"cheap": 10, "quality": 10}, candidate_seconds=20, campaign_seconds=20)
    atomic_json(path, contract)
    _, requests = registered(setup, freeze=True)
    request = requests[0]
    queue = Queue(root / "queue", report_root=root / "reports/forge")
    entry = queue.submit(request, request["calibration_campaign"])
    for name, score in (("cheap", 0.), ("quality", 1.)):
        claim = queue.claim(SLOTS)
        assert claim["job"]["task_id"] == name
        raw = {"evidence": {"observations": [{"step": i, "score": score} for i in range(1, 25)],
                            "live": {"score": score}, "scoring_weights": "live"}}
        grade = views.grade_result(request["tasks"][name], raw)
        atomic_json(Path(claim["worker"]["directory"]) / "terminal.json", {
            "token": claim["worker"]["token"], "attempt_status": "completed", "result": raw,
            "elapsed_seconds": 1., "grading": {"raw_hash": stable_hash(raw),
                "source_digest": request["source"]["digest"], "grades": {name: grade}}})
        queue.collect()
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert state["submissions"][entry["request"]["request_id"]]["status"] == "completed"
    assert state["campaigns"][request["calibration_campaign"]["id"]]["spent_seconds"] == 2.
    # The ordinary lane has distinct keys and cannot inherit diagnostic PASS.
    pin_legacy(root)
    ordinary = resolve_idea(root, "negative1", through_tier=3, queue_root=root / "queue",
                            execution_backend="cpu", freeze_source=True)
    queue.submit(ordinary, {"id": "ordinary", "budget_seconds": 20, "candidate_budget_seconds": 20})
    assert queue.claim(SLOTS)["job"]["task_id"] == "cheap"


def test_queue_requires_registered_lane_repository_binding_and_frozen_campaign(setup):
    from experiments.forge.queue import Queue
    root, _, _ = setup
    _, requests = registered(setup, freeze=True)
    request = requests[0]
    with pytest.raises(ValueError, match="repository-bound"):
        Queue(root / "unbound").submit(request, request["calibration_campaign"])
    altered = {**request["calibration_campaign"], "budget_seconds": 1000}
    with pytest.raises(ValueError, match="campaign"):
        Queue(root / "forged", report_root=root / "reports/forge").submit(request, altered)
    assert not (root / "forged/queue/state.json").exists()
