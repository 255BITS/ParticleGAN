"""Saved-evidence calibration tests; no model or optimizer is executed."""
import gzip
import json
from hashlib import sha256
from pathlib import Path
import shutil
from copy import deepcopy

import pytest

from experiments.forge import calibration as c
from experiments.forge.contracts import atomic_json, read_json, file_hash, stable_hash

ROOT = Path(__file__).resolve().parents[1]


def task(status, seconds=None):
    return {"gate_status": status, "calibration_cost": {"wall_seconds": seconds}}


def test_failure_requires_a_scientific_fail_and_unknown_cost_is_not_zero():
    blocked = c._decision({"a": task("BLOCKED"), "b": task("INCOMPLETE")}, ["a", "b", "c"])
    assert blocked["decision"] == "UNKNOWN"
    assert blocked["blocked_tasks"] == ["a"]
    assert blocked["error_or_incomplete_tasks"] == ["b"]
    assert blocked["cost"]["wall_seconds"] is None
    failed = c._decision({"a": task("FAIL", 3), "b": task("BLOCKED")}, ["a", "b"])
    assert failed["decision"] == "FAIL"
    assert failed["cost"]["known_wall_seconds"] == 3
    assert failed["cost"]["missing_tasks"] == ["b"]


def test_error_denominators_exclude_unknown_without_hiding_it():
    def row(smoke, ref, classification):
        return {"smoke": c._decision({"x": task(smoke, 2)}, ["x"]),
                "reference": c._decision({"y": task(ref, 100)}, ["y"]), "classification": classification}
    rows = [row("PASS", "FAIL", "false_accept"), row("FAIL", "FAIL", "true_reject"),
            row("PASS", "PASS", "true_accept"), row("BLOCKED", "FAIL", "unknown")]
    stats = c._stats(rows)
    assert stats["false_accept_fraction"] == .5
    assert stats["false_reject_fraction"] == 0
    assert stats["all_reference_negatives"] == 3
    assert stats["reference_negatives"] == 2
    assert stats["unknown"] == stats["blocked_lineages"] == 1
    assert stats["paired_fraction"] == .75


def test_reference_cannot_reuse_a_smoke_predicate(tmp_path):
    config_dir = tmp_path / "configs/forge/calibration"
    config_dir.mkdir(parents=True)
    (config_dir / "bad.json").write_text(json.dumps({"criteria": "criteria.json", "smoke_tasks": ["a"], "reference_tasks": ["a"]}))
    (config_dir / "criteria.json").write_text("{}")
    with pytest.raises(ValueError, match="smoke predicates"):
        c.calibrate(tmp_path, "bad")


def test_snapshot_cost_requires_both_compressed_and_decoded_hash(monkeypatch, tmp_path):
    payload = b'{"seconds": 7.5}'
    raw = gzip.compress(payload, mtime=0)

    class Source:
        def __init__(self, *args):
            pass

        def read(self, path):
            return raw

    monkeypatch.setattr(c, "Source", Source)
    recorded = {"raw_result": {"snapshot_receipt": {"path": "r.json.gz", "revision": "frozen", "sha256": sha256(raw).hexdigest()},
                               "artifact_sha256": sha256(payload).hexdigest()}}
    assert c._task_cost(tmp_path, recorded, {})["wall_seconds"] == 7.5
    recorded["raw_result"]["artifact_sha256"] = "wrong"
    result = c._task_cost(tmp_path, recorded, {})
    assert result["wall_seconds"] is None
    assert "decoded artifact hash mismatch" in result["gap"]


def test_package_conflict_invalidates_a_lineage(tmp_path):
    spec = {"id": "q", "candidate_id": "pr217_qr_native_adapter", "source_path": "p",
            "prior_cohort": "cloud", "runtime_cohort": "cuda"}
    card = {"candidate_id": spec["candidate_id"], "candidate_revision": "rev", "record_id": "one",
            "source": {"path": "p", "revision": "commit"}, "provenance": {"package_sha256": "wrong"},
            "task_results": [{"task_id": "a", "gate_status": "PASS"}, {"task_id": "b", "gate_status": "PASS"}]}
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card))
    row = c._lineage(tmp_path, spec, [(path, card)], {"smoke_tasks": ["a"], "reference_tasks": ["b"]}, {})
    assert row["classification"] == "unknown"
    assert row["smoke"]["task_statuses"] == {"a": "INVALID"}
    assert row["conflicts"]


@pytest.fixture
def saved_replay(tmp_path, monkeypatch):
    # Keep output away from tracked reports, while reading only immutable git blobs.
    profiles = ROOT / "configs/forge/calibration"
    if not profiles.exists() or not (ROOT / "reports/forge/records").exists():
        pytest.skip("historical fixtures unavailable")
    destination = tmp_path / "configs/forge/calibration"
    destination.mkdir(parents=True)
    for name in ("initial.json", "quick-discriminator.json", "criteria-v1.json"):
        shutil.copyfile(profiles / name, destination / name)
    specs = json.loads((profiles / "initial.json").read_text())["lineages"]
    selector = {(r["candidate_id"], r["source_path"]) for r in specs}
    records = tmp_path / "reports/forge/records"
    records.mkdir(parents=True)
    for path in (ROOT / "reports/forge/records").glob("history-*.json"):
        record = json.loads(path.read_text())
        if (record.get("candidate_id"), record.get("source", {}).get("path")) in selector:
            shutil.copyfile(path, records / path.name)
    source = c.Source
    try:
        source(ROOT, c.FOLLOWUP_REVISION, (c.LRFREE_ROOT,))
    except Exception:
        pytest.skip("pinned historical git objects unavailable")
    monkeypatch.setattr(c, "Source", lambda _, revision, prefixes: source(ROOT, revision, prefixes))
    return tmp_path


def test_initial_replay_reports_known_false_accept_and_measured_cost(saved_replay):
    result = c.calibrate(saved_replay)
    stats = result["descriptive_totals_not_pooled_adoption"]
    assert stats["paired"] == 6 and stats["unknown"] == 4
    assert stats["false_accept_fraction"] == pytest.approx(1 / 3)
    assert stats["false_reject"] == 0
    rows = {r["id"]: r for r in result["matrix"]}
    assert rows["gapfill-rg5-bcap"]["classification"] == "false_accept"
    assert rows["gapfill-k3p"]["smoke"]["cost"]["wall_seconds"] == pytest.approx(19.33470469713211)
    assert rows["row-em-renew-all22"]["smoke"]["decision"] == "UNKNOWN"
    assert result["adoption"] == "BLOCKED" and result["training_seconds_spent"] == 0
    assert c.calibrate(saved_replay) == result


def test_alternate_does_not_erase_pr217_native_failure(saved_replay):
    result = c.calibrate(saved_replay, "quick-discriminator")
    rows = {r["id"]: r for r in result["matrix"]}
    assert rows["gapfill-rg5-bcap"]["classification"] == "true_reject"
    assert rows["pr217_qr_native_adapter"]["classification"] == "false_accept"
    assert {"grid100", "rotated100", "staggered100"} <= set(rows["pr217_qr_native_adapter"]["reference"]["failed_tasks"])
    assert result["descriptive_totals_not_pooled_adoption"]["false_accept_fraction"] == .2
    assert result["descriptive_totals_not_pooled_adoption"]["paired_fraction"] == .8
    assert result["adoption"] == "BLOCKED"
    assert any(r["prior_cohort"] == "new_learned_mog_sigma_0.025" and r["stats"]["lineages"] == 0 for r in result["cohorts"])


def test_later_host_correction_and_bounded_proposal_remain_explicit(saved_replay):
    c.calibrate(saved_replay)
    base = saved_replay / "reports/forge/calibration"
    audit = json.loads((base / "followup-source-audit.json").read_text())
    new = audit["new_distinct_lineage"]
    assert new["diagnostic_host"]["native_passes"] == 3
    assert new["canonical_host"]["status"] == "FAIL"
    assert new["canonical_host"]["terminal_checks_passed"] == 4
    assert len(audit["unbound_user_reported_claims"]) == 3
    proposal = json.loads((base / "missing-runs.json").read_text())
    assert proposal["maximum_training_wall_seconds"] == 1800
    assert proposal["maximum_jobs"] == 6 and proposal["training_allowance_seconds"] == 0
    assert all(run["status"] == "BLOCKED" and run["package_sha256"] and run["fixture_parameter_sha256"] is None for run in proposal["runs"])
    template = json.loads((saved_replay / "reports/forge/promotion-contract-template.json").read_text())
    assert template["authoritative_template"] == "configs/forge/promotion-template.json"
    assert template["execution_authorized"] is template["is_candidate_claim"] is False


def current_request(name):
    prior = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
    tasks = {name: {"schema_version": 1, "id": name, "adapter": "fixture",
        "execution": {"initializer": "deterministic_orthogonal", "steps": 24, "prior": prior, "fixture_id": name}, "dependencies": [], "requires_capabilities": [],
        "evaluation": {"kind": "transfer_sustained", "thresholds": [["score", ">=", 1.]], "scoring_weights": "live"}}
        for name in ("cheap", "quality")}
    return {"candidate": {"id": name, "prior": prior, "recipe_overrides": {"mechanism": name}},
        "candidate_revision": stable_hash(name), "source": {"digest": stable_hash("source")},
        "protocol": {"seed": 0, "rng": {"version": "fixture"}}, "rng": {"data": 1, "prior": 2},
        "runtime": {"backend": "cpu", "threads": 1}, "tasks": tasks,
        "view": {"assignments": [{"task": "cheap", "qualification_tier": 1, "importance": "required"},
                                 {"task": "quality", "qualification_tier": 2, "importance": "required"}]},
        "jobs": [{"task_id": task, "compatibility_key": stable_hash((name, task)),
                  "science": {"compute": {"backend": "cpu", "threads": 1}}} for task in tasks]}


def current_attempt(root, request, name, score=1., *, status="PASS", costs=None, omit=(), retry=None, raw_status=None):
    from experiments.forge.sampling import executed_receipt, PUBLIC_PRIOR_CLEAN
    costs = costs or {"cheap": .05, "quality": 1.}
    rows = [{"task_id": job["task_id"], "compatibility_key": job["compatibility_key"], "gate_status": status,
             "cost": {"wall_seconds": costs.get(job["task_id"])},
             "evidence": {"observations": [{"step": i, "score": score} for i in range(1, 25)],
                          "live": {"score": score}, "scoring_weights": "live",
                          **(executed_receipt(PUBLIC_PRIOR_CLEAN, eval_output_noise="clean")
                             if "sampling_contract_version" in request["tasks"][job["task_id"]]["evaluation"] else {})}}
            for job in request["jobs"] if job["task_id"] not in omit]
    result = {"attempt_id": name, "candidate_revision": request["candidate_revision"], "task_results": rows}
    resolved = {"request": request}
    if retry:
        result["retry_of"] = resolved["retry_of"] = retry
    if raw_status:
        result["raw"] = {"attempt_status": raw_status}
    directory = root / "reports/forge/attempts" / name
    atomic_json(directory / "request.json", resolved)
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result),
        "source": request["source"], "runtime": request["runtime"]})
    return result


@pytest.fixture
def current(tmp_path):
    requests = [current_request(name) for name in ("positive", "negative1", "negative2")]
    for index, request in enumerate(requests):
        # Both negative controls carry a counterfeit PASS stamp; raw curves fail.
        current_attempt(tmp_path, request, request["candidate"]["id"], score=1. if index == 0 else 0.)
    directory = tmp_path / "configs/forge/calibration"
    criteria_path = directory / "criteria.json"
    atomic_json(criteria_path, read_json(ROOT / "configs/forge/calibration/criteria-v1.json"))
    profile = {"schema_version": 1, "id": "current", "revision": 1, "evidence_scope": "current",
        "criteria": "criteria.json", "criteria_sha256": file_hash(criteria_path), "smoke_tasks": ["cheap"],
        "reference_tasks": ["quality"], "reference_scope": "Independent fixed quality fixture",
        "scoring_weights": "live", "training_allowance_seconds": 0,
        "lineages": [{"id": r["candidate"]["id"], "candidate_id": r["candidate"]["id"],
                      "candidate_revision": r["candidate_revision"]} for r in requests],
        "cohort": c.calibration_cohort(requests[0], ["cheap", "quality"])}
    atomic_json(directory / "current.json", profile)
    return tmp_path, profile, requests


def bind_current(root, request):
    report = c.calibrate(root, "current")
    path = root / "reports/forge/calibration/current.json"
    request["view"]["calibration"] = {"status": "accepted", "report": str(path.relative_to(root)),
        "report_sha256": file_hash(path), "profile_sha256": report["profile_sha256"],
        "criteria_sha256": report["criteria_sha256"], "cohort_sha256": report["cohorts"][0]["sha256"]}
    return report


def test_current_mog_regrades_controls_and_can_satisfy_frozen_acceptance(current):
    root, _, requests = current
    report = bind_current(root, requests[0])
    assert report["adoption"] == "PASS" and report["complete"]
    assert report["cohorts"][0]["stats"]["reference_negatives"] == 2
    assert [r["classification"] for r in report["matrix"]] == ["true_accept", "true_reject", "true_reject"]
    assert report["training_seconds_spent"] == 0
    assert not report["current_qualification_reuse"]
    assert c.verify_calibration(root, requests[0])["report_sha256"] == file_hash(root / "reports/forge/calibration/current.json")
    assert c.calibrate(root, "current") == report


@pytest.mark.parametrize("problem", ["missing_task", "missing_cost", "certificate", "truncated", "duplicate"])
def test_current_incomplete_or_conflicting_evidence_cannot_adopt(current, problem):
    root, _, requests = current
    request = requests[1]
    if problem == "missing_task":
        current_attempt(root, request, "negative1", score=0., omit=("quality",))
    elif problem == "missing_cost":
        current_attempt(root, request, "negative1", score=0., costs={"cheap": .05, "quality": None})
    elif problem == "certificate":
        (root / "reports/forge/attempts/negative1/evidence.json").unlink()
    elif problem == "truncated":
        directory = root / "reports/forge/attempts/unresolved"
        atomic_json(directory / "request.json", {"request": request})
    else:
        current_attempt(root, request, "conflict", score=1.)
    report = c.calibrate(root, "current")
    assert report["adoption"] == "BLOCKED"
    row = report["matrix"][1]
    if problem == "missing_cost":
        assert row["reference"]["cost"]["wall_seconds"] is None
        assert row["reference"]["cost"]["missing_tasks"] == ["quality"]
    elif problem == "truncated":
        assert report["receipt_issues"][0]["attempt_id"] == "unresolved"
    else:
        assert row["classification"] == "unknown"


@pytest.mark.parametrize("dimension", ["source", "seed", "rng", "runtime", "compute", "prior", "candidate_revision", "task", "evaluator"])
def test_current_incompatible_receipt_stays_unknown_without_cross_cohort_credit(current, dimension):
    root, _, requests = current
    request = deepcopy(requests[1])
    if dimension == "source":
        request["source"]["digest"] = stable_hash("other-source")
    elif dimension == "seed":
        request["protocol"]["seed"] = 1
    elif dimension == "rng":
        request["rng"]["data"] = 99
    elif dimension == "runtime":
        request["runtime"]["backend"] = "cuda"
    elif dimension == "compute":
        request["jobs"][0]["science"]["compute"]["threads"] = 4
    elif dimension == "prior":
        request["candidate"]["prior"]["sigma"] = .03
    elif dimension == "candidate_revision":
        request["candidate_revision"] = stable_hash("other-candidate")
    elif dimension == "task":
        request["tasks"]["quality"]["execution"]["steps"] = 48
    else:
        request["tasks"]["quality"]["evaluation"]["thresholds"][0][2] = .5
    current_attempt(root, request, "negative1", score=0.)
    report = c.calibrate(root, "current")
    assert report["adoption"] == "BLOCKED" and not report["complete"]
    assert report["matrix"][1]["classification"] == "unknown"
    assert report["cohorts"][0]["stats"]["lineages"] == 3


def test_current_cohort_excludes_recipe_knobs_and_policy_but_keeps_initialization(current):
    _, _, requests = current
    cohort = c.calibration_cohort(requests[0], ["cheap", "quality"])
    assert c.calibration_cohort(requests[1], ["quality", "cheap"]) == cohort
    request = deepcopy(requests[0])
    request["view"] = {"revision": 500, "calibration": {"status": "accepted"}}
    assert c.calibration_cohort(request, ["cheap", "quality"]) == cohort
    request["candidate"]["initializer"] = "different"
    assert c.calibration_cohort(request, ["cheap", "quality"]) != cohort


def test_current_verified_repair_preserves_failed_attempt_cost(current):
    root, _, requests = current
    previous = current_attempt(root, requests[1], "negative1", status="INCOMPLETE", raw_status="timeout")
    retry = {"attempt_id": "negative1", "result_hash": stable_hash(previous), "reason": "worker repaired",
             "authorized_at": "2026-09-28T00:00:00Z"}
    current_attempt(root, requests[1], "repaired", score=0., retry=retry)
    report = c.calibrate(root, "current")
    assert report["adoption"] == "PASS"
    row = report["matrix"][1]
    assert row["classification"] == "true_reject"
    assert row["smoke"]["cost"]["wall_seconds"] == .1
    assert row["reference"]["cost"]["wall_seconds"] == 2.
    assert len(row["inputs"]) == 2


def test_grouped_receipt_cost_counted_once_not_once_per_predicate():
    tasks = {name: {"gate_status": "PASS", "calibration_cost": {"wall_seconds": 7.,
              "cost_units": {"shared-attempt": 7.}}} for name in ("hold", "extension")}
    result = c._decision(tasks, list(tasks))
    assert result["cost"]["wall_seconds"] == 7.
    assert result["cost"]["known_task_count"] == 2


@pytest.mark.parametrize("problem", ["cloud", "shared_group", "renamed_measurement", "duplicate_lineage", "changed_criteria", "robustness_seed"])
def test_current_profile_rejects_inherited_or_dependent_reference(current, problem):
    root, profile, _ = current
    if problem == "cloud":
        profile["cohort"]["identity"]["prior"]["kind"] = "particle_cloud"
    elif problem == "shared_group":
        for task in profile["cohort"]["identity"]["tasks"].values():
            task["execution_group"] = "one-run"
    elif problem == "renamed_measurement":
        profile["cohort"]["identity"]["tasks"]["quality"]["execution"] = profile["cohort"]["identity"]["tasks"]["cheap"]["execution"]
    elif problem == "duplicate_lineage":
        profile["lineages"][1]["candidate_revision"] = profile["lineages"][0]["candidate_revision"]
    elif problem == "robustness_seed":
        profile["cohort"]["identity"]["protocol"]["seed"] = 1729
    else:
        criteria_path = root / "configs/forge/calibration/criteria.json"
        criteria = read_json(criteria_path)
        criteria["maximum_false_accept_fraction"] = 1.
        atomic_json(criteria_path, criteria)
    profile["cohort"]["sha256"] = stable_hash(profile["cohort"]["identity"])
    atomic_json(root / "configs/forge/calibration/current.json", profile)
    with pytest.raises(ValueError):
        c.calibrate(root, "current")


@pytest.mark.parametrize("problem", ["report", "profile", "criteria", "receipt", "retier", "source"])
def test_accepted_report_requires_unchanged_complete_evidence_and_actual_screen(current, problem):
    root, _, requests = current
    request = requests[0]
    bind_current(root, request)
    if problem in {"profile", "criteria"}:
        name = "current" if problem == "profile" else "criteria"
        path = root / f"configs/forge/calibration/{name}.json"
        value = read_json(path)
        value["tampered"] = True
        atomic_json(path, value)
    elif problem == "report":
        path = root / "reports/forge/calibration/current.json"
        report = read_json(path)
        report["matrix"][0]["classification"] = "false_accept"
        atomic_json(path, report)
        request["view"]["calibration"]["report_sha256"] = file_hash(path)
    elif problem == "receipt":
        current_attempt(root, requests[1], "negative1", score=1.)
    elif problem == "retier":
        request["view"]["assignments"][1]["qualification_tier"] = 1
    else:
        request["source"]["digest"] = stable_hash("new-source")
    with pytest.raises(ValueError, match="calibration"):
        c.verify_calibration(root, request)


def test_authorized_promotion_or_other_seed_evidence_does_not_change_screen_calibration(current):
    root, _, requests = current
    request = requests[0]
    original = bind_current(root, request)
    for index in range(2):
        promotion = deepcopy(request)
        promotion["protocol"].update(seed=index, promotion_namespace="registered-finished-stage")
        current_attempt(root, promotion, f"promotion-{index}", score=0.)
    assert c.calibrate(root, "current") == original
    assert c.verify_calibration(root, request)


@pytest.mark.parametrize("marker", [False, True, "stripped"])
def test_unregistered_diagnostic_flag_cannot_supply_calibration_evidence(current, marker):
    root, _, requests = current
    request = deepcopy(requests[1])
    request["view"]["evidence_scope"] = "calibration_diagnostic"
    if marker == "stripped":
        request["view"].pop("evidence_scope")
        for job in request["jobs"]:
            job["science"]["evidence_use"] = "calibration_diagnostic"
            job["qualification_compatibility_key"] = job["compatibility_key"]
    elif marker:
        request["calibration_lane"] = {"registration_id": "forged", "lineage_id": "negative1"}
    current_attempt(root, request, "negative1", score=0.)
    report = c.calibrate(root, "current")
    assert report["adoption"] == "BLOCKED"
    assert report["matrix"][1]["classification"] == "unknown"
    assert "invalid diagnostic registration" in report["matrix"][1]["conflicts"][0]
    assert report["matrix"][1]["reference"]["cost"]["wall_seconds"] == 1.
