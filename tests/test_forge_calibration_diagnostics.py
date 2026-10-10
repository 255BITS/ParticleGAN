"""Extra registered measurements cannot manufacture calibration credit.

All receipts are synthetic software fixtures; no training is launched.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge import calibration as c, calibration_lane as lane, views
from experiments.forge.api import CapabilityError
from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.planning import resolve_idea
from test_forge_promotion import setup as promotion_setup, save_attempt


@pytest.fixture
def diagnostics(promotion_setup):
    root, _, _, _ = promotion_setup
    task = read_json(root / "configs/forge/tasks/quality.json")
    task["id"] = task["execution"]["fixture_id"] = "release-cloud"
    task["execution"]["prior"] = {"kind": "particle_cloud", "sigma": 0,
        "standardize": False, "learnable": True,
        "exception_reason": "Explicit release-replay diagnostic; no MoG reference credit."}
    atomic_json(root / "configs/forge/tasks/release-cloud.json", task)
    view_path = root / "configs/forge/views/stability.json"
    view = read_json(view_path)
    view["assignments"].append({"task": "release-cloud", "qualification_tier": 2,
                                "importance": "diagnostic", "order": 1})
    atomic_json(view_path, view)
    profile_path = root / "configs/forge/calibration/current-fixture.json"
    profile = read_json(profile_path)
    requests = {}
    for name, filename, score in (("finished", "qualification", 1.),
                                  ("negative1", "calibration-negative1", 0.),
                                  ("negative2", "calibration-negative2", 0.)):
        request = resolve_idea(root, name, through_tier=3, execution_backend="cpu")
        requests[name] = request
        save_attempt(root, request, filename, score=score, cost={"cheap": .05, "quality": 1.},
                     tasks=["cheap", "quality"])
    profile.update(diagnostic_tasks=["release-cloud"],
                   cohort=c.calibration_cohort(requests["finished"], ["cheap", "quality", "release-cloud"]))
    atomic_json(profile_path, profile)
    contract = {"schema_version": 1, "id": "release-diagnostic", "profile": "current-fixture",
        "profile_sha256": file_hash(profile_path), "view": "stability",
        "selections": [{"lineage_id": "finished", "tasks": ["release-cloud"],
                         "reason": "Measure released cloud law separately from the MoG reference."}],
        "budgets": {"task_seconds": {"release-cloud": 10}, "candidate_seconds": 10, "campaign_seconds": 10},
        "execution_backend": "cpu", "cuda_model": None,
        "purpose": "Paired formulation context without changing either decision denominator.",
        "failure_policy": "continue_registered_diagnostics", "qualification_reuse": False}
    path = root / "release-lane.json"
    atomic_json(path, contract)
    lane.register(root, path)
    diagnostic_request = lane.plan_calibration(root, contract["id"], root / "queue")[0]
    return root, profile, requests, diagnostic_request


def test_failed_expensive_diagnostic_is_separate_from_reference_costs_and_adoption(diagnostics):
    root, _, requests, diagnostic = diagnostics
    save_attempt(root, diagnostic, "release-failure", score=0., cost=100., tasks=["release-cloud"])
    report = c.calibrate(root, "current-fixture")
    assert report["adoption"] == "PASS" and report["complete"]
    assert report["diagnostic_tasks"] == ["release-cloud"]
    assert report["smoke_tasks"] == ["cheap"] and report["reference_tasks"] == ["quality"]
    rows = {row["id"]: row for row in report["matrix"]}
    positive = rows["finished"]
    assert positive["classification"] == "true_accept"
    assert positive["reference"]["task_statuses"] == {"quality": "PASS"}
    assert positive["reference"]["cost"]["wall_seconds"] == 1.
    assert positive["smoke"]["cost"]["wall_seconds"] == .05
    assert positive["diagnostics"]["task_statuses"] == {"release-cloud": "FAIL"}
    assert positive["diagnostics"]["cost"]["wall_seconds"] == 100.
    assert positive["diagnostics"]["reference_credit"] is False
    assert positive["diagnostics"]["current_qualification_reuse"] is False
    assert rows["negative1"]["diagnostics"]["task_statuses"] == {"release-cloud": "NOT_RUN"}
    assert rows["negative1"]["diagnostics"]["cost"]["wall_seconds"] is None
    checks = report["cohorts"][0]["checks"]
    assert checks["maximum_smoke_to_reference_wall_ratio"]["actual"] == .05
    assert report["cohorts"][0]["stats"]["reference_positives"] == 1
    markdown = (root / "reports/forge/calibration/current-fixture.md").read_text()
    assert "Separate diagnostics provide no independent reference or qualification credit" in markdown
    assert "100.000" in markdown

    # Promotion verification must bind the complete diagnostic task identity,
    # even though its scientific FAIL has no calibration decision credit.
    request = requests["finished"]
    report_path = root / "reports/forge/calibration/current-fixture.json"
    request["view"]["calibration"] = {"status": "accepted", "report": str(report_path.relative_to(root)),
        "report_sha256": file_hash(report_path), "profile_sha256": report["profile_sha256"],
        "criteria_sha256": report["criteria_sha256"], "cohort_sha256": report["cohorts"][0]["sha256"]}
    assert c.verify_calibration(root, request)["cohort_sha256"] == report["cohorts"][0]["sha256"]
    changed = deepcopy(request)
    changed["tasks"]["release-cloud"]["execution"]["prior"]["sigma"] = .1
    with pytest.raises(ValueError, match="cohort"):
        c.verify_calibration(root, changed)


def test_missing_reference_cannot_be_replaced_by_passing_diagnostic(diagnostics):
    root, _, requests, diagnostic = diagnostics
    save_attempt(root, requests["finished"], "qualification", score=1., cost=.05, tasks=["cheap"])
    save_attempt(root, diagnostic, "release-pass", score=1., cost=100., tasks=["release-cloud"])
    report = c.calibrate(root, "current-fixture")
    positive = next(row for row in report["matrix"] if row["id"] == "finished")
    assert report["adoption"] == "BLOCKED" and not report["complete"]
    assert positive["reference"]["decision"] == "UNKNOWN"
    assert positive["reference"]["unknown_tasks"] == ["quality"]
    assert positive["reference"]["cost"]["wall_seconds"] is None
    assert positive["diagnostics"]["decision"] == "PASS"
    assert report["cohorts"][0]["stats"]["reference_positives"] == 0


def test_diagnostic_registration_and_import_preserve_exact_nonqualifying_identity(diagnostics):
    root, profile, requests, diagnostic = diagnostics
    lane.verify_request(root, diagnostic)
    qualification = views.qualify(diagnostic["view"], diagnostic["tasks"], [], candidate=diagnostic["candidate"])
    assert qualification["status"] == "DIAGNOSTIC" and qualification["qualified_tier"] == 0
    save_attempt(root, diagnostic, "release-original", score=0., cost=3., tasks=["release-cloud"])
    bindings = c.diagnostic_imports(root, "release-diagnostic", {"finished": ["release-cloud"]})
    assert len(bindings) == 1 and bindings[0]["qualification_reuse"] is False
    alternate = {**profile, "id": "imported-release", "diagnostic_imports": bindings}
    atomic_json(root / "configs/forge/calibration/imported-release.json", alternate)
    report = c.calibrate(root, "imported-release")
    assert report["adoption"] == "PASS"
    row = next(row for row in report["matrix"] if row["id"] == "finished")
    assert row["diagnostics"]["cost"]["wall_seconds"] == 3.
    assert row["reference"]["cost"]["wall_seconds"] == 1.
    assert c.calibration_cohort(requests["finished"], c.profile_task_ids(profile)) == profile["cohort"]

    # No silent diagnostic/scoring tuning after registration or during import.
    path = root / "configs/forge/tasks/release-cloud.json"
    task = read_json(path)
    task["evaluation"]["thresholds"][0][2] = .5
    atomic_json(path, task)
    with pytest.raises(CapabilityError, match="frozen cohort"):
        lane.plan_calibration(root, "release-diagnostic", root / "queue")
    changed = deepcopy(diagnostic)
    changed["tasks"]["release-cloud"] = task
    with pytest.raises(CapabilityError, match="exact frozen"):
        lane.verify_request(root, changed)
    alternate["cohort"] = c.calibration_cohort(changed, c.profile_task_ids(alternate))
    atomic_json(root / "configs/forge/calibration/imported-release.json", alternate)
    report = c.calibrate(root, "imported-release")
    assert report["adoption"] == "BLOCKED"
    row = next(row for row in report["matrix"] if row["id"] == "finished")
    assert row["diagnostics"]["task_statuses"]["release-cloud"] == "INVALID"
    assert row["conflicts"]


@pytest.mark.parametrize("mutation", ["duplicate", "reference", "group", "renamed_measurement", "unbound"])
def test_diagnostics_cannot_alias_reference_or_escape_complete_cohort(diagnostics, mutation):
    _, profile, requests, _ = diagnostics
    config = deepcopy(profile)
    request = deepcopy(requests["finished"])
    if mutation == "duplicate":
        config["diagnostic_tasks"] *= 2
    elif mutation == "reference":
        config["diagnostic_tasks"] = ["quality"]
    elif mutation == "group":
        request["tasks"]["release-cloud"]["execution"]["execution_group"] = "quality"
        config["cohort"] = c.calibration_cohort(request, c.profile_task_ids(config))
    elif mutation == "renamed_measurement":
        request["tasks"]["release-cloud"]["execution"] = deepcopy(request["tasks"]["quality"]["execution"])
        config["cohort"] = c.calibration_cohort(request, c.profile_task_ids(config))
    else:
        config["cohort"] = c.calibration_cohort(request, ["cheap", "quality"])
    criteria = read_json(Path(__file__).resolve().parents[1] / "configs/forge/calibration/criteria-v1.json")
    with pytest.raises(ValueError, match="diagnostic|diagnostics"):
        c._current_profile(config, criteria)


def test_profiles_without_optional_diagnostics_keep_legacy_report_shape(promotion_setup):
    root, _, _, candidate = promotion_setup
    config = read_json(root / "configs/forge/calibration/current-fixture.json")
    assert c.profile_task_ids(config) == ["cheap", "quality"]
    report = c.calibrate(root, "current-fixture")
    assert report["adoption"] == "PASS" and "diagnostic_tasks" not in report
    assert all("diagnostics" not in row for row in report["matrix"])
    assert c.calibration_cohort(candidate, c.profile_task_ids(config)) == config["cohort"]
