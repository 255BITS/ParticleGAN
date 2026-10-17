"""Logical calibration controls; no training or invented reference credit."""
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest

from experiments.forge.calibration_feasibility import decision_feasibility, preflight
from experiments.forge.contracts import read_json, atomic_json
from experiments.forge.__main__ import main
from experiments.forge.api import CapabilityError
from test_forge_calibration_lane import setup as lane_setup
from test_forge_promotion import setup as promotion_setup


ROOT = Path(__file__).resolve().parents[1]
CRITERIA = read_json(ROOT / "configs/forge/calibration/criteria-v1.json")


def matrix(pairs, *, cost=True):
    return [{"id": str(index), "cohort_sha256": "same-frozen-cohort",
             "smoke": {"decision": smoke, "cost": {"wall_seconds": .05 if cost else None}},
             "reference": {"decision": reference, "cost": {"wall_seconds": 1. if cost else None}}}
            for index, (smoke, reference) in enumerate(pairs)]


def test_all_smoke_rejected_cannot_satisfy_positive_minimum_without_false_reject():
    rows = matrix([("FAIL", "FAIL"), ("FAIL", "UNKNOWN"), ("FAIL", "UNKNOWN")])
    before = deepcopy(rows)
    result = decision_feasibility(rows, CRITERIA)
    assert result["status"] == "INFEASIBLE"
    assert result["possible_completion_counts"] is None
    assert result["unknown_decisions"] == 2
    assert rows == before


def test_all_reference_negative_is_infeasible_even_with_unknown_smoke():
    assert decision_feasibility(matrix([("UNKNOWN", "FAIL")] * 3), CRITERIA)["status"] == "INFEASIBLE"


def test_unknown_positive_can_make_profile_possible_without_adoption_or_cost_credit():
    rows = matrix([("UNKNOWN", "UNKNOWN"), ("FAIL", "FAIL"), ("FAIL", "FAIL")], cost=False)
    result = decision_feasibility(rows, CRITERIA)
    assert result["status"] == "POSSIBLE"
    assert result["possible_completion_counts"] == {
        "true_accept": 1, "false_accept": 0, "true_reject": 2, "false_reject": 0}
    assert len(result["missing_cost_vectors"]) == 6
    assert result["qualification_input"] is result["default_adoption"] is result["training_authorized"] is False
    assert rows[0]["smoke"]["decision"] == "UNKNOWN"


def test_declared_lineage_minimum_and_cost_floor_can_prove_impossibility():
    assert decision_feasibility(matrix([("PASS", "PASS"), ("FAIL", "FAIL")]), CRITERIA)["status"] == "INFEASIBLE"
    rows = matrix([("PASS", "PASS"), ("FAIL", "FAIL"), ("FAIL", "FAIL")])
    rows[0]["smoke"]["cost"] = {"wall_seconds": None, "known_wall_seconds": 901.}
    assert "already-paid smoke" in decision_feasibility(rows, CRITERIA)["reasons"][0]
    rows[0]["smoke"]["cost"] = {"wall_seconds": .2}
    assert "cost ratio" in decision_feasibility(rows, CRITERIA)["reasons"][0]


def test_nonzero_error_tolerances_use_correct_reference_denominators():
    criteria = {**CRITERIA, "maximum_false_reject_fraction": .5}
    rows = matrix([("PASS", "PASS"), ("FAIL", "PASS"), ("FAIL", "FAIL"), ("FAIL", "FAIL")])
    assert decision_feasibility(rows, criteria)["status"] == "POSSIBLE"
    criteria["maximum_false_reject_fraction"] = .49
    assert decision_feasibility(rows, criteria)["status"] == "INFEASIBLE"


@pytest.mark.parametrize("error", ["reject", "accept"])
def test_exact_fraction_boundary_uses_same_division_as_final_criteria(error):
    criterion = "maximum_false_" + error + "_fraction"
    pairs = ([("PASS", "PASS")] * 48 + [("FAIL", "PASS")] + [("FAIL", "FAIL")] * 2
             if error == "reject" else
             [("PASS", "PASS")] + [("PASS", "FAIL")] + [("FAIL", "FAIL")] * 48)
    assert decision_feasibility(matrix(pairs), {**CRITERIA, criterion: 1 / 49})["status"] == "POSSIBLE"
    assert decision_feasibility(matrix(pairs), {**CRITERIA, criterion: 1 / 50})["status"] == "INFEASIBLE"


def test_diagnostics_cannot_rescue_reference_negatives():
    rows = matrix([("UNKNOWN", "FAIL")] * 3)
    for row in rows:
        row["diagnostics"] = {"decision": "PASS", "cost": {"wall_seconds": 0.}}
    assert decision_feasibility(rows, CRITERIA)["status"] == "INFEASIBLE"


@pytest.mark.parametrize("mutation", ["cohort", "identity", "nonfinite", "nondecision", "badcriteria"])
def test_malformed_or_pooled_constraints_are_rejected(mutation):
    rows = matrix([("PASS", "PASS"), ("FAIL", "FAIL"), ("FAIL", "FAIL")])
    criteria = deepcopy(CRITERIA)
    if mutation == "cohort": rows[0]["cohort_sha256"] = "other"
    elif mutation == "identity": rows[0]["id"] = rows[1]["id"]
    elif mutation == "nonfinite": rows[0]["smoke"]["cost"]["wall_seconds"] = float("nan")
    elif mutation == "nondecision": rows[0]["smoke"]["decision"] = "BLOCKED"
    else: criteria["minimum_reference_positives"] = True
    with pytest.raises(ValueError):
        decision_feasibility(rows, criteria)


def test_solver_agrees_with_independent_exhaustive_completion():
    # Oracle enumerates individual scientific outcomes, independent of the
    # implementation's count-state pruning and cost logic.
    variants = [(s, r) for s in ("PASS", "FAIL", "UNKNOWN") for r in ("PASS", "FAIL", "UNKNOWN")]
    for pairs in product(variants, repeat=3):
        assignments = [list(product(("PASS", "FAIL") if s == "UNKNOWN" else (s,),
                                    ("PASS", "FAIL") if r == "UNKNOWN" else (r,))) for s, r in pairs]
        possible = False
        for completion in product(*assignments):
            positives = [smoke for smoke, reference in completion if reference == "PASS"]
            negatives = [smoke for smoke, reference in completion if reference == "FAIL"]
            if len(positives) >= 1 and len(negatives) >= 2 and all(x == "PASS" for x in positives) and all(x == "FAIL" for x in negatives):
                possible = True
                break
        assert (decision_feasibility(matrix(pairs), CRITERIA)["status"] == "POSSIBLE") == possible


def test_preflight_is_read_only_and_missing_cost_stays_unavailable(promotion_setup):
    root, _, _, _ = promotion_setup
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = preflight(root, "current-fixture")
    assert result["feasibility"]["status"] == "POSSIBLE"
    assert result["observed_criteria_status"] == "PASS"
    assert result["qualification_input"] is result["default_adoption"] is result["training_authorized"] is False
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_bound_legacy_report_does_not_reapply_newer_host_validation(promotion_setup, monkeypatch):
    from experiments.forge import calibration
    root, _, _, _ = promotion_setup
    monkeypatch.setattr(calibration, "_evaluate_current", lambda *a: pytest.fail("archived decisions were regraded"))
    result = preflight(root, "current-fixture")
    assert result["decision_source"]["scope"] == "bound_published_report"
    assert result["qualification_input"] is False


@pytest.mark.parametrize("kind", ["conflict", "matching_extra_cost", "incomplete", "malformed", "retry"])
def test_new_unbound_attempts_block_saved_publication_without_regrading(promotion_setup, monkeypatch, kind):
    from experiments.forge import calibration
    from test_forge_promotion import save_attempt
    root, _, _, candidate = promotion_setup
    published = read_json(root / "reports/forge/calibration/current-fixture.json")
    directory = save_attempt(root, candidate, "later-attempt", score=0. if kind == "conflict" else 1.)
    if kind == "incomplete":
        (directory / "result.json").unlink()
    elif kind == "malformed":
        (directory / "request.json").write_text("{")
    elif kind == "retry":
        resolved = read_json(directory / "request.json")
        resolved["retry_of"] = {"attempt_id": "qualification"}
        atomic_json(directory / "request.json", resolved)
    monkeypatch.setattr(calibration, "_evaluate_current", lambda *a: pytest.fail("archived decisions were regraded"))
    result = preflight(root, "current-fixture")
    assert result["status"] == "BLOCKED"
    assert any(issue["attempt_id"] == "later-attempt" for issue in result["receipt_issues"])
    assert result["observed_criteria_status"] == published["adoption"]
    assert [(row["smoke"], row["reference"]) for row in result["matrix_decisions"]] == [
        (row["smoke"]["decision"], row["reference"]["decision"]) for row in published["matrix"]]


def test_unrelated_readable_candidate_and_scientific_cohort_do_not_stale_publication(promotion_setup):
    from test_forge_promotion import save_attempt
    root, _, _, candidate = promotion_setup
    other = deepcopy(candidate)
    other["candidate"]["id"] = "unrelated"
    save_attempt(root, other, "unrelated-attempt")
    other = deepcopy(candidate)
    other["protocol"]["seed"] = 17
    save_attempt(root, other, "incompatible-cohort")
    assert preflight(root, "current-fixture")["status"] == "POSSIBLE"


def test_unavailable_originals_block_new_registration_without_erasing_archived_decisions(lane_setup):
    from experiments.forge import calibration_lane
    root, path, _ = lane_setup
    published = read_json(root / "reports/forge/calibration/current-fixture.json")
    entry = next(entry for row in published["matrix"] for entry in row["inputs"])
    original = root / "reports/forge/attempts" / entry["attempt_id"] / "result.json"
    original.unlink()
    result = preflight(root, "current-fixture")
    assert result["status"] == "BLOCKED" and result["receipt_issues"]
    assert [(r["smoke"], r["reference"]) for r in result["matrix_decisions"]] == [
        (r["smoke"]["decision"], r["reference"]["decision"]) for r in published["matrix"]]
    with pytest.raises(CapabilityError, match="unresolved original receipt issues"):
        calibration_lane.register(root, path)
    assert not (root / "reports/forge/calibration-lanes/diagnose-negative").exists()


def test_bound_published_matrix_cannot_drop_a_required_task(promotion_setup):
    root, _, _, _ = promotion_setup
    path = root / "reports/forge/calibration/current-fixture.json"
    published = read_json(path)
    published["matrix"][0]["smoke"]["task_statuses"].clear()
    atomic_json(path, published)
    with pytest.raises(ValueError, match="required tasks"):
        preflight(root, "current-fixture")


def test_infeasible_registration_stops_before_source_freezing(lane_setup):
    from experiments.forge import calibration_lane
    root, path, _ = lane_setup
    config_path = root / "configs/forge/calibration/current-fixture.json"
    config = read_json(config_path)
    # Changing the reference criterion makes the existing three-lineage fixture
    # unable to supply a positive without changing a retained negative.
    config["lineages"] = [row for row in config["lineages"] if row["id"] != "finished"]
    atomic_json(config_path, config)
    contract = read_json(path)
    from experiments.forge.contracts import file_hash
    contract["profile_sha256"] = file_hash(config_path)
    atomic_json(path, contract)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(CapabilityError, match="cannot meet its frozen adoption criteria"):
        calibration_lane.register(root, path)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not (root / "reports/forge/calibration-lanes/diagnose-negative").exists()


def test_cli_returns_before_queue_construction(promotion_setup, monkeypatch, capsys):
    from experiments.forge import queue
    root, _, _, _ = promotion_setup
    monkeypatch.setattr(queue, "Queue", lambda *a, **kw: pytest.fail("read-only preflight constructed a queue"))
    assert main(["--root", str(root), "calibration-preflight", "--profile", "current-fixture", "--require-feasible"]) == 0
    assert '"training_authorized": false' in capsys.readouterr().out


def test_cli_nonzero_for_infeasible_and_receipt_issues(monkeypatch, capsys):
    import experiments.forge.calibration_feasibility as module
    for result in ({"feasibility": {"status": "INFEASIBLE"}, "receipt_issues": []},
                   {"feasibility": {"status": "POSSIBLE"}, "receipt_issues": [{"attempt_id": "missing"}]}):
        monkeypatch.setattr(module, "preflight", lambda *a: result)
        assert main(["calibration-preflight", "--profile", "fixture", "--require-feasible"]) == 1
        assert main(["calibration-preflight", "--profile", "fixture"]) == 0
        capsys.readouterr()
