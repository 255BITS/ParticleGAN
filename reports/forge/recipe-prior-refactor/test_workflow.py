"""Bounded admission/reporting checks; no training or queue submission."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import subprocess

import pytest

PATH = Path(__file__).with_name("workflow.py")
SPEC = importlib.util.spec_from_file_location("recipe_prior_refactor_workflow", PATH)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def test_baseline_is_actual_committed_develop_bytes():
    data = workflow.baseline()
    for name, descriptor in data["task_cards"].items():
        original = subprocess.check_output(["git", "show", f'{data["source_commit"]}:{descriptor["original_path"]}'], cwd=workflow.ROOT)
        assert original == (workflow.OUT / "legacy-task-cards" / f"{name}.json").read_bytes()
    for path, expected in ((data["family_current_path"], data["family_current_sha256"]),
                           (data["registry_path"], data["registry_sha256"])):
        import hashlib
        assert hashlib.sha256(subprocess.check_output(["git", "show", f'{data["source_commit"]}:{path}'], cwd=workflow.ROOT)).hexdigest() == expected


@pytest.mark.parametrize("family", ["atlas", "bcap-pure", "e22", "k3p", "ka2", "r1r2", "release07-gan-v3"])
def test_fresh_recipe_resolves_all_original_selected_settings(family):
    from experiments.forge.api import resolve_public_recipe
    leader = next(x for x in workflow.baseline()["leaders"] if x["selection"]["family"] == family)
    card = workflow.candidate_for(leader)
    actual = asdict(resolve_public_recipe(card))
    # Test the public constructor/validation, including its alias/API migration,
    # against a recipe resolved independently in the real old source checkout.
    for key, value in leader["portable_recipe"].items():
        assert workflow.stable_hash(actual[key]) == workflow.stable_hash(value), key
    assert actual["prior_reg"] == (0.05 if family == "release07-gan-v3" else 0.0)
    assert actual["prior_update"] == "learned"
    assert actual["prior_regularizer"] == "vicreg"
    assert actual["prior_l2"] == 0.0


def test_view_preserves_full_question_denominators_and_timeouts():
    view, tasks = workflow.ordinary_contract(workflow.ROOT)
    assert len(view["assignments"]) == 30
    tier2 = [a for a in view["assignments"] if a["qualification_tier"] == 2]
    assert len(tier2) == 21
    assert sum(tasks[a["task"]]["resources"]["timeout_seconds"] for a in tier2) == 40500
    assert workflow.PER_CANDIDATE * 7 == workflow.CAMPAIGN_CAP == 308280
    assert workflow.PER_CANDIDATE - workflow.ORDINARY_ALLOWANCE == 1020


@pytest.mark.parametrize("part", ["architecture", "gate", "prior_law", "budget", "cadence"])
def test_question_projection_retains_scientific_changes(part):
    old = workflow.read_json(workflow.OUT / "legacy-task-cards/gaussian1d_smoke.json")
    altered = deepcopy(old)
    if part == "architecture":
        altered["execution"]["steps"] += 1
    elif part == "gate":
        altered["evaluation"]["thresholds"][0][2] += 1
    elif part == "prior_law":
        altered["execution"]["prior"]["sigma"] *= 2
    elif part == "budget":
        altered["resources"]["timeout_seconds"] += 1
    else:
        altered["evaluation"]["observations"] += 1
    assert workflow.question(old) != workflow.question(altered)


def request_fixture():
    return dict(view=dict(assignments=[
        dict(task="trajectory", qualification_tier=2, importance="required"),
        dict(task="gaussian1d_stability", qualification_tier=2, importance="required"),
        dict(task="five_word_joint_hold", qualification_tier=2, importance="required")]),
        tasks=dict(trajectory=dict(dependencies=[]),
            gaussian1d_stability=dict(dependencies=[dict(task="gaussian1d_smoke", kind="checkpoint")]),
            five_word_joint_hold=dict(dependencies=[dict(task="five_word_joint_smoke", kind="checkpoint")])))


@pytest.mark.parametrize("status", ["PASS", "FAIL", "INCOMPLETE", "INVALID", "BLOCKED"])
def test_completed_ordinary_task_is_never_scientifically_retried(status):
    producers, selected, blocked = workflow.diagnostic_selection(request_fixture(),
        dict(trajectory=dict(gate_status=status)))
    assert "trajectory" not in selected
    assert not producers


@pytest.mark.parametrize("status", ["FAIL", "INCOMPLETE", "INVALID", "BLOCKED", None])
def test_unsuccessful_or_missing_prefix_cannot_be_retried(status):
    results = {} if status is None else dict(gaussian1d_smoke=dict(gate_status=status))
    producers, selected, blocked = workflow.diagnostic_selection(request_fixture(), results)
    assert "gaussian1d_smoke" not in producers
    assert "gaussian1d_stability" not in selected
    assert "gaussian1d_stability" in blocked
    assert "trajectory" in selected


def test_successful_prefixes_are_explicit_duplicate_dependencies_only():
    producers, selected, blocked = workflow.diagnostic_selection(request_fixture(),
        dict(gaussian1d_smoke=dict(gate_status="PASS"), five_word_joint_smoke=dict(gate_status="PASS")))
    assert producers == ["five_word_joint_smoke", "gaussian1d_smoke"]
    assert selected == ["trajectory", "gaussian1d_stability", "five_word_joint_hold"]
    assert not blocked


def test_unsupported_host_is_not_silently_replaced():
    request = request_fixture()
    request["tasks"]["trajectory"]["preflight_blockers"] = ["unsupported serving contract"]
    _, selected, blocked = workflow.diagnostic_selection(request, {})
    assert "trajectory" not in selected
    assert blocked["trajectory"] == ["unsupported serving contract"]


def test_declared_candidate_task_maps_are_readable_legacy_controls():
    declarations = workflow.declarations_for(workflow.ROOT)
    assert len(declarations) == 42
    studies = [d for p, d in declarations.items() if "/studies/" in p]
    assert len(studies) == 7
    for study in studies:
        assert len(study["control"]["task_map"]) == 28
        assert all(f"configs/forge/tasks/{v}.json" in declarations for v in study["control"]["task_map"].values())


def test_legacy_aliases_change_only_the_binding_identifier():
    declarations = workflow.declarations_for(workflow.ROOT)
    for name in workflow.baseline()["task_cards"]:
        original = workflow.read_json(workflow.OUT / "legacy-task-cards" / f"{name}.json")
        alias = deepcopy(declarations[f"configs/forge/tasks/{workflow.control_task(name)}.json"])
        alias["id"] = name
        assert alias == original


def test_declaration_cannot_overwrite_a_prior_registration(tmp_path):
    path = tmp_path / "registration.json"
    workflow.write_fresh(path, {"identity": 1})
    workflow.write_fresh(path, {"identity": 1})
    with pytest.raises(ValueError, match="Immutable declaration changed"):
        workflow.write_fresh(path, {"identity": 2})


def grading_fixture():
    evidence = dict(observations=[dict(step=1, cdf_ks=0.04)], sampling_law="public_prior_without_output_noise")
    raw_result = dict(task_results=dict(gaussian1d_smoke=dict(evidence=evidence)))
    grade = dict(gate_status="PASS", status="pass", metrics=dict(cdf_ks=0.04), evaluator_result=dict(method="frozen-full-gate"))
    grading = dict(raw_hash=workflow.stable_hash(raw_result), source_digest="exact-scientific-source", grades=dict(gaussian1d_smoke=grade))
    return dict(request=dict(requires_independent_grading=True, source=dict(digest="exact-scientific-source")),
        result=dict(raw=dict(attempt_status="completed", result=raw_result, grading=grading),
            task_results=[dict(task_id="gaussian1d_smoke", evidence=deepcopy(evidence), **deepcopy(grade))]))


def test_saved_grading_accepts_original_queue_merge_contract():
    proof = workflow.saved_grading(grading_fixture())
    assert proof["independently_graded"] is True
    assert proof["source_digest"] == "exact-scientific-source"


@pytest.mark.parametrize("tamper", ["missing", "source", "raw_input", "task_id", "gate", "method", "metrics", "evidence"])
def test_saved_grading_fails_closed_on_tampering(tamper):
    attempt = grading_fixture()
    grading = attempt["result"]["raw"]["grading"]
    row = attempt["result"]["task_results"][0]
    if tamper == "missing":
        del attempt["result"]["raw"]["grading"]
    elif tamper == "source":
        grading["source_digest"] = "other-source"
    elif tamper == "raw_input":
        attempt["result"]["raw"]["result"]["task_results"]["gaussian1d_smoke"]["evidence"]["observations"][0]["cdf_ks"] = 0.5
    elif tamper == "task_id":
        row["task_id"] = "other-task"
    elif tamper == "gate":
        row["gate_status"] = "FAIL"
    elif tamper == "method":
        row["evaluator_result"]["method"] = "other-method"
    elif tamper == "metrics":
        row["metrics"]["cdf_ks"] = 0.5
    else:
        row["evidence"]["sampling_law"] = "other-law"
    with pytest.raises(ValueError):
        workflow.saved_grading(attempt)


def test_noncompleted_attempt_cannot_claim_pass():
    attempt = grading_fixture()
    attempt["result"]["raw"]["attempt_status"] = "timeout"
    with pytest.raises(ValueError, match="Noncompleted"):
        workflow.saved_grading(attempt)
