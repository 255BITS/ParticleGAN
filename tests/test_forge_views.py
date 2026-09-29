"""Qualification is evidence-derived, prerequisite-safe, and independent of tier labels."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.forge.views import (
    grade_result, load_tasks, load_view, qualify, task_evaluation_fingerprint,
    task_execution_fingerprint, task_fingerprint, validate_view, view_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


def simple_task(name):
    return dict(schema_version=1, id=name, adapter="test_host",
                execution=dict(steps=24, produces_state=True),
                evaluation=dict(kind="transfer_sustained", thresholds=[["error", "<=", 1.0]],
                                scoring_weights="live"), resources={},
                requires_capabilities=[], dependencies=[])


def simple_view(*assignments):
    return dict(schema_version=1, id="test", revision=1, goal="quality", eligibility={},
                assignments=[dict(task=n, qualification_tier=t, importance=i, order=k)
                             for k, (n, t, i) in enumerate(assignments)])


def receipt(task, *, bad_steps=(), final=.5):
    steps = task["execution"]["steps"]
    names = {r[0] for r in task["evaluation"]["thresholds"]}
    assert names == {"error"}
    observations = [dict(step=math.ceil(i * steps / 24), error=2. if i in bad_steps else .5)
                    for i in range(1, 25)]
    return dict(task_id=task["id"], compatibility_key="compatible", gate_status="PASS",
                evidence=dict(observations=observations, live=dict(error=final)),
                cost=dict(seconds=2.))


def test_initial_inventory_has_complete_quality_and_distinct_claim_views():
    tasks = load_tasks(ROOT)
    stability = load_view(ROOT, "discriminator_stability")
    quality = load_view(ROOT, "quality_coverage")
    assert len(quality["assignments"]) == 22
    assert sum(t["evaluation"]["kind"] == "transfer_sustained" for t in tasks.values()) == 19
    smoke = [a["task"] for a in stability["assignments"] if a["qualification_tier"] == 1]
    assert smoke == ["two_pole", "unused_token_hold", "ae_gan_hold"]
    assert sum(tasks[n]["execution"]["steps"] for n in smoke) == 530
    assert all("qualification_tier" not in t for t in tasks.values())
    assert tasks["ring_hold"]["execution"]["execution_group"] == tasks["ring_extension"]["execution"]["execution_group"]
    assert tasks["ring_hold"]["execution"]["max_total_steps"] == 7500
    assert "target_shift_recovery" not in {a["task"] for a in stability["assignments"]}
    assert "target_shift_recovery" in {a["task"] for a in load_view(ROOT, "adaptation")["assignments"]}
    assert "clockfree_audit" in {a["task"] for a in load_view(ROOT, "clockfree_continuous")["assignments"]}
    for name, task in tasks.items():
        prior = task["execution"]["prior"]
        if prior["kind"] == "particle_cloud":
            assert prior["kind"] == "particle_cloud" and prior["sigma"] == 0
            assert prior["exception_reason"]
        else:
            assert prior["kind"] == "mog" and prior["sigma"] > 0 and prior["learnable"]
    assert tasks["ae_gan_hold"]["execution"]["prior"]["kind"] == "mog"


def test_uninterrupted_execution_group_cannot_cross_tier_caps():
    tasks=load_tasks(ROOT)
    view=load_view(ROOT,"discriminator_stability")
    next(a for a in view["assignments"] if a["task"]=="ring_hold")["qualification_tier"]=2
    with pytest.raises(ValueError,match="uninterrupted execution group"):
        validate_view(view,tasks)


def test_frozen_thresholds_reference_current_existing_evaluators():
    from benchmarks.transfer_suite.protocol import required_tasks
    from benchmarks.transfer_suite.vector_tasks import TASKS as vectors
    from benchmarks.transfer_suite.image_tasks import TASKS as images
    tasks = load_tasks(ROOT)
    for spec in required_tasks() + [s for s in vectors + images if s["tier"] == "ranking"]:
        thresholds = spec["thresholds"]
        if isinstance(thresholds, dict):
            thresholds = [["modes", ">=", thresholds["modes"]], ["hq", ">=", thresholds["hq_min"]]]
        assert tasks[spec["name"]]["evaluation"]["thresholds"] == json.loads(json.dumps(thresholds))
    for task in tasks.values():
        for path, digest in task["evaluation"].get("sources", {}).items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_retiering_reuses_evidence_without_changing_execution_or_evaluator_identity():
    tasks = {n: simple_task(n) for n in "abcd"}
    view = simple_view(("a", 1, "required"), ("b", 2, "required"),
                       ("c", 2, "required"), ("d", 3, "required"))
    frozen = deepcopy(view)
    results = [receipt(t) for t in tasks.values()]
    before = qualify(view, tasks, results)
    fingerprints = {n: task_fingerprint(t) for n, t in tasks.items()}
    moved = deepcopy(view)
    moved["revision"] = 2
    moved["assignments"][1]["qualification_tier"] = 1
    after = qualify(moved, tasks, results)
    assert before["qualified_tier"] == after["qualified_tier"] == 3
    assert after["task_statuses"] == before["task_statuses"]
    assert after["next_task"] is None
    assert view == frozen
    assert {n: task_fingerprint(t) for n, t in tasks.items()} == fingerprints
    assert view_fingerprint(moved) != view_fingerprint(view)
    changed = deepcopy(tasks["a"])
    changed["evaluation"]["thresholds"][0][2] = .1
    assert task_execution_fingerprint(changed) == task_execution_fingerprint(tasks["a"])
    assert task_evaluation_fingerprint(changed) != task_evaluation_fingerprint(tasks["a"])


@pytest.mark.parametrize("mutation,match", [
    (lambda v,t: v["assignments"][0].update(task="missing"), "missing task"),
    (lambda v,t: v["assignments"][0].update(qualification_tier=2), "empty required tier"),
    (lambda v,t: v["assignments"].append(deepcopy(v["assignments"][0])), "duplicate assignment"),
    (lambda v,t: t["a"].update(qualification_tier=1), "only in view"),
    (lambda v,t: t["a"].update(dependencies=[dict(task="b", kind="gate")]), "later tier"),
    (lambda v,t: t["b"].update(dependencies=[dict(task="missing", kind="gate")]), "missing from view"),
    (lambda v,t: t["b"].update(dependencies=[dict(task="a", kind="checkpoint")]), "continuation_of"),
])
def test_invalid_retiering_and_dependency_edits_are_rejected(mutation, match):
    tasks = {n: simple_task(n) for n in "ab"}
    view = simple_view(("a",1,"required"),("b",2,"required"))
    mutation(view,tasks)
    with pytest.raises(ValueError, match=match):
        validate_view(view,tasks)


def test_same_tier_cycles_and_checkpoints_without_producer_are_rejected():
    tasks = {n:simple_task(n) for n in "ab"}
    view = simple_view(("a",1,"required"),("b",1,"required"))
    tasks["a"]["dependencies"] = [dict(task="b",kind="gate")]
    tasks["b"]["dependencies"] = [dict(task="a",kind="gate")]
    with pytest.raises(ValueError,match="cycle"):
        validate_view(view,tasks)
    tasks["a"]["dependencies"] = []
    tasks["a"]["execution"]["produces_state"] = False
    tasks["b"]["dependencies"][0]["kind"] = "checkpoint"
    tasks["b"]["execution"]["continuation_of"] = "a"
    with pytest.raises(ValueError,match="does not produce state"):
        validate_view(view,tasks)


def test_required_failure_stops_next_tier_and_keeps_partial_qualification():
    tasks = {n:simple_task(n) for n in "abc"}
    view = simple_view(("a",1,"required"),("b",2,"required"),("c",3,"required"))
    result = qualify(view,tasks,[receipt(tasks["a"]),receipt(tasks["b"],final=2.)])
    assert result["qualified_tier"] == 1
    assert result["status"] == "FAIL" and result["next_task"] is None
    assert result["task_statuses"]["c"] == "NOT_RUN"
    assert result["required_total"] == 3 and result["required_passed"] == 1
    assert result["cost_seconds"] == 4


def test_next_task_respects_prerequisites_within_tier():
    tasks = {n:simple_task(n) for n in "ab"}
    tasks["a"]["dependencies"] = [dict(task="b",kind="gate")]
    view = simple_view(("a",1,"required"),("b",1,"required"))
    assert qualify(view,tasks,[])["next_task"] == "b"
    assert qualify(view,tasks,[receipt(tasks["b"])])["next_task"] == "a"


def test_dependency_failures_propagate_before_ordered_leaderboard_rows():
    tasks = {n: simple_task(n) for n in "abc"}
    tasks["a"]["dependencies"] = [dict(task="b", kind="gate")]
    tasks["b"]["dependencies"] = [dict(task="c", kind="gate")]
    view = simple_view(("a", 1, "required"), ("b", 1, "required"), ("c", 1, "required"))
    result = qualify(view, tasks, [receipt(tasks["a"]), receipt(tasks["b"]), receipt(tasks["c"], final=2.)])
    assert result["task_statuses"] == {"a": "BLOCKED", "b": "BLOCKED", "c": "FAIL"}
    assert result["required_passed"] == 0


def test_diagnostic_failure_does_not_veto_or_change_required_denominator():
    tasks = {n:simple_task(n) for n in "ab"}
    view = simple_view(("a",1,"required"),("b",1,"diagnostic"))
    result = qualify(view,tasks,[receipt(tasks["a"]),receipt(tasks["b"],final=2.)])
    assert result["qualified_tier"] == 1 and result["eligible"]
    assert result["required_total"] == result["required_passed"] == 1
    assert result["task_statuses"]["b"] == "FAIL"


def test_registered_calibration_view_has_no_qualification_even_when_all_metrics_pass():
    tasks = {"a": simple_task("a")}
    view = simple_view(("a", 1, "diagnostic"))
    with pytest.raises(ValueError, match="empty required tier"):
        validate_view(view, tasks)
    view["evidence_scope"] = "calibration_diagnostic"
    validate_view(view, tasks)
    missing = qualify(view, tasks, [])
    assert missing["status"] == "DIAGNOSTIC" and not missing["diagnostic_complete"]
    result = qualify(view, tasks, [receipt(tasks["a"])])
    assert result["diagnostic_complete"] and result["task_statuses"] == {"a": "PASS"}
    assert result["qualified_tier"] == 0 and not result["eligible"]
    assert not result["current_qualification_reuse"] and not result["next_tasks"]
    result = qualify(view, tasks, [receipt(tasks["a"], final=2.)])
    assert result["diagnostic_complete"] and result["task_statuses"] == {"a": "FAIL"}
    assert result["status"] == "DIAGNOSTIC" and result["qualified_tier"] == 0


@pytest.mark.parametrize("importance,tier", [("required", 1), ("ranking", 1), ("diagnostic", 2)])
def test_calibration_scope_cannot_hide_required_or_higher_tier_qualification(importance, tier):
    view = simple_view(("a", tier, importance))
    view["evidence_scope"] = "calibration_diagnostic"
    with pytest.raises(ValueError, match="only diagnostic tasks in Tier 1"):
        validate_view(view, {"a": simple_task("a")})


def test_raw_api_error_becomes_blocked_without_erasing_error_or_denominator():
    task = simple_task("a")
    raw = dict(task_id="a",gate_status="ERROR",error="unsupported keyword",applicability=dict(status="unsupported",reason="host lacks new field"))
    result = qualify(simple_view(("a",1,"required")),{"a":task},[raw])
    assert result["status"] == "BLOCKED" and result["required_total"] == 1
    assert result["tasks"][0]["raw_status"] == "ERROR"
    assert raw["gate_status"] == "ERROR"
    raw.pop("applicability")
    assert grade_result(task,raw)["status"] == "INCOMPLETE"


def test_pass_stamp_and_ema_cannot_substitute_for_live_evidence():
    task = simple_task("a")
    assert grade_result(task,{"gate_status":"PASS"})["status"] == "INCOMPLETE"
    raw = receipt(task)
    raw["evidence"]["scoring_weights"] = "ema"
    assert grade_result(task,raw)["status"] == "INVALID"


@pytest.mark.parametrize("mutation,status", [
    (lambda e:e["observations"].pop(),"INCOMPLETE"),
    (lambda e:e["observations"].append(dict(e["observations"][-1])),"INVALID"),
    (lambda e:e["observations"][5].update(error=float("inf")),"INVALID"),
    (lambda e:e["observations"][5].update(error=True),"INVALID"),
    (lambda e:e["observations"][5].update(step=5.5),"INVALID"),
    (lambda e:e["observations"][5].pop("error"),"INVALID"),
    (lambda e:e["live"].clear(),"INCOMPLETE"),
])
def test_whole_observation_evidence_is_required(mutation,status):
    task=simple_task("a"); raw=receipt(task)
    mutation(raw["evidence"])
    assert grade_result(task,raw)["status"] == status


def test_transient_or_early_good_checkpoint_cannot_hide_terminal_failure():
    task=simple_task("a")
    assert grade_result(task,receipt(task,bad_steps=[20]))["status"] == "FAIL"
    assert grade_result(task,receipt(task,bad_steps=[19]))["status"] == "PASS"
    assert grade_result(task,receipt(task,final=2.))["status"] == "FAIL"


def test_smoke_requires_learning_and_rng_activation_guards():
    task=simple_task("a")
    task["evaluation"]["guards"]=dict(finite_state=True,optimizer_roles=["prior"],mechanism_exercised=True,rng_isolation=True)
    raw=receipt(task)
    assert grade_result(task,raw)["status"] == "INCOMPLETE"
    raw["evidence"]["guards"]=dict(all_finite=True,optimizer_updates=dict(prior=0),hooks_exercised=True,unintended_rng_deviations=0)
    assert grade_result(task,raw)["status"] == "FAIL"
    raw["evidence"]["guards"]["optimizer_updates"]["prior"]=24
    assert grade_result(task,raw)["status"] == "BLOCKED"  # A boolean cannot prove activation.
    from experiments.forge.mechanisms import NAMES
    mechanisms = {name: dict(requested=False, enabled=False, calls=0, eligible=0, applied=0)
                  for name in NAMES}
    mechanisms["critic_penalty"].update(requested=True, enabled=True, calls=24, eligible=24, applied=24)
    raw["evidence"]["guards"]["mechanism_audit"] = {"schema_version": 1, "mechanisms": mechanisms}
    assert grade_result(task,raw)["status"] == "PASS"
    raw["evidence"]["guards"]["unintended_rng_deviations"]=1
    assert grade_result(task,raw)["status"] == "INVALID"


def test_conflicting_compatible_attempts_cannot_cherry_pick_a_pass():
    task=simple_task("a"); view=simple_view(("a",1,"required"))
    result=qualify(view,{"a":task},[receipt(task),receipt(task,final=2.)])
    assert result["status"] == "INVALID"


def ring_evidence(*,bad_step=None,truncate=0,run_id="one"):
    # First 200 checks acquire at 1400, followed by 1200 hold and 300 extension.
    points=[dict(step=s,modes=8,hq=.8 if s==bad_step else .95) for s in range(1201,2901-truncate)]
    return dict(dense=points,continuity=dict(mode="uninterrupted",run_id=run_id))


def test_ring_first_convergence_and_immediate_extension_are_separate_verdicts():
    tasks=load_tasks(ROOT)
    raw=dict(evidence=ring_evidence(bad_step=2700))
    assert grade_result(tasks["ring_hold"],raw)["status"] == "PASS"
    assert grade_result(tasks["ring_extension"],raw)["status"] == "FAIL"
    assert grade_result(tasks["ring_extension"],dict(evidence=ring_evidence(truncate=1)))["status"] == "INCOMPLETE"
    raw=dict(evidence=ring_evidence(bad_step=1500))
    assert grade_result(tasks["ring_hold"],raw)["status"] == "FAIL"


def test_ring_missing_dense_check_and_restarted_state_are_rejected():
    tasks=load_tasks(ROOT); raw=dict(evidence=ring_evidence())
    raw["evidence"]["dense"].pop(10)
    assert grade_result(tasks["ring_hold"],raw)["status"] == "INVALID"
    raw=dict(evidence=ring_evidence()); raw["evidence"].pop("continuity")
    assert grade_result(tasks["ring_extension"],raw)["status"] == "BLOCKED"


def test_checkpoint_continuation_requires_same_parent_state():
    tasks={n:simple_task(n) for n in "ab"}
    tasks["b"]["dependencies"]=[dict(task="a",kind="checkpoint")]
    tasks["b"]["execution"]["continuation_of"]="a"
    view=simple_view(("a",1,"required"),("b",2,"required"))
    a,b=receipt(tasks["a"]),receipt(tasks["b"])
    a["evidence"]["continuity"]=dict(mode="uninterrupted",run_id="parent")
    b["evidence"]["continuity"]=dict(mode="uninterrupted",run_id="other")
    assert qualify(view,tasks,[a,b])["status"] == "BLOCKED"
    b["evidence"]["continuity"]["run_id"]="parent"
    assert qualify(view,tasks,[a,b])["qualified_tier"] == 2
    a["evidence"]["continuity"]["state_sha256"] = "f" * 64
    b["evidence"]["continuity"] = dict(mode="verified_checkpoint", full_state_verified=True,
        parent_state_sha256="f" * 64, restore_proof={"status": "PASS", "source_sha256": "a" * 64})
    blocked = qualify(view, tasks, [a,b])
    assert blocked["qualified_tier"] == 1 and blocked["task_statuses"]["b"] == "BLOCKED"
    assert "no supported proof contract" in str(blocked["blockers"])


def test_native_gate_requires_artifacts_and_rejects_wrong_budget(tmp_path):
    task=load_tasks(ROOT)["grid100"]
    assert grade_result(task,dict(gate_status="PASS",evidence=dict(coverage="PASS",accuracy="PASS")))["status"] == "INCOMPLETE"
    (tmp_path/"grid100").mkdir()
    (tmp_path/"grid100/config.json").write_text(json.dumps(dict(steps=1000)))
    from experiments.forge.artifacts import manifest_artifacts
    raw=dict(evidence=dict(artifact_root=str(tmp_path), artifact_manifest=manifest_artifacts(tmp_path)))
    assert grade_result(task,raw)["status"] == "INVALID"


@pytest.mark.parametrize("name,key,value", [
    ("two_pole", "minimum_stable_checks", 6),
    ("two_pole", "observations", 30),
    ("two_pole", "evaluator", "another.module:gate"),
    ("grid100", "minimum_stable_checks", 7),
    ("grid100", "holdout_samples", 200000),
    ("grid100", "accuracy_limits", {"mass_tv": .01}),
    ("grid100", "coverage_thresholds", {"min_modes": 101}),
    ("ring_hold", "thresholds", [["hq", ">=", .99]]),
    ("target_shift_recovery", "stationary_checks", 6),
    ("target_shift_recovery", "deadline_checks", 82),
    ("target_shift_recovery", "minimum_frozen_passing", 1),
    ("clockfree_audit", "conditions", ["horizon"]),
    ("clockfree_audit", "training_feedback", True),
])
def test_fixed_evaluator_metadata_cannot_silently_override_its_predicate(name, key, value):
    tasks = load_tasks(ROOT)
    tasks[name]["evaluation"][key] = value
    view = load_view(ROOT, "adaptation" if name == "target_shift_recovery" else
        "clockfree_continuous" if name == "clockfree_audit" else "discriminator_stability")
    with pytest.raises(ValueError, match="fixed|unsupported override"):
        validate_view(view, tasks)
    assert grade_result(tasks[name], {"evidence": {}})["status"] == "INVALID"


def test_clockfree_claim_metadata_and_independent_parity_are_both_required():
    task=load_tasks(ROOT)["clockfree_audit"]
    view=simple_view((task["id"],1,"required"))
    view["eligibility"]=dict(requires_capabilities=["named_rng"],claim_contract=dict(learning="clockfree",shared_settings=True))
    evidence=dict(comparisons=[dict(condition=c,permitted_state_sha256="0"*64,rng_state_sha256="1"*64,
                                   reference_sha256="2"*64,changed_sha256="2"*64)
                               for c in task["evaluation"]["conditions"]],
                  source_audit=dict(source_sha256={"learner.py":"3"*64},allowed_state=["parameters","moments"],unexplained_clock_dependencies=[]))
    raw=dict(task_id=task["id"],evidence=evidence)
    assert qualify(view,{task["id"]:task},[raw])["status"] == "BLOCKED"
    candidate=dict(capabilities=["named_rng"],claim_contract=dict(learning="clockfree",shared_settings=True))
    assert qualify(view,{task["id"]:task},[raw],candidate=candidate)["status"] == "INCOMPLETE"
    raw["evidence"]["comparisons"][0]["changed_sha256"]="4"*64
    assert qualify(view,{task["id"]:task},[raw],candidate=candidate)["status"] == "INCOMPLETE"


def test_paired_adaptation_rejects_unmatched_or_truncated_control():
    task=load_tasks(ROOT)["target_shift_recovery"]
    assert grade_result(task,dict(evidence=dict(active={},frozen={})))["status"] == "INCOMPLETE"


def test_native_checkpoint_dependency_binds_the_selected_parent(monkeypatch):
    # Isolate dependency reduction; real saved-state verification is covered by
    # the public native continuation adapter test and does not use this stub.
    from experiments.forge import views
    parent, child = simple_task("prefix"), simple_task("continuation")
    child.update(adapter="native100_continuation", dependencies=[{"task": "prefix", "kind": "checkpoint"}])
    child["execution"]["continuation_of"] = "prefix"
    view = simple_view(("prefix", 1, "required"), ("continuation", 2, "required"))
    monkeypatch.setattr(views, "grade_result", lambda *args: views._verdict("PASS", "independent grader fixture"))
    checkpoint, manifest = {"state_sha256": "fixed"}, {"sha256": "tree"}
    first = {"task_id": "prefix", "compatibility_key": "prefix-key", "_attempt_id": "prefix-attempt",
             "evidence": {"checkpoint": checkpoint, "artifact_manifest": manifest}}
    second = {"task_id": "continuation", "evidence": {"prefix_parity": {
        "checkpoint": checkpoint, "artifact_manifest": manifest,
        "prerequisite": {"compatibility_key": "prefix-key", "attempt_id": "prefix-attempt"}}}}
    tasks = {"prefix": parent, "continuation": child}
    assert qualify(view, tasks, [first, second])["qualified_tier"] == 2
    second["evidence"]["prefix_parity"]["prerequisite"]["attempt_id"] = "another-candidate"
    assert qualify(view, tasks, [first, second])["task_statuses"]["continuation"] == "BLOCKED"
