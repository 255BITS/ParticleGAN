"""Bounded pacing-study admission and whole-recipe decisions; no GAN training."""
import importlib.util
from pathlib import Path
import sys
import time

import pytest

from experiments.forge.configuration_search import _base_declaration, _declarations, _grid, _resolved_recipe
from experiments.forge.contracts import read_json, stable_hash
from test_forge_queue import SLOTS, campaign, finish, grade, request
from test_forge_workers import SLEEP_PROGRAM, await_file, worker_request
from experiments.forge.queue import process_identity


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location(
        "dualnorm_pacing_study", ROOT / "reports/forge/dualnorm-pacing-v2/run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def declarations(study):
    contract = read_json(ROOT / "reports/forge/dualnorm-pacing-v2/contract.json")
    stage_a = read_json(ROOT / contract["stage_a_spec"])
    return contract, study.declared_specs(contract, stage_a)


def trial(contract, name, passed=3, *, lr=.01, ratio=1.5, prior=.03):
    tasks = [{"task": task, "qualification_tier": 1, "importance": "required",
              "gate_status": "PASS" if index < passed else "FAIL",
              "attempt_id": f"attempt-{name}-{index}",
              "compatibility_key": f"key-{name}-{index}"}
             for index, task in enumerate(contract["required_tasks"])]
    tasks.append({"task": contract["diagnostic_task"], "qualification_tier": 1,
                  "importance": "diagnostic", "gate_status": "PASS",
                  "attempt_id": f"attempt-{name}-clock", "compatibility_key": f"key-{name}-clock"})
    return {"candidate_id": f"candidate-{name}", "configuration_id": name,
            "source_digest": "source", "runtime_cohort": {"execution_backend": "cuda"},
            "protocol_hash": "protocol", "policy_fingerprint": "policy",
            "submission_status": "blocked", "submission_blockers": [], "tasks": tasks,
            "settings": {"lr": lr, "d_lr_mult": ratio, "prior_lr_mult": prior / lr},
            "resolved_recipe": {"lr": lr, "d_lr_mult": ratio,
                                "prior_lr_mult": prior / lr, "optimizer_momentum": 0.}}


def test_reachable_searches_are_finite_and_optimizer_only(study, declarations):
    contract, entries = declarations
    assert len(entries) == 101
    assert sum(len(_grid(entry["spec"]["grid"])) for entry in entries) == 240
    assert {entry["stage"] for entry in entries} == {"A", "B", "C"}
    assert contract["maximum_admitted_recipes"] == 25
    assert contract["maximum_paid_reserved_seconds"] == 25 * 2520
    assert contract["maximum_elapsed_seconds"] == 12 * 3600
    assert contract["automatic_retries"] == contract["automatic_expansions"] == 0
    baseline = _resolved_recipe(_base_declaration(ROOT, entries[0]["spec"]))
    allowed = {"optimizer_family", "optimizer_momentum", "lr", "d_lr_mult", "prior_lr_mult"}
    for entry in entries:
        spec = entry["spec"]
        assert spec["tuning_through_tier"] == 1
        assert spec["view"] == "discriminator_stability"
        assert spec["campaign"]["candidate_budget_seconds"] == 2520
        assert set().union(*(row.keys() for row in _grid(spec["grid"]))) <= allowed
    for card, _ in _declarations(ROOT, entries[0]["spec"]):
        recipe = card["resolved_configuration_recipe"]
        changes = {key for key, value in recipe.items()
                   if stable_hash(value) != stable_hash(baseline[key])}
        assert changes <= allowed
        assert recipe["reg_coeff"] == 1 and recipe["reg_arm"] == "b_cap"


def test_failure_with_missing_independent_peer_cannot_activate(study, declarations):
    contract, _ = declarations
    control = trial(contract, "b")
    unfinished = trial(contract, "a", passed=4)
    unfinished["tasks"][5].update(gate_status="UNKNOWN", attempt_id=None)
    assert not study.execution_complete(unfinished)
    assert not study.whole_selection([control, unfinished])["selection_complete"]
    assert not study.activation("B", [control, unfinished], control, contract)["eligible"]


@pytest.mark.parametrize("field", ["source_digest", "runtime_cohort", "protocol_hash", "policy_fingerprint"])
def test_different_evidence_cohorts_cannot_be_selected_together(study, declarations, field):
    contract, _ = declarations
    first, second = trial(contract, "a"), trial(contract, "b", passed=4)
    second[field] = {"execution_backend": "cpu"} if field == "runtime_cohort" else "different"
    with pytest.raises(ValueError):
        study.whole_selection([first, second])


def test_whole_hash_tiebreak_and_conditional_absolute_prior(study, declarations):
    contract, entries = declarations
    control = trial(contract, "z")
    winner = trial(contract, "a", passed=4, ratio=.75, prior=.003)
    selection = study.whole_selection([control, winner])
    assert selection["selected_candidate_id"] == winner["candidate_id"]
    assert selection["required_pass_count"] == 4
    assert not selection["qualified"]
    tied = trial(contract, "b", passed=4, ratio=2., prior=.1)
    assert study.whole_selection([tied, winner])["selected_candidate_id"] == winner["candidate_id"]
    activation = study.activation("B", [control, winner], control, contract)
    assert activation["eligible"]
    branch = study.branch_for("B", activation["pace"], entries)
    rows = _grid(branch["spec"]["grid"])
    assert {row["lr"] for row in rows} == {.012, .016, .022}
    assert all(row["d_lr_mult"] == .75 for row in rows)
    assert all(row["lr"] * row["prior_lr_mult"] == pytest.approx(.003) for row in rows)


def test_incomplete_selected_evidence_and_regressing_score_do_not_advance(study, declarations):
    contract, _ = declarations
    control = trial(contract, "z", passed=4)
    incomplete = trial(contract, "a", passed=5)
    incomplete["tasks"][-1]["gate_status"] = "INCOMPLETE"
    assert not study.activation("B", [control, incomplete], control, contract)["eligible"]
    assert not study.activation("B", [trial(contract, "a", passed=3)], control, contract)["eligible"]


def test_deadline_denial_survives_refresh_without_spending(study, tmp_path):
    queue = study.DeadlineQueue(tmp_path / "queue", deadline=100.,
                                cleanup_grace_seconds=30., clock=lambda: 70., grader=grade)
    queue.submit(request(tmp_path, cap=1, cost=10), campaign())
    assert queue.claim(SLOTS) is None
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert next(iter(state["submissions"].values()))["status"] == "blocked"
    assert state["campaigns"]["pilot"]["spent_seconds"] == 0
    assert not any(job["attempts"] for job in state["jobs"].values())


def test_deadline_preserves_running_peer_and_full_task_allowance(study, tmp_path):
    now = [0.]
    queue = study.DeadlineQueue(tmp_path / "queue", deadline=100.,
                                cleanup_grace_seconds=5., clock=lambda: now[0], grader=grade)
    req = request(tmp_path, cap=1, cost=10)
    req["execution_policy"] = {"schema_version": 1, "mode": "complete_current_tier"}
    for assignment in req["view"]["assignments"]:
        assignment["qualification_tier"] = 1
    queue.submit(req, campaign())
    claimed = queue.claim(SLOTS)
    assert claimed["job"]["budget_seconds"] == 10
    now[0] = 94.
    assert queue.claim(SLOTS) is None
    assert next(iter(queue.inspect()["submissions"].values()))["status"] == "running"
    finish(claimed, measured=1)
    queue.collect()
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert next(iter(state["submissions"].values()))["status"] == "blocked"
    assert sum(len(job["attempts"]) for job in state["jobs"].values()) == 1
    assert sum(job.get("reserved_seconds", 0) for job in state["jobs"].values()) == 0


def test_deadline_still_admits_an_independent_peer_that_fits(study, tmp_path):
    queue = study.DeadlineQueue(tmp_path / "queue", deadline=100.,
                                cleanup_grace_seconds=5., clock=lambda: 92., grader=grade)
    req = request(tmp_path, cap=1, cost=10)
    req["execution_policy"] = {"schema_version": 1, "mode": "complete_current_tier"}
    for assignment in req["view"]["assignments"]:
        assignment["qualification_tier"] = 1
    req["jobs"][2]["budget_seconds"] = 2
    queue.submit(req, campaign())
    claimed = queue.claim(SLOTS)
    assert claimed["job"]["task_id"] == "t3"
    assert claimed["job"]["budget_seconds"] == 2
    finish(claimed, measured=3)
    queue.collect()
    assert queue.claim(SLOTS) is None
    assert sum(len(job["attempts"]) for job in queue.inspect()["jobs"].values()) == 1


def test_campaign_cleanup_waits_for_real_supervisor_and_descendants(study, tmp_path):
    queue = study.DeadlineQueue(tmp_path / "queue", deadline=time.monotonic() + 60,
                                cleanup_grace_seconds=30., grader=grade)
    req = worker_request(tmp_path, SLEEP_PROGRAM, budget=20)
    queue.submit(req, campaign(study.CAMPAIGN))
    claimed = queue.claim(SLOTS)
    process = queue.launch(claimed)
    try:
        # Includes importing the real serial-autograd dependency in the child.
        child = await_file(Path(claimed["worker"]["directory"]) / "grandchild.json", timeout=10)
    finally:
        study.cancel_campaign(queue, "software test of campaign shutdown")
        study.finish_supervision(queue)
    assert process.wait(timeout=3) == 1
    supervision = study.supervision_state(queue)
    assert not supervision["running_jobs"] and not supervision["active_leases"]
    assert process_identity(child["pid"]) is None
    assert len(supervision["jobs"][0]["attempts"]) == 1
    assert supervision["state"]["campaigns"][study.CAMPAIGN]["reserved_seconds"] == 0


def test_expired_clock_before_first_admission_is_an_unmeasured_stop(study, declarations, tmp_path, monkeypatch):
    from experiments.forge.contracts import atomic_json

    contract, entries = declarations
    frozen = {"input_digest": "frozen", "control_candidate_id": "control",
              "source": {"origin_commit": "source-commit", "digest": "source-digest"}}
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "REPORT", tmp_path / "reports/forge/dualnorm-pacing-v2")
    monkeypatch.setattr(study, "verify_frozen", lambda *_: frozen)
    started = time.monotonic() - 43201
    clock = {"schema_version": 1, "study_binding_sha256": "frozen",
             "source": {"commit": "source-commit", "digest": "source-digest"},
             "control_candidate_id": "control", "boot_id": study._boot_id(),
             "started_at": "earlier", "started_monotonic": started,
             "deadline_monotonic": started + 43200, "deadline_utc": "expired"}
    clock["input_digest"] = stable_hash(clock)
    queue_root = tmp_path / "queue"
    atomic_json(queue_root / "execution-clock.json", clock)
    result = study.execute(queue_root, contract, entries, ["0", "1"])
    assert result["status"] == "stopped_incomplete"
    assert result["admitted_recipes"] == 0 and result["selection"] is None
    assert result["all_attempts_supervised_finished"]
    assert result["accounting"] == {}


@pytest.mark.parametrize("devices", [["cpu"], ["0"], ["0", "0"], ["0", "1", "2"]])
def test_driver_rejects_a_different_worker_policy(study, devices):
    with pytest.raises(ValueError):
        study.validate_devices(devices)
