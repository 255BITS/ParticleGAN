"""Versioned complete-tier scheduling without training or accelerator allocation."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge.execution_policy import DEFAULT, policy
from experiments.forge.queue import Queue
from test_forge_queue import SLOTS, campaign, finish, grade, request


def full_request(tmp_path):
    req = request(tmp_path)
    req["execution_policy"] = deepcopy(DEFAULT)
    req["view"]["assignments"][1]["qualification_tier"] = 1
    req["view"]["assignments"][2]["qualification_tier"] = 2
    return req


@pytest.mark.parametrize("status", ["FAIL", "INCOMPLETE", "INVALID", "BLOCKED"])
def test_nonpass_finishes_independent_current_tier_before_veto(tmp_path, status):
    queue = Queue(tmp_path / "queue", grader=lambda task, raw: {"gate_status": status if task["id"] == "t1" else "PASS"})
    queue.submit(full_request(tmp_path), campaign())
    first = queue.claim(SLOTS)
    finish(first)
    queue.collect()
    peer = queue.claim(SLOTS)
    assert peer["job"]["task_id"] == "t2"
    assert queue.claim(SLOTS) is None  # Tier 2 still awaits the whole current tier.
    finish(peer)
    queue.collect()
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    entry = next(iter(state["submissions"].values()))
    assert entry["status"] == "blocked" and f"t1: {status}" in entry["reason"]
    assert state["jobs"][full_request(tmp_path)["jobs"][2]["compatibility_key"]]["status"] == "pending"
    assert state["campaigns"]["pilot"]["spent_seconds"] == 2


def test_independent_jobs_can_share_slots_but_next_tier_needs_both_passes(tmp_path, monkeypatch):
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(full_request(tmp_path), campaign())
    first, peer = queue.claim(SLOTS), queue.claim(SLOTS)
    assert [first["job"]["task_id"], peer["job"]["task_id"]] == ["t1", "t2"]
    # Hold the simulated peer worker's execution lease while collecting first.
    # Otherwise an unlaunched claim correctly recovers as an orphan error.
    peer_lease = Path(peer["worker"]["directory"]) / "execution.lock"
    monkeypatch.setattr("experiments.forge.queue.lease_held", lambda path: path == peer_lease)
    finish(first)
    queue.collect()
    assert queue.claim(SLOTS) is None
    finish(peer, measured=2)
    queue.collect()
    next_tier = queue.claim(SLOTS)
    assert next_tier["job"]["task_id"] == "t3"


@pytest.mark.parametrize("kind", ["gate", "checkpoint", "data"])
def test_unsatisfied_dependency_skips_child_and_finishes_independent_peer(tmp_path, kind):
    req = full_request(tmp_path)
    for assignment in req["view"]["assignments"]:
        assignment["qualification_tier"] = 1
    req["tasks"]["t2"]["dependencies"] = [{"task": "t1", "kind": kind}]
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(req, campaign())
    first = queue.claim(SLOTS)
    finish(first, measured=99)
    queue.collect()
    independent = queue.claim(SLOTS)
    assert independent["job"]["task_id"] == "t3"
    finish(independent, measured=3)
    queue.collect()
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert state["jobs"][req["jobs"][1]["compatibility_key"]]["attempts"] == []
    assert "t2: prerequisites are unsatisfied: t1" in next(iter(state["submissions"].values()))["reason"]


def test_task_blocker_spends_nothing_and_does_not_suppress_peer(tmp_path):
    req = full_request(tmp_path)
    req["tasks"]["t1"]["preflight_blockers"] = ["unsupported host"]
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(req, campaign())
    peer = queue.claim(SLOTS)
    assert peer["job"]["task_id"] == "t2"
    finish(peer, measured=2)
    queue.collect()
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert state["jobs"][req["jobs"][0]["compatibility_key"]]["attempts"] == []
    assert "unsupported host" in next(iter(state["submissions"].values()))["reason"]


def test_grouped_child_blocker_skips_whole_group_and_preserves_peer(tmp_path):
    req = full_request(tmp_path)
    for assignment in req["view"]["assignments"]:
        assignment["qualification_tier"] = 1
    req["jobs"][0]["task_ids"] = ["t1", "t2"]
    req["jobs"].pop(1)
    req["tasks"]["t2"]["preflight_blockers"] = ["child lacks adapter"]
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(req, campaign())
    peer = queue.claim(SLOTS)
    assert peer["job"]["task_id"] == "t3"
    finish(peer, measured=3)
    queue.collect()
    assert queue.claim(SLOTS) is None
    assert sum(len(job["attempts"]) for job in queue.inspect()["jobs"].values()) == 1


def test_optional_nonpass_does_not_block_higher_tier_after_required_pass(tmp_path):
    req = full_request(tmp_path)
    req["view"]["assignments"][1]["importance"] = "diagnostic"
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(req, campaign())
    first = queue.claim(SLOTS)
    finish(first)
    queue.collect()
    optional = queue.claim(SLOTS)
    finish(optional, measured=99)
    queue.collect()
    assert queue.claim(SLOTS)["job"]["task_id"] == "t3"


def test_infrastructure_repair_retry_reserves_new_attempt_and_can_unblock_tier(tmp_path):
    req = full_request(tmp_path)
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(req, campaign())
    first = queue.claim(SLOTS)
    finish(first, status="error", elapsed=4)
    queue.collect()
    peer = queue.claim(SLOTS)
    finish(peer, measured=2, elapsed=3)
    queue.collect()
    assert queue.claim(SLOTS) is None
    queue.retry(first["job"]["compatibility_key"], reason="repaired worker environment")
    retried = queue.claim(SLOTS)
    assert retried["job"]["task_id"] == "t1"
    finish(retried, elapsed=2)
    queue.collect()
    assert queue.claim(SLOTS)["job"]["task_id"] == "t3"
    assert queue.inspect()["campaigns"]["pilot"]["spent_seconds"] == 9


def test_budget_failure_for_large_task_still_allows_smaller_independent_task(tmp_path):
    req = full_request(tmp_path)
    req["jobs"][0]["budget_seconds"] = 15
    req["jobs"][1]["budget_seconds"] = 5
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(req, campaign(budget=10))
    peer = queue.claim(SLOTS)
    assert peer["job"]["task_id"] == "t2"
    finish(peer, measured=2)
    queue.collect()
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert state["jobs"][req["jobs"][0]["compatibility_key"]]["attempts"] == []
    assert "budget" in next(iter(state["submissions"].values()))["reason"]


def test_cancellation_stops_remaining_current_tier_work(tmp_path):
    queue = Queue(tmp_path / "queue", grader=grade)
    entry = queue.submit(full_request(tmp_path), campaign())
    first = queue.claim(SLOTS)
    finish(first, measured=99)
    queue.collect()
    queue.cancel(entry["request"]["request_id"])
    assert queue.claim(SLOTS) is None
    assert sum(len(job["attempts"]) for job in queue.inspect()["jobs"].values()) == 1


def test_policy_changes_request_identity_and_reuses_exact_scientific_jobs(tmp_path):
    queue = Queue(tmp_path / "queue", grader=grade)
    req = full_request(tmp_path)
    legacy = deepcopy(req)
    legacy.pop("execution_policy")
    old = queue.submit(legacy, campaign())
    first = queue.claim(SLOTS)
    finish(first, measured=99)
    queue.collect()
    assert queue.claim(SLOTS) is None
    new = queue.submit(req, campaign())
    assert old["request"]["request_id"] != new["request"]["request_id"]
    peer = queue.claim(SLOTS)
    assert peer["job"]["task_id"] == "t2"
    assert len(queue.inspect()["jobs"]) == 3
    assert len(queue.inspect()["jobs"][first["job"]["compatibility_key"]]["attempts"]) == 1
    assert policy(legacy)["mode"] == "fail_fast"


@pytest.mark.parametrize("value", [None, {}, {"schema_version": True, "mode": "complete_current_tier"},
                                  {"schema_version": 2, "mode": "complete_current_tier"},
                                  {"schema_version": 1, "mode": "keep_going"},
                                  {"schema_version": 1, "mode": []}])
def test_invalid_execution_policy_rejected_before_mutation(tmp_path, value):
    req = request(tmp_path)
    req["execution_policy"] = value
    queue = Queue(tmp_path / "queue", grader=grade)
    with pytest.raises(ValueError, match="execution_policy"):
        queue.submit(req, campaign())
    assert queue.inspect()["submissions"] == {}
