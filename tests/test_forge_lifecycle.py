"""Administrative decisions preserve failed evidence and exact readout lineage."""
from copy import deepcopy
import json

import pytest

from experiments.forge import knowledge, lifecycle, planning
from experiments.forge.contracts import atomic_json, read_json, stable_hash


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    requests = {}
    for name, revision in (("idea", "revision-a"), ("successor", "revision-b")):
        candidate = {"schema_version": 1, "id": name, "hypothesis": "A substantive mechanism change",
                     "changed_factors": ["Synthetic substantive mechanism for lifecycle software controls"],
                     "goal": "stability", "mechanism_class": "structural"}
        atomic_json(tmp_path / f"configs/forge/ideas/{name}.json", candidate)
        requests[name] = {"candidate": candidate, "candidate_revision": revision,
                          "source": {"digest": "source-" + revision}, "runtime": {"python": "fixture"},
                          "execution_backend": "cpu", "jobs": [{"task_id": "cheap", "compatibility_key": "key"}]}
    monkeypatch.setattr(planning, "resolve_idea", lambda root, name, **kwargs: deepcopy(requests[name]))
    return tmp_path, requests


def attempt(root, request, name="attempt-one", *, seconds=2.5):
    result = {"schema_version": 1, "attempt_id": name, "candidate_revision": request["candidate_revision"],
              "task_results": [{"task_id": "cheap", "compatibility_key": "key", "gate_status": "FAIL",
                                "raw_status": "completed", "metrics": {"error": 2},
                                "cost": {"wall_seconds": seconds}, "evidence": {"measured": True}}]}
    directory = root / "reports/forge/attempts" / name
    atomic_json(directory / "request.json", {"request": request})
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result),
                "source": request["source"], "runtime": request["runtime"]})
    return directory


def readout(root, request):
    attempts, _ = knowledge._attempts(root)
    selected = [a for a in attempts if a["request"]["candidate_revision"] == request["candidate_revision"]]
    record = {"schema_version": 1, "record_id": "readout-fixture", "record_type": "scientific",
              "evidence_scope": "current", "lifecycle": "concluded", "candidate_id": request["candidate"]["id"],
              "candidate_revision": request["candidate_revision"], "task_results": [row for a in selected for row in a["task_results"]],
              "attempt_ids": [a["attempt_id"] for a in selected],
              "provenance": {"attempts": [{"attempt_id": a["attempt_id"], "result_hash": a["result_hash"]} for a in selected]},
              "conclusion": "Fails smoke", "comparison": "Worse than the fixed control", "next_action": "Change the mechanism"}
    path = root / "reports/forge/records/readout-fixture.json"
    atomic_json(path, record)
    return path, record


def test_unattempted_abandonment_is_immutable_idempotent_and_does_not_qualify(cohort):
    root, requests = cohort
    record = lifecycle.abandon(root, "idea", " Insufficient expected value ")
    assert record["lifecycle"] == "abandoned"
    assert record["task_results"] == record["attempt_ids"] == []
    assert record["qualification_reuse"] is False
    assert record["cost_at_disposition"]["wall_seconds"] is None
    path = root / "reports/forge/records" / (record["record_id"] + ".json")
    saved = path.read_bytes()
    assert lifecycle.abandon(root, "idea", "Insufficient expected value") == record
    assert path.read_bytes() == saved
    with pytest.raises(ValueError, match="immutable disposition"):
        lifecycle.abandon(root, "idea", "Rewrite the reason")
    with pytest.raises(ValueError, match="abandoned"):
        lifecycle.ensure_open(root, requests["idea"])
    lifecycle.ensure_open(root, requests["successor"])


def test_attempted_failure_requires_and_preserves_concluded_readout_and_cost(cohort):
    root, requests = cohort
    directory = attempt(root, requests["idea"])
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match="concluded readout"):
        lifecycle.abandon(root, "idea", "No further allocation")
    path, explanation = readout(root, requests["idea"])
    prior_readout = path.read_bytes()
    record = lifecycle.abandon(root, "idea", "No further allocation")
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert path.read_bytes() == prior_readout
    assert record["provenance"]["concluded_readout"] == {
        "record_id": explanation["record_id"], "sha256": stable_hash(explanation), "snapshot": explanation}
    assert record["cost_at_disposition"]["wall_seconds"] == 2.5
    assert record["lifecycle_events"][0]["from"] == "concluded"
    attempts, _ = knowledge._attempts(root)
    records, _ = knowledge._records(root)
    base = {"lifecycle": "concluded", "pending_readout": False, "status": "FAIL", "qualified_tier": 0,
            "task_results": attempts[0]["task_results"], "cost": {"wall_seconds": 2.5}}
    overlaid = lifecycle.overlay(requests["idea"], attempts, records, {}, base)
    assert overlaid["lifecycle"] == "abandoned"
    assert {key: overlaid[key] for key in ("status", "qualified_tier", "task_results", "cost")} == {
        key: base[key] for key in ("status", "qualified_tier", "task_results", "cost")}


@pytest.mark.parametrize("status", ["queued", "running", "paused"])
def test_active_other_runtime_cohort_blocks_disposition(cohort, status):
    root, requests = cohort
    request = deepcopy(requests["idea"])
    request["execution_backend"] = "cuda"
    location = root / "external-queue"
    atomic_json(location / "queue/state.json", {"submissions": {"active": {"request": request, "status": status}}})
    with pytest.raises(ValueError, match="active queued/running/paused"):
        lifecycle.abandon(root, "idea", "Stopped allocation", queue_root=location)
    assert not list((root / "reports/forge/records").glob("lifecycle-*.json"))


def test_supersession_binds_existing_successor_and_preserves_predecessor(cohort):
    root, requests = cohort
    attempt(root, requests["idea"])
    readout(root, requests["idea"])
    record = lifecycle.supersede(root, "idea", "successor", "Address the measured failure")
    assert record["successor"] == "successor@revision-b"
    assert record["successor_identity"]["source"] == requests["successor"]["source"]
    assert record["candidate_revision"] == "revision-a"
    assert record["lifecycle"] == "superseded"
    assert record["cost_at_disposition"]["wall_seconds"] == 2.5


@pytest.mark.parametrize("replacement, message", [("missing", "Unknown"), ("idea", "different candidate")])
def test_invalid_successors_do_not_write_dispositions(cohort, replacement, message):
    root, _ = cohort
    with pytest.raises(ValueError, match=message):
        lifecycle.supersede(root, "idea", replacement, "A reason")
    assert not list((root / "reports/forge/records").glob("lifecycle-*.json"))


def test_terminal_successor_is_not_an_open_replacement(cohort):
    root, _ = cohort
    lifecycle.abandon(root, "successor", "Rejected independently")
    with pytest.raises(ValueError, match="Successor revision is already"):
        lifecycle.supersede(root, "idea", "successor", "Try the replacement")


def test_multiple_attempted_revisions_require_explicit_selection(cohort):
    root, requests = cohort
    attempt(root, requests["idea"])
    another = deepcopy(requests["idea"])
    another["candidate_revision"] = "revision-another"
    attempt(root, another, "attempt-two")
    readout(root, requests["idea"])
    with pytest.raises(ValueError, match="Multiple attempted revisions"):
        lifecycle.abandon(root, "idea", "Stop")
    with pytest.raises(ValueError, match="Ambiguous revision prefix"):
        lifecycle.abandon(root, "idea@revision-a", "Stop")
    # An unambiguous saved full revision remains selectable even after the live source changes.
    another["candidate_revision"] = "revision-b"
    directory = root / "reports/forge/attempts/attempt-two"
    result = read_json(directory / "result.json")
    result["candidate_revision"] = "revision-b"
    atomic_json(directory / "request.json", {"request": another})
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result), "source": another["source"], "runtime": another["runtime"]})
    record = lifecycle.abandon(root, "idea@revision-a", "Stop")
    assert record["candidate_revision"] == "revision-a"


@pytest.mark.parametrize("change", ["new_attempt", "changed_result"])
def test_readout_must_cover_exact_current_result_set(cohort, change):
    root, requests = cohort
    attempt(root, requests["idea"])
    readout(root, requests["idea"])
    if change == "new_attempt":
        attempt(root, requests["idea"], "attempt-two")
    else:
        attempt(root, requests["idea"], seconds=9)
    with pytest.raises(ValueError, match="concluded readout"):
        lifecycle.abandon(root, "idea", "Stop")


def test_unpaired_durable_attempt_cannot_be_silently_abandoned(cohort):
    root, requests = cohort
    atomic_json(root / "reports/forge/attempts/unpaired/request.json", {"request": requests["idea"]})
    with pytest.raises(ValueError, match="incomplete durable attempt"):
        lifecycle.abandon(root, "idea", "Stop")


def test_later_attempt_or_active_queue_does_not_hide_new_readout_obligation(cohort):
    root, requests = cohort
    attempt(root, requests["idea"])
    readout(root, requests["idea"])
    lifecycle.abandon(root, "idea", "Stop")
    attempt(root, requests["idea"], "unauthorized-later-attempt")
    attempts, _ = knowledge._attempts(root)
    records, _ = knowledge._records(root)
    base = {"lifecycle": "awaiting_readout", "pending_readout": True, "status": "FAIL"}
    row = lifecycle.overlay(requests["idea"], attempts, records, {}, base)
    assert row["lifecycle"] == "awaiting_readout" and row["pending_readout"]
    assert "lifecycle_conflict" in row


def test_receipt_tampering_is_rejected_instead_of_overwriting_history(cohort):
    root, requests = cohort
    record = lifecycle.abandon(root, "idea", "Stop")
    path = root / "reports/forge/records" / (record["record_id"] + ".json")
    record["reason"] = "Rewritten"
    atomic_json(path, record)
    with pytest.raises(ValueError, match="Invalid immutable lifecycle"):
        lifecycle.ensure_open(root, requests["idea"])


@pytest.mark.parametrize("reason", ["", "  ", None])
def test_empty_reason_is_rejected_before_writes(cohort, reason):
    root, _ = cohort
    with pytest.raises(ValueError, match="nonempty reason"):
        lifecycle.abandon(root, "idea", reason)
    assert not (root / "runs").exists()


@pytest.mark.parametrize("cancel_before_readout", [False, True])
def test_real_queue_refuses_resubmit_and_retry_after_disposition(cohort, cancel_before_readout):
    # Reuse the queue's no-training worker-receipt fixture; launch is never called.
    from test_forge_queue import SLOTS, campaign, finish, grade, request as queue_request
    from experiments.forge.queue import Queue
    root, requests = cohort
    req = queue_request(root, "idea", cap=1)
    requests["idea"] = req
    queue = Queue(root / "queue", report_root=root / "reports/forge", grader=grade)
    submission = queue.submit(req, campaign())
    claim = queue.claim(SLOTS)
    finish(claim, status="error", elapsed=3.25)
    assert queue.collect() == 1
    if cancel_before_readout:
        queue.cancel(submission["request"]["request_id"])
    explanation = knowledge.readout(root, "idea", "Execution failed", "No scientific comparison", "Retire this revision")
    disposition = lifecycle.abandon(root, "idea", "No further allocation", queue_root=queue.root)
    before = queue.inspect()
    assert disposition["readout_record_id"] == explanation["record_id"]
    assert disposition["cost_at_disposition"]["wall_seconds"] == 3.25
    with pytest.raises(ValueError, match="revision is abandoned"):
        queue.submit(req, campaign())
    with pytest.raises(ValueError, match="revision is abandoned"):
        queue.submit(req, campaign("another-campaign"))
    with pytest.raises(ValueError, match="revision is abandoned|open.*subscrib|authorized.*subscrib"):
        queue.retry(claim["job"]["compatibility_key"], reason="Runtime repaired")
    assert queue.inspect() == before
    assert queue.claim(SLOTS) is None
    assert before["campaigns"]["pilot"]["spent_seconds"] == 3.25
    assert before["jobs"][claim["job"]["compatibility_key"]]["result"]["task_results"][0]["gate_status"] == "INCOMPLETE"


def test_real_queue_claim_rejects_stale_operational_reactivation(cohort):
    from test_forge_queue import SLOTS, campaign, grade, request as queue_request
    from experiments.forge.queue import Queue
    root, requests = cohort
    req = queue_request(root, "idea", cap=1)
    requests["idea"] = req
    queue = Queue(root / "queue", report_root=root / "reports/forge", grader=grade)
    submission = queue.submit(req, campaign())
    request_id = submission["request"]["request_id"]
    queue.cancel(request_id)
    lifecycle.abandon(root, "idea", "Not worth running", queue_root=queue.root)
    # Model an old operational snapshot restored after the immutable decision.
    with queue.state() as state:
        state["submissions"][request_id].update(status="queued", lifecycle="ready")
        for job in state["jobs"].values():
            job["subscribers"].append(request_id)
    assert queue.claim(SLOTS) is None
    state = queue.inspect()
    assert state["submissions"][request_id]["status"] == "blocked"
    assert "abandoned" in state["submissions"][request_id]["reason"]
    assert not any(job["attempts"] for job in state["jobs"].values())
    assert state["campaigns"]["pilot"]["spent_seconds"] == 0


def test_shared_open_subscriber_can_repair_without_reviving_disposed_idea(cohort):
    from test_forge_queue import SLOTS, campaign, finish, grade, request as queue_request
    from experiments.forge.queue import Queue
    root, requests = cohort
    req = queue_request(root, "idea", cap=1)
    requests["idea"] = req
    queue = Queue(root / "queue", report_root=root / "reports/forge", grader=grade)
    first = queue.submit(req, campaign())
    claim = queue.claim(SLOTS)
    finish(claim, status="error", elapsed=3.25)
    queue.collect()
    knowledge.readout(root, "idea", "Execution failed", "No scientific outcome", "Retire this declaration")
    lifecycle.abandon(root, "idea", "No further allocation", queue_root=queue.root)
    original = root / "reports/forge/attempts" / claim["worker"]["attempt"] / "result.json"
    original_bytes = original.read_bytes()
    alias = deepcopy(req)
    alias["candidate"]["id"] = "open-equivalent-idea"
    second = queue.submit(alias, campaign())
    key = claim["job"]["compatibility_key"]
    queue.retry(key, reason="Repair for the still-open compatible subscriber")
    state = queue.inspect()
    assert state["jobs"][key]["subscribers"] == [second["request"]["request_id"]]
    assert state["submissions"][first["request"]["request_id"]]["status"] == "concluded"
    replacement = queue.claim(SLOTS)
    assert replacement["request"]["candidate"]["id"] == "open-equivalent-idea"
    finish(replacement, elapsed=1.0)
    queue.collect()
    assert original.read_bytes() == original_bytes
    assert queue.inspect()["campaigns"]["pilot"]["spent_seconds"] == 4.25
    attempts, _ = knowledge._attempts(root)
    records, _ = knowledge._records(root)
    # Boards share compatible science, but administration belongs to each idea.
    state = knowledge._lifecycle(req, attempts, records, {str(queue.root): queue.inspect()})
    assert state["lifecycle"] == "abandoned"
    assert not state["pending_readout"]
    assert "lifecycle_conflict" not in state
    with pytest.raises(ValueError, match="abandoned"):
        queue.submit(req, campaign())


def test_cancelled_request_requires_explicit_resubmission_then_repair(cohort):
    from test_forge_queue import SLOTS, campaign, finish, grade, request as queue_request
    from experiments.forge.queue import Queue
    root, requests = cohort
    req = queue_request(root, "idea", cap=1)
    requests["idea"] = req
    queue = Queue(root / "queue", report_root=root / "reports/forge", grader=grade)
    first = queue.submit(req, campaign())
    request_id = first["request"]["request_id"]
    claim = queue.claim(SLOTS)
    queue.cancel(request_id)
    finish(claim, status="cancelled", elapsed=.5)
    queue.collect()
    key = claim["job"]["compatibility_key"]
    original = root / "reports/forge/attempts" / claim["worker"]["attempt"] / "result.json"
    original_bytes = original.read_bytes()
    with pytest.raises(ValueError, match="open subscribed request"):
        queue.retry(key, reason="Repair without authorization")
    resumed = queue.submit(req, campaign())
    state = queue.inspect()
    assert resumed["request"]["request_id"] == request_id
    assert len(state["submissions"]) == 1
    assert request_id in state["jobs"][key]["subscribers"]
    assert state["jobs"][key]["result"]["raw"]["attempt_status"] == "cancelled"
    assert queue.claim(SLOTS) is None  # Re-enqueue preserves the stopped result.
    queue.retry(key, reason="Explicitly repaired cancelled execution")
    repaired = queue.claim(SLOTS)
    assert repaired["request_id"] == request_id
    assert repaired["retry_of"]["attempt_id"] == claim["worker"]["attempt"]
    finish(repaired, elapsed=.75)
    queue.collect()
    state = queue.inspect()
    assert len(state["jobs"][key]["attempts"]) == 2
    assert state["campaigns"]["pilot"]["spent_seconds"] == 1.25
    assert original.read_bytes() == original_bytes


def test_cli_supersession_and_board_preserve_failed_verdict_cost_and_successor(cohort, capsys):
    from test_forge_knowledge import make_task
    from experiments.forge.__main__ import main
    root, requests = cohort
    # The real board validates declarations before the synthetic resolver.
    # Exercise that validation rather than mocking away a malformed fixture.
    for name, request in requests.items():
        assert planning.load_idea(root, name) == request["candidate"]
    task = make_task("cheap")
    view = {"schema_version": 1, "id": "stability", "revision": 1, "goal": "stability", "eligibility": {},
            "assignments": [{"task": "cheap", "qualification_tier": 1, "importance": "required", "order": 0}]}
    atomic_json(root / "configs/forge/tasks/cheap.json", task)
    atomic_json(root / "configs/forge/views/stability.json", view)
    for request in requests.values():
        request.update(tasks={"cheap": task}, view=view, protocol={"id": "fixture", "seed": 0})
    directory = attempt(root, requests["idea"])
    result = read_json(directory / "result.json")
    result["task_results"][0]["evidence"] = {"observations": [{"step": step, "error": 2.} for step in range(1, 25)],
                                              "live": {"error": 2.}}
    atomic_json(directory / "result.json", result)
    certificate = read_json(directory / "evidence.json")
    certificate["result_hash"] = stable_hash(result)
    atomic_json(directory / "evidence.json", certificate)
    explanation = knowledge.readout(root, "idea", "Fails smoke", "Worse than the fixed control", "Test the successor")
    before = next(row for row in knowledge.board(root, "stability")["current_rows"] if row["candidate_id"] == "idea")
    readout_path = root / "reports/forge/records" / (explanation["record_id"] + ".json")
    preserved = readout_path.read_bytes()
    assert main(["--root", str(root), "--queue-root", str(root / "queue"), "supersede", "idea",
                 "--successor", "successor", "--reason", "Address the measured failure"]) == 0
    record = json.loads(capsys.readouterr().out)
    board = knowledge.board(root, "stability")
    after = next(row for row in board["current_rows"] if row["candidate_id"] == "idea")
    assert before["status"] == after["status"] == "FAIL"
    assert before["qualified_tier"] == after["qualified_tier"] == 0
    assert before["cost"] == after["cost"] and after["cost"]["wall_seconds"] == 2.5
    assert before["counts"] == after["counts"] == {"FAIL": 1}
    assert after["lifecycle"] == "superseded"
    assert after["successor"] == "successor@revision-b"
    assert after["disposition_record_id"] == record["record_id"]
    assert after["disposition_reason"] == "Address the measured failure"
    assert not after["pending_readout"]
    assert readout_path.read_bytes() == preserved
    compiled = read_json(root / "reports/forge/leaderboards/stability.json")
    assert next(row for row in compiled["current_rows"] if row["candidate_id"] == "idea")["successor"] == "successor@revision-b"
    assert next(row for row in board["current_rows"] if row["candidate_id"] == "successor")["qualified_tier"] == 0
