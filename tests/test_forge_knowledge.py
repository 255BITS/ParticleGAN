from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge import knowledge
from experiments.forge.contracts import atomic_json, read_json, stable_hash


def make_task(name):
    return {"schema_version": 1, "id": name, "adapter": "test", "execution": {"steps": 24},
            "evaluation": {"kind": "transfer_sustained", "thresholds": [["error", "<=", 1.0]],
                           "scoring_weights": "live"}, "resources": {}, "dependencies": [], "requires_capabilities": []}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tasks = {name: make_task(name) for name in ("cheap", "quality")}
    view = {"schema_version": 1, "id": "stability", "revision": 1, "goal": "stability", "eligibility": {},
            "assignments": [{"task": "cheap", "qualification_tier": 1, "importance": "required", "order": 0},
                            {"task": "quality", "qualification_tier": 2, "importance": "required", "order": 0}]}
    for name, task in tasks.items():
        atomic_json(tmp_path / f"configs/forge/tasks/{name}.json", task)
    atomic_json(tmp_path / "configs/forge/views/stability.json", view)
    candidate = {"id": "idea", "goal": "stability", "hypothesis": "Test a mechanism", "mechanism_class": "structural"}
    atomic_json(tmp_path / "configs/forge/ideas/idea.json", candidate)
    request = {"candidate": candidate, "candidate_revision": "revision-a", "source": {"digest": "source-a"},
               "runtime": {"python": "fixed"}, "protocol": {"id": "fixed"}, "view": view, "tasks": tasks,
               "jobs": [{"task_id": name, "task_ids": [name], "compatibility_key": name + "-key"} for name in tasks]}
    monkeypatch.setattr(knowledge, "_current_request", lambda *args: deepcopy(request))
    return tmp_path, request


def save_attempt(root, request, task_id="cheap", *, name="attempt-one", final=.5, stamp="PASS"):
    row = {"task_id": task_id, "compatibility_key": task_id + "-key", "gate_status": stamp,
           "evidence": {"observations": [{"step": i, "error": final} for i in range(1, 25)],
                        "live": {"error": final}}, "cost": {"wall_seconds": 2.5}, "raw_status": "completed"}
    result = {"schema_version": 1, "attempt_id": name, "candidate_revision": request["candidate_revision"], "task_results": [row]}
    directory = root / "reports/forge/attempts" / name
    atomic_json(directory / "request.json", {"request": request})
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result), "source": request["source"],
                                             "runtime": request["runtime"]})
    return directory


def historical(root):
    record = {"schema_version": 1, "record_id": "history-a", "candidate_id": "idea", "candidate_revision": "revision-a",
              "evidence_scope": "historical", "record_type": "scientific", "source": {"path": "reports/old.json"},
              "hypothesis": "Critic anchor experiment", "conclusion": "Useful quality with a late failure.",
              "next_action": "Test sustained hold.", "task_results": [{"task_id": "cheap", "gate_status": "PASS", "cost": {"seconds": 8}}]}
    atomic_json(root / "reports/forge/records/history-a.json", record)
    return record


def test_historical_pass_never_promotes_matching_candidate(setup):
    root, _ = setup
    historical(root)
    result = knowledge.board(root, "stability")
    assert result["current_rows"][0]["qualified_tier"] == 0
    assert result["historical_rows"][0]["qualified_tier"] is None
    assert result["historical_rows"][0]["cost"]["wall_seconds"] == 8


def test_current_curve_is_regraded_and_readout_pending(setup):
    root, request = setup
    save_attempt(root, request, final=2., stamp="PASS")
    row = knowledge.board(root, "stability")["current_rows"][0]
    assert row["status"] == "FAIL"
    assert row["qualified_tier"] == 0
    assert row["pending_readout"]
    assert row["cost"]["wall_seconds"] == 2.5


def test_changed_compatibility_key_does_not_reuse_old_pass(setup):
    root, request = setup
    save_attempt(root, request)
    request["jobs"][0]["compatibility_key"] = "new-key"
    result = knowledge.board(root, "stability")
    assert result["current_rows"][0]["qualified_tier"] == 0
    assert len(result["archived_attempts"]) == 1
    assert result["pinned_rows"][0]["counts"] == {"PASS": 1}
    assert result["pinned_rows"][0]["cost"]["wall_seconds"] == 2.5
    assert result["pinned_rows"][0] in result["rows"]


def test_retiering_reuses_unchanged_evidence_without_queue_side_effect(setup):
    root, request = setup
    save_attempt(root, request)
    save_attempt(root, request, "quality", name="attempt-two")
    before = knowledge.board(root, "stability")
    view_path = root / "configs/forge/views/stability.json"
    view = read_json(view_path)
    view["assignments"][1]["qualification_tier"] = 1
    view["revision"] = 2
    atomic_json(view_path, view)
    after = knowledge.board(root, "stability")
    assert before["current_rows"][0]["qualified_tier"] == 2
    assert after["current_rows"][0]["qualified_tier"] == 1
    assert after["current_rows"][0]["qualification"]["required_passed"] == 2
    assert not (root / "runs").exists()


def test_missing_or_tampered_durable_certificate_blocks_qualification(setup):
    root, request = setup
    directory = save_attempt(root, request)
    result = read_json(directory / "result.json")
    result["task_results"][0]["evidence"]["live"]["error"] = .1
    atomic_json(directory / "result.json", result)
    board = knowledge.board(root, "stability")
    assert board["conflicts"]
    assert board["current_rows"][0]["qualified_tier"] == 0
    assert board["current_rows"][0]["status"] == "BLOCKED"


def test_conflicting_compatible_attempts_remain_invalid(setup):
    root, request = setup
    save_attempt(root, request, final=.5)
    save_attempt(root, request, name="attempt-two", final=2.)
    row = knowledge.board(root, "stability")["current_rows"][0]
    assert row["status"] == "INVALID"
    assert row["qualified_tier"] == 0


def repair_attempt(root, request, *, prior_status="error", prior_gate="INCOMPLETE"):
    previous = save_attempt(root, request, name="original", stamp=prior_gate)
    prior = read_json(previous / "result.json")
    prior["raw"] = {"attempt_status": prior_status}
    prior["task_results"][0]["raw_status"] = prior_status
    atomic_json(previous / "result.json", prior)
    atomic_json(previous / "evidence.json", {"result_hash": stable_hash(prior),
        "source": request["source"], "runtime": request["runtime"]})
    retry = save_attempt(root, request, name="repair")
    link = {"attempt_id": "original", "result_hash": stable_hash(prior),
            "reason": "Repaired interrupted worker", "authorized_at": "2026-09-29T00:00:00Z"}
    result = read_json(retry / "result.json")
    result["retry_of"] = link
    atomic_json(retry / "result.json", result)
    atomic_json(retry / "request.json", {"request": request, "retry_of": link})
    atomic_json(retry / "evidence.json", {"result_hash": stable_hash(result),
        "source": request["source"], "runtime": request["runtime"]})
    return previous, retry


def test_repaired_execution_qualifies_but_preserves_error_and_all_costs(setup):
    root, request = setup
    repair_attempt(root, request)
    board = knowledge.board(root, "stability")
    assert not board["conflicts"]
    row = board["current_rows"][0]
    assert row["qualified_tier"] == 1
    assert row["cost"]["wall_seconds"] == 5
    assert row["attempt_ids"] == ["original", "repair"]
    assert row["retry_history"][0]["superseded_by"] == "repair"
    assert row["retry_history"][0]["task_results"][0]["gate_status"] == "INCOMPLETE"
    request["candidate_revision"] = "new-source"
    pinned = knowledge.board(root, "stability")["pinned_rows"][0]
    assert pinned["counts"] == {"PASS": 1}
    assert pinned["cost"]["wall_seconds"] == 5
    assert pinned["retry_history"]


@pytest.mark.parametrize("prior_status,prior_gate", [("completed", "FAIL"), ("error", "BLOCKED"),
                                                     ("completed", "PASS"), ("error", "INVALID")])
def test_retry_cannot_hide_scientific_or_applicability_outcomes(setup, prior_status, prior_gate):
    root, request = setup
    repair_attempt(root, request, prior_status=prior_status, prior_gate=prior_gate)
    result = knowledge.board(root, "stability")
    assert result["conflicts"]
    assert result["current_rows"][0]["qualified_tier"] == 0
    assert not result["current_rows"][0]["retry_history"]


def test_missing_or_changed_prior_receipt_invalidates_retry(setup):
    root, request = setup
    previous, _ = repair_attempt(root, request)
    (previous / "result.json").unlink()
    result = knowledge.board(root, "stability")
    assert result["conflicts"]
    assert result["current_rows"][0]["qualified_tier"] == 0


def test_unlinked_infrastructure_failure_is_not_silently_ignored(setup):
    root, request = setup
    save_attempt(root, request, name="original", stamp="INCOMPLETE")
    save_attempt(root, request, name="unlinked-success")
    assert knowledge.board(root, "stability")["current_rows"][0]["qualified_tier"] == 0


def test_calibration_diagnostics_never_qualify_but_keep_costs_and_readout(setup):
    root, request = setup
    diagnostic = deepcopy(request)
    # Deliberately identical keys test the reducer's additional scope guard;
    # production registration also places jobs in a separate queue namespace.
    diagnostic["calibration_lane"] = {"registration_id": "fixture", "qualification_reuse": False}
    save_attempt(root, diagnostic)
    board = knowledge.board(root, "stability")
    assert board["current_rows"][0]["qualified_tier"] == 0
    assert not board["pinned_rows"]
    row = board["calibration_rows"][0]
    assert row["status"] == "DIAGNOSTIC" and row["qualification_reuse"] is False
    assert row["cost"]["wall_seconds"] == 2.5
    assert row["counts"] == {"PASS": 1}
    assert knowledge.compile_memory(root)["pending_readout"] == ["idea"]
    record = knowledge.readout(root, "idea", "Diagnostic only", "No promotion", "Complete calibration")
    assert record["evidence_scope"] == "calibration_diagnostic" and not record["qualification_reuse"]
    assert not knowledge.compile_memory(root)["pending_readout"]


def test_compilation_is_deterministic_and_never_ingests_generated_memory(setup):
    root, request = setup
    historical(root)
    first = knowledge.compile_memory(root)
    before = (root / "reports/forge/EXPERIMENT_MEMORY.md").read_bytes()
    (root / "reports/forge/EXPERIMENT_MEMORY.md").write_text("Injected generated content must not become input")
    second = knowledge.compile_memory(root)
    assert first == second
    assert (root / "reports/forge/EXPERIMENT_MEMORY.md").read_bytes() == before
    inputs = read_json(root / "reports/forge/compilation.json")["input_hashes"]
    assert "reports/forge/EXPERIMENT_MEMORY.md" not in inputs
    assert "reports/forge/compilation.json" not in inputs


def test_compiler_links_to_single_published_table_instead_of_creating_another(setup):
    root, _ = setup
    current = root / "reports/forge/technique-inventory.md"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text("# Current technique leaderboard\n")
    atomic_json(current.with_suffix(".json"), {"publication_scope": "current_technique_inventory", "view": "stability"})
    duplicate = root / "reports/forge/leaderboards/stability.md"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_text("Old generated goal table\n")
    knowledge.compile_memory(root)
    assert current.read_text() == "# Current technique leaderboard\n"
    assert not duplicate.exists()
    assert "[stability](technique-inventory.md)" in (root / "reports/forge/EXPERIMENT_MEMORY.md").read_text()
    # The full qualification reducer remains available as numerical evidence.
    assert (root / "reports/forge/leaderboards/stability.json").is_file()
    assert knowledge.leaderboard_path(root, "another-goal") == Path("reports/forge/leaderboards/another-goal.md")


def test_recall_returns_useful_negative_history_and_unknown_goal(setup):
    root, _ = setup
    historical(root)
    result = knowledge.recall(root, "anchor", "discriminator_stability")
    assert result[0]["candidate_id"] == "idea"
    assert "late failure" in result[0]["conclusion"]
    assert result[0]["goal_applicability"] == "unknown"


def test_readout_is_bound_to_exact_revision_and_updates_memory(setup):
    root, request = setup
    save_attempt(root, request, final=2.)
    record = knowledge.readout(root, "idea", "Fails smoke", "Worse than control", "Revise the mechanism")
    assert record["lifecycle"] == "concluded"
    assert record["candidate_revision"] == "revision-a"
    assert knowledge.board(root, "stability")["current_rows"][0]["pending_readout"] is False
    assert "Fails smoke" in (root / "reports/forge/EXPERIMENT_MEMORY.md").read_text()


def test_readout_rejects_ambiguous_revision_and_empty_recommendation(setup):
    root, request = setup
    save_attempt(root, request)
    other = deepcopy(request)
    other["candidate_revision"] = "revision-b"
    save_attempt(root, other, name="attempt-two")
    with pytest.raises(ValueError, match="Multiple revisions"):
        knowledge.readout(root, "idea", "result", "comparison", "next")
    with pytest.raises(ValueError, match="requires"):
        knowledge.readout(root, "idea@revision-a", "result", "comparison", "")


def test_grouped_task_cost_is_charged_once():
    rows = [{"_attempt_id": "shared", "cost": {"wall_seconds": 10}} for _ in range(2)]
    assert knowledge._cost(rows)["wall_seconds"] == 10


def test_cpu_cuda_cohorts_never_pool_partial_tiers(setup, monkeypatch):
    root, request = setup
    cpu, cuda = deepcopy(request), deepcopy(request)
    for backend, resolved in (("cpu", cpu), ("cuda", cuda)):
        resolved["execution_backend"] = backend
        resolved["compute_profiles"] = {backend: {"backend": backend, "model": backend + "-machine"}}
    # Even an old sparse key scheme must not combine backend cohorts.
    save_attempt(root, cpu, name="cpu-cheap")
    save_attempt(root, cuda, task_id="quality", name="cuda-quality")
    monkeypatch.setattr(knowledge, "_current_request", lambda root, idea, view, backend, model: deepcopy(cpu if backend == "cpu" else cuda))
    # Real keys differ by backend; reflect that in the resolved declarations and receipts.
    for backend, resolved in (("cpu", cpu), ("cuda", cuda)):
        for job in resolved["jobs"]:
            job["compatibility_key"] = backend + "-" + job["compatibility_key"]
    for name, resolved in (("cpu-cheap", cpu), ("cuda-quality", cuda)):
        directory = root / "reports/forge/attempts" / name
        result = read_json(directory / "result.json")
        result["task_results"][0]["compatibility_key"] = resolved["execution_backend"] + "-" + result["task_results"][0]["compatibility_key"]
        atomic_json(directory / "request.json", {"request": resolved})
        atomic_json(directory / "result.json", result)
        atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result), "source": resolved["source"], "runtime": resolved["runtime"]})
    rows = knowledge.board(root, "stability")["current_rows"]
    assert len(rows) == 2
    by_backend = {r["runtime_cohort"]["execution_backend"]: r for r in rows}
    assert by_backend["cpu"]["qualified_tier"] == 1
    assert by_backend["cuda"]["qualified_tier"] == 0
    assert by_backend["cpu"]["attempt_ids"] == ["cpu-cheap"]
    assert by_backend["cuda"]["attempt_ids"] == ["cuda-quality"]


def test_changed_source_stays_pinned_without_live_regrading(setup):
    root, request = setup
    save_attempt(root, request, final=2, stamp="PASS")
    # A malicious/stale unchanged compatibility key cannot bypass source identity.
    request["source"]["digest"] = "changed-grader-source"
    result = knowledge.board(root, "stability")
    assert result["current_rows"][0]["qualified_tier"] == 0
    archived = result["pinned_rows"][0]
    assert archived["status"] == "PASS"  # explicitly recorded, not regraded
    assert archived["qualified_tier"] is None
    assert archived["qualification_reuse"] is False
    assert "never regraded" in archived["grading"]


def queue_state(root, request, status):
    request.update(queue_root=str(root / "external-queue"), request_id="request-one")
    path = Path(request["queue_root"]) / "queue/state.json"
    atomic_json(path, {"schema_version": 1, "submissions": {
        "request-one": {"request": request, "status": status, "lifecycle": "running"}},
        "jobs": {}, "campaigns": {}, "events": []})
    return path


def test_active_queue_suppresses_pending_and_blocks_readout(setup):
    root, request = setup
    state_path = queue_state(root, request, "running")
    save_attempt(root, request)
    row = knowledge.board(root, "stability")["current_rows"][0]
    assert row["lifecycle"] == "running"
    assert row["pending_readout"] is False
    with pytest.raises(ValueError, match="active queued/running/paused"):
        knowledge.readout(root, "idea", "partial", "comparison", "next")
    assert read_json(state_path)["submissions"]["request-one"]["status"] == "running"
    assert not list((root / "reports/forge/records").glob("readout-*.json"))


def test_stopped_readout_persists_concluded_and_retry_needs_new_readout(setup):
    root, request = setup
    state_path = queue_state(root, request, "blocked")
    save_attempt(root, request)
    knowledge.readout(root, "idea", "smoke complete", "comparison", "next")
    entry = read_json(state_path)["submissions"]["request-one"]
    assert entry["status"] == entry["lifecycle"] == "concluded"
    assert knowledge.board(root, "stability")["current_rows"][0]["pending_readout"] is False
    save_attempt(root, request, task_id="quality", name="retry-two")
    row = knowledge.board(root, "stability")["current_rows"][0]
    assert row["pending_readout"] is True
    assert row["lifecycle"] == "awaiting_readout"
    knowledge.readout(root, "idea", "both tasks complete", "comparison", "next")
    assert knowledge.board(root, "stability")["current_rows"][0]["pending_readout"] is False
