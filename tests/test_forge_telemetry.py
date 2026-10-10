"""Measured telemetry and deterministic accounting without experiment training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.forge import telemetry as t
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from test_forge_calibration import current_attempt, current_request


def test_nested_phase_timing_is_exclusive_and_sampling_is_not_double_counted():
    clock = iter((0., 2., 5., 10., 12., 16.))
    timer = t.PhaseTimer(clock=lambda: next(clock))
    with timer.measure("evaluation"):
        with timer.measure("sampling"):
            pass
    with timer.measure("training_updates"):
        pass
    saved = timer.snapshot()
    assert saved["phases"] == {"training_updates": {"seconds": 4., "calls": 1},
        "evaluation": {"seconds": 7., "calls": 1}, "sampling": {"seconds": 3., "calls": 1}}
    assert saved["measured_seconds"] == 14.


def test_interrupted_phase_still_records_cost_without_swallowing_failure():
    clock = iter((1., 4.))
    calls = []
    timer = t.PhaseTimer(clock=lambda: next(clock), synchronize=lambda: calls.append("synchronized"))
    with pytest.raises(RuntimeError, match="failed"):
        with timer.measure("training_updates"):
            raise RuntimeError("failed")
    assert timer.snapshot()["phases"]["training_updates"] == {"seconds": 3., "calls": 1}
    assert len(calls) == 2


def test_memory_probe_reports_distinct_process_and_allocator_scopes(monkeypatch):
    calls = []
    cuda = SimpleNamespace(init=lambda: None, reset_peak_memory_stats=lambda device: calls.append(device),
        max_memory_allocated=lambda device: 100, max_memory_reserved=lambda device: 200)
    monkeypatch.setattr(t, "peak_rss", lambda: 300)
    measured = t.MemoryProbe(device="cuda:0", torch_module=SimpleNamespace(cuda=cuda, device=torch.device)).snapshot()
    assert calls == [torch.device("cuda:0")] and measured["process_peak_rss_bytes"] == 300
    assert measured["cuda_peak_allocated_bytes"] == 100
    assert measured["cuda_peak_reserved_bytes"] == 200
    unavailable = t.MemoryProbe(device="cuda:0").snapshot()
    assert unavailable["cuda_peak_allocated_bytes"] is None
    assert unavailable["cuda_unavailable_reason"]
    assert t.MemoryProbe().snapshot()["cuda_peak_reserved_bytes"] is None


@pytest.mark.parametrize("device,index", [("0", 0), (0, 0), ("cuda:0", 0), (torch.device("cuda:0"), 0),
                                         ("1", 1), (torch.device("cuda:1"), 1),
                                         ("cuda", 2), (torch.device("cuda"), 2)])
def test_cuda_probe_initializes_allocator_and_canonicalizes_worker_devices(device, index):
    calls, state = [], {"initialized": False, "current": 2}
    def initialize():
        calls.append("init")
        state["initialized"] = True
    def allocator(operation, resolved):
        # The real reset API can fail this way before its lazy CUDA state exists.
        if not state["initialized"]:
            raise RuntimeError("Invalid device argument ")
        assert isinstance(resolved, torch.device) and resolved == torch.device("cuda", index)
        calls.append(operation)
        return {"reset": None, "allocated": 100 + index, "reserved": 200 + index}[operation]
    cuda = SimpleNamespace(init=initialize, current_device=lambda: state["current"],
        reset_peak_memory_stats=lambda resolved: allocator("reset", resolved),
        max_memory_allocated=lambda resolved: allocator("allocated", resolved),
        max_memory_reserved=lambda resolved: allocator("reserved", resolved))
    module = SimpleNamespace(cuda=cuda, device=torch.device)
    probe = t.MemoryProbe(device=device, torch_module=module)
    state["current"] = 7  # A later current-device change cannot redirect this probe.
    measured = probe.snapshot()
    assert calls == ["init", "reset", "allocated", "reserved"]
    assert measured["cuda_peak_allocated_bytes"] == 100 + index
    assert measured["cuda_peak_reserved_bytes"] == 200 + index
    assert "cuda_unavailable_reason" not in measured


@pytest.mark.parametrize("stage", ["init", "reset", "allocated", "reserved"])
def test_cuda_probe_preserves_real_instrumentation_failures_as_unavailable(stage):
    def invoke(operation, *args):
        if operation == stage:
            raise RuntimeError("unavailable at " + stage)
        return 123
    cuda = SimpleNamespace(init=lambda: invoke("init"),
        reset_peak_memory_stats=lambda resolved: invoke("reset", resolved),
        max_memory_allocated=lambda resolved: invoke("allocated", resolved),
        max_memory_reserved=lambda resolved: invoke("reserved", resolved))
    measured = t.MemoryProbe(device="0", torch_module=SimpleNamespace(cuda=cuda, device=torch.device)).snapshot()
    assert measured["cuda_peak_allocated_bytes"] is None and measured["cuda_peak_reserved_bytes"] is None
    assert measured["cuda_unavailable_reason"] == "unavailable at " + stage


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_cpu_probe_never_initializes_or_queries_cuda(device):
    cuda = SimpleNamespace(init=lambda: pytest.fail("CPU telemetry must not initialize CUDA"),
                           reset_peak_memory_stats=lambda *_: pytest.fail("CPU telemetry must not query CUDA"))
    measured = t.MemoryProbe(device=device, torch_module=SimpleNamespace(cuda=cuda, device=torch.device)).snapshot()
    assert measured["cuda_peak_allocated_bytes"] is None and measured["cuda_peak_reserved_bytes"] is None
    assert "cuda_unavailable_reason" not in measured


def test_legacy_inclusive_time_is_renamed_recursively_without_mutation():
    raw = {"cost": {"training_seconds": 10.}, "task_results": {"extension": {"cost": {"training_seconds": 10.}}}}
    normalized = t.normalize_adapter_costs(raw)
    assert "training_seconds" not in normalized["cost"]
    assert normalized["cost"]["adapter_reported_inclusive_seconds"] == 10.
    assert "training_seconds" not in normalized["task_results"]["extension"]["cost"]
    assert raw["cost"]["training_seconds"] == 10.


def test_observed_concurrency_counts_overlaps_and_distinct_physical_devices():
    rows = [{"attempt_id": "a", "device": "0", "started_at": 0., "finished_at": 4.},
            {"attempt_id": "b", "device": "1", "started_at": 2., "finished_at": 6.},
            {"attempt_id": "c", "device": "1", "started_at": 4., "finished_at": 5.},
            {"attempt_id": "bad", "device": "2", "started_at": 9., "finished_at": 8.}]
    result = t.observed_concurrency(rows)
    assert result["peak_workers"] == result["peak_distinct_gpu_devices"] == 2
    assert result["measured_attempts"] == 3
    assert result["mean_active_workers"] == 1.5
    assert t.observed_concurrency([])["peak_workers"] is None


def request(name="candidate", cap=2):
    value = current_request(name)
    value.update(through_tier=cap, request_id=name + "-request")
    value["view"].update(schema_version=1, id="quality", revision=1, goal="quality", eligibility={})
    for index, assignment in enumerate(value["view"]["assignments"]):
        assignment["order"] = index
    for task in value["tasks"].values():
        task["resources"] = {}
    return value


def save(root, req, name, *, score=1., status="PASS", elapsed=4., tasks=("cheap",), interval=None, retry=None):
    result = current_attempt(root, req, name, score=score, status=status,
                             omit=tuple(set(req["tasks"]) - set(tasks)), retry=retry)
    result["cost_owner"] = {"request": req["request_id"], "revision": req["candidate_revision"], "campaign": "fixture"}
    result["raw"] = {"attempt_status": "completed" if status in {"PASS", "FAIL"} else "error",
        "elapsed_seconds": elapsed, "result": {"cost": {"phase_timing": {"phases": {
            "training_updates": {"seconds": 1.}, "evaluation": {"seconds": .5}, "sampling": {"seconds": .25}}}},
            "telemetry": {"memory": {"process_peak_rss_bytes": 12345}}}}
    if interval:
        result["raw"]["telemetry"] = {"interval": interval}
    directory = root / "reports/forge/attempts" / name
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result), "source": req["source"], "runtime": req["runtime"]})
    return result


def state(root, requests, results):
    queue_root = root / "runs/telemetry-fixture"
    jobs = {}
    for req in requests:
        for job in req["jobs"]:
            key = job["compatibility_key"]
            matches = [result for result in results if any(r["compatibility_key"] == key for r in result["task_results"])]
            jobs[key] = {"definition": job, "status": "terminal" if matches else "pending",
                "attempts": [{"attempt_id": r["attempt_id"]} for r in matches], "result": matches[-1] if matches else None}
    atomic_json(queue_root / "queue/state.json", {"jobs": jobs,
        "submissions": {req["request_id"]: {"request": req, "status": "blocked"} for req in requests},
        "charges": [{"attempt_id": r["attempt_id"], "seconds": r["raw"]["elapsed_seconds"], "owner": r["cost_owner"]} for r in results]})
    return queue_root


def test_failed_tier_records_cost_and_only_requested_unattempted_work_as_avoided(tmp_path):
    req = request()
    failed = save(tmp_path, req, "smoke", score=0., status="FAIL", elapsed=3.,
                  interval={"started_at": 10., "finished_at": 13., "device": "cpu"})
    queue = state(tmp_path, [req], [failed])
    report = t.summarize_automation(tmp_path, queue)
    assert report["spend"]["wall_seconds"] == 3.
    assert report["tier_rejections"] == {"1": 1}
    assert report["tasks_avoided"] == report["execution_jobs_avoided"] == 1
    assert report["cost_to_reject"]["mean_seconds"] == 3.
    assert report["qualification_units"][0]["status"] == "FAIL"
    assert report["concurrency"]["peak_workers"] == 1
    assert report["memory"]["process_peak_rss_bytes"]["maximum_bytes"] == 12345
    assert report["component_timing"]["seconds"] == {"training_updates": 1., "evaluation": .5, "sampling": .25}
    assert not report["public_default_claim"] and report["flops"]["value"] is None
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert t.summarize_automation(tmp_path, queue) == report
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_cap_excluded_work_is_never_counted_as_avoided(tmp_path):
    req = request(cap=1)
    failed = save(tmp_path, req, "smoke", score=0., status="FAIL")
    report = t.summarize_automation(tmp_path, state(tmp_path, [req], [failed]))
    assert report["tasks_avoided"] == 0


def test_requested_cap_above_existing_view_does_not_invent_a_new_qualified_tier(tmp_path):
    req = request(cap=3)  # This view has only two declared tiers.
    result = save(tmp_path, req, "complete", tasks=("cheap", "quality"))
    report = t.summarize_automation(tmp_path, state(tmp_path, [req], [result]))
    assert report["qualification_units"][0]["status"] == "PASS"
    assert report["qualification_units"][0]["qualified_tier"] == 2


def test_shared_subscribers_reuse_evidence_without_duplicate_spend_or_outcomes(tmp_path):
    req = request()
    second = deepcopy(req)
    second["request_id"] = "second-subscriber"
    second["candidate"]["id"] = "renamed"
    result = save(tmp_path, req, "qualified", tasks=("cheap", "quality"), elapsed=5.)
    report = t.summarize_automation(tmp_path, state(tmp_path, [req, second], [result]))
    assert report["spend"]["wall_seconds"] == 5.
    assert len(report["qualification_units"]) == 1
    assert report["cost_to_qualify"]["mean_seconds"] == 5.
    assert report["reuse"] == {"submitted_execution_demands": 4, "evidence_uses": 4,
                              "unique_evidence_jobs": 2, "reused_evidence_uses": 2}


def test_frozen_verdicts_are_observational_not_regraded_with_current_code(tmp_path, monkeypatch):
    req = request(cap=1)
    # A pinned result can differ from the current evaluator: retain its certified
    # frozen verdict in telemetry without making any current qualification claim.
    result = save(tmp_path, req, "pinned", score=0., status="PASS")
    import experiments.forge.views as views
    monkeypatch.setattr(views, "grade_result", lambda *a, **k: pytest.fail("live grading forbidden"))
    monkeypatch.setattr(views, "qualify", lambda *a, **k: pytest.fail("live qualification forbidden"))
    report = t.summarize_automation(tmp_path, state(tmp_path, [req], [result]))
    assert report["qualification_units"][0]["status"] == "PASS"
    assert "Recorded attainment" in report["qualification_scope"]
    assert not report["public_default_claim"]


def test_missing_queue_cap_cost_and_intervals_are_explicitly_unavailable(tmp_path):
    req = request()
    req.pop("through_tier")
    result = save(tmp_path, req, "legacy", elapsed=None)
    report = t.summarize_automation(tmp_path, tmp_path / "moved-queue")
    assert not report["queue_available"] and report["tasks_avoided"] is None
    assert report["reuse"]["evidence_uses"] is None
    assert report["requests_with_unavailable_cap"] == 1
    assert report["qualification_units"] == []
    assert report["spend"]["wall_seconds"] is None and report["spend"]["unknown_cost_attempts"] == 1
    assert report["concurrency"]["peak_workers"] is None
    assert report["concurrency"]["missing_interval_attempts"] == 1


def test_certified_retry_replaces_incomplete_attainment_but_retains_every_paid_attempt(tmp_path):
    req = request(cap=1)
    old = save(tmp_path, req, "error", status="INCOMPLETE", elapsed=2.)
    retry = {"attempt_id": "error", "result_hash": stable_hash(old), "reason": "infrastructure repaired", "authorized_at": "fixture"}
    new = save(tmp_path, req, "repair", retry=retry, elapsed=3.)
    report = t.summarize_automation(tmp_path, state(tmp_path, [req], [old, new]))
    assert report["qualification_units"][0]["status"] == "PASS"
    assert report["cost_to_qualify"]["mean_seconds"] == 5.
    assert report["execution_error_rate"] == {"numerator": 1, "denominator": 2, "fraction": .5}
    assert report["incomplete_task_rate"]["numerator"] == 1


def test_invalid_receipt_cannot_qualify_and_unknown_charge_disagreement_is_visible(tmp_path):
    req = request(cap=1)
    result = save(tmp_path, req, "invalid")
    queue = state(tmp_path, [req], [result])
    (tmp_path / "reports/forge/attempts/invalid/evidence.json").unlink()
    report = t.summarize_automation(tmp_path, queue)
    assert report["qualification_units"][0]["status"] == "INVALID"
    assert report["attempt_outcomes"] == {"invalid_receipt": 1}
    assert report["spend"]["wall_seconds"] == 4.  # paid queue charge is still real
    assert report["incomplete_task_rate"]["fraction"] == 1.
    result = save(tmp_path, req, "invalid", elapsed=5.)
    report = t.summarize_automation(tmp_path, queue)
    assert report["spend"]["wall_seconds"] is None
    assert any("disagree" in row["reason"] for row in report["receipt_issues"])


def test_diagnostic_and_promotion_receipts_never_confer_ordinary_attainment(tmp_path):
    req = request(cap=1)
    req["calibration_lane"] = {"registration_id": "fixture"}
    result = save(tmp_path, req, "diagnostic")
    report = t.summarize_automation(tmp_path, state(tmp_path, [req], [result]))
    assert report["qualification_units"] == []
    assert report["cost_to_qualify"]["outcomes"] == 0
    assert report["spend"]["wall_seconds"] == 4.


def test_runtime_normalizes_aggregate_cost_and_records_memory_without_training(tmp_path, monkeypatch):
    import experiments.forge.adapters as adapters
    from experiments.forge import runtime
    import torch
    req = request(cap=1)
    req["candidate"]["claim_contract"] = {"scoring_weights": "live"}
    req["campaign_id"] = "fixture"
    job = req["jobs"][0]
    job["resources"] = {"cpu_threads": 1}
    path = tmp_path / "request.json"
    atomic_json(path, {"request": req, "job": job, "worker": {"attempt": "runtime-fixture", "device": "cpu"}})
    monkeypatch.setattr(adapters, "run_task", lambda *args: {"cost": {"training_seconds": .1}, "evidence": {}})
    previous = torch.get_num_threads()
    try:
        assert runtime.execute(path) == 0
    finally:
        torch.set_num_threads(previous)
    result = read_json(tmp_path / "raw-result.json")
    assert "training_seconds" not in result["cost"]
    assert result["cost"]["adapter_reported_inclusive_seconds"] == .1
    assert result["telemetry"]["memory"]["process_peak_rss_bytes"] > 0
    assert result["telemetry"]["timing"]["unattributed_runner_seconds"] is None
    assert result["cost"]["runner_seconds"] >= result["cost"]["adapter_seconds"]


def test_real_supervised_process_receipt_has_actual_interval_and_peak_memory(tmp_path):
    from test_forge_workers import worker_request
    from test_forge_queue import campaign, grade, SLOTS
    from experiments.forge.queue import Queue
    req = worker_request(tmp_path, '''
from pathlib import Path
import sys
from .contracts import atomic_json
atomic_json(Path(sys.argv[1]).parent / "raw-result.json", {"measured": 1})
''', budget=10)
    queue = Queue(tmp_path / "queue", grader=grade)
    queue.submit(req, campaign())
    claim = queue.claim(SLOTS)
    # This receipt control includes two real Torch-importing processes. Its
    # software allowance accommodates cold startup during concurrent jobs.
    process = queue.launch(claim)
    try:
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            queue.cancel(claim["request"]["request_id"])
            process.wait(timeout=10)
    terminal = read_json(Path(claim["worker"]["directory"]) / "terminal.json")
    measured = terminal["telemetry"]
    assert measured["interval"]["finished_at"] >= measured["interval"]["started_at"]
    assert measured["runner_process_seconds"] > 0 and measured["grading_process_seconds"] is None
    assert measured["maximum_reaped_child_peak_rss_bytes"] > 0
