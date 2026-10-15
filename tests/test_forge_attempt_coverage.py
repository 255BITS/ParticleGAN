"""Attempt coverage cannot change qualification or duplicate durable work."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from experiments.forge import attempt_coverage as runner
from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.family_reports import _count, score


def test_coverage_retains_qualified_numerator_and_original_missing_counts():
    tasks = [
        dict(status="PASS", execution_recorded=True),
        dict(status="UNKNOWN", execution_recorded=False,
             run_attempt=dict(status="FAIL", attempted=True)),
        dict(status="BLOCKED", execution_recorded=False,
             run_attempt=dict(status="BLOCKED", attempted=True, block_kind="incompatible")),
        dict(status="UNKNOWN", execution_recorded=False,
             run_attempt=dict(status="BLOCKED", attempted=True, block_kind="prerequisite")),
        dict(status="UNKNOWN", execution_recorded=False,
             run_attempt=dict(status="PENDING", attempted=False)),
    ]
    before = deepcopy(tasks)
    result = _count(tasks)
    assert result["passed"] == 1 and result["not_run"] == 3 and result["incompatible"] == 1
    assert score(result) == "1/5 (1 not run, 1 incompatible, 1 blocked, 1 diagnostic attempts)"
    assert tasks == before


def test_recovery_preserves_live_workers_and_never_replays_interrupted_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "process_identity", lambda pid: "live" if pid == 10 else None)
    live = dict(status="RUNNING", directory=str(tmp_path / "live"), pid=10, process_identity="live")
    gone = dict(status="RUNNING", directory=str(tmp_path / "gone"), pid=11, process_identity="old")
    finished = dict(status="RUNNING", directory=str(tmp_path / "finished"), pid=12, process_identity="old")
    atomic_json(tmp_path / "finished/result.json", dict(status="BLOCKED", attempted=True, reason="constructor"))
    (tmp_path / "finished/run.log").write_text("actual constructor traceback")
    state = dict(jobs=dict(live=live, gone=gone, finished=finished))
    runner.recover(state)
    assert live["status"] == "RUNNING"
    assert gone["status"] == "INTERRUPTED" and gone["attempted"]
    assert finished["status"] == "BLOCKED" and finished["result_sha256"] and finished["log_sha256"]
    runner.recover(state)
    assert gone["status"] == "INTERRUPTED"


def test_scheduler_caps_four_tests_two_per_gpu_and_resumes_without_replay(tmp_path, monkeypatch):
    root, directory = tmp_path / "repo", tmp_path / "runs"
    jobs = {str(i): dict(key=str(i), family="fixture", task=f"task{i}", status="PENDING",
                         resources=dict(gpus=1, gpu_memory_mb=1), dependencies=[]) for i in range(9)}
    atomic_json(directory / "state.json", dict(jobs=jobs))
    monkeypatch.setattr(runner, "device_slots", lambda *a, **kw: [
        dict(device=gpu, slot=slot, memory_mb=100) for gpu in ("0", "1") for slot in (0, 1)])
    monkeypatch.setattr(runner, "process_identity", lambda pid: str(pid))
    monkeypatch.setattr(runner.time, "sleep", lambda delay: None)
    starts = []

    class Process:
        def __init__(self, command, **kwargs):
            path = Path(command[-1]).parent
            starts.append(path)
            self.pid = 100 + len(starts)
            atomic_json(path / "result.json", dict(status="BLOCKED", attempted=True, reason="fixture"))

        def poll(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    assert runner.drain(root, directory, gpus=["0", "1"])["pending"] == 0
    active = {"0": 0, "1": 0}
    for event in map(json.loads, (directory / "events.jsonl").read_text().splitlines()):
        active[event["gpu"]] += 1 if event["event"] == "started" else -1
        assert sum(active.values()) <= 4 and max(active.values()) <= 2
    assert active == {"0": 0, "1": 0} and len(starts) == 9
    runner.drain(root, directory, gpus=["0", "1"])
    assert len(starts) == 9
    report = read_json(root / runner.REPORT)
    assert report["qualification_input"] is False
    assert all(r["evidence"] == runner.REPORT.as_posix() for r in report["records"])


def test_checkpoint_parent_failure_remains_a_blocker_but_failed_gate_does_not():
    state = dict(jobs={"parent": dict(family="f", task="parent", status="FAIL")})
    job = dict(family="f", dependencies=[dict(task="parent", kind="gate")])
    assert runner.dependency_blockers(job, state) == []
    job["dependencies"][0]["kind"] = "checkpoint"
    assert runner.dependency_blockers(job, state) == ["checkpoint parent parent: FAIL"]


def test_uninterrupted_hold_cannot_be_restarted_from_a_passing_scalar(tmp_path):
    path = tmp_path / "raw.json"
    atomic_json(path, {"evidence": {"continuity": {"mode": "uninterrupted"}}})
    parent = dict(family="f", task="ring_hold", status="PASS", origin="new_attempt", directory=str(tmp_path))
    state = dict(jobs={"parent": parent})
    job = dict(family="f", dependencies=[dict(task="ring_hold", kind="checkpoint")])
    assert runner.prerequisite_bindings(job, state) == {}
    assert "certified artifact binding unavailable" in runner.dependency_blockers(job, state)[0]


def test_research_roster_excludes_historical_seed_studies():
    from experiments.forge.research_attempts import variants
    groups = variants(Path(__file__).resolve().parents[1])
    assert len(groups) == 19
    assert all(len(g["required"]) == 29 for g in groups)
    assert len({g["scope"] for g in groups}) == 19
    assert sum(len(g["records"]) for g in groups) == 52


def test_research_backfill_fills_empty_cells_without_replacing_original_verdicts(tmp_path):
    from experiments.forge.research_tiers import _row
    atomic_json(tmp_path / runner.REPORT, {"records": [
        dict(research_scope="frozen-scope", task="missing", status="PASS", attempted=True),
        dict(research_scope="frozen-scope", task="original", status="PASS", attempted=True),
        dict(research_scope="other-scope", task="unbound", status="PASS", attempted=True)]})
    original = {"original": "FAIL"}
    text = _row(tmp_path, tmp_path / "board.md", "Research", "details.md", original,
                {"1": {"original", "missing", "unbound"}, "2": set(), "3": set()}, scope="frozen-scope")
    assert "1/3 (1 not run)" in text
    assert original == {"original": "FAIL"}


def test_missing_completion_publisher_stops_without_restart_or_replaying_work(tmp_path, monkeypatch):
    directory = tmp_path / "runs"
    atomic_json(directory / "state.json", dict(jobs={}))
    monkeypatch.setattr(runner, "device_slots", lambda *args, **kwargs: [])
    def unexpected_subprocess(*args, **kwargs):
        pytest.fail("a missing publisher must be detected before launching a subprocess")
    monkeypatch.setattr(runner.subprocess, "run", unexpected_subprocess)
    with pytest.raises(SystemExit) as stopped:
        runner.drain(tmp_path / "repo", directory, gpus=["0", "1"],
                     completion_hook=tmp_path / "missing-publisher.py")
    assert stopped.value.code == 2
    assert read_json(directory / "state.json")["jobs"] == {}
    template = Path(__file__).resolve().parents[1] / "ops/forge-attempt-coverage.service.example"
    assert "RestartPreventExitStatus=2" in template.read_text()
