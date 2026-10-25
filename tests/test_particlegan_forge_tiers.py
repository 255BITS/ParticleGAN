"""Regression tests for particlegan_forge qualification, reports and namespace (review #6-#9, #18, #19)."""
from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from particlegan_forge import (
    NAMESPACE,
    CandidateSpec,
    ForgeRegistry,
    ForgeWorkspace,
    MetricBound,
    TaskSpec,
    TierSpec,
    TierView,
    grade_sustained_observations,
    resolve_device,
)
from particlegan_forge.__main__ import main as cli_main

REPO = Path(__file__).resolve().parents[1]


def _task(task_id="t1", **kw):
    base = dict(id=task_id, description="toy", adapter="toy", steps=4, observations=4, minimum_stable_checks=2,
                thresholds=[MetricBound("score", ">=", 0.5)])
    base.update(kw)
    return TaskSpec(**base)


def _ok(score=1.0, **overrides):
    def adapter(*, task, candidate, streams, attempt_dir, workspace_root, device, steps_override, log_event):
        out = {"observations": [{"step": s, "metrics": {"score": score}} for s in task.observation_steps()],
               "all_finite": True, "unintended_rng_deviations": 0, "completed_steps": task.steps}
        out.update(overrides)
        for key, value in list(out.items()):
            if value is _MISSING:
                del out[key]
        return out
    return adapter


_MISSING = object()


def _workspace(tmp_path, adapter, *, tiers=None, tasks=None):
    registry = ForgeRegistry()
    registry.adapter("toy")(adapter)
    ws = ForgeWorkspace(tmp_path, registry=registry)
    for task in tasks or [_task()]:
        ws.register_task(task)
    ws.register_view(TierView.from_tiers(id="v", goal="toy", tiers=tiers or [TierSpec(tier=1, required=["t1"])]))
    ws.register_candidate(CandidateSpec(id="c", technique="c", changed_factors=["baseline"]))
    return ws, registry


# --- #8 ---------------------------------------------------------------------------------------

def test_task_without_thresholds_is_rejected_at_construction_and_load():
    with pytest.raises(ValueError, match="threshold"):
        _task(thresholds=[])
    data = _task().to_dict()
    data["evaluation"]["thresholds"] = []
    with pytest.raises(ValueError, match="threshold"):
        TaskSpec.from_dict(data)


# --- #6 ---------------------------------------------------------------------------------------

def test_candidate_change_invalidates_stale_pass(tmp_path):
    ws, _ = _workspace(tmp_path, _ok())
    ws.run_task("c", "t1")
    assert ws.qualify("c", "v")["status"] == "PASS"
    ws.register_candidate(CandidateSpec(id="c", technique="c", changed_factors=["lr"], recipe_overrides={"lr": 1e-3}))
    q = ws.qualify("c", "v")
    outcome = q["tiers"][1]["tasks"]["t1"]
    assert outcome["status"] == "NOT_RUN" and "candidate_fingerprint" in outcome["reason"]
    assert q["status"] != "PASS" and not q["eligible"]


def test_adapter_exception_overwrites_stale_pass_with_invalid(tmp_path):
    ws, registry = _workspace(tmp_path, _ok())
    ws.run_task("c", "t1")
    assert ws.qualify("c", "v")["status"] == "PASS"

    def broken(**_):
        raise RuntimeError("boom")

    registry._adapters["toy"] = broken
    receipt = ws.run_task("c", "t1")
    assert receipt["verdict"] == "INVALID" and "boom" in receipt["grade"]["reason"]
    # Even ignoring the adapter-source change, the recorded failure replaces the PASS.
    assert ForgeWorkspace(tmp_path).qualify("c", "v")["tiers"][1]["tasks"]["t1"]["status"] == "INVALID"
    assert ws.qualify("c", "v")["status"] == "INVALID"
    events = (ws.queue_root / "c" / "t1" / "run.log").read_text()
    assert "attempt_failed" in events and "RuntimeError: boom" in events


@pytest.mark.parametrize("field,value", [("rng_version", "forge-rng-v0"), ("engine_source_sha256", "0" * 64),
                                         ("adapter_source_sha256", "0" * 64), ("seed", 7)])
def test_receipt_binding_mismatch_is_not_counted(tmp_path, field, value):
    ws, _ = _workspace(tmp_path, _ok())
    ws.run_task("c", "t1")
    path = ws._receipt_path("c", "t1", "cpu")
    rec = json.loads(path.read_text())
    rec["binding"][field] = value
    path.write_text(json.dumps(rec))
    assert ws.qualify("c", "v")["tiers"][1]["tasks"]["t1"]["status"] == "NOT_RUN"


def test_legacy_receipt_without_binding_is_stale(tmp_path):
    ws, _ = _workspace(tmp_path, _ok())
    ws.run_task("c", "t1")
    path = ws._receipt_path("c", "t1", "cpu")
    rec = json.loads(path.read_text())
    del rec["binding"], rec["cohort_key"]
    path.write_text(json.dumps(rec))
    assert ws.qualify("c", "v")["status"] == "INCOMPLETE"


def test_unregistered_adapter_is_reported_unverified_and_not_eligible(tmp_path):
    ws, _ = _workspace(tmp_path, _ok())
    ws.run_task("c", "t1")
    q = ForgeWorkspace(tmp_path).qualify("c", "v")
    assert q["status"] == "PASS" and q["adapter_source_verified"] is False and q["eligible"] is False
    assert ws.qualify("c", "v")["eligible"] is True


# --- #7 / #9 ----------------------------------------------------------------------------------

def test_compile_is_receipt_derived_and_namespaced(tmp_path):
    forge_reports = tmp_path / "reports/forge"
    forge_reports.mkdir(parents=True)
    sentinels = {name: f"live {name}\n" for name in ("EXPERIMENT_MEMORY.md", "technique-inventory.md",
                                                       "EXPERIMENTS_BY_TIER.md")}
    for name, text in sentinels.items():
        (forge_reports / name).write_text(text)
    (tmp_path / "configs/forge/tasks").mkdir(parents=True)
    (tmp_path / "configs/forge/tasks/other.json").write_text(json.dumps({"id": "other", "evaluation": {}}))

    ws, _ = _workspace(tmp_path, _ok(score=0.1))
    ws.run_task("c", "t1")
    out = ws.compile_reports(view_id="v")
    for name, text in sentinels.items():
        assert (forge_reports / name).read_text() == text
    assert all(Path(p).parent == tmp_path / "reports" / NAMESPACE for p in out.values())
    memory = Path(out["experiment_memory"]).read_text()
    assert "fast-weight-bcap" not in memory and "Recommendations" not in memory
    assert "`c`" in memory and "**FAIL**" in memory and "score >= 0.5" in memory
    ebt = Path(out["experiments_by_tier"]).read_text()
    assert "480" not in ebt and "batch_size=1" not in ebt
    assert "FAIL" in Path(out["leaderboard"]).read_text()


def test_experiments_forge_schema_files_are_refused(tmp_path):
    ws = ForgeWorkspace(tmp_path)
    ws.tasks_dir.mkdir(parents=True)
    (ws.tasks_dir / "x.json").write_text(json.dumps({"id": "x", "adapter": "a", "evaluation": {}}))
    with pytest.raises(ValueError, match="experiments/forge"):
        ws.load_tasks()
    data = TierView.from_tiers(id="v", goal="g", tiers=[TierSpec(tier=1, required=["x"])]).to_dict()
    del data["namespace"]
    with pytest.raises(ValueError, match="namespace"):
        TierView.from_dict(data)


def test_cli_commands_on_repository_do_not_touch_experiments_forge(tmp_path, capsys):
    memory = REPO / "reports/forge/EXPERIMENT_MEMORY.md"
    before = memory.read_bytes() if memory.exists() else None
    assert cli_main(["--root", str(REPO), "--queue-root", str(tmp_path), "validate"]) == 0
    assert json.loads(capsys.readouterr().out)["namespace"] == NAMESPACE
    assert cli_main(["--root", str(REPO), "--queue-root", str(tmp_path), "board", "--view", "absent"]) == 2
    assert "not found" in capsys.readouterr().err
    assert (memory.read_bytes() if memory.exists() else None) == before


def test_cli_run_board_compile_round_trip(tmp_path, capsys, monkeypatch):
    import types
    import sys

    module = types.ModuleType("pgf_test_provider")
    registry = ForgeRegistry()
    registry.adapter("toy")(_ok())
    module.REGISTRY = registry
    monkeypatch.setitem(sys.modules, "pgf_test_provider", module)
    _workspace(tmp_path, _ok())
    args = ["--root", str(tmp_path), "--provider", "pgf_test_provider"]
    assert cli_main([*args, "run", "c", "--view", "v"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert cli_main([*args, "board", "--view", "v"]) == 0
    assert json.loads(capsys.readouterr().out)["c"]["eligible"] is True
    assert cli_main([*args, "compile", "--view", "v"]) == 0
    assert (tmp_path / "reports" / NAMESPACE / "leaderboard.md").is_file()


# --- #18 --------------------------------------------------------------------------------------

def test_diagnostic_only_tier_is_held_and_does_not_block(tmp_path):
    tasks = [_task("t1"), _task("d2"), _task("t3")]
    tiers = [TierSpec(tier=1, required=["t1"]), TierSpec(tier=2, diagnostic=["d2"]),
             TierSpec(tier=3, required=["t3"])]
    ws, _ = _workspace(tmp_path, _ok(), tiers=tiers, tasks=tasks)
    ws.run_task("c", "t1")
    ws.run_task("c", "t3")
    q = ws.qualify("c", "v")
    assert [q["tiers"][t]["status"] for t in (1, 2, 3)] == ["PASS", "HELD", "PASS"]
    assert q["status"] == "PASS" and q["qualified_tier"] == 3 and q["held_tiers"] == [2]


def test_held_highest_tier_and_diagnostic_only_view_are_held(tmp_path):
    tasks = [_task("t1"), _task("r2")]
    ws, _ = _workspace(tmp_path, _ok(), tasks=tasks,
                       tiers=[TierSpec(tier=1, required=["t1"]), TierSpec(tier=2, ranking=["r2"])])
    ws.run_task("c", "t1")
    q = ws.qualify("c", "v")
    assert q["tiers"][2]["status"] == "HELD" and q["status"] == "HELD" and not q["eligible"]

    ws.register_view(TierView.from_tiers(id="d", goal="toy", tiers=[TierSpec(tier=1, diagnostic=["t1"])]))
    assert ws.qualify("c", "d")["status"] == "HELD"


def test_invalid_required_task_reports_invalid_tier(tmp_path):
    ws, _ = _workspace(tmp_path, _ok(all_finite=_MISSING))
    ws.run_task("c", "t1")
    assert ws.qualify("c", "v")["tiers"][1]["status"] == "INVALID"


# --- #19 --------------------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["all_finite", "unintended_rng_deviations", "completed_steps"])
def test_missing_adapter_evidence_is_invalid(tmp_path, field):
    ws, _ = _workspace(tmp_path, _ok(**{field: _MISSING}))
    receipt = ws.run_task("c", "t1")
    assert receipt["verdict"] == "INVALID" and field in receipt["grade"]["reason"]


def test_grade_function_never_defaults_evidence_to_pass():
    task = _task()
    rows = [{"step": s, "metrics": {"score": 1.0}} for s in task.observation_steps()]
    assert grade_sustained_observations(task, rows)["status"] == "INVALID"
    assert grade_sustained_observations(task, rows, all_finite=True, unintended_rng_deviations=0,
                                        completed_steps=4)["status"] == "PASS"


def test_timeout_is_enforced(tmp_path):
    def slow(**kwargs):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                time.sleep(0.05)
            except Exception:  # an adapter swallowing Exception cannot hide the deadline
                pass
        return _ok()(**kwargs)

    ws, _ = _workspace(tmp_path, slow, tasks=[_task(timeout_seconds=1)])
    t0 = time.monotonic()
    receipt = ws.run_task("c", "t1")
    assert time.monotonic() - t0 < 5
    assert receipt["verdict"] == "INCOMPLETE" and "timeout" in receipt["grade"]["reason"]


def _evaluating_adapter(stream_family):
    def adapter(*, task, candidate, streams, attempt_dir, workspace_root, device, steps_override, log_event, evaluate):
        train = streams.generator("noise")
        for step in task.observation_steps():
            import torch

            torch.rand(1, generator=train)

            def measure():
                torch.rand(1, generator=streams.generator(stream_family))
                torch.rand(1)  # global draw: isolated and restored
                return {"score": 1.0}

            evaluate(step, measure)
        return {"all_finite": True, "unintended_rng_deviations": 0, "completed_steps": task.steps}
    return adapter


def test_evaluate_helper_uses_isolated_evaluation(tmp_path):
    ws, _ = _workspace(tmp_path, _evaluating_adapter("eval"))
    receipt = ws.run_task("c", "t1")
    assert receipt["verdict"] == "PASS" and len(receipt["observations"]) == 4


def test_evaluate_helper_detects_training_stream_mutation(tmp_path):
    ws, _ = _workspace(tmp_path, _evaluating_adapter("noise"))
    receipt = ws.run_task("c", "t1")
    assert receipt["verdict"] == "INVALID" and receipt["unintended_rng_deviations"] > 0


# --- device -----------------------------------------------------------------------------------

def test_device_resolution_ignores_receipt_counts(tmp_path):
    ws, _ = _workspace(tmp_path, _ok())
    ws.run_task("c", "t1", device="cpu")
    fake = ws.reports_dir / "receipts/cuda/c"
    fake.mkdir(parents=True)
    for i in range(3):
        (fake / f"x{i}.json").write_text("{}")
    assert resolve_device(None) == "cpu"
    assert ws.qualify("c", "v")["device"] == "cpu"
    assert resolve_device("cuda:1") == "cuda"
    with pytest.raises(ValueError):
        resolve_device("cpu:0")


def test_device_is_part_of_receipt_cohort(tmp_path):
    ws, _ = _workspace(tmp_path, _ok())
    ws.run_task("c", "t1", device="cpu")
    src = ws._receipt_path("c", "t1", "cpu")
    dst = ws._receipt_path("c", "t1", "cuda")
    dst.parent.mkdir(parents=True)
    dst.write_bytes(src.read_bytes())
    outcome = ws.qualify("c", "v", device="cuda")["tiers"][1]["tasks"]["t1"]
    assert outcome["status"] == "NOT_RUN" and "device" in outcome["reason"]


# --- PR #480 review follow-ups ----------------------------------------------------------------

def test_deadline_is_disarmed_before_deterministic_scope_restores_flags(tmp_path, monkeypatch):
    import contextlib
    import signal

    import particlegan_forge.tiers as tiers_mod

    real_scope = tiers_mod.deterministic_scope
    timer_at_restore = []

    @contextlib.contextmanager
    def probing_scope(seed=0):
        with real_scope(seed=seed):
            try:
                yield
            finally:
                timer_at_restore.append(signal.getitimer(signal.ITIMER_REAL)[0])

    monkeypatch.setattr(tiers_mod, "deterministic_scope", probing_scope)
    ws, _ = _workspace(tmp_path, _ok(), tasks=[_task(timeout_seconds=60)])
    assert ws.run_task("c", "t1")["verdict"] == "PASS"
    assert timer_at_restore == [0.0]


@pytest.mark.parametrize("field,value", [("completed_steps", "np"), ("unintended_rng_deviations", "np"),
                                         ("completed_steps", 4.0)])
def test_integral_evidence_scalars_are_accepted(tmp_path, field, value):
    import numpy as np

    if value == "np":
        value = np.int64(4 if field == "completed_steps" else 0)
    ws, _ = _workspace(tmp_path, _ok(**{field: value}))
    assert ws.run_task("c", "t1")["verdict"] == "PASS"
    task = _task()
    rows = [{"step": s, "metrics": {"score": 1.0}} for s in task.observation_steps()]
    for bad in (True, 4.5, -1, "4"):
        assert grade_sustained_observations(task, rows, all_finite=True, unintended_rng_deviations=0,
                                            completed_steps=bad)["status"] == "INVALID"


def test_var_keyword_adapter_does_not_receive_evaluate(tmp_path):
    inner = _ok()

    def forwarding(**kwargs):
        return inner(**kwargs)  # fixed-signature inner adapter would reject evaluate=

    ws, _ = _workspace(tmp_path, forwarding)
    assert ws.run_task("c", "t1")["verdict"] == "PASS"


def test_fractional_timeout_round_trips_through_from_dict():
    for timeout in (0.5, 1.5, 7):
        loaded = TaskSpec.from_dict(_task(timeout_seconds=timeout).to_dict())
        assert loaded.timeout_seconds == timeout and type(loaded.timeout_seconds) is type(timeout)


def test_tier_statuses_match_experiments_forge_views(tmp_path):
    ws, _ = _workspace(tmp_path, _ok(score=0.0), tasks=[_task("t1"), _task("t2")],
                       tiers=[TierSpec(tier=1, required=["t1"]), TierSpec(tier=2, required=["t2"])])
    ws.run_task("c", "t1")
    q = ws.qualify("c", "v")
    assert q["tiers"][1]["status"] == "FAIL" and not q["tiers"][1]["gated_by_lower_tier"]
    # The tier above a failing tier keeps its own status (NOT_RUN), not BLOCKED.
    assert q["tiers"][2]["status"] == "NOT_RUN" and q["tiers"][2]["gated_by_lower_tier"]
    assert q["status"] == "FAIL" and q["qualified_tier"] == 0

    ws2, _ = _workspace(tmp_path / "w2", _ok())
    q2 = ws2.qualify("c", "v")
    assert q2["tiers"][1]["status"] == "NOT_RUN" and q2["status"] == "INCOMPLETE"
