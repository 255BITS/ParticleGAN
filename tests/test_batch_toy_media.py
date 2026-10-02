"""Self-contained software evidence controls; no scientific training/history."""
from copy import deepcopy
import hashlib
import json
import subprocess

from PIL import Image
import pytest

from benchmarks.toy_audit import batch_toy_media as media


def save(path, value):
    media.write(path, value)


def archive(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    (root / "source.py").write_text("# Synthetic source identity; never a scientific execution.\n")
    protocol = {"external_steps_per_arm": 500, "external_total_seconds": 60, "threads": 1,
                "arms": {"G16": {"D_batch": 16, "G_batch": 16}, "G64": {"D_batch": 16, "G_batch": 64}},
                "source_hashes": {media.DRIVER: media.sha(root / "source.py")},
                "primary_metric": {"threshold_both_arm_convergence_ratio": .9, "threshold_G64_over_G16": .9}}
    save(root / "protocol.json", protocol)
    arms, published = {}, {}
    for name, end in (("G16", .2), ("G64", .15)):
        def observation(step):
            return {"live_excess_error": 1 - (1 - end) * step / 500,
                    "served_excess_error": 1 - (1 - end) * step / 500,
                    "irreducible_population_error": .0225, "served_source": "fast"}
        rows = []
        for step in range(1, 501):
            row = {"step": step, "g_batch": int(name[1:]), "d_batch": 16, "dense_rows": 128,
                   "loss_g": .7, "loss_d": .7, "penalty": .1, "sigma": 1.3,
                   "row_event": {"state": None}, "settle": {"scale": 1.},
                   "quadratic_weights": [.1, -.1], "caller_panel_sha256": "f" * 64}
            if step in media.MEDIA_STEPS: row["evaluation"] = observation(step)
            rows.append(row)
        (root / (name + ".jsonl")).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        arms[name] = {"steps": 500, "initial": observation(0), "endpoint": observation(500),
            "health": {"finite_steps": 500, "min_dense_rows": 128, "ka2_applied_calls": 500,
                       **{k: True for k in ("ownership", "frozen_host", "critic_score_active_gradients", "generator_live_gradients")}},
            "control": {"rows": {"counters": {"updates": 500}}, "counters": {"evals": 31, "probes": 248}},
            "caller_stream_hashes": {key: "f" * 64 for key in media.STREAMS},
            "caller_panel_history_sha256": hashlib.sha256("\n".join(["f" * 64] * 500).encode()).hexdigest()}
        published[name] = {"initial": observation(0), "endpoint": observation(500),
            "evaluation_points": [{"step": r["step"], **r["evaluation"]} for r in rows if "evaluation" in r]}
    checks = media.original_gates(arms, streams_match=True, source_match=True)
    result = {"arms": arms, "checks": checks, "status": "passed", "failure": None,
              "elapsed_seconds": 27., "protocol_sha256": media.sha(root / "protocol.json"),
              "fixture_sha256": "a" * 64, "teacher_sha256": "b" * 64,
              "initial_hashes": {role: "c" * 64 for role in ("generator", "encoder", "critic", "router", "table")}}
    save(root / "result.json", result)
    card = {"protocol_sha256": media.sha(root / "protocol.json"), "publication": {
        "protocol_sha256": media.sha(root / "protocol.json"), "original_scientific_driver_sha256": media.sha(root / "source.py")},
        "artifact_hashes": {name: media.sha(root / name) for name in ("result.json", "G16.jsonl", "G64.jsonl")},
        "arms": published, "real100": {"quality_gate": {"status": "FAILED", "scope": "Separate synthetic software control"}},
        **{key: deepcopy(result[key]) for key in ("fixture_sha256", "teacher_sha256", "initial_hashes")}}
    return root, card


def test_retained_actual_step_media_keeps_original_bytes_and_separate_task_fail(tmp_path, monkeypatch):
    root, card = archive(tmp_path)
    monkeypatch.setattr(media, "load_card", lambda: card)
    monkeypatch.setattr(media, "reproduce", lambda *args: pytest.fail("ordinary export must not train"))
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    output = tmp_path / "review"
    receipt = media.export(root, output)
    assert receipt["data"]["scientific_status"] == "PASS"
    assert receipt["data"]["actual_task"]["status"] == "FAILED"
    assert receipt["training_or_model_rescoring_during_export"] is False
    assert receipt["frozen_176_campaign_changed"] is False
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    with Image.open(output / "goal.gif") as gif: assert gif.n_frames == 12


@pytest.mark.parametrize("change", ("partial", "step", "observation", "endpoint", "nonfinite", "weights", "panel", "health", "streams", "empty_streams", "missing_stream", "extra_stream", "invalid_stream", "fixture", "teacher", "init", "missing_identity", "fake_pass", "missing_protocol", "wall"))
def test_fresh_protocol_rejects_incomplete_or_false_evidence_before_media(tmp_path, change):
    root, card = archive(tmp_path)
    assert media.verify(root, card, media.Inputs(), fresh=True)["scientific_status"] == "PASS"
    result = json.loads((root / "result.json").read_text())
    trace = root / "G64.jsonl"; rows = [json.loads(line) for line in trace.read_text().splitlines()]
    if change == "partial": result["arms"]["G64"]["steps"] = 499
    elif change == "step": rows.pop(12)
    elif change == "observation": rows[49].pop("evaluation")
    elif change == "endpoint": result["arms"]["G64"]["endpoint"]["live_excess_error"] = .001
    elif change == "nonfinite": rows[1]["settle"]["scale"] = float("nan")
    elif change == "weights": rows[1]["quadratic_weights"][0] = None
    elif change == "panel": rows[1]["caller_panel_sha256"] = "e" * 64
    elif change == "health": result["arms"]["G64"]["health"]["ownership"] = False
    elif change == "streams": result["arms"]["G64"]["caller_panel_history_sha256"] = "different"
    elif change == "empty_streams": result["arms"]["G64"]["caller_stream_hashes"] = {}
    elif change == "missing_stream": result["arms"]["G64"]["caller_stream_hashes"].pop("d_data")
    elif change == "extra_stream": result["arms"]["G64"]["caller_stream_hashes"]["extra"] = "f" * 64
    elif change == "invalid_stream": result["arms"]["G64"]["caller_stream_hashes"]["d_data"] = "invalid"
    elif change == "fixture": result["fixture_sha256"] = "different"
    elif change == "teacher": result["teacher_sha256"] = "different"
    elif change == "init": result["initial_hashes"]["table"] = "different"
    elif change == "missing_identity": result.pop("fixture_sha256")
    elif change == "fake_pass": result["checks"]["G64_improves_10_percent"] = False
    elif change == "missing_protocol": (root / "protocol.json").unlink()
    else: result["elapsed_seconds"] = 61.
    save(root / "result.json", result)
    trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises((ValueError, FileNotFoundError)): media.verify(root, card, media.Inputs(), fresh=True)


def test_historical_hash_guard_rejects_changed_trace(tmp_path):
    root, card = archive(tmp_path)
    path = root / "G16.jsonl"; path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="identity"): media.verify(root, card, media.Inputs())


def test_reproduction_invokes_exact_public_caller_once_with_cpu_and60s_cap(tmp_path, monkeypatch):
    source = {"fixed": "synthetic-source"}; calls = []
    monkeypatch.setattr(media, "active_sources", lambda *args: deepcopy(source))
    def run(command, **kwargs):
        calls.append((command, kwargs)); return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(media.subprocess, "run", run)
    root = tmp_path / "new"
    assert media.reproduce(root, {}) == root / "attempt"
    assert len(calls) == 1 and calls[0][0][1] == media.DRIVER
    assert calls[0][1]["timeout"] == 60 and calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(ValueError): media.reproduce(root, {})


def test_exit_zero_with_partial_result_cannot_publish_default_pass(tmp_path, monkeypatch, capsys):
    root, card = archive(tmp_path)
    result = json.loads((root / "result.json").read_text()); result["arms"]["G64"]["steps"] = 499
    save(root / "result.json", result)
    source = {"scope": "synthetic source only"}
    save(root.parent / "reproduction.json", {"source": source, "cap_seconds": 60, "returncode": 0})
    monkeypatch.setattr(media, "load_card", lambda: card)
    monkeypatch.setattr(media, "active_sources", lambda *args: deepcopy(source))
    monkeypatch.setattr(media, "reproduce", lambda *args: root)
    output = tmp_path / "review"
    assert media.main(["--input", str(root), "--output", str(output), "--reproduce"]) == 2
    assert "partial" in capsys.readouterr().err and not output.exists()


def test_fixed_gate_rejects_no_relative_gain_despite_absolute_convergence(tmp_path):
    root, card = archive(tmp_path)
    result = json.loads((root / "result.json").read_text())
    arms = result["arms"]; arms["G64"]["endpoint"] = deepcopy(arms["G16"]["endpoint"])
    checks = media.original_gates(arms, streams_match=True, source_match=True)
    assert checks["G16_converges_10_percent"] and checks["G64_converges_10_percent"]
    assert checks["G64_improves_10_percent"] is False


def test_exit_one_with_complete_passing_rows_cannot_become_exit_zero(tmp_path, monkeypatch, capsys):
    root, card = archive(tmp_path)
    source = {"scope": "synthetic source only"}
    save(root.parent / "reproduction.json", {"source": source, "cap_seconds": 60, "returncode": 1})
    monkeypatch.setattr(media, "load_card", lambda: card)
    monkeypatch.setattr(media, "active_sources", lambda *args: deepcopy(source))
    monkeypatch.setattr(media, "reproduce", lambda *args: root)
    output = tmp_path / "review"
    assert media.main(["--input", str(root), "--output", str(output), "--reproduce"]) == 2
    assert "exit code contradicts" in capsys.readouterr().err and not output.exists()


def test_complete_numeric_fail_preserves_failure_media_and_exit_one(tmp_path, monkeypatch):
    root, card = archive(tmp_path)
    result = json.loads((root / "result.json").read_text())
    endpoint = result["arms"]["G64"]["endpoint"]
    endpoint["live_excess_error"] = endpoint["served_excess_error"] = .19
    trace = root / "G64.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[-1]["evaluation"] = deepcopy(endpoint)
    trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result["checks"] = media.original_gates(result["arms"], streams_match=True, source_match=True)
    result["status"] = "failed"
    save(root / "result.json", result)
    source = {"scope": "synthetic source only"}
    save(root.parent / "reproduction.json", {"source": source, "cap_seconds": 60, "returncode": 1})
    monkeypatch.setattr(media, "load_card", lambda: card)
    monkeypatch.setattr(media, "active_sources", lambda *args: deepcopy(source))
    monkeypatch.setattr(media, "reproduce", lambda *args: root)
    output = tmp_path / "review"
    assert media.main(["--input", str(root), "--output", str(output), "--reproduce"]) == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["data"]["scientific_status"] == "FAIL"
    assert receipt["data"]["checks"]["G64_improves_10_percent"] is False
    with Image.open(output / "goal.gif") as gif: assert gif.n_frames == 12


@pytest.mark.parametrize("timeout", [False, True])
def test_child_error_or_timeout_cannot_publish_media(tmp_path, monkeypatch, capsys, timeout):
    monkeypatch.setattr(media, "load_card", lambda: {})
    monkeypatch.setattr(media, "active_sources", lambda *args: {"scope": "synthetic source only"})
    def run(command, **kwargs):
        if timeout: raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 2)
    monkeypatch.setattr(media.subprocess, "run", run)
    output = tmp_path / "review"
    assert media.main(["--input", str(tmp_path / "new"), "--output", str(output), "--reproduce"]) == 2
    assert "INCOMPLETE:" in capsys.readouterr().err and not output.exists()
