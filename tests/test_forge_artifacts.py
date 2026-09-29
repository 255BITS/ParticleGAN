"""Certified saved inputs retain identity across relocation, never mutation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from experiments.forge.artifacts import manifest_artifacts, verify_artifacts
from experiments.forge.views import grade_result, load_tasks


ROOT = Path(__file__).resolve().parents[1]


def artifact_tree(path):
    problem = path / "grid100"
    problem.mkdir(parents=True)
    (problem / "config.json").write_text(json.dumps({"steps": 7000, "eval_interval": 250,
        "eval_samples": 20000, "early_eval_steps": [0, 1, 10, 25, 50, 100]}))
    (problem / "summary.json").write_text("{}")
    (problem / "events.jsonl").write_text('{"event":"eval"}\n')
    # Bulk-byte fixtures test certification; the original numerical evaluator
    # is exercised by native adapter tests, not replaced with fake oracle data.
    (problem / "holdout_samples.npz").write_bytes(bytes(range(256)) * 400)
    return path


def test_manifest_is_complete_relative_and_portable(tmp_path):
    original = artifact_tree(tmp_path / "original")
    manifest = manifest_artifacts(original)
    assert set(manifest["files"]) == {"grid100/" + name for name in
        ("config.json", "summary.json", "events.jsonl", "holdout_samples.npz")}
    assert manifest["file_count"] == 4
    assert manifest["total_bytes"] == sum(p.stat().st_size for p in original.rglob("*") if p.is_file())
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    assert manifest_artifacts(relocated) == manifest
    verify_artifacts(relocated, manifest)


@pytest.mark.parametrize("change", ["same_size_bytes", "missing", "added", "events", "summary"])
def test_any_evaluator_input_mutation_is_rejected(tmp_path, change):
    root = artifact_tree(tmp_path / "native")
    manifest = manifest_artifacts(root)
    array = root / "grid100/holdout_samples.npz"
    if change == "same_size_bytes":
        data = bytearray(array.read_bytes())
        data[-1] ^= 1
        array.write_bytes(data)
    elif change == "missing":
        array.unlink()
    elif change == "added":
        (root / "grid100/extra.json").write_text("{}")
    else:
        (root / ("grid100/events.jsonl" if change == "events" else "grid100/summary.json")).write_text("changed")
    with pytest.raises(ValueError, match="artifact .*changed"):
        verify_artifacts(root, manifest)


def test_manifest_cannot_escape_or_misdescribe_its_tree(tmp_path):
    root = artifact_tree(tmp_path / "native")
    original = manifest_artifacts(root)
    unsafe = deepcopy(original)
    unsafe["files"]["../outside"] = next(iter(unsafe["files"].values()))
    with pytest.raises(ValueError, match="relative paths"):
        verify_artifacts(root, unsafe)
    changed = deepcopy(original)
    changed["files"]["grid100/config.json"]["size"] += 1
    with pytest.raises(ValueError, match="digest or totals"):
        verify_artifacts(root, changed)
    (root / "linked.json").symlink_to(root / "grid100/config.json")
    with pytest.raises(ValueError, match="symlink"):
        manifest_artifacts(root)


def test_native_grader_checks_certified_bytes_before_calling_original_gate(tmp_path, monkeypatch):
    from benchmarks.toy100 import accuracy_gate
    root = artifact_tree(tmp_path / "native")
    evidence = {"artifact_root": str(root), "artifact_manifest": manifest_artifacts(root)}
    calls = []
    def original_gate(path, **kwargs):
        calls.append((path, kwargs))
        return {"problems": {"grid100": {"status": "PASS", "reason": "numerical gate fixture"}}}
    monkeypatch.setattr(accuracy_gate, "evaluate_suite", original_gate)
    task = load_tasks(ROOT)["grid100"]
    assert grade_result(task, {"evidence": evidence})["status"] == "PASS"
    assert len(calls) == 1
    array = root / "grid100/holdout_samples.npz"
    array.write_bytes(b"x" * array.stat().st_size)
    assert grade_result(task, {"evidence": evidence})["status"] == "INVALID"
    assert len(calls) == 1
    array.unlink()
    assert grade_result(task, {"evidence": evidence})["status"] == "INVALID"
    assert len(calls) == 1


def test_native_grader_needs_manifest_and_exact_declared_early_schedule(tmp_path):
    root = artifact_tree(tmp_path / "native")
    task = load_tasks(ROOT)["grid100"]
    evidence = {"artifact_root": str(root)}
    assert grade_result(task, {"evidence": evidence})["status"] == "INCOMPLETE"
    config = root / "grid100/config.json"
    values = json.loads(config.read_text())
    values["early_eval_steps"].remove(25)
    config.write_text(json.dumps(values))
    evidence["artifact_manifest"] = manifest_artifacts(root)
    grade = grade_result(task, {"evidence": evidence})
    assert grade["status"] == "INVALID" and "early observation" in str(grade["reasons"])


def test_native_artifact_changed_during_evaluation_is_rejected(tmp_path, monkeypatch):
    from benchmarks.toy100 import accuracy_gate
    root = artifact_tree(tmp_path / "native")
    evidence = {"artifact_root": str(root), "artifact_manifest": manifest_artifacts(root)}
    def changing_gate(path, **kwargs):
        (path / "grid100/events.jsonl").write_text("changed during evaluation")
        return {"problems": {"grid100": {"status": "PASS", "reason": "must not qualify"}}}
    monkeypatch.setattr(accuracy_gate, "evaluate_suite", changing_gate)
    assert grade_result(load_tasks(ROOT)["grid100"], {"evidence": evidence})["status"] == "INVALID"
