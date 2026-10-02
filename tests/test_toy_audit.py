"""Audit hooks must preserve the experiment they claim to observe."""
from copy import deepcopy
import inspect
import json

import torch

from benchmarks.toy_audit.capture import Capture
from benchmarks.transfer_suite import image_tasks, vector_tasks


def scientific_result(result):
    """Wall time and timestamps are observational, never parity targets."""
    if isinstance(result, dict):
        return {k:scientific_result(v) for k,v in result.items()
                if k not in {"seconds","controller_seconds","created_at","stable_from_seconds","confirmed_seconds"}}
    if isinstance(result,list):
        return [scientific_result(v) for v in result]
    return result


def test_image_capture_preserves_training_and_evaluation(tmp_path):
    torch.set_num_threads(1)
    spec=deepcopy(image_tasks.TASKS[0])|dict(steps=24)
    original=image_tasks.run_episode(spec,vector_tasks.fixed_policy(),fixed=True)
    capture=Capture(tmp_path)
    with capture.installed():
        observed=image_tasks.run_episode(spec,vector_tasks.fixed_policy(),fixed=True)
    assert not original.get("error") and not observed.get("error")
    assert scientific_result(original)==scientific_result(observed)
    assert capture.records[0]["frames"]==24


def test_failed_contrast_exit_preserves_completed_evidence(tmp_path, monkeypatch):
    from benchmarks.toy_audit import replay
    source=tmp_path/"source"
    source.mkdir()
    (source/"reproduce_arms.py").write_text(
        "from benchmarks.transfer_suite import image_tasks, vector_tasks\n"
        "spec=dict(image_tasks.TASKS[0], steps=24)\n"
        "image_tasks.run_episode(spec,vector_tasks.fixed_policy(),fixed=True)\n"
        "raise SystemExit(1)\n")
    output=tmp_path/"artifact"
    monkeypatch.setattr("sys.argv",["replay","--source",str(source),"--output",str(output)])
    assert replay.main()==0
    receipt=json.loads((output/"replay.json").read_text())
    assert receipt["status"]=="COMPLETE" and receipt["proposal_exit_code"]==1
    assert receipt["episodes"]==1


def test_vector_capture_preserves_training_and_reserved_contract(tmp_path):
    torch.set_num_threads(1)
    spec=deepcopy(vector_tasks.RESERVED[0])|dict(steps=24)
    original=vector_tasks.run_episode(spec,vector_tasks.fixed_policy(),fixed=True,allow_reserved=True)
    capture=Capture(tmp_path)
    with capture.installed():
        assert "allow_reserved" in inspect.signature(vector_tasks.run_episode).parameters
        observed=vector_tasks.run_episode(spec,vector_tasks.fixed_policy(),fixed=True,allow_reserved=True)
    assert not original.get("error") and not observed.get("error")
    assert scientific_result(original)==scientific_result(observed)
