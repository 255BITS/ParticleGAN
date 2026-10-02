"""Synthetic archive controls: publication launches no training or rescoring."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from benchmarks.toy_audit import api_contract, api_publish, api_run


def archive(path, *, updates=600, default_updates=600, samples=32, frames=9,
            terminal=5, metric_count=24, failures=(), list_thresholds=False):
    path.mkdir()
    case = {"id": "publication-software-control", "title": "Synthetic publication control",
            "goal": "Verify immutable software archive metadata; no scientific training",
            "scope": "Synthetic software-only receipt, never a trained experiment",
            "kind": "software", "legacy_ids": ["publication-software-control"],
            "default_steps": default_updates, "batch_size": 4, "eval_samples": 32,
            "thresholds": [["finite", ">=", 1]] if list_thresholds else {"observations": metric_count},
            "sampling": "synthetic software arrays", "evaluation_observations": metric_count,
            "terminal_observations": terminal}
    metric_steps = api_contract.evaluation_steps(updates, min(updates, metric_count) + 1)
    media_steps = api_contract.evaluation_steps(updates, frames)
    union = sorted(set(metric_steps) | set(media_steps))
    images = [Image.new("RGB", (8, 8), (index * 31 % 255, index * 17 % 255, index))
              for index in range(len(media_steps))]
    images[0].save(path / "goal.gif", save_all=True, append_images=images[1:], duration=100, optimize=False)
    arrays = {f"step{step}_view0_{role}": np.full((4, 2), step + (role == "samples"), dtype=np.float32)
              for step in union for role in ("target", "samples")}
    np.savez_compressed(path / "observations.npz", **arrays)
    (path / "final-state.pt").write_bytes(b"synthetic software identity control; no actual checkpoint")
    observations = [{"step": step, "passed": step not in failures,
                     "failed_bounds": [] if step not in failures else ["finite >= 1"],
                     "metrics": {"finite": int(step not in failures)},
                     "views": [{"kind": "scatter", "title": "Synthetic software view"}]}
                    for step in union]
    full = updates >= default_updates and samples >= 32 and len(metric_steps) - 1 >= metric_count
    last = [record for record in observations if record["step"] > 0 and record["step"] in metric_steps][-terminal:]
    sustained = len(last) == terminal and all(record["passed"] for record in last)
    failed = list(observations[-1]["failed_bounds"])
    if not full:
        failed.append("default budget, evaluation draw count or metric cadence not completed")
    if not sustained:
        failed.append(f"last {terminal} post-update metric observations do not all pass")
    receipt = {"schema": "particlegan_api_toy_run_v1", "case": case,
               "status": "COMPLETE", "source_unchanged": True, "completed_updates": updates,
               "protocol": {"updates": updates, "default_updates": default_updates,
                            "evaluation_samples": samples, "default_evaluation_samples": 32,
                            "terminal_observations": terminal, "metric_observations": metric_count,
                            "media_frames": frames, "metric_evaluation_steps": metric_steps,
                            "media_steps": media_steps, "evaluation_steps": union},
               "metric_passed": observations[-1]["passed"], "sustained_metric_passed": sustained,
               "default_protocol_complete": full, "passed": full and sustained,
               "verdict": "PASS" if full and sustained else "FAIL", "failed_bounds": failed,
               "gif_frames": len(media_steps), "observations": observations,
               "source": {"commit": "synthetic-software-only", "files_sha256": {}},
               "runtime": {"device": "cpu"}, "recipe": {"name": "synthetic-software-only"},
               "api_components": ["synthetic-software-only"], "seed": 0,
               "artifacts": {name: {"sha256": api_run.file_hash(path / name), "bytes": (path / name).stat().st_size}
                             for name in ("goal.gif", "observations.npz", "final-state.pt")}}
    save(path, receipt)
    return receipt


def save(path, receipt):
    api_run.write_json(path / "receipt.json", receipt)


def test_valid_union_keeps_all_metric_arrays_but_only_requested_media_frames(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path, frames=3, list_thresholds=True)
    assert len(receipt["observations"]) == 25
    assert receipt["gif_frames"] == 3
    assert api_publish.verify_run(path) == receipt


def test_valid_incomplete_run_stays_fail_despite_passing_metrics(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path, updates=16)
    assert api_publish.verify_run(path)["verdict"] == "FAIL"
    assert receipt["metric_passed"] and receipt["sustained_metric_passed"]


def test_forged_16_of_600_completion_and_pass_is_rejected(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path, updates=16)
    receipt.update(default_protocol_complete=True, passed=True, verdict="PASS", failed_bounds=[])
    save(path, receipt)
    with pytest.raises(ValueError, match="default_protocol_complete"):
        api_publish.verify_run(path)


def test_failed_terminal_metric_cannot_claim_sustained_pass(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path, failures=(600,))
    receipt.update(sustained_metric_passed=True, passed=True, verdict="PASS", failed_bounds=[])
    save(path, receipt)
    with pytest.raises(ValueError, match="sustained_metric_passed"):
        api_publish.verify_run(path)


@pytest.mark.parametrize("field", ["metric_passed", "sustained_metric_passed", "default_protocol_complete", "passed", "verdict", "failed_bounds"])
def test_each_stored_grade_field_must_match_rederivation(tmp_path, field):
    path = tmp_path / "control"
    receipt = archive(path)
    receipt[field] = "FAIL" if field == "verdict" else ["invented failure"] if field == "failed_bounds" else False
    save(path, receipt)
    with pytest.raises(ValueError):
        api_publish.verify_run(path)


@pytest.mark.parametrize("change", ["missing", "duplicate", "out_of_order"])
def test_exact_union_coverage_rejects_missing_duplicate_or_reordered_observations(tmp_path, change):
    path = tmp_path / "control"
    receipt = archive(path)
    if change == "missing":
        del receipt["observations"][2]
    elif change == "duplicate":
        receipt["observations"].insert(2, deepcopy(receipt["observations"][1]))
    else:
        receipt["observations"][1], receipt["observations"][2] = receipt["observations"][2], receipt["observations"][1]
    save(path, receipt)
    with pytest.raises(ValueError, match="step coverage"):
        api_publish.verify_run(path)


@pytest.mark.parametrize("field,value", [("default_updates", 16), ("default_evaluation_samples", 8),
                                         ("terminal_observations", 1), ("metric_observations", 8)])
def test_declared_protocol_counts_cannot_replace_case_defaults(tmp_path, field, value):
    path = tmp_path / "control"
    receipt = archive(path)
    receipt["protocol"][field] = value
    save(path, receipt)
    with pytest.raises(ValueError, match="differs from registered case"):
        api_publish.verify_run(path)


@pytest.mark.parametrize("frames", [1, 3, 31])
def test_gif_frame_knob_cannot_lower_required_metric_cadence(tmp_path, frames):
    path = tmp_path / "control"
    receipt = archive(path, frames=frames, failures=(575,))
    actual = api_publish.verify_run(path)
    assert len(actual["protocol"]["metric_evaluation_steps"]) == 25
    assert actual["protocol"]["metric_observations"] == 24
    assert actual["metric_passed"] and not actual["sustained_metric_passed"]
    assert actual["verdict"] == "FAIL"
    assert actual["gif_frames"] == len(receipt["protocol"]["media_steps"])


def test_extra_media_only_failure_does_not_change_the_frozen_terminal_grade(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path, frames=8)
    media_only = next(step for step in receipt["protocol"]["media_steps"]
                      if step not in receipt["protocol"]["metric_evaluation_steps"])
    for record in receipt["observations"]:
        record["passed"] = record["step"] != media_only
        record["failed_bounds"] = [] if record["passed"] else ["finite >= 1"]
        record["metrics"] = {"finite": int(record["passed"])}
    save(path, receipt)
    assert api_publish.verify_run(path)["verdict"] == "PASS"


@pytest.mark.parametrize("field", ["metric_evaluation_steps", "media_steps", "evaluation_steps"])
def test_forged_step_lists_cannot_declare_a_looser_protocol(tmp_path, field):
    path = tmp_path / "control"
    receipt = archive(path)
    receipt["protocol"][field] = [0, 600]
    save(path, receipt)
    with pytest.raises(ValueError, match="frozen step coverage"):
        api_publish.verify_run(path)


@pytest.mark.parametrize("value,bounds", [(1, []), (False, []), (True, ["invented failure"]), (False, [""])])
def test_observation_pass_flags_and_failures_must_be_binary_and_consistent(tmp_path, value, bounds):
    path = tmp_path / "control"
    receipt = archive(path)
    receipt["observations"][0].update(passed=value, failed_bounds=bounds)
    save(path, receipt)
    with pytest.raises(ValueError):
        api_publish.verify_run(path)


def test_completion_count_must_equal_the_executed_protocol(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path)
    receipt["completed_updates"] = 16
    save(path, receipt)
    with pytest.raises(ValueError, match="executed update budget"):
        api_publish.verify_run(path)


def test_changed_numeric_capture_cannot_publish_under_old_identity(tmp_path):
    path = tmp_path / "control"
    archive(path)
    with (path / "observations.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        api_publish.verify_run(path)


def test_non_media_metric_array_removal_fails_even_after_hash_is_rebound(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path, frames=3)
    with np.load(path / "observations.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files if key != "step25_view0_samples"}
    np.savez_compressed(path / "observations.npz", **arrays)
    receipt["artifacts"]["observations.npz"] = {"sha256": api_run.file_hash(path / "observations.npz"),
                                               "bytes": (path / "observations.npz").stat().st_size}
    save(path, receipt)
    with pytest.raises(ValueError, match="observations differ"):
        api_publish.verify_run(path)


def test_extra_goal_view_cannot_be_hidden_by_valid_artifact_hashes(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path)
    receipt["observations"][1]["views"].append(deepcopy(receipt["observations"][1]["views"][0]))
    save(path, receipt)
    with pytest.raises(ValueError, match="observations differ"):
        api_publish.verify_run(path)


def test_mismatched_decoded_gif_frames_fail_even_with_a_new_hash(tmp_path):
    path = tmp_path / "control"
    receipt = archive(path, frames=3)
    a, b = Image.new("RGB", (8, 8), "black"), Image.new("RGB", (8, 8), "white")
    a.save(path / "goal.gif", save_all=True, append_images=[b], duration=100)
    receipt["artifacts"]["goal.gif"] = {"sha256": api_run.file_hash(path / "goal.gif"),
                                       "bytes": (path / "goal.gif").stat().st_size}
    save(path, receipt)
    with pytest.raises(ValueError, match="goal GIF lost"):
        api_publish.verify_run(path)
