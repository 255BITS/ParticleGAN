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


def media_review(path, raw_path, receipt):
    """Synthetic render identity control; no optimizer, evaluator or provider."""
    path.mkdir(parents=True)
    count = receipt["gif_frames"]
    images = [Image.new("RGB", (8, 8), (255 - index * 13 % 255, index * 23 % 255, 100))
              for index in range(count)]
    images[0].save(path / "reviewed-goal.gif", save_all=True, append_images=images[1:],
                   duration=100, optimize=False)
    review = {"schema": "particlegan_api_toy_media_review_v1",
              "case_id": receipt["case"]["id"], "raw_receipt": str((raw_path / "receipt.json").resolve()),
              "raw_receipt_sha256": api_run.file_hash(raw_path / "receipt.json"),
              "raw_artifacts": deepcopy(receipt["artifacts"]),
              "training_source_commit": receipt["source"]["commit"],
              "renderer_source": {"commit": "separate-renderer-software-control",
                                  "files_sha256": {f"benchmarks/toy_audit/{name}.py": "a" * 64
                                                   for name in ("api_run", "api_reframe", "api_contract")}},
              "verdict": receipt["verdict"], "metric_passed": receipt["metric_passed"],
              "sustained_metric_passed": receipt["sustained_metric_passed"],
              "default_protocol_complete": receipt["default_protocol_complete"],
              "failed_bounds": deepcopy(receipt["failed_bounds"]),
              "final_metrics": deepcopy(receipt["observations"][-1]["metrics"]),
              "media_steps": deepcopy(receipt["protocol"]["media_steps"]),
              "reviewed_gif": {"file": "reviewed-goal.gif",
                               "sha256": api_run.file_hash(path / "reviewed-goal.gif"),
                               "bytes": (path / "reviewed-goal.gif").stat().st_size, "frames": count},
              "annotations": {"default_verdict_displayed": receipt["verdict"],
                              "goal_annotations": [{"step": receipt["protocol"]["media_steps"][0],
                                                    "view": 0, "reference_camera": "synthetic-control"}],
                              "numeric_observations_changed": False},
              "training_or_rescoring": False, "raw_files_unchanged": True}
    api_run.write_json(path / "media-review.json", review)
    return review


@pytest.mark.parametrize("updates,failures", [(600, ()), (16, ()), (600, (600,))])
def test_reviewed_media_preserves_original_pass_short_and_failure_grades(tmp_path, updates, failures):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    receipt = archive(raw, updates=updates, frames=3, failures=failures)
    before = {name: (raw / name).read_bytes()
              for name in ("receipt.json", "goal.gif", "observations.npz", "final-state.pt")}
    review = media_review(reviewed, raw, receipt)
    assert api_publish.verify_media_review(raw, reviewed) == review
    assert api_publish.verify_run(raw) == receipt
    assert before == {name: (raw / name).read_bytes() for name in before}


@pytest.mark.parametrize("field,value", [
    ("schema", "unbound-render"), ("case_id", "another-case"),
    ("raw_receipt", "receipt.json"), ("raw_receipt", "/unrelated/receipt.json"),
    ("raw_receipt_sha256", "0" * 64), ("training_source_commit", "different-training-source"),
    ("verdict", "FAIL"), ("metric_passed", False), ("metric_passed", 1),
    ("sustained_metric_passed", False), ("default_protocol_complete", False),
    ("failed_bounds", ["invented failure"]), ("final_metrics", {"finite": 0}),
    ("training_or_rescoring", True), ("training_or_rescoring", 0),
    ("raw_files_unchanged", False), ("raw_files_unchanged", 1),
])
def test_media_review_cannot_change_original_identity_metrics_or_grade(tmp_path, field, value):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    receipt = archive(raw, frames=3)
    review = media_review(reviewed, raw, receipt)
    review[field] = value
    api_run.write_json(reviewed / "media-review.json", review)
    with pytest.raises(ValueError):
        api_publish.verify_media_review(raw, reviewed)


def test_media_review_cannot_substitute_original_artifacts(tmp_path):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    review = media_review(reviewed, raw, archive(raw, frames=3))
    review["raw_artifacts"]["goal.gif"]["sha256"] = review["reviewed_gif"]["sha256"]
    api_run.write_json(reviewed / "media-review.json", review)
    with pytest.raises(ValueError, match="raw_artifacts"):
        api_publish.verify_media_review(raw, reviewed)


@pytest.mark.parametrize("steps", [[0, 600], [0, 300, 300, 600], [600, 300, 0], [0.0, 300, 600]])
def test_media_review_cannot_change_retained_state_steps(tmp_path, steps):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    review = media_review(reviewed, raw, archive(raw, frames=3))
    review["media_steps"] = steps
    api_run.write_json(reviewed / "media-review.json", review)
    with pytest.raises(ValueError, match="media steps"):
        api_publish.verify_media_review(raw, reviewed)


@pytest.mark.parametrize("change", ["missing_source", "missing_file", "invalid_hash",
                                    "wrong_label", "changed_observations", "uncaptured_step", "uncaptured_view"])
def test_review_requires_renderer_identity_and_valid_goal_annotations(tmp_path, change):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    review = media_review(reviewed, raw, archive(raw, frames=3))
    if change == "missing_source":
        del review["renderer_source"]
    elif change == "missing_file":
        del review["renderer_source"]["files_sha256"]["benchmarks/toy_audit/api_reframe.py"]
    elif change == "invalid_hash":
        review["renderer_source"]["files_sha256"]["benchmarks/toy_audit/api_run.py"] = "z" * 64
    elif change == "wrong_label":
        review["annotations"]["default_verdict_displayed"] = "FAIL"
    elif change == "changed_observations":
        review["annotations"]["numeric_observations_changed"] = True
    elif change == "uncaptured_step":
        review["annotations"]["goal_annotations"][0]["step"] = 25
    else:
        review["annotations"]["goal_annotations"][0]["view"] = 1
    api_run.write_json(reviewed / "media-review.json", review)
    with pytest.raises(ValueError):
        api_publish.verify_media_review(raw, reviewed)


@pytest.mark.parametrize("field,value", [("file", "../goal.gif"), ("sha256", "0" * 64),
                                        ("bytes", 1), ("frames", 2), ("frames", 3.0)])
def test_reviewed_gif_identity_and_media_count_are_checked(tmp_path, field, value):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    review = media_review(reviewed, raw, archive(raw, frames=3))
    review["reviewed_gif"][field] = value
    api_run.write_json(reviewed / "media-review.json", review)
    with pytest.raises(ValueError):
        api_publish.verify_media_review(raw, reviewed)


def test_reviewed_gif_decoded_count_is_checked_even_with_rebound_identity(tmp_path):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    review = media_review(reviewed, raw, archive(raw, frames=3))
    a, b = Image.new("RGB", (8, 8), "black"), Image.new("RGB", (8, 8), "white")
    a.save(reviewed / "reviewed-goal.gif", save_all=True, append_images=[b], duration=100)
    review["reviewed_gif"].update(sha256=api_run.file_hash(reviewed / "reviewed-goal.gif"),
                                  bytes=(reviewed / "reviewed-goal.gif").stat().st_size)
    api_run.write_json(reviewed / "media-review.json", review)
    with pytest.raises(ValueError, match="lost actual observed states"):
        api_publish.verify_media_review(raw, reviewed)


@pytest.mark.parametrize("change", ["error", "source_changed", "forged_full", "changed_numeric_capture"])
def test_review_does_not_weaken_original_execution_or_archive_checks(tmp_path, change):
    raw, reviewed = tmp_path / "raw", tmp_path / "review"
    receipt = archive(raw, frames=3, updates=16 if change == "forged_full" else 600)
    if change == "error":
        receipt["status"] = "ERROR"
    elif change == "source_changed":
        receipt["source_unchanged"] = False
    elif change == "forged_full":
        receipt.update(default_protocol_complete=True, passed=True, verdict="PASS", failed_bounds=[])
    else:
        with (raw / "observations.npz").open("ab") as handle:
            handle.write(b"changed")
    save(raw, receipt)
    media_review(reviewed, raw, receipt)
    with pytest.raises(ValueError):
        api_publish.verify_media_review(raw, reviewed)


@pytest.mark.parametrize("reviewed_media", [False, True])
def test_publication_copies_selected_media_and_keeps_raw_evidence_and_grades(tmp_path, monkeypatch, reviewed_media):
    runs, review_root = tmp_path / "runs", tmp_path / "review"
    runs.mkdir()
    name = "publication-software-control"
    raw = runs / name
    receipt = archive(raw, frames=3, updates=16)
    api_run.write_json(runs / "summary.json", {"cases": [{"id": name}]})
    reviewed = review_root / name
    review = media_review(reviewed, raw, receipt)
    monkeypatch.setattr(api_contract, "discover", lambda: {name: receipt["case"]})
    monkeypatch.setattr(api_run, "inventory", lambda cases: {"coverage": {"missing": [], "api_variants": 1}})
    output = tmp_path / "publication"
    api_publish.publish([runs], output, media_review=review_root if reviewed_media else None)
    published = api_publish.read(output / "runs.json")
    row = published["cases"][0]
    source = reviewed / "reviewed-goal.gif" if reviewed_media else raw / "goal.gif"
    assert (output / row["gif"]).read_bytes() == source.read_bytes()
    assert row["gif_sha256"] == api_run.file_hash(source)
    assert row["raw_artifacts"] == receipt["artifacts"]
    assert row["raw_receipt_sha256"] == api_run.file_hash(raw / "receipt.json")
    assert row["verdict"] == "FAIL" and row["metric_passed"] and not row["default_protocol_complete"]
    assert published["media_review_used"] is reviewed_media
    assert published["training_or_rescoring_by_publication"] is False
    if reviewed_media:
        provenance = row["media_review"]
        assert provenance["raw_sidecar_sha256"] == api_run.file_hash(reviewed / "media-review.json")
        assert published["media_renderer_source_identities"][provenance["renderer_source_identity"]] == review["renderer_source"]
        assert published["source_identities"][row["source_identity"]] == receipt["source"]
    else:
        assert "media_review" not in row and not published["media_renderer_source_identities"]


def test_media_review_cli_option_passes_only_the_separate_archive(tmp_path, monkeypatch):
    calls = []
    def fake_publish(runs, output, *, media_review=None):
        calls.append((runs, output, media_review))
        return {"coverage": {"api_variants": 1, "missing": []}, "missing_api_media": []}
    monkeypatch.setattr(api_publish, "publish", fake_publish)
    assert api_publish.main(["--runs", str(tmp_path / "raw"), "--output", str(tmp_path / "published"),
                             "--media-review", str(tmp_path / "reviewed")]) == 0
    assert calls == [([tmp_path / "raw"], tmp_path / "published", tmp_path / "reviewed")]


@pytest.mark.parametrize("value", ["nan", "1", None, {"finite": 1}, float("nan"), float("inf")])
def test_passing_metrics_cannot_be_nonnumeric_or_nonfinite(tmp_path, value):
    raw = tmp_path / "raw"
    receipt = archive(raw, frames=3)
    receipt["observations"][0]["metrics"]["finite"] = value
    # Includes malformed numeric NaN/Inf JSON as well as the runner's string encoding.
    (raw / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="must be finite numeric"):
        api_publish.verify_run(raw)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_passing_samples_cannot_be_nonfinite_even_after_artifact_hash_is_rebound(tmp_path, value):
    raw = tmp_path / "raw"
    receipt = archive(raw, frames=3)
    with np.load(raw / "observations.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["step0_view0_samples"][0, 0] = value
    np.savez_compressed(raw / "observations.npz", **arrays)
    receipt["artifacts"]["observations.npz"].update(sha256=api_run.file_hash(raw / "observations.npz"),
                                                   bytes=(raw / "observations.npz").stat().st_size)
    save(raw, receipt)
    with pytest.raises(ValueError, match="nonfinite actual samples"):
        api_publish.verify_run(raw)


def test_nonfinite_failed_observation_remains_visible_as_failure_evidence(tmp_path):
    raw = tmp_path / "raw"
    receipt = archive(raw, frames=3, failures=(0,))
    receipt["observations"][0]["metrics"]["finite"] = "nan"
    with np.load(raw / "observations.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["step0_view0_samples"][0, 0] = np.nan
    np.savez_compressed(raw / "observations.npz", **arrays)
    receipt["artifacts"]["observations.npz"].update(sha256=api_run.file_hash(raw / "observations.npz"),
                                                   bytes=(raw / "observations.npz").stat().st_size)
    save(raw, receipt)
    assert api_publish.verify_run(raw)["observations"][0]["passed"] is False
