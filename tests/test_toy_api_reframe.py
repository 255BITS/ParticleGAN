"""Media-only recovery controls; synthetic archives launch no training."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from particlegan import get_recipe
from benchmarks.toy_audit import api_contract, api_publish, api_reframe, api_run
from test_toy_api_publish import archive, save


def raw_archive(tmp_path, *, failures=(), name="publication-software-control"):
    root = tmp_path / "raw"
    root.mkdir(exist_ok=True)
    path = root / name
    receipt = archive(path, frames=3, metric_count=6, failures=failures)
    receipt["case"]["id"] = name
    save(path, receipt)
    return path, receipt


def raw_bytes(path):
    return {name: (path / name).read_bytes()
            for name in ("receipt.json", "goal.gif", "observations.npz", "final-state.pt")}


def test_reframe_preserves_all_raw_bytes_metrics_gates_and_media_steps(tmp_path, monkeypatch):
    path, receipt = raw_archive(tmp_path, failures=(200,))
    assert receipt["metric_passed"] and receipt["verdict"] == "FAIL"
    before = raw_bytes(path)
    captured = {}
    actual = api_run.render_gif

    def render(case, records, output, **kwargs):
        captured.update(case=deepcopy(case), records=deepcopy(records), kwargs=kwargs)
        return actual(case, records, output, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("media review must not construct or train a fixture")

    monkeypatch.setattr(api_run, "render_gif", render)
    monkeypatch.setattr(api_contract, "build", forbidden)
    output = tmp_path / "review" / path.name
    review = api_reframe.review_run(path, output)
    assert raw_bytes(path) == before
    assert review["raw_files_unchanged"] and review["training_or_rescoring"] is False
    assert captured["case"] == receipt["case"]
    assert [record["step"] for record in captured["records"]] == receipt["protocol"]["media_steps"]
    assert len(receipt["observations"]) > len(captured["records"])
    assert captured["kwargs"]["final_verdict"] == review["verdict"] == "FAIL"
    assert review["annotations"]["default_verdict_displayed"] == "FAIL"
    assert review["annotations"]["numeric_observations_changed"] is False
    assert review["raw_artifacts"] == receipt["artifacts"]
    assert review["raw_receipt_sha256"] == api_run.file_hash(path / "receipt.json")
    for field in ("metric_passed", "sustained_metric_passed", "default_protocol_complete", "failed_bounds"):
        assert review[field] == receipt[field]
    assert review["final_metrics"] == receipt["observations"][-1]["metrics"]
    assert set(review["renderer_source"]["files_sha256"]) == {
        f"benchmarks/toy_audit/{name}" for name in ("api_run.py", "api_reframe.py", "api_contract.py")}
    with np.load(path / "observations.npz", allow_pickle=False) as arrays:
        originals = {record["step"]: record for record in receipt["observations"]}
        for record in captured["records"]:
            assert record["metrics"] == originals[record["step"]]["metrics"]
            assert record["failed_bounds"] == originals[record["step"]]["failed_bounds"]
            for index, view in enumerate(record["views"]):
                for role in ("target", "samples"):
                    np.testing.assert_array_equal(view[role], arrays[f"step{record['step']}_view{index}_{role}"])
    gif_path = output / review["reviewed_gif"]["file"]
    assert review["reviewed_gif"]["sha256"] == api_run.file_hash(gif_path)
    assert review["reviewed_gif"]["bytes"] == gif_path.stat().st_size
    with Image.open(gif_path) as gif:
        assert gif.n_frames == review["reviewed_gif"]["frames"] == 3


@pytest.mark.parametrize("mutation", ["samples", "metric"])
def test_renderer_mutation_is_rejected_without_changing_raw_capture(tmp_path, monkeypatch, mutation):
    path, _ = raw_archive(tmp_path)
    before = raw_bytes(path)
    actual = api_run.render_gif

    def mutating(case, records, output, **kwargs):
        annotations = actual(case, records, output, **kwargs)
        if mutation == "samples":
            records[0]["views"][0]["samples"][0, 0] += 100
        else:
            records[0]["metrics"]["finite"] = 7
        return annotations

    monkeypatch.setattr(api_run, "render_gif", mutating)
    output = tmp_path / "review" / path.name
    with pytest.raises(ValueError, match="changed retained observations"):
        api_reframe.review_run(path, output)
    assert raw_bytes(path) == before
    assert not (output / "media-review.json").exists()
    assert not (output / "reviewed-goal.gif").exists()


def test_original_archive_and_previous_review_cannot_be_overwritten(tmp_path):
    path, _ = raw_archive(tmp_path)
    before = raw_bytes(path)
    with pytest.raises(ValueError, match="outside the original"):
        api_reframe.review_run(path, path)
    output = tmp_path / "review" / path.name
    output.mkdir(parents=True)
    sentinel = output / "reviewed-goal.gif"
    sentinel.write_bytes(b"existing immutable reviewed media")
    with pytest.raises(FileExistsError):
        api_reframe.review_run(path, output)
    assert sentinel.read_bytes() == b"existing immutable reviewed media"
    assert raw_bytes(path) == before


def test_forged_original_verdict_is_rejected_before_review_is_created(tmp_path):
    path, receipt = raw_archive(tmp_path, failures=(200,))
    receipt.update(verdict="PASS", passed=True, sustained_metric_passed=True, failed_bounds=[])
    save(path, receipt)
    before = raw_bytes(path)
    output = tmp_path / "review" / path.name
    with pytest.raises(ValueError, match="sustained_metric_passed"):
        api_reframe.review_run(path, output)
    assert raw_bytes(path) == before and not output.exists()


def test_archive_reports_error_and_pending_without_fabricating_goal_media(tmp_path):
    path, receipt = raw_archive(tmp_path)
    receipt["status"] = "ERROR"
    save(path, receipt)
    (path.parent / "pending-case").mkdir()
    before = raw_bytes(path)
    output = tmp_path / "review"
    records = api_reframe.review_archive(path.parent, output)
    assert {record["execution_status"] for record in records} == {"ERROR", "PENDING"}
    assert all(record["status"] == "SKIPPED" for record in records)
    assert not list(output.glob("*/reviewed-goal.gif"))
    assert not list(output.glob("*/media-review.json"))
    assert raw_bytes(path) == before


def test_process_workers_keep_distinct_case_reviews_and_original_capture_bytes(tmp_path):
    first, _ = raw_archive(tmp_path, name="software-case-a")
    second, _ = raw_archive(tmp_path, name="software-case-b")
    before = {path.name: raw_bytes(path) for path in (first, second)}
    output = tmp_path / "review"
    records = api_reframe.review_archive(first.parent, output, jobs=2)
    assert [record["id"] for record in records] == ["software-case-a", "software-case-b"]
    assert all(record["status"] == "REVIEWED" for record in records)
    assert {path.name: raw_bytes(path) for path in (first, second)} == before
    for record in records:
        review = json.loads((output / record["media_review"]).read_text())
        assert review["case_id"] == record["id"] and review["verdict"] == "PASS"


def failed_archive(tmp_path, name):
    """Synthetic ERROR controls with registered metadata, never trained evidence."""
    root = tmp_path / "raw"
    root.mkdir()
    path = root / name
    case = api_run.json_value(api_contract.discover()[name])
    receipt = archive(path, updates=case["default_steps"], default_updates=case["default_steps"],
                      samples=case["eval_samples"], frames=9)
    export = name in api_reframe._EXPORT_CASES
    completed = 800 if export else 1200
    receipt["case"] = case
    receipt["protocol"]["evaluation_samples"] = case["eval_samples"]
    receipt["protocol"]["default_evaluation_samples"] = case["eval_samples"]
    receipt.update(status="ERROR", passed=False, verdict="FAIL", completed_updates=completed,
                   default_protocol_complete=export)
    receipt["observations"] = [o for o in receipt["observations"] if o["step"] <= completed]
    arrays = {}
    for record in receipt["observations"]:
        record.update(passed=export, failed_bounds=[] if export else ["hq >= 0.9", "max_cov_eigen <= 1.5", "max_radial_ks <= 0.1"])
        record["metrics"] = {"normalized_rmse": .02} if export else {"hq": .8, "max_cov_eigen": 70., "max_radial_ks": .2}
        record["views"] = [{"kind": "image" if export else "scatter", "title": "Synthetic failure control"}]
        for role in ("target", "samples"):
            shape = (8, 2, 1, 32) if export else (8, 2)
            arrays[f"step{record['step']}_view0_{role}"] = np.full(shape, record["step"] / 800, dtype=np.float32)
    np.savez_compressed(path / "observations.npz", **arrays)
    source_files = {"benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_contract.py",
                    f"benchmarks/toy_audit/{case['provider']}.py", "particlegan/recipes.py"}
    receipt["source"] = {"commit": api_reframe._FAILURE_SOURCE,
                         "files_sha256": {key: api_reframe._frozen_file_hash(key) for key in source_files}}
    receipt["seed"] = 24002
    receipt["runtime"] = {"device": "cpu", "torch_threads": 1}
    if export:
        recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=16,
            lr=.000204, d_lr_mult=1.5, prior_lr_mult=10., output_noise_std=1.3,
            betas=(0., .999), birth_death_backend="auto", reopen_guard="settled")
        receipt.update(metric_passed=True, sustained_metric_passed=True,
                       failed_bounds=[api_reframe._EXPORT_ERROR], artifacts={}, gif_frames=0)
        receipt["api_components"] = ["Recipe", "E22Policy", "RoutedRows", "GANLoss", "native policy lifecycle", "public initialization"]
        (path / "goal.gif").unlink()
        (path / "final-state.pt").unlink()
    else:
        recipe = get_recipe("atlas", num_particles=256, z_dim=4, batch_size=128)
        receipt.update(metric_passed=None, sustained_metric_passed=None,
            failed_bounds=["API execution or metric error: RuntimeError: scientific prerequisite failed at update1200: " + str(receipt["observations"][-1]["failed_bounds"])])
        receipt["api_components"] = ["particlegan.Recipe", "particlegan.GANTrainer", "particlegan.ParticlePrior"]
        receipt["gif_frames"] = len([step for step in receipt["protocol"]["media_steps"] if step <= completed])
        images = [Image.new("RGB", (8, 8), (index * 31 % 255, index * 17 % 255, index))
                  for index in range(receipt["gif_frames"])]
        images[0].save(path / "goal.gif", save_all=True, append_images=images[1:], duration=100, optimize=False)
        receipt["artifacts"] = {name: {"sha256": api_run.file_hash(path / name), "bytes": (path / name).stat().st_size}
                               for name in ("goal.gif", "observations.npz", "final-state.pt")}
    receipt["recipe"] = api_run.json_value(recipe.to_dict())
    save(path, receipt)
    return path, receipt


@pytest.mark.parametrize("name", ["api-critic-lag-current", "api-ring8-shift"])
def test_failure_supplement_keeps_error_and_actual_terminal_without_qualification(tmp_path, name):
    path, receipt = failed_archive(tmp_path, name)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    output = tmp_path / "failure-review" / name
    review = api_reframe.review_failure_run(path, output)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    assert review["original_execution_status"] == "ERROR" and review["verdict"] == "FAIL"
    assert review["qualification_upgrade"] is False and review["training_or_rescoring"] is False
    assert review["final_metrics"] == receipt["observations"][-1]["metrics"]
    assert review["observations"] == receipt["observations"]
    assert review["original_media_steps"] == receipt["protocol"]["media_steps"]
    assert review["media_steps"][-1] == receipt["completed_updates"]
    assert review["annotations"]["default_verdict_displayed"] == review["displayed_verdict"]
    assert not (output / "media-review.json").exists()
    with pytest.raises(ValueError, match="unsuccessful API execution"):
        api_publish.verify_run(path)
    if name in api_reframe._EXPORT_CASES:
        assert review["raw_artifacts"] == {}
        assert review["artifact_attestations"] == {"observations.npz": "review_time_only"}
        assert "numeric gate PASS" in review["displayed_verdict"]
        assert review["annotations"]["goal_annotations"]
    else:
        assert review["terminal_frame_added"] and review["media_steps"] == [0, 450, 900, 1200]
        assert set(review["artifact_attestations"].values()) == {"original_receipt"}
        assert "continuation unattempted" in review["displayed_verdict"]


@pytest.mark.parametrize("change", ["source", "source_hash", "runtime", "recipe", "error", "missing_step", "reordered_step", "shape"])
def test_unknown_or_incomplete_failure_evidence_is_rejected(tmp_path, change):
    path, receipt = failed_archive(tmp_path, "api-critic-lag-current")
    if change == "source":
        receipt["source"]["commit"] = "different-source"
    elif change == "source_hash":
        receipt["source"]["files_sha256"]["particlegan/recipes.py"] = "a" * 64
    elif change == "runtime":
        receipt["runtime"]["torch_threads"] = 8
    elif change == "recipe":
        receipt["recipe"]["lr"] *= 2
    elif change == "error":
        receipt["failed_bounds"] = ["different execution failure"]
    elif change == "missing_step":
        del receipt["observations"][2]
    elif change == "reordered_step":
        receipt["observations"][2:4] = reversed(receipt["observations"][2:4])
    else:
        with np.load(path / "observations.npz", allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["step0_view0_target"] = arrays["step0_view0_target"].reshape(8, 1, 2, 32)
        np.savez_compressed(path / "observations.npz", **arrays)
    save(path, receipt)
    with pytest.raises(ValueError):
        api_reframe.verify_failure_view(path)


def test_failure_views_are_opt_in_and_separate_from_completed_reviews(tmp_path):
    path, _ = failed_archive(tmp_path, "api-critic-lag-current")
    plain = api_reframe.review_archive(path.parent, tmp_path / "plain-review")
    assert plain[0]["status"] == "SKIPPED"
    enabled = api_reframe.review_archive(path.parent, tmp_path / "failure-review", include_failure_views=True)
    assert enabled[0]["status"] == "FAILURE_VIEW" and enabled[0]["verdict"] == "FAIL"
    assert "failure_review" in enabled[0] and "media_review" not in enabled[0]
