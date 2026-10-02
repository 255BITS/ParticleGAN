"""Media-only recovery controls; synthetic archives launch no training."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from benchmarks.toy_audit import api_contract, api_reframe, api_run
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
