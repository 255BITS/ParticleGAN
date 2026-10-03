"""The accuracy gate must audit real clouds and reject inconsistent evidence."""

import json
import shutil

import numpy as np
import pytest
import torch

from benchmarks.toy100.accuracy import PROTOCOL, evaluate_accuracy
from benchmarks.toy100.accuracy_gate import (
    HOLDOUT_N,
    HOLDOUT_SEED_OFFSETS,
    score_run,
)
from benchmarks.toy100.problems import sample_real


PROBLEM = "grid100"
COVERAGE_PASS = {"status": "PASS", "passed": True}


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def oracle_evidence(tmp_path_factory):
    """Five actual 20k target draws plus an independent 100k holdout."""
    directory = tmp_path_factory.mktemp("accuracy-oracle")
    checks = directory / "quality_checks"
    checks.mkdir()
    rng = torch.Generator(device="cpu").manual_seed(8021)
    def cloud(n):
        return sample_real(PROBLEM, n, generator=rng).numpy()

    events = []
    for step in range(1, 6):
        arrays = {name: cloud(20_000) for name in ("live", "ema", "target")}
        np.savez(checks / f"step_{step:06d}.npz", **arrays)
        for model in ("live", "ema"):
            events.append({"event": "eval", "step": step, "model": model,
                           "accuracy": evaluate_accuracy(arrays[model], PROBLEM)})
    (directory / "events.jsonl").write_text(
        "\n".join(json.dumps(row, allow_nan=False) for row in events) + "\n"
    )
    holdout = {name: cloud(HOLDOUT_N) for name in ("live", "ema", "target")}
    np.savez(directory / "holdout_samples.npz", **holdout)
    summary = {
        "eval_steps": [0, 1, 2, 3, 4, 5],
        "config": {"eval_samples": 20_000},
        "accuracy": {
            "protocol": PROTOCOL, "check_steps": [1, 2, 3, 4, 5],
            "sample_count": 20_000, "holdout_samples": HOLDOUT_N,
            "holdout_seed_offsets": HOLDOUT_SEED_OFFSETS,
        },
        "holdout": {name: evaluate_accuracy(samples, PROBLEM)
                    for name, samples in holdout.items()},
    }
    _write_json(directory / "summary.json", summary)
    return directory


def _copy_evidence(tmp_path, source):
    destination = tmp_path / "run"
    shutil.copytree(source, destination)
    return destination


def test_five_oracle_checks_and_independent_holdout_pass(oracle_evidence):
    row = score_run(oracle_evidence, PROBLEM, COVERAGE_PASS)
    assert row["status"] == "PASS"
    assert row["passed"]
    assert [check["step"] for check in row["terminal_checks"]] == [1, 2, 3, 4, 5]
    assert all(check["passed"] for check in row["terminal_checks"])
    assert row["holdout_metrics"]["passed"]
    assert row["oracle_metrics"]["passed"]


def test_forged_terminal_accuracy_receipt_is_invalid(tmp_path, oracle_evidence):
    directory = _copy_evidence(tmp_path, oracle_evidence)
    path = directory / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-2]["accuracy"]["center_rms_sigma"] = 0.0
    path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    row = score_run(directory, PROBLEM, COVERAGE_PASS)
    assert row["status"] == "INVALID"
    assert "recorded accuracy differs" in row["reason"]


def test_forged_ema_receipt_is_invalid_even_though_ema_is_diagnostic(tmp_path, oracle_evidence):
    directory = _copy_evidence(tmp_path, oracle_evidence)
    path = directory / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[1]["accuracy"]["radial_ks"] = 0.0
    path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    row = score_run(directory, PROBLEM, COVERAGE_PASS)
    assert row["status"] == "INVALID"
    assert "saved ema samples" in row["reason"]


def test_forged_holdout_receipt_is_invalid(tmp_path, oracle_evidence):
    directory = _copy_evidence(tmp_path, oracle_evidence)
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    summary["holdout"]["live"]["mass_tv"] = 0.0
    _write_json(path, summary)
    row = score_run(directory, PROBLEM, COVERAGE_PASS)
    assert row["status"] == "INVALID"
    assert "recorded holdout differs" in row["reason"]


def test_corrupt_oracle_target_cannot_support_a_pass(tmp_path, oracle_evidence):
    directory = _copy_evidence(tmp_path, oracle_evidence)
    path = directory / "holdout_samples.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in ("live", "ema", "target")}
    arrays["target"] = np.zeros_like(arrays["target"])
    np.savez(path, **arrays)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    # Forge the diagnostic receipt as well: the gate must check the oracle.
    summary["holdout"]["target"] = evaluate_accuracy(arrays["target"], PROBLEM)
    _write_json(summary_path, summary)
    row = score_run(directory, PROBLEM, COVERAGE_PASS)
    assert row["status"] == "INVALID"
    assert "target reference fails" in row["reason"]


def test_missing_terminal_cloud_is_invalid(tmp_path, oracle_evidence):
    directory = _copy_evidence(tmp_path, oracle_evidence)
    (directory / "quality_checks" / "step_000003.npz").unlink()
    row = score_run(directory, PROBLEM, COVERAGE_PASS)
    assert row["status"] == "INVALID"
    assert not row["passed"]


def test_original_coverage_remains_mandatory(oracle_evidence):
    row = score_run(oracle_evidence, PROBLEM, {"status": "FAIL", "passed": False})
    assert row["status"] == "FAIL"
    assert not row["passed"]
    assert all(check["passed"] for check in row["terminal_checks"])
    assert row["holdout_metrics"]["passed"]


def test_corrupt_holdout_archive_is_invalid(tmp_path, oracle_evidence):
    directory = _copy_evidence(tmp_path, oracle_evidence)
    (directory / "holdout_samples.npz").write_bytes(b"not a valid NPZ")
    row = score_run(directory, PROBLEM, COVERAGE_PASS)
    assert row["status"] == "INVALID"
    assert not row["passed"]
