"""Tiny CPU contracts for the public-step numerical benchmark, not GAN tests."""
import json

import pytest
import torch
import torch._dynamo  # Complete lazy optimizer imports before installing CUDA-port mocks.

from benchmarks import gram_profile
from particlegan.optim import GramGuardError, dualnorm


@pytest.fixture(autouse=True)
def forbid_cpu_benchmark_accelerator_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU benchmark must not query/synchronize an accelerator")

    for name in ("synchronize", "get_rng_state", "is_available", "device_count", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.mark.parametrize("options", [dict(size=1), dict(size=1153), dict(size=True),
    dict(matrices=0), dict(matrices=113), dict(repeats=0), dict(repeats=4),
    dict(warmup=-1), dict(warmup=2), dict(biases=1), dict(case="unknown"),
    dict(device="meta"), dict(device="cuda")])
def test_benchmark_rejects_unbounded_or_implicit_requests(options):
    with pytest.raises(ValueError):
        gram_profile.run_benchmark(**{**dict(device="cpu", size=2, matrices=1), **options})


def test_tiny_public_steps_use_identical_fp32_inputs_and_fresh_warmups():
    rng = torch.get_rng_state().clone()
    settings = (torch.get_default_dtype(), torch.get_num_threads(), torch.get_float32_matmul_precision())
    report = gram_profile.run_benchmark(device="cpu", size=2, matrices=2,
                                      repeats=2, warmup=1, biases=True)
    assert report["status"] == "PASS" and report["inputs_RNG_settings_unchanged"]
    assert report["seed"] == 0 and "not actual Sana" in report["fixture_scope"]
    assert len(report["records"]) == 6
    assert sum(not row["warmup"] for row in report["records"]) == 4
    for row in report["records"]:
        assert row["status"] == "PASS" and row["exception"] is None
        assert row["input_identity"] == report["input_identity"]
        assert len(row["actual_FP32_displacements"]) == 4
        assert all(value["dtype"] == "torch.float32" for value in row["actual_FP32_displacements"])
        assert row["clocks"] == [1] * 4  # Every warmup/repeat starts from fresh state.
        assert row["complete_optimizer_step_cuda_ms"] is None
        assert row["complete_optimizer_step_wall_seconds"] >= 0
        assert row["CPU64_oracle_error"]["maximum_absolute"] < 1e-7
    assert torch.equal(torch.get_rng_state(), rng)
    assert settings == (torch.get_default_dtype(), torch.get_num_threads(), torch.get_float32_matmul_precision())
    assert all(len(value["sha256"]) == 64 for bank in report["input_identity"].values() for value in bank)
    assert "not default FP32 parity" in report["numerical_acceptance"]
    json.dumps(report, allow_nan=False)


def test_rank_deficient_case_retains_public_fp32_cutoff_accuracy():
    report = gram_profile.run_benchmark(device="cpu", size=2, matrices=1, case="rank_deficient")
    assert report["status"] == "PASS"
    for row in report["records"]:
        assert row["CPU64_oracle_error"]["maximum_absolute"] < 1e-7


def test_near_cutoff_is_honest_hold_with_partial_state_evidence():
    report = gram_profile.run_benchmark(device="cpu", size=2, matrices=2, case="near_cutoff")
    assert report["status"] == "HOLD" and report["inputs_RNG_settings_unchanged"]
    selected = [row for row in report["records"] if row["profile"] == gram_profile.PROFILE]
    assert len(selected) == 1
    row = selected[0]
    assert row["exception"]["type"] == "GramGuardError" and row["exception"]["numerical_guard"]
    assert row["partial_state_discard_required"] and row["clocks"] == [1, 0]
    assert len(row["actual_FP32_displacements"]) == 2


def test_guard_exception_is_preserved_and_never_retried(monkeypatch):
    calls = []

    def hold(*args, **kwargs):
        calls.append(1)
        raise GramGuardError("explicit fixture HOLD")

    monkeypatch.setattr(dualnorm, "guarded_gram_polar", hold)
    report = gram_profile.run_benchmark(device="cpu", size=2, matrices=1, repeats=3)
    selected = [r for r in report["records"] if r["profile"] == gram_profile.PROFILE]
    assert calls == [1] and len(selected) == 1
    assert selected[0]["exception"]["message"] == "explicit fixture HOLD"
    assert selected[0]["status"] == "HOLD" and report["status"] == "HOLD"


def test_cli_validates_limits_before_writing(tmp_path):
    output = tmp_path / "not-created.json"
    with pytest.raises(SystemExit) as error:
        gram_profile.main(["--device", "cpu", "--matrices", "113", "--output", str(output)])
    assert error.value.code == 2 and not output.exists()


def test_cli_saves_hold_evidence_and_returns_nonzero(tmp_path):
    output = tmp_path / "hold.json"
    code = gram_profile.main(["--device", "cpu", "--size", "2", "--matrices", "1",
                              "--case", "near_cutoff", "--output", str(output)])
    assert code == 2
    result = json.loads(output.read_text())
    assert result["status"] == "HOLD" and len(result["source_sha256"]) == 4
