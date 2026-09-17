"""Training-loop contracts beyond the isolated Hopfield read implementation."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def example(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/100gaussians.py"
    spec = importlib.util.spec_from_file_location("hopfield_training_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "save_fake_scatter", lambda *args, **kwargs: None)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(previous)


def test_training_optimizes_clamps_and_emas_beta_once(example, monkeypatch, tmp_path):
    reads = []
    original_read = example.HopfieldRead
    original_step = torch.optim.Adam.step
    beta_steps = []

    def capture_read(prior, **kwargs):
        read = original_read(prior, **kwargs)
        reads.append((read, prior.z.detach().clone()))
        return read

    def step_with_beta_overshoot(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        live = reads[0][0]
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        beta_occurrences = sum(p is live.log_beta for p in parameters)
        if beta_occurrences:
            assert beta_occurrences == 1
            assert sum(p is live.prior.z for p in parameters) == 1
            assert live.log_beta.grad is not None
            assert torch.isfinite(live.log_beta.grad)
            assert live.log_beta.grad.abs() > 0
            beta_steps.append(1)
            # Force an optimizer overshoot so omitting the training-loop clamp
            # fails independently of how far a normal one-step Adam update moves.
            with torch.no_grad():
                live.log_beta.fill_(math.log(256) + 1)
        return result

    monkeypatch.setattr(example, "HopfieldRead", capture_read)
    monkeypatch.setattr(torch.optim.Adam, "step", step_with_beta_overshoot)
    monkeypatch.setattr(example, "mode_coverage", lambda *args, **kwargs: (0, 0.0))
    result = example.train(
        epochs=1, steps_per_epoch=1, batch_size=16, num_particles=100,
        read="hopfield", learn_beta=True, beta=16, ema_decay=0.5,
        out_dir=str(tmp_path), device_str="cpu", return_details=True,
    )
    assert len(beta_steps) == 1
    assert result["ema_read"].prior is result["ema_prior"]
    assert not result["ema_read"].log_beta.requires_grad
    assert float(result["read"].log_beta.detach()) == pytest.approx(math.log(256))
    assert float(result["ema_read"].log_beta) == pytest.approx(
        0.5 * math.log(16) + 0.5 * math.log(256)
    )
    initial_prior = reads[0][1]
    torch.testing.assert_close(
        result["ema_prior"].z,
        initial_prior * 0.5 + result["prior"].z.detach() * 0.5,
    )


def test_study_logs_completed_steps_and_evaluates_exact_final(example, tmp_path):
    result = example.train(
        epochs=1, steps_per_epoch=3, batch_size=16, num_particles=100,
        read="uniform", dataset="imbalanced", study_metrics=True,
        n_eval=128, eval_batch_size=64, log_interval=2,
        out_dir=str(tmp_path), device_str="cpu", return_details=True,
    )
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [2, 3]
    assert [row["step"] for row in result["history"]] == [2, 3]
    final = json.loads((tmp_path / "final_metrics.json").read_text())
    assert final == result["final"]
    assert final["step"] == 3
    assert "steps_to_tv" in final
    assert len(final["mode_weights"]) == 100
    assert result["final_samples"].shape == (128, 2)
    floor = json.loads((tmp_path / "sampling_floor.json").read_text())
    assert floor["repeats"] == 20
    assert len(floor["values"]) == 20
    assert floor["mean"] > 0
    persisted = np.load(tmp_path / "target_weights.npy")
    np.testing.assert_array_equal(persisted, result["target_weights"].cpu().numpy())
    assert persisted.max() / persisted.min() == pytest.approx(100)
