"""Adversarial stress testing and degenerate mock verification for AE-DDGAN paired metrics."""
import math
import pytest
import torch

from benchmarks.toy_audit.api_model_ddgan import MixtureTarget
from benchmarks.toy_audit.api_model_ae_ddgan import paired_metrics
from experiments.forge.rng import NamedStreams


def rng(component):
    return NamedStreams(0).generator("eval", component=component)


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_oracle_perfect_reconstruction(kind):
    """Oracle clean == reconstructed must pass all bounds."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    shuffled = clean.roll(1, 0)
    metrics, failures = paired_metrics(clean, clean, shuffled, target)
    assert len(failures) == 0, f"Expected 0 failures, got {failures}"
    assert metrics["paired_nmse"] == 0.0
    assert metrics["paired_worst_axis_nrmse"] == 0.0
    assert metrics["paired_shuffled_gain"] > 1000.0


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_degenerate_mock_zero(kind):
    """Zero mock: reconstructed is all zeros. Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    reconstructed = torch.zeros_like(clean)
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert "paired_shuffled_gain" in failures
    assert metrics["paired_nmse"] > 0.80
    assert metrics["paired_worst_axis_nrmse"] > 0.80
    assert abs(metrics["paired_shuffled_gain"] - 1.0) < 1e-4


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_degenerate_mock_mean(kind):
    """Mean mock: reconstructed is the batch empirical mean. Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    reconstructed = clean.mean(0).expand_as(clean)
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert "paired_shuffled_gain" in failures
    assert abs(metrics["paired_nmse"] - 1.0) < 0.10
    assert abs(metrics["paired_worst_axis_nrmse"] - 1.0) < 0.10
    assert abs(metrics["paired_shuffled_gain"] - 1.0) < 1e-4


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_degenerate_mock_independent_sample(kind):
    """Independent Sample mock: reconstructed is an independent draw from target.
    Marginal distribution is PERFECT, but sample pairing is absent. Must fail closed.
    """
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target1"))
    reconstructed = target.sample(4096, rng("target2"))
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert "paired_shuffled_gain" in failures
    assert abs(metrics["paired_nmse"] - 2.0) < 0.15
    assert abs(metrics["paired_worst_axis_nrmse"] - math.sqrt(2.0)) < 0.15
    assert abs(metrics["paired_shuffled_gain"] - 1.0) < 0.15


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_degenerate_mock_normal(kind):
    """Normal mock: reconstructed is standard Gaussian N(0, I). Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    reconstructed = torch.randn(clean.shape, generator=rng("normal"))
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert "paired_shuffled_gain" in failures
    assert metrics["paired_nmse"] > 0.50
    assert metrics["paired_worst_axis_nrmse"] > 0.50
    assert abs(metrics["paired_shuffled_gain"] - 1.0) < 0.20


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_degenerate_mock_rolled(kind):
    """Rolled mock: reconstructed is clean rolled by 1 (or 5).
    Identical empirical marginal, but completely un-paired. Must fail closed.
    """
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    reconstructed = clean.roll(5, 0)
    shuffled = clean.roll(10, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert "paired_shuffled_gain" in failures
    assert abs(metrics["paired_nmse"] - 2.0) < 0.15
    assert abs(metrics["paired_worst_axis_nrmse"] - math.sqrt(2.0)) < 0.15
    assert abs(metrics["paired_shuffled_gain"] - 1.0) < 0.10


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_degenerate_mock_constant(kind):
    """Constant mock: reconstructed is constant offset c = 5.0. Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    reconstructed = torch.full_like(clean, 5.0)
    shuffled = reconstructed.roll(1, 0)
    metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
    assert "paired_nmse" in failures
    assert "paired_worst_axis_nrmse" in failures
    assert "paired_shuffled_gain" in failures
    assert metrics["paired_nmse"] > 10.0
    assert metrics["paired_worst_axis_nrmse"] > 3.0
    assert abs(metrics["paired_shuffled_gain"] - 1.0) < 1e-4


@pytest.mark.parametrize("kind", ["gaussian", "bimodal"])
def test_degenerate_mock_nonfinite(kind):
    """Non-finite mock: reconstructed contains NaN or Inf. Must fail closed."""
    target = MixtureTarget(kind)
    clean = target.sample(4096, rng("target"))
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        reconstructed = clean.clone()
        reconstructed[0, 0] = bad_val
        shuffled = reconstructed.roll(1, 0)
        metrics, failures = paired_metrics(clean, reconstructed, shuffled, target)
        assert "paired_nonfinite" in failures
        assert math.isinf(metrics["paired_nmse"])
        assert math.isinf(metrics["paired_worst_axis_nrmse"])
        assert metrics["paired_shuffled_gain"] == 0.0
