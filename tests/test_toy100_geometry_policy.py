"""Benchmark-local geometry ablations retain explicit, unlabeled initialization."""

import math

import pytest
import torch

from benchmarks.toy100.problems import sample_real
from benchmarks.toy100.train import (
    EMPIRICAL_INIT_SEED_OFFSET, _model_policy_receipt, make_trainer,
    resolve_config,
)


def _config(model: str):
    return resolve_config({
        "model": "gan", "problem": "grid100", "device": "cpu", "steps": 10,
        "seed": 1234, "z_dim": 2, "num_particles": 256, "batch_size": 128,
        "toy100_model": model, "output_noise_std": 0.0,
    })


@pytest.mark.parametrize("model,prior_kind,identity", [
    ("affine_square_v1", "uniform_square", True),
    ("affine_normal_v1", "normal", True),
    ("affine_square_random_v1", "uniform_square", False),
    ("affine_normal_random_v1", "normal", False),
    ("affine_empirical_box_v1", "empirical_box", True),
    ("affine_empirical_box_random_v1", "empirical_box", False),
])
def test_affine_geometry_receipt(model, prior_kind, identity):
    config, recipe = _config(model)
    trainer = make_trainer(config, recipe)
    receipt = _model_policy_receipt(trainer, config)
    assert receipt["prior_initialization"] == prior_kind
    assert receipt["generator_base_parameters"] == 6
    assert receipt["prior_shape"] == [256, 2]
    assert receipt["generator_initialization"] == (
        "identity" if identity else "torch_linear_default"
    )
    if identity:
        assert receipt["generator_initial_weight"] == [[1.0, 0.0], [0.0, 1.0]]
        assert receipt["generator_initial_bias"] == [0.0, 0.0]
    else:
        assert receipt["generator_initial_weight"] != [[1.0, 0.0], [0.0, 1.0]]
        assert max(abs(x) for row in receipt["generator_initial_weight"] for x in row) <= math.sqrt(.5)
    if prior_kind == "empirical_box":
        stream = torch.Generator().manual_seed(config["seed"] + EMPIRICAL_INIT_SEED_OFFSET)
        initial_batch = sample_real("grid100", config["batch_size"], generator=stream)
        assert receipt["init_data_lower"] == initial_batch.amin(dim=0).tolist()
        assert receipt["init_data_upper"] == initial_batch.amax(dim=0).tolist()
        assert receipt["init_data_samples"] == config["batch_size"]


def test_unknown_or_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="unsupported toy100_model"):
        _config("affine_target_centers")
    with pytest.raises(ValueError, match="requires z_dim=2"):
        resolve_config({"model": "gan", "toy100_model": "affine_normal_v1", "z_dim": 4})
