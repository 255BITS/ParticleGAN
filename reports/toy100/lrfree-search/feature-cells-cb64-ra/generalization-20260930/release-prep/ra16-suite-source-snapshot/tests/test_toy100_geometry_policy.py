"""Benchmark-local geometry ablations retain explicit, unlabeled initialization."""

import hashlib
import json
import math
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.toy100.problems import sample_real
from benchmarks.toy100.train import (
    EMPIRICAL_INIT_SEED_OFFSET, _model_policy_receipt, make_trainer,
    resolve_config, train,
)
from benchmarks.toy_suite import _check_toy100_policy
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy


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
    ("affine_moment_box_v1", "moment_box", True),
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
    if prior_kind == "moment_box":
        stream = torch.Generator().manual_seed(config["seed"] + EMPIRICAL_INIT_SEED_OFFSET)
        initial_batch = sample_real("grid100", config["batch_size"], generator=stream)
        mean = initial_batch.mean(dim=0)
        std = initial_batch.std(dim=0, unbiased=False)
        half_width = math.sqrt(3.0) * std
        assert receipt["init_data_mean"] == mean.tolist()
        assert receipt["init_data_std"] == std.tolist()
        assert receipt["init_data_lower"] == (mean - half_width).tolist()
        assert receipt["init_data_upper"] == (mean + half_width).tolist()
        assert receipt["init_data_samples"] == config["batch_size"]


def test_unknown_or_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="unsupported toy100_model"):
        _config("affine_target_centers")
    with pytest.raises(ValueError, match="requires z_dim=2"):
        resolve_config({"model": "gan", "toy100_model": "affine_normal_v1", "z_dim": 4})


def test_moment_box_archive_regrade_binds_unlabelled_calibration_data(tmp_path):
    config = dict(
        model="gan", problem="grid100", device="cpu", seed=1234, steps=6,
        z_dim=2, num_particles=32, batch_size=16, d_hidden=8, n_hidden=1,
        fourier=0, eval_samples=128, snapshot_samples=16,
        eval_interval=3, snapshot_interval=3, log_interval=3, threads=1,
        output_noise_std=.029, input_noise_std=.5,
        toy100_model="affine_moment_box_v1",
    )
    summary = train(config, tmp_path)
    resolved = summary["config"]
    policy = declared_model_policy(config)
    assert _check_toy100_policy(tmp_path, summary, resolved, policy)
    card = summary["model_policy"]
    assert card["init_data_file"] == "initialization-samples.npy"
    assert card["init_data_samples"] == 16

    altered = deepcopy(summary)
    altered["model_policy"]["init_data_lower"][0] += .1
    with pytest.raises(ValueError, match="sample/formula differs"):
        _check_toy100_policy(tmp_path, altered, resolved, policy)

    path = tmp_path / "initialization-samples.npy"
    original = path.read_bytes()
    samples = np.load(path, allow_pickle=False)
    samples[0, 0] += .1
    np.save(path, samples, allow_pickle=False)
    with pytest.raises(ValueError, match="seeded calibration data differs"):
        _check_toy100_policy(tmp_path, summary, resolved, policy)
    path.write_bytes(original)
    assert _check_toy100_policy(tmp_path, summary, resolved, policy)

    # A self-consistent forged batch used to pass the old checker: the file,
    # its SHA-256, and all four moment fields agree, but its seeded origin does
    # not. The saved initial prior remains the real one.
    forged = deepcopy(summary)
    samples = np.load(path, allow_pickle=False)
    samples[0, 0] += np.float32(.001)
    np.save(path, samples, allow_pickle=False)
    tensor = torch.from_numpy(samples)
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0, unbiased=False)
    half_width = math.sqrt(3.0) * std
    forged["model_policy"].update(
        init_data_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
        init_data_mean=mean.tolist(), init_data_std=std.tolist(),
        init_data_lower=(mean - half_width).tolist(),
        init_data_upper=(mean + half_width).tolist(),
    )
    with pytest.raises(ValueError, match="seeded calibration hash differs"):
        _check_toy100_policy(tmp_path, forged, resolved, policy)
    path.write_bytes(original)

    forged = deepcopy(summary)
    forged["model_policy"]["prior_initial_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seeded affine initialization differs"):
        _check_toy100_policy(tmp_path, forged, resolved, policy)
    forged = deepcopy(summary)
    forged["model_policy"]["prior_initial_min"] += .001
    with pytest.raises(ValueError, match="seeded affine initialization differs"):
        _check_toy100_policy(tmp_path, forged, resolved, policy)


@pytest.mark.parametrize("model", [
    "affine_square_v1", "affine_normal_v1", "affine_square_random_v1",
    "affine_normal_random_v1", "affine_empirical_box_v1",
    "affine_empirical_box_random_v1", "affine_moment_box_v1",
])
def test_all_cpu_affine_archives_replay_initialization(tmp_path, model):
    config = dict(
        model="gan", problem="rotated100", device="cpu", seed=91, steps=2,
        z_dim=2, num_particles=32, batch_size=16, d_hidden=8, n_hidden=1,
        fourier=0, eval_samples=64, snapshot_samples=16,
        eval_interval=2, snapshot_interval=2, log_interval=2, threads=1,
        output_noise_std=(0.0 if model == "affine_normal_v1" else .029),
        input_noise_std=0.0, toy100_model=model,
    )
    directory = tmp_path / model
    summary = train(config, directory)
    assert _check_toy100_policy(
        directory, summary, summary["config"], declared_model_policy(config),
    )
    if "empirical_box" in model:
        forged = deepcopy(summary)
        forged["model_policy"]["init_data_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="seeded calibration hash differs"):
            _check_toy100_policy(
                directory, forged, summary["config"], declared_model_policy(config),
            )
    if "random" in model:
        forged = deepcopy(summary)
        weight = forged["model_policy"]["generator_initial_weight"]
        weight[0][0] *= .99  # still inside the ordinary Linear initialization bound
        forged["model_policy"]["generator_initial_weight_sha256"] = hashlib.sha256(
            struct.pack("<4f", *(x for row in weight for x in row)),
        ).hexdigest()
        with pytest.raises(ValueError, match="seeded affine initialization differs"):
            _check_toy100_policy(
                directory, forged, summary["config"], declared_model_policy(config),
            )


def test_historical_cpu_square_policy_archive_still_regrades():
    directory = Path(__file__).resolve().parents[1] / "reports/toy100/shared22/toy100/grid100"
    summary = json.loads((directory / "summary.json").read_text())
    config = summary["config"]
    assert _check_toy100_policy(
        directory, summary, config, declared_model_policy(config),
    )
