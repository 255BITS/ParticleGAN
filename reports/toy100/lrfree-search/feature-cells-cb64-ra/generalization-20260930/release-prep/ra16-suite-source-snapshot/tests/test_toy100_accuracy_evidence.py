"""Holdout generation must be separate from both optimizer and preview streams."""

import numpy as np
import torch

from benchmarks.toy100.accuracy import evaluate_accuracy
from benchmarks.toy100.accuracy_evidence import AccuracyEvidence
from benchmarks.toy100.accuracy_gate import HOLDOUT_N
from benchmarks.toy100.metrics import evaluate_samples
from benchmarks.toy100.problems import sample_real
from benchmarks.toy100.train import make_trainer, resolve_config


def test_terminal_and_holdout_samples_are_replayable_and_preserve_training_rng(tmp_path):
    config, recipe = resolve_config({
        "steps": 1000, "seed": 23, "device": "cpu", "num_particles": 64,
        "batch_size": 8, "g_hidden": 8, "d_hidden": 8, "n_hidden": 1,
        "output_noise_std": .029, "input_noise_std": .5,
    })
    trainer = make_trainer(config, recipe)
    target = sample_real("grid100", 20_000, generator=torch.Generator().manual_seed(24))
    recorder = AccuracyEvidence(config, tmp_path, [0, 100, 250, 500, 750, 1000], target)
    cloud = sample_real("grid100", 20_000, generator=torch.Generator().manual_seed(25))
    coverage = evaluate_samples(cloud, "grid100")
    expected = evaluate_accuracy(cloud, "grid100", gate_metrics=coverage)
    for step in (100, 250, 500, 750, 1000):
        for model in ("live", "ema"):
            assert recorder.observe(step, model, cloud, coverage) == expected
    with np.load(tmp_path / "quality_checks/step_001000.npz") as saved:
        np.testing.assert_array_equal(saved["live"], cloud.numpy())
        np.testing.assert_array_equal(saved["target"], target.numpy())

    global_rng = torch.random.get_rng_state().clone()
    d_rng = trainer.D.noise_stream.get_state().clone()
    training_latent_rng = trainer.latent_generator.get_state().clone()
    metadata, holdout = recorder.finish(trainer)
    assert metadata["check_steps"] == [100, 250, 500, 750, 1000]
    assert metadata["holdout_samples"] == HOLDOUT_N
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert torch.equal(d_rng, trainer.D.noise_stream.get_state())
    assert torch.equal(training_latent_rng, trainer.latent_generator.get_state())
    with np.load(tmp_path / "holdout_samples.npz") as saved:
        original = {key: saved[key].copy() for key in saved.files}
        for key in ("live", "ema", "target"):
            assert saved[key].shape == (HOLDOUT_N, 2)
            assert evaluate_accuracy(saved[key], "grid100") == holdout[key]
        assert not np.array_equal(saved["target"][:20_000], target.numpy())
    metadata_again, holdout_again = recorder.finish(trainer)
    assert metadata_again == metadata and holdout_again == holdout
    with np.load(tmp_path / "holdout_samples.npz") as saved:
        for key, points in original.items():
            np.testing.assert_array_equal(points, saved[key])
