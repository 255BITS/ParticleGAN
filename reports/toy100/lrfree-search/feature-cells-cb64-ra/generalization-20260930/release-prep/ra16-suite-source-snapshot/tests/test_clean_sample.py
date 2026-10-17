"""Sampling and toy100 evaluation score the clean generator, not training noise."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.training import output_noise_std
from benchmarks.toy100 import accuracy_evidence
from benchmarks.toy100.models import (
    IsolatedOutputNoise, OutputNoise, clean_output_noise, sample_clean,
)
from benchmarks.toy100.train import _set_output_sigma, make_trainer, resolve_config, train


def _trainer(seed=0, steps=8):
    torch.manual_seed(seed)
    recipe = get_recipe(num_particles=12, z_dim=3, batch_size=6, total_steps=steps,
                        output_noise_std=.05, output_noise_warmup=0.0)
    g = nn.Sequential(nn.Linear(3, 8), nn.LeakyReLU(.2), nn.Linear(8, 2))
    d = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1))
    return GANTrainer(recipe, g, d)


def _real(step):
    return torch.randn(6, 2, generator=torch.Generator().manual_seed(100 + step))


def test_sample_is_clean_by_default_and_noisy_path_matches_the_old_formula():
    trainer = _trainer()
    trainer.step(_real(0))
    assert output_noise_std(trainer.recipe, trainer.completed_steps) == pytest.approx(.05)
    clean = trainer.sample(64, generator=torch.Generator().manual_seed(3))
    again = trainer.sample(64, generator=torch.Generator().manual_seed(3))
    assert torch.equal(clean, again)
    ema_clean = trainer.sample(64, ema=True, generator=torch.Generator().manual_seed(3))

    # The previous implementation: G(latent) + sigma * eps, both from the stream.
    stream = torch.Generator().manual_seed(3)
    with torch.no_grad():
        flag = trainer.prior.training
        trainer.prior.eval()
        latent, _ = trainer.prior.sample(64, generator=stream)
        trainer.prior.train(flag)
        mean = trainer.G(latent)
        old = mean + .05 * torch.randn(mean.shape, generator=stream)
    assert torch.equal(clean, mean)
    noisy = trainer.sample(64, generator=torch.Generator().manual_seed(3), output_noise=True)
    assert torch.equal(noisy, old)
    assert not torch.equal(noisy, clean)
    assert not torch.equal(ema_clean, trainer.sample(
        64, ema=True, generator=torch.Generator().manual_seed(3), output_noise=True))
    with pytest.raises(ValueError, match="boolean"):
        trainer.sample(4, output_noise=1)


@pytest.mark.parametrize("mode", ["clean", "noisy"])
def test_sampling_between_steps_leaves_the_training_trajectory_bitwise_unchanged(mode):
    reference, sampled = _trainer(seed=5), _trainer(seed=5)
    for step in range(6):
        sampled.sample(32, output_noise=mode == "noisy")
        sampled.sample(16, ema=True, output_noise=mode == "noisy")
        left, right = reference.step(_real(step)), sampled.step(_real(step))
        for key in ("loss_d", "loss_g", "penalty"):
            assert torch.equal(left[key], right[key])
    a, b = reference.state_dict(), sampled.state_dict()
    for name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for key, tensor in a["models"][name].items():
            assert torch.equal(tensor, b["models"][name][key]), (name, key)
    for name in ("latent_generator", "penalty_generator", "noise_generator"):
        assert torch.equal(a["streams"][name], b["streams"][name])
    assert torch.equal(a["cpu_rng"], b["cpu_rng"])


def _toy_config(**overrides):
    return {
        "steps": 4, "seed": 37, "device": "cpu", "threads": 1,
        "num_particles": 64, "batch_size": 16,
        "g_hidden": 16, "d_hidden": 16, "n_hidden": 1,
        "eval_samples": 1024, "snapshot_samples": 32,
        "eval_interval": 2, "snapshot_interval": 4,
        "early_eval_steps": [0, 4], "log_interval": 4,
        "output_noise_std": .029,
        **overrides,
    }


@pytest.mark.parametrize("extra", [{}, {"output_noise_rng": "isolated"}])
def test_toy100_evaluation_draws_no_output_noise(monkeypatch, tmp_path, extra):
    draws, sampling = [], [0]
    original_sample = GANTrainer.sample

    def sample(self, *args, **kwargs):
        sampling[0] += 1
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            sampling[0] -= 1

    for cls in (OutputNoise, IsolatedOutputNoise):
        def spy(self, prediction, _original=cls._randn):
            draws.append(sampling[0] > 0)
            return _original(self, prediction)
        monkeypatch.setattr(cls, "_randn", spy)
    monkeypatch.setattr(GANTrainer, "sample", sample)
    summary = train(_toy_config(**extra), tmp_path / "run")
    assert summary["status"] == "complete"
    assert summary["eval_output_noise"] == "clean"
    # Training still draws output noise; no draw happens inside evaluation.
    assert draws and not any(draws)
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert {e["step"] for e in events if e.get("event") == "eval"} == {0, 2, 4}


def test_toy100_sample_clean_and_holdout_skip_training_noise(tmp_path):
    resolved, recipe = resolve_config(_toy_config(output_noise_warmup=0.0))
    trainer = make_trainer(resolved, recipe)
    _set_output_sigma(trainer, resolved, 0)
    assert trainer.G.std == pytest.approx(.029)
    seed = lambda: torch.Generator().manual_seed(9)
    clean = sample_clean(trainer, 256, generator=seed())
    noisy = trainer.sample(256, generator=seed())
    with clean_output_noise((trainer.G,)):
        assert trainer.G._clean and not trainer.ema_G._clean
    assert not trainer.G._clean
    trainer.G.std = 0.0
    assert torch.equal(clean, trainer.sample(256, generator=seed()))
    trainer.G.std = .029
    assert not torch.equal(clean, noisy)

    evidence = accuracy_evidence.AccuracyEvidence(
        resolved, tmp_path, [4], torch.zeros(resolved["eval_samples"], 2))
    before = torch.get_rng_state().clone()
    _, _ = evidence.finish(trainer)
    assert torch.equal(torch.get_rng_state(), before)
    saved = np.load(tmp_path / "holdout_samples.npz")
    with clean_output_noise((trainer.G, trainer.ema_G)):
        latent = torch.Generator().manual_seed(
            resolved["seed"] + accuracy_evidence.HOLDOUT_SEED_OFFSETS["latent"])
        expected = trainer.sample(accuracy_evidence.HOLDOUT_N, generator=latent)
    assert np.array_equal(saved["live"], expected.numpy())


def test_toy100_isolated_trainer_accepts_the_output_noise_flag():
    resolved, recipe = resolve_config(_toy_config(output_noise_rng="isolated"))
    trainer = make_trainer(resolved, recipe)
    seed = lambda: torch.Generator().manual_seed(9)
    assert torch.equal(trainer.sample(8, generator=seed()),
                       trainer.sample(8, generator=seed(), output_noise=True))
    with pytest.raises(ValueError, match="boolean"):
        trainer.sample(8, output_noise="yes")
