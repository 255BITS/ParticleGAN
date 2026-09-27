"""Sampling and toy100 evaluation include output noise (the sampling law);
clean draws are an explicit diagnostic, and neither touches training."""

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


def test_sample_adds_output_noise_by_default_and_clean_is_opt_in():
    trainer = _trainer()
    trainer.step(_real(0))
    assert output_noise_std(trainer.recipe, trainer.completed_steps) == pytest.approx(.05)
    noisy = trainer.sample(64, generator=torch.Generator().manual_seed(3))
    again = trainer.sample(64, generator=torch.Generator().manual_seed(3))
    assert torch.equal(noisy, again)
    clean = trainer.sample(64, generator=torch.Generator().manual_seed(3), output_noise=False)
    ema_noisy = trainer.sample(64, ema=True, generator=torch.Generator().manual_seed(3))

    # The sampling law: G(latent) + sigma * eps, both from the sampling stream.
    stream = torch.Generator().manual_seed(3)
    with torch.no_grad():
        flag = trainer.prior.training
        trainer.prior.eval()
        latent, _ = trainer.prior.sample(64, generator=stream)
        trainer.prior.train(flag)
        mean = trainer.G(latent)
        old = mean + .05 * torch.randn(mean.shape, generator=stream)
    assert torch.equal(noisy, old)
    assert torch.equal(clean, mean)
    assert torch.equal(noisy, trainer.sample(
        64, generator=torch.Generator().manual_seed(3), output_noise=True))
    assert not torch.equal(noisy, clean)
    assert not torch.equal(ema_noisy, trainer.sample(
        64, ema=True, generator=torch.Generator().manual_seed(3), output_noise=False))
    with pytest.raises(ValueError, match="boolean"):
        trainer.sample(4, output_noise=1)


@pytest.mark.parametrize("mode", ["default", "clean", "noisy"])
def test_sampling_between_steps_leaves_the_training_trajectory_bitwise_unchanged(mode):
    reference, sampled = _trainer(seed=5), _trainer(seed=5)
    for step in range(6):
        options = {} if mode == "default" else {"output_noise": mode == "noisy"}
        sampled.sample(32, **options)
        sampled.sample(16, ema=True, **options)
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
def test_toy100_evaluation_draws_output_noise(monkeypatch, tmp_path, extra):
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
    assert summary["eval_output_noise"] == "noisy"
    # Training draws output noise, and so does evaluation (live and EMA).
    assert any(draws) and not all(draws)
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert {e["step"] for e in events if e.get("event") == "eval"} == {0, 2, 4}


def test_toy100_holdout_is_noisy_and_sample_clean_is_a_diagnostic(tmp_path):
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
    latent = lambda: torch.Generator().manual_seed(
        resolved["seed"] + accuracy_evidence.HOLDOUT_SEED_OFFSETS["latent"])
    with torch.random.fork_rng():
        torch.manual_seed(resolved["seed"] + accuracy_evidence.HOLDOUT_SEED_OFFSETS["noise"])
        expected = trainer.sample(accuracy_evidence.HOLDOUT_N, generator=latent())
    assert np.array_equal(saved["live"], expected.numpy())
    assert not np.array_equal(
        saved["live"], sample_clean(trainer, accuracy_evidence.HOLDOUT_N, generator=latent()).numpy())


def test_toy100_isolated_trainer_accepts_the_output_noise_flag():
    resolved, recipe = resolve_config(_toy_config(output_noise_rng="isolated"))
    trainer = make_trainer(resolved, recipe)
    seed = lambda: torch.Generator().manual_seed(9)
    assert torch.equal(trainer.sample(8, generator=seed()),
                       trainer.sample(8, generator=seed(), output_noise=True))
    with pytest.raises(ValueError, match="boolean"):
        trainer.sample(8, output_noise="yes")


def test_toy100_default_recipe_evaluation_samples_with_recipe_output_noise(monkeypatch, tmp_path):
    sigmas, sampling = [], [0]
    original_sample, original_generate = GANTrainer.sample, GANTrainer._generate

    def sample(self, *args, **kwargs):
        sampling[0] += 1
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            sampling[0] -= 1

    def generate(model, latent, sigma, stream):
        sigmas.append((sampling[0] > 0, sigma))
        return original_generate(model, latent, sigma, stream)

    monkeypatch.setattr(GANTrainer, "sample", sample)
    monkeypatch.setattr(GANTrainer, "_generate", staticmethod(generate))
    config = {key: value for key, value in _toy_config().items() if key != "output_noise_std"}
    summary = train({**config, "recipe_defaults": "particlegan"}, tmp_path / "run")
    assert summary["status"] == "complete"
    assert summary["config"]["recipe_output_noise_std"] == pytest.approx(.029)
    evaluation = [sigma for inside, sigma in sigmas if inside]
    assert evaluation and all(sigma == pytest.approx(.029) for sigma in evaluation)
    assert any(not inside and sigma > 0 for inside, sigma in sigmas)
