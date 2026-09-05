"""Bounded CPU checks of prior semantics and actual training reproducibility."""

from functools import partial
from importlib import import_module
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import train_arm
from experiments.compare_priors import collect, generate
from lib.particle_prior import FreshGaussianPrior, make_prior
from lib.toy_models import SimpleMLPGenerator, mode_coverage


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_prior_controls_preserve_initialization_and_alias():
    draws, weights = [], []
    for kind in ("particles", "frozen_gaussian", "gaussian", "fresh_gaussian"):
        torch.manual_seed(71)
        prior = make_prior(kind, num_particles=8, z_dim=4)
        draws.append(prior.z.detach().clone())
        weights.append(next(SimpleMLPGenerator().parameters()).detach().clone())
        assert bool(list(prior.parameters())) == (kind == "particles")
    for draw, weight in zip(draws[1:], weights[1:]):
        assert torch.equal(draws[0], draw)
        assert torch.equal(weights[0], weight)


def test_fresh_noise_ignores_reference_buffer_and_has_no_particle_indices():
    prior = FreshGaussianPrior(num_particles=8, z_dim=4)
    reference, _ = prior.sample(8, fixed_first_n=True)
    assert torch.equal(reference, prior.z)
    prior.z.fill_(float("nan"))
    global_state = torch.get_rng_state()
    draw, idx = prior.sample(64, generator=torch.Generator().manual_seed(9))
    repeat, _ = prior.sample(64, generator=torch.Generator().manual_seed(9))
    assert idx is None
    assert torch.isfinite(draw).all()
    assert torch.equal(draw, repeat)
    assert draw.unique(dim=0).shape[0] == 64
    assert torch.equal(torch.get_rng_state(), global_state)


def test_coverage_uses_fresh_samples_and_restores_model_mode():
    class Recorder(torch.nn.Module):
        def forward(self, z):
            self.seen = z.detach().clone()
            return z

    prior = FreshGaussianPrior(num_particles=8, z_dim=2)
    prior.z.fill_(float("nan"))
    model = Recorder().eval()
    mode_coverage(model, prior, torch.device("cpu"), n_eval=64,
                  sample_generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(model.seen).all()
    assert not model.training


@pytest.mark.parametrize("kind", ["particles", "frozen_gaussian", "fresh_gaussian"])
def test_eval_interval_does_not_change_actual_training(tmp_path, kind):
    """Exercise all updates, diagnostics, final sampling and checkpoint writes.

    Use an interpolation penalty as well: its random draws must remain isolated
    from evaluation and from latent draws. Only batch/cloud sizes and plotting
    cost are reduced; losses, models, optimizers and final metrics are real.
    """
    snapshots = []
    for interval in (1, 4):
        out = tmp_path / str(interval)
        cfg = dict(train_arm.DEFAULTS, total_steps=4, eval_interval=interval,
                   prior=kind, arm="e_interp", coeff=0.02,
                   spectral=(kind != "particles"), out_dir=str(out))
        with patch.multiple(train_arm, BATCH_SIZE=32, EVAL_N=64, GRAD_STATS_N=16,
                            FINAL_N=128, NUM_PARTICLES=256), \
                patch.object(train_arm, "save_fake_scatter"), \
                patch.object(train_arm, "mode_coverage", partial(mode_coverage, n_eval=128)):
            summary = train_arm.train(cfg, torch.device("cpu"))
        state = torch.load(out / "ckpt.pt", weights_only=True)
        samples = np.load(out / "final_samples.npy")
        assert summary["final"]["unique_samples"] == np.unique(samples, axis=0).shape[0]
        if kind != "particles":
            assert summary["spectral"]["status"] == "unsupported"
        snapshots.append((state, samples, summary))
    first, second = snapshots
    for module in ("G", "D", "prior", "ema_G", "ema_prior"):
        for name, tensor in first[0][module].items():
            assert torch.equal(tensor, second[0][module][name]), (module, name)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2]["final"]["w1_exact"] == second[2]["final"]["w1_exact"]


def test_examples_share_the_same_training_function():
    particles = import_module("examples.100gaussians")
    gaussian = import_module("examples.100gaussians_no_particle_prior")
    assert gaussian.train.func is particles.train
    assert gaussian.train.keywords == {
        "prior_kind": "fresh_gaussian", "out_dir": "100gaussians_no_particles_samples",
    }


@pytest.mark.parametrize("kind", ["particles", "frozen_gaussian", "fresh_gaussian"])
def test_shared_example_executes_each_prior(tmp_path, kind):
    example = import_module("examples.100gaussians")
    with patch.object(example, "save_fake_scatter"), \
            patch.object(example, "mode_coverage", partial(mode_coverage, n_eval=128)):
        prior, generator, discriminator = example.train(
            epochs=1, steps_per_epoch=2, batch_size=32, num_particles=128,
            prior_kind=kind, out_dir=str(tmp_path), device_str="cpu",
        )
    assert bool(list(prior.parameters())) == (kind == "particles")
    assert all(torch.isfinite(p).all() for p in generator.parameters())
    assert all(torch.isfinite(p).all() for p in discriminator.parameters())


def test_comparison_configs_are_paired_and_recipe_is_fixed(tmp_path):
    import yaml

    manifest = generate(tmp_path, seeds=(23001, 23002), total_steps=7)
    recipes = []
    for run in manifest["runs"]:
        cfg = yaml.safe_load(Path(run["config"]).read_text())
        assert cfg.pop("prior") == run["prior"]
        assert cfg.pop("seed") == run["seed"]
        cfg.pop("out_dir")
        recipes.append(cfg)
    assert len(recipes) == 6
    assert all(recipe == recipes[0] for recipe in recipes)
    assert recipes[0]["arm"] == "b_cap"
    assert recipes[0]["lr"] == 6e-4
    with pytest.raises(ValueError, match="already exists"):
        generate(tmp_path, seeds=(23003,), total_steps=10)
    assert manifest["runtime"]["packages"]["torch"]


def test_comparison_rejects_mismatched_results(tmp_path):
    import yaml

    manifest = generate(tmp_path, seeds=(23001,), total_steps=7)
    for run in manifest["runs"]:
        config = yaml.safe_load(Path(run["config"]).read_text())
        out = Path(run["out_dir"])
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({
            "config": config, "final": {"hq": 0.5}, "collapse_events": 0,
        }))
    assert len(collect(manifest)) == 3
    summary_path = Path(manifest["runs"][0]["out_dir"]) / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["config"]["seed"] = 1234
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="mismatch"):
        collect(manifest)
