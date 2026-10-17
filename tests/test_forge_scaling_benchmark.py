"""Software/measurement controls, not scientific quality or throughput claims."""
import json

import pytest
import torch

from particlegan import GANTrainer, get_recipe
from experiments.forge.contracts import file_hash, stable_hash
from experiments.forge.scaling_benchmark import (DCGANCritic, DCGANGenerator,
                                                recipe_blockers, run_benchmark, validate_spec)
from experiments.forge.state import require_consistent_rng


SPEC = dict(image_size=32, batch_size=2, base_channels=2, max_channels=8,
            latent_dim=4, n_particles=8, warmup_updates=1, measured_updates=2,
            max_seconds=30.)


def recipe(**overrides):
    return get_recipe("bispf", batch_size=2, z_dim=4, num_particles=8,
                      total_steps=3, prior_kind="mog", standardize=False).replace(**overrides)


@pytest.mark.parametrize("delta", [dict(seed=1), dict(image_size=48), dict(measured_updates=0),
                                   dict(max_seconds=float("inf")), dict(profile_updates=1),
                                   dict(prior=dict(kind="mog", sigma=0, standardize=False)),
                                   dict(unrecognized=True)])
def test_bounds_prevent_undeclared_protocols(delta):
    with pytest.raises(ValueError):
        validate_spec({**SPEC, **delta})


def test_fixed_channel_cap_and_resolution_shape():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for size in (32, 64, 128, 256):
            g, d = DCGANGenerator(size, 4, 2, 8), DCGANCritic(size, 2, 8)
            images = g(torch.zeros(2, 4))
            assert images.shape == (2, 3, size, size)
            assert d(images).shape == (2, 1)
            assert max(module.out_channels for model in (g, d) for module in model.modules()
                       if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))) <= 8
    finally:
        torch.set_num_threads(previous)


def test_recipe_preflight_keeps_task_resources_and_normalization_explicit():
    assert not recipe_blockers(recipe(), SPEC)
    assert any("batch_size" in value for value in recipe_blockers(recipe(batch_size=4), SPEC))
    with pytest.raises(ValueError, match="standardize"):
        run_benchmark(recipe(), {**SPEC, "prior": dict(kind="mog", sigma=.1, standardize=True)}, "/tmp/unused")


def test_matched_streams_real_updates_checkpoint_and_media(tmp_path):
    before_rng = torch.get_rng_state().clone()
    before_settings = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    left = run_benchmark(recipe(), SPEC, tmp_path / "left", "cpu")
    right = run_benchmark(recipe(prior_l2=.01), SPEC, tmp_path / "right", "cpu")
    assert left["status"] == right["status"] == "PASS"
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_settings == (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    for key in ("initial_model_sha256", "initial_parameter_sha256", "initialization_sha256", "batch_sequence_sha256"):
        assert left["identity"][key] == right["identity"][key]
    assert left["identity"]["global_recipe_sha256"] != right["identity"]["global_recipe_sha256"]
    assert left["identity"]["task_sha256"] == stable_hash(left["spec"])
    assert left["identity"]["recipe_sha256"] == stable_hash(left["recipe"])
    for item in (left, right):
        assert item["validity"]["changed_generator"] and item["validity"]["changed_discriminator"]
        assert item["metrics"]["measured_updates"] == 2
        assert item["metrics"]["update_ms_p95"] >= item["metrics"]["update_ms_median"] > 0
        assert item["metrics"]["images_per_second"] > 0
        assert item["metrics"]["peak_allocated_mib"] is None
    checkpoint = torch.load(tmp_path / "left/state.pt", weights_only=False)
    require_consistent_rng(checkpoint["context"])
    bindings = checkpoint["context"]["streams"]["manifest"]["bindings"].values()
    assert {(row["family"], row["component"]) for row in bindings} >= {
        ("init", "generator"), ("init", "discriminator"), ("data", "critic"), ("data", "generator"),
        ("noise", "prior"), ("eval", "sampler"), ("eval", "target")}
    trace = [json.loads(line) for line in (tmp_path / "left/updates.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in trace] == ["warmup", "measured", "measured"]
    assert all(row["finite_losses"] for row in trace)
    assert file_hash(tmp_path / "left/updates.jsonl") == left["artifact_sha256"]["trace"]
    from PIL import Image
    with Image.open(tmp_path / "left/training.gif") as image:
        assert image.n_frames == 3
    with pytest.raises(FileExistsError):
        run_benchmark(recipe(), SPEC, tmp_path / "left", "cpu")


def test_noop_optimizer_control_fails_parameter_update_gate(tmp_path, monkeypatch):
    original = GANTrainer.__init__
    def build(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.opt_g.step = lambda: None
        self.opt_d.step = lambda: None
    monkeypatch.setattr(GANTrainer, "__init__", build)
    result = run_benchmark(recipe(), SPEC, tmp_path / "noop", "cpu")
    assert result["status"] == "FAIL"
    assert result["validity"]["finite_model_state"]
    assert result["validity"]["full_budget"]
    assert not result["validity"]["changed_generator"]
    assert not result["validity"]["changed_discriminator"]


def test_nonfinite_loss_control_retains_failure_and_stops(tmp_path, monkeypatch):
    original = GANTrainer.step
    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result["loss_g"] = torch.tensor(float("nan"))
        return result
    monkeypatch.setattr(GANTrainer, "step", step)
    result = run_benchmark(recipe(), SPEC, tmp_path / "nonfinite", "cpu")
    assert result["status"] == "FAIL" and result["benchmark_status"] == "NONFINITE"
    assert result["metrics"]["completed_updates"] == 1
    assert result["validity"]["finite_update_fraction"] == 0
    assert not result["validity"]["full_budget"]
    assert (tmp_path / "nonfinite/state.pt").is_file()


@pytest.mark.parametrize("name", ["e22", "atlas"])
def test_schedule_free_cloud_policy_is_preserved(tmp_path, name):
    spec = {**SPEC, "n_particles": 64, "prior": dict(kind="particle_cloud", sigma=0., standardize=False,
                exception_reason="Explicit finite-table performance workload; no quality claim")}
    configured = get_recipe(name, batch_size=2, z_dim=4, num_particles=64,
                            prior_kind="particles", standardize=False)
    assert configured.total_steps is None and configured.particle_birth_death
    assert not recipe_blockers(configured, spec)
    result = run_benchmark(configured, spec, tmp_path / name, "cpu")
    assert result["status"] == "PASS", result.get("error")
    assert result["recipe"] == configured.to_dict()
    assert result["api"]["recipe"] == configured.to_dict()
    assert result["api"]["policy_lifecycle"]["completed_steps"] == 3
    assert result["api"]["policy_lifecycle"]["external_max_steps"] == 3
    assert result["prior_policy"]["trainable_locations"]
    state = torch.load(tmp_path / name / "state.pt", weights_only=False)
    require_consistent_rng(state["context"])
    assert state["context"]["trainer"]["recipe"]["total_steps"] is None
    assert state["context"]["trainer"]["max_steps"] == 3
