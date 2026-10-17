"""Native learnable output-noise behavior and replay guarantees."""

import json

import pytest
import torch
from torch import nn

from benchmarks.toy100.models import LearnableOutputScale, OutputNoise
from benchmarks.toy100.train import (
    _set_output_sigma, make_trainer, resolve_config, train,
)


def _config(**overrides):
    return {
        "steps": 4, "seed": 37, "device": "cpu", "threads": 1,
        "num_particles": 64, "batch_size": 16,
        "g_hidden": 16, "d_hidden": 16, "n_hidden": 1,
        "eval_samples": 1024, "snapshot_samples": 32,
        "eval_interval": 4, "snapshot_interval": 4,
        "early_eval_steps": [0, 4], "log_interval": 4,
        "output_noise_std": .029, "output_noise_warmup": .5,
        "output_noise_learnable": True,
        **overrides,
    }


def test_learnable_scale_validation_and_fixed_noise_schema():
    scale = LearnableOutputScale(.029)
    assert scale.initial_std == .029
    assert float(scale().detach()) == pytest.approx(.029, rel=1e-6)
    for initial in (0, -1, float("inf"), True):
        with pytest.raises(ValueError, match="positive finite"):
            LearnableOutputScale(initial)

    old, _ = resolve_config({"steps": 4})
    assert "output_noise_learnable" not in old
    with pytest.raises(ValueError, match="requires output_noise_std"):
        resolve_config({"output_noise_learnable": True, "output_noise_std": 0})
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_config({"output_noise_learnable": True, "output_noise_std": -1})
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_config({"output_noise_learnable": 1, "output_noise_std": .029})

    fixed = OutputNoise(nn.Identity(), .029)
    assert fixed.output_scale is None
    assert fixed.effective_std() == .029
    assert list(fixed.state_dict()) == []
    fixed.std = 0.0
    before = torch.get_rng_state().clone()
    points = torch.ones(4, 2)
    assert torch.equal(fixed(points), points)
    assert torch.equal(torch.get_rng_state(), before)


def test_scale_gets_generator_gradient_update_ema_and_exact_replay():
    resolved, recipe = resolve_config(_config())
    trainer = make_trainer(resolved, recipe)
    parameter = trainer.G.output_scale.raw_scale
    assert sum(value is parameter for group in trainer.opt_g.param_groups
               for value in group["params"]) == 1
    assert any(value is parameter for value in trainer.opt_g.param_groups[0]["params"])
    assert all(value is not parameter for group in trainer.opt_g.param_groups[1:]
               for value in group["params"])
    assert all(value is not parameter for group in trainer.opt_d.param_groups
               for value in group["params"])
    assert trainer.G.output_scale is not trainer.ema_G.output_scale

    _set_output_sigma(trainer, resolved, 0)
    assert float(trainer.G.effective_std().detach()) == 0.0
    before_rng = torch.get_rng_state().clone()
    trainer.sample(8, generator=torch.Generator().manual_seed(33))
    assert torch.equal(torch.get_rng_state(), before_rng)
    real = torch.randn(16, 2, generator=torch.Generator().manual_seed(81))
    trainer.step(real)
    assert parameter.grad is None  # Warmup starts with zero output noise.
    _set_output_sigma(trainer, resolved, trainer.completed_steps)
    before_update = parameter.detach().clone()
    before_ema = trainer.ema_G.output_scale.raw_scale.detach().clone()
    assert float(trainer.G.effective_std().detach()) == pytest.approx(.0145, rel=1e-5)
    trainer.step(real)
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert float(parameter.grad.abs()) > 0
    assert not torch.equal(parameter.detach(), before_update)
    expected_ema = before_ema * recipe.ema_decay + parameter.detach() * (1 - recipe.ema_decay)
    torch.testing.assert_close(trainer.ema_G.output_scale.raw_scale, expected_ema)
    _set_output_sigma(trainer, resolved, trainer.completed_steps)

    checkpoint = trainer.state_dict()
    assert "output_scale.raw_scale" in checkpoint["models"]["G"]
    assert "output_scale.raw_scale" in checkpoint["models"]["ema_G"]
    before_rng = torch.get_rng_state().clone()
    before_latent = trainer.latent_generator.get_state().clone()
    before_eval = trainer.eval_generator.get_state().clone()
    live = trainer.sample(8, generator=torch.Generator().manual_seed(71))
    ema = trainer.sample(8, ema=True, generator=torch.Generator().manual_seed(71))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.equal(trainer.latent_generator.get_state(), before_latent)
    assert torch.equal(trainer.eval_generator.get_state(), before_eval)
    assert not torch.equal(live, ema)

    trainer.step(real)
    _set_output_sigma(trainer, resolved, trainer.completed_steps)
    uninterrupted = trainer.state_dict()
    restored = make_trainer(resolved, recipe)
    restored.load_state_dict(checkpoint)
    _set_output_sigma(restored, resolved, restored.completed_steps)
    torch.testing.assert_close(
        restored.sample(8, generator=torch.Generator().manual_seed(71)), live,
    )
    torch.testing.assert_close(
        restored.sample(8, ema=True, generator=torch.Generator().manual_seed(71)), ema,
    )
    restored.step(real)
    _set_output_sigma(restored, resolved, restored.completed_steps)
    replayed = restored.state_dict()
    for model in ("G", "D", "prior", "ema_G", "ema_prior"):
        for key in uninterrupted["models"][model]:
            torch.testing.assert_close(
                uninterrupted["models"][model][key], replayed["models"][model][key],
                rtol=0, atol=0,
            )


def test_runner_records_actual_live_and_ema_scales(tmp_path):
    summary = train(_config(), tmp_path / "learnable")
    receipt = summary["learnable_output_noise"]
    assert receipt["added_trainable_parameters"] == 1
    assert receipt["optimizer"] == "G" and receipt["optimizer_group"] == 0
    assert receipt["initial_output_sigma_live"] == 0
    assert receipt["final_output_sigma_live"] > 0
    events = [json.loads(line) for line in (tmp_path / "learnable" / "events.jsonl").read_text().splitlines()]
    initial = [row for row in events if row["event"] == "eval" and row["step"] == 0]
    assert [row["output_sigma"] for row in initial] == [0, 0]
    final = [row for row in events if row["event"] == "eval" and row["step"] == 4]
    assert final[0]["output_sigma"] == receipt["final_output_sigma_live"]
    assert final[1]["output_sigma"] == receipt["final_output_sigma_ema"]
    assert all("output_sigma_live" in row and "output_sigma_ema" in row
               for row in events if row["event"] == "train")
