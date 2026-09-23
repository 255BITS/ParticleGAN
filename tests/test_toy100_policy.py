"""Native affine/square model and shared dense-network schedule evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.toy100.config import resolve_problem_config, validate_manifest
from benchmarks.toy100 import __main__ as cli
from benchmarks.toy100.models import InputNoise, OutputNoise
from benchmarks.toy100.schedule import (
    policy_multipliers, policy_rate_action, step_with_policy,
)
from benchmarks.toy100.train import (
    _source_provenance, make_trainer, resolve_config, train, verify_source_archive,
)
from particlegan.recipes import learning_rate_scale


def _small_config(**overrides):
    return {
        "problem": "grid100", "seed": 31, "steps": 6, "device": "cpu",
        "z_dim": 2, "num_particles": 32, "batch_size": 8,
        "d_hidden": 8, "n_hidden": 1, "fourier": 3,
        "eval_samples": 128, "snapshot_samples": 16,
        "eval_interval": 3, "snapshot_interval": 3,
        "log_interval": 3, "threads": 1,
        "toy100_model": "affine_square_v1", "network_lr_horizon_cap": 3,
        **overrides,
    }


def test_policy_fields_are_optional_global_and_validated():
    historical, _ = resolve_config({"steps": 6})
    assert "toy100_model" not in historical
    assert "network_lr_horizon_cap" not in historical
    assert "benchmarks/toy100/schedule.py" not in _source_provenance()["source_sha256"]
    candidate = json.loads((Path(__file__).resolve().parents[1]
                            / "configs/toy100/accuracy_shared_policy.json").read_text())
    assert candidate["toy100_model"] == "affine_square_v1"
    assert candidate["network_lr_horizon_cap"] == 1600
    assert candidate["prior_lr_mult"] == 2.0
    assert candidate["input_noise_anneal_end"] == .1
    assert candidate["fourier"] == 3
    assert all(resolve_problem_config(candidate, problem)["toy100_model"] == "affine_square_v1"
               for problem in ("grid100", "rotated100", "staggered100"))
    manifest = {"steps": 6, "z_dim": 2, "toy100_model": "affine_square_v1",
                "network_lr_horizon_cap": 3,
                "problem_overrides": {"staggered100": {"batch_size": 16}}}
    configs = [resolve_problem_config(manifest, problem) for problem in
               ("grid100", "rotated100", "staggered100")]
    assert all(config["toy100_model"] == "affine_square_v1" for config in configs)
    assert all(config["network_lr_horizon_cap"] == 3 for config in configs)
    for field, value in (("toy100_model", "affine_square_v1"),
                         ("network_lr_horizon_cap", 3)):
        with pytest.raises(ValueError, match="cannot set"):
            validate_manifest({"problem_overrides": {"grid100": {field: value}}})
    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError, match="network_lr_horizon_cap"):
            resolve_config({"network_lr_horizon_cap": bad})
    with pytest.raises(ValueError, match="z_dim=2"):
        resolve_config({"toy100_model": "affine_square_v1"})
    with pytest.raises(ValueError, match="toy100_model"):
        resolve_config({"toy100_model": "unlisted"})


def test_affine_square_initialization_matches_scratch_rng_order():
    config, recipe = resolve_config(_small_config(output_noise_std=.029,
                                                   input_noise_std=.5))
    trainer = make_trainer(config, recipe)
    assert isinstance(trainer.G, OutputNoise)
    assert isinstance(trainer.G.model, torch.nn.Linear)
    assert isinstance(trainer.D, InputNoise)
    np.testing.assert_array_equal(trainer.G.model.weight.detach().numpy(), np.eye(2))
    np.testing.assert_array_equal(trainer.G.model.bias.detach().numpy(), np.zeros(2))
    assert trainer.prior.z.shape == (32, 2)
    assert torch.all(trainer.prior.z >= -5) and torch.all(trainer.prior.z <= 5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        reference = recipe.make_prior(learnable=True)
        with torch.no_grad():
            reference.z.uniform_(-5.0, 5.0)
        torch.nn.Linear(2, 2)  # The scratch probe consumes its default init.
    torch.testing.assert_close(trainer.prior.z, reference.z, rtol=0, atol=0)
    assert sum(p.numel() for p in trainer.G.parameters()) == 6


def test_policy_rates_keep_prior_on_full_budget_and_restore_checkpoint_bases():
    config, recipe = resolve_config(_small_config())
    torch.set_num_threads(1)
    trainer = make_trainer(config, recipe)
    bases = [list(row) for row in trainer.initial_lrs]
    real = torch.randn(8, 2)
    for completed in range(4):
        step_with_policy(trainer, real, network_lr_horizon_cap=3,
                         generator_real=lambda: real)
        assert trainer.initial_lrs == bases
        receipt = policy_rate_action(trainer, completed + 1,
                                     network_lr_horizon_cap=3)
        network, prior = policy_multipliers(completed, 6, .6, .05, 3)
        assert receipt["lr_g"] == pytest.approx(bases[0][0] * network)
        assert receipt["lr_prior"] == pytest.approx(bases[0][1] * prior)
        assert receipt["lr_d"] == pytest.approx(bases[1][0] * network)
        assert trainer.state_dict()["initial_lrs"] == bases
    assert policy_multipliers(4, 6, .6, .05, 3)[0] == .05
    assert policy_multipliers(4, 6, .6, .05, 3)[1] > .05
    assert policy_multipliers(4, 6, .6, .05, 6) == (
        learning_rate_scale(4, 6, .6, .05),
        learning_rate_scale(4, 6, .6, .05),
    )
    assert policy_multipliers(4, 6, .6, 0.0, 3)[0] == 0.0
    assert policy_multipliers(4, 6, .6, 0.0, 3)[1] > 0.0


def test_noop_horizon_and_checkpoint_replay_are_exact(monkeypatch):
    config, recipe = resolve_config(_small_config())
    torch.set_num_threads(1)
    real = torch.randn(8, 2)
    first = make_trainer(config, recipe)
    second = make_trainer(config, recipe)
    rng = torch.get_rng_state()
    ordinary = first.step(real, generator_real=lambda: real)
    torch.set_rng_state(rng)
    capped_noop = step_with_policy(second, real, network_lr_horizon_cap=6,
                                   generator_real=lambda: real)
    for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
        torch.testing.assert_close(ordinary[key], capped_noop[key], rtol=0, atol=0)
    for left, right in zip(first.G.parameters(), second.G.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    checkpoint = second.state_dict()
    step_with_policy(second, real, network_lr_horizon_cap=3,
                     generator_real=lambda: real)
    replay = make_trainer(config, recipe)
    replay.load_state_dict(checkpoint)
    step_with_policy(replay, real, network_lr_horizon_cap=3,
                     generator_real=lambda: real)
    for name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for left, right in zip(getattr(second, name).parameters(),
                               getattr(replay, name).parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert second.initial_lrs == replay.initial_lrs == checkpoint["initial_lrs"]

    def broken_step(*args, **kwargs):
        assert second.opt_g._optimizer_step_pre_hooks
        assert second.opt_d._optimizer_step_pre_hooks
        raise RuntimeError("injected failure")

    monkeypatch.setattr(second, "step", broken_step)
    with pytest.raises(RuntimeError, match="injected failure"):
        step_with_policy(second, real, network_lr_horizon_cap=3)
    assert second.initial_lrs == checkpoint["initial_lrs"]
    assert not second.opt_g._optimizer_step_pre_hooks
    assert not second.opt_d._optimizer_step_pre_hooks


def test_policy_update_matches_original_scratch_hook_order(monkeypatch):
    """The role-specific production hooks reproduce the archived probe math."""
    from particlegan import training as training_module

    config, recipe = resolve_config(_small_config(output_noise_std=.029,
                                                   input_noise_std=.5))
    torch.set_num_threads(1)
    real = torch.randn(8, 2)
    reference = make_trainer(config, recipe)
    production = make_trainer(config, recipe)
    reference.D.sigma = production.D.sigma = .5
    ordinary_scale = training_module.learning_rate_scale
    reference_g_step = reference.opt_g.step

    def reset_prior_before_g(*args, **kwargs):
        full = ordinary_scale(reference.completed_steps, recipe.total_steps,
                              recipe.lr_anneal_start, recipe.lr_floor)
        reference.opt_g.param_groups[1]["lr"] = reference.initial_lrs[0][1] * full
        return reference_g_step(*args, **kwargs)

    reference.opt_g.step = reset_prior_before_g
    for _ in range(4):
        rng = torch.get_rng_state()
        with monkeypatch.context() as patch:
            patch.setattr(training_module, "learning_rate_scale",
                          lambda step, total, start, floor: ordinary_scale(
                              step, min(total, 3), start, floor,
                          ))
            old = reference.step(real, generator_real=lambda: real)
        torch.set_rng_state(rng)
        new = step_with_policy(production, real, network_lr_horizon_cap=3,
                               generator_real=lambda: real)
        for key in ("loss_d", "loss_g", "loss_gan", "prior_regularization", "penalty"):
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        for name in ("G", "D", "prior", "ema_G", "ema_prior"):
            for left, right in zip(getattr(reference, name).parameters(),
                                   getattr(production, name).parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert [group["lr"] for group in reference.opt_g.param_groups] == [
            group["lr"] for group in production.opt_g.param_groups
        ]
        assert reference.opt_d.param_groups[0]["lr"] == production.opt_d.param_groups[0]["lr"]


def test_policy_run_archives_source_and_logs_actual_rates(tmp_path):
    run_dir = tmp_path / "policy"
    summary = train(_small_config(output_noise_std=.029,
                                  output_noise_warmup=.2,
                                  input_noise_std=.5), run_dir)
    assert summary["status"] == "complete"
    model = summary["model_policy"]
    assert model["toy100_model"] == "affine_square_v1"
    assert model["generator_class"] == "Linear"
    assert model["generator_base_parameters"] == 6
    assert model["prior_initialization"] == "uniform_square"
    assert model["prior_shape"] == [32, 2]
    assert model["prior_initial_min"] >= -5 and model["prior_initial_max"] <= 5
    assert model["generator_initial_weight"] == [[1.0, 0.0], [0.0, 1.0]]
    assert model["generator_initial_bias"] == [0.0, 0.0]
    provenance = summary["provenance"]
    assert provenance == json.loads((run_dir / "provenance.json").read_text())
    for name in ("benchmarks/toy100/schedule.py", "benchmarks/toy100/config.py",
                 "benchmarks/toy100/__main__.py"):
        assert name in provenance["source_sha256"]
    verify_source_archive(run_dir, provenance)
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "source.tar.gz").write_bytes((run_dir / "source.tar.gz").read_bytes())
    verify_source_archive(moved, provenance)
    corrupted = bytearray((moved / "source.tar.gz").read_bytes())
    corrupted[-1] ^= 1
    (moved / "source.tar.gz").write_bytes(corrupted)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_source_archive(moved, provenance)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    actions = [row for row in events if row["event"] == "train"]
    assert len(actions) == 6
    assert all("lr_g" in row and "lr_prior" in row and "lr_d" in row for row in actions)
    assert actions[-1]["network_multiplier"] == .05
    assert actions[-1]["prior_multiplier"] > .05
    assert hashlib.sha256((run_dir / "source.tar.gz").read_bytes()).hexdigest() == (
        provenance["source_archive_sha256"]
    )


def test_cli_rejects_policy_source_mismatch_even_when_a_fake_gate_passes(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    manifest = _small_config()
    manifest.pop("problem")
    config_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "train", lambda config, folder: {
        "status": "complete", "provenance": {"source_sha256": {"wrong.py": "0" * 64}},
    })
    monkeypatch.setattr(cli, "evaluate_suite", lambda *args, **kwargs: {
        "status": "PASS", "passed_problems": 3, "required_problems": 3,
    })
    args = SimpleNamespace(config=config_path, output=tmp_path / "run", problem=None,
                           steps=None, device=None, no_render=True,
                           require_accuracy=False)
    assert cli._run(args) == 1
    declaration = json.loads((args.output / "run_manifest.json").read_text())
    assert declaration["policy_source_sha256"] == _source_provenance(
        include_policy=True)["source_sha256"]
