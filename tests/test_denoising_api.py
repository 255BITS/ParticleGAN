"""Regression contracts for recipes consumed by the conditional research loops."""
import copy
from pathlib import Path

import pytest
import torch

from experiments.config import read_config
from experiments import train_denoising, train_trajectory
from particlegan import ParticleRegularizer, learning_rate_scale
from particlegan.diffusion import DrawSource


@pytest.mark.parametrize("trainer,config", [
    (train_denoising, "configs/denoising/default.toml"),
    (train_denoising, "configs/denoising/joint_ucd/time_class.yaml"),
    (train_denoising, "configs/denoising/budget56k/ddgan_ucd_learned_noise_56k_s24002.yaml"),
    (train_denoising, "configs/denoising/budget56k/ddgan_concat_gaussian_56k_s24002.yaml"),
    (train_trajectory, "configs/trajectory/default.yaml"),
    (train_trajectory, "configs/trajectory/ablations_1k/learned_noise.yaml"),
    (train_trajectory, "configs/trajectory/ablations_1k/one_shot.yaml"),
])
def test_recipe_preserves_optimizer_updates_and_weighted_regularization(trainer, config):
    cfg = {**trainer.DEFAULTS, **read_config(Path(__file__).parents[1] / config)}
    recipe = trainer.training_recipe(cfg)
    g, d = torch.nn.Linear(4, 2), torch.nn.Linear(2, 1)
    prior = DrawSource(cfg["prior"], 8, 4, cfg["seed"], "cpu")
    noise = DrawSource(cfg["noise"], 8, 2, cfg["seed"] + 1, "cpu")
    old_g, old_d, old_prior, old_noise = copy.deepcopy((g, d, prior, noise))
    opt_g, opt_d = recipe.make_optimizers(g, d, prior)
    if noise.kind == "learned":
        opt_g.add_param_group({"params": list(noise.parameters()), "lr": cfg["lr"] * cfg["noise_lr_mult"]})
    groups = [{"params": list(old_g.parameters()), "lr": cfg["lr"]}]
    for source, key in ((old_prior, "prior_lr_mult"), (old_noise, "noise_lr_mult")):
        if source.kind == "learned":
            groups.append({"params": list(source.parameters()), "lr": cfg["lr"] * cfg[key]})
    old_opts = (torch.optim.Adam(groups, betas=(cfg["beta1"], .999)),
                torch.optim.Adam(old_d.parameters(), lr=cfg["lr"] * cfg["d_lr_mult"], betas=(cfg["beta1"], .999)))
    # Compare multiple real Adam updates, including each separately rated source.
    for current, original in zip((opt_g, opt_d), old_opts):
        assert len(current.param_groups) == len(original.param_groups)
        for _ in range(3):
            for optimizer in (current, original):
                optimizer.zero_grad()
                loss = sum(p.square().sum() for group in optimizer.param_groups for p in group["params"])
                loss.backward()
                optimizer.step()
            for new_group, old_group in zip(current.param_groups, original.param_groups):
                assert new_group["lr"] == old_group["lr"]
                for p, q in zip(new_group["params"], old_group["params"]):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
    rows = prior.table[:4].detach().requires_grad_()
    actual = recipe.make_prior_regularizer()(rows)
    expected = cfg["prior_reg"] * ParticleRegularizer()(rows)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, rows)[0],
                               torch.autograd.grad(expected, rows)[0], rtol=0, atol=0)
    if trainer is train_trajectory:
        assert all(learning_rate_scale(step, recipe.total_steps, recipe.lr_anneal_start,
                                       recipe.lr_floor) == 1 for step in (0, 6000, 10000))
