"""Regression contracts for recipes consumed by the conditional research loops."""
import copy
from pathlib import Path

import pytest
import torch

from experiments.config import read_config
from experiments import train_denoising, train_trajectory
from particlegan import ParticleRegularizer, learning_rate_scale
from particlegan.diffusion import DrawSource


@pytest.mark.parametrize("model", ["gan", "ddgan"])
def test_mog_latent_updates_and_checkpoint_sampling(model):
    from lib.denoising_toy import ToyGenerator, ToyDiscriminator, generate
    from particlegan import DDGAN, MoGParticlePrior

    cfg = {**train_denoising.DEFAULTS, "model": model, "prior": "mog",
           "num_particles": 16, "hidden": 16, "depth": 1,
           "prior_betas": [.5, .999]}
    train_denoising.validate(cfg)
    recipe = train_denoising.training_recipe(cfg)
    prior = train_denoising.make_prior(cfg, "cpu")
    assert isinstance(prior, MoGParticlePrior)
    g, d = ToyGenerator(cfg), ToyDiscriminator(cfg)
    opt, _ = recipe.make_optimizers(g, d, prior)
    assert opt.param_groups[1]["betas"] == (.5, .999)
    before = prior.z.detach().clone()
    sigma = prior.sigma.clone()
    rng = torch.Generator().manual_seed(42)
    z, ids = prior.sample(32, rng)
    c = torch.arange(32) % cfg["classes"]
    xt, t = torch.randn(32, 2), torch.full((32,), 2, dtype=torch.long)
    schedule = DDGAN(cfg["alpha_bar"])
    pred = g(z, c, xt, t)
    if model == "ddgan":
        pred = schedule.reverse(pred, xt, t, torch.randn_like(xt))
    loss = pred.square().mean() + recipe.make_prior_regularizer()(prior.z[ids.unique()])
    loss.backward()
    assert torch.isfinite(prior.z.grad).all()
    opt.step()
    assert not torch.equal(prior.z, before)
    torch.testing.assert_close(prior.sigma, sigma, rtol=0, atol=0)
    restored = train_denoising.make_prior(cfg, "cpu")
    restored.load_state_dict(copy.deepcopy(prior.state_dict()))
    noise = DrawSource("gaussian", 8, 2, cfg["seed"], "cpu")

    def sample(source):
        rngs = [torch.Generator().manual_seed(k) for k in range(3)]
        return generate(g, source, noise, schedule, c, *rngs)

    actual = sample(prior)
    assert actual.shape == (32, 2) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, sample(restored), rtol=0, atol=0)


@pytest.mark.parametrize("options", [{"sigma_rel": -.1}, {"standardize": "yes"},
                                    {"prior_betas": [1., .999]}])
def test_denoising_rejects_invalid_mog_settings(options):
    with pytest.raises(ValueError):
        train_denoising.validate({**train_denoising.DEFAULTS, "prior": "mog", **options})


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
