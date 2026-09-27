import importlib.util
from pathlib import Path

import pytest
import torch

from particlegan import GANTrainer, get_recipe
from benchmarks.toy_runner import ToyRun, run
from lib.toy_models import sample_100gaussians


def load_example():
    spec = importlib.util.spec_from_file_location("grid_example", Path(__file__).parents[1] / "examples/100gaussians.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small(**overrides):
    return get_recipe(**{"num_particles": 32, "batch_size": 16, "total_steps": 20, **overrides})


@pytest.mark.parametrize("overrides", [{}, {"reg_kappa": 1.25, "reg_coeff": 3., "lr": .00051, "prior_reg": .05}])
def test_runner_matches_public_trainer_updates(overrides):
    """The problem-only example on the shared runner trains bit-for-bit like GANTrainer."""
    torch.set_num_threads(1)
    problem, recipe = load_example().Gaussians100(), small(**overrides)
    toy = ToyRun(problem, recipe=recipe)
    nets = problem.networks(recipe, 0)
    trainer = GANTrainer(recipe, nets.generator, nets.critics, prior=nets.prior, seed=0)
    data = torch.Generator().manual_seed(0)
    for _ in range(recipe.total_steps):
        toy.step()
        trainer.step(sample_100gaussians(16, "cpu", generator=data),
                     generator_real=lambda: sample_100gaussians(16, "cpu", generator=data))
    pairs = [(toy.nets.generator, trainer.G), (toy.nets.critics, trainer.D), (toy.nets.prior, trainer.prior),
             (toy.ema_nets.generator, trainer.ema_G), (toy.ema_nets.prior, trainer.ema_prior)]
    for ours, theirs in pairs:
        for key, tensor in ours.state_dict().items():
            assert torch.equal(tensor, theirs.state_dict()[key]), key


def test_grid_observer_preserves_training():
    torch.set_num_threads(1)
    problem, recipe = load_example().Gaussians100(), small(total_steps=6)
    plain, observed = ToyRun(problem, recipe=recipe), ToyRun(problem, recipe=recipe)
    for _ in range(recipe.total_steps):
        plain.step()
        observed.step()
        torch.randn(100)
        observed.measure()
        observed.measure(ema=True)
    a, b = plain.state_dict(), observed.state_dict()
    for x, y in zip(a["generator_side"] + a["ema"], b["generator_side"] + b["ema"]):
        for key in x:
            assert torch.equal(x[key], y[key]), key


@pytest.mark.parametrize("prior_kind", ["particles", "mog", "frozen_gaussian", "fresh_gaussian"])
def test_every_prior_control_runs_on_the_recipe(prior_kind):
    torch.set_num_threads(1)
    result = run(load_example().Gaussians100(prior_kind, distribution=True), recipe=small(total_steps=4))
    assert result["verdict"] in ("PASS", "FAIL")
    assert {"modes", "hq", "distribution"} <= set(result["ema"])


def test_grid_study_arms_are_recipes():
    from benchmarks.locked_shared.grid_study import ARMS, arm_recipe
    recipes = {name: arm_recipe(overrides) for name, overrides in ARMS}
    stock = recipes["stock"]
    assert (stock.lr, stock.betas, stock.reg_coeff, stock.reg_kappa, stock.prior_reg) == (.0006, (0., .999), 1., 1., 1.)
    assert (stock.batch_size, stock.num_particles, stock.total_steps) == (256, 20_000, 7000)
    assert recipes["penalty_only"].reg_kappa == 1.25 and recipes["toy_transfer"].prior_reg == .05
