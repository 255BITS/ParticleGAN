"""The two-pole cloud is a problem-only toy on the shared runner."""
import ast
from pathlib import Path

import pytest
import torch

from benchmarks.locked_shared import two_pole
from benchmarks.toy_runner import ToyRun, run
from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_two_pole_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/two_pole.py").read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer",
                "manual_seed", "torch.manual_seed"} & names
    for banned in ('["lr"]', "GANLoss", "GradientPenalty", "make_gan_loss", "make_b_cap", "LOCKED_SHARED",
                   "noise_policy"):
        assert banned not in source


def test_two_pole_optimizers_come_from_the_recipe():
    toy = ToyRun(two_pole.TwoPole())
    shipped = get_recipe()
    assert toy.recipe.lr == shipped.lr and toy.recipe.betas == shipped.betas
    assert toy.recipe.batch_size == toy.recipe.num_particles == two_pole.N_PARTICLES
    assert toy.opt_g.direct_response is not None  # the particle table is the direct-particle group
    assert [p for g in toy.opt_g.param_groups for p in g["params"]] == [toy.nets.generator.particles]
    out = toy.step()
    assert "particle_l2" in out and toy.opt_g.completed_steps == 1


def test_two_pole_runs_and_is_deterministic():
    recipe = two_pole.TwoPole().recipe().replace(total_steps=12)
    first = run(two_pole.TwoPole(), recipe=recipe, observe_every=4)
    again = run(two_pole.TwoPole(), recipe=recipe, observe_every=4)
    assert first["live"] == again["live"] and [p["step"] for p in first["curve"]] == [4, 8, 12]
    assert first["verdict"] == two_pole.TwoPole().verdict(first["live"])
    assert {"mean_abs", "grad_med", "nearest", "cover_score"} <= set(first["live"])


def test_stranger_arm_only_feels_the_l2_pull():
    stranger = ToyRun(two_pole.TwoPole(pairing="stranger"))
    with torch.no_grad():
        stranger.nets.generator.particles.fill_(0.5)
    stranger.step()
    # The critic never sees the particles, so the pull toward the origin is all they feel.
    assert bool((stranger.nets.generator.particles < 0.5).all())
    with pytest.raises(ValueError):
        two_pole.TwoPole(pairing="nearest")
