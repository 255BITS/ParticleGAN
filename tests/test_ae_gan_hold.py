"""ae_gan_hold is a problem-only toy on the shared runner."""

import ast
from pathlib import Path

import pytest

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.hosts import ae_gan_hold
from benchmarks.toy_runner import run
from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_ae_gan_hold_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/hosts/ae_gan_hold.py").read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer",
                "make_optimizers", "make_gradient_penalty", "requires_grad_", "set_rng_state"} & names
    assert '["lr"]' not in source and "gan_v3" not in source and "legacy" not in source


def test_ae_gan_hold_runs_on_the_shipped_recipe_and_is_deterministic():
    problem = ae_gan_hold.AEGanHold()
    recipe = problem.recipe()
    shipped = get_recipe("ae_gan")
    changed = {k for k, v in recipe.to_dict().items() if v != shipped.to_dict()[k]}
    assert changed <= {"z_dim", "num_particles", "batch_size", "total_steps"}
    assert recipe.total_steps == ae_gan_hold.STEPS  # the schedule's horizon is the host budget
    short = recipe.replace(total_steps=20)
    first = run(problem, recipe=short, observe_every=10)
    again = run(ae_gan_hold.AEGanHold(), recipe=short, observe_every=10)
    assert first["live"] == again["live"] and first["ema"] == again["ema"]
    assert {"recon_mse", "hold", "verdict"} <= set(first["live"])
    assert first["verdict"] == ae_gan_hold.verdict(first["live"])


def test_baseline_routes_ae_gan_hold_through_the_runner(monkeypatch):
    monkeypatch.setitem(baseline.BUDGETS, "ae_gan_hold", 24)
    monkeypatch.setattr(ae_gan_hold.AEGanHold, "recipe",
                        lambda self: get_recipe("ae_gan", num_particles=12, batch_size=16, total_steps=24))
    result = baseline.run_toy("ae_gan_hold", baseline.Candidate("probe", particle_l2=0.0))
    assert set(result["live"]) == {"recon_mse", "hold"} and set(result["ema"]) == {"recon_mse", "hold"}
    assert len(result["observations"]) == 24
    with pytest.raises(ValueError, match="noise from its recipe"):
        baseline.run_toy("ae_gan_hold", baseline.Candidate("probe"), noise_policy=object())
