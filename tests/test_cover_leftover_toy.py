"""cover_leftover as a problem-only toy on the shared runner."""
import ast
from pathlib import Path

import pytest

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.hosts import cover_leftover
from benchmarks.toy_runner import ToyRun, run
from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_cover_leftover_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/hosts/cover_leftover.py").read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer",
                "ParticleRegularizer", "GANLoss", "GradientPenalty", "noise_policy"} & names
    assert '["lr"]' not in source


def test_cover_leftover_roles_come_from_the_recipe():
    toy = ToyRun(cover_leftover.CoverLeftover())
    assert [g["role"] for g in toy.opt_g.param_groups] == ["network", "prior", "prior"]
    out = toy.step()
    assert "cover" in out and "loss_d" in out
    assert toy.recipe.to_dict() == get_recipe(z_dim=4, num_particles=12, batch_size=32, total_steps=800).to_dict()
    assert "cover" not in ToyRun(cover_leftover.CoverLeftover("cover_zero")).step()


def test_cover_leftover_runs_deterministically_and_feeds_the_baseline(monkeypatch):
    problem = cover_leftover.CoverLeftover()
    recipe = problem.recipe().replace(total_steps=20)
    first = run(problem, recipe=recipe, observe_every=10)
    again = run(cover_leftover.CoverLeftover(), recipe=recipe, observe_every=10)
    assert first["live"] == again["live"] and first["ema"] == again["ema"]
    assert first["verdict"] == ("PASS" if first["live"]["pass"] else "FAIL")
    monkeypatch.setitem(baseline.BUDGETS, "cover_leftover", 24)
    monkeypatch.setattr(cover_leftover, "GATE_STEPS", 24)
    result = baseline.run_toy("cover_leftover", baseline.Candidate("probe"))
    assert len(result["observations"]) == 24 and set(result["ema"]) == set(result["live"])
    with pytest.raises(ValueError, match="recipe"):
        baseline.run_toy("cover_leftover", baseline.Candidate("probe"), noise_policy=object())
