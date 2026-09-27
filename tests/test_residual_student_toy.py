"""The residual student is a problem-only toy on the shared runner."""
import ast
import json
from pathlib import Path

import pytest

from benchmarks.locked_shared import baseline
from benchmarks.locked_shared.hosts import residual_student as rs
from benchmarks.toy_runner import run
from particlegan import get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_residual_student_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/hosts/residual_student.py").read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer",
                "noise_policy", "ParticleRegularizer", "ParticlePrior", "PROTOCOL"} & names
    assert '["lr"]' not in source and "GANLoss" not in source and "GradRegularizer" not in source


def test_residual_student_runs_on_the_recipe_and_is_deterministic():
    problem = rs.ResidualStudent()
    recipe = problem.recipe().replace(total_steps=20)
    first = run(problem, recipe=recipe, observe_every=10)
    again = run(rs.ResidualStudent(), recipe=recipe, observe_every=10)
    assert first["live"] == again["live"] and first["ema"] == again["ema"]
    assert [p["step"] for p in first["curve"]] == [10, 20]
    assert first["recipe"]["lr"] == get_recipe().lr  # shipped optimizer settings, task shape only
    assert first["verdict"] == problem.verdict(first["live"])
    assert int(problem.mask.sum()) == rs.N_IDENTITIES


def test_residual_student_logs_its_named_losses(tmp_path):
    log = tmp_path / "rs.log"
    problem = rs.ResidualStudent()
    run(problem, recipe=problem.recipe().replace(total_steps=4), observe_every=4, log_path=log)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert {"cover", "residual", "loss_d", "loss_g"} <= set(rows[0])
    assert rows[-1]["event"] == "final"


def test_baseline_route_and_noise_policy_refusal():
    raw = rs.train_residual_student(steps=4)
    assert {"identity_mse", "success_rate", "wrong_pad_rate", "live", "hold"} <= set(raw)
    with pytest.raises(ValueError, match="noise from its recipe"):
        baseline.run_toy("residual_student", baseline.Candidate("locked_shared"), noise_policy=object())
