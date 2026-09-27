"""mid_scale_identity is a problem-only toy on the shared runner."""

import ast
from pathlib import Path

import pytest

from particlegan import get_recipe
from benchmarks.locked_shared.hosts import mid_scale_identity as msi
from benchmarks.locked_shared.observation import recording
from benchmarks.toy_runner import run

ROOT = Path(__file__).resolve().parents[1]


def test_mid_scale_identity_is_problem_only():
    source = (ROOT / "benchmarks/locked_shared/hosts/mid_scale_identity.py").read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer", "noise_policy.input",
                "FORMULATION", "delayed_cosine"} & names
    assert '["lr"]' not in source and "GANLoss" not in source and "GradientPenalty" not in source


def test_mid_scale_identity_runs_on_the_recipe_and_is_deterministic():
    problem = msi.MidScaleIdentity()
    recipe = problem.recipe().replace(total_steps=20)
    first = run(problem, recipe=recipe, observe_every=10)
    again = run(msi.MidScaleIdentity(), recipe=recipe, observe_every=10)
    assert first["live"] == again["live"] and first["ema"] == again["ema"]
    assert [p["step"] for p in first["curve"]] == [10, 20]
    assert first["recipe"]["lr"] == get_recipe().lr  # shipped optimizer settings, task shape only
    assert first["verdict"] == ("PASS" if first["live"]["pass"] else "FAIL")


def test_run_arm_feeds_recorders_and_rejects_legacy_noise():
    with recording(4) as recorder:
        row = msi.run_arm("locked", steps=4)
    assert recorder.curve and recorder.curve[-1]["step"] == 4 and {"arm", "ema", "hold", "identity_at_mid"} <= set(row)
    with pytest.raises(ValueError, match="recipe"):
        msi.run_arm("locked", steps=2, noise_policy=object())


def test_missing_minus_grid_cannot_pass():
    student = msi.MidScaleResidual()
    row = msi.score_hold(student, scales=msi.ANIMA_SMILE_SCALES)
    assert not row["pass"] and "missing_minus" in row["fail_reasons"]
