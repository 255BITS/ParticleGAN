"""The unipolar toy is problem-only on the shared runner (benchmarks.toy_runner)."""
import ast
from pathlib import Path

import pytest

from benchmarks.locked_shared.hosts import unipolar
from benchmarks.toy_runner import ToyRun, run
from particlegan import get_recipe

SOURCE = Path(unipolar.__file__)


def test_unipolar_is_problem_only():
    source = SOURCE.read_text()
    names = {ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Attribute, ast.Name))}
    assert not {"torch.optim", "torch.optim.Adam", "learning_rate_scale", "schedule_optimizer"} & names
    for banned in ('["lr"]', "initial_lr", "GANLoss", "GradientPenalty", "noise_policy"):
        assert banned not in source


@pytest.mark.parametrize("arm", unipolar.ARMS)
def test_every_arm_trains_on_recipe_optimizers(arm):
    toy = ToyRun(unipolar.Unipolar(arm))
    assert toy.recipe.lr == get_recipe().lr and toy.recipe.batch_size == unipolar.N_ROWS
    assert type(toy.opt_g).__name__ == "K3PGeneratorAdam"
    assert set(toy.opt_d) == (set() if arm == "mse_only" else {"critic"})
    before = [p.detach().clone() for p in toy.nets.generator.parameters()]
    losses = toy.step()
    assert ("mse" in losses) == (arm == "mse_only")
    assert any((a != b).any() for a, b in zip(before, toy.nets.generator.parameters()))
    assert toy.opt_g.completed_steps == 1


def test_run_arm_is_deterministic_and_reports_live_and_ema():
    recipe = unipolar.Unipolar().recipe().replace(total_steps=20)
    first, again = unipolar.run_arm(recipe=recipe), unipolar.run_arm(recipe=recipe)
    assert first["live"] == again["live"] and first["hold"] == again["hold"]
    assert {k: v for k, v in first.items() if k != "live_curve"} == {k: v for k, v in again.items() if k != "live_curve"}
    assert {"cover", "off_caption", "neu_hold", "live", "hold", "live_curve"} <= set(first)
    assert first["live"]["verdict"] in ("PASS", "FAIL")


def test_gates_read_the_true_plus_pole():
    result = run(unipolar.Unipolar("polarity_flipped"), recipe=get_recipe(batch_size=8, total_steps=5))
    assert result["live"]["cos_plus"] <= 0.0
