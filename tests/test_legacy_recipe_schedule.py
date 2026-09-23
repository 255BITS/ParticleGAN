"""The custom-host control must use the candidate's LR schedule fields."""

import torch

from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite.compare_defaults import optimizer_defaults
from particlegan import get_recipe, learning_rate_scale


def test_legacy_control_uses_declared_anneal_start_and_floor():
    recipe = get_recipe().replace(lr_anneal_start=0.4, lr_floor=0.01)
    applied = []
    with optimizer_defaults(recipe, applied):
        control = evaluate.FixedControl({"schedule": "cosine"}, 100)
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=0.005)
        control.step(optimizer, 0, role="g")
        control.step(optimizer, 80, role="g")
    expected = learning_rate_scale(80, 100, 0.4, 0.01)
    assert applied[0]["lr"] == recipe.lr
    assert optimizer.param_groups[0]["lr"] == recipe.lr * expected
    assert control.trace == [dict(step=0, role="g", multiplier=1.0),
                             dict(step=80, role="g", multiplier=expected)]
    assert expected != learning_rate_scale(80, 100, 0.6, 0.05)
