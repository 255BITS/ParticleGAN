import torch
import pytest

from benchmarks.learned_lr_evaluation import FixedSchedule, control_host_schedules, optimizer_role
from benchmarks.locked_shared import baseline, two_pole
from particlegan import learning_rate_scale


def test_fixed_schedule_replaces_rates_and_preserves_group_ratios():
    first, second = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.Adam([{"params": [first], "lr": .1}, {"params": [second], "lr": .3}])
    controller = FixedSchedule(100, cosine=True)
    controller.step(optimizer, 0, "g")
    for step in (60, 80, 99):
        for group in optimizer.param_groups:
            group["lr"] = 999.
        controller.step(optimizer, step, "g")
        assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(
            [base * learning_rate_scale(step, 100, .6, .05) for base in (.1, .3)])


def test_phase_bridge_rejects_unknown_optimizer_and_restores_callbacks():
    optimizer = object()
    for key in ("opt_g", "opt_p", "opt"):
        assert optimizer_role(optimizer, {key: optimizer}) == "g"
    assert optimizer_role(optimizer, {"opt_d": optimizer}) == "d"
    with pytest.raises(RuntimeError, match="no declared"):
        optimizer_role(optimizer, {})


def test_host_schedule_bridge_is_inert():
    # Hosts no longer expose schedule_optimizer: a controller cannot rewrite
    # their rates from outside any more.
    assert not hasattr(two_pole, "schedule_optimizer")
    with control_host_schedules(FixedSchedule(80, cosine=True)) as timing:
        pass
    assert timing == {"seconds": 0., "calls": 0}
