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
    original = two_pole.schedule_optimizer
    with pytest.raises(RuntimeError, match="interrupted"):
        with control_host_schedules(FixedSchedule(80, cosine=False)):
            assert two_pole.schedule_optimizer is not original
            raise RuntimeError("interrupted")
    assert two_pole.schedule_optimizer is original


def test_existing_cosine_toy_matches_research_bridge_exactly():
    config = baseline.Candidate("test", reg_coeff=3., reg_kappa=1.25,
                                particle_l2=0., lr_multiplier=.85, lr_schedule="cosine")
    reference = baseline.run_toy("two_pole", config)
    from dataclasses import replace
    with control_host_schedules(FixedSchedule(80, cosine=True)) as timing:
        observed = baseline.run_toy("two_pole", replace(config, lr_schedule="host"))
    assert reference["live"] == observed["live"]
    assert timing["calls"] == 160
    assert timing["seconds"] > 0
