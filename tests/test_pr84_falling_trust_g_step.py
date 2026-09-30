"""G step ×0.5 only while G's own-curvature ratio is falling.

Stall reach stays the default. The scale is 0.5 and the window is fixed.
The curvature bound stays .25: this is not the always-on .125 kill.
"""

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import (
    TRUST_FALL_SCALE, TRUST_FALL_WINDOW, ReachRecorder, pr84_reach_candidate,
    scale_adam_displacement, trust_is_falling,
)


def _rho_records(values):
    return [dict(g=dict(rho=value, factor=1.)) for value in values]


def test_falling_is_a_negative_delta_over_the_fixed_window():
    assert TRUST_FALL_WINDOW == 8 and TRUST_FALL_SCALE == .5
    rising = [1, 2, 3, 4, 5, 6, 7, 8]
    flat = [5.] * 8
    falling = [5.0, 4.2, 3.8, 3.1, 2.9, 2.2, 1.9, 0.9]
    assert trust_is_falling(falling)
    assert not trust_is_falling(rising)
    assert not trust_is_falling(flat)
    assert not trust_is_falling(falling[:7])
    assert trust_is_falling(rising + falling)


def test_scale_halves_the_adam_displacement_and_leaves_the_moment_step():
    parameter = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.Adam([parameter], lr=.1)
    parameter.grad = torch.ones(3)
    fresh = torch.nn.Parameter(torch.zeros(3))
    fresh_opt = torch.optim.Adam([fresh], lr=.1)
    fresh.grad = torch.ones(3)
    torch.optim.Adam.step(fresh_opt)

    def ordinary(optimizer, closure=None):
        return torch.optim.Adam.step(optimizer, closure)

    assert scale_adam_displacement(ordinary, lambda optimizer: [parameter], .5)(opt) is None
    assert torch.allclose(parameter, .5 * fresh)
    assert opt.state[parameter]["step"] == fresh_opt.state[fresh]["step"]


def test_recorder_halves_only_a_falling_g_step():
    recorder = ReachRecorder(start_step=0)
    recorder.trust_fall = True
    recorder.phase = 1
    recorder.passthrough = False
    generator = torch.nn.Parameter(torch.tensor([4.]))
    critic = torch.nn.Parameter(torch.tensor([8.]))
    opt_d = torch.optim.Adam([critic], lr=.1)
    opt_g = torch.optim.Adam([generator], lr=.1)
    recorder.optimizers = (opt_d, opt_g)
    recorder.row = {}
    seen = {}

    def super_step(self, optimizer, ordinary_step, closure=None):
        seen["optimizer"] = optimizer
        seen["scale_wrapped"] = ordinary_step is not raw
        ordinary_step(optimizer)
        return "super"

    def raw(optimizer, closure=None):
        for p in optimizer.param_groups[0]["params"]:
            p.add_(2.)
        return "raw"

    recorder.records = _rho_records([5, 4, 3, 2, 2, 2, 2, 1])
    with _patch_super(super_step):
        assert recorder.step(opt_g, raw) == "super"
    assert seen["scale_wrapped"] and generator.item() == 5.
    assert recorder.row["g_step_scale"] == .5 and recorder.trust_fall_halves == 1
    assert recorder.row["g_trust_delta"] < 0

    generator2 = torch.nn.Parameter(torch.tensor([4.]))
    opt_g2 = torch.optim.Adam([generator2], lr=.1)
    recorder.optimizers = (opt_d, opt_g2)
    recorder.records = _rho_records([1, 1, 1, 1, 2, 3, 4, 5])
    with _patch_super(super_step):
        recorder.step(opt_g2, raw)
    assert generator2.item() == 6.
    assert recorder.row["g_step_scale"] == 1. and recorder.trust_fall_halves == 1

    critic2 = torch.nn.Parameter(torch.tensor([8.]))
    opt_d2 = torch.optim.Adam([critic2], lr=.1)
    recorder.optimizers = (opt_d2, opt_g2)
    recorder.records = _rho_records([5, 4, 3, 2, 2, 2, 2, 1])
    with _patch_super(super_step):
        recorder.step(opt_d2, raw)
    assert critic2.item() == 10.

    recorder.records = _rho_records([5, 4, 3])
    parked = torch.nn.Parameter(torch.tensor([1.]))
    opt_park = torch.optim.Adam([parked], lr=.1)
    recorder.optimizers = (opt_d, opt_park)
    with _patch_super(super_step):
        recorder.step(opt_park, raw)
    assert parked.item() == 3. and recorder.row["g_step_scale"] == 1.


def test_trust_fall_off_leaves_the_g_step_and_the_curvature_bound():
    with pr84_reach_candidate(task="mode_hold", ramp="stall") as (recorder, _source):
        assert recorder.trust_fall is False
        assert recorder.curvature_bound == base.G_CURVATURE_BOUND == .25
        assert recorder.ramp == "stall"
    with pr84_reach_candidate(task="mode_hold", ramp="stall", trust_fall=True) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.trust_fall is True
        assert recorder.ramp == "stall" and recorder.game_bound is False
        assert recorder.curvature_bound == .25
        receipt = recorder.receipt()
        assert receipt["trust_fall_scale"] == .5
        assert receipt["trust_fall_window"] == 8
        assert receipt["g_curvature_bound"] == .25
        assert receipt["cpu"]
    assert base.SmoothedBothBoundRecorder is not ReachRecorder


def _patch_super(fn):
    from unittest.mock import patch
    return patch.object(base.SmoothedBothBoundRecorder, "step", fn)
