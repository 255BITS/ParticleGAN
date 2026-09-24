import pytest
import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import ReachRecorder, pr84_reach_candidate, reach_width


@pytest.mark.parametrize("sharpness", [.01, .1, .15, .3])
def test_reach_is_pr84_width_below_slope_utilisation_threshold(sharpness):
    assert reach_width(sharpness) == min(base.SMOOTH_WIDTH_CAP, .5 / sharpness)


def test_reach_peaks_at_the_b_cap_slope():
    assert reach_width(1.) == pytest.approx(.5)
    assert reach_width(.8) == pytest.approx(reach_width(1.25))
    assert reach_width(.6) < reach_width(.9) < reach_width(1.)
    assert reach_width(1., reach=1.) == pytest.approx(1.)


def test_stall_needs_saturated_d_and_collapsed_g_trust():
    with pr84_reach_candidate(task="mode_hold", ramp="stall") as (recorder, _source):
        assert not recorder._stalled(1.)
        recorder.records = [dict(g=dict(factor=.05))] * 60
        assert recorder._stalled(.6) and not recorder._stalled(.59)
        recorder.records = [dict(g=dict(factor=.2))] * 60
        assert not recorder._stalled(1.)


def test_factory_installs_and_restores_reach_recorder():
    with pr84_reach_candidate(task="mode_hold", reach=.7) as (recorder, _source):
        assert isinstance(recorder, ReachRecorder)
        assert recorder.reach == .7
        assert recorder.sustained_g_scale is None
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)


def test_sustained_arm_waits_for_two_hundred_full_ring_updates_then_halves_g():
    """199 full-ring updates match stall reach. The 200th arms and halves G+prior only."""
    ordinary = torch.optim.Adam.step

    def run(scale, script):
        torch.manual_seed(0)
        g = torch.nn.Parameter(torch.tensor([1., -0.5]))
        prior = torch.nn.Parameter(torch.tensor([0.25]))
        d = torch.nn.Parameter(torch.tensor([2.]))
        opt_g = torch.optim.Adam([g, prior], lr=.1, betas=(0., .9))
        opt_d = torch.optim.Adam([d], lr=.1, betas=(0., .9))
        recorder = ReachRecorder(start_step=0)
        recorder.curvature_bound = 1e6
        recorder.d_curvature_bound = 1e6
        recorder.sustained_g_scale = scale
        planned = list(script)
        recorder._ring_coverage = lambda local: planned.pop(0)
        for step in range(len(script)):
            for _phase in recorder.phases(step, opt_d, opt_g, {"means": torch.zeros(8, 2)}):
                d.grad = torch.ones_like(d)
                recorder.step(opt_d, ordinary)
                g.grad = torch.tensor([1., -1.])
                prior.grad = torch.tensor([0.4])
                recorder.step(opt_g, ordinary)
        return recorder, g.detach().clone(), prior.detach().clone(), d.detach().clone()

    full = [(8, 1.)] * 200
    plain = run(None, full)
    held = run(.5, full)
    assert not plain[0].armed and plain[0].post_arm_g_steps == 0
    assert held[0].armed and held[0].arm_update == 199 and held[0].post_arm_g_steps == 1
    assert torch.equal(held[3], plain[3])
    assert not torch.equal(held[1], plain[1])
    assert held[0].records[198]["g_half"] is False and held[0].records[199]["g_half"] is True
    assert held[0].records[198]["hold"] == 199 and held[0].records[199]["hold"] == 200

    broken = [(8, 1.)] * 50 + [(8, .899)] + [(7, 1.)] + [(8, .9)] * 199
    reset = run(.5, broken)
    assert not reset[0].armed and reset[0]._hold == 199 and reset[0].post_arm_g_steps == 0
    assert reset[0]._max_hold == 199


def test_sustained_arm_keeps_halving_after_the_ring_drops_and_spares_d():
    ordinary = torch.optim.Adam.step
    g = torch.nn.Parameter(torch.tensor([1.]))
    d = torch.nn.Parameter(torch.tensor([2.]))
    opt_g = torch.optim.Adam([g], lr=.1, betas=(0., .9))
    opt_d = torch.optim.Adam([d], lr=.1, betas=(0., .9))
    recorder = ReachRecorder(start_step=0)
    recorder.curvature_bound = 1e6
    recorder.sustained_g_scale = .5
    recorder._hold = 199
    script = [(8, .9), (0, 0.)]
    recorder._ring_coverage = lambda local: script.pop(0)

    def update(step):
        before_g, before_d = g.detach().clone(), d.detach().clone()
        for _phase in recorder.phases(step, opt_d, opt_g, {"means": torch.zeros(8, 2)}):
            d.grad = torch.ones_like(d)
            recorder.step(opt_d, ordinary)
            g.grad = torch.ones_like(g)
            recorder.step(opt_g, ordinary)
        return before_g, before_d

    before_g, before_d = update(0)
    assert recorder.armed and recorder.post_arm_g_steps == 1
    armed_g, armed_d = g.detach().clone(), d.detach().clone()
    # A second update still halves G after the ring collapses, and D keeps moving.
    update(1)
    assert recorder.armed and recorder.post_arm_g_steps == 2
    assert not torch.equal(g.detach(), armed_g)
    assert not torch.equal(d.detach(), armed_d)
    assert opt_g.state[g]["step"] == opt_d.state[d]["step"] == 2
    full = ReachRecorder(start_step=0)
    full.curvature_bound = 1e6
    g2 = torch.nn.Parameter(before_g.clone())
    d2 = torch.nn.Parameter(before_d.clone())
    opt_g2 = torch.optim.Adam([g2], lr=.1, betas=(0., .9))
    opt_d2 = torch.optim.Adam([d2], lr=.1, betas=(0., .9))
    for _phase in full.phases(0, opt_d2, opt_g2, {}):
        d2.grad = torch.ones_like(d2)
        full.step(opt_d2, ordinary)
        g2.grad = torch.ones_like(g2)
        full.step(opt_g2, ordinary)
    assert torch.allclose(armed_g, before_g + .5 * (g2.detach() - before_g))
    assert torch.equal(armed_d, d2.detach())


def test_sustained_arm_before_the_hold_matches_stall_reach():
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    def run(scale):
        torch.manual_seed(0)
        policy = NoisePolicy(.029, .5, .1, 1200)
        with pr84_reach_candidate(task="mode_hold", ramp="stall", sustained_g_scale=scale) as (recorder, _):
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2), seed=0,
                                               noise_policy=policy)
        return result, recorder.armed, recorder.post_arm_g_steps, torch.get_rng_state().clone()

    plain, held = run(None), run(.5)
    assert plain[0] == held[0]
    assert not held[1] and held[2] == 0
    assert torch.equal(plain[3], held[3])
