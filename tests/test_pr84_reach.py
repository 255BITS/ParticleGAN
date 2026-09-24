import pytest

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
        assert recorder.post_acquire_g_scale is None
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)


def test_post_acquire_arms_once_on_full_ring_and_labels_the_window():
    from benchmarks.locked_shared.observation import Recorder, ring_listener

    with pr84_reach_candidate(task="mode_hold", ramp="stall", post_acquire_g_scale=.5) as (recorder, _source):
        assert ring_listener() == recorder.note_ring
        assert recorder.post_acquire_g_scale == .5 and not recorder.armed
        recorder.note_ring(50, 7, 1.)
        recorder.note_ring(100, 8, .89)
        assert not recorder.armed
        Recorder(1200).record(150, lambda: {"modes": 8, "hq": .9, "effective_modes": 8.})
        assert recorder.armed and recorder.arm_update == 150 and recorder.arm_phase == "cold_acquire"
        recorder.note_ring(200, 0, 0.)
        assert recorder.arm_update == 150 and recorder.post_arm_g_steps == 0
    assert ring_listener() is None

    with pr84_reach_candidate(task="mode_hold", ramp="stall", start_step=1000,
                              post_acquire_g_scale=.5) as (recorder, _source):
        recorder.note_ring(1000, 8, .95)
        assert recorder.arm_phase == "warm"
    with pr84_reach_candidate(task="mode_hold", ramp="stall", post_acquire_g_scale=.5) as (recorder, _source):
        recorder.note_ring(1300, 8, .91)
        assert recorder.arm_phase == "stay"
    with pytest.raises(ValueError):
        with pr84_reach_candidate(task="mode_hold", post_acquire_g_scale=.25):
            pass


def _g_displacement(armed, *, passthrough=False, phase=1, role="g"):
    import torch
    init = torch.tensor([1., -2., .5])
    grad = torch.tensor([.4, -.2, .1])
    p = torch.nn.Parameter(init.clone())
    q = torch.nn.Parameter(torch.zeros(1))
    opt_g = torch.optim.Adam([p], lr=2e-3, betas=(0., .99))
    opt_d = torch.optim.Adam([q], lr=2e-3, betas=(0., .99))
    recorder = ReachRecorder()
    recorder.post_acquire_g_scale = .5
    recorder.armed = armed
    recorder.optimizers = (opt_d, opt_g)
    recorder.rows = {opt_d: dict(calls=0), opt_g: dict(calls=0)}
    recorder.phase = phase
    recorder.passthrough = passthrough
    recorder.game_bound = False
    recorder.g_base = [p.detach().clone()]
    recorder.row = {}
    target, opt = (opt_g, opt_g) if role == "g" else (opt_d, opt_d)
    param = p if role == "g" else q
    param.grad = grad[:param.numel()].clone()
    before = param.detach().clone()
    recorder.step(opt, torch.optim.Adam.step)
    return param.detach() - before, recorder.post_arm_g_steps, opt.param_groups[0]["lr"]


def test_post_acquire_halves_only_the_armed_g_adam_step():
    full, steps_off, lr_off = _g_displacement(False)
    half, steps_on, lr_on = _g_displacement(True)
    assert steps_off == 0 and steps_on == 1
    assert half.allclose(full * .5, atol=1e-6, rtol=0.)
    assert lr_off == lr_on == 2e-3
    held, steps_pass, _ = _g_displacement(True, passthrough=True)
    assert steps_pass == 0 and held.allclose(full)
    d_full, _, _ = _g_displacement(False, phase=0, role="d")
    d_armed, steps_d, _ = _g_displacement(True, phase=0, role="d")
    assert steps_d == 0 and d_armed.allclose(d_full)
