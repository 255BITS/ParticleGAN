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
        assert recorder.post_arm_g_bound is None
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)


def test_post_arm_sticks_on_first_full_ring_check():
    from benchmarks.locked_shared.observation import Recorder, ring_listener

    with pr84_reach_candidate(task="mode_hold", ramp="stall", post_arm_g_bound=.125) as (recorder, _source):
        assert ring_listener() == recorder.note_ring
        assert recorder.curvature_bound == .25 and not recorder.armed
        recorder.note_ring(50, 7, 1.)
        recorder.note_ring(100, 8, .89)
        assert not recorder.armed
        Recorder(1200).record(150, lambda: {"modes": 8, "hq": .9, "effective_modes": 8.})
        assert recorder.armed and recorder.arm_update == 150 and recorder.arm_phase == "cold_acquire"
        recorder.note_ring(200, 0, 0.)
        assert recorder.arm_update == 150 and recorder.post_arm_fires == 0
    assert ring_listener() is None

    with pr84_reach_candidate(task="mode_hold", ramp="stall", start_step=1000,
                              post_arm_g_bound=.125) as (recorder, _source):
        recorder.note_ring(1000, 8, .95)
        assert recorder.arm_phase == "warm"
    with pr84_reach_candidate(task="mode_hold", ramp="stall", post_arm_g_bound=.125) as (recorder, _source):
        recorder.note_ring(1300, 8, .91)
        assert recorder.arm_phase == "stay"
    with pytest.raises(ValueError):
        with pr84_reach_candidate(task="mode_hold", post_arm_g_bound=.2):
            pass


def _phase2_placement(armed):
    import torch
    p = torch.nn.Parameter(torch.tensor([3.]))
    q = torch.nn.Parameter(torch.zeros(1))
    opt_g = torch.optim.Adam([p], lr=1e-3, betas=(0., .99))
    opt_d = torch.optim.Adam([q], lr=1e-3, betas=(0., .99))
    recorder = ReachRecorder()
    recorder.post_arm_g_bound = .125
    recorder.armed = armed
    recorder.optimizers = (opt_d, opt_g)
    recorder.rows = {opt_d: dict(calls=0), opt_g: dict(calls=0)}
    recorder.phase = 2
    recorder.game_bound = False
    recorder.passthrough = False
    recorder.row = dict(outer_step=11)
    recorder.g_base = [torch.tensor([0.])]
    recorder.g1 = [torch.tensor([1.])]
    recorder.gg0 = [torch.tensor([0.])]
    recorder.metric_g = [torch.tensor(1.)]
    p.grad = torch.tensor([1.])
    recorder.step(opt_g, torch.optim.Adam.step)
    return p.detach().clone(), recorder


def test_post_arm_clamps_only_the_g_trust_radius(capsys):
    import json
    import torch
    pre, rec_pre = _phase2_placement(False)
    post, rec_post = _phase2_placement(True)
    err = capsys.readouterr()
    rows = [json.loads(line) for line in err.out.splitlines() if line.startswith("{")]
    assert pre.item() == pytest.approx(.25)
    assert post.item() == pytest.approx(.125)
    assert rec_pre.curvature_bound == rec_post.curvature_bound == .25
    assert rec_pre.post_arm_fires == 0 and rec_post.post_arm_fires == 1
    assert rec_pre.d_curvature_bound == 3.
    assert rows[0]["armed"] is False and rows[0]["g_bound_before"] == rows[0]["g_bound_after"] == .25
    assert rows[0]["update"] == 11 and rows[0]["post_arm_fires"] == 0
    assert rows[1]["armed"] is True and rows[1]["g_bound_before"] == .25 and rows[1]["g_bound_after"] == .125
    assert rows[1]["post_arm_fires"] == 1
    # Phase-1 G Adam is the full step. The clamp is not a global x0.5.
    full = _adam_g_move(False)
    armed = _adam_g_move(True)
    assert torch.allclose(full, armed)


def _adam_g_move(armed):
    import torch
    p = torch.nn.Parameter(torch.tensor([1., -2., .5]))
    q = torch.nn.Parameter(torch.zeros(1))
    opt_g = torch.optim.Adam([p], lr=2e-3, betas=(0., .99))
    opt_d = torch.optim.Adam([q], lr=2e-3, betas=(0., .99))
    recorder = ReachRecorder()
    recorder.post_arm_g_bound = .125
    recorder.armed = armed
    recorder.optimizers = (opt_d, opt_g)
    recorder.rows = {opt_d: dict(calls=0), opt_g: dict(calls=0)}
    recorder.phase = 1
    recorder.game_bound = False
    recorder.row = {}
    recorder.g_base = [p.detach().clone()]
    p.grad = torch.tensor([.4, -.2, .1])
    before = p.detach().clone()
    recorder.step(opt_g, torch.optim.Adam.step)
    return p.detach() - before
