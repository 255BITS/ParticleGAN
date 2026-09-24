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
        assert recorder.mode_drop is False
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)


def test_mode_drop_freezes_g_only_after_arming_and_resumes_at_seven():
    """≤6 skips the G write after an 8-mode arm; ≥7 resumes. D still steps."""
    ordinary = torch.optim.Adam.step
    x = torch.nn.Parameter(torch.tensor([1.]))
    y = torch.nn.Parameter(torch.tensor([2.]))
    opt_g = torch.optim.Adam([x], lr=.1, betas=(0., .9))
    opt_d = torch.optim.Adam([y], lr=.1, betas=(0., .9))
    recorder = ReachRecorder(start_step=0)
    recorder.mode_drop = True
    script = [(8, 1.), (6, .4), (7, .95)]
    recorder._ring_coverage = lambda local: script.pop(0)

    def update():
        for _phase in recorder.phases(recorder.outer_steps, opt_d, opt_g, {"means": torch.zeros(8, 2)}):
            y.grad = torch.ones_like(y)
            recorder.step(opt_d, ordinary)
            x.grad = torch.ones_like(x)
            recorder.step(opt_g, ordinary)

    update()
    assert recorder._armed and recorder._armed_at == 0 and recorder._skip_steps == []
    armed_g, armed_d = x.detach().clone(), y.detach().clone()
    assert not torch.equal(armed_g, torch.tensor([1.]))
    update()
    assert torch.equal(x.detach(), armed_g)
    assert not torch.equal(y.detach(), armed_d)
    assert recorder._skip_steps == [2]
    assert opt_g.state[x]["step"] == opt_d.state[y]["step"] == 2
    held = y.detach().clone()
    update()
    assert not torch.equal(x.detach(), armed_g)
    assert not torch.equal(y.detach(), held)
    assert recorder._skip_steps == [2]
    assert recorder.records[1]["g_skipped"] and not recorder.records[0]["g_skipped"]
    assert recorder.records[1]["g"]["skipped"] is True


def test_mode_drop_before_arming_matches_stall_reach():
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    def run(mode_drop):
        torch.manual_seed(0)
        policy = NoisePolicy(.029, .5, .1, 1200)
        with pr84_reach_candidate(task="mode_hold", ramp="stall", mode_drop=mode_drop) as (recorder, _):
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2), seed=0,
                                               noise_policy=policy)
        return result, recorder._armed, recorder._skip_steps, torch.get_rng_state().clone()

    plain, dropped = run(False), run(True)
    assert plain[0] == dropped[0]
    assert not dropped[1] and dropped[2] == []
    assert torch.equal(plain[3], dropped[3])
