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
    assert base.SmoothedBothBoundRecorder is not ReachRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, ReachRecorder)


def test_post_arm_beta1_arms_once_on_full_ring_and_labels_the_window():
    from benchmarks.locked_shared.observation import Recorder, ring_listener

    with pr84_reach_candidate(task="mode_hold", ramp="stall", post_arm_g_beta1=0) as (recorder, _source):
        assert ring_listener() == recorder.note_ring
        assert recorder.post_arm_g_beta1 == 0 and not recorder.armed
        recorder.note_ring(50, 7, 1.)
        recorder.note_ring(100, 8, .89)
        assert not recorder.armed
        Recorder(1200).record(150, lambda: {"modes": 8, "hq": .9, "effective_modes": 8.})
        assert recorder.armed and recorder.arm_update == 150 and recorder.arm_phase == "cold_acquire"
        recorder.note_ring(200, 0, 0.)
        assert recorder.arm_update == 150 and recorder.post_arm_g_steps == 0 and not recorder.momentum_applied
    assert ring_listener() is None

    with pr84_reach_candidate(task="mode_hold", ramp="stall") as (recorder, _source):
        assert ring_listener() is None
        recorder.note_ring(150, 8, 1.)
        assert not recorder.armed

    with pr84_reach_candidate(task="mode_hold", ramp="stall", start_step=1000,
                              post_arm_g_beta1=0) as (recorder, _source):
        recorder.note_ring(1000, 8, .95)
        assert recorder.arm_phase == "warm"
    with pr84_reach_candidate(task="mode_hold", ramp="stall", post_arm_g_beta1=0) as (recorder, _source):
        recorder.note_ring(1300, 8, .91)
        assert recorder.arm_phase == "stay"
    with pytest.raises(ValueError):
        with pr84_reach_candidate(task="mode_hold", post_arm_g_beta1=.5):
            pass


def _g_step(opt_g, opt_d, param, grad, *, armed, beta1_kill=True, passthrough=False, phase=1, role="g",
            host_step=41):
    import torch
    recorder = ReachRecorder()
    recorder.post_arm_g_beta1 = 0 if beta1_kill else None
    recorder.armed = armed
    recorder.arm_update = 40
    recorder.arm_phase = "cold_acquire"
    recorder.optimizers = (opt_d, opt_g)
    recorder.rows = {opt_d: dict(calls=0), opt_g: dict(calls=0)}
    recorder.phase = phase
    recorder.passthrough = passthrough
    recorder.game_bound = False
    recorder.host_step = host_step
    recorder.g_base = [p.detach().clone() for p in recorder._params(opt_g)]
    recorder.row = {}
    target = param if role == "g" else opt_d.param_groups[0]["params"][0]
    opt = opt_g if role == "g" else opt_d
    for group in opt.param_groups:
        for item in group["params"]:
            if item.grad is None:
                item.grad = torch.zeros_like(item)
    target.grad = grad[:target.numel()].reshape(target.shape).clone()
    if role == "g":
        for group in opt_g.param_groups:
            for item in group["params"]:
                if item is not param and item.numel() == 2:
                    item.grad = torch.tensor([.3, -.1])
    before = target.detach().clone()
    beta2 = opt.param_groups[0]["betas"][1]
    lr, eps = opt.param_groups[0]["lr"], opt.param_groups[0]["eps"]
    recorder.step(opt, torch.optim.Adam.step)
    return dict(delta=target.detach() - before, recorder=recorder, beta2=beta2, lr=lr, eps=eps,
                betas=opt.param_groups[0]["betas"], d_betas=opt_d.param_groups[0]["betas"])


def _pair(beta1, *, fill=None):
    import torch
    p = torch.nn.Parameter(torch.tensor([1., -2., .5]))
    q = torch.nn.Parameter(torch.zeros(2))
    z = torch.nn.Parameter(torch.tensor([.25, -.5]))
    opt_g = torch.optim.Adam([
        {"params": [p], "lr": 2e-3, "betas": (beta1, .999)},
        {"params": [z], "lr": 4e-3, "betas": (beta1, .99)},
    ], lr=2e-3, betas=(beta1, .999))
    opt_d = torch.optim.Adam([q], lr=2e-3, betas=(.9, .999))
    grad = torch.tensor([.4, -.2, .1])
    p.grad = grad.clone()
    z.grad = torch.tensor([.3, -.1])
    q.grad = torch.tensor([.2, -.4])
    opt_g.step()
    opt_d.step()
    if fill is not None:
        with torch.no_grad():
            for group in opt_g.param_groups:
                for param in group["params"]:
                    opt_g.state[param]["exp_avg"].fill_(fill)
    opt_g.zero_grad()
    opt_d.zero_grad()
    return opt_g, opt_d, p, q, z, grad


def test_unarmed_g_adam_matches_stall_reach_and_keeps_beta1():
    import torch
    opt_g, opt_d, p, q, z, grad = _pair(0.0)
    held = _g_step(opt_g, opt_d, p, grad, armed=False)
    assert held["recorder"].post_arm_g_steps == 0 and not held["recorder"].momentum_applied
    assert held["betas"][0] == 0.0 and held["betas"][1] == pytest.approx(.999)
    assert opt_g.param_groups[1]["betas"] == pytest.approx((0.0, .99))
    assert held["d_betas"][0] == pytest.approx(.9)
    opt_g2, opt_d2, p2, _, z2, grad2 = _pair(0.0)
    plain = torch.optim.Adam.step
    before = p2.detach().clone()
    p2.grad = grad2.clone()
    z2.grad = torch.tensor([.3, -.1])
    plain(opt_g2)
    assert torch.allclose(held["delta"], p2.detach() - before)
    assert torch.equal(opt_g.state[p]["exp_avg"], opt_g2.state[p2]["exp_avg"])


def test_armed_step_zeros_exp_avg_once_and_drops_stale_momentum():
    import torch
    opt_g, opt_d, p, q, z, grad = _pair(.9, fill=3.)
    stale = opt_g.state[p]["exp_avg"].clone()
    assert torch.all(stale == 3.)
    beta2 = opt_g.param_groups[0]["betas"][1]
    eps = opt_g.param_groups[0]["eps"]
    lr = opt_g.param_groups[0]["lr"]
    sq = opt_g.state[p]["exp_avg_sq"].clone()
    step = opt_g.state[p]["step"].clone() if torch.is_tensor(opt_g.state[p]["step"]) else opt_g.state[p]["step"]
    z_sq = opt_g.state[z]["exp_avg_sq"].clone()
    z_step = opt_g.state[z]["step"].clone() if torch.is_tensor(opt_g.state[z]["step"]) else opt_g.state[z]["step"]
    d_beta = opt_d.param_groups[0]["betas"]
    armed = _g_step(opt_g, opt_d, p, grad, armed=True)
    rec = armed["recorder"]
    assert rec.momentum_applied and rec.exp_avg_zeroed and rec.exp_avg_zero_count == 2
    assert rec.momentum_apply_update == 42 and rec.post_arm_g_steps == 1
    assert rec.g_beta1_before == pytest.approx(.9)
    assert opt_g.param_groups[0]["betas"] == pytest.approx((0.0, .999))
    assert opt_g.param_groups[1]["betas"] == pytest.approx((0.0, .99))
    assert opt_g.param_groups[0]["lr"] == lr and opt_g.param_groups[1]["lr"] == 4e-3
    assert opt_g.param_groups[0]["eps"] == eps and opt_g.defaults["betas"][0] == 0.0
    assert opt_g.defaults["betas"][1] == pytest.approx(.999)
    assert armed["d_betas"] == d_beta
    assert beta2 == pytest.approx(.999)

    pref = torch.nn.Parameter(torch.tensor([1., -2., .5]))
    # Match the post-warmup parameter, which _pair already stepped once before the kill.
    with torch.no_grad():
        pref.copy_(p.detach() - armed["delta"])
    ref = torch.optim.Adam([pref], lr=lr, betas=(0.0, beta2), eps=eps)
    ref.state[pref] = {"step": step.clone() if torch.is_tensor(step) else step,
                       "exp_avg": torch.zeros_like(pref),
                       "exp_avg_sq": sq.clone()}
    pref.grad = grad.clone()
    before = pref.detach().clone()
    torch.optim.Adam.step(ref)
    assert torch.allclose(armed["delta"], pref.detach() - before, atol=0., rtol=0.)

    opt_hold, opt_d_hold, p_hold, _, _, grad_hold = _pair(.9, fill=3.)
    held = _g_step(opt_hold, opt_d_hold, p_hold, grad_hold, armed=True, passthrough=True)
    assert held["recorder"].post_arm_g_steps == 0 and not held["recorder"].exp_avg_zeroed
    assert held["betas"][0] == pytest.approx(.9)
    # Passthrough still takes the host Adam step, so the stale moment is mixed in, not wiped.
    assert torch.allclose(opt_hold.state[p_hold]["exp_avg"], .9 * 3. + .1 * grad_hold)

    # Second real step keeps β1 at 0 and does not zero exp_avg again.
    opt_g.state[p]["exp_avg"].fill_(5.)
    p.grad = grad.clone()
    z.grad = torch.tensor([.3, -.1])
    zero_calls = {"n": 0}
    original_zero = torch.Tensor.zero_

    def spy_zero(self, *args, **kwargs):
        if self is opt_g.state[p]["exp_avg"] or self is opt_g.state[z]["exp_avg"]:
            zero_calls["n"] += 1
        return original_zero(self, *args, **kwargs)

    torch.Tensor.zero_ = spy_zero
    try:
        armed["recorder"].step(opt_g, torch.optim.Adam.step)
    finally:
        torch.Tensor.zero_ = original_zero
    assert zero_calls["n"] == 0
    assert rec.exp_avg_zero_count == 2 and rec.post_arm_g_steps == 2
    assert opt_g.param_groups[0]["betas"][0] == 0.0
    assert torch.allclose(opt_g.state[p]["exp_avg"], grad)

    # Unarmed step with the same stale moment is a different displacement.
    opt_u, opt_du, p_u, _, _, grad_u = _pair(.9, fill=3.)
    unarmed = _g_step(opt_u, opt_du, p_u, grad_u, armed=False)
    assert not torch.allclose(unarmed["delta"], armed["delta"])
    assert unarmed["betas"][0] == pytest.approx(.9)
