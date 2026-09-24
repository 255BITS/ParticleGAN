"""Post-arm common-mode null on G's output gradient. No retune, no centering of y."""

import torch
from torch import nn

from benchmarks.locked_shared.observation import Recorder, notify_ring, set_ring_listener
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_common_mode_grad_null import (
    ARM_HQ, CommonModeGradNullRecorder, pr84_common_mode_grad_null, subtract_common_mode,
)
from reports.toy100.pr84_reach_candidate import ReachRecorder


def test_subtract_removes_only_the_batch_mean():
    grad = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    original = grad.clone()
    nulled, mean = subtract_common_mode(grad)
    assert torch.allclose(mean, torch.tensor([3., 4.]))
    assert torch.allclose(nulled, grad - mean)
    assert torch.allclose(nulled.mean(0), torch.zeros(2))
    assert torch.equal(grad, original)


def test_pure_translation_gradient_is_zero_and_zero_mean_is_unchanged():
    rows = torch.tensor([[0.4, -0.2]]).expand(8, 2).clone()
    nulled, mean = subtract_common_mode(rows)
    assert torch.allclose(nulled, torch.zeros_like(rows))
    assert torch.allclose(mean, rows[0])
    shaped = torch.tensor([[1., -1.], [-1., 1.], [0.5, -0.5], [-0.5, 0.5]])
    kept, _ = subtract_common_mode(shaped)
    assert torch.allclose(kept, shaped)


def test_hook_leaves_y_raw_and_drops_translation_before_the_weights():
    torch.manual_seed(0)
    layer = nn.Linear(3, 2)
    z = torch.randn(6, 3)
    recorder = CommonModeGradNullRecorder(start_step=0)
    recorder.armed = True
    recorder.enabled = True
    recorder.passthrough = False
    recorder.phase = 1
    recorder.row = {"outer_step": 4}
    with torch.no_grad():
        raw = layer(z).clone()
    handle = layer.register_forward_hook(recorder._on_forward)
    try:
        y = layer(z)
        assert torch.equal(y, raw)
        y.sum().backward()
    finally:
        handle.remove()
    assert torch.allclose(layer.weight.grad, torch.zeros_like(layer.weight.grad))
    assert torch.allclose(layer.bias.grad, torch.zeros_like(layer.bias.grad))
    assert len(recorder._phase_l2) == 1 and recorder._phase_l2[0] > 0


def test_shape_gradient_still_reaches_the_weights():
    torch.manual_seed(1)
    layer = nn.Linear(3, 2)
    state = {k: v.clone() for k, v in layer.state_dict().items()}
    z = torch.randn(4, 3)
    weights = torch.tensor([1., -1., 0.5, -0.5])

    def run(null):
        fresh = nn.Linear(3, 2)
        fresh.load_state_dict(state)
        y = fresh(z)
        if null:
            y.register_hook(lambda grad: subtract_common_mode(grad)[0])
        (y[:, 0] * weights).sum().backward()
        return fresh.weight.grad.clone(), fresh.bias.grad.clone()

    raw_w, raw_b = run(False)
    null_w, null_b = run(True)
    assert torch.allclose(null_w, raw_w)
    assert torch.allclose(null_b, raw_b)
    assert raw_w.abs().sum() > 0


def test_detached_discriminator_path_does_not_consume_the_null():
    torch.manual_seed(2)
    layer = nn.Linear(3, 2)
    z = torch.randn(5, 3)
    recorder = CommonModeGradNullRecorder(start_step=0)
    recorder.armed = True
    recorder.enabled = True
    recorder.passthrough = False
    recorder.phase = 1
    critic = nn.Linear(2, 1)
    handle = layer.register_forward_hook(recorder._on_forward)
    try:
        y = layer(z)
        # D scores the detached fake, the same cut the ring host uses.
        critic(y.detach()).sum().backward()
    finally:
        handle.remove()
    assert recorder._phase_l2 == []
    assert layer.weight.grad is None
    assert critic.weight.grad is not None and critic.weight.grad.abs().sum() > 0


def test_adam_step_is_not_skipped_and_betas_stay():
    torch.manual_seed(3)
    layer = nn.Linear(2, 2)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2, betas=(0.0, 0.99))
    z = torch.randn(4, 2)
    y = layer(z)
    y.register_hook(lambda grad: subtract_common_mode(grad)[0])
    # Mixed common mode and shape, so the step is a real non-translation update.
    (y.square().sum() + y.sum()).backward()
    before = [p.detach().clone() for p in layer.parameters()]
    opt.step()
    assert opt.param_groups[0]["betas"] == (0.0, 0.99)
    assert all(int(opt.state[p]["step"]) == 1 for p in layer.parameters())
    assert any(not torch.equal(p, b) for p, b in zip(layer.parameters(), before))


def test_arm_is_sticky_on_the_logged_ring_check_only():
    recorder = CommonModeGradNullRecorder(start_step=0)
    previous = set_ring_listener(recorder.note_ring)
    try:
        notify_ring(40, 7, 1.0)
        notify_ring(50, 8, ARM_HQ - 1e-6)
        assert not recorder.armed
        notify_ring(650, 8, ARM_HQ)
        assert recorder.armed and recorder.arm_update == 650
        assert recorder.arm_phase == "cold_acquire"
        notify_ring(700, 6, 0.0)
        assert recorder.arm_update == 650
    finally:
        set_ring_listener(previous)
    warm = CommonModeGradNullRecorder(start_step=1000)
    warm.note_ring(1000, 8, 0.95)
    assert warm.arm_phase == "warm" and warm.arm_update == 1000
    late = CommonModeGradNullRecorder(start_step=0)
    late.note_ring(1210, 8, 0.91)
    assert late.arm_phase == "stay"


def test_hook_stays_off_until_armed_and_enabled():
    layer = nn.Linear(2, 2)
    recorder = CommonModeGradNullRecorder(start_step=0)
    recorder._sync_hook(layer, 0)
    assert recorder._hook is None
    recorder.armed = True
    recorder.enabled = False
    recorder._sync_hook(layer, 0)
    assert recorder._hook is None
    recorder.enabled = True
    recorder._sync_hook(layer, 0)
    assert recorder._hook is not None
    recorder.enabled = False
    recorder._sync_hook(layer, 0)
    assert recorder._hook is None


def test_scheduled_record_notifies_and_ignores_checks_without_a_ring():
    hits = []
    previous = set_ring_listener(lambda step, modes, hq: hits.append((step, modes, hq)))
    try:
        recorder = Recorder(24)
        recorder.record(1, lambda: {"modes": 8, "hq": 0.95, "mse": 1.0})
        recorder.record(2, lambda: {"mse": 0.2})
    finally:
        set_ring_listener(previous)
    assert hits == [(1, 8, 0.95)]


def test_two_armed_updates_null_once_per_g_step_and_keep_adam():
    from benchmarks.locked_shared import mode_hold

    with pr84_common_mode_grad_null(task="mode_hold") as (recorder, _source):
        recorder.note_ring(0, 8, ARM_HQ)
        mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2), seed=0)
    assert [row["step"] for row in recorder.null_rows] == [1, 2]
    assert all(row["l2"] > 0 for row in recorder.null_rows)
    assert recorder._hook is None


def test_factory_is_stall_reach_and_restores_the_host():
    previous = set_ring_listener(None)
    try:
        with pr84_common_mode_grad_null(task="mode_hold") as (recorder, _source):
            assert isinstance(recorder, CommonModeGradNullRecorder)
            assert isinstance(recorder, ReachRecorder)
            assert recorder.ramp == "stall" and recorder.reach == 0.5
            assert recorder.game_bound is False
            assert recorder.curvature_bound == 0.25
            assert recorder.d_curvature_bound == 3.0
    finally:
        set_ring_listener(previous)
    assert base.SmoothedBothBoundRecorder is not CommonModeGradNullRecorder
    assert not issubclass(base.SmoothedBothBoundRecorder, CommonModeGradNullRecorder)
