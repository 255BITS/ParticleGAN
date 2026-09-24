"""Ordering, anchored operator math, exact controls and sample accounting."""

import pytest
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from reports.toy100.alternating_implicit_scratch import AlternatingImplicitRecorder, alternating_implicit


def test_bilinear_matches_closed_form_alternating_map_implicit_solve():
    d = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    g = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .9), eps=1e-12)
    og = torch.optim.Adam([g], lr=.1, betas=(0., .9), eps=1e-12)
    recorder = AlternatingImplicitRecorder(krylov_dim=4, linear_tolerance=1e-8)
    draws = []
    for phase in recorder.phases(0, od, og, {}):
        draws.append(torch.rand(3))
        d.grad = -g.detach().clone()
        recorder.step(od, torch.optim.Adam.step)
        g.grad = d.detach().clone()
        recorder.step(og, torch.optim.Adam.step)
    pd = .1 / (1 + 1e-12)
    d1 = 2 + pd
    pg = .1 / (d1 + 1e-12)
    matrix = torch.tensor([[1., -pd], [pg, 1. + pg * pd]], dtype=torch.float64)
    rhs = torch.tensor([pd, -pg * d1], dtype=torch.float64)
    delta = torch.linalg.solve(matrix, rhs)
    assert d.item() == pytest.approx(2 + delta[0].item(), abs=2e-10)
    assert g.item() == pytest.approx(1 + delta[1].item(), abs=2e-10)
    assert recorder.base_replays[0]['raw_gradients_bitwise_equal']
    assert recorder.solves[-1]['nonlinear_relative_residual'] < 1e-8
    assert all(torch.equal(draws[0], draw) for draw in draws)
    assert od.state[d]['step'] == og.state[g]['step'] == 1
    for row in recorder.receipt()['optimizers']:
        assert row['calls'] == 1 + len(recorder.queries)
    assert recorder.receipt()['shared_gate_eligible'] is False


def test_zero_field_rests_without_queries_and_keeps_one_moment_update():
    d = torch.nn.Parameter(torch.tensor([0.]))
    g = torch.nn.Parameter(torch.tensor([0.]))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .9))
    og = torch.optim.Adam([g], lr=.1, betas=(0., .9))
    recorder = AlternatingImplicitRecorder()
    for phase in recorder.phases(0, od, og, {}):
        d.grad = -g.detach().clone()
        recorder.step(od, torch.optim.Adam.step)
        g.grad = d.detach().clone()
        recorder.step(og, torch.optim.Adam.step)
    assert d.item() == g.item() == 0
    assert recorder.queries == recorder.base_replays == []
    assert recorder.solves[0]['zero_field']
    assert od.state[d]['step'] == og.state[g]['step'] == 1


class CapturePolicy(NoisePolicy):
    def register_generator_optimizer(self, opt_g, opt_d):
        self.optimizers = (opt_d, opt_g)
        return super().register_generator_optimizer(opt_g, opt_d)


def host(context=None, steps=2):
    torch.set_num_threads(1)
    policy = CapturePolicy(.029, .5, .1, 1200, output_noise_rng='isolated')
    if context is None:
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps),
                                           noise_policy=policy, diagnostics=True)
        recorder = None
    else:
        with context as (recorder, _):
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps),
                                               noise_policy=policy, diagnostics=True)
    state = []
    for optimizer in policy.optimizers:
        for group in optimizer.param_groups:
            for p in group['params']:
                state.extend([p.detach().clone(), *[value.detach().clone()
                             for value in optimizer.state[p].values() if isinstance(value, torch.Tensor)]])
    rng = [torch.get_rng_state().clone(), policy.input_stream.get_state().clone(),
           policy.output_stream.get_state().clone()]
    return result, state, rng, policy.receipt(), recorder


def test_active_explicit_control_is_exact_ordinary_alternating_host():
    plain = host()
    control = host(alternating_implicit(explicit_control=True))
    assert plain[0] == control[0]
    assert plain[3] == control[3]
    assert all(torch.equal(left, right) for left, right in zip(plain[1] + plain[2], control[1] + control[2]))
    assert control[-1].outer_steps == 2
    assert control[-1].queries == []
    assert all(row['calls'] == 2 for row in control[-1].receipt()['optimizers'])


def test_delayed_activation_preserves_original_host_exactly():
    plain = host()
    control = host(alternating_implicit(start_step=1000))
    assert plain[0] == control[0]
    assert plain[3] == control[3]
    assert all(torch.equal(left, right) for left, right in zip(plain[1] + plain[2], control[1] + control[2]))
    assert control[-1].outer_steps == 0


def test_active_host_exact_base_replay_and_one_batch_rng_moments():
    plain = host(steps=1)
    active = host(alternating_implicit(), steps=1)
    recorder = active[-1]
    assert len(recorder.base_replays) == recorder.outer_steps == 1
    assert recorder.base_replays[0]['raw_gradients_bitwise_equal']
    assert all(torch.equal(left, right) for left, right in zip(plain[2], active[2]))
    assert plain[3]['step_calls'] == active[3]['step_calls'] == 1
    assert recorder.rng_replay_verified == len(recorder.queries)
    assert recorder.solves[-1]['accepted']
    for row in recorder.receipt()['optimizers']:
        assert row['calls'] == 1 + len(recorder.queries)
        assert all(step == 1 for group in row['groups'] for step in group['moment_steps'])
