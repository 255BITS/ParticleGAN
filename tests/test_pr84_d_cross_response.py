"""Signs, exact rest, accepted-point queries and source-adapter parity."""

from unittest.mock import patch

import pytest
import torch

from reports.toy100.pr84_d_cross_response import (
    DCrossResponseRecorder, pr84_d_cross_response, response_point,
)


def test_bilinear_cross_response_is_the_declared_stable_alternating_map():
    a, b = .2, .3
    state = torch.tensor([2., 1.], dtype=torch.float64)
    d_star = state[0] - a * state[1]
    g_plus = state[1] + b * d_star
    d_plus = response_point([d_star[None]], [torch.tensor([a])],
                            [state[1:]], [g_plus[None]])[0][0]
    matrix = torch.tensor([[1-a*b, -a*(1-a*b)], [b, 1-a*b]], dtype=torch.float64)
    assert torch.allclose(torch.stack([d_plus, g_plus]), matrix @ state, atol=1e-8, rtol=0)
    assert torch.linalg.det(matrix).item() == pytest.approx(1-a*b)
    assert torch.allclose(torch.linalg.eigvals(matrix).abs(),
                          torch.full((2,), (1-a*b)**.5, dtype=torch.float64))


def test_response_vanishes_when_opponent_does_not_move_and_rejects_nonfinite():
    d = torch.tensor([2., -1.])
    field = torch.tensor([3., 4.])
    assert torch.equal(response_point([d], [torch.ones(2)], [field], [field])[0], d)
    assert torch.equal(d, torch.tensor([2., -1.]))
    with pytest.raises(FloatingPointError):
        response_point([d], [torch.ones(2)], [field], [torch.full((2,), float('inf'))])


@pytest.mark.parametrize('bound_active', [False, True])
def test_actual_accepted_d_and_g_are_used_once_with_correct_query_accounting(bound_active):
    d = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    g = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .99))
    og = torch.optim.Adam([g], lr=.2, betas=(0., .99))
    rec = DCrossResponseRecorder()
    if bound_active:
        rec.d_curvature_bound = .01
    original = torch.optim.Adam.step
    query_points = []
    before_rng = torch.get_rng_state().clone()
    for phase in rec.phases(0, od, og, {}):
        d.grad = d.detach() + g.detach()
        if phase >= 3:
            query_points.append((float(d.detach()), float(g.detach())))
            rec.capture_d_query(od)
        else:
            rec.step(od, original)
            g.grad = -d.detach().clone()
            rec.step(og, original)
    expected = response_point(rec.d_star, rec.metric_d,
                             [rec.d_star[0] + rec.g_base[0]],
                             [rec.d_star[0] + g.detach()])[0]
    assert torch.allclose(d, expected, atol=1e-15, rtol=0)
    assert g.item() > 1
    assert all(x[0] == rec.d_star[0].item() for x in query_points)
    assert query_points[-1][1] == g.item()
    if bound_active:
        assert query_points[0][1] == 1
    assert rec.additional_d_fields == rec.d_query_rng_verified == 1 + int(bound_active)
    assert rec.cross_records[0]['base_field_reused'] == (not bound_active)
    assert rec.rows[od]['calls'] == rec.rows[og]['calls'] == 3
    assert int(od.state[d]['step']) == int(og.state[g]['step']) == 1
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_zero_field_keeps_both_players_still_and_moments_advance_once():
    d = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    g = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .99))
    og = torch.optim.Adam([g], lr=.2, betas=(0., .99))
    rec = DCrossResponseRecorder()
    original = torch.optim.Adam.step
    for phase in rec.phases(0, od, og, {}):
        d.grad = torch.zeros_like(d)
        if phase >= 3:
            rec.capture_d_query(od)
        else:
            rec.step(od, original)
            g.grad = torch.zeros_like(g)
            rec.step(og, original)
    assert not torch.count_nonzero(d) and not torch.count_nonzero(g)
    assert rec.cross_records[0]['correction_parameter_norm'] == 0
    assert int(od.state[d]['step']) == int(og.state[g]['step']) == 1


def _host(context, task='mode_hold', steps=2):
    from benchmarks.locked_shared import mode_hold, trajectory
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.pr84_smoothed_parity import _state_sha
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200 if task == 'mode_hold' else 400,
                         output_noise_rng='isolated')
    with context as (recorder, _):
        if task == 'mode_hold':
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps),
                                               noise_policy=policy, diagnostics=True)
        else:
            with patch.dict(trajectory.PROTOCOL, {'steps': steps}):
                result = trajectory.train(noise_policy=policy, diagnostics=True)
        rng = [torch.get_rng_state().clone(), recorder._local['stream'].get_state().clone()
               if 'stream' in recorder._local else None, policy.input_stream.get_state().clone(),
               policy.output_stream.get_state().clone()]
    return result, _state_sha(recorder), policy.receipt(), rng, recorder


@pytest.mark.parametrize('task', ['mode_hold', 'trajectory'])
def test_disabled_response_preserves_exact_original_full_host_state(task):
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _host(pr84_smoothed_candidate(task=task), task)
    disabled = _host(pr84_d_cross_response(task=task, correction=False), task)
    assert original[:3] == disabled[:3]
    assert original[-1].records == disabled[-1].records
    assert disabled[-1].additional_d_fields == 0


def test_active_host_replays_only_d_blocks_without_advancing_training_rng_or_moments():
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _host(pr84_smoothed_candidate())
    active = _host(pr84_d_cross_response())
    assert all(torch.equal(a, b) if a is not None else b is None
               for a, b in zip(original[3], active[3]))
    rec = active[-1]
    assert rec.additional_d_fields == rec.d_query_rng_verified
    assert active[2]['step_calls'] == original[2]['step_calls'] == 2
    assert active[2]['output_train_calls'] - original[2]['output_train_calls'] == rec.additional_d_fields
    for opt, row in rec.rows.items():
        assert row['calls'] == 6
        assert all(int(opt.state[p]['step']) == 2 for group in opt.param_groups for p in group['params'])


def test_query_error_restores_actual_g_d_and_full_rng():
    d = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    g = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .99))
    og = torch.optim.Adam([g], lr=.2, betas=(0., .99))
    rec = DCrossResponseRecorder()
    original = torch.optim.Adam.step
    expected_rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='synthetic D query failure'):
        iterator = rec.phases(0, od, og, {})
        try:
            for phase in iterator:
                if phase >= 3:
                    torch.rand(5)
                    raise RuntimeError('synthetic D query failure')
                d.grad = g.detach().clone()
                rec.step(od, original)
                g.grad = -d.detach().clone()
                rec.step(og, original)
                if phase == 2:
                    expected_g, expected_d = g.detach().clone(), d.detach().clone()
        finally:
            iterator.close()
    assert torch.equal(g, expected_g) and torch.equal(d, expected_d)
    assert torch.equal(torch.get_rng_state(), expected_rng)
