"""Focused finite-GH9 adapter checks without a host training run."""

from copy import deepcopy

import pytest
import torch

from reports.toy100.forward_kl_free_filter import quadrature, cross_entropy
from reports.toy100.forward_kl_neural_v2 import (
    WIDTH, ForwardKLV2Recorder,
)


class Prior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z = torch.nn.Parameter(torch.zeros((12, 2)))


class Policy:
    def __init__(self):
        self.input_stream = torch.Generator().manual_seed(101)
        self.output_stream = torch.Generator().manual_seed(102)
        self.output_noise_learnable = False
        self.output_scale = None
        self.output_sigma = .029


class Recipe:
    particle_l2 = 0.


def setup_recorder(target_mode='improve'):
    torch.set_num_threads(1)
    generator = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        generator.weight.copy_(torch.eye(2))
    prior = Prior()
    critic = torch.nn.Linear(2, 1)
    opt_g = torch.optim.Adam(list(generator.parameters()) + list(prior.parameters()),
                             lr=.01)
    opt_d = torch.optim.Adam(critic.parameters(), lr=.01)
    params = [p for group in opt_g.param_groups for p in group['params']]
    pre = [p.detach().clone() for p in params]
    pre_points = generator(prior.z).detach().clone()
    # Simulate the one native Adam update that must remain accounted for even
    # when the likelihood correction ultimately rests.
    for p in params:
        p.grad = -torch.ones_like(p)
    for p in critic.parameters():
        p.grad = torch.ones_like(p)
    opt_d.step()
    opt_g.step()
    moment_d = deepcopy(opt_d.state_dict())
    moment_g = deepcopy(opt_g.state_dict())
    stream = torch.Generator().manual_seed(103)
    policy = Policy()
    real = torch.full((128, 2), .2)
    locations, weights = quadrature(real, WIDTH, 9)
    variance = WIDTH**2 + policy.output_sigma**2

    def operator(history, bank, points, width, sigma):
        assert width == WIDTH and sigma == policy.output_sigma
        assert torch.equal(history, real) and torch.equal(bank, real)
        assert torch.equal(points, pre_points)
        target = (torch.full((12, 2), .2, dtype=torch.float64)
                  if target_mode == 'improve' else points.double().clone())
        before = float(cross_entropy(locations, weights, points, variance))
        after = float(cross_entropy(locations, weights, target, variance))
        row = dict(selected=('GH5_WHOLE_ACCEPTED_BY_GH9' if target_mode == 'improve'
                             else 'EXACT_REST'), initial_audit9=before,
                   final_audit9=after)
        return target, row, (locations, weights)

    recorder = ForwardKLV2Recorder(start_step=0, target_operator=operator)
    recorder._local = dict(generator=generator, prior=prior, critic=critic,
        opt_d=opt_d, opt_g=opt_g, noise_policy=policy, recipe=Recipe(),
        slow=None, step=0, stream=stream)
    recorder.real = real
    recorder.pre_points = pre_points
    recorder.g_base = pre
    recorder.row = {}
    rng = (torch.get_rng_state().clone(), stream.get_state().clone(),
           policy.input_stream.get_state().clone(),
           policy.output_stream.get_state().clone())
    return recorder, opt_g, moment_d, moment_g, rng


def assert_adam_and_rng(recorder, moments_d, moments_g, rng):
    local = recorder._local
    # Tensor-inclusive structural equality without depending on dict order.
    from reports.toy100.pr84_critic_refinement_capture import _sha
    assert _sha(local['opt_d'].state_dict()) == _sha(moments_d)
    assert _sha(local['opt_g'].state_dict()) == _sha(moments_g)
    policy = local['noise_policy']
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert torch.equal(local['stream'].get_state(), rng[1])
    assert torch.equal(policy.input_stream.get_state(), rng[2])
    assert torch.equal(policy.output_stream.get_state(), rng[3])


def test_converged_neural_target_strict_gh9_descent_and_once_adam_state():
    recorder, opt_g, md, mg, rng = setup_recorder()
    recorder.correct(opt_g)
    row = recorder.corrections[-1]
    assert row['selected'] == 'FINITE_GH9_FITTED_TARGET'
    assert row['fit']['status'] == 'CONVERGED'
    assert row['final_gh9'] < row['pre_gh9']
    assert row['bank_count'] == 1 and row['history_samples'] == 128
    assert recorder.gradient_checks == recorder.rng_checks == recorder.owner_checks == 1
    assert_adam_and_rng(recorder, md, mg, rng)


def test_failed_neural_fit_rests_exactly_without_native_fallback(monkeypatch):
    recorder, opt_g, md, mg, rng = setup_recorder()
    prior = recorder._local['prior']
    pre = [p.clone() for p in recorder.g_base]

    def fake_fit(*args, **kwargs):
        with torch.no_grad():
            prior.z.add_(.1)
        return dict(status='BUDGET', records=[])

    monkeypatch.setattr('reports.toy100.forward_kl_neural_v2.fit_output_targets',fake_fit)
    recorder.correct(opt_g)
    assert recorder.corrections[-1]['selected'] == 'EXACT_REST'
    assert recorder.corrections[-1]['fit']['status'] == 'BUDGET'
    assert all(torch.equal(p, saved) for p, saved in
               zip(recorder._params(opt_g), pre))
    assert_adam_and_rng(recorder, md, mg, rng)


def test_pure_exact_rest_skips_fit_and_discards_native_parameter_move(monkeypatch):
    recorder, opt_g, md, mg, rng = setup_recorder(target_mode='rest')
    native = [p.clone() for p in recorder._params(opt_g)]
    assert any(not torch.equal(p, saved) for p, saved in
               zip(native, recorder.g_base))

    def must_not_fit(*args, **kwargs):
        raise AssertionError('exact output rest must skip neural fitting')

    monkeypatch.setattr('reports.toy100.forward_kl_neural_v2.fit_output_targets',must_not_fit)
    recorder.correct(opt_g)
    assert recorder.corrections[-1]['selected'] == 'EXACT_REST'
    assert recorder.corrections[-1]['fit']['status'] == 'SKIPPED_EXACT_REST'
    assert recorder.receipt()['additional_joint_output_jacobians'] == 0
    assert all(torch.equal(p, saved) for p, saved in
               zip(recorder._params(opt_g), recorder.g_base))
    assert_adam_and_rng(recorder, md, mg, rng)


def test_resume_requires_exact_completed_host_clock():
    bank = torch.zeros((128, 2), dtype=torch.float32)
    source = ForwardKLV2Recorder(start_step=0)
    source.banks = [bank]
    source.learner_bank_count = 1
    source.learner_last_observed_step = 1
    state = source.learner_state_dict()
    resume = ForwardKLV2Recorder(start_step=1, history_mode='resume')
    resume.load_learner_state_dict(state)
    assert resume.loaded_history and resume.learner_last_observed_step == 1
    with pytest.raises(ValueError):
        resume.load_learner_state_dict(state)
    bad = deepcopy(state)
    bad['last_bank_id'] = True
    with pytest.raises(ValueError):
        ForwardKLV2Recorder(start_step=1, history_mode='resume').load_learner_state_dict(bad)
    with pytest.raises(ValueError):
        ForwardKLV2Recorder(start_step=2, history_mode='resume').load_learner_state_dict(state)


def test_missing_resume_state_stops_before_likelihood_update():
    recorder, opt_g, _, _, _ = setup_recorder()
    recorder.history_mode = 'resume'
    with pytest.raises(RuntimeError, match='missing source-bound likelihood history'):
        recorder.correct(opt_g)
    assert recorder.learner_bank_count == 0 and not recorder.banks


def test_oracle_quality_receipt_is_rejected_and_pre_g_parameters_restored():
    recorder, opt_g, _, _, _ = setup_recorder()
    original = recorder.target_operator

    def bad_operator(*args):
        target, row, finite9 = original(*args)
        row['final_quality'] = {'hq': 1.}
        return target, row, finite9

    recorder.target_operator = bad_operator
    with pytest.raises(RuntimeError, match='oracle-free'):
        recorder.correct(opt_g)
    assert recorder.learner_bank_count == 0 and not recorder.banks
    assert all(torch.equal(p, saved) for p, saved in
               zip(recorder._params(opt_g), recorder.g_base))


def test_actual_pure_operator_and_neural_fit_on_cached_cold_real_bank():
    """Exercise the default operator's autograd and the actual GN fit path."""
    pytest.importorskip('reports.toy100.forward_kl_gh9_remembered')
    import hashlib
    import json
    from pathlib import Path
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.sample_anchor_free1200 import load_states

    cold, _, _ = load_states()
    stream = torch.Generator().set_state(cold['rng']['data'])
    bank = mode_hold.sample_ring(mode_hold.ring_means(), 128,
                                 mode_hold.SIGMA, stream)
    first = json.loads((Path(__file__).resolve().parents[1]/
        'reports/toy100/continuous-evidence/round8-forward-kl-first-bank/v2/result.json').read_text())
    assert hashlib.sha256(bank.contiguous().numpy().tobytes()).hexdigest() == (
        first['declaration']['native_real_bank_sha256']['cold1'])

    # Small affine neural map with native one-step Adam state; this is a
    # source-bound correction smoke, not a replay of the saved cold model.
    recorder, opt_g, md, mg, rng = setup_recorder()
    recorder.real = bank
    recorder.target_operator = None
    recorder.correct(opt_g)
    row = recorder.corrections[-1]
    assert row['pure']['selected'] == 'GH5_WHOLE_ACCEPTED_BY_GH9'
    assert 'initial_quality' not in row['pure']
    assert 'final_quality' not in row['pure']
    assert row['selected'] == 'FINITE_GH9_FITTED_TARGET'
    assert row['fit']['status'] == 'CONVERGED'
    assert row['final_gh9'] < row['pre_gh9']
    assert recorder.receipt()['pure_operator_source_sha256'] is not None
    assert_adam_and_rng(recorder, md, mg, rng)
