"""Cold-noise objective, conditioning and frozen-warm-equivalence checks."""

from unittest.mock import patch

import pytest
import torch

from reports.toy100 import pr84_critic_refinement as warm
from reports.toy100.pr84_critic_refinement_cold import (
    ColdCriticRefinementRecorder, cached_d_loss, pr84_critic_refinement_cold,
)


@pytest.mark.parametrize('conditional', [False, True])
def test_four_noise_draws_preserve_actual_penalty_and_conditioning(conditional):
    from particlegan import GANLoss, GradientPenalty
    class Quadratic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(5., dtype=torch.float64))
        def forward(self, *args):
            data = args[-1]
            base = self.weight * data.square().sum(-1)
            return base if len(args) == 1 else base + 3. * args[0].sum(-1)
    critic = Quadratic()
    real = torch.tensor([[.8, -.4], [.3, .6]], dtype=torch.float64)
    fake = torch.tensor([[.2, .1], [-.5, .7]], dtype=torch.float64)
    noises = [torch.full_like(real, value) for value in (.1, -.2, .3, -.4)]
    slow = torch.tensor([[2., -3.], [5., 7.]], dtype=torch.float64) if conditional else None
    bank = dict(real=real, fake=fake, slow=slow, input_sigma=.5, input_noise=noises,
                task='trajectory' if conditional else 'mode_hold', penalty_step=1)
    gan, cap = GANLoss('logistic', 'rp'), GradientPenalty('b_cap', coeff=1., kappa=1.)
    actual = cached_d_loss(critic, bank, gan, cap, 999)[0]
    score = lambda x: critic(x) if slow is None else critic(slow, x)
    logistic = gan.d_loss(score(real+.5*noises[0]), score(fake+.5*noises[1]))
    def penalty(data, noise):
        gradient = 2 * critic.weight * (data + .5 * noise)
        norm = (gradient.square().sum(-1)+1e-12).sqrt()
        return torch.relu(norm-1).square().mean()
    expected = logistic + .5*(penalty(real, noises[2])+penalty(fake, noises[3]))
    wrong_reused_noise = logistic + .5*(penalty(real, noises[0])+penalty(fake, noises[1]))
    assert torch.allclose(actual, expected, atol=1e-13, rtol=0)
    assert not torch.allclose(actual, wrong_reused_noise)
    assert torch.allclose(torch.autograd.grad(actual, critic.weight)[0],
                          torch.autograd.grad(expected, critic.weight)[0], atol=1e-13, rtol=0)


@pytest.mark.parametrize('task', ['mode_hold', 'trajectory'])
def test_disabled_support_is_exact_original_host_state_and_observations(task):
    from tests.test_pr84_opponent_prediction import _host
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _host(pr84_smoothed_candidate(task=task), task)
    disabled = _host(pr84_critic_refinement_cold(task=task, refinement=False), task)
    assert original[:3] == disabled[:3]
    assert original[-1].records == disabled[-1].records


def _cold_host(context, task, isolated):
    from benchmarks.locked_shared import mode_hold, trajectory
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.pr84_smoothed_parity import _state_sha
    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200 if task == 'mode_hold' else 400,
                         output_noise_warmup=.2,
                         output_noise_rng='isolated' if isolated else None)
    with context as (rec, _):
        if task == 'mode_hold':
            result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=2),
                                               noise_policy=policy, diagnostics=True)
        else:
            with patch.dict(trajectory.PROTOCOL, {'steps': 2}):
                result = trajectory.train(pairing='stranger', noise_policy=policy, diagnostics=True)
        rng = [torch.get_rng_state().clone(), policy.input_stream.get_state().clone(),
               None if policy.output_stream is None else policy.output_stream.get_state().clone()]
        if task == 'mode_hold':
            rng.append(rec._local['stream'].get_state().clone())
        return result, _state_sha(rec), policy.receipt(), rng, rec


@pytest.mark.parametrize('task', ['mode_hold', 'trajectory'])
@pytest.mark.parametrize('isolated', [False, True])
def test_active_nonzero_noise_has_exact_first_gradient_rng_and_one_moment_per_player(task, isolated):
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _cold_host(pr84_smoothed_candidate(task=task), task, isolated)
    seen = []
    ordinary = ColdCriticRefinementRecorder.step
    def observed(self, optimizer, ordinary_step, closure=None):
        if optimizer is self.optimizers[1] and self.phase in (1, 2):
            seen.append((self.outer_steps, self.phase, self._smooth_width,
                         [p.detach().clone() for p in self._params(self.optimizers[0])]))
        return ordinary(self, optimizer, ordinary_step, closure)
    with patch.object(ColdCriticRefinementRecorder, 'step', observed):
        active = _cold_host(pr84_critic_refinement_cold(task=task), task, isolated)
    rec = active[-1]
    assert active[2] == original[2]
    assert all(torch.equal(a, b) if a is not None else b is None
               for a, b in zip(active[3], original[3]))
    assert rec.bank_rng_verified == rec.fit_rng_verified == rec.parity_gradient_evaluations == 2
    assert rec.rng_replay_verified == 4
    assert [row['host_update'] for row in rec.refinement_records] == [1, 2]
    assert all(row['first_bank_gradient_bitwise_equal'] for row in rec.refinement_records)
    assert all(row['input_sigma'] > 0 for row in rec.refinement_records)
    assert all(row['bank_pairs'] == (1024 if task == 'mode_hold' else 96)
               for row in rec.refinement_records)
    assert all(row['closure_calls'] <= 80 and row['iterations'] <= 40
               for row in rec.refinement_records)
    for left, right in zip(seen[::2], seen[1::2]):
        assert left[:3] == (right[0], 1, right[2])
        assert all(torch.equal(a, b) for a, b in zip(left[3], right[3]))
    for opt, record in rec.rows.items():
        assert record['calls'] == 6
        assert all(int(opt.state[p]['step']) == 2 for group in opt.param_groups for p in group['params'])
    assert rec.receipt()['fit_sample_pairs_evaluated'] == sum(
        row['bank_pairs']*row['closure_calls'] for row in rec.refinement_records)
    if task == 'trajectory':
        assert torch.equal(rec._bank['slow'], rec._local['slow'].repeat(8, 1))
        assert torch.equal(rec._bank['real'], rec._local['paired'].repeat(8, 1))
        assert not torch.equal(rec._bank['real'][:12], rec._local['fast'])
        assert all(row['frozen_g_stencil_width'] == 0 for row in rec.refinement_records)


def test_zero_input_noise_extension_is_bitwise_frozen_warm_adapter():
    from tests.test_pr84_critic_refinement import _late_host
    from reports.toy100.pr84_smoothed_parity import _state_sha, _without_runtime_timing
    original = _late_host(warm.pr84_critic_refinement(), steps=2)
    original_hash = _state_sha(original[0])
    extended = _late_host(pr84_critic_refinement_cold(), steps=2)
    assert _state_sha(extended[0]) == original_hash
    assert extended[1] == original[1]
    for old, new in zip(original[0].refinement_records, extended[0].refinement_records):
        assert _without_runtime_timing(old) == _without_runtime_timing({key: new[key] for key in old})
    assert all(torch.equal(a, b) if a is not None else b is None
               for a, b in zip(original[2], extended[2]))
