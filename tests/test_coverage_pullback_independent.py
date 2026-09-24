"""Independent closed-form, nonlinear acceptance and state-preservation audit."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from reports.toy100.coverage_pullback import centroid_pullback, coverage_loss, prior_adam_metric


class LinearMap(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.as_tensor(matrix, dtype=torch.float64))

    def forward(self, z):
        return z @ self.weight.T


def test_full_rank_projection_matches_exact_weighted_minimum_norm_solution():
    model = LinearMap([[2., 0., 1., 0.], [0., 3., 0., 1.]])
    z = torch.nn.Parameter(torch.zeros((1, 4), dtype=torch.float64))
    metric = torch.tensor([[.5, 2., 1., 4.]], dtype=torch.float64)
    real = torch.tensor([[.6, -.4]], dtype=torch.float64)
    row = centroid_pullback(model, z, real, metric)
    # J P J^T=diag(3,22); J^T times its inverse times (.6,-.4),
    # then left multiplication by P, gives these four entries exactly.
    expected = torch.tensor([[.2, -2.4 / 22, .2, -1.6 / 22]], dtype=torch.float64)
    assert torch.allclose(z, expected, atol=1e-13, rtol=0)
    assert row['accepted'] and row['alpha'] == 1
    assert row['numerical_rank'] == [2]
    assert torch.allclose(model(z), real, atol=1e-13, rtol=0)
    assert row['coverage_after'] < 1e-25
    # Adding a null-space vector leaves output unchanged and increases
    # the P^-1 norm, an independent check of the selected solution.
    null = torch.tensor([[1., 0., -2., 0.]], dtype=torch.float64)
    assert torch.equal(model(null), torch.zeros((1, 2), dtype=torch.float64))
    assert ((z + null).square() / metric).sum() > (z.square() / metric).sum()


def test_rank_deficient_jacobian_projects_target_and_still_checks_actual_loss():
    model = LinearMap([[1., 0.], [2., 0.]])
    z = torch.nn.Parameter(torch.zeros((1, 2), dtype=torch.float64))
    real = torch.tensor([[1., 0.]], dtype=torch.float64)
    row = centroid_pullback(model, z, real, torch.tensor([[.3, 5.]], dtype=torch.float64))
    assert torch.allclose(z, torch.tensor([[.2, 0.]], dtype=torch.float64), atol=1e-13, rtol=0)
    assert row['accepted'] and row['alpha'] == 1
    assert row['numerical_rank'] == [1]
    assert row['coverage_before'] == pytest.approx(1.)
    assert row['coverage_after'] == pytest.approx(.8)


def test_nonlinear_output_requires_measured_backtracking():
    class Exp(torch.nn.Module):
        def forward(self, z):
            return z.exp()

    z = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
    real = torch.tensor([[10.]], dtype=torch.float64)
    row = centroid_pullback(Exp(), z, real, torch.tensor([[2.]], dtype=torch.float64))
    assert row['accepted'] and row['alpha'] == .25
    assert z.item() == pytest.approx(2.25)
    assert row['coverage_after'] == pytest.approx(float(coverage_loss(real, z.exp()).detach()))
    assert row['coverage_after'] < row['coverage_before']


def test_exhausted_nonlinear_trial_restores_prior_and_rng():
    class Exp(torch.nn.Module):
        def forward(self, z):
            return z.exp()

    z = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
    rng = torch.get_rng_state().clone()
    row = centroid_pullback(Exp(), z, torch.tensor([[10.]], dtype=torch.float64),
                            torch.ones_like(z), max_halves=0)
    assert not row['accepted'] and row['alpha'] == 0
    assert row['trial_evaluations'] == 1
    assert row['coverage_after'] == row['coverage_before'] == 81
    assert torch.equal(z, torch.zeros_like(z))
    assert torch.equal(torch.get_rng_state(), rng)


def test_ties_leave_empty_cells_exactly_fixed():
    model = LinearMap([[1.]])
    z = torch.nn.Parameter(torch.zeros((2, 1), dtype=torch.float64))
    row = centroid_pullback(model, z, torch.tensor([[2.]], dtype=torch.float64), torch.ones_like(z))
    assert row['assigned_counts'] == [1, 0]
    assert row['empty_cells'] == 1
    assert torch.equal(z, torch.tensor([[2.], [0.]], dtype=torch.float64))


def test_zero_jacobian_rests_and_zero_metric_is_rejected_without_mutation():
    model = LinearMap([[0.]])
    z = torch.nn.Parameter(torch.tensor([[.4]], dtype=torch.float64))
    base = z.detach().clone()
    rng = torch.get_rng_state().clone()
    row = centroid_pullback(model, z, torch.tensor([[1.]], dtype=torch.float64), torch.ones_like(z))
    assert not row['accepted'] and row['alpha'] == 0
    assert torch.equal(z, base) and torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(FloatingPointError, match='metric'):
        centroid_pullback(model, z, torch.tensor([[1.]], dtype=torch.float64), torch.zeros_like(z))
    assert torch.equal(z, base) and torch.equal(torch.get_rng_state(), rng)


def test_correction_preserves_network_and_all_adam_state_and_rng():
    model = LinearMap([[2., 1.], [0., 3.]])
    z = torch.nn.Parameter(torch.tensor([[.1, -.2], [2., 3.]], dtype=torch.float64))
    optimizer = torch.optim.Adam([{'params': list(model.parameters()), 'lr': .01},
                                 {'params': [z], 'lr': .02}], betas=(0., .99), eps=1e-8)
    model.weight.grad = torch.ones_like(model.weight)
    z.grad = torch.tensor([[.2, -.3], [.4, .5]], dtype=torch.float64)
    optimizer.step()
    metric = prior_adam_metric(optimizer, z)
    expected = .02 / (optimizer.state[z]['exp_avg_sq'].div(.01).sqrt() + 1e-8)
    assert torch.allclose(metric, expected, atol=0, rtol=2e-15)
    saved_state = deepcopy(optimizer.state_dict())
    saved_weight = model.weight.detach().clone()
    saved_grads = [p.grad.clone() for p in [model.weight, z]]
    rng = torch.get_rng_state().clone()
    real = model(z).detach() + torch.tensor([[.2, .1], [-.3, .4]], dtype=torch.float64)
    row = centroid_pullback(model, z, real, metric)
    assert row['accepted']
    assert torch.equal(model.weight, saved_weight)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(p.grad, g) for p, g in zip([model.weight, z], saved_grads))
    after = optimizer.state_dict()
    assert after['param_groups'] == saved_state['param_groups']
    for key, values in saved_state['state'].items():
        for name, value in values.items():
            assert torch.equal(value, after['state'][key][name])


def test_generator_exception_during_trial_restores_prior():
    class RejectTrial(torch.nn.Module):
        def forward(self, z):
            # jacfwd enables gradients, while actual candidate checks run
            # under no_grad. Base z=0 evaluates successfully in both paths.
            if not torch.is_grad_enabled() and bool((z > .5).any()):
                raise RuntimeError('synthetic trial failure')
            return z

    z = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='synthetic trial failure'):
        centroid_pullback(RejectTrial(), z, torch.tensor([[1.]], dtype=torch.float64), torch.ones_like(z))
    assert torch.equal(z, torch.zeros_like(z))
    assert torch.equal(torch.get_rng_state(), rng)


def _mode_host(context, steps=3):
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.pr84_smoothed_parity import _state_sha

    torch.set_num_threads(1)
    policy = NoisePolicy(.029, .5, .1, 1200, output_noise_rng='isolated')
    with context as (recorder, _):
        result = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(steps=steps),
                                           noise_policy=policy, diagnostics=True)
    return result, _state_sha(recorder), policy.receipt(), recorder


def test_disabled_correction_is_exact_pr84_host_state_and_noise():
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.coverage_smoothed_candidate import coverage_smoothed_candidate

    baseline = _mode_host(pr84_smoothed_candidate())
    observed = _mode_host(coverage_smoothed_candidate(correction=False))
    assert baseline[:3] == observed[:3]
    assert observed[-1].batch_replays_verified == 6
    assert observed[-1].coverage_records == []


def test_conditional_trajectory_is_bitwise_unchanged_and_skips_correction():
    from benchmarks.locked_shared import trajectory
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    from reports.toy100.coverage_smoothed_candidate import coverage_smoothed_candidate
    from reports.toy100.pr84_smoothed_parity import _state_sha

    def run(context):
        policy = NoisePolicy(.029, .5, .1, 400, output_noise_rng='isolated')
        with patch.dict(trajectory.PROTOCOL, {'steps': 3}):
            with context as (recorder, _):
                result = trajectory.train(noise_policy=policy, diagnostics=True)
        return result, _state_sha(recorder), policy.receipt(), recorder

    baseline = run(pr84_smoothed_candidate(task='trajectory'))
    candidate = run(coverage_smoothed_candidate(task='trajectory'))
    assert baseline[:3] == candidate[:3]
    assert candidate[-1].coverage_records == []
    assert candidate[-1].batch_replays_verified == 0


def test_active_host_corrects_after_phase_two_before_ema_without_noise_draws():
    from benchmarks.locked_shared import mode_hold
    from reports.toy100 import coverage_smoothed_candidate as module
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate

    baseline = _mode_host(pr84_smoothed_candidate(), steps=1)
    seen = []
    original = module.CoverageSmoothedRecorder._correct

    def audited(self, real, opt_g):
        assert self.phase == 2 and self.outer_steps == 0
        assert self.rows[opt_g]['calls'] == 3
        assert all(int(opt_g.state[p]['step']) == 1 for group in opt_g.param_groups for p in group['params'])
        assert real is self.real_samples[0]
        assert torch.equal(real, self.phase_zero_samples[0])
        # The second draw is distinct and must never be silently substituted.
        assert not torch.equal(real, self.real_samples[1])
        ema_before = self._local['ema_z'].clone()
        original(self, real, opt_g)
        prior = self._local['prior']
        expected_ema = ema_before.mul(self._local['recipe'].ema).add(prior.z.detach(),
                                                                   alpha=1 - self._local['recipe'].ema)
        seen.append(expected_ema)

    with patch.object(module.CoverageSmoothedRecorder, '_correct', audited):
        active = _mode_host(module.coverage_smoothed_candidate(), steps=1)
    recorder = active[-1]
    assert len(seen) == len(recorder.coverage_records) == 1
    assert torch.equal(recorder._local['ema_z'], seen[0])
    assert baseline[2] == active[2]  # every noise-policy call/draw counter
    assert recorder.rng_replay_verified == recorder.batch_replays_verified == 2
    for optimizer, row in recorder.rows.items():
        assert row['calls'] == 3
        assert all(int(optimizer.state[p]['step']) == 1 for group in optimizer.param_groups for p in group['params'])
