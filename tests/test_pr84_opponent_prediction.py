"""Prediction mathematics and faithful temporary-opponent integration."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from reports.toy100.pr84_opponent_prediction import (
    OpponentPredictionRecorder, predicted_opponent, pr84_opponent_prediction,
)


def test_fixed_metric_bilinear_prediction_contracts_rotation_without_zero_pull():
    # D minimizes d*g; G minimizes -d*g. D*=d-a*g and G responds
    # to 2D*-d. For fixed a,b and 0<a*b<1, both eigenvalues have
    # modulus sqrt(1-a*b), whereas ordinary alternation has determinant1.
    a, b = .2, .3
    matrix = torch.tensor([[1., -a], [b, 1 - 2 * a * b]], dtype=torch.float64)
    before = torch.tensor([2., 1.], dtype=torch.float64)
    d_star = before[0] - a * before[1]
    d_probe = predicted_opponent([before[:1]], [d_star[None]])[0][0]
    actual = torch.stack((d_star, before[1] + b * d_probe))
    assert torch.allclose(actual, matrix @ before, atol=1e-15, rtol=0)
    assert torch.linalg.det(matrix).item() == pytest.approx(1 - a * b)
    assert torch.allclose(torch.linalg.eigvals(matrix).abs(),
                          torch.full((2,), (1 - a * b) ** .5, dtype=torch.float64))
    zero = torch.zeros(3, dtype=torch.float64)
    assert torch.equal(predicted_opponent([zero], [zero])[0], zero)


def test_prediction_does_not_mutate_inputs_and_rejects_nonfinite_probe():
    base = torch.tensor([2., -1.])
    material = torch.tensor([3., 4.])
    result = predicted_opponent([base], [material])[0]
    assert torch.equal(result, torch.tensor([4., 9.]))
    assert torch.equal(base, torch.tensor([2., -1.]))
    assert torch.equal(material, torch.tensor([3., 4.]))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        predicted_opponent([base], [torch.tensor([float('inf'), 1.])])


def test_recorded_bilinear_step_keeps_actual_d_and_updates_moments_once():
    d = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    g = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .99))
    og = torch.optim.Adam([g], lr=.2, betas=(0., .99))
    rec = OpponentPredictionRecorder()
    d_queries, g_queries = [], []
    ordinary = torch.optim.Adam.step
    for phase in rec.phases(0, od, og, {}):
        d_queries.append(float(d.detach()))
        od.zero_grad()
        d.grad = g.detach().clone()
        rec.step(od, ordinary)
        g_queries.append(float(d.detach()))
        og.zero_grad()
        g.grad = -d.detach().clone()
        rec.step(og, ordinary)
        assert not rec._probe_active
    expected_d = 2 - .1 / (1 + od.param_groups[0]['eps'])
    expected_probe = 2 * expected_d - 2
    expected_g = 1 + .2 * expected_probe / (abs(expected_probe) + og.param_groups[0]['eps'])
    assert d.item() == pytest.approx(expected_d, abs=1e-14)
    assert g.item() == pytest.approx(expected_g, abs=1e-14)
    assert d_queries == pytest.approx([2., expected_d, expected_d])
    assert g_queries == pytest.approx([expected_d, expected_probe, expected_probe])
    assert int(od.state[d]['step']) == int(og.state[g]['step']) == 1
    assert rec.rows[od]['calls'] == rec.rows[og]['calls'] == 3
    assert rec.prediction_queries == rec.prediction_restores == 2
    assert rec.records[0]['g']['factor'] == 1


def test_zero_game_field_stays_still_without_a_parameter_pull():
    d = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    g = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .99))
    og = torch.optim.Adam([g], lr=.2, betas=(0., .99))
    rec = OpponentPredictionRecorder()
    ordinary = torch.optim.Adam.step
    before = torch.get_rng_state().clone()
    for phase in rec.phases(0, od, og, {}):
        d.grad = torch.zeros_like(d)
        rec.step(od, ordinary)
        g.grad = torch.zeros_like(g)
        rec.step(og, ordinary)
    assert torch.equal(d, torch.zeros_like(d))
    assert torch.equal(g, torch.zeros_like(g))
    assert torch.equal(torch.get_rng_state(), before)
    assert int(od.state[d]['step']) == int(og.state[g]['step']) == 1
    assert rec.records[0]['d']['rho'] == rec.records[0]['g']['rho'] == 0
    assert rec.prediction_records[0]['critic_step_norm'] == 0


def _host(context, task='mode_hold', steps=3):
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
    return result, _state_sha(recorder), policy.receipt(), recorder


@pytest.mark.parametrize('task', ['mode_hold', 'trajectory'])
def test_disabled_prediction_is_exact_original_models_moments_ema_rng_and_metrics(task):
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _host(pr84_smoothed_candidate(task=task), task)
    disabled = _host(pr84_opponent_prediction(task=task, prediction=False), task)
    assert original[:3] == disabled[:3]
    assert original[-1].records == disabled[-1].records
    assert disabled[-1].prediction_queries == disabled[-1].prediction_restores == 0


def test_active_host_has_no_new_draws_restores_d_and_freezes_g_operator():
    from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate
    original = _host(pr84_smoothed_candidate(), steps=2)
    seen = []
    real_step = OpponentPredictionRecorder.step

    def audited(self, optimizer, ordinary_step, closure=None):
        if optimizer is self.optimizers[0]:
            assert not self._probe_active
        elif self.phase in (1, 2):
            assert self._probe_active
            assert all(torch.equal(p, v) for p, v in
                       zip(self._params(self.optimizers[0]), self._probe_d))
            seen.append((self.outer_steps, self.phase, self._smooth_width,
                         [p.detach().clone() for p in self._params(self.optimizers[0])]))
        return real_step(self, optimizer, ordinary_step, closure)

    with patch.object(OpponentPredictionRecorder, 'step', audited):
        active = _host(pr84_opponent_prediction(), steps=2)
    assert original[2] == active[2]
    rec = active[-1]
    assert rec.prediction_queries == rec.prediction_restores == rec.rng_replay_verified == 4
    assert not rec._probe_active
    for base, proposal in zip(seen[::2], seen[1::2]):
        assert base[:3] == (proposal[0], 1, proposal[2])
        assert all(torch.equal(a, b) for a, b in zip(base[3], proposal[3]))
    for opt, row in rec.rows.items():
        assert row['calls'] == 6
        assert all(int(opt.state[p]['step']) == 2 for group in opt.param_groups for p in group['params'])


def test_prediction_context_restores_materialized_d_on_exception():
    d = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    g = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    od = torch.optim.Adam([d], lr=.1, betas=(0., .99))
    og = torch.optim.Adam([g], lr=.2, betas=(0., .99))
    with pytest.raises(RuntimeError, match='synthetic G forward failure'):
        with pr84_opponent_prediction() as (rec, _):
            for phase in rec.phases(0, od, og, {}):
                d.grad = g.detach().clone()
                rec.step(od, rec_context_ordinary_step)
                if phase == 1:
                    assert rec._probe_active
                    saved = [p.clone() for p in rec._materialized_d]
                    raise RuntimeError('synthetic G forward failure')
                g.grad = -d.detach().clone()
                rec.step(og, rec_context_ordinary_step)
    assert torch.equal(d, saved[0])
    assert not rec._probe_active


# Capture the unpatched ordinary optimizer implementation for the synthetic
# context test; the context itself deliberately patches torch.optim.Adam.step.
rec_context_ordinary_step = torch.optim.Adam.step
