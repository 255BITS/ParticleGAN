"""Small public-host parity checks; no scientific horizon run is executed."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import pytest
import torch

from experiments.forge.behavior_adapters import BehaviorComponents, run_behavior
from experiments.forge.two_pole_observer import _snapshot, diagnostic_contract
from experiments.forge.views import load_tasks


ROOT = Path(__file__).resolve().parents[1]


def task(steps=8, horizon=4):
    value = deepcopy(load_tasks(ROOT)['two_pole'])
    value['id'] = 'two-pole-horizon-unit'
    value['execution'].update(steps=steps, original_schedule_horizon=horizon,
        horizon_diagnostic={'schema_version': 1, 'kind': 'two_pole_force_v1',
                            'prefix_horizon': 4, 'checkpoints': [4, steps]})
    return value


def request(noisy=False):
    return {'protocol': {'seed': 0}, 'candidate': {'recipe_preset': 'k3p', 'recipe_overrides': {
        'lr': .0012, 'reg_coeff': 170., 'network_lr_horizon_cap': None,
        'input_noise_std': .5 if noisy else 0., 'output_noise_std': .02 if noisy else 0.}}}


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def saved_state(components):
    public = components.optimizers['generator'].optimizers[0]
    particles = components.role_parameters['prior'][0]
    return _snapshot({'positions': particles, 'critic': components.models['discriminator'].state_dict(),
        'generator_optimizer': public.state_dict(), 'critic_optimizer': components.optimizers['discriminator'].state_dict(),
        'streams': components.context.streams.state_dict(), 'torch_rng_state': torch.get_rng_state()})


@pytest.mark.parametrize('noisy', [False, True])
def test_observer_preserves_complete_ordinary_prefix_state_and_rng(tmp_path, monkeypatch, noisy):
    ordinary = deepcopy(load_tasks(ROOT)['two_pole'])
    ordinary['execution']['steps'] = 4
    states = {}
    original_bind, original_checkpoint = BehaviorComponents.bind, BehaviorComponents.checkpoint

    def bind(self, **kwargs):
        result = original_bind(self, **kwargs)
        states[(self.task['id'], 0)] = saved_state(self)
        return result

    def checkpoint(self, step, measure):
        original_checkpoint(self, step, measure)
        if step == 4:
            states[(self.task['id'], step)] = saved_state(self)

    monkeypatch.setattr(BehaviorComponents, 'bind', bind)
    monkeypatch.setattr(BehaviorComponents, 'checkpoint', checkpoint)
    before = torch.get_rng_state().clone()
    baseline = run_behavior(request(noisy), ordinary, tmp_path / 'ordinary')
    extended = run_behavior(request(noisy), task(), tmp_path / 'extended')
    assert torch.equal(before, torch.get_rng_state())
    assert_equal(states[('two_pole', 0)], states[('two-pole-horizon-unit', 0)])
    assert_equal(states[('two_pole', 4)], states[('two-pole-horizon-unit', 4)])
    checkpoints = torch.load(tmp_path / 'extended/diagnostic-checkpoints.pt', weights_only=False)
    assert_equal(checkpoints['initial'], states[('two_pole', 0)])
    assert_equal(checkpoints['checkpoints'][4], states[('two_pole', 4)])
    assert extended['evidence']['diagnostic_observations'][:4] == baseline['evidence']['observations']
    assert 'horizon_diagnostic' not in baseline['evidence']
    assert not (tmp_path / 'ordinary/force-trace.pt').exists()
    assert extended['evidence']['guards']['optimizer_updates'] == {'prior': 8, 'discriminator': 8}


def test_observed_gradients_rates_displacements_and_artifact_bindings(tmp_path):
    result = run_behavior(request(), task(horizon=8), tmp_path)
    receipt = result['evidence']['horizon_diagnostic']
    assert receipt['execution_updates'] == receipt['schedule_horizon'] == 8
    assert receipt['optimizer_updates_added'] == receipt['sampling_draws_added'] == 0
    assert [row['step'] for row in receipt['milestones']] == [4, 8]
    assert [row['step'] for row in result['evidence']['observations']] == list(range(1, 9))
    records = torch.load(tmp_path / 'force-trace.pt', weights_only=False)
    for row in records:
        assert_equal(row['displacement'], row['positions'] - row['positions_before'])
        assert_equal(row['particle_l2_gradient'], 2 * .02 * row['positions_before'] / 12)
        assert_equal(row['adversarial_gradient'], row['total_gradient'] - row['particle_l2_gradient'])
        assert_equal(row['critic_payoff_parameter_gradient'],
                     row['critic_total_parameter_gradient'] - row['critic_penalty_parameter_gradient'])
        assert row['actual_generator_lr'] == row['scheduled_generator_lr'] * row['direct_gain']
        assert 1 <= row['direct_gain'] <= 2
        assert row['actual_generator_betas'] == [.0, .9]
        if row['step'] in (4, 8):
            assert row['critic_diagnostics']['particle_input_gradients'].shape == (12, 1)
    for artifact in receipt['artifacts']:
        path = tmp_path / artifact['path']
        assert artifact['bytes'] == path.stat().st_size
        assert artifact['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['applied']['recipe']['total_steps'] == 8
    assert result['applied']['field_ownership']['recipe_fields']['total_steps']['source'] == 'task.execution.original_schedule_horizon'


def test_declared_800_horizons_keep_exact_prefix_and_terminal_observation_contracts():
    for horizon in (80, 800):
        declared = task(800, horizon)
        declared['execution']['horizon_diagnostic'].update(prefix_horizon=80, checkpoints=[80, 200, 400, 800])
        components = BehaviorComponents(request(), declared)
        assert components.recipe.total_steps == horizon
        assert diagnostic_contract(declared)['checkpoints'] == [80, 200, 400, 800]
        prefix = {math.ceil(i * 80 / 24) for i in range(1, 25)}
        assert len(prefix) == 24 and 80 in prefix
        terminal = {math.ceil(i * 800 / 24) for i in range(1, 25)}
        assert len(terminal) == 24 and sorted(terminal)[-5:] == [667, 700, 734, 767, 800]


@pytest.mark.parametrize('mutate', [
    lambda value: value.update(id='two_pole'),
    lambda value: value['execution'].update(host='unused_token_hold'),
    lambda value: value['execution'].update(original_schedule_horizon=True),
    lambda value: value['execution'].update(original_schedule_horizon=6),
    lambda value: value['execution']['horizon_diagnostic'].update(checkpoints=[8, 4]),
    lambda value: value['execution']['horizon_diagnostic'].update(checkpoints=[4]),
    lambda value: value['execution']['horizon_diagnostic'].update(checkpoints=[4, 4, 8]),
    lambda value: value['execution'].pop('horizon_diagnostic'),
])
def test_invalid_or_unscoped_observer_blocks_before_execution(tmp_path, mutate):
    declared = task()
    mutate(declared)
    with pytest.raises(ValueError):
        run_behavior(request(), declared, tmp_path / 'blocked')
    assert not (tmp_path / 'blocked').exists()
