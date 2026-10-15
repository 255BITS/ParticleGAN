"""Bounded arithmetic/routing/state contracts, not quality experiments."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.component_history import ComponentHistoryDirectionBlendOptimizer
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.direction_blend import DirectionBlendOptimizer
from particlegan.optim.gradient_signal import GradientSignal


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def vector_optimizer(mode='adversarial', **kwargs):
    parameter = nn.Parameter(torch.tensor([.2, -.3], dtype=torch.float64))
    optimizer = ComponentHistoryDirectionBlendOptimizer([parameter], generator_history_mode=mode,
                                                        lr=.1, **kwargs)
    return parameter, optimizer


def linear_step(parameter, optimizer, force, residual=(0., 1.)):
    optimizer.zero_grad(set_to_none=True)
    adversarial = (parameter * parameter.new_tensor(force)).sum()
    other = (parameter * parameter.new_tensor(residual)).sum()
    constraint_geometry_backward(adversarial + other, optimizer, (adversarial,),
                                 adversarial_loss=adversarial)
    gradient = parameter.grad.clone()
    normals = optimizer._protected.clone()
    optimizer.step()
    return gradient, normals


def test_recovered_estimator_matches_frozen_source_receipts_without_rng_or_gradient_mutation():
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    groups = [dict(params=[parameter], role='generator')]
    signal = GradientSignal(('generator',))
    # Receipts from exact frozen gradient_signal.py SHA b7fd16b80604e7...
    expected = [(1., 1., 4., 0.),
                (0., 1.991150442477876, 0., 2.4999999999999996),
                (0., 2.9649122807017543, 0., 2.5946745562130173),
                (0., 3.9133446205941205, 0., 2.5594857831072946)]
    rng = torch.get_rng_state().clone()
    for force, receipt in zip(([2., 0.], [0., 1.], [-1., 0.], [2., 0.]), expected):
        parameter.grad = torch.tensor(force, dtype=parameter.dtype)
        before = parameter.grad.clone()
        signal.scale(groups)
        last = signal.packets['generator']['last']
        assert tuple(last.values()) == receipt
        assert torch.equal(parameter.grad, before)
        signal.validate(signal.state_dict(), groups)
    assert torch.equal(torch.get_rng_state(), rng)
    assert signal.packets['generator']['count'] == 4


def test_explicit_channel_uses_same_estimator_and_retains_total_gradients():
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    groups = [dict(params=[parameter], role='generator')]
    explicit, ordinary = GradientSignal(('generator',)), GradientSignal(('generator',))
    for force in ([1., 2.], [2., 3.], [.5, -1.]):
        gradient = torch.tensor(force, dtype=parameter.dtype)
        parameter.grad = gradient
        ordinary.scale(groups)
        parameter.grad = torch.full_like(gradient, 7.)
        explicit.scale(groups, gradients={'generator': [gradient]})
        assert torch.equal(parameter.grad, torch.full_like(gradient, 7.))
        equal(ordinary.state_dict(), explicit.state_dict())


def test_selective_gradient_keeps_residual_fresh_and_protected_normals_raw():
    parameter, optimizer = vector_optimizer()
    linear_step(parameter, optimizer, (1., 0.))
    gradient, normals = linear_step(parameter, optimizer, (-1., 0.))
    assert optimizer.gradient_signal.packets['generator']['last']['rho'] == 0.
    assert torch.equal(gradient, torch.tensor([0., 1.], dtype=parameter.dtype))
    assert torch.equal(normals, torch.tensor([[-1., 0.]], dtype=parameter.dtype))
    assert optimizer.state[parameter]['step'] == 2
    assert optimizer.constraint_geometry_stats['steps'] == 2


def test_only_effective_generator_role_changes_including_direct_particles():
    generator = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    encoder = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    prior = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    direct = nn.Parameter(torch.eye(2, dtype=torch.float64))
    recipe = get_recipe('bcap', generator_history_mode='adversarial')
    optimizer = recipe.make_generator_optimizer([
        dict(params=[generator], role='generator'), dict(params=[encoder], role='encoder'),
        dict(params=[prior], role='prior'), dict(params=[direct], forge_role='prior')],
        latent_table=prior, direct_particles=[direct])
    assert [g['role'] for g in optimizer.param_groups] == ['generator', 'encoder', 'prior', 'generator']
    for sign in (1., -1.):
        optimizer.zero_grad(set_to_none=True)
        optimizer.set_sampled_rows(prior, torch.tensor([0, 2]))
        adversarial = sign * (generator.sum() + direct.sum() + encoder.sum() + prior.sum())
        other = 2 * generator.sum() + 2 * direct.sum() + 3 * encoder.sum() + 4 * prior.sum()
        constraint_geometry_backward(adversarial + other, optimizer, (adversarial,),
                                     adversarial_loss=adversarial)
        if sign == -1.:
            assert optimizer.gradient_signal.packets['generator']['last']['rho'] == 0.
            assert torch.equal(generator.grad, torch.full_like(generator, 2.))
            assert torch.equal(direct.grad, torch.full_like(direct, 2.))
            assert torch.equal(encoder.grad, torch.full_like(encoder, 2.))
            assert torch.equal(prior.grad, torch.full_like(prior, 3.))
            assert len(optimizer.gradient_signal.packets['generator']['means']) == 2
        optimizer.step()
    assert torch.equal(prior[1], torch.zeros_like(prior[1]))
    assert all(optimizer.state[p]['step'] == 2 for p in (generator, encoder, prior, direct))


def test_actual_float32_update_distinguishes_selective_from_whole_normalization_control():
    def build(mode):
        p = nn.Parameter(torch.tensor([.1234567, -.2345678]))
        if mode == 'none':
            opt = DirectionBlendOptimizer([p], lr=.01, smoothing=0., eps=1e-8)
        else:
            opt = ComponentHistoryDirectionBlendOptimizer([p], generator_history_mode=mode,
                                                           lr=.01, smoothing=0., eps=1e-8)
        return p, opt
    pairs = [build(mode) for mode in ('none', 'whole', 'adversarial')]
    for parameter, optimizer in pairs:
        linear_step(parameter, optimizer, (1., 0.))
    displacements = []
    for parameter, optimizer in pairs:
        old = parameter.detach().clone()
        # Strong consistent residual makes whole-gradient rho positive.
        linear_step(parameter, optimizer, (-1., 0.), residual=(0., 10.))
        displacements.append(parameter.detach() - old)
    assert pairs[1][1].gradient_signal.packets['generator']['last']['rho'] > 0.
    assert pairs[2][1].gradient_signal.packets['generator']['last']['rho'] == 0.
    torch.testing.assert_close(displacements[0], displacements[1], rtol=0., atol=2e-8)
    assert not torch.equal(displacements[0], displacements[2])
    assert displacements[2][0].item() == 0.
    assert displacements[2][1].item() < 0.


@pytest.mark.parametrize('mode', ['adversarial', 'whole'])
def test_pending_backward_checkpoint_resumes_without_advancing_history_twice(mode):
    parameter, optimizer = vector_optimizer(mode)
    linear_step(parameter, optimizer, (1., 0.))
    optimizer.zero_grad(set_to_none=True)
    adversarial = -parameter[0]
    constraint_geometry_backward(adversarial + parameter[1], optimizer, (adversarial,),
                                 adversarial_loss=adversarial)
    checkpoint = deepcopy(optimizer.state_dict())
    restored_parameter, restored = vector_optimizer(mode)
    with torch.no_grad():
        restored_parameter.copy_(parameter)
    restored_parameter.grad = parameter.grad.clone()
    restored.load_state_dict(checkpoint)
    optimizer.step()
    restored.step()
    equal(optimizer.state_dict(), restored.state_dict())
    assert torch.equal(parameter, restored_parameter)
    assert restored.gradient_signal.packets['generator']['count'] == 2
    with pytest.raises(ValueError, match='component-history backward'):
        restored.step()


def trainer(mode='none'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = get_recipe('bcap', z_dim=2, num_particles=8, batch_size=4, total_steps=6,
                            prior_kind='mog', standardize=False, generator_history_mode=mode,
                            kinetic_transport_weight=1., kinetic_transport_local_weight=1.)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        critic = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for model in (generator, critic, prior):
            deterministic_orthogonal_(model, seed=0)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0)


@pytest.mark.parametrize('mode', ['adversarial', 'whole'])
def test_public_trainer_exact_resume_and_first_step_disabled_parity(mode):
    candidate, baseline = trainer(mode), trainer()
    batch = torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]])
    baseline.step(batch)
    candidate.step(batch)
    # One observation has zero variance; the archived denominator epsilon can
    # make rho infinitesimally below one without changing this float32 update.
    equal(candidate.G.state_dict(), baseline.G.state_dict())
    equal(candidate.D.state_dict(), baseline.D.state_dict())
    assert candidate.opt_g.gradient_signal.packets['generator']['last']['rho'] == pytest.approx(1., abs=1e-12)
    candidate.step(batch)
    checkpoint = candidate.state_dict()
    expected_metrics = candidate.step(batch)
    expected = candidate.state_dict()
    restored = trainer(mode)
    restored.load_state_dict(checkpoint)
    equal(expected_metrics, restored.step(batch))
    equal(expected, restored.state_dict())
    assert restored.opt_g.gradient_signal.packets['generator']['count'] == 3
    assert 'component_history' not in baseline.opt_g.state_dict()
    assert 'generator_history_mode' not in baseline.recipe.to_dict()
    assert not hasattr(baseline.opt_d, 'gradient_signal')


@pytest.mark.parametrize('mutation', ['mode', 'placement', 'roles', 'nan', 'weight', 'clock', 'shape', 'receipt', 'pending'])
def test_malformed_history_checkpoint_rejected_before_mutation(mutation):
    parameter, optimizer = vector_optimizer()
    linear_step(parameter, optimizer, (1., 0.))
    before = deepcopy(optimizer.state_dict())
    saved = deepcopy(before)
    meta = saved['component_history']
    packet = meta['signal']['packets']['generator']
    if mutation == 'mode':
        meta['mode'] = 'whole'
    elif mutation == 'placement':
        meta['placement'] = 'after_normalization'
    elif mutation == 'roles':
        meta['signal']['roles'] = ['encoder']
    elif mutation == 'nan':
        packet['means'][0][0] = float('nan')
    elif mutation == 'weight':
        packet['weight'] = .9
    elif mutation == 'clock':
        saved['constraint_geometry']['stats']['steps'] = 7
    elif mutation == 'shape':
        packet['means'][0] = torch.zeros(3, dtype=torch.float64)
    elif mutation == 'receipt':
        packet['last']['other'] = 1.
    elif mutation == 'pending':
        meta['pending'] = True
    with pytest.raises(ValueError):
        optimizer.load_state_dict(saved)
    equal(before, optimizer.state_dict())


def test_active_missing_explicit_channel_and_nonfinite_inputs_fail_closed():
    parameter, optimizer = vector_optimizer()
    loss = parameter.sum()
    with pytest.raises(ValueError, match='explicit current adversarial'):
        constraint_geometry_backward(loss, optimizer, (loss,))
    assert parameter.grad is None
    assert not optimizer.gradient_signal.packets
    parameter, optimizer = vector_optimizer()
    adversarial = parameter.sum() * float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        constraint_geometry_backward(adversarial, optimizer, (parameter.sum(),), adversarial_loss=adversarial)
    assert not optimizer.gradient_signal.packets


def test_partially_failed_step_blocks_reuse_and_export_until_explicit_valid_restore(monkeypatch):
    from particlegan.optim.dualnorm import NormalizedOptimizer
    parameter, optimizer = vector_optimizer()
    linear_step(parameter, optimizer, (1., 0.))
    saved_parameter = parameter.detach().clone()
    checkpoint = deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    loss = -parameter[0] + parameter[1]
    constraint_geometry_backward(loss, optimizer, (-parameter[0],), adversarial_loss=-parameter[0])
    original_step = NormalizedOptimizer.step

    @torch.no_grad()
    def partial_failure(self, closure=None):
        first = self.param_groups[0]['params'][0]
        first.add_(123.)
        self.state[first]['step'] += 1
        raise RuntimeError('injected numerical failure after parameter mutation')

    monkeypatch.setattr(NormalizedOptimizer, 'step', partial_failure)
    with pytest.raises(RuntimeError, match='injected numerical failure'):
        optimizer.step()
    assert not torch.equal(parameter, saved_parameter)
    for call in (optimizer.step, optimizer.state_dict,
                 lambda: optimizer.backward_with_component_history(loss, adversarial_loss=loss),
                 lambda: optimizer.bind_protected_losses((loss,))):
        with pytest.raises(RuntimeError, match='restore a valid checkpoint'):
            call()
    malformed = deepcopy(checkpoint)
    malformed['component_history']['mode'] = 'whole'
    with pytest.raises(ValueError):
        optimizer.load_state_dict(malformed)
    with pytest.raises(RuntimeError, match='restore a valid checkpoint'):
        optimizer.state_dict()
    monkeypatch.setattr(NormalizedOptimizer, 'step', original_step)
    # Optimizer restore does not restore model weights; an application must
    # restore both members of its saved full checkpoint before continuing.
    with torch.no_grad():
        parameter.copy_(saved_parameter)
    optimizer.load_state_dict(checkpoint)
    equal(checkpoint, optimizer.state_dict())
    reference_parameter, reference = vector_optimizer()
    with torch.no_grad():
        reference_parameter.copy_(saved_parameter)
    reference.load_state_dict(checkpoint)
    linear_step(parameter, optimizer, (-1., 0.))
    linear_step(reference_parameter, reference, (-1., 0.))
    assert torch.equal(parameter, reference_parameter)
    equal(optimizer.state_dict(), reference.state_dict())


@pytest.mark.parametrize('settings', [dict(generator_history_mode='unknown'),
                                     dict(generator_history_mode='whole', constraint_geometry_mode='none'),
                                     dict(generator_history_mode='adversarial', prior_proximal_strength=1.)])
def test_unsupported_recipe_compositions_reject(settings):
    with pytest.raises(ValueError):
        get_recipe('bcap', **settings)


@pytest.mark.parametrize('host', ['two_pole', 'trajectory', 'ae_gan_hold', 'unused_token_hold',
                                  'mid_scale_identity', 'unipolar', 'residual_student', 'cover_leftover'])
@pytest.mark.parametrize('mode', ['adversarial', 'whole'])
def test_all_public_component_hosts_route_current_channel_and_one_joint_step(tmp_path, host, mode):
    from experiments.forge.behavior_adapters import run_behavior
    from experiments.forge.views import load_tasks
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root / 'configs/forge/configurations' /
        'bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json').read_text())
    candidate['recipe_overrides'].update(generator_history_mode=mode,
        constraint_geometry_mode='direction_blend', kinetic_transport_weight=1., kinetic_transport_local_weight=1.)
    task = deepcopy(load_tasks(root)[host])
    task['execution'].update(steps=2, transport_consumer='output_marginal_v1')
    task['task_cohort'] = 'component_output_transport_v1'
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_behavior(dict(protocol=dict(seed=0), candidate=candidate), task, tmp_path)
    finally:
        torch.set_num_threads(threads)
    guards = result['evidence']['guards']
    assert guards['all_finite'] and guards['hooks_exercised']
    assert guards['unintended_rng_deviations'] == 0
    assert all(count == 2 for count in guards['optimizer_updates'].values())
    saved = torch.load(tmp_path / 'component-state.pt', weights_only=False)
    joint, = saved['optimizers']['generator']
    signal = joint['component_history']['signal']
    assert signal['roles'] == ['generator']
    assert signal['packets']['generator']['count'] == 2
    assert not joint['component_history']['pending']
    assert joint['constraint_geometry']['stats']['steps'] == 2
    assert len(signal['packets']['generator']['means']) == sum(
        len(group['params']) for group in joint['param_groups'] if group['role'] == 'generator')
    if host == 'ae_gan_hold':
        assert any(group['role'] == 'encoder' for group in joint['param_groups'])
    if host == 'two_pole':
        assert all(group['role'] == 'generator' for group in joint['param_groups'])


@pytest.mark.parametrize('mode', ['adversarial', 'whole'])
def test_word_history_excludes_encoder_and_replays_full_joint_checkpoint(mode):
    from benchmarks.toy_audit.api_images import WordFixture
    from experiments.forge.api import FormulationContext
    from experiments.forge.state import state_digest

    def build():
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root / 'configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json').read_text())
        context = FormulationContext(recipe_preset='bcap',
            recipe_overrides=dict(base['recipe_overrides'], num_particles=5, z_dim=2, batch_size=256, total_steps=20000,
                generator_history_mode=mode, kinetic_transport_weight=1., kinetic_transport_local_weight=1.),
            prior=dict(kind='particle_cloud', sigma=0., standardize=False, learnable=True,
                       exception_reason='software check of the original finite vocabulary'),
            seed=0, device='cpu', initializer='deterministic_orthogonal', execution_path='public_components',
            component_transport='output_marginal_v1')
        return WordFixture(device='cpu', seed=0, recipe_name=None, max_steps=3, components=context), context

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture, context = build()
        assert [group['role'] for group in fixture.opt_g.param_groups] == ['generator', 'encoder', 'prior']
        assert not fixture.opt_g.gradient_signal.packets
        fixture.step()
        saved = deepcopy(fixture.state_dict())
        streams = deepcopy(context.streams.state_dict())
        expected = fixture.step()
        resumed, resumed_context = build()
        resumed_context.streams.load_state_dict(streams)
        resumed.policy.load_state_dict(saved['api_state'])
        resumed.data_generator.set_state(saved['data_generator'])
        resumed.restore_component_transport(saved.get('component_transport'))
        assert state_digest(expected) == state_digest(resumed.step())
        assert state_digest(fixture.state_dict()) == state_digest(resumed.state_dict())
        assert state_digest(context.streams.state_dict()) == state_digest(resumed_context.streams.state_dict())
        packet = resumed.opt_g.gradient_signal.packets['generator']
        assert packet['count'] == 2
        assert len(packet['means']) == len(list(resumed.G.parameters()))
    finally:
        torch.set_num_threads(threads)
