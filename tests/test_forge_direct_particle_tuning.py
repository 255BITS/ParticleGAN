"""Existing direct-coordinate moment tuning, without GAN training."""
import importlib.util
from pathlib import Path

import pytest
import torch

from particlegan import Recipe
from particlegan.recipes import learning_rate_scale
from experiments.forge import configuration_search as search
from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.techniques import recipe_field_active, validate_same_technique
from test_forge_configuration_search import checkout, spec


def _direct_host(checkout):
    path = checkout / 'configs/forge/tasks/t1.json'
    task = read_json(path)
    task['execution'].update(host='two_pole', prior={
        'kind': 'particle_cloud', 'sigma': 0., 'standardize': False, 'learnable': True,
        'exception_reason': 'Fixture for existing direct generated-coordinate optimizer.'})
    task['evaluation'].update(sampling_law='learned_particles_and_critic_gradient',
                              eval_output_noise='not_applied_to_measurement')
    atomic_json(path, task)


def test_direct_beta_search_keeps_global_recipe_and_existing_configuration_identity(checkout, spec):
    _direct_host(checkout)
    spec['grid'] = {'direct_particle_betas': [[0., .99], [0., .999]]}
    original = {path: path.read_bytes() for path in checkout.rglob('*.json')}
    planned = search.plan_search(checkout, checkout / 'runs', spec)
    assert len(planned['trials']) == 2
    assert all(len(trial['tasks']) == 3 for trial in planned['trials'])
    assert len({trial['technique_signature']['digest'] for trial in planned['trials']}) == 1
    assert original == {path: path.read_bytes() for path in checkout.rglob('*.json')}
    paths = search.materialize_search(checkout, spec)
    assert {path.stem for path in paths} == {trial['candidate_id'] for trial in planned['trials']}
    for path in paths:
        card = read_json(path)
        search.validate_configuration_declaration(card, root=checkout)
        assert search.configuration_id(card, resolved_recipe=card['resolved_configuration_recipe']) == card['configuration_id']
        assert set(card['recipe_overrides']) == {'direct_particle_betas'}
    frozen = {path: path.read_bytes() for path in paths}
    assert search.materialize_search(checkout, spec) == paths
    assert frozen == {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize('value', [[False, .99], [0, True], [0, 1.], [0, float('nan')], [0], None])
def test_invalid_direct_beta_pairs_block_before_cards(checkout, spec, value):
    spec['grid'] = {'direct_particle_betas': [value]}
    with pytest.raises(ValueError, match='invalid hyperparameter value'):
        search.materialize_search(checkout, spec)
    assert not (checkout / 'configs/forge/configurations').exists()


@pytest.mark.parametrize('value', [[.1, .99], [0., 0.]])
def test_direct_moment_activation_changes_require_structural_idea(checkout, spec, value):
    _direct_host(checkout)
    spec['grid'] = {'direct_particle_betas': [value]}
    with pytest.raises(ValueError, match='direct_particle_moments'):
        search.materialize_search(checkout, spec)
    assert not (checkout / 'configs/forge/configurations').exists()


def test_direct_beta_axis_requires_an_actual_direct_optimizer(checkout, spec):
    recipe = Recipe()
    direct = {'adapter': 'transfer_behavior', 'id': 'two_pole', 'execution': {'host': 'two_pole'}}
    assert recipe_field_active('direct_particle_betas', recipe, task=direct)
    assert not recipe_field_active('direct_particle_betas', recipe, task={'adapter': 'word_joint'})
    plain = Recipe(optimizer_family='adam', reg_arm='a_r1r2', d_guard_ratio=0,
                   reg_anchor_weight=0, latent_damping_max_rate=0, direct_particle_gain=False)
    assert not recipe_field_active('direct_particle_betas', plain, task=direct)
    validate_same_technique(recipe, recipe.replace(direct_particle_betas=(0, .999)))
    spec['grid'] = {'direct_particle_betas': [[0., .99]]}
    with pytest.raises(ValueError, match='inactive or task-owned'):
        search.materialize_search(checkout, spec)
    assert not (checkout / 'configs/forge/configurations').exists()


def _analysis_module():
    path = Path(__file__).resolve().parents[1] / 'reports/forge/k3p-global-tier1-v3/direct-rate-analysis.py'
    spec = importlib.util.spec_from_file_location('direct_rate_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('beta2', [.9, .99, .999])
def test_zero_origin_displacement_bound_contains_public_optimizer_with_changing_gradients(beta2):
    bound = _analysis_module().displacement_bound
    recipe = Recipe(critic_formulation='k3p', lr=.0006, total_steps=80,
                    direct_particle_betas=(0, beta2), network_lr_horizon_cap=None)
    particles = torch.nn.Parameter(torch.zeros(12, 1, dtype=torch.float64))
    optimizer = recipe.make_generator_optimizer([particles], direct_particles=[particles])
    # Synthetic externally supplied gradients cover decay, sign reversals and
    # late spikes. No model, loss, data sampling or qualification is executed.
    pattern = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(12, 1)
    for step in range(1, 81):
        optimizer.param_groups[0]['lr'] = recipe.lr * learning_rate_scale(step - 1, 80, .6, .05)
        magnitude = 1e-4 if step % 9 < 6 else 10.
        particles.grad = pattern * magnitude * (-1 if step % 13 == 0 else 1)
        optimizer.step()
        maximum = bound(lr=recipe.lr, beta2=beta2, steps=step, horizon=80, start=.6, floor=.05)
        assert float(particles.detach().abs().max()) <= maximum + 1e-12
    if beta2 == .9:
        assert maximum == pytest.approx(.22666346, abs=1e-8)
        assert maximum < .3
