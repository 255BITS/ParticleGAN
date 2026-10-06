"""Branch-only software contract tests; no numerical qualification training."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.forge import atlas_two_pole, noisy_prior_adapters, noisy_prior_tier1
from experiments.forge.adapters import adapter_preflight
from experiments.forge.api import CapabilityError, FormulationContext, task_formulation_context
from experiments.forge.priors import resolve_prior

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = ('gaussian1d_acquisition', 'two_pole', 'ring16_acquisition')


def parent(name):
    return json.loads((ROOT / 'configs/forge/tasks' / (name + '.json')).read_text())


def variant(name):
    return noisy_prior_tier1.make_variant(parent(name))


def candidate():
    return dict(recipe_preset='atlas', recipe_overrides={},
                requires_capabilities=['particle_cloud', 'a2', 'named_rng', 'policy_controls', 'policy_serving'])


def protocol():
    return json.loads((ROOT / noisy_prior_tier1.PROTOCOL_PATH).read_text())


@pytest.mark.parametrize('name', SUPPORTED)
def test_full_context_and_preflight_bind_fixed_task_without_models(name, monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight constructed a network')
    monkeypatch.setattr(torch.nn.Linear, '__init__', forbidden)
    task = variant(name)
    context = task_formulation_context(candidate(), task, protocol(), root=ROOT)
    full = asdict(context.recipe)
    assert len(full) == 79 and full['total_steps'] is None
    assert full['prior_kind'] == 'noisy_particles'
    assert full['continuous_policy'] == 'dv12' and full['row_policy'] == 'independent'
    assert full['particle_birth_death'] and full['row_evidence_gate']
    assert context.capabilities()['live_sampling'] is True
    assert context.capabilities()['served_sampling'] is False
    assert context.prior_config == resolve_prior(task['execution']['prior'], explicit=True)
    assert context._trainer is None and context._policy is None
    assert adapter_preflight(task, candidate(), root=ROOT) == []
    from particlegan import get_recipe
    reference = asdict(get_recipe('atlas', **noisy_prior_adapters.task_resources(task),
                                 sigma_rel=0., standardize=False))
    reference['prior_kind'] = 'noisy_particles'
    assert full == reference


@pytest.mark.parametrize('name', ('unused_token_hold', 'five_word_joint_acquisition'))
def test_unimplemented_owner_is_blocked_before_any_constructor(name, monkeypatch):
    import torch
    monkeypatch.setattr(torch.nn.Linear, '__init__', lambda *a, **k: pytest.fail('blocked owner constructed a model'))
    reasons = adapter_preflight(variant(name), candidate(), root=ROOT)
    assert reasons
    with pytest.raises(CapabilityError):
        task_formulation_context(candidate(), variant(name), protocol(), root=ROOT)


@pytest.mark.parametrize('field,value', [('sigma', .03), ('standardize', True), ('learnable', False)])
def test_fixed_prior_law_mutation_refused(field, value):
    task = variant('ring16_acquisition')
    task['execution']['prior'][field] = value
    with pytest.raises(ValueError):
        noisy_prior_tier1.validate(task)
    assert adapter_preflight(task, candidate(), root=ROOT)


@pytest.mark.parametrize('change', [
    {'recipe_overrides': {'lr': .01}}, {'recipe_overrides': {'total_steps': 80}},
    {'host_adaptation': {'schema_version': 1, 'recipe_fields': ['z_dim']}, 'recipe_overrides': {'z_dim': 1}},
    {'recipe_preset': 'bcap'}, {'extensions': {'latent_damping_max_rate': .2}},
    {'initializer': 'supplied'},
])
def test_new_cohort_is_not_a_blanket_recipe_or_adaptation_waiver(change):
    spec = candidate(); spec.update(change)
    task = variant('two_pole')
    assert adapter_preflight(task, spec, root=ROOT)
    with pytest.raises(CapabilityError):
        task_formulation_context(spec, task, protocol(), root=ROOT)


def test_mutated_gate_horizon_source_and_missing_cohort_are_refused():
    for key in ('gate', 'horizon', 'source', 'cohort'):
        task = variant('two_pole')
        if key == 'gate': task['evaluation']['thresholds'][0][2] = .2
        elif key == 'horizon': task['execution']['steps'] = 79
        elif key == 'source': task['evaluation']['sources']['benchmarks/transfer_suite/protocol.py'] = '0' * 64
        else: task.pop('task_cohort')
        assert adapter_preflight(task, candidate(), root=ROOT)
        assert noisy_prior_adapters.validate_evidence(task, {})['status'] == 'INVALID'


def test_direct_public_context_cannot_change_width_or_omit_fixed_resources():
    task = variant('ring16_acquisition')
    kwargs = dict(recipe_preset='atlas', policy_task=task, prior=task['execution']['prior'])
    with pytest.raises(CapabilityError, match='fixed task resources'):
        FormulationContext(**kwargs)
    prior = deepcopy(task['execution']['prior']); prior['sigma'] = .03
    with pytest.raises(CapabilityError, match='fixed parent-width'):
        FormulationContext(**{**kwargs, 'prior': prior}, recipe_overrides=noisy_prior_adapters.task_resources(task))


def test_noisy_two_pole_full_recipe_changes_only_prior_kind():
    task = variant('two_pole')
    binding = atlas_two_pole.resolve_binding(ROOT, candidate(), task, protocol())
    ordinary = atlas_two_pole.resolve_binding(ROOT, candidate(), parent('two_pole'), protocol())
    expected = deepcopy(ordinary['recipe']); expected['prior_kind'] = 'noisy_particles'
    assert binding['recipe'] == expected
    assert atlas_two_pole.digest(expected) == atlas_two_pole.NOISY_EFFECTIVE_RECIPE_SHA256
    assert binding['source_contract']['observation'] == ordinary['source_contract']['observation']
    assert binding['source_contract']['objective'] == ordinary['source_contract']['objective']
    assert binding['source_contract']['prior_type_substitution']['sampler_calls'] == 0
    assert binding['source_contract']['prior_type_substitution']['constructor_rng_draws'] == 0
    assert binding['source_contract']['prior_type_substitution']['same_table_alias'] is True


def test_guard_refuses_factory_before_model_construction(monkeypatch):
    import torch
    task = variant('two_pole')
    binding = atlas_two_pole.resolve_binding(ROOT, candidate(), task, protocol())
    monkeypatch.setattr(torch.nn.Linear, '__init__', lambda *a, **k: pytest.fail('guard ran too late'))
    calls = []
    def denied():
        calls.append('guard')
        raise ValueError('no admitted Source')
    with pytest.raises(ValueError, match='no admitted Source'):
        atlas_two_pole.construct_owner(ROOT, binding, source_guard=denied)
    assert calls == ['guard']


def test_live_restore_uses_same_actual_prior_and_ready_owner():
    import torch
    from particlegan.noisy_particle_prior import NoisyParticlePrior
    prior = NoisyParticlePrior.from_table(torch.nn.Parameter(torch.zeros(4, 2)), sigma=.025)
    calls = []
    policy = SimpleNamespace(prior=prior, table=prior.z, _phase='ready', completed_steps=1, _fast=[object()])
    def release():
        calls.append('release'); policy._fast = None
    policy._serve_release = release
    trainer = SimpleNamespace(prior=prior, policy=policy, completed_steps=1)
    noisy_prior_adapters.restore_live_owner(trainer)
    assert calls == ['release'] and trainer.prior.z is policy.table
    policy._phase = 'updating'
    with pytest.raises(ValueError): noisy_prior_adapters.restore_live_owner(trainer)
    assert calls == ['release']


def evidence_for(task):
    steps = task['execution']['steps']
    import math
    clocks = [math.ceil(i * steps / 24) for i in range(1, 25)]
    kernel = dict(kind='noisy_particle_cloud', code_path='particlegan.noisy_particle_prior.NoisyParticlePrior',
                  sigma=.02500000037252903 if task['execution']['prior']['sigma'] else 0.,
                  sigma_units='raw_latent_coordinates', standardize=False, learned_width=False,
                  row_weights='uniform', zero_sigma_consumes_no_kernel_rng=True)
    return dict(policy_controls=dict(cohort=task['task_cohort'], completed_steps=steps,
        implementation_observed=True, requested_owners_bound=True, lifecycle={'complete': True}),
        policy_observations=[dict(task_id=task['id'], parent_task_id=task['prior_substitution_parent']['task'],
            completed_steps=clock, observed=True, table_alias_preserved=True, policy_owner='particlegan.UpdatePolicy',
            weights='live', sampling_law=task['evaluation']['sampling_law'],
            eval_output_noise=task['evaluation']['eval_output_noise'], sampling_calls_added_by_receipt=0,
            prior=deepcopy(kernel)) for clock in clocks],
        policy_purity=[dict(completed_steps=clock, pure=True, before_sha256='a'*64, after_sha256='a'*64)
                       for clock in clocks])


@pytest.mark.parametrize('name', SUPPORTED)
def test_original_reader_join_accepts_float32_width_without_renaming_it(name):
    task = variant(name); evidence = evidence_for(task)
    assert noisy_prior_adapters.validate_evidence(task, evidence) is None
    if name != 'two_pole':
        assert evidence['policy_observations'][0]['prior']['sigma'] != .025


@pytest.mark.parametrize('mutation', ['selected', 'width', 'extra_draw', 'clock', 'purity', 'malformed'])
def test_observation_guard_refuses_law_clock_purity_and_malformed_evidence(mutation):
    task = variant('ring16_acquisition'); evidence = evidence_for(task)
    if mutation == 'selected': evidence['policy_observations'][0]['weights'] = 'state_selected'
    elif mutation == 'width': evidence['policy_observations'][0]['prior']['sigma'] = .03
    elif mutation == 'extra_draw': evidence['policy_observations'][0]['sampling_calls_added_by_receipt'] = 1
    elif mutation == 'clock': evidence['policy_observations'].pop()
    elif mutation == 'purity': evidence['policy_purity'][0]['after_sha256'] = 'b'*64
    else: evidence['policy_observations'] = [None]
    result = noisy_prior_adapters.validate_evidence(task, evidence)
    assert result['status'] in ('INVALID', 'INCOMPLETE')


def test_catalog_support_carries_actual_variants_parents_view_and_protocol():
    files = set()
    for name in noisy_prior_tier1.PARENTS:
        task = variant(name)
        files.update(noisy_prior_adapters.supporting_source_paths(task))
        assert 'configs/forge/tasks/' + name + '.json' in files
        assert str(noisy_prior_tier1.VARIANT_DIRECTORY / (task['id'] + '.json')) in files
    assert str(noisy_prior_tier1.VIEW_PATH) in files
    assert str(noisy_prior_tier1.PROTOCOL_PATH) in files
    assert noisy_prior_adapters.supporting_source_paths(parent('two_pole')) == ()


def test_ordinary_bcap_and_original_atlas_routes_are_not_replaced():
    ordinary = parent('two_pole')
    bcap = dict(recipe_preset='bcap', recipe_overrides={})
    context = task_formulation_context(bcap, ordinary, root=ROOT)
    assert context.recipe.total_steps == 80 and context._noisy_task is False
    atlas = task_formulation_context(candidate(), ordinary, protocol(), root=ROOT)
    assert atlas._ordinary_two_pole and not atlas._noisy_task
    assert atlas.recipe.prior_kind == 'particles'
    assert atlas.ordinary_two_pole_binding['recipe']['total_steps'] is None



def test_ae_context_and_preflight_delegate_current79_without_model_construction(monkeypatch):
    import torch
    from experiments.forge import atlas_noisy_ae
    monkeypatch.setattr(torch.nn.Linear, '__init__', lambda *a, **k: pytest.fail('AE metadata constructed a model'))
    task = variant('ae_gan_hold')
    context = task_formulation_context(candidate(), task, protocol(), root=ROOT)
    full = asdict(context.recipe)
    assert atlas_noisy_ae.canonical(full) == atlas_noisy_ae.canonical(atlas_noisy_ae.RESOLVED_RECIPE)
    assert full['total_steps'] is None and full['encoder_mode'] == 'none'
    assert (full['num_particles'], full['z_dim'], full['batch_size']) == (12, 2, 64)
    assert context.noisy_ae_binding['recipe'] == json.loads(atlas_noisy_ae.canonical(full))
    assert context._trainer is None and context._policy is None
    assert adapter_preflight(task, candidate(), root=ROOT) == []


def test_ae_shared_validator_delegates_to_exact_owner_guard(monkeypatch):
    from experiments.forge import atlas_noisy_ae
    calls = []
    marker = {'status': 'INVALID', 'reason': 'missing original measured AE owner'}
    def guard(task, evidence):
        calls.append((task['id'], evidence)); return marker
    monkeypatch.setattr(atlas_noisy_ae, 'validate_evidence', guard)
    task = variant('ae_gan_hold'); evidence = {'sentinel': True}
    assert noisy_prior_adapters.validate_evidence(task, evidence) is marker
    assert calls == [(task['id'], evidence)]


def test_ae_dispatch_preserves_single_ordinary_owner_call(monkeypatch, tmp_path):
    from experiments.forge import adapters, atlas_noisy_ae
    calls = []
    task = variant('ae_gan_hold')
    request = {'candidate': candidate(), 'tasks': {task['id']: task},
               'source': {'snapshot_path': str(ROOT)}}
    marker = {'ordinary_owner_called': True}
    def run(request, task, output, device):
        calls.append((request, task, output, device)); return marker
    monkeypatch.setattr(atlas_noisy_ae, 'run_behavior', run)
    monkeypatch.setattr(atlas_noisy_ae, 'blockers', lambda *a, **k: [])
    assert adapters._dispatch_task(request, {'task_id': task['id']}, tmp_path, 'cpu') is marker
    assert calls == [(request, task, tmp_path, 'cpu')]
