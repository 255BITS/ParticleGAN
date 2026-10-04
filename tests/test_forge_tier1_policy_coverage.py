"""Software controls for scoped policy support; no qualification experiments."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.adapters import adapter_preflight
from experiments.forge.api import task_formulation_context
from experiments.forge.tier1_policy import materialize, validate, clock_measurement
from experiments.forge.sampling import grade_sampling
from experiments.forge.views import grade_result

ROOT = Path(__file__).resolve().parents[1]


def task(name):
    return json.loads((ROOT / 'configs/forge/tasks' / (name + '.json')).read_text())


def candidate(name):
    return json.loads((ROOT / 'configs/forge/ideas' / (name + '.json')).read_text())


@pytest.mark.parametrize('family', ['atlas', 'e22'])
@pytest.mark.parametrize('name', ['gaussian1d_acquisition', 'two_pole', 'ring16_acquisition', 'clockfree_audit'])
def test_supported_policy_preflight_keeps_family_recipe(family, name):
    parent = task(name); before = deepcopy(parent)
    variant = materialize(parent, ROOT)
    assert adapter_preflight(variant, candidate(family), root=ROOT) == []
    context = task_formulation_context(candidate(family), variant, device='cpu', root=ROOT)
    assert context.recipe_preset == family
    assert context.recipe.total_steps is None
    assert context.prior_config['kind'] == 'particle_cloud'
    assert parent == before
    assert variant['evaluation']['thresholds'] == parent['evaluation']['thresholds'] if 'thresholds' in parent['evaluation'] else variant['evaluation']['conditions'] == parent['evaluation']['conditions']


@pytest.mark.parametrize('family', ['atlas', 'e22'])
@pytest.mark.parametrize('name', ['unused_token_hold', 'ae_gan_hold', 'five_word_joint_acquisition'])
def test_incompatible_independent_owners_are_blocked(family, name):
    variant = materialize(task(name), ROOT)
    blockers = adapter_preflight(variant, candidate(family), root=ROOT)
    assert blockers and any('RoutedRows' in value or 'routed AE' in value or 'BiGAN' in value for value in blockers)


def test_coherent_threshold_changes_cannot_rebind_original_parent():
    variant = materialize(task('ring16_acquisition'), ROOT)
    variant['evaluation']['thresholds'][0][2] = 1
    with pytest.raises(ValueError, match='changed parent settings'):
        validate(variant, root=ROOT)


def test_original_sampling_receipt_cannot_fill_selected_policy_cell():
    variant = materialize(task('ring16_acquisition'), ROOT)
    assert grade_sampling(variant, {'sampling_contract_version': 1,
        'sampling_law': 'public_prior_without_output_noise', 'eval_output_noise': 'clean'})['status'] == 'INVALID'


def test_policy_metrics_without_actual_hooks_cannot_pass():
    variant = materialize(task('ring16_acquisition'), ROOT)
    evidence = {name: variant['evaluation'][name] for name in ('sampling_contract_version','sampling_law','eval_output_noise','scoring_weights')}
    assert grade_result(variant, {'evidence': evidence})['status'] in {'INCOMPLETE', 'INVALID'}


def test_direct_table_software_control_observes_real_lifecycle():
    from experiments.forge.policy_behavior_adapters import TwoPolePolicyFixture
    variant = materialize(task('two_pole'), ROOT)
    fixture = TwoPolePolicyFixture({'candidate': candidate('e22'), 'protocol': {'seed': 0}}, variant)
    fixture.step(); fixture.step()
    measured = fixture.observe()
    assert set(measured) == {'mean_abs', 'grad_med'}
    assert fixture.audit.receipt(2)['complete']
    assert fixture.guards()['all_finite']
    saved = fixture.state_dict()
    fixture.load_state_dict(saved)
    assert fixture.completed_steps == 2


def test_measurement_scope_is_different_from_clockfree_claim():
    parent = task('clockfree_audit')
    variant = clock_measurement(parent, ROOT)
    assert variant['id'] != parent['id']
    assert {key: value for key, value in variant['evaluation'].items() if key != 'sources'} == parent['evaluation']
    assert variant['execution']['clock_audit_parent']['id'] == parent['id']
    assert adapter_preflight(variant, candidate('k3p'), root=ROOT) == []


def test_scalar_policy_software_control_retains_selected_observation(tmp_path):
    from experiments.forge.adapters import _Run
    from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
    variant = materialize(task('gaussian1d_acquisition'), ROOT)
    context = task_formulation_context(candidate('atlas'), variant, device='cpu', root=ROOT)
    trainer = context.build_trainer(*build_vector_models(context, resolve_vector_spec(variant)))
    run = _Run(context, trainer, tmp_path, variant)
    run.step(torch.zeros(context.recipe.batch_size, 1))
    from benchmarks.toy_audit.gaussian1d_quality import score_samples
    from experiments.forge.adapters import _save_observer_outputs
    saved_samples = []
    def observe():
        values = run.sample(8)
        saved_samples.append({'step': 1, 'samples': values.detach().cpu()})
        return score_samples(values, resolve_vector_spec(variant), 1)
    measured = run.evaluate(observe)
    descriptor = _save_observer_outputs(tmp_path, 'observed-samples.pt', saved_samples, kind='scored_vector_samples_v1')
    result = run.receipt({'observations': [{'step': 1, **measured}], 'live': measured, 'saved_observer_outputs': descriptor})
    assert result['evidence']['policy_controls']['implementation_observed']
    assert result['evidence']['policy_observations'][0]['observed']
    assert result['evidence']['policy_purity'][0]['pure']
    assert result['evidence']['guards']['all_finite']
    from experiments.forge.tier1_media import render
    before = context.streams.audit()
    media = render(variant, {**result, 'gate_status': 'FAIL'}, tmp_path, tmp_path / 'goal.gif')
    assert media['optimizer_updates_added'] == media['sampling_draws_added'] == 0
    assert context.streams.compare(before, context.streams.audit())['unintended_rng_deviations'] == 0


def test_policy_clockproof_roundtrip_is_a_property_control(tmp_path):
    from experiments.forge.clockfree import run_clockfree, verify_probe
    variant = materialize(task('clockfree_audit'), ROOT)
    result = run_clockfree({'candidate': candidate('e22'), 'protocol': {'seed': 0}}, variant, tmp_path, 'cpu')
    comparisons, audit = verify_probe(variant, result['evidence'])
    assert len(comparisons) == 4
    assert audit['unexplained_clock_dependencies']
    grade = grade_result(variant, result)
    assert grade['status'] in {'FAIL', 'BLOCKED'}
    from experiments.forge.tier1_media import render
    assert render(variant, {**result, 'gate_status': grade['status']}, tmp_path, tmp_path / 'goal.gif')['observation_count'] == 4


def test_clean_scheduled_clock_measurement_returns_failure(tmp_path):
    from experiments.forge.clockfree import run_clockfree
    variant = clock_measurement(task('clockfree_audit'), ROOT)
    result = run_clockfree({'candidate': candidate('k3p'), 'protocol': {'seed': 0}}, variant, tmp_path, 'cpu')
    assert grade_result(variant, result)['status'] == 'FAIL'


@pytest.mark.parametrize('family', ['atlas', 'e22'])
def test_frozen_request_contains_parents_and_revalidates_sources(family, tmp_path):
    from experiments.forge.planning import resolve_idea
    request = resolve_idea(ROOT, family, view_id='tier1_policy_coverage', queue_root=tmp_path,
                           freeze_source=True)
    snapshot = Path(request['source']['snapshot_path'])
    assert not request['preflight_blockers']
    for variant in request['tasks'].values():
        parent_name = variant['policy_parent']['id']
        assert (snapshot / 'configs/forge/tasks' / (parent_name + '.json')).is_file()
        reasons = adapter_preflight(variant, request['candidate'], root=snapshot)
        assert bool(reasons) == (parent_name in {'unused_token_hold', 'ae_gan_hold', 'five_word_joint_acquisition'})
