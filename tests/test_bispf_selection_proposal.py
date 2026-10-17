"""Selection proposals fail closed and never change a selected recipe."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge import scaling_publication
from test_forge_scaling_publication import receipt, write_receipt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bispf_selection_proposal', ROOT / 'reports/forge/bispf-optimized/propose_selection.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


@pytest.fixture
def quality(tmp_path, monkeypatch):
    current = read_json(ROOT / 'reports/forge/technique-inventory.json')
    incumbent = next(row for row in current['rows'] if row.get('trainer_family') == 'bispf')
    before = current['recipe_contracts'][incumbent['bindings']['recipe_sha256']]
    atomic_json(tmp_path / 'reports/forge/technique-inventory.json',
                {'rows': [incumbent], 'recipe_contracts': {incumbent['bindings']['recipe_sha256']: before}})
    view = read_json(ROOT / 'configs/forge/views/discriminator_stability.json')
    atomic_json(tmp_path / 'configs/forge/views/discriminator_stability.json', view)
    row = deepcopy(incumbent)
    row.update(candidate_id='optimized', candidate_revision='new-revision', pending_readout=False)
    after = {**before, 'optimizer_svd_backend': 'newton_schulz'}
    row['bindings']['recipe_sha256'] = stable_hash(after)
    row['bindings']['source_digest'] = 'new-source'
    row['bindings']['source_origin_commit'] = 'a' * 40
    row['bindings']['recorded_source_origin_commits'] = ['a' * 40]
    actual = deepcopy(row)
    actual.update(status='PASS', qualified_tier=3)
    qualification = {'view_revision': view['revision'], 'policy_fingerprint': stable_hash(view), 'rows': [actual]}
    report = {'publication_scope': 'frozen_source', 'rows': [row], 'frozen_source': {'commit': 'a' * 40},
              'recipe_contracts': {stable_hash(after): after},
              'comparison_policy_projection': {'actual_qualification': qualification,
                                               'actual_qualification_sha256': stable_hash(qualification)}}
    monkeypatch.setattr(selection, '_published_report', lambda *args: (report, {'json_sha256': 'proof'}))
    monkeypatch.setattr(selection, '_validate_published_row', lambda *args: None)
    return tmp_path, report, incumbent


def test_quality_proposal_includes_every_incumbent_pass_including_clock(quality):
    root, _, incumbent = quality
    _, proof = selection.quality_proof(root, Path('source.json'), 'optimized')
    expected = {name for name, status in selection.statuses(incumbent).items() if status == 'PASS'}
    assert len(expected) == 19
    assert set(proof['incumbent_passes']) == expected
    assert 'clockfree_audit_measurement_v1' in expected


@pytest.mark.parametrize('origin_stage', ['absent', 'frozen_placeholder', 'published'])
def test_quality_origin_annotations_are_validated_without_mutating_actual_proof(quality, origin_stage):
    root, report, _ = quality
    actual = report['comparison_policy_projection']['actual_qualification']
    bindings = actual['rows'][0]['bindings']
    if origin_stage != 'published':
        bindings.pop('recorded_source_origin_commits')
        bindings.pop('source_origin_commit')
        if origin_stage == 'frozen_placeholder':
            bindings['source_origin_commit'] = None
    report['comparison_policy_projection']['actual_qualification_sha256'] = stable_hash(actual)
    before = deepcopy(report)
    row, proof = selection.quality_proof(root, Path('source.json'), 'optimized')
    assert report == before
    assert row['bindings']['source_origin_commit'] == proof['source_commit'] == 'a' * 40


@pytest.mark.parametrize('location', ['actual', 'projected'])
@pytest.mark.parametrize('change', ['wrong_origin', 'wrong_list', 'missing_origin', 'missing_list', 'multiple', 'empty_list'])
def test_quality_rejects_forged_wrong_or_partial_origin_claims(quality, location, change):
    root, report, _ = quality
    actual = report['comparison_policy_projection']['actual_qualification']
    row = actual['rows'][0] if location == 'actual' else report['rows'][0]
    bindings = row['bindings']
    if change == 'wrong_origin':
        bindings['source_origin_commit'] = 'b' * 40
    elif change == 'wrong_list':
        bindings['recorded_source_origin_commits'] = ['b' * 40]
    elif change == 'missing_origin':
        bindings.pop('source_origin_commit')
    elif change == 'missing_list':
        bindings.pop('recorded_source_origin_commits')
    elif change == 'multiple':
        bindings['recorded_source_origin_commits'].append('a' * 40)
    else:
        bindings['recorded_source_origin_commits'] = []
    report['comparison_policy_projection']['actual_qualification_sha256'] = stable_hash(actual)
    with pytest.raises(ValueError, match='source origin annotations'):
        selection.quality_proof(root, Path('source.json'), 'optimized')


@pytest.mark.parametrize('change', ['source_digest', 'recipe_sha256', 'task_contracts', 'cohort'])
def test_origin_compatibility_never_hides_other_scientific_binding_or_cohort_changes(quality, change):
    root, report, _ = quality
    actual = report['comparison_policy_projection']['actual_qualification']
    row = actual['rows'][0]
    row['bindings'].pop('recorded_source_origin_commits')
    row['bindings']['source_origin_commit'] = None
    if change == 'cohort':
        row['cohort'] = 'foreign'
    else:
        row['bindings'][change] = {'foreign': 'contract'} if change == 'task_contracts' else 'foreign'
    report['comparison_policy_projection']['actual_qualification_sha256'] = stable_hash(actual)
    with pytest.raises(ValueError, match='actual executed-policy'):
        selection.quality_proof(root, Path('source.json'), 'optimized')


def replace_quality_recipe(report, recipe):
    row = report['rows'][0]
    row['bindings']['recipe_sha256'] = stable_hash(recipe)
    report['recipe_contracts'] = {stable_hash(recipe): recipe}
    actual = report['comparison_policy_projection']['actual_qualification']
    actual['rows'][0]['bindings'] = deepcopy(row['bindings'])
    report['comparison_policy_projection']['actual_qualification_sha256'] = stable_hash(actual)


@pytest.mark.parametrize('rule', ['polar', 'frobenius', 'quintic'])
def test_quality_accepts_only_the_declared_single_optimizer_delta(quality, rule):
    root, report, incumbent = quality
    inventory = read_json(root / 'reports/forge/technique-inventory.json')
    before = inventory['recipe_contracts'][incumbent['bindings']['recipe_sha256']]
    after = {**before, 'optimizer_convolution_rule': rule}
    if rule == 'polar':
        after['optimizer_svd_backend'] = 'newton_schulz'
    replace_quality_recipe(report, after)
    _, proof = selection.quality_proof(root, Path('source.json'), 'optimized')
    expected = ({'optimizer_svd_backend': ['native', 'newton_schulz']} if rule == 'polar'
                else {'optimizer_convolution_rule': ['polar', rule]})
    assert proof['recipe_delta'] == expected
    assert proof['optimizer_contract']['convolution_rule'] == rule
    assert proof['optimizer_contract']['convolution_update_version'] == (
        'per_offset_polar_v1' if rule == 'polar' else f'per_offset_{rule}_hybrid_v1')
    assert proof['recipe_sha256'] == stable_hash(after)
    assert proof['native_recipe_sha256'] == stable_hash(before)
    assert proof['native_recipe_comparison_sha256'] == selection.recipe_comparison_hash(before)


@pytest.mark.parametrize('change', ['combined_backend', 'rate', 'prior_policy', 'unknown_rule', 'actual_source'])
def test_alternative_quality_rejects_additional_changes_or_foreign_actual_cohort(quality, change):
    root, report, incumbent = quality
    inventory = read_json(root / 'reports/forge/technique-inventory.json')
    before = inventory['recipe_contracts'][incumbent['bindings']['recipe_sha256']]
    after = {**before, 'optimizer_convolution_rule': 'frobenius'}
    if change == 'combined_backend':
        after['optimizer_svd_backend'] = 'newton_schulz'
    elif change == 'rate':
        after['lr'] *= 2
    elif change == 'prior_policy':
        after['prior_update'] = 'frozen'
    elif change == 'unknown_rule':
        after['optimizer_convolution_rule'] = 'other'
    replace_quality_recipe(report, after)
    if change == 'actual_source':
        actual = report['comparison_policy_projection']['actual_qualification']
        actual['rows'][0]['bindings']['source_digest'] = 'foreign'
        report['comparison_policy_projection']['actual_qualification_sha256'] = stable_hash(actual)
    with pytest.raises(ValueError, match='BISPF selection'):
        selection.quality_proof(root, Path('source.json'), 'optimized')


@pytest.mark.parametrize('change', ['clock_regression', 'unfinished', 'recipe_delta', 'actual_proof'])
def test_quality_proposal_rejects_regression_or_incomplete_evidence(quality, change):
    root, report, _ = quality
    row = report['rows'][0]
    qualification = report['comparison_policy_projection']['actual_qualification']
    if change == 'clock_regression':
        for item in row['nonrequired_tasks']:
            if item['task_id'] == 'clockfree_audit_measurement_v1':
                item['status'] = 'FAIL'
        qualification['rows'][0]['nonrequired_tasks'] = deepcopy(row['nonrequired_tasks'])
    elif change == 'unfinished':
        qualification['rows'][0]['pending_readout'] = True
    elif change == 'recipe_delta':
        report['recipe_contracts'][row['bindings']['recipe_sha256']]['lr'] = .02
    else:
        report['comparison_policy_projection']['actual_qualification_sha256'] = 'changed'
    if change != 'actual_proof':
        report['comparison_policy_projection']['actual_qualification_sha256'] = stable_hash(qualification)
    with pytest.raises(ValueError, match='BISPF selection'):
        selection.quality_proof(root, Path('source.json'), 'optimized')


def performance_fixture(tmp_path, *, slow=False, unknown=True, missing=False, wrong_source=False,
                        rule='polar', wrong_rule=False):
    tasks = ['dcgan_performance_' + suffix for suffix in ('r32', 'r64', 'r128', 'r256', 'b64', 'w128')]
    atomic_json(tmp_path / 'configs/forge/views/dcgan_performance.json',
                {'assignments': [{'task': task} for task in tasks]})
    manifests = []
    for index, candidate in enumerate(('optimized', selection.NATIVE, selection.R1R2)):
        entries = []
        for task in tasks:
            if missing and index == 0 and task == tasks[-1]:
                continue
            data = receipt(candidate=candidate)
            if index == 0:
                data['recipe']['optimizer_convolution_rule'] = 'other' if wrong_rule else rule
                if rule == 'polar':
                    data['recipe']['optimizer_svd_backend'] = 'newton_schulz'
                data['identity']['recipe_sha256'] = stable_hash(data['recipe'])
                data['identity']['global_recipe_sha256'] = stable_hash({key: value for key, value in data['recipe'].items()
                    if key not in {'batch_size', 'num_particles', 'z_dim', 'total_steps'}})
            data['task_id'] = task
            data['forge_binding'] = {'source_digest': 'new-source', 'candidate_revision': 'new-revision'}
            if wrong_source and index == 0:
                data['identity']['source']['commit'] = 'b' * 40
                source = deepcopy(data['identity']['source']); source.pop('sha256')
                data['identity']['source']['sha256'] = stable_hash(source)
            data['hardware']['contention_status'] = 'unknown' if unknown else 'clear'
            latency = (90. if slow else 50.) if index == 0 else 800. if index == 1 else 40.
            data['metrics'].update(update_ms_median=latency, update_ms_p95=latency * 1.1)
            entries.append(write_receipt(tmp_path, data, name=candidate + '-' + task + '.json', task=task))
        manifest = scaling_publication.build_manifest(tmp_path, entries)
        path = tmp_path / ('manifest-' + str(index) + '.json')
        atomic_json(path, manifest); manifests.append(path)
    return manifests


def performance_quality(rule='polar'):
    return dict(source_commit='a' * 40, source_digest='new-source', candidate_revision='new-revision',
                optimizer_contract={'convolution_rule': rule,
                                    'svd_backend': 'newton_schulz' if rule == 'polar' else 'native'})


@pytest.mark.parametrize('rule', ['polar', 'frobenius', 'quintic'])
def test_observed_performance_targets_remain_unpaired_when_contention_unknown(tmp_path, rule):
    manifests = performance_fixture(tmp_path, rule=rule)
    proof = selection.performance_proof(tmp_path, manifests, 'optimized',
                performance_quality(rule))
    assert not proof['certified_paired_speed_ranking']
    assert len(proof['observations']) == 6
    assert proof['observations']['dcgan_performance_r256']['observed_native_speedup'] == 16


@pytest.mark.parametrize('change', ['slow', 'missing', 'wrong_source', 'wrong_rule'])
def test_performance_proposal_refuses_target_misses_or_incompatible_evidence(tmp_path, change):
    manifests = performance_fixture(tmp_path, **{change: True})
    with pytest.raises(ValueError, match='BISPF selection'):
        selection.performance_proof(tmp_path, manifests, 'optimized',
            performance_quality())


def test_propose_preserves_current_selection_and_every_other_family(quality, monkeypatch):
    root, report, _ = quality
    selected = root / selection.CURRENT_SELECTION
    original = {'selections': [{'trainer_family': 'bispf', 'candidate_id': 'bispf-v2'},
                               {'trainer_family': 'other', 'candidate_id': 'unchanged'}], 'default_adoption': False}
    atomic_json(selected, original)
    before = selected.read_bytes()
    monkeypatch.setattr(selection, 'performance_proof', lambda *args: {'passed': True})
    monkeypatch.setattr(selection, 'wide_proof', lambda *args: {'passed': True})
    proposal = selection.propose(root, source_report='source.json', candidate='optimized', manifests=[], wide_attempts=[])
    assert selected.read_bytes() == before
    assert proposal['original_selection_sha256'] == file_hash(selected)
    assert not proposal['selection_applied'] and not proposal['default_adoption']
    assert proposal['proposed_selection']['selections'][1] == original['selections'][1]
    assert proposal['proposed_selection']['selections'][0]['candidate_id'] == 'optimized'
    assert not (root / proposal['exact_archive_path']).exists()


@pytest.fixture
def wide(tmp_path, monkeypatch):
    attempts, envelopes, results = [], {}, {}
    native_recipe, fast_recipe = {'optimizer_svd_backend': 'native'}, {'optimizer_svd_backend': 'newton_schulz'}
    quality = dict(source_commit='a' * 40, source_digest='source', candidate_revision='new',
                   recipe_sha256=stable_hash(fast_recipe), native_recipe_sha256=stable_hash(native_recipe),
                   recipe_comparison_sha256=selection.recipe_comparison_hash(fast_recipe),
                   native_recipe_comparison_sha256=selection.recipe_comparison_hash(native_recipe))
    for candidate in ('optimized', selection.NATIVE):
        for task in selection.WIDE:
            definition = read_json(ROOT / 'configs/forge/tasks' / (task + '.json'))
            atomic_json(tmp_path / 'configs/forge/tasks' / (task + '.json'), definition)
            attempt = candidate + '-' + ('stripes' if 'stripes' in task else 'bars')
            directory = tmp_path / 'reports/forge/attempts' / attempt
            request = dict(candidate={'id': candidate, 'resolved_recipe': fast_recipe if candidate == 'optimized' else native_recipe},
                           candidate_revision='new' if candidate == 'optimized' else 'native', protocol={'seed': 0},
                           source={'origin_commit': quality['source_commit'], 'digest': quality['source_digest']},
                           view={'id': 'bispf_fastpath_quality', 'evidence_scope': 'research_diagnostic'},
                           tasks={task: definition}, queue_root=str(tmp_path / 'queue'), campaign_id='wide')
            row = dict(task_id=task, gate_status='PASS' if 'stripes' in task else 'FAIL',
                       evidence={'observations': [{} for _ in range(24)],
                                 'guards': {'all_finite': True, 'unintended_rng_deviations': 0}},
                       cost={'completed_steps': 600, 'optimizer_updates': {key: 600 for key in ('generator', 'discriminator', 'prior')}},
                       initialization={'G': 'shared', 'D': 'shared', 'prior': 'shared'}, rng={'data': 'shared'}, metrics={'score': 1})
            atomic_json(directory / 'request.json', {'request': request})
            atomic_json(directory / 'result.json', {'task_results': [row]})
            gif = tmp_path / 'queue/wide' / attempt / 'training.gif'
            gif.parent.mkdir(parents=True)
            gif.write_bytes(b'GIF89a-software-fixture')
            attempts.append(attempt); envelopes[attempt] = request; results[attempt] = row
    monkeypatch.setattr(selection, 'project_receipt', lambda *args: {
        'provenance': {'canonical_result_hash': 'certified', 'original_files': {}}})
    return tmp_path, quality, attempts, envelopes, results


def rewrite_wide(root, attempts, envelopes, results):
    for attempt in attempts:
        directory = root / 'reports/forge/attempts' / attempt
        atomic_json(directory / 'request.json', {'request': envelopes[attempt]})
        atomic_json(directory / 'result.json', {'task_results': [results[attempt]]})


def test_exercised_wide_quality_preserves_passes_and_keeps_shared_failure_visible(wide):
    root, quality, attempts, _, _ = wide
    proof = selection.wide_proof(root, attempts, 'optimized', quality)
    bars = proof['outcomes'][selection.WIDE[1]]
    assert bars['native']['status'] == bars['optimized']['status'] == 'FAIL'
    assert proof['outcomes'][selection.WIDE[0]]['optimized']['status'] == 'PASS'
    assert not proof['qualification_reuse']


def test_wide_native_allows_only_the_new_implicit_polar_default(wide):
    root, quality, attempts, envelopes, results = wide
    for request in envelopes.values():
        request['candidate']['resolved_recipe']['optimizer_convolution_rule'] = 'polar'
    rewrite_wide(root, attempts, envelopes, results)
    assert selection.wide_proof(root, attempts, 'optimized', quality)
    envelopes[selection.NATIVE + '-stripes']['candidate']['resolved_recipe']['optimizer_convolution_rule'] = 'quintic'
    rewrite_wide(root, attempts, envelopes, results)
    with pytest.raises(ValueError, match='global recipe'):
        selection.wide_proof(root, attempts, 'optimized', quality)


@pytest.mark.parametrize('rule', ['frobenius', 'quintic'])
def test_wide_alternatives_require_the_exact_qualified_rule(wide, rule):
    root, quality, attempts, envelopes, results = wide
    recipe = {'optimizer_svd_backend': 'native', 'optimizer_convolution_rule': rule}
    quality['recipe_sha256'] = stable_hash(recipe)
    quality['recipe_comparison_sha256'] = selection.recipe_comparison_hash(recipe)
    for request in envelopes.values():
        if request['candidate']['id'] == 'optimized':
            request['candidate']['resolved_recipe'] = deepcopy(recipe)
    rewrite_wide(root, attempts, envelopes, results)
    assert selection.wide_proof(root, attempts, 'optimized', quality)
    envelopes['optimized-stripes']['candidate']['resolved_recipe']['optimizer_convolution_rule'] = 'polar'
    rewrite_wide(root, attempts, envelopes, results)
    with pytest.raises(ValueError, match='global recipe'):
        selection.wide_proof(root, attempts, 'optimized', quality)


@pytest.mark.parametrize('change', ['budget', 'native_pass_loss', 'no_positive', 'recipe', 'revision', 'initialization',
                                    'source_commit', 'source_manifest', 'gif', 'task'])
def test_wide_quality_rejects_missing_or_incompatible_training_evidence(wide, change):
    root, quality, attempts, envelopes, results = wide
    stripes = 'optimized-stripes'
    if change == 'budget':
        results[stripes]['cost']['completed_steps'] = 599
    elif change == 'native_pass_loss':
        results[selection.NATIVE + '-bars']['gate_status'] = 'PASS'
    elif change == 'no_positive':
        results[stripes]['gate_status'] = 'FAIL'
        results[selection.NATIVE + '-stripes']['gate_status'] = 'FAIL'
    elif change == 'recipe':
        envelopes[stripes]['candidate']['resolved_recipe'] = {'optimizer_svd_backend': 'changed'}
    elif change == 'revision':
        envelopes[stripes]['candidate_revision'] = 'foreign'
    elif change == 'initialization':
        results[stripes]['initialization'] = {'G': 'different'}
    elif change == 'source_commit':
        envelopes[stripes]['source']['origin_commit'] = 'b' * 40
    elif change == 'source_manifest':
        envelopes[stripes]['source']['digest'] = 'foreign'
    elif change == 'gif':
        (root / 'queue/wide' / stripes / 'training.gif').unlink()
    else:
        envelopes[stripes]['tasks'][selection.WIDE[0]]['execution']['steps'] = 599
    rewrite_wide(root, attempts, envelopes, results)
    with pytest.raises(ValueError, match='BISPF selection'):
        selection.wide_proof(root, attempts, 'optimized', quality)
