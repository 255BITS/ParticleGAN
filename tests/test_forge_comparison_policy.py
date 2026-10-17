"""Historical comparison policy changes counts, never original task verdicts."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from test_forge_current_technique_inventory import ROOT, evidence, publication


def test_actual_registered_dispositions_are_a_valid_comparison_projection():
    actual = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v12.json')
    recorded = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v9.json')
    publication._validate_comparison_policy(actual, recorded)


@pytest.mark.parametrize('change', ['task', 'tier', 'order', 'eligibility', 'calibration', 'same_revision'])
def test_comparison_policy_refuses_changes_beyond_dispositions(change):
    actual = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v12.json')
    recorded = read_json(ROOT / 'configs/forge/view-history/discriminator_stability-v9.json')
    if change == 'task':
        recorded['assignments'][0]['task'] = 'another_task'
    elif change == 'tier':
        recorded['assignments'][0]['qualification_tier'] = 2
    elif change == 'order':
        recorded['assignments'].reverse()
    elif change in {'eligibility', 'calibration'}:
        recorded[change] = {'changed': True}
    else:
        recorded['revision'] = actual['revision']
    with pytest.raises(ValueError, match='comparison policy'):
        publication._validate_comparison_policy(actual, recorded)


def comparison_fixture():
    actual = dict(view='stability', view_revision=11, policy_fingerprint='actual',
                  tier_requirements={'1': ['acquire'], '2': [], '3': ['hold']})
    row = dict(candidate_id='optimized', candidate_revision='revision', cohort='cohort',
               attempt_ids=['attempt'], bindings={'trainer_family': 'family', 'task_contracts': {'acquire': 'a', 'held': 'b', 'hold': 'c'}},
               runtime_cohort={'backend': 'cuda'}, cost={'wall_seconds': 10}, status='PASS', qualified_tier=3,
               tiers={'1': {'passed': 1, 'total': 1}, '2': {'passed': 0, 'total': 0}, '3': {'passed': 1, 'total': 1}},
               tasks=[{'task_id': 'acquire', 'status': 'PASS'}, {'task_id': 'hold', 'status': 'PASS'}],
               nonrequired_tasks=[{'task_id': 'held', 'status': 'FAIL'}])
    actual['rows'] = [row]
    projected = deepcopy(actual)
    projected.update(view_revision=9, policy_fingerprint='comparison')
    projected['rows'][0].update(status='FAIL', qualified_tier=1)
    projected['rows'][0]['tasks'].append(projected['rows'][0]['nonrequired_tasks'].pop())
    projected['rows'][0]['tiers']['2'] = {'passed': 0, 'total': 1}
    comparison = dict(policy={'revision': 9}, path='policy.json', sha256='policy-hash',
                      reference_task_contracts={'family': deepcopy(row['bindings']['task_contracts'])})
    return actual, projected, comparison


def test_projection_preserves_original_qualification_and_all_numerical_verdicts():
    actual, projected, comparison = comparison_fixture()
    before = deepcopy(actual)
    publication._comparison_projection(actual, projected, comparison)
    assert actual == before
    assert projected['rows'][0]['qualified_tier'] == 1
    assert projected['rows'][0]['executed_policy_qualification']['qualified_tier'] == 3
    proof = projected['comparison_policy_projection']
    assert proof['actual_qualification']['rows'] == actual['rows']
    assert proof['actual_qualification_sha256'] == stable_hash(proof['actual_qualification'])
    assert proof['qualification_reuse'] is proof['qualification_input'] is False


@pytest.mark.parametrize('change', ['verdict', 'task_contract', 'runtime', 'attempt', 'missing'])
def test_projection_rejects_changed_evidence_or_missing_original(change):
    actual, projected, comparison = comparison_fixture()
    row = projected['rows'][0]
    if change == 'verdict':
        row['tasks'][-1]['status'] = 'PASS'
    elif change == 'task_contract':
        row['bindings']['task_contracts']['acquire'] = 'different-gate-budget-or-sampling'
    elif change == 'runtime':
        row['runtime_cohort'] = {'backend': 'cpu'}
    elif change == 'attempt':
        row['attempt_ids'] = ['foreign']
    else:
        projected['rows'] = []
    with pytest.raises(ValueError, match='comparison'):
        publication._comparison_projection(actual, projected, comparison)


def test_explicit_cached_comparison_preserves_registered_rows_without_regrading(evidence):
    root, manifest = evidence
    publication.publish_current(root)
    original = read_json(root / 'reports/forge/technique-inventory.json')['rows']
    manifest_hash = file_hash(root / publication.EVIDENCE_MANIFEST)
    policy_path = root / 'configs/forge/views/discriminator_stability.json'
    recorded_path = root / 'configs/forge/view-history/discriminator_stability-v2.json'
    atomic_json(recorded_path, read_json(policy_path))
    actual = read_json(policy_path)
    actual['revision'] = 3
    atomic_json(policy_path, actual)
    result = publication.publish_current(root, comparison_policy=recorded_path)
    assert read_json(result['json'])['rows'] == original
    assert file_hash(root / publication.EVIDENCE_MANIFEST) == manifest_hash


def test_independent_projection_requires_a_frozen_source(tmp_path):
    with pytest.raises(ValueError, match='exact frozen source'):
        publication.regenerate(tmp_path, comparison_policy=Path('policy.json'))


def test_pending_snapshot_overlay_is_read_only_hash_checked_and_restored_on_failure(tmp_path):
    from experiments.forge.test_performance import Inputs
    pending = tmp_path / 'reports/forge/technique-evidence/new.json'
    text = '{"new_source": true}\n'
    import hashlib
    expected = hashlib.sha256(text.encode()).hexdigest()
    original = Inputs.__init__
    with pytest.raises(RuntimeError, match='render failed'):
        with publication._pending_publication_inputs(tmp_path, (pending, text)):
            inputs = Inputs(tmp_path)
            assert inputs.read('reports/forge/technique-evidence/new.json', sha256=expected) == {'new_source': True}
            with pytest.raises(ValueError, match='bytes changed'):
                inputs.read('reports/forge/technique-evidence/new.json', sha256='wrong')
            assert not pending.exists()
            raise RuntimeError('render failed')
    assert Inputs.__init__ is original
    assert not pending.exists()
    with pytest.raises(FileNotFoundError):
        Inputs(tmp_path).read('reports/forge/technique-evidence/new.json')
    pending.parent.mkdir(parents=True)
    pending.write_text('{}\n')
    with pytest.raises(ValueError, match='conflicts'):
        with publication._pending_publication_inputs(tmp_path, (pending, text)):
            pytest.fail('conflicting immutable source must not be overlaid')
    assert pending.read_text() == '{}\n'


@pytest.mark.parametrize('captured', [False, True])
def test_frozen_catalog_overlay_preserves_captured_presence_and_absence(tmp_path, captured):
    name = 'reports/transfer_suite/unadjusted/leading_profile.json'
    live, frozen = tmp_path / 'live', tmp_path / 'frozen'
    source, target = live / name, frozen / name
    source.parent.mkdir(parents=True)
    source.write_text('{"live": "later optional profile"}\n')
    sibling = source.with_name('readout.json')
    sibling.write_text('{"knowledge": true}\n')
    files = {}
    if captured:
        target.parent.mkdir(parents=True)
        target.write_text('{"frozen": "original profile"}\n')
        files[name] = file_hash(target)
    closure = publication._catalog_support_closure([name], {'recorded-source': {'files': files}})
    publication._overlay_reports(live / 'reports', frozen / 'reports',
                                 excluded=[Path(p).relative_to('reports') for p in closure['preserved_absent_paths']])
    assert target.exists() == captured
    if captured:
        assert target.read_text() == '{"frozen": "original profile"}\n'
        assert closure['captured_present_paths'] == [name]
        assert not target.is_symlink()
    else:
        assert closure['preserved_absent_paths'] == [name]
    assert (target.parent / sibling.name).read_bytes() == sibling.read_bytes()
    assert source.read_text() == '{"live": "later optional profile"}\n'


def test_catalog_closure_retains_original_study_support_and_rejects_conflicting_source_presence():
    name = 'reports/profile.json'
    sources = {'sparse': {'files': {}}, 'full': {'files': {name: 'original-hash'}}}
    with pytest.raises(ValueError, match='conflicting captured'):
        publication._catalog_support_closure([name], sources)
    closure = publication._catalog_support_closure([name], {'sparse': sources['sparse']}, {name: 'study-hash'})
    assert closure['preserved_absent_paths'] == []
    with pytest.raises(ValueError, match='unsafe'):
        publication._catalog_support_closure(['reports/../escape.json'], {'source': {'files': {}}})


def test_catalog_inventory_uses_frozen_profile_rules_in_a_fresh_process(tmp_path):
    package = tmp_path / 'experiments/forge'
    package.mkdir(parents=True)
    (package.parent / '__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (package / 'views.py').write_text(
        "def load_tasks(root):\n    return {'task': {'evaluation': {'sources': {'reports/evaluator.py': 'hash'}}}}\n")
    (package / 'hostprofiles.py').write_text(
        "def profile_source_paths(task):\n    return ['reports/frozen-profile.json']\n")
    assert publication._frozen_catalog_support(tmp_path) == ['reports/evaluator.py', 'reports/frozen-profile.json']


@pytest.mark.parametrize('missing', ['proof', 'measured', 'both'])
def test_projection_refuses_to_turn_recorded_study_into_silent_unresolved_rows(missing):
    request = {'candidate': {'id': 'optimized'}, 'view': {'id': 'stability'}, 'execution_backend': 'cuda'}
    studies = [{'request': request}]
    result = {'registered_study_reconstruction': [{'candidate_id': 'optimized'}],
              'rows': [{'candidate_id': 'optimized', 'attempt_ids': ['original']} ]}
    publication._validate_recorded_study_projection(studies, result, 'stability', 'cuda')
    if missing in {'proof', 'both'}:
        result['registered_study_reconstruction'] = []
    if missing in {'measured', 'both'}:
        result['rows'] = [{'candidate_id': 'optimized', 'attempt_ids': []}]
    with pytest.raises(ValueError, match='lost recorded study admission or measured rows'):
        publication._validate_recorded_study_projection(studies, result, 'stability', 'cuda')
    publication._validate_recorded_study_projection(studies, result, 'another-view', 'cuda')


@pytest.mark.parametrize('scope', ['research_diagnostic', 'calibration_diagnostic'])
def test_diagnostic_study_does_not_require_an_ordinary_qualifying_row(scope):
    studies = [{'request': {'candidate': {'id': 'diagnostic'}, 'view': {'id': 'view', 'evidence_scope': scope},
                            'execution_backend': 'cuda'}}]
    publication._validate_recorded_study_projection(studies, {'rows': []}, 'view', 'cuda')
