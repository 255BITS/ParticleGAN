"""Protect bounded admission, exact paid reuse, registry and archive guards."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('objective_memory_workflow', Path(__file__).with_name('workflow.py'))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def manifest():
    return dict(schema_version=1, round=1, id=workflow.CAMPAIGN,
        max_candidates=5, baseline_role='baseline', paid_cap_seconds=8000, searches=[],
        candidates=[dict(role=role, candidate=role, study=f'{role}-study') for role in sorted(workflow.ROLES)],
        campaign=dict(id=workflow.CAMPAIGN, budget_seconds=12600, candidate_budget_seconds=2520))


@pytest.mark.parametrize('change', [dict(round=2), dict(id='unregistered'), dict(max_candidates=6),
                                    dict(paid_cap_seconds=8001), dict(baseline_role='adam'),
                                    dict(searches=[dict(id='extra-search')])])
def test_finite_registration_rejects_extra_scope(change):
    value = manifest()
    value.update(change)
    with pytest.raises(ValueError):
        workflow.validate_manifest(value)


@pytest.mark.parametrize('field,change', [('budget_seconds', -1), ('candidate_budget_seconds', -1)])
def test_full_scientific_reservations_cannot_be_shortened(field, change):
    value = manifest()
    value['campaign'][field] += change
    with pytest.raises(ValueError, match='full task allowance'):
        workflow.validate_manifest(value)


def test_an_extra_or_duplicated_role_cannot_replace_a_factorial_cell():
    value = manifest()
    value['candidates'][0]['role'] = value['candidates'][1]['role']
    with pytest.raises(ValueError, match='explicit global-transport and AMSGrad-memory'):
        workflow.validate_manifest(value)


def control_fixture(monkeypatch):
    controls = {role: dict(candidate=workflow.ZERO_CANDIDATE if role == 'adam' else
                          workflow.M110_CANDIDATE if role == 'm110' else 'scheduled',
                          source_digest='original-source', runtime_sha256='original-runtime',
                          protocol_sha256='seed0-public', task_contracts={t: 'fixed' for t in workflow.shared.TASKS},
                          task_keys={t: f'{role}-{t}' for t in workflow.shared.TASKS}) for role in ('baseline', 'adam', 'm110')}
    resolved = {role: dict(projection=deepcopy(projection)) for role, projection in controls.items()}
    jobs = {key: dict(result=dict(attempt_id=f'attempt-{key}'), attempts=['one'])
            for projection in controls.values() for key in projection['task_keys'].values()}
    rows = [dict(role=role, task_id=task, candidate_id=controls.get(role, {}).get('candidate', role),
                 attempt_id=f'attempt-{role}-{task}') for role in ('baseline', 'adam', 'ada', 'spectral', 'critic_spectral')
            for task in workflow.shared.TASKS]
    predecessor = dict(complete=True, round_id='r1r2-spectral-repair-v1',
                       certified_results=rows, task_cells=deepcopy(rows), media=deepcopy(rows))
    charges = [dict(attempt_id=f'historical-{i}', seconds=float(i), owner=dict(request=f'old-{i}')) for i in range(189)]
    predecessor['aggregate_charge_attempt_ids'] = [charge['attempt_id'] for charge in charges]
    audit = dict(status='PASS', inputs_sha256=dict(publication='predecessor-bytes',
                 aggregate189_charge_index=workflow.shared.stable_hash({c['attempt_id']: c for c in charges})))
    role_rows = [dict(role='m110', task_id=task, candidate_id=workflow.M110_CANDIDATE,
                     attempt_id=f'attempt-m110-{task}') for task in workflow.shared.TASKS]
    monkeypatch.setattr(workflow, 'original_requests', lambda *_args: (resolved, {}))
    monkeypatch.setattr(workflow.shared, 'file_hash', lambda *_args: 'predecessor-bytes')
    def read(path):
        if path.name == 'state.json':
            return dict(jobs=jobs, charges=charges)
        if path.name == 'results.json':
            return dict(certified_results=role_rows) if path.parent.parent.name == 'r1r2-role-repair' else predecessor
        if path.name == 'collection-audit.json':
            return audit
        if path.parent.name == 'r1r2-role-repair':
            return dict(candidates=controls)
        return dict(candidates=dict(baseline=controls['baseline'], zero=controls['adam']))
    monkeypatch.setattr(workflow.shared, 'read_json', read)
    return controls, resolved, jobs, predecessor, audit


@pytest.mark.parametrize('role', ['baseline', 'adam', 'm110'])
@pytest.mark.parametrize('field', ['source_digest', 'runtime_sha256', 'protocol_sha256', 'task_contracts'])
def test_cached_control_refuses_changed_scientific_binding(role, field, tmp_path, monkeypatch):
    _, resolved, _, _, _ = control_fixture(monkeypatch)
    resolved[role]['projection'][field] = 'changed'
    with pytest.raises(ValueError, match='retain exact measured'):
        workflow.requests(tmp_path, tmp_path, {})


@pytest.mark.parametrize('role', ['baseline', 'adam', 'm110'])
@pytest.mark.parametrize('change', ['missing', 'unmeasured', 'retry'])
def test_each_control_requires_seven_original_single_attempt_results(role, change, tmp_path, monkeypatch):
    controls, _, jobs, _, _ = control_fixture(monkeypatch)
    key = controls[role]['task_keys']['ring16_acquisition']
    if change == 'missing':
        jobs.pop(key)
    elif change == 'unmeasured':
        jobs[key]['result'] = None
    else:
        jobs[key]['attempts'].append('retry')
    with pytest.raises(ValueError, match='seven original one-attempt results'):
        workflow.requests(tmp_path, tmp_path, {})


def test_control_cannot_switch_to_another_measured_constant(tmp_path, monkeypatch):
    _, resolved, _, _, _ = control_fixture(monkeypatch)
    resolved['adam']['projection']['candidate'] = 'another-measured-recipe'
    with pytest.raises(ValueError, match='exact measured zero-momentum'):
        workflow.requests(tmp_path, tmp_path, {})


def test_matching_controls_return_saved_requests_without_admission(tmp_path, monkeypatch):
    _, resolved, _, _, _ = control_fixture(monkeypatch)
    assert workflow.requests(tmp_path, tmp_path, {}) == (resolved, {})


def test_shared_paid_cap_keeps_prior_charges_and_concurrent_full_allowances(monkeypatch):
    monkeypatch.setattr(workflow.shared.Queue, '_available_budget', lambda *_a, **_kw: (True, None))
    queue = object.__new__(workflow.shared.BoundedQueue)
    queue.paid_cap_seconds = 8000
    state = dict(charges=[dict(seconds=4001), dict(seconds=2799)], jobs={'active': dict(reserved_seconds=300)})
    assert queue._available_budget(state, {}, 900)[0]
    assert not queue._available_budget(state, {}, 901)[0]
    state['charges'].append(dict(seconds=1))
    assert not queue._available_budget(state, {}, 900)[0]


def archive_fixture(tmp_path, monkeypatch):
    archive, original = tmp_path / 'new', tmp_path / 'original'
    archive.mkdir()
    (original / 'queue').mkdir(parents=True)
    (archive / 'queue').symlink_to(original / 'queue', target_is_directory=True)
    monkeypatch.setattr(workflow, 'ARCHIVE', archive)
    monkeypatch.setattr(workflow, 'ORIGINAL', original)
    previous = (original, tmp_path / 'five-track', tmp_path / 'roles', tmp_path / 'direction-floor', tmp_path / 'spectral')
    monkeypatch.setattr(workflow, 'PRIOR_ARCHIVES', previous)
    return archive, previous


def test_storage_requires_the_exact_original_queue_symlink(tmp_path, monkeypatch):
    archive, _ = archive_fixture(tmp_path, monkeypatch)
    (archive / 'queue').unlink()
    (archive / 'queue').mkdir()
    with pytest.raises(ValueError, match='exact original queue'):
        workflow.storage(archive)


def test_storage_counts_all_six_archives_against_one_cap(tmp_path, monkeypatch):
    archive, previous = archive_fixture(tmp_path, monkeypatch)
    readings = []
    def read(path):
        readings.append(path)
        return dict(bytes=workflow.shared.ARCHIVE_CAP // 6 + 1, free_bytes=workflow.shared.FREE_MIN)
    monkeypatch.setattr(workflow, 'original_storage', read)
    with pytest.raises(ValueError, match='six research archives'):
        workflow.storage(archive)
    assert readings == [archive, *previous]


def test_storage_retains_per_archive_headroom_checks(tmp_path, monkeypatch):
    archive, previous = archive_fixture(tmp_path, monkeypatch)
    def read(path):
        if path == previous[-1]:
            raise ValueError('Archive cap/headroom exhausted')
        return dict(bytes=1, free_bytes=workflow.shared.FREE_MIN)
    monkeypatch.setattr(workflow, 'original_storage', read)
    with pytest.raises(ValueError, match='headroom exhausted'):
        workflow.storage(archive)


def test_configuration_controls_stay_in_the_original_registry(tmp_path, monkeypatch):
    cid = workflow.ZERO_CANDIDATE
    path = tmp_path / f'configs/forge/configurations/{cid}.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    idea = f'configs/forge/ideas/{cid}.json'
    monkeypatch.setattr(workflow, 'original_declarations', lambda *_a: {idea})
    resolved = dict(adam=dict(card=dict(candidate=cid)))
    paths = workflow.declaration_paths(tmp_path, {}, resolved)
    assert paths == {str(path.relative_to(tmp_path))} and not (tmp_path / idea).exists()


@pytest.mark.parametrize('change', ['incomplete', 'missing_case', 'missing_media', 'wrong_campaign', 'failed_audit', 'stale_audit'])
def test_predecessor_must_be_complete_and_independently_audited_before_building_requests(change, tmp_path, monkeypatch):
    _, _, _, predecessor, audit = control_fixture(monkeypatch)
    if change == 'incomplete': predecessor['complete'] = False
    elif change == 'missing_case': predecessor['certified_results'].pop()
    elif change == 'missing_media': predecessor['media'].pop()
    elif change == 'wrong_campaign': predecessor['round_id'] = 'unfinished-other-campaign'
    elif change == 'failed_audit': audit['status'] = 'FAIL'
    else: audit['inputs_sha256']['publication'] = 'different-publication-bytes'
    def forbidden(*_args):
        pytest.fail('An unfinished predecessor reached new request preparation')
    monkeypatch.setattr(workflow, 'original_requests', forbidden)
    with pytest.raises(ValueError, match='predecessor|Predecessor'):
        workflow.requests(tmp_path, tmp_path, {})


def test_m110_cannot_switch_to_the_prior_momentum_variant(tmp_path, monkeypatch):
    _, resolved, _, _, _ = control_fixture(monkeypatch)
    resolved['m110']['projection']['candidate'] = 'r1r2-role-m111-v1'
    with pytest.raises(ValueError, match='exact measured role-momentum'):
        workflow.requests(tmp_path, tmp_path, {})


@pytest.mark.parametrize('role', ['baseline', 'adam', 'm110'])
def test_a_replaced_cached_attempt_is_rejected_even_when_one_attempt_remains(role, tmp_path, monkeypatch):
    controls, _, jobs, _, _ = control_fixture(monkeypatch)
    jobs[controls[role]['task_keys']['ring16_acquisition']]['result']['attempt_id'] = 'replacement-attempt'
    with pytest.raises(ValueError, match='exact completed cached attempt'):
        workflow.requests(tmp_path, tmp_path, {})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'seconds', 'owner'])
def test_all189_historical_charges_are_retained_with_original_owners_and_values(change, tmp_path, monkeypatch):
    control_fixture(monkeypatch)
    state = workflow.shared.read_json(tmp_path / 'state.json')
    if change == 'missing': state['charges'].pop()
    elif change == 'duplicate': state['charges'].append(deepcopy(state['charges'][0]))
    elif change == 'seconds': state['charges'][0]['seconds'] += 1
    else: state['charges'][0]['owner']['request'] = 'new-owner'
    with pytest.raises(ValueError, match='historical189|Duplicate paid charge|Historical charge'):
        workflow.requests(tmp_path, tmp_path, {})
