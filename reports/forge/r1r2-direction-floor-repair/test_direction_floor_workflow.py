"""Protect bounded admission, exact paid reuse, registry and archive guards."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('direction_floor_workflow', Path(__file__).with_name('workflow.py'))
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
    with pytest.raises(ValueError, match='explicit native optimizer'):
        workflow.validate_manifest(value)


def control_fixture(monkeypatch):
    controls = {role: dict(candidate=workflow.ZERO_CANDIDATE if role == 'adam' else 'scheduled',
                          source_digest='original-source', runtime_sha256='original-runtime',
                          protocol_sha256='seed0-public', task_contracts={t: 'fixed' for t in workflow.shared.TASKS},
                          task_keys={t: f'{role}-{t}' for t in workflow.shared.TASKS}) for role in ('baseline', 'adam')}
    resolved = {role: dict(projection=deepcopy(projection)) for role, projection in controls.items()}
    jobs = {key: dict(result=dict(attempt_id=f'attempt-{key}'), attempts=['one'])
            for projection in controls.values() for key in projection['task_keys'].values()}
    monkeypatch.setattr(workflow, 'original_requests', lambda *_args: (resolved, {}))
    def read(path):
        if path.name == 'state.json':
            return dict(jobs=jobs)
        return dict(candidates=dict(baseline=controls['baseline'], zero=controls['adam']))
    monkeypatch.setattr(workflow.shared, 'read_json', read)
    return controls, resolved, jobs


@pytest.mark.parametrize('role', ['baseline', 'adam'])
@pytest.mark.parametrize('field', ['source_digest', 'runtime_sha256', 'protocol_sha256', 'task_contracts'])
def test_cached_control_refuses_changed_scientific_binding(role, field, tmp_path, monkeypatch):
    _, resolved, _ = control_fixture(monkeypatch)
    resolved[role]['projection'][field] = 'changed'
    with pytest.raises(ValueError, match='retain exact measured'):
        workflow.requests(tmp_path, tmp_path, {})


@pytest.mark.parametrize('role', ['baseline', 'adam'])
@pytest.mark.parametrize('change', ['missing', 'unmeasured', 'retry'])
def test_each_control_requires_seven_original_single_attempt_results(role, change, tmp_path, monkeypatch):
    controls, _, jobs = control_fixture(monkeypatch)
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
    _, resolved, _ = control_fixture(monkeypatch)
    resolved['adam']['projection']['candidate'] = 'another-measured-recipe'
    with pytest.raises(ValueError, match='exact measured zero-momentum'):
        workflow.requests(tmp_path, tmp_path, {})


def test_matching_controls_return_saved_requests_without_admission(tmp_path, monkeypatch):
    _, resolved, _ = control_fixture(monkeypatch)
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
    previous = (original, tmp_path / 'five-track', tmp_path / 'roles')
    monkeypatch.setattr(workflow, 'PRIOR_ARCHIVES', previous)
    return archive, previous


def test_storage_requires_the_exact_original_queue_symlink(tmp_path, monkeypatch):
    archive, _ = archive_fixture(tmp_path, monkeypatch)
    (archive / 'queue').unlink()
    (archive / 'queue').mkdir()
    with pytest.raises(ValueError, match='exact original queue'):
        workflow.storage(archive)


def test_storage_counts_all_four_archives_against_one_cap(tmp_path, monkeypatch):
    archive, previous = archive_fixture(tmp_path, monkeypatch)
    readings = []
    def read(path):
        readings.append(path)
        return dict(bytes=workflow.shared.ARCHIVE_CAP // 4, free_bytes=workflow.shared.FREE_MIN)
    monkeypatch.setattr(workflow, 'original_storage', read)
    with pytest.raises(ValueError, match='four research archives'):
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
