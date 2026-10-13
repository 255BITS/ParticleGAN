"""Protect paid admission, exact control reuse and the registered trainer delta."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('role_workflow', Path(__file__).with_name('workflow.py'))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def manifest():
    return workflow.shared.read_json(Path(__file__).with_name('manifest.json'))


@pytest.mark.parametrize('field,value', [('round', 2), ('max_candidates', 11),
                                        ('paid_cap_seconds', 8001), ('baseline_role', 'm111')])
def test_admission_keeps_the_finite_registered_scope(field, value):
    data = manifest()
    data[field] = value
    with pytest.raises(ValueError):
        workflow.validate_manifest(data)


def test_full_reservation_cannot_be_shortened():
    data = manifest()
    data['campaign']['budget_seconds'] -= 1
    with pytest.raises(ValueError):
        workflow.validate_manifest(data)


def test_prior_charges_and_concurrent_reservations_remain_paid(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.shared.Queue, '_available_budget', lambda *_a, **_kw: (True, None))
    queue = workflow.shared.BoundedQueue(tmp_path, paid_cap_seconds=8000)
    state = dict(charges=[dict(seconds=2732), dict(seconds=4100)],
                 jobs={'active': dict(reserved_seconds=300)})
    assert not queue._available_budget(state, {}, 900)[0]
    assert queue._available_budget(state, {}, 868)[0]


def test_a_new_queue_cannot_rerun_the_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ARCHIVE', tmp_path)
    (tmp_path / 'queue').mkdir()
    with pytest.raises(ValueError, match='original artifacts'):
        workflow.storage(tmp_path)


@pytest.mark.parametrize('role', ['baseline', 'm000', 'm111'])
def test_each_control_rejects_a_changed_scientific_identity(role, tmp_path, monkeypatch):
    controls = {r: dict(candidate=r, source_digest='original', task_keys={r: r})
                for r in ['baseline', 'm000', 'm111']}
    resolved = {r: dict(projection=dict(p)) for r, p in controls.items()}
    resolved[role]['projection']['source_digest'] = 'changed'
    monkeypatch.setattr(workflow, 'original_requests', lambda *_a: (resolved, {}))
    def read(path):
        if path.name == 'state.json':
            return dict(jobs={r: dict(result={}, attempts=['one']) for r in controls})
        if 'five-track' in str(path):
            return dict(candidates=dict(baseline=controls['baseline'], momentum=controls['m111']))
        return dict(candidates=dict(zero=controls['m000']))
    monkeypatch.setattr(workflow.shared, 'read_json', read)
    with pytest.raises(ValueError, match='retain exact measured'):
        workflow.requests(tmp_path, tmp_path, {})


def test_factorial_changes_only_declared_global_momentum_roles():
    parent = workflow.shared.read_json(workflow.ROOT / 'configs/forge/ideas/r1r2-five-track-momentum-v1.json')
    fields = {'betas', 'd_betas', 'prior_betas', 'amsgrad'}
    for card in manifest()['candidates']:
        if card['role'] in {'baseline', 'm000', 'm111'}:
            continue
        recipe = workflow.shared.read_json(workflow.ROOT / f'configs/forge/ideas/{card["candidate"]}.json')
        source, actual = parent['recipe_overrides'], recipe['recipe_overrides']
        assert {k: v for k, v in actual.items() if k not in fields} == {
            k: v for k, v in source.items() if k not in fields}
        expected = [.5, .5, .5] if card['role'] == 'ams_m111' else [int(x) * .5 for x in card['role'][1:]]
        observed = [(actual['d_betas'] or actual['betas'])[0], actual['betas'][0],
                    (actual['prior_betas'] or actual['betas'])[0]]
        assert observed == expected
        assert all((actual.get(k) or actual['betas'])[1] == .99
                   for k in ['betas', 'd_betas', 'prior_betas'])
        assert actual['amsgrad'] == (card['role'] == 'ams_m111')
