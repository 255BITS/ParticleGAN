"""Saved-request topology guards; no file-backed queue or paid claim is used."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('energy_administrative_drain', Path(__file__).with_name('administrative_drain.py'))
admin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admin)


@pytest.fixture(scope='module')
def original():
    return admin.shared.read_json(admin.frozen.ARCHIVE / 'queue/queue/state.json')


@pytest.fixture
def topology(original, monkeypatch):
    state = deepcopy(original)
    @contextmanager
    def memory_state(_queue):
        yield state
    # Even Queue.collect uses this memory-only transaction; no atomic writer.
    monkeypatch.setattr(admin.frozen.OriginalBoundedQueue, 'state', memory_state)
    return state


def queue(cls=admin.AdministrativeQueue):
    return cls(admin.frozen.ARCHIVE / 'queue', paid_cap_seconds=8000)


def test_exact_29_job_requests_reproduce_the_frozen_collector_missing_key(topology):
    q = queue(admin.frozen.ScopedQueue)
    with q.scoped('collect'), q.state() as state:
        assert len(state['jobs']) == 14
        assert all(len(s['request']['jobs']) == 29 for s in state['submissions'].values())
        with pytest.raises(KeyError):
            q._results(state, next(iter(state['submissions'].values())))


def test_admin_collector_retains_all58_and_preserves_every_old_job_and_cost(topology):
    before = deepcopy(topology)
    q = queue()
    with q.scoped('collect'), q.state() as state:
        assert len(state['jobs']) == 58 and len(state['charges']) == 217
        for s in state['submissions'].values():
            assert q._results(state, s) == []
        assert state['charges'] is topology['charges']
    assert q.collect() == 0
    assert topology == before


def test_foreign_pending_tier2_cannot_enter_claims_or_collections(topology, monkeypatch):
    foreign = deepcopy(next(iter(topology['submissions'][rid] for rid in admin.REQUESTS.values())))
    foreign['request'].update(request_id='foreign', campaign_id='foreign', through_tier=2)
    foreign['status'] = 'queued'
    topology['submissions']['foreign'] = foreign
    topology['jobs']['foreign-running'] = dict(status='running', reserved_seconds=100, attempts=[])
    before = deepcopy(foreign)
    observed = []
    def claim(q, _slots, **kw):
        with q.state() as state:
            assert kw['campaign_filter'] == admin.frozen.CAMPAIGN
            assert set(state['submissions']) == set(admin.REQUESTS.values())
            assert 'foreign-running' in state['jobs']  # Full global reservations.
            for s in state['submissions'].values():
                eligible, _reason, _running = q._eligible(state, s)
                allowed = {j['compatibility_key'] for j in s['request']['jobs'] if admin.shared.Queue._authorized(s['request'], j)}
                assert len(allowed) == 7 and set(eligible) <= allowed
                assert all(state['jobs'][k]['definition']['task_id'] in admin.frozen.BUDGETS for k in eligible)
            observed.append(True)
        return None
    monkeypatch.setattr(admin.frozen.OriginalBoundedQueue, 'claim', claim)
    q = queue()
    assert q.claim([]) is None and observed
    with q.scoped('collect'), q.state() as state:
        assert 'foreign-running' not in state['jobs'] and 'foreign' not in state['submissions']
    assert topology['submissions']['foreign'] == before
    assert topology['jobs']['foreign-running']['reserved_seconds'] == 100
    assert len(topology['charges']) == 217


@pytest.mark.parametrize('change', ['attempt', 'subscriber', 'result', 'reservation'])
def test_higher_tier_bookkeeping_cannot_become_scientific_work(topology, change):
    request = topology['submissions'][admin.REQUESTS['energy_u']]['request']
    definition = next(j for j in request['jobs'] if not admin.shared.Queue._authorized(request, j))
    job = topology['jobs'][definition['compatibility_key']]
    if change == 'attempt': job['attempts'] = [dict(attempt_id='forbidden')]
    elif change == 'subscriber': job['subscribers'] = ['foreign']
    elif change == 'result': job['result'] = dict(attempt_id='forbidden')
    else: job['reserved_seconds'] = 1
    q = queue()
    with pytest.raises(ValueError, match='Higher-tier bookkeeping'), q.scoped('collect'), q.state():
        pass


@pytest.fixture
def coordinator(monkeypatch):
    commit = 'c' * 40
    actual_run = admin.subprocess.run
    def run(args, **kwargs):
        if args[:3] == ['git', 'merge-base', '--is-ancestor']:
            assert args[3:] == [admin.EXECUTION, commit]
            return SimpleNamespace(returncode=0)
        return actual_run(args, **kwargs)
    monkeypatch.setattr(admin.subprocess, 'run', run)
    monkeypatch.setattr(admin, 'git_text', lambda _root, *args: commit if args[0]=='rev-parse' else '\n'.join(sorted(admin.ADMIN_FILES)))
    calls = []
    monkeypatch.setattr(admin, 'git_files', lambda _root, revision, expected: calls.append((revision, deepcopy(expected))))
    return commit, calls


def test_coordinator_c_is_separate_from_all_actual_scientific_b_inputs(coordinator):
    commit, calls = coordinator
    receipt = admin.verify(admin.ROOT, commit)
    assert receipt['coordinator_commit'] == commit
    assert receipt['scientific_execution_commit'] == admin.EXECUTION
    assert receipt['requests'] == admin.REQUESTS
    assert receipt['source_files'] == 1216 and receipt['declarations'] == 36
    assert receipt['paid_attempts_before'] == 217
    science = [(revision, expected) for revision, expected in calls if len(expected)==1216]
    assert [revision for revision, _ in science] == [admin.EXECUTION, commit]
    assert science[0][1] == science[1][1]


@pytest.mark.parametrize('change', ['old_coordinator', 'science_change', 'progress_origin'])
def test_admin_commit_c_cannot_replace_science_b_or_admitted_progress(coordinator, monkeypatch, change):
    commit, _ = coordinator
    if change == 'old_coordinator': commit = admin.EXECUTION
    elif change == 'science_change':
        monkeypatch.setattr(admin, 'git_text', lambda _root, *args: commit if args[0]=='rev-parse' else '\n'.join([*admin.ADMIN_FILES, 'particlegan/recipes.py']))
    else:
        original_read = admin.shared.read_json
        def read(path):
            value = original_read(path)
            if path.name.endswith('-progress.json'):
                value['source_commit'] = commit
            return value
        monkeypatch.setattr(admin.shared, 'read_json', read)
    with pytest.raises(ValueError, match='administrative commit|Coordinator C|original admitted B'):
        admin.verify(admin.ROOT, commit)


def test_actual_committed_b_science_blobs_match_the_frozen_child_snapshot(original):
    source = original['submissions'][admin.REQUESTS['energy_u']]['request']['source']
    admin.git_files(admin.ROOT, admin.EXECUTION, source['files'])
    corrupted = dict(source['files'])
    corrupted['particlegan/recipes.py'] = '0' * 64
    with pytest.raises(ValueError, match='Committed bytes changed'):
        admin.git_files(admin.ROOT, admin.EXECUTION, corrupted)
