"""Read-only admission/collection adversaries; queue fixtures stay in memory."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('energy_prior_freeze_workflow', Path(__file__).with_name('workflow.py'))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def manifest():
    return workflow.shared.read_json(Path(__file__).with_name('manifest.json'))


@pytest.fixture(scope='module')
def actual():
    # Resolve real READY declarations, source/task bindings and the existing
    # exact cache without preparing, submitting, constructing or training.
    value = manifest()
    resolved, searches = workflow.requests(workflow.ROOT, workflow.ORIGINAL / 'queue', value)
    assert not searches
    state = workflow.shared.read_json(workflow.ORIGINAL / 'queue/queue/state.json')
    return dict(manifest=value, resolved=resolved, state=state,
                recipes={role: workflow.public_recipes(item['request']) for role, item in resolved.items()})


@pytest.fixture
def campaign(actual, monkeypatch):
    data = {k: deepcopy(v) for k, v in actual.items() if k != 'state'}
    data['state'] = {**actual['state'], 'charges': deepcopy(actual['state']['charges']),
        'jobs': {k: {**j, 'attempts': deepcopy(j['attempts']), 'subscribers': list(j['subscribers'])}
                 for k, j in actual['state']['jobs'].items()}}
    original_read = workflow.shared.read_json
    monkeypatch.setattr(workflow.shared, 'read_json', lambda p:
        data['state'] if p.name == 'state.json' else original_read(p))
    monkeypatch.setattr(workflow, 'original_requests', lambda *_: (data['resolved'], {}))
    monkeypatch.setattr(workflow, 'public_recipes', lambda request:
        data['recipes'][next(role for role, item in data['resolved'].items() if item['request'] is request)])
    return data


def checked(data):
    return workflow.requests(workflow.ROOT, workflow.ORIGINAL / 'queue', data['manifest'])


def test_real_ready_studies_preserve_source_and_only_policy_delta(actual):
    assert set(actual['resolved']) == workflow.ROLES
    assert all(item['request']['study']['campaign'] == actual['manifest']['campaign']
               and item['request']['study_review']['status'] == 'READY'
               and item['projection']['source_digest'] == workflow.SOURCE for item in actual['resolved'].values())
    assert len(actual['state']['charges']) == 231
    for label in ['global', *workflow.shared.TASKS]:
        old, new = (actual['recipes'][r][0] if label == 'global' else actual['recipes'][r][1][label]
                    for r in ('energy_u', 'frozen_prior'))
        assert {k for k in old if old[k] != new[k]} == {'prior_update'}
        assert (old['prior_update'], new['prior_update']) == ('learned', 'frozen')


@pytest.mark.parametrize('field,value', [
    ('source_digest', 'a' * 64), ('paid_cap_seconds', 8001), ('baseline_role', 'frozen_prior'),
    ('round', 2), ('max_candidates', 3), ('searches', [dict(id='automatic')])])
def test_manifest_rejects_new_source_cap_or_scope(field, value):
    data = manifest(); data[field] = value
    with pytest.raises(ValueError): workflow.validate_manifest(data)


@pytest.mark.parametrize('field', ['budget_seconds', 'candidate_budget_seconds', 'id'])
def test_full_campaign_allowances_and_identity_cannot_change(field):
    data = manifest(); data['campaign'][field] = 2519 if field != 'id' else 'another'
    with pytest.raises(ValueError): workflow.validate_manifest(data)


@pytest.mark.parametrize('role', ['energy_u', 'frozen_prior'])
def test_both_real_studies_need_one_full_campaign(campaign, role):
    campaign['resolved'][role]['request']['study']['campaign']['budget_seconds'] -= 1
    with pytest.raises(ValueError, match='Both new studies'): checked(campaign)


@pytest.mark.parametrize('field', ['source_digest', 'runtime_sha256', 'protocol_sha256',
    'view_sha256', 'task_contracts', 'execution_policy', 'candidate_revision'])
def test_task_source_protocol_and_recipe_identity_are_fixed(campaign, field):
    campaign['resolved']['frozen_prior']['projection'][field] = 'changed'
    with pytest.raises(ValueError): checked(campaign)


@pytest.mark.parametrize('host', ['global', 'ring16_acquisition', 'five_word_joint_smoke', 'two_pole'])
@pytest.mark.parametrize('field,value', [('energy_distance_weight', 0.), ('reg_coeff', .01),
    ('loss', 'least_squares'), ('prior_betas', (.5, .99)), ('prior_update', 'learned')])
def test_no_second_delta_or_inactive_policy_substitution(campaign, host, field, value):
    recipe = campaign['recipes']['frozen_prior'][0] if host == 'global' else campaign['recipes']['frozen_prior'][1][host]
    recipe[field] = value
    with pytest.raises(ValueError, match='only actual recipe delta'): checked(campaign)


def test_false_learned_capability_requirement_is_rejected(campaign):
    campaign['resolved']['frozen_prior']['request']['candidate']['requires_capabilities'].append('learned_locations')
    with pytest.raises(ValueError, match='learned-location'): workflow.prior_delta(campaign['resolved'])


def test_pinned_original_host_binding_cannot_be_bypassed(campaign):
    request = campaign['resolved']['frozen_prior']['request']
    source = next(iter(request['tasks']['five_word_joint_smoke']['evaluation']['sources']))
    request['source']['files'][source] = 'changed'
    with pytest.raises(ValueError, match='pinned original'): checked(campaign)


@pytest.mark.parametrize('change', ['seconds', 'owner', 'remove', 'duplicate'])
def test_all231_historical_costs_and_owners_remain_exact(campaign, change):
    charge = campaign['state']['charges'][0]
    if change == 'seconds': charge['seconds'] += 1
    elif change == 'owner': charge['owner']['request'] = 'foreign'
    elif change == 'remove': campaign['state']['charges'].pop()
    else: campaign['state']['charges'].append(deepcopy(charge))
    with pytest.raises(ValueError): checked(campaign)


@pytest.mark.parametrize('change', ['attempt', 'projection', 'cached-key'])
def test_exact_cached_parent_is_required(campaign, change):
    parent = campaign['resolved']['energy_u']['projection']
    task = 'ring16_acquisition'; job = campaign['state']['jobs'][parent['task_keys'][task]]
    if change == 'attempt': job['attempts'].append(dict(attempt_id='replay'))
    elif change == 'projection': parent['jobs_sha256'] = 'different'
    else: campaign['resolved']['frozen_prior']['projection']['task_keys'][task] = parent['task_keys'][task]
    with pytest.raises(ValueError): checked(campaign)


def test_admission_refuses_foreign_reservation_that_cannot_fit_word900(campaign):
    campaign['state']['jobs']['foreign-reservation'] = dict(reserved_seconds=130., attempts=[], result=None)
    with pytest.raises(ValueError, match='full900 word'): checked(campaign)


@pytest.mark.parametrize('task', sorted(workflow.shared.TASKS))
def test_each_original_full_task_allowance_is_held(actual, task):
    request = deepcopy(actual['resolved']['frozen_prior']['request'])
    next(j for j in request['jobs'] if j['task_id'] == task)['budget_seconds'] -= 1
    with pytest.raises(ValueError): workflow.request_projection(request)


def test_paid_budget_preserves_full_word_reservation(actual, monkeypatch, tmp_path):
    queue = workflow.ScopedQueue(tmp_path, paid_cap_seconds=8000, on_completion=None)
    monkeypatch.setattr(workflow.shared.Queue, '_available_budget', lambda *_a, **_kw: (True, None))
    state = {**actual['state'], 'jobs': dict(actual['state']['jobs'])}; request = actual['resolved']['frozen_prior']['request']
    assert queue._available_budget(state, request, 900)[0]
    state['jobs']['foreign'] = dict(reserved_seconds=130.)
    allowed, reason = queue._available_budget(state, request, 900)
    assert not allowed and 'full next task' in reason


@pytest.fixture
def topology(actual, monkeypatch, tmp_path):
    # Actual complete29-job requests, including22 unsubscribed higher keys.
    state = {**actual['state'], 'submissions': dict(actual['state']['submissions']),
             'jobs': {k: {**j, 'attempts': deepcopy(j['attempts']), 'subscribers': list(j['subscribers'])}
                      for k, j in actual['state']['jobs'].items()}}
    requests, projections = {}, {}
    for role, item in actual['resolved'].items():
        request = deepcopy(item['request']); rid = f'new-{role}'
        request.update(request_id=rid, campaign_id=workflow.CAMPAIGN)
        requests[role] = rid; projections[role] = item['projection']
        state['submissions'][rid] = dict(request=request, status='queued', campaign_id=workflow.CAMPAIGN)
        for definition in request['jobs']:
            key = definition['compatibility_key']; authorized = workflow.shared.Queue._authorized(request, definition)
            if key not in state['jobs']:
                state['jobs'][key] = dict(status='pending', attempts=[], result=None, subscribers=[], reserved_seconds=0., definition=definition)
            if authorized and rid not in state['jobs'][key]['subscribers']:
                state['jobs'][key]['subscribers'].append(rid)
    progress = dict(phase='admitted', requests=requests, registration_sha256='registration')
    original_read = workflow.shared.read_json
    monkeypatch.setattr(workflow.shared, 'read_json', lambda p:
        progress if p.name.endswith('-progress.json') else dict(candidates=projections) if p.name == 'registration.json' else original_read(p))
    monkeypatch.setattr(workflow.shared, 'file_hash', lambda _p: 'registration')
    @contextmanager
    def memory_state(_self): yield state
    monkeypatch.setattr(workflow.energy.OriginalBoundedQueue, 'state', memory_state)
    queue = workflow.ScopedQueue(tmp_path, paid_cap_seconds=8000, on_completion=None)
    return queue, state, requests


def test_collection_reads_all58_keys_without_erasing_raw_history(topology):
    queue, state, requests = topology
    before = workflow.shared.stable_hash(state)
    with queue.scoped('collect'), queue.state() as selected:
        assert set(selected['submissions']) == set(requests.values())
        assert len(selected['jobs']) == 58
        for sub in selected['submissions'].values():
            assert len(sub['request']['jobs']) == 29
            queue._results(selected, sub)
        assert selected['charges'] is state['charges']
    assert workflow.shared.stable_hash(state) == before


def test_claim_scope_retains_foreign_costs_but_authorizes_only14(topology, monkeypatch):
    queue, state, requests = topology
    state['jobs']['foreign-running'] = dict(state='running', reserved_seconds=100., attempts=[], subscribers=['foreign'])
    state['submissions']['foreign'] = dict(status='queued', request=dict(request_id='foreign'))
    def inspect_claim(_self, _slots, **kwargs):
        assert kwargs['campaign_filter'] == workflow.CAMPAIGN
        with queue.state() as selected:
            assert 'foreign' not in selected['submissions'] and 'foreign-running' in selected['jobs']
            authorized = [j for sub in selected['submissions'].values() for j in sub['request']['jobs']
                          if workflow.shared.Queue._authorized(sub['request'], j)]
            assert len(authorized) == 14 and all(j['task_id'] in workflow.shared.TASKS for j in authorized)
            assert selected['jobs']['foreign-running']['reserved_seconds'] == 100.
        return None
    monkeypatch.setattr(workflow.energy.OriginalBoundedQueue, 'claim', inspect_claim)
    assert queue.claim([]) is None
    with queue.scoped('collect'), queue.state() as selected:
        assert 'foreign-running' not in selected['jobs']


@pytest.mark.parametrize('change', ['attempt', 'result', 'subscriber', 'reservation'])
def test_higher_bookkeeping_cannot_become_paid_or_subscribed(topology, change):
    queue, state, _ = topology
    request = state['submissions']['new-frozen_prior']['request']
    definition = next(j for j in request['jobs'] if not workflow.shared.Queue._authorized(request, j))
    job = state['jobs'][definition['compatibility_key']]
    if change == 'attempt': job['attempts'].append(dict(attempt_id='bad'))
    elif change == 'result': job['result'] = dict(attempt_id='bad')
    elif change == 'subscriber': job['subscribers'].append('foreign')
    else: job['reserved_seconds'] = 1
    with pytest.raises(ValueError, match='bookkeeping'):
        with queue.scoped('collect'), queue.state(): pass


def test_storage_pauses_only_own_campaign(topology, monkeypatch):
    queue, _, _ = topology; calls = []
    monkeypatch.setattr(workflow.shared.Queue, 'pause', lambda _self, campaign, paused: calls.append((campaign, paused)))
    queue.pause('old-science', True); queue.pause(workflow.CAMPAIGN, True)
    assert calls == [(workflow.CAMPAIGN, True)]


def test_claim_defers_word_until_all_six_nonword_jobs_are_terminal(topology):
    queue, state, _ = topology
    sub = state['submissions']['new-frozen_prior']; request = sub['request']
    definitions = [j for j in request['jobs'] if workflow.shared.Queue._authorized(request, j)]
    word = next(j['compatibility_key'] for j in definitions if j['task_id'] == 'five_word_joint_smoke')
    before = workflow.shared.stable_hash(state)
    # Original lifecycle/collection eligibility includes words. Claim alone
    # defers them while even the separate clock remains unfinished.
    normal = queue._eligible(state, sub)
    with queue.scoped('claim'):
        early = queue._eligible(state, sub)
    assert word in normal[0] and word not in early[0]
    assert workflow.shared.stable_hash(state) == before
    for definition in definitions:
        if definition['task_id'] == 'five_word_joint_smoke': continue
        state['jobs'][definition['compatibility_key']].update(status='terminal', result=dict(task_results=[
            dict(task_id=definition['task_id'], gate_status='FAIL')]))
    with queue.scoped('claim'):
        ready = queue._eligible(state, sub)
    assert ready[0] == [word]
    assert next(j for j in definitions if j['compatibility_key'] == word)['budget_seconds'] == 900


def test_no_media_collection_rejected_before_saved_publication_work():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='GIF'): workflow.collect(SimpleNamespace(media=False))


def test_ten_archives_share_combined_storage_cap(monkeypatch, tmp_path):
    original = tmp_path / 'original'; (original / 'queue').mkdir(parents=True)
    archive = tmp_path / 'new'; archive.mkdir(); (archive / 'queue').symlink_to(original / 'queue')
    monkeypatch.setattr(workflow, 'ORIGINAL', original); monkeypatch.setattr(workflow, 'ARCHIVE', archive)
    monkeypatch.setattr(workflow, 'PRIOR_ARCHIVES', tuple(tmp_path / f'old{i}' for i in range(9)))
    monkeypatch.setattr(workflow, 'original_storage', lambda _p: dict(bytes=workflow.shared.ARCHIVE_CAP // 10, free_bytes=workflow.shared.FREE_MIN))
    assert len(workflow.storage(archive)['archives']) == 10
    monkeypatch.setattr(workflow, 'original_storage', lambda _p: dict(bytes=workflow.shared.ARCHIVE_CAP // 10 + 1, free_bytes=workflow.shared.FREE_MIN))
    with pytest.raises(ValueError, match='ten'): workflow.storage(archive)
