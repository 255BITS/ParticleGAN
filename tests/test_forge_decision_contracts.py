"""Numerical intent and coordinator controls; no training is executed."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge import decision_contracts as decisions, knowledge
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash, validate_idea
from experiments.forge.planning import new_idea, plan_summary, resolve_idea
from experiments.forge.queue import Queue
from test_forge_planning import checkout


from forge_legacy_fixtures import pin_legacy

@pytest.fixture
def ready(checkout):
    pin_legacy(checkout)
    path = new_idea(checkout, 'successor', 'base', goal='stability', hypothesis='A changed exercised learning rate')
    idea = read_json(path)
    idea.update(recipe_overrides={'lr': .006}, changed_factors=['learning rate'], mechanism_rationale='Example software control')
    atomic_json(path, idea)
    request = resolve_idea(checkout, 'successor', execution_backend='cpu')
    expected = request['decision_review']['expected']
    evidence = checkout / 'reports/prior.json'
    atomic_json(evidence, {'records': [{'record_id': 'original-failure', 'candidate_id': 'base', 'gate_status': 'FAIL'}]})
    contract = idea['decision_contract']
    contract.update(status='ready', prior_evidence=[{
        'path': str(evidence.relative_to(checkout)), 'sha256': file_hash(evidence), 'selector': ['records', 0],
        'identity': {'record_id': 'original-failure', 'candidate_id': 'base'}, 'use': 'motivation_only'}],
        candidate_binding_sha256=expected['candidate_binding_sha256'], substantive_delta=expected['substantive_delta'],
        prediction={'task_id': 't1', 'metric': 'score', 'op': '>=', 'threshold': .8, 'phase': 'final'},
        falsifier={'task_id': 't1', 'metric': 'score', 'op': '<', 'threshold': .5, 'phase': 'final'},
        competing_explanation='A transient gain can arise from contraction rather than sustained quality.')
    contract['control'].update(task_map=expected['task_map'], binding_sha256=expected['control_binding_sha256'])
    contract['scope'].update({key: expected[key] for key in (
        'task_ids', 'protocol_sha256', 'source_digest', 'execution_backend', 'runtime_cohort_sha256', 'jobs_sha256')},
        candidate_budget_seconds=10, campaign_budget_seconds=30)
    atomic_json(path, idea)
    return checkout, resolve_idea(checkout, 'successor', execution_backend='cpu'), path


def campaign(name='bounded', *, cap=10):
    return {'id': name, 'candidate_budget_seconds': cap, 'budget_seconds': 30}


def row(score, name='t1', gate='PASS'):
    return {'task_id': name, 'gate_status': gate, 'metrics': {'score': score}}


def test_new_draft_shows_actual_delta_and_blocks_before_snapshot_or_queue(checkout):
    path = new_idea(checkout, 'draft', 'base', goal='stability')
    assert read_json(path)['schema_version'] == 2
    draft = resolve_idea(checkout, 'draft', execution_backend='cpu')
    assert plan_summary(draft)['decision_contract']['status'] == 'BLOCKED'
    assert draft['decision_review']['actual_bindings']['candidate']['prior']['t1']['kind'] == 'mog'
    with pytest.raises(ValueError, match='submission blocked.*draft'):
        resolve_idea(checkout, 'draft', freeze_source=True, queue_root=checkout / 'queue', execution_backend='cpu')
    queue = Queue(checkout / 'queue', report_root=checkout / 'reports/forge')
    with pytest.raises(ValueError, match='admission blocked.*draft'):
        queue.submit(draft, campaign())
    assert not (checkout / 'queue').exists()


def test_ready_contract_copies_exercised_fields_and_admits_one_bounded_round(ready):
    root, request, _ = ready
    assert request['preflight_blockers'] == []
    assert request['decision_review']['status'] == 'READY'
    assert any(item['path'][:1] == ['recipe'] for item in request['candidate']['decision_contract']['substantive_delta'])
    queue = Queue(root / 'queue', report_root=root / 'reports/forge')
    first = queue.submit(request, campaign('a'))
    queue.submit(request, campaign('b'))
    state = queue.inspect()
    assert len(state['decision_rounds']) == 1
    assert state['jobs'][request['jobs'][0]['compatibility_key']]['subscribers'] == [first['request']['request_id'], next(
        key for key, value in state['submissions'].items() if value['request']['campaign_id'] == 'b')]
    with pytest.raises(ValueError, match='frozen bounded-round caps'):
        queue.submit(request, campaign('wide', cap=11))


@pytest.mark.parametrize('category,mutate', [
    ('recipe', lambda task: task['execution'].update(recipe_overrides={'lr': .008})),
    ('prior', lambda task: task['execution']['prior'].update(sigma=.04)),
    ('initialization', lambda task: task['execution'].update(fixed_initialization={'particles': [[1, 1]]})),
    ('host', lambda task: task['execution'].update(warmup_steps=123)),
    ('budget', lambda task: task['resources'].update(timeout_seconds=11)),
    ('sampling', lambda task: task['evaluation'].update(eval_output_noise='noisy')),
    ('evaluation', lambda task: task['evaluation'].update(thresholds=[['score', '>=', .2]])),
])
def test_contract_binds_changed_actual_task_conditions(ready, category, mutate):
    root, request, _ = ready
    changed = deepcopy(request)
    mutate(changed['tasks']['t1'])
    review = decisions.inspect_contract(root, changed)
    assert review['status'] == 'BLOCKED'
    assert review['expected']['candidate_binding_sha256'] != request['candidate']['decision_contract']['candidate_binding_sha256']


def test_full_fixed_and_component_initialization_policies_are_visible(ready):
    root, request, _ = ready
    task = deepcopy(request['tasks']['t1'])
    task['execution']['fixed_initialization'] = {'particles': [[0, 0]]}
    task['execution']['host_definition'] = {'initialization': {'generator': {'method': 'orthogonal', 'gain': 1}}}
    before = decisions._bindings(request['candidate'], {'t1': task}, request['protocol'], root)
    task['execution']['fixed_initialization']['particles'] = [[1, 1]]
    task['execution']['host_definition']['initialization']['generator']['gain'] = 2
    after = decisions._bindings(request['candidate'], {'t1': task}, request['protocol'], root)
    assert before['initialization']['t1']['fixed_initialization']['particles'] == [[0, 0]]
    assert before['initialization']['t1']['component_policies']['generator']['gain'] == 1
    assert any(item['path'][0] == 'initialization' for item in decisions.delta(before, after))


@pytest.mark.parametrize('field', ['execution_backend', 'runtime', 'compute_profiles', 'protocol', 'source'])
def test_contract_binds_the_scientific_cohort(ready, field):
    root, request, _ = ready
    changed = deepcopy(request)
    if field == 'execution_backend':
        changed[field] = 'cuda'
    elif field == 'source':
        changed[field]['digest'] = 'a' * 64
    else:
        changed[field]['changed'] = 'cohort'
    assert decisions.inspect_contract(root, changed)['status'] == 'BLOCKED'


def test_changed_control_prior_evidence_and_frozen_receipt_block_admission(ready):
    root, request, _ = ready
    changed = deepcopy(request)
    changed['decision_admission']['contract_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='mismatched frozen decision receipt'):
        decisions.validate_admission(changed, campaign(), root=root)
    source = root / 'reports/prior.json'
    source.write_text('{}')
    assert 'prior evidence' in '; '.join(decisions.inspect_contract(root, request)['blockers'])


@pytest.mark.parametrize('tamper', ['missing', 'v1', 'no_version'])
def test_new_idea_cannot_bypass_by_downgrading_version(ready, tamper):
    root, request, _ = ready
    changed = deepcopy(request)
    if tamper == 'missing':
        changed['candidate'].pop('decision_contract')
    else:
        changed['candidate'].pop('decision_contract')
        if tamper == 'v1':
            changed['candidate']['schema_version'] = 1
        else:
            changed['candidate'].pop('schema_version')
    with pytest.raises(ValueError, match='decision_contract|legacy'):
        decisions.validate_admission(changed, campaign(), root=root)


def test_legacy_declaration_and_saved_identity_remain_supported(checkout):
    pin_legacy(checkout)
    request = resolve_idea(checkout, 'base', execution_backend='cpu')
    decisions.validate_admission(request, campaign(), root=checkout)
    assert 'decision_admission' not in request
    original = deepcopy(request)
    path = checkout / 'configs/forge/ideas/base.json'
    value = read_json(path)
    value['hypothesis'] = 'Modified new declaration'
    atomic_json(path, value)
    with pytest.raises(ValueError, match='legacy v1 declaration changed'):
        decisions.validate_admission(request, campaign(), root=checkout)
    assert original == request  # No migration/reconstruction of saved identities.


@pytest.mark.parametrize('score,outcome', [(1, 'prediction_observed'), (.3, 'falsified'), (.6, 'inconclusive')])
def test_terminal_rules_stop_or_require_review_never_grant_qualification(ready, score, outcome):
    _, request, _ = ready
    result = decisions.evaluate(request, [row(score)])
    assert result['outcome'] == outcome
    assert result['next_action'] == decisions.OUTCOMES[outcome]
    assert result['execution_authorized'] is result['qualification_input'] is False
    assert result['causal_judgment'] == 'requires_review'


@pytest.mark.parametrize('rows', [[], [row(None)], [row(True)], [row(float('nan'))], [row(1, gate='INVALID')],
                                [row(1), row(.3)], [row(1), row(1)]])
def test_missing_nonfinite_invalid_and_duplicate_evidence_is_incomplete(ready, rows):
    _, request, _ = ready
    result = decisions.evaluate(request, rows)
    assert result['outcome'] == 'incomplete'
    if len(rows) > 1:
        assert result['duplicate_task_ids'] == ['t1']
        assert result['observed']['prediction']['value'] is None


def test_falsifier_has_precedence_if_signatures_overlap(ready):
    _, request, _ = ready
    request['candidate']['decision_contract']['falsifier'].update(op='>=', threshold=.5)
    assert decisions.evaluate(request, [row(1)])['outcome'] == 'falsified'


def test_cross_campaign_charges_and_reservations_remain_under_one_round_cap(ready):
    root, request, _ = ready
    queue = Queue(root / 'queue', report_root=root / 'reports/forge')
    queue.submit(request, campaign('a'))
    second = queue.submit(request, campaign('b'))['request']
    key = request['jobs'][0]['compatibility_key']
    with queue.state() as state:
        state['jobs'][key]['attempts'].append({'attempt_id': 'paid-failure'})
        state['charges'].append({'attempt_id': 'paid-failure', 'owner': {'campaign': 'a', 'revision': request['candidate_revision']}, 'seconds': 1})
        assert queue._available_budget(state, second, 10, job_key=key)[0] is False
        assert queue._available_budget(state, second, 9, job_key=key)[0] is True
        state['jobs'][key]['reserved_seconds'] = 9
        assert queue._available_budget(state, second, 1, job_key=key)[0] is False
        # Deferred tasks outside the round are governed by ordinary campaign caps.
        assert queue._available_budget(state, second, 1, job_key=request['jobs'][1]['compatibility_key'])[0] is True


def test_round_caps_cannot_be_reset_by_different_narrative_or_contract(ready):
    root, request, _ = ready
    queue = Queue(root / 'queue', report_root=root / 'reports/forge')
    queue.submit(request, campaign('a'))
    changed = deepcopy(request)
    changed['candidate']['decision_contract']['scope']['candidate_budget_seconds'] = 11
    review = decisions.inspect_contract(root, changed)
    changed['decision_admission'] = review['receipt']
    with pytest.raises(ValueError, match='round budget is immutable'):
        queue.submit(changed, campaign('b', cap=11))


def save_attempt(root, request, *, name='attempt', score=1, backend=None):
    request = deepcopy(request)
    if backend:
        request['execution_backend'] = backend
    item = row(score)
    item.update(compatibility_key=request['jobs'][0]['compatibility_key'], raw_status='completed', cost={'wall_seconds': 1})
    result = {'schema_version': 1, 'attempt_id': name, 'candidate_revision': request['candidate_revision'], 'task_results': [item]}
    path = root / 'reports/forge/attempts' / name
    atomic_json(path / 'request.json', {'request': request})
    atomic_json(path / 'result.json', result)
    atomic_json(path / 'evidence.json', {'result_hash': stable_hash(result), 'source': request['source'], 'runtime': request['runtime']})
    return path


def test_readout_publishes_frozen_decision_without_rewriting_receipts(ready, monkeypatch):
    root, request, _ = ready
    path = save_attempt(root, request)
    hashes = {file.name: file_hash(file) for file in path.glob('*.json')}
    monkeypatch.setattr(knowledge, 'compile_memory', lambda root: {})
    result = knowledge.readout(root, 'successor', 'Observed scalar signature', 'Causality requires diagnostics', 'Review saved diagnostics')
    assert result['decision_outcomes'][0]['outcome'] == 'prediction_observed'
    assert result['decision_outcomes'][0]['qualification_input'] is False
    assert hashes == {file.name: file_hash(file) for file in path.glob('*.json')}


def test_outcomes_keep_compute_cohorts_and_original_receipt_provenance_separate(ready):
    root, request, _ = ready
    save_attempt(root, request, name='cpu')
    save_attempt(root, request, name='cuda', backend='cuda', score=.3)
    attempts, _ = knowledge._attempts(root)
    results = decisions.concluded_outcomes(attempts)
    assert len(results) == 2
    assert {result['outcome'] for result in results} == {'incomplete', 'prediction_observed'}
    assert all(len(result['provenance']) == 1 for result in results)


@pytest.mark.parametrize('field,value', [('max_rounds', 2), ('candidate_budget_seconds', True)])
def test_unbounded_or_boolean_scope_is_rejected(ready, field, value):
    _, request, _ = ready
    contract = deepcopy(request['candidate']['decision_contract'])
    contract['scope'][field] = value
    with pytest.raises(ValueError):
        decisions.validate_shape(contract)


def test_v2_schema_requires_contract_and_no_automatic_terminal_continuation(ready):
    _, request, _ = ready
    candidate = deepcopy(request['candidate'])
    candidate.pop('decision_contract')
    with pytest.raises(ValueError, match='decision_contract'):
        validate_idea(candidate)
    contract = deepcopy(request['candidate']['decision_contract'])
    contract['terminal_rules']['prediction_observed'] = 'continue_training'
    with pytest.raises(ValueError, match='no automatic continuation'):
        decisions.validate_shape(contract)


def test_metadata_only_revision_cannot_reset_the_physical_round(ready):
    root, request, path = ready
    queue = Queue(root / 'queue', report_root=root / 'reports/forge')
    queue.submit(request, campaign('a'))
    idea = read_json(path)
    idea['api_changes'] = {'description': 'Documentation wording only; no API implementation change'}
    atomic_json(path, idea)
    changed = resolve_idea(root, 'successor', execution_backend='cpu')
    idea['decision_contract']['scope']['jobs_sha256'] = changed['decision_review']['expected']['jobs_sha256']
    atomic_json(path, idea)
    changed = resolve_idea(root, 'successor', execution_backend='cpu')
    assert changed['decision_review']['status'] == 'READY'
    assert changed['candidate_revision'] != request['candidate_revision']
    assert decisions.round_definition(changed)['round_id'] == decisions.round_definition(request)['round_id']
    with pytest.raises(ValueError, match='round budget is immutable'):
        queue.submit(changed, campaign('b'))


def test_job_namespace_tampering_cannot_reset_a_ready_cap(ready):
    root, request, _ = ready
    changed = deepcopy(request)
    job = changed['jobs'][0]
    job['science']['extra_namespace'] = 'new'
    job['compatibility_key'] = stable_hash(job['science'])
    with pytest.raises(ValueError, match='admission blocked'):
        decisions.validate_admission(changed, campaign(), root=root)
    assert decisions.round_definition(changed)['round_id'] == decisions.round_definition(request)['round_id']


def test_tight_cap_running_payer_transfer_keeps_one_physical_reservation(ready):
    root, request, _ = ready
    queue_root = root / 'queue'
    request = resolve_idea(root, 'successor', execution_backend='cpu', freeze_source=True, queue_root=queue_root)
    queue = Queue(queue_root, report_root=root / 'reports/forge')
    first = queue.submit(request, campaign('a'))['request']['request_id']
    second = queue.submit(request, {**campaign('b'), 'accept_shared_cost_transfer': True})['request']['request_id']
    claim = queue.claim([{'device': 'cpu', 'slot': 0, 'memory_mb': 100}])
    assert claim is not None
    queue.cancel(first)
    state = queue.inspect()
    job = state['jobs'][claim['job']['compatibility_key']]
    assert job['status'] == 'running'
    assert job['cost_owner']['request'] == second
    assert state['campaigns']['a']['reserved_seconds'] == 0
    assert state['campaigns']['b']['reserved_seconds'] == 10
    assert job['reserved_seconds'] == 10


def test_explicit_retry_keeps_original_paid_cost_across_campaigns(ready):
    root, _, path = ready
    idea = read_json(path)
    idea['decision_contract']['scope']['candidate_budget_seconds'] = 20
    atomic_json(path, idea)
    queue_root = root / 'queue'
    request = resolve_idea(root, 'successor', execution_backend='cpu', freeze_source=True, queue_root=queue_root)
    queue = Queue(queue_root, report_root=root / 'reports/forge')
    queue.submit(request, campaign('a', cap=20))
    queue.submit(request, campaign('b', cap=20))
    claim = queue.claim([{'device': 'cpu', 'slot': 0, 'memory_mb': 100}], campaign_filter='a')
    key = claim['job']['compatibility_key']
    atomic_json(Path(claim['worker']['directory']) / 'terminal.json', {
        'token': claim['worker']['token'], 'attempt_status': 'error', 'reason': 'Software infrastructure fixture',
        'elapsed_seconds': 11, 'result': {}})
    queue.collect()
    queue.retry(key, reason='Explicit infrastructure repair, retaining the original charge')
    assert queue.claim([{'device': 'cpu', 'slot': 0, 'memory_mb': 100}], campaign_filter='b') is None
    state = queue.inspect()
    assert sum(item['seconds'] for item in state['charges']) == 11
    assert state['jobs'][key]['retry_of']['attempt_id'] == claim['worker']['attempt']
    assert any('decision round budget' in (entry['reason'] or '') for entry in state['submissions'].values())


def test_registered_finite_search_preserves_original_boundaries_and_rejects_forgery(ready):
    from experiments.forge import configuration_search as search
    root, _, _ = ready
    spec = {'schema_version': 1, 'id': 'finite', 'trainer_family': 'family', 'base_candidate': 'base',
            'grid': {'lr': [.003, .005]}, 'tuning_through_tier': 1, 'view': 'stability', 'execution_backend': 'cpu',
            'protocol': 'screening', 'protocol_hash': stable_hash(read_json(root / 'configs/forge/protocols/screening.json')),
            'campaign': {'id': 'finite-search', 'candidate_budget_seconds': 10, 'budget_seconds': 20}}
    queue = Queue(root / 'queue', report_root=root / 'reports/forge')
    report = search.enqueue_search(root, root / 'queue', spec, queue=queue)
    assert report['submitted_count'] == 2
    request = next(iter(queue.inspect()['submissions'].values()))['request']
    decisions.validate_admission(request, spec['campaign'], root=root)
    changed = deepcopy(request)
    changed['source']['digest'] = 'a' * 64
    with pytest.raises(ValueError, match='exact bounded search registration'):
        decisions.validate_admission(changed, spec['campaign'], root=root)
    with pytest.raises(ValueError, match='exact bounded search registration'):
        decisions.validate_admission(request, {**spec['campaign'], 'id': 'unregistered'}, root=root)
    # A caller marker cannot replace the original registration.
    (root / 'reports/forge/configuration-search/finite.json').unlink()
    with pytest.raises(ValueError, match='exact bounded search registration'):
        decisions.validate_admission(request, spec['campaign'], root=root)


def test_complete_ready_scope_cannot_omit_an_authorized_task(ready):
    root, request, _ = ready
    changed = deepcopy(request)
    changed['through_tier'] = 2
    with pytest.raises(ValueError, match='every authorized task'):
        decisions.inspect_contract(root, changed)


@pytest.mark.parametrize('change', ['none', 'descriptor_only'])
def test_unexercised_or_descriptor_only_delta_does_not_authorize_spend(ready, change):
    root, _, path = ready
    idea = read_json(path)
    idea['recipe_overrides'] = {}
    if change == 'descriptor_only':
        task = read_json(root / 'configs/forge/tasks/t1.json')
        task['id'] = 't1-control'
        task['execution']['fixed_initialization'] = {'particles': [[0, 0]]}
        atomic_json(root / 'configs/forge/tasks/t1-control.json', task)
        idea['decision_contract']['control']['task_map'] = {'t1': 't1-control'}
    # Draft review exposes actual values without fabricating an exercised change.
    idea['decision_contract']['status'] = 'draft'
    atomic_json(path, idea)
    draft = resolve_idea(root, 'successor', execution_backend='cpu')
    expected = draft['decision_review']['expected']
    idea['decision_contract'].update(status='ready', candidate_binding_sha256=expected['candidate_binding_sha256'],
                                     substantive_delta=expected['substantive_delta'])
    idea['decision_contract']['control']['binding_sha256'] = expected['control_binding_sha256']
    idea['decision_contract']['scope']['jobs_sha256'] = expected['jobs_sha256']
    if not expected['substantive_delta']:
        idea['decision_contract']['substantive_delta'] = [{'path': ['recipe'], 'before': None, 'after': None}]
    atomic_json(path, idea)
    request = resolve_idea(root, 'successor', execution_backend='cpu')
    assert request['decision_review']['status'] == 'BLOCKED'
    assert any('exercised' in reason for reason in request['decision_review']['blockers'])


def test_evidence_identity_and_unsafe_selectors_cannot_substitute_a_different_trial(ready):
    root, request, _ = ready
    changed = deepcopy(request)
    evidence = changed['candidate']['decision_contract']['prior_evidence'][0]
    evidence['identity']['record_id'] = 'a-different-success'
    assert 'contradicts its bound source' in '; '.join(decisions.inspect_contract(root, changed)['blockers'])
    evidence['path'] = '../outside.json'
    assert 'unsafe' in '; '.join(decisions.inspect_contract(root, changed)['blockers'])
