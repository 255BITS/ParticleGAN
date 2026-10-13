"""Admission adversaries: exact caches, public penalty delta, owners and bounds."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('weak_penalty_workflow', Path(__file__).with_name('workflow.py'))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def manifest():
    return dict(schema_version=1, round=1, id=workflow.CAMPAIGN, max_candidates=3,
        baseline_role='baseline', paid_cap_seconds=8000, searches=[],
        candidates=[dict(role=r, candidate=c, study=s) for r, (c, s) in workflow.CARDS.items()],
        campaign=dict(id=workflow.CAMPAIGN, budget_seconds=7560, candidate_budget_seconds=2520))


@pytest.mark.parametrize('change', [dict(round=2), dict(id='unregistered'), dict(max_candidates=4),
    dict(paid_cap_seconds=8001), dict(baseline_role='non_saturating'), dict(searches=[dict(id='extra')])])
def test_extra_scope_is_rejected(change):
    value = manifest()
    value.update(change)
    with pytest.raises(ValueError):
        workflow.validate_manifest(value)


@pytest.mark.parametrize('field', ['budget_seconds', 'candidate_budget_seconds'])
@pytest.mark.parametrize('delta', [-1, 1])
def test_full_reservations_cannot_change(field, delta):
    value = manifest()
    value['campaign'][field] += delta
    with pytest.raises(ValueError, match='full task allowance'):
        workflow.validate_manifest(value)


@pytest.mark.parametrize('field', ['role', 'candidate', 'study'])
def test_card_study_or_duplicate_role_substitution_is_rejected(field):
    value = manifest()
    value['candidates'][-1][field] = value['candidates'][0][field]
    with pytest.raises(ValueError, match='exact declared cards'):
        workflow.validate_manifest(value)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    projections = {role: dict(candidate=card[0], candidate_revision=f'revision-{role}',
        source_digest=workflow.SOURCE, runtime_sha256=workflow.RUNTIME, protocol_sha256='seed0-public',
        view_sha256='original-view', execution_policy=dict(mode='complete_current_tier'),
        full_allowance_seconds=2520, jobs_sha256=f'jobs-{role}',
        task_contracts={t: 'fixed-task' for t in workflow.shared.TASKS},
        task_keys={t: f'{role}-{t}' for t in workflow.shared.TASKS}) for role, card in workflow.CARDS.items()}
    resolved = {role: dict(projection=deepcopy(p), request=dict(role=role)) for role, p in projections.items()}
    controls = ('baseline', 'non_saturating')
    control_keys = [(r, t, projections[r]['task_keys'][t]) for r in controls for t in sorted(workflow.shared.TASKS)]
    charges, jobs, certificates = [], {}, {}
    seconds = workflow.HISTORICAL_SECONDS / 210
    for i in range(210):
        aid, owner = f'original-{i}', dict(request=f'old-{i}', campaign='original', revision=f'rev-{i}')
        charge = dict(attempt_id=aid, owner=owner, seconds=seconds)
        result = dict(attempt_id=aid, cost_owner=deepcopy(owner),
                      task_results=[dict(cost=dict(wall_seconds=seconds))])
        key = control_keys[i][2] if i < 14 else f'historical-key-{i}'
        jobs[key] = dict(attempts=[dict(attempt_id=aid)], cost_owner=deepcopy(owner), result=result)
        charges.append(charge)
        certificates[aid] = deepcopy(result)
    rows = []
    previous_roles = ('baseline', 'ams_m110', 'non_saturating')
    for role in previous_roles:
        for task in sorted(workflow.shared.TASKS):
            p = projections.get(role, dict(candidate=role, candidate_revision=role, source_digest=workflow.SOURCE))
            aid = jobs[p['task_keys'][task]]['result']['attempt_id'] if role in controls else f'other-{role}-{task}'
            rows.append(dict(role=role, task_id=task, candidate_id=p['candidate'],
                             candidate_revision=p['candidate_revision'], source_digest=p['source_digest'], attempt_id=aid))
    publication = dict(complete=True, round_id=workflow.PREVIOUS_CAMPAIGN,
        certified_results=rows, task_cells=[dict(**r, status='FAIL') for r in rows], media=deepcopy(rows),
        aggregate_charge_attempt_ids=[c['attempt_id'] for c in charges], aggregate_paid_attempts=210,
        aggregate_total_paid_seconds=workflow.HISTORICAL_SECONDS,
        exact_certificate_hashes={aid: {n:'exact-certificate' for n in ('request', 'result', 'evidence')}
                                  for aid in certificates})
    audit = dict(status='PASS', inputs_sha256=dict(publication='exact-publication', registration='exact-registration',
        aggregate210_charge_index=workflow.shared.stable_hash({c['attempt_id']:c for c in charges})))
    density = dict(publication_sha256='exact-publication', registration_sha256='exact-registration',
        source_digest=workflow.SOURCE, runtime_sha256=workflow.RUNTIME, recipe_count=1,
        recomputed_metrics=['sample_count','modes','mass_tv','hq','component_covariance_error','component_min_eigen_ratio'],
        recomputed_saved_observations=96, diagnosis_rng_states_unchanged=True,
        new_optimizer_updates=0, new_sampling_draws=0,
        rings=[dict(**deepcopy(next(r for r in rows if r['role']=='non_saturating'
                                   and r['task_id']=='ring16_acquisition')),
                    recomputed_observations=96, max_absolute_gate_metric_difference={k:0. for k in
                    ['sample_count','modes','mass_tv','hq','component_covariance_error','component_min_eigen_ratio']})])
    previous = dict(candidates=deepcopy(projections))
    previous['candidates']['weak_gamma']['task_keys'] = {t:f'old-other-{t}' for t in workflow.shared.TASKS}
    state = dict(jobs=jobs, charges=charges, submissions={})
    parent = dict(loss='non_saturating', reg_arm='a_r1r2', reg_coeff=.1, reg_every=1,
        critic_r1_real=True, optimizer_family='adam', adam_variant='pytorch', amsgrad=True,
        betas=(.5,.99), prior_betas=(0.,.99), d_betas=None, lr=.0085, d_lr_mult=1., prior_lr_mult=1.,
        eps=1e-8, d_eps=None, prior_eps=None, lr_schedule='constant', prior_update='learned',
        prior_reg=0., prior_l2=0., kinetic_transport_weight=0., kinetic_transport_local_weight=0.,
        output_noise_std=0., input_noise_std=0., total_steps=None)
    recipes = {r:(deepcopy(parent), {t:dict(**deepcopy(parent), batch_size=128) for t in workflow.shared.TASKS})
               for r in controls + ('weak_gamma',)}
    recipes['weak_gamma'][0]['reg_coeff'] = .01
    for recipe in recipes['weak_gamma'][1].values(): recipe['reg_coeff'] = .01
    def read(path):
        if path.name == 'state.json': return state
        if path.name == 'results.json': return publication
        if path.name == 'registration.json': return previous
        if path.name == 'collection-audit.json': return audit
        if path.name == 'density-audit.json': return density
        if path.name == 'result.json': return certificates[path.parent.name]
        if path.name.endswith('-progress.json'): return progress
        raise AssertionError(path)
    def file_hash(path):
        if path.name == 'results.json': return 'exact-publication'
        if path.name == 'registration.json': return 'exact-registration'
        return 'exact-certificate'
    progress = dict(requests=dict(weak_gamma='current'))
    monkeypatch.setattr(workflow.shared, 'read_json', read)
    monkeypatch.setattr(workflow.shared, 'file_hash', file_hash)
    monkeypatch.setattr(workflow, 'original_requests', lambda *_a: (resolved, {}))
    monkeypatch.setattr(workflow, 'public_recipes', lambda request: recipes[request['role']])
    return dict(root=tmp_path, queue=tmp_path/'queue', projections=projections, resolved=resolved,
                state=state, publication=publication, audit=audit, density=density,
                previous=previous, recipes=recipes, certificates=certificates, progress=progress)


def admission(c):
    return workflow.requests(c['root'], c['queue'], manifest())


def test_matching_controls_and_one_penalty_delta_need_no_admission(campaign):
    assert admission(campaign) == (campaign['resolved'], {})


@pytest.mark.parametrize('change', ['incomplete', 'case', 'gif', 'duplicate_cell', 'wrong_campaign',
                                  'failed_audit', 'stale_publication', 'stale_registration'])
def test_predecessor_closure_is_required_before_request_resolution(campaign, monkeypatch, change):
    p, a = campaign['publication'], campaign['audit']
    if change == 'incomplete': p['complete'] = False
    elif change == 'case': p['certified_results'].pop()
    elif change == 'gif': p['media'].pop()
    elif change == 'duplicate_cell': p['task_cells'][-1] = deepcopy(p['task_cells'][0])
    elif change == 'wrong_campaign': p['round_id'] = 'another'
    elif change == 'failed_audit': a['status'] = 'FAIL'
    else: a['inputs_sha256']['publication' if change == 'stale_publication' else 'registration'] = 'changed'
    monkeypatch.setattr(workflow, 'original_requests', lambda *_a: pytest.fail('Unclosed predecessor reached preparation'))
    with pytest.raises(ValueError, match='predecessor|Predecessor'): admission(campaign)


@pytest.mark.parametrize('change', ['publication', 'registration', 'missing_panels', 'metric', 'rng', 'updates', 'draws'])
def test_density_receipt_cannot_be_stale_or_add_training(campaign, change):
    d = campaign['density']
    if change in ('publication', 'registration'): d[f'{change}_sha256'] = 'stale'
    elif change == 'missing_panels': d['recomputed_saved_observations'] -= 1
    elif change == 'metric': d['rings'][0]['max_absolute_gate_metric_difference']['hq'] = .01
    elif change == 'rng': d['diagnosis_rng_states_unchanged'] = False
    else: d['new_optimizer_updates' if change == 'updates' else 'new_sampling_draws'] = 1
    with pytest.raises(ValueError, match='density audit'): admission(campaign)


@pytest.mark.parametrize('role', ['baseline', 'non_saturating'])
@pytest.mark.parametrize('field', ['candidate_revision', 'jobs_sha256', 'task_keys'])
def test_cached_scientific_projection_substitution_is_rejected(campaign, role, field):
    campaign['resolved'][role]['projection'][field] = 'changed'
    with pytest.raises(ValueError, match='exact measured'): admission(campaign)


@pytest.mark.parametrize('field', ['source_digest', 'runtime_sha256', 'protocol_sha256', 'view_sha256',
                                 'task_contracts', 'execution_policy', 'full_allowance_seconds'])
def test_new_candidate_cannot_change_source_tasks_gates_or_protocol(campaign, field):
    campaign['resolved']['weak_gamma']['projection'][field] = 'changed'
    with pytest.raises(ValueError, match='source and runtime|fixed condition'): admission(campaign)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'owner', 'seconds', 'missing_result', 'retry', 'walltime', 'result_owner'])
def test_all210_original_paid_attempts_and_owners_are_preserved(campaign, change):
    s = campaign['state']
    job = s['jobs']['historical-key-209']
    if change == 'missing': s['charges'].pop()
    elif change == 'duplicate': s['charges'].append(deepcopy(s['charges'][0]))
    elif change == 'owner': s['charges'][0]['owner']['request'] = 'new-owner'
    elif change == 'seconds': s['charges'][0]['seconds'] += 1
    elif change == 'missing_result': job['result'] = None
    elif change == 'retry': job['attempts'].append(dict(attempt_id='retry'))
    elif change == 'walltime': job['result']['task_results'][0]['cost']['wall_seconds'] += 1
    else: job['result']['cost_owner']['request'] = 'replacement'
    with pytest.raises(ValueError, match='historical210|Duplicate paid|Historical'): admission(campaign)


@pytest.mark.parametrize('role', ['baseline', 'non_saturating'])
@pytest.mark.parametrize('change', ['attempt', 'candidate', 'certificate'])
def test_cached_case_or_certificate_substitution_is_rejected(campaign, role, change):
    row = next(r for r in campaign['publication']['certified_results'] if r['role'] == role)
    if change == 'attempt': row['attempt_id'] = 'replacement'
    elif change == 'candidate': row['candidate_id'] = 'other-recipe'
    else: campaign['certificates'][row['attempt_id']]['altered'] = True
    with pytest.raises(ValueError, match='cached cases|certificate/result'): admission(campaign)


@pytest.mark.parametrize('host', ['global', *sorted(workflow.shared.TASKS)])
@pytest.mark.parametrize('field,value', [('reg_coeff',.02), ('loss','relativistic'), ('amsgrad',False), ('prior_betas',(.5,.99)),
                                      ('prior_update','frozen'), ('batch_size',64), ('output_noise_std',.1)])
def test_any_additional_actual_public_recipe_delta_is_rejected(campaign, host, field, value):
    recipes = campaign['recipes']['weak_gamma']
    recipe = recipes[0] if host == 'global' else recipes[1][host]
    recipe[field] = value
    with pytest.raises(ValueError, match='one actual public positive penalty-field delta'): admission(campaign)


@pytest.mark.parametrize('field,value', [('reg_coeff',0.), ('reg_every',2), ('reg_arm','bcap'),
                                      ('critic_r1_real',False), ('amsgrad',False), ('lr',.004)])
def test_parent_must_keep_positive_every_update_r1r2_and_exact_optimizer(campaign, field, value):
    campaign['recipes']['non_saturating'][0][field] = value
    with pytest.raises(ValueError, match='every-update paired R1/R2'): admission(campaign)


def test_existing_new_job_requires_this_rounds_sole_admission(campaign):
    key = next(iter(campaign['projections']['weak_gamma']['task_keys'].values()))
    campaign['state']['jobs'][key] = dict(attempts=[], subscribers=['another'])
    with pytest.raises(ValueError, match='sole frozen admission'): admission(campaign)


@pytest.mark.parametrize('change', ['retry', 'other_subscriber', 'changed_projection'])
def test_resuming_cannot_retry_or_rebind_new_jobs(campaign, monkeypatch, change):
    c = campaign
    (c['root']/f'{workflow.CAMPAIGN}-progress.json').touch()
    key = next(iter(c['projections']['weak_gamma']['task_keys'].values()))
    job = dict(attempts=[], subscribers=['current'])
    c['state']['jobs'][key] = job
    c['state']['submissions']['current'] = dict(request=c['resolved']['weak_gamma']['request'])
    projected = deepcopy(c['projections']['weak_gamma'])
    monkeypatch.setattr(workflow.shared, 'request_projection', lambda _r: projected)
    if change == 'retry': job['attempts'] = ['first', 'retry']
    elif change == 'other_subscriber': job['subscribers'] = ['another']
    else: projected['task_contracts'] = 'changed'
    with pytest.raises(ValueError, match='sole frozen admission'): admission(c)


@pytest.mark.parametrize('attempts', [[], [dict(attempt_id='sole-current-attempt')]])
def test_exact_frozen_admission_can_resume_without_recreating_jobs(campaign, monkeypatch, attempts):
    c = campaign
    (c['root']/f'{workflow.CAMPAIGN}-progress.json').touch()
    key = next(iter(c['projections']['weak_gamma']['task_keys'].values()))
    c['state']['jobs'][key] = dict(attempts=attempts, subscribers=['current'])
    c['state']['submissions']['current'] = dict(request=c['resolved']['weak_gamma']['request'])
    monkeypatch.setattr(workflow.shared, 'request_projection', lambda _r:c['projections']['weak_gamma'])
    assert admission(c) == (c['resolved'], {})


@pytest.mark.parametrize('change', ['source', 'runtime', 'omitted_metric', 'other_attempt', 'partial_panel_count'])
def test_density_must_include_six_exact_metrics_for_the_original_ns_attempt(campaign, change):
    d = campaign['density']
    if change == 'source': d['source_digest'] = 'changed'
    elif change == 'runtime': d['runtime_sha256'] = 'changed'
    elif change == 'omitted_metric': d['rings'][0]['max_absolute_gate_metric_difference'].pop('hq')
    elif change == 'other_attempt': d['rings'][0]['attempt_id'] = 'other'
    else: d['rings'][0]['recomputed_observations'] = 95
    with pytest.raises(ValueError, match='density audit'): admission(campaign)


def test_a_foreign_paid_attempt_cannot_expand_the_campaign(campaign):
    campaign['state']['charges'].append(dict(attempt_id='foreign', seconds=1, owner={}))
    with pytest.raises(ValueError, match='one registered seven-job'): admission(campaign)


def test_paid_cap_includes_prior_costs_and_concurrent_full_allowances(monkeypatch):
    monkeypatch.setattr(workflow.shared.Queue, '_available_budget', lambda *_a, **_kw:(True,None))
    queue = object.__new__(workflow.shared.BoundedQueue)
    queue.paid_cap_seconds = 8000
    state = dict(charges=[dict(seconds=workflow.HISTORICAL_SECONDS)], jobs=dict(other=dict(reserved_seconds=900)))
    assert queue._available_budget(state, {}, 300)[0]
    assert not queue._available_budget(state, {}, 900)[0]


def archive_fixture(tmp_path, monkeypatch):
    archive, original = tmp_path/'new', tmp_path/'original'
    archive.mkdir()
    (original/'queue').mkdir(parents=True)
    (archive/'queue').symlink_to(original/'queue', target_is_directory=True)
    prior = (original, *(tmp_path/f'prior-{i}' for i in range(6)))
    monkeypatch.setattr(workflow, 'ARCHIVE', archive)
    monkeypatch.setattr(workflow, 'ORIGINAL', original)
    monkeypatch.setattr(workflow, 'PRIOR_ARCHIVES', prior)
    return archive, prior


def test_only_the_exact_original_queue_symlink_is_allowed(tmp_path, monkeypatch):
    archive, _ = archive_fixture(tmp_path, monkeypatch)
    (archive/'queue').unlink()
    (archive/'queue').mkdir()
    with pytest.raises(ValueError, match='exact original queue'): workflow.storage(archive)


def test_all_eight_archives_share_one_storage_cap(tmp_path, monkeypatch):
    archive, prior = archive_fixture(tmp_path, monkeypatch)
    visited = []
    def storage(path):
        visited.append(path)
        return dict(bytes=workflow.shared.ARCHIVE_CAP//8+1, free_bytes=workflow.shared.FREE_MIN)
    monkeypatch.setattr(workflow, 'original_storage', storage)
    with pytest.raises(ValueError, match='eight research archives'): workflow.storage(archive)
    assert visited == [archive, *prior]


def test_each_prior_archive_must_retain_free_space(tmp_path, monkeypatch):
    archive, prior = archive_fixture(tmp_path, monkeypatch)
    def storage(path):
        if path == prior[-1]: raise ValueError('Archive cap/headroom exhausted')
        return dict(bytes=1, free_bytes=workflow.shared.FREE_MIN)
    monkeypatch.setattr(workflow, 'original_storage', storage)
    with pytest.raises(ValueError, match='headroom exhausted'): workflow.storage(archive)
