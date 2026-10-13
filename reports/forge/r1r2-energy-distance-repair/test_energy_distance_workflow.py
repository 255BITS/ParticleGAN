"""Admission adversaries for fresh source, one energy law and honest budgets."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('energy_workflow', Path(__file__).with_name('workflow.py'))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
NEW_SOURCE = 'a' * 64


def manifest():
    return dict(schema_version=1, round=1, id=workflow.CAMPAIGN, max_candidates=2,
        baseline_role='non_saturating', paid_cap_seconds=8000, searches=[], source_digest=NEW_SOURCE,
        candidates=[dict(role=r, candidate=c, study=s) for r, (c, s) in workflow.CARDS.items()],
        campaign=dict(id=workflow.CAMPAIGN, budget_seconds=5040, candidate_budget_seconds=2520))


@pytest.mark.parametrize('change', [dict(round=2), dict(id='another'), dict(max_candidates=3),
    dict(paid_cap_seconds=8001), dict(baseline_role='baseline'), dict(searches=[dict(id='grid')]),
    dict(source_digest=None), dict(source_digest=workflow.OLD_SOURCE), dict(source_digest='unpinned')])
def test_unregistered_scope_and_unfrozen_or_old_source_are_rejected(change):
    value = manifest()
    value.update(change)
    with pytest.raises(ValueError): workflow.validate_manifest(value)


@pytest.mark.parametrize('field', ['budget_seconds', 'candidate_budget_seconds'])
@pytest.mark.parametrize('delta', [-1, 1])
def test_full_reservations_cannot_change(field, delta):
    value = manifest()
    value['campaign'][field] += delta
    with pytest.raises(ValueError, match='full task allowances'): workflow.validate_manifest(value)


@pytest.mark.parametrize('field', ['role', 'candidate', 'study'])
def test_exact_cards_and_new_studies_cannot_be_substituted(field):
    value = manifest()
    value['candidates'][-1][field] = value['candidates'][0][field]
    with pytest.raises(ValueError, match='exact declared cards'): workflow.validate_manifest(value)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    parent = dict(candidate=workflow.CARDS['non_saturating'][0], candidate_revision='old-ns-revision',
        source_digest=workflow.OLD_SOURCE, runtime_sha256=workflow.RUNTIME, protocol_sha256='seed0-public',
        view_sha256='original-view', execution_policy=dict(mode='complete_current_tier'),
        full_allowance_seconds=2520, jobs_sha256='old-jobs',
        task_contracts={t:'fixed-task' for t in workflow.shared.TASKS},
        task_keys={t:f'old-ns-{t}' for t in workflow.shared.TASKS})
    projections = {role: dict(**{k:v for k,v in deepcopy(parent).items()
                                 if k not in {'candidate', 'candidate_revision', 'source_digest', 'task_keys', 'jobs_sha256'}},
        candidate=card[0], candidate_revision='fresh-ns-revision' if role=='non_saturating' else 'energy-revision',
        source_digest=NEW_SOURCE, jobs_sha256=f'new-jobs-{role}',
        task_keys={t:f'fresh-{role}-{t}' for t in workflow.shared.TASKS}) for role, card in workflow.CARDS.items()}
    requests = {role:dict(role=role, request_id=f'current-{role}', campaign_id=workflow.CAMPAIGN,
        candidate=dict(id=card[0]), study=dict(id=card[1],campaign=deepcopy(manifest()['campaign'])),
        source=dict(digest=NEW_SOURCE,files={'native.py':'original-native'}),
        tasks={t:dict(evaluation=dict(sources={'native.py':'original-native'})) for t in workflow.shared.TASKS})
        for role, card in workflow.CARDS.items()}
    resolved = {role:dict(request=requests[role], projection=projections[role]) for role in workflow.ROLES}
    seconds = workflow.HISTORICAL_SECONDS / workflow.HISTORICAL_ATTEMPTS
    charges, jobs = [], {}
    for i in range(workflow.HISTORICAL_ATTEMPTS):
        aid, owner = f'original-{i}', dict(request=f'old-{i}', campaign='original', revision=f'rev-{i}')
        charges.append(dict(attempt_id=aid, owner=deepcopy(owner), seconds=seconds))
        jobs[f'historical-{i}'] = dict(attempts=[dict(attempt_id=aid)], cost_owner=deepcopy(owner),
            result=dict(attempt_id=aid, cost_owner=deepcopy(owner), task_results=[dict(cost=dict(wall_seconds=seconds))]))
    rows = [dict(role=r, task_id=t, candidate_id=r, candidate_revision=r, source_digest=workflow.OLD_SOURCE,
                 attempt_id=f'{r}-{t}') for r in ('baseline','non_saturating','weak_gamma') for t in workflow.shared.TASKS]
    publication = dict(complete=True, round_id=workflow.PREVIOUS_CAMPAIGN, certified_results=rows,
        task_cells=[dict(**r,status='FAIL') for r in rows], media=deepcopy(rows),
        aggregate_charge_attempt_ids=[c['attempt_id'] for c in charges], aggregate_paid_attempts=217,
        aggregate_total_paid_seconds=workflow.HISTORICAL_SECONDS)
    audit = dict(status='PASS', inputs_sha256=dict(publication='exact-publication', registration='exact-registration',
        aggregate217_charge_index=workflow.shared.stable_hash({c['attempt_id']:c for c in charges})))
    metrics = ['sample_count','modes','mass_tv','hq','component_covariance_error','component_min_eigen_ratio']
    ring = next(r for r in rows if r['role']=='weak_gamma' and r['task_id']=='ring16_acquisition')
    density = dict(publication_sha256='exact-publication', registration_sha256='exact-registration',
        source_digest=workflow.OLD_SOURCE, runtime_sha256=workflow.RUNTIME, recipe_count=1,
        recomputed_saved_observations=96, recomputed_metrics=metrics, diagnosis_rng_states_unchanged=True,
        new_optimizer_updates=0, new_sampling_draws=0,
        rings=[dict(**deepcopy(ring),recomputed_observations=96,max_absolute_gate_metric_difference={k:0. for k in metrics})])
    previous = dict(candidates=dict(non_saturating=parent),
        declarations={f"configs/forge/ideas/{parent['candidate']}.json":'exact-old-card'})
    state = dict(jobs=jobs, charges=charges, submissions={})
    recipe = dict(loss='non_saturating', loss_labels=(0.,1.,1.), reg_arm='a_r1r2', reg_coeff=.1,
        reg_coeff_end=None, reg_every=1, critic_r1_real=True, optimizer_family='adam', adam_variant='pytorch',
        amsgrad=True, betas=(.5,.99), prior_betas=(0.,.99), d_betas=None, beta2_end=None,
        lr=.0085, d_lr_mult=1., prior_lr_mult=1., eps=1e-8, d_eps=None, prior_eps=None,
        lr_schedule='constant', prior_update='learned', prior_reg=0., prior_l2=0., prior_proximal_strength=0.,
        kinetic_transport_weight=0., kinetic_transport_local_weight=0., input_noise_std=0., output_noise_std=0.,
        ema_decay=0., serve_average=0., critic_step_mode='none', constraint_geometry_mode='none', energy_distance_weight=0.)
    recipes = {r:({**deepcopy(recipe),'energy_distance_weight':float(r=='energy_u')},
                  {t:{**deepcopy(recipe),'energy_distance_weight':float(r=='energy_u'),'batch_size':128}
                   for t in workflow.shared.TASKS}) for r in workflow.ROLES}
    progress = dict(phase='admitted',requests={r:f'current-{r}' for r in workflow.ROLES},registration_sha256='exact-registration')
    registration = dict(candidates=deepcopy(projections), manifest=manifest())
    def read(path):
        if path.name=='state.json': return state
        if path.name=='results.json': return publication
        if path.name=='registration.json': return previous if str(path.parent).endswith(workflow.PREDECESSOR) else registration
        if path.name=='collection-audit.json': return audit
        if path.name=='density-audit.json': return density
        if path.name.endswith('-progress.json'): return progress
        raise AssertionError(path)
    def file_hash(path):
        if path.name=='results.json': return 'exact-publication'
        if path.name=='registration.json': return 'exact-registration'
        if path.name=='native.py': return 'original-native'
        if path.name==f"{parent['candidate']}.json": return 'exact-old-card'
        raise AssertionError(path)
    monkeypatch.setattr(workflow.shared,'read_json',read)
    monkeypatch.setattr(workflow.shared,'file_hash',file_hash)
    monkeypatch.setattr(workflow,'original_requests',lambda *_a:(resolved,{}))
    monkeypatch.setattr(workflow,'public_recipes',lambda request:recipes[request['role']])
    fresh_revisions={r:p['candidate_revision'] for r,p in projections.items()}
    monkeypatch.setattr(workflow,'source_revision',lambda request:fresh_revisions[request['role']])
    monkeypatch.setattr(workflow.shared,'request_projection',lambda request:projections[request['role']])
    return dict(root=tmp_path,queue=tmp_path/'queue',parent=parent,resolved=resolved,state=state,
        publication=publication,audit=audit,density=density,previous=previous,recipes=recipes,
        projections=projections,progress=progress,registration=registration)


def admission(c):
    return workflow.requests(c['root'],c['queue'],manifest())


def test_one_structural_delta_requires_both_fresh_source_recipes(campaign):
    assert admission(campaign)==(campaign['resolved'],{})


def test_actual_public_field_is_only_delta_on_global_and_all_seven_hosts():
    tasks={t:workflow.shared.read_json(workflow.ROOT/f'configs/forge/tasks/{t}.json') for t in workflow.shared.TASKS}
    protocol=workflow.shared.read_json(workflow.ROOT/'configs/forge/protocols/screening.json')
    resolved={r:dict(request=dict(candidate=workflow.shared.read_json(workflow.ROOT/f'configs/forge/ideas/{c}.json'),
                   tasks=tasks,protocol=protocol)) for r,(c,_s) in workflow.CARDS.items()}
    workflow.energy_delta(resolved)


@pytest.mark.parametrize('role',sorted(workflow.ROLES))
@pytest.mark.parametrize('field,value',[('id','separate-campaign'),('budget_seconds',2520),('candidate_budget_seconds',2519)])
def test_study_campaign_must_match_the_single_declared_full_campaign(campaign,role,field,value):
    campaign['resolved'][role]['request']['study']['campaign'][field]=value
    with pytest.raises(ValueError,match='actual studies must share'): admission(campaign)


def test_real_resolved_study_campaigns_share_the_full_fresh_source_comparison(monkeypatch):
    from experiments.forge import planning
    monkeypatch.setattr(planning,'snapshot_source',lambda *_a,**_kw:pytest.fail('Read-only study guard attempted a source freeze'))
    value=workflow.validate_manifest(workflow.shared.read_json(workflow.OUT/'manifest.json'))
    resolved,searches=workflow.original_requests(workflow.ROOT,workflow.ARCHIVE/'queue',value)
    assert set(resolved)==workflow.ROLES and not searches
    workflow.study_campaigns(resolved,value)
    assert all(r['request']['study_review']['status']=='READY' for r in resolved.values())
    assert all(r['projection']['full_allowance_seconds']==2520 for r in resolved.values())


@pytest.mark.parametrize('change',['incomplete','case','gif','duplicate_cell','campaign','audit','stale_audit','density','density_rng','density_updates','density_metric'])
def test_predecessor_closure_and_saved_only_density_are_required(campaign,monkeypatch,change):
    p,a,d=campaign['publication'],campaign['audit'],campaign['density']
    if change=='incomplete': p['complete']=False
    elif change=='case': p['certified_results'].pop()
    elif change=='gif': p['media'].pop()
    elif change=='duplicate_cell': p['task_cells'][-1]=deepcopy(p['task_cells'][0])
    elif change=='campaign': p['round_id']='other'
    elif change=='audit': a['status']='FAIL'
    elif change=='stale_audit': a['inputs_sha256']['publication']='stale'
    elif change=='density': d['recomputed_saved_observations']=95
    elif change=='density_rng': d['diagnosis_rng_states_unchanged']=False
    elif change=='density_updates': d['new_optimizer_updates']=1
    else: d['rings'][0]['max_absolute_gate_metric_difference']['hq']=.01
    monkeypatch.setattr(workflow,'original_requests',lambda *_a:pytest.fail('Unclosed evidence reached resolver'))
    with pytest.raises(ValueError,match='predecessor|Predecessor'): admission(campaign)


@pytest.mark.parametrize('role',sorted(workflow.ROLES))
@pytest.mark.parametrize('field',['source_digest','runtime_sha256','protocol_sha256','view_sha256','task_contracts','execution_policy','full_allowance_seconds'])
def test_fresh_source_still_preserves_every_original_question(campaign,role,field):
    campaign['projections'][role][field]='changed'
    with pytest.raises(ValueError,match='new source|fixed condition'): admission(campaign)


@pytest.mark.parametrize('revision',['old-ns-revision','other'])
def test_the_fresh_control_revision_must_be_rebuilt_for_the_new_source(campaign,revision):
    campaign['projections']['non_saturating']['candidate_revision']=revision
    with pytest.raises(ValueError,match='fresh-source candidate revision'): admission(campaign)


def test_original_ns_control_card_bytes_cannot_change(campaign,monkeypatch):
    original=workflow.shared.file_hash
    monkeypatch.setattr(workflow.shared,'file_hash',lambda p:'changed' if p.name.endswith('ams-m110-v1.json') else original(p))
    with pytest.raises(ValueError,match='parent card bytes'): admission(campaign)


@pytest.mark.parametrize('change',['file','snapshot'])
def test_pinned_original_native_hosts_cannot_be_bypassed(campaign,monkeypatch,change):
    if change=='file':
        original=workflow.shared.file_hash
        monkeypatch.setattr(workflow.shared,'file_hash',lambda p:'changed' if p.name=='native.py' else original(p))
    else: campaign['resolved']['energy_u']['request']['source']['files']['native.py']='changed'
    with pytest.raises(ValueError,match='audit|pinned original host'): admission(campaign)


@pytest.mark.parametrize('host',['global',*sorted(workflow.shared.TASKS)])
@pytest.mark.parametrize('field,value',[('reg_coeff',.01),('loss','relativistic'),('amsgrad',False),
    ('prior_update','frozen'),('kinetic_transport_weight',1.),('energy_distance_weight',.5)])
def test_no_second_actual_recipe_delta_or_host_weight_substitution(campaign,host,field,value):
    recipe=campaign['recipes']['energy_u'][0] if host=='global' else campaign['recipes']['energy_u'][1][host]
    recipe[field]=value
    with pytest.raises(ValueError,match='one raw energy-distance'): admission(campaign)


@pytest.mark.parametrize('host',['global',*sorted(workflow.shared.TASKS)])
@pytest.mark.parametrize('field,value',[('reg_coeff',0.),('reg_every',2),('critic_r1_real',False),('prior_betas',(.5,.99))])
def test_positive_every_update_r1r2_and_parent_role_histories_are_held(campaign,host,field,value):
    recipe=campaign['recipes']['non_saturating'][0] if host=='global' else campaign['recipes']['non_saturating'][1][host]
    recipe[field]=value
    with pytest.raises(ValueError,match='positive every-update R1/R2'): admission(campaign)


@pytest.mark.parametrize('change',['missing','duplicate','owner','seconds','missing_result','retry','walltime','result_owner'])
def test_all217_historical_attempts_and_original_owners_are_retained(campaign,change):
    s=campaign['state'];job=s['jobs']['historical-216']
    if change=='missing': s['charges'].pop()
    elif change=='duplicate': s['charges'].append(deepcopy(s['charges'][0]))
    elif change=='owner': s['charges'][0]['owner']['request']='new'
    elif change=='seconds': s['charges'][0]['seconds']+=1
    elif change=='missing_result': job['result']=None
    elif change=='retry': job['attempts'].append(dict(attempt_id='retry'))
    elif change=='walltime': job['result']['task_results'][0]['cost']['wall_seconds']+=1
    else: job['result']['cost_owner']['request']='new'
    with pytest.raises(ValueError,match='historical217|Duplicate paid|Historical'): admission(campaign)


def test_an_old_job_key_cannot_fill_even_the_fresh_control(campaign):
    campaign['projections']['non_saturating']['task_keys']['two_pole']='historical-216'
    with pytest.raises(ValueError,match='Fourteen fresh-source'): admission(campaign)


def test_a_foreign_paid_attempt_cannot_expand_the_campaign(campaign):
    campaign['state']['charges'].append(dict(attempt_id='foreign',seconds=1,owner={}))
    with pytest.raises(ValueError,match='fourteen-job contrast'): admission(campaign)


@pytest.mark.parametrize('role',sorted(workflow.ROLES))
@pytest.mark.parametrize('change',['no_admission','foreign_subscriber','retry','changed_projection'])
def test_resumption_cannot_rebind_or_retry_either_fresh_role(campaign,role,change):
    c=campaign;rid=c['progress']['requests'][role];key=next(iter(c['projections'][role]['task_keys'].values()))
    (c['root']/f'{workflow.CAMPAIGN}-progress.json').touch()
    job=dict(attempts=[],subscribers=[rid]);c['state']['jobs'][key]=job
    c['state']['submissions'][rid]=dict(request=deepcopy(c['resolved'][role]['request']))
    if change=='no_admission': (c['root']/f'{workflow.CAMPAIGN}-progress.json').unlink()
    elif change=='foreign_subscriber': job['subscribers'].append('another')
    elif change=='retry': job['attempts']=[dict(attempt_id='first'),dict(attempt_id='retry')]
    else: c['state']['submissions'][rid]['request']['role']='non_saturating' if role=='energy_u' else 'energy_u'
    with pytest.raises(ValueError,match='sole frozen admission'): admission(c)


@pytest.mark.parametrize('task',sorted(workflow.BUDGETS))
@pytest.mark.parametrize('delta',[-1,1])
def test_no_task_can_change_its_full_reservation(monkeypatch,task,delta):
    request=dict(through_tier=1,view=dict(assignments=[dict(task=t,qualification_tier=1) for t in workflow.shared.TASKS]),
        jobs=[dict(task_id=t,compatibility_key=t,budget_seconds=b) for t,b in workflow.BUDGETS.items()])
    monkeypatch.setattr(workflow,'original_projection',lambda _: {})
    next(j for j in request['jobs'] if j['task_id']==task)['budget_seconds']+=delta
    with pytest.raises(ValueError,match='word900'): workflow.request_projection(request)


def test_both_word_allowances_can_fit_serially_but_not_concurrently(monkeypatch):
    monkeypatch.setattr(workflow.shared.Queue,'_available_budget',lambda *_a,**_kw:(True,None))
    queue=object.__new__(workflow.ScopedQueue);queue.paid_cap_seconds=8000
    state=dict(charges=[dict(seconds=workflow.HISTORICAL_SECONDS)],jobs=dict(other=dict(reserved_seconds=900)))
    assert not queue._available_budget(state,{},900)[0]
    state['jobs']['other']['reserved_seconds']=0
    state['charges'].append(dict(seconds=100))
    assert queue._available_budget(state,{},900)[0]
    state['charges'].append(dict(seconds=500))
    assert not queue._available_budget(state,{},900)[0]


@pytest.mark.parametrize('mode',['claim','collect'])
def test_scoped_queue_preserves_foreign_history_and_charges(campaign,monkeypatch,mode):
    c=campaign;raw=c['state'];raw['campaigns']={};raw['events']=[]
    for role,rid in c['progress']['requests'].items():
        request=c['resolved'][role]['request'];request['jobs']=[dict(task_id=t,compatibility_key=k) for t,k in c['projections'][role]['task_keys'].items()]
        raw['submissions'][rid]=dict(request=request)
        for job in request['jobs']: raw['jobs'][job['compatibility_key']]=dict(status='pending')
    raw['submissions']['foreign']=dict(status='queued',request=dict(campaign_id='old',through_tier=2))
    raw['jobs']['foreign-running']=dict(status='running',reserved_seconds=100)
    before=deepcopy(raw['submissions']['foreign'])
    @contextmanager
    def state(_self): yield raw
    monkeypatch.setattr(workflow.OriginalBoundedQueue,'state',state)
    monkeypatch.setattr(workflow.shared.Queue,'_authorized',staticmethod(lambda _r,_j:True))
    queue=workflow.ScopedQueue(c['queue'],paid_cap_seconds=8000)
    with queue.scoped(mode),queue.state() as selected:
        assert set(selected['submissions'])==set(c['progress']['requests'].values())
        assert selected['charges'] is raw['charges']
        assert ('foreign-running' in selected['jobs'])==(mode=='claim')
        selected['submissions']['current-energy_u']['status']='running'
    assert raw['submissions']['foreign']==before
    assert raw['jobs']['foreign-running']['reserved_seconds']==100
    assert raw['submissions']['current-energy_u']['status']=='running'


def test_drain_cannot_select_historical_campaign(monkeypatch,tmp_path):
    queue=workflow.ScopedQueue(tmp_path,paid_cap_seconds=8000)
    with pytest.raises(ValueError,match='historical campaign'): queue.claim([],campaign_filter='old')
    calls=[]
    monkeypatch.setattr(workflow,'original_drain',lambda *_a,**kw:calls.append(kw))
    workflow.drain(queue,['0'])
    assert calls[0]['campaign']==workflow.CAMPAIGN
    with pytest.raises(ValueError,match='scoped new campaign'): workflow.drain(queue,['0'],campaign='old')


def test_storage_stop_only_pauses_the_new_campaign(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(workflow.shared.Queue,'pause',lambda _q,c,p:calls.append((c,p)))
    queue=workflow.ScopedQueue(tmp_path,paid_cap_seconds=8000)
    queue.pause('old',True);queue.pause(workflow.CAMPAIGN,True)
    assert calls==[(workflow.CAMPAIGN,True)]


@pytest.mark.parametrize('change',['absent','stream','init','batches'])
def test_native_collection_requires_actual_paired_conditions(change):
    protocol=dict(initial_tensors_sha256='same',actual_target_batches=['same'],rng_checkpoints=dict(all_named_streams_checkpointed=True))
    rows=[dict(task_id='two_pole',role=r,comparison_protocol=deepcopy(protocol)) for r in workflow.ROLES]
    if change=='absent': rows[0]['comparison_protocol']=None
    elif change=='stream': rows[0]['comparison_protocol']['rng_checkpoints']['all_named_streams_checkpointed']=False
    else: rows[0]['comparison_protocol']['initial_tensors_sha256' if change=='init' else 'actual_target_batches']='different'
    with pytest.raises(ValueError,match='actual initialization|Actual comparison condition'): workflow.comparison_audit(rows)


def test_one_measured_peer_does_not_receive_a_matched_comparison_pass():
    row=dict(task_id='two_pole',role='energy_u',comparison_protocol=dict(initial_tensors_sha256='same',
        actual_target_batches=['same'],rng_checkpoints=dict(all_named_streams_checkpointed=True)))
    assert workflow.comparison_audit([row])[0]['status']=='INCOMPLETE_MATCHED_COMPARISON'


def test_collect_requires_actual_training_media_before_any_work():
    with pytest.raises(ValueError,match='actual-training GIF'): workflow.collect(SimpleNamespace(media=False))


@pytest.mark.parametrize('change',['none','other_device','different_warmed','constructor','model','external_stream','trainer_stream','missing_stream','missing_branch','missing_frame','proof_hash'])
def test_own_clock_saved_tensor_and_stream_proof_is_required(monkeypatch,change):
    import torch
    names=[('data','clock_probe','training'),('data','target','training'),('eval','clock_probe','samples'),
        ('eval','sampler','samples'),('eval','target','reference'),('init','discriminator','construction'),
        ('init','generator','construction'),('init','prior','locations'),('noise','critic','input'),
        ('noise','generator','output'),('noise','models','stochastic_layers'),('noise','penalty','training'),
        ('noise','prior','gaussian'),('prior','latent','indices')]
    trainer_names=['latent_generator','penalty_generator','eval_generator','noise_generator','input_noise_generator','prior_noise_generator','model_generator']
    keys=[json.dumps([*n,'cuda:0']) for n in names]
    initial=dict(initializer='deterministic_orthogonal',
        initialization={k:dict(initial_state_sha256='f'*64) for k in ('generator','discriminator','prior')},
        streams=dict(states={k:torch.tensor([1,2],dtype=torch.uint8) for k in keys},
        manifest=dict(seed=0,bindings={k:dict(seed=17) for k in keys})),
        trainer=dict(models=dict(G=dict(weight=torch.tensor([.3]))),
                     streams={k:torch.tensor([1,2],dtype=torch.uint8) for k in trainer_names}))
    branches=['reference','step_label','horizon','evaluation_cadence','restart']
    proof=dict(initial=initial,branch_initial={b:deepcopy(initial) for b in branches},
               trajectories={b:[deepcopy(initial) for _ in range(4)] for b in branches})
    proofs={r:deepcopy(proof) for r in workflow.ROLES}
    selected=proofs['energy_u']
    if change=='other_device':
        for state in [selected['initial'],*selected['branch_initial'].values(),*(s for v in selected['trajectories'].values() for s in v)]:
            state['streams']['states']={k.replace('cuda:0','cuda:1'):v for k,v in state['streams']['states'].items()}
            state['streams']['manifest']['bindings']={k.replace('cuda:0','cuda:1'):v for k,v in state['streams']['manifest']['bindings'].items()}
    elif change=='different_warmed':
        for state in [selected['initial'],*selected['branch_initial'].values()]: state['trainer']['models']['G']['weight']+=1
    elif change=='constructor': selected['initial']['initialization']['prior']['initial_state_sha256']='e'*64
    elif change=='model': selected['initial']['trainer']['models']['G']['weight']+=1
    elif change=='external_stream': selected['initial']['streams']['states'][keys[0]][0]=3
    elif change=='trainer_stream': selected['initial']['trainer']['streams'][trainer_names[0]][0]=3
    elif change=='missing_stream': selected['trajectories']['reference'][0]['streams']['states'].pop(keys[0])
    elif change=='missing_branch': selected['branch_initial'].pop('restart')
    elif change=='missing_frame': selected['trajectories']['reference'].pop()
    monkeypatch.setattr(torch,'load',lambda p,**_kw:proofs[Path(p).parent.name])
    monkeypatch.setattr(workflow.shared,'file_hash',lambda p:'changed' if change=='proof_hash' else 'saved-proof')
    rows=[dict(role=r,task_id=workflow.CLOCK) for r in workflow.ROLES]
    exports=[dict(role=r,task_id=workflow.CLOCK,receipt=dict(source_inputs={f'/fixture/{r}/comparisons.pt':'saved-proof'})) for r in workflow.ROLES]
    if change in {'none','other_device','different_warmed'}:
        audit=workflow.clock_comparison(rows,exports)
        assert audit['status']=='PASS' and all(v['saved_states_checked']==26 for v in audit['own_clock_saved_proofs'].values())
        assert all(v['pre_warmup_tensor_copy_available'] is False for v in audit['own_clock_saved_proofs'].values())
    else:
        with pytest.raises(ValueError,match='Own-clock|own-clock'): workflow.clock_comparison(rows,exports)


def archive_fixture(tmp_path,monkeypatch):
    archive,original=tmp_path/'new',tmp_path/'original';archive.mkdir();(original/'queue').mkdir(parents=True)
    (archive/'queue').symlink_to(original/'queue',target_is_directory=True)
    prior=(original,*(tmp_path/f'prior-{i}' for i in range(7)))
    monkeypatch.setattr(workflow,'ARCHIVE',archive);monkeypatch.setattr(workflow,'ORIGINAL',original)
    monkeypatch.setattr(workflow,'PRIOR_ARCHIVES',prior)
    return archive,prior


def test_all_nine_archives_share_the_original_storage_cap(tmp_path,monkeypatch):
    archive,prior=archive_fixture(tmp_path,monkeypatch);visited=[]
    def storage(p): visited.append(p);return dict(bytes=workflow.shared.ARCHIVE_CAP//9+1,free_bytes=workflow.shared.FREE_MIN)
    monkeypatch.setattr(workflow,'original_storage',storage)
    with pytest.raises(ValueError,match='nine research archives'): workflow.storage(archive)
    assert visited==[archive,*prior]


def test_each_archive_retains_the_free_floor(tmp_path,monkeypatch):
    archive,prior=archive_fixture(tmp_path,monkeypatch)
    def storage(p):
        if p==prior[-1]: raise ValueError('Archive cap/headroom exhausted')
        return dict(bytes=1,free_bytes=workflow.shared.FREE_MIN)
    monkeypatch.setattr(workflow,'original_storage',storage)
    with pytest.raises(ValueError,match='headroom exhausted'): workflow.storage(archive)
