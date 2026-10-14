"""Software fixtures only: these assertions do not publish training results."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
import torch

from experiments.forge.contracts import stable_hash, file_hash
from experiments.forge.extended_results import (
    applicability, plan_extension, evaluate_extension, render_status, unique_attempt_cost, board_rows)
from experiments.forge.extended_continuation import verify_extension_prefix
from experiments.forge.state import state_digest

ROOT = Path(__file__).resolve().parents[1]


def fixture(name='gaussian1d_stability'):
    task = json.loads((ROOT / f'configs/forge/tasks/{name}.json').read_text())
    original = dict(attempt_id='original-software-fixture', gate_status='FAIL',
        candidate_revision='candidate-fixture', source_digest='source-fixture', protocol_sha256='protocol-fixture',
        checkpoint=dict(schema='public_context_v1',step=6000,sha256='file-fixture',state_sha256='state-fixture'),
        schedule_horizon=1000, executed_updates=6000, observation_steps=[42,84,1000,4000,6000],
        phase_evidence={'original_first_hold':'FAIL','frozen':'FAIL'})
    kind=task['evaluation']['kind']
    if kind=='gaussian_stability':
        schedule={'reacquisition':[42,84,125,167,209], 'shift_hold':[209+42*i for i in range(1,25)]}
    elif kind in {'gaussian_smoke','word_smoke'}:
        schedule={'confirmed_acquisition':[42,84,125]}
    elif kind=='native_accuracy':
        schedule={'terminal_coverage':[250,500,750,1000,1250], 'independent_holdout':[1250]}
    elif kind in {'ring_hold','ring_extension'}:
        schedule={'confirmed_acquisition':list(range(1,201)), 'first_later_hold':list(range(201,1401))}
        if kind=='ring_extension':schedule['endurance_extension']=list(range(1401,1701))
    else:
        schedule={'extended_hold':[42*i for i in range(1,task['evaluation']['observations']+1)]}
    scope='confirmed_acquisition' if kind in {'gaussian_smoke','word_smoke'} else 'eventual_recovery_window'
    policy=plan_extension(task,original,maximum_extra_updates=3000,maximum_wall_seconds=100,
        observation_extra_steps=schedule,registered_at='2026-10-14T01:00:00Z',scope=scope)['policy']
    rows=[]
    for point in policy['append_observation_schedule']:
        values={}
        for name,op,bound in policy['numerical_bounds_by_phase'][point['phase']]:
            # Resolve duplicate lower/upper bounds inside the frozen interval.
            if name not in values:values[name]=bound
            elif op=='>=':values[name]=max(values[name],bound)
            elif op=='<=':values[name]=min(values[name],bound)
        row=dict(**point,evaluation_sha256=policy['evaluation_sha256'],independent_evaluator=True,
            guards_passed=True,values=values,state_sha256='final-state-fixture',artifact_sha256='sample-fixture',
            independent_same_state_confirmation=True,confirmation_state_sha256='final-state-fixture',confirmation_values=deepcopy(values))
        if point['phase']=='independent_holdout':row['sample_count']=policy['independent_holdout_samples']
        rows.append(row)
    extra=rows[-1]['extra_step']
    continuation=dict(policy_sha256=stable_hash(policy),started_at='2026-10-14T01:00:01Z',
        parent_attempt_id=original['attempt_id'],candidate_revision=original['candidate_revision'],
        source_digest=original['source_digest'],protocol_sha256=original['protocol_sha256'],
        parent_checkpoint=deepcopy(original['checkpoint']),
        prefix_verification=dict(policy_sha256=stable_hash(policy),restored_exactly=True,history_reset=False,checkpoint_step=6000,checkpoint_sha256='file-fixture',checkpoint_state_sha256='state-fixture',restored_state_sha256='state-fixture'),
        schedule_horizon=1000,attempts=[dict(execution_attempt_id='execution-fixture',executed_updates=extra,wall_seconds=2.)],
        completed_steps=6000+extra,observations=rows,preserved_original_observation_steps=deepcopy(original['observation_steps']),
        original_phase_evidence_sha256=stable_hash(original['phase_evidence']),status='completed',
        final_checkpoint=dict(step=6000+extra,sha256='final-file-fixture',state_sha256='final-state-fixture'),final_state_verified=True)
    return task,original,policy,continuation


def result(parts):return evaluate_extension(*parts)['extended_result']


def test_later_success_never_regrades_the_deadline_failure():
    parts=fixture();before=deepcopy(parts);wrapped=evaluate_extension(*parts);r=wrapped['extended_result']
    assert parts==before and wrapped['original_result']==parts[1]
    assert r['extended_status']=='PASS' and r['original_grade']=='FAIL'
    assert r['benchmark_tasks_passed_original']==0 and r['original_qualification_changed'] is False
    assert r['phases']['reacquisition']['required_checks']==5 and r['phases']['shift_hold']['required_checks']==24
    assert 'FAIL (original deadline); PASS after +' in render_status(wrapped)


def test_first_hit_sustained_window_and_end_state_are_distinct_after_collapse():
    parts=list(fixture());row=parts[3]['observations'][-1];row['values']['cdf_ks']=0.9
    r=result(parts)
    assert r['extended_status']=='FAIL' and r['phases']['shift_hold']['first_hit_extra_step'] is not None
    assert r['phases']['shift_hold']['end_state']=='FAIL' and r['phases']['shift_hold']['first_sustained_extra_step'] is None
    smoke=list(fixture('gaussian1d_smoke'));smoke[3]['observations'][-1]['values']['cdf_ks']=0.9
    r=result(smoke)
    assert r['extended_status']=='FAIL' and r['absolute_first_confirmation_step']==6042


def test_interruption_is_incomplete_even_with_passing_partial_window():
    parts=list(fixture());c=parts[3];c['status']='timeout';c['completed_steps']=6209;c['observations']=c['observations'][:5]
    c['final_checkpoint']['step']=6209
    assert result(parts)['extended_status']=='INCOMPLETE'


@pytest.mark.parametrize('field', ['candidate_revision','source_digest','protocol_sha256','parent_attempt_id','schedule_horizon','policy_sha256'])
def test_identity_and_horizon_changes_are_rejected(field):
    parts=list(fixture());parts[3][field]='changed'
    with pytest.raises(ValueError):result(parts)


@pytest.mark.parametrize('edit', ['missing','reordered','duplicate','off_cadence','beyond_cap','changed_criteria','changed_original','changed_checkpoint','reset','rng','changed_positions','uncertified_final','wrong_end_state','unregistered'])
def test_evidence_and_original_contract_rejections(edit):
    parts=list(fixture());task,original,policy,c=parts
    if edit=='missing':c['observations'].pop(0)
    elif edit=='reordered':c['observations'][0:2]=list(reversed(c['observations'][0:2]))
    elif edit=='duplicate':c['observations'].insert(1,deepcopy(c['observations'][0]))
    elif edit=='off_cadence':c['observations'][0]['extra_step']+=1
    elif edit=='beyond_cap':c['completed_steps']=10000;c['attempts'][0]['executed_updates']=4000
    elif edit=='changed_criteria':task['evaluation']['thresholds'][-1][-1]=1
    elif edit=='changed_original':original['gate_status']='PASS'
    elif edit=='changed_checkpoint':c['parent_checkpoint']['sha256']='changed'
    elif edit=='reset':c['prefix_verification']['history_reset']=True
    elif edit=='rng':c['prefix_verification']['restored_exactly']=False
    elif edit=='changed_positions':c['preserved_original_observation_steps']=[42,6001]
    elif edit=='uncertified_final':c['final_state_verified']=False
    elif edit=='wrong_end_state':c['final_checkpoint']['state_sha256']='different'
    elif edit=='unregistered':c['started_at']=policy['registered_at']
    with pytest.raises(ValueError):result(parts)


def test_word_earlier_checkpoint_and_charged_retry_work_have_independent_clocks():
    task,original,policy,c=fixture('five_word_joint_smoke')
    original.update(executed_updates=20001,schedule_horizon=20000)
    original['checkpoint'].update(step=8334,schema='joint_word_v1')
    policy=plan_extension(task,original,maximum_extra_updates=3000,maximum_wall_seconds=100,
        observation_extra_steps={'confirmed_acquisition':[42,84,125]},registered_at=policy['registered_at'],scope='confirmed_acquisition')['policy']
    c.update(policy_sha256=stable_hash(policy),schedule_horizon=20000,parent_checkpoint=deepcopy(original['checkpoint']),
        completed_steps=8459,prefix_verification=dict(policy_sha256=stable_hash(policy),restored_exactly=True,history_reset=False,checkpoint_step=8334,checkpoint_sha256='file-fixture',checkpoint_state_sha256='state-fixture',restored_state_sha256='state-fixture'))
    c['final_checkpoint']['step']=8459
    c['attempts']=[dict(execution_attempt_id='interrupted-fixture',executed_updates=50,wall_seconds=1.),
        dict(execution_attempt_id='retry-fixture',executed_updates=125,wall_seconds=2.)]
    c['attempts'].append(deepcopy(c['attempts'][1]))
    r=result((task,original,policy,c))
    assert r['extended_status']=='PASS' and r['extra_training_updates']==125
    assert r['extra_executed_updates_including_retries']==175 and r['total_charged_training_updates']==20176
    assert r['absolute_first_hit_step']==8376 and r['absolute_end_step']==8459
    assert r['measured_extra_wall_seconds']==3.
    c['attempts'][0]['executed_updates']=3000
    with pytest.raises(ValueError,match='interrupted/retry'):result((task,original,policy,c))


def test_shared_groups_and_conflicting_costs():
    row=dict(execution_attempt_id='same',executed_updates=5,wall_seconds=2.)
    assert unique_attempt_cost([row,deepcopy(row)])==(5,2.)
    changed=deepcopy(row);changed['wall_seconds']=3.
    with pytest.raises(ValueError,match='conflicting'):unique_attempt_cost([row,changed])


@pytest.mark.parametrize('name', ['gaussian1d_smoke','five_word_joint_smoke'])
def test_smoke_requires_independent_same_state_numeric_confirmation(name):
    parts=list(fixture(name));parts[3]['observations'][-1]['confirmation_values']={}
    assert result(parts)['extended_status']=='FAIL'
    parts=list(fixture(name));parts[3]['observations'][-1]['confirmation_state_sha256']='changed'
    assert result(parts)['extended_status']=='FAIL'


def test_native_coverage_and_independent_100k_fidelity_are_both_required():
    parts=list(fixture('grid100'));assert result(parts)['extended_status']=='PASS'
    c=parts[3];hold=next(row for row in c['observations'] if row['phase']=='independent_holdout')
    hold['values']['accuracy.radial_ks']=1.
    assert result(parts)['extended_status']=='FAIL'
    hold['values']['accuracy.radial_ks']=0.;hold['sample_count']=99999
    assert result(parts)['extended_status']=='FAIL'
    hold['sample_count']=100000
    coverage=next(row for row in c['observations'] if row['phase']=='terminal_coverage')
    coverage['values']['coverage.min_modes']=99
    assert result(parts)['extended_status']=='FAIL'


@pytest.mark.parametrize('name', ['ring_hold','ring_extension'])
def test_ring_dense_first_later_hold_and_original_failure_are_preserved(name):
    parts=list(fixture(name));r=result(parts)
    assert r['extended_status']=='PASS' and r['original_grade']=='FAIL'
    assert r['phases']['first_later_hold']['required_checks']==1200
    parts[3]['original_phase_evidence_sha256']='changed'
    with pytest.raises(ValueError,match='first-hold'):result(parts)


def test_nontraining_and_unsupported_hosts_spend_nothing():
    task,original,_,_=fixture();task['evaluation']['kind']='clockfree_parity'
    assert applicability(task,original)['status']=='NOT_APPLICABLE'
    assert plan_extension(task,original,maximum_extra_updates=1,maximum_wall_seconds=1,observation_extra_steps=[],registered_at='x')['policy'] is None
    task['evaluation']['kind']='transfer_sustained';task['adapter']='transfer_behavior'
    assert applicability(task,original)['status']=='BLOCKED'
    task['adapter']='transfer_vector';original['checkpoint'].pop('schema')
    assert applicability(task,original)['status']=='BLOCKED'


def test_policy_cannot_shorten_required_windows_or_move_native_cadence():
    parts=list(fixture());parts[2]['required_windows']['shift_hold']=1
    with pytest.raises(ValueError):result(parts)
    task,original,policy,_=fixture('grid100')
    with pytest.raises(ValueError,match='cadence'):
        plan_extension(task,original,maximum_extra_updates=3000,maximum_wall_seconds=10,
            observation_extra_steps={'terminal_coverage':[251,500,750,1000,1250],'independent_holdout':[1250]},registered_at=policy['registered_at'])


def test_report_renderer_preserves_original_grade_and_all_units(tmp_path):
    task,original,policy,c=fixture();path=tmp_path/'reports/forge';path.mkdir(parents=True)
    (path/'extended-result-board.json').write_text(json.dumps({'rows':[dict(task=task,original=original,policy=policy,continuation=c)]}))
    text='\n'.join(board_rows(tmp_path))
    assert 'FAIL (original deadline); PASS after +' in text and 'updates' in text and 'wall seconds' in text
    assert original['gate_status']=='FAIL'


@pytest.mark.parametrize('schema', ['public_context_v1','joint_word_v1'])
def test_failure_eligible_prefix_checks_real_checkpoint_optimizer_rng_and_horizon(tmp_path,schema):
    optimizer={'state':{0:{'step':torch.tensor(7.),'exp_avg':torch.ones(2)}}}
    trainer={'completed_steps':7,'optimizers':[deepcopy(optimizer),deepcopy(optimizer)],'rng':torch.tensor([1,2],dtype=torch.uint8)}
    recipe={'total_steps':1000}
    state={'trainer':trainer,'recipe':recipe} if schema=='public_context_v1' else {'fixture':{'api_state':trainer,'recipe':recipe},'streams':{'rng':torch.tensor([3])}}
    parent=tmp_path/'parent.pt';restore=tmp_path/'restore.pt';torch.save(state,parent);torch.save(deepcopy(state),restore)
    original={'gate_status':'FAIL','checkpoint':dict(schema=schema,step=7,sha256=file_hash(parent),state_sha256=state_digest(state))}
    policy=dict(kind='failure_eligible_training_extension_v1',original_result_sha256=stable_hash(original),checkpoint=original['checkpoint'],original_schedule_horizon=1000)
    proof=verify_extension_prefix(policy,original,parent,restore)
    assert proof['restored_exactly'] and proof['original_grade']=='FAIL'
    assert proof['ordinary_passing_parent_dependency_weakened'] is False
    modified=deepcopy(state)
    target=modified['trainer'] if schema=='public_context_v1' else modified['fixture']['api_state']
    target['rng'][0]=9;torch.save(modified,restore)
    with pytest.raises(ValueError,match='restore'):verify_extension_prefix(policy,original,parent,restore)
    torch.save(state,restore);policy['original_schedule_horizon']=2000
    with pytest.raises(ValueError,match='horizon'):verify_extension_prefix(policy,original,parent,restore)
    policy['original_schedule_horizon']=1000;target=trainer;target['optimizers'][0]['state'][0]['step']=torch.tensor(6.)
    torch.save(state,parent);torch.save(state,restore);original['checkpoint'].update(sha256=file_hash(parent),state_sha256=state_digest(state));policy['original_result_sha256']=stable_hash(original)
    with pytest.raises(ValueError,match='optimizer'):verify_extension_prefix(policy,original,parent,restore)


def test_final_checkpoint_preserves_public_formulation_and_named_rng_contract(tmp_path):
    from experiments.forge.api import TRAINER_STREAM_BINDINGS
    from experiments.forge.extended_continuation import verify_extension_final
    recipe={'total_steps':1000}
    streams={name:torch.tensor([i,2],dtype=torch.uint8) for i,name in enumerate(TRAINER_STREAM_BINDINGS)}
    named=dict(manifest={'bindings':{name:dict(zip(('family','component','purpose'),binding))
                                   for name,binding in TRAINER_STREAM_BINDINGS.items()}},states=deepcopy(streams))
    trainer=dict(schema=1,recipe=deepcopy(recipe),optimizer_options={},penalty_options={},device='cpu',dtype='float32',
        requires_grad=True,initial_lrs=[.1,.1],completed_steps=7,streams=deepcopy(streams),
        optimizers=[{'state':{0:{'step':7,'exp_avg':torch.ones(2)}}} for _ in range(2)])
    parent=dict(schema=1,api_version=1,recipe=recipe,prior={'kind':'mog'},extensions={},initializer='deterministic_orthogonal',
        initialization={},trainer=trainer,streams=named,models={'G':torch.ones(2)})
    final=deepcopy(parent);final['trainer']['completed_steps']=9
    for opt in final['trainer']['optimizers']:opt['state'][0]['step']=9
    final['models']['G'][0]=2
    p=tmp_path/'parent.pt';f=tmp_path/'final.pt';torch.save(parent,p);torch.save(final,f)
    cp=dict(schema='public_context_v1',step=7,sha256=file_hash(p),state_sha256=state_digest(parent))
    fc=dict(step=9,sha256=file_hash(f),state_sha256=state_digest(final))
    policy=dict(checkpoint=cp,original_schedule_horizon=1000)
    assert verify_extension_final(policy,p,fc,f)['final_state_verified']
    for edit in ('horizon','optimizer','streams','formulation'):
        broken=deepcopy(final)
        if edit=='horizon':broken['recipe']['total_steps']=2000
        elif edit=='optimizer':broken['trainer']['optimizers'][0]['state'][0]['step']=8
        elif edit=='streams':broken['streams']['manifest']['bindings'][next(iter(TRAINER_STREAM_BINDINGS))]['purpose']='different'
        else:broken['trainer']['initial_lrs']=[.2,.2]
        torch.save(broken,f);fc.update(sha256=file_hash(f),state_sha256=state_digest(broken))
        with pytest.raises(ValueError):verify_extension_final(policy,p,fc,f)


def test_native_holdout_must_use_the_final_coverage_state():
    parts=list(fixture('grid100'))
    hold=next(row for row in parts[3]['observations'] if row['phase']=='independent_holdout')
    hold['state_sha256']='earlier-state'
    with pytest.raises(ValueError,match='end-state'):result(parts)


def test_nontraining_and_unsupported_rows_render_without_training_credit(tmp_path):
    task,original,_,_=fixture();task['evaluation']['kind']='api_contract'
    path=tmp_path/'reports/forge';path.mkdir(parents=True)
    (path/'extended-result-board.json').write_text(json.dumps({'rows':[dict(task=task,original=original)]}))
    text='\n'.join(board_rows(tmp_path));assert 'NOT_APPLICABLE' in text and '| 0 / 0 | 0 |' in text
    task['evaluation']['kind']='transfer_sustained';task['adapter']='transfer_behavior'
    (path/'extended-result-board.json').write_text(json.dumps({'rows':[dict(task=task,original=original)]}))
    assert 'BLOCKED' in '\n'.join(board_rows(tmp_path))


def test_json_round_trip_keeps_policy_and_first_hit_window_cost_units(tmp_path):
    task,original,policy,c=fixture();parts=json.loads(json.dumps((task,original,policy,c)))
    assert result(parts)==result((task,original,policy,c))
    path=tmp_path/'reports/forge';path.mkdir(parents=True)
    (path/'extended-result-board.json').write_text(json.dumps({'rows':[dict(task=task,original=original,policy=policy,continuation=c)]}))
    text='\n'.join(board_rows(tmp_path))
    assert 'First hit / sustained confirmation' in text and 'observation checks' in text
    assert 'Accepted / charged extra updates' in text and 'Extra measured wall seconds' in text


def test_cost_wall_cap_and_nonfinite_boolean_numeric_receipts_are_rejected():
    parts=list(fixture());parts[3]['attempts'][0]['wall_seconds']=101
    with pytest.raises(ValueError,match='bounded extension'):result(parts)
    parts=list(fixture());parts[3]['observations'][-1]['values']['cdf_ks']=False
    assert result(parts)['extended_status']=='FAIL'
    parts[3]['observations'][-1]['values']['cdf_ks']=float('nan')
    assert result(parts)['extended_status']=='FAIL'


def test_ring_first_later_hold_cannot_be_replaced_by_a_selected_suffix():
    task,original,policy,_=fixture('ring_hold')
    schedule={'confirmed_acquisition':list(range(1,201)), 'first_later_hold':list(range(201,1402))}
    with pytest.raises(ValueError,match='every observation'):
        plan_extension(task,original,maximum_extra_updates=3000,maximum_wall_seconds=10,
            observation_extra_steps=schedule,registered_at=policy['registered_at'])


def test_cli_writes_a_separate_result_and_keeps_input_bytes(monkeypatch,tmp_path):
    from experiments.forge.extended_results import main
    task,original,policy,c=fixture();path=tmp_path/'receipt.json';output=tmp_path/'result.json'
    path.write_text(json.dumps(dict(task=task,original=original,policy=policy,continuation=c)));before=path.read_bytes()
    monkeypatch.setattr('sys.argv',['extended_results',str(path),'--output',str(output)])
    main();data=json.loads(output.read_text())
    assert path.read_bytes()==before and data['original_result']['gate_status']=='FAIL'
    assert data['extended_result']['extended_status']=='PASS'
    monkeypatch.setattr('sys.argv',['extended_results',str(path),'--output',str(path)])
    with pytest.raises(SystemExit):main()


def test_word_hold_requires_independent_confirmation_at_every_new_check():
    parts=list(fixture('five_word_joint_hold'));assert result(parts)['extended_status']=='PASS'
    parts[3]['observations'][0]['confirmation_values']={}
    assert result(parts)['extended_status']=='FAIL'


def test_paired_adaptation_remains_blocked_without_a_coupled_grader_adapter():
    task=json.loads((ROOT/'configs/forge/tasks/target_shift_recovery.json').read_text())
    original=fixture()[1]
    assert applicability(task,original)['status']=='BLOCKED'


def test_absolute_checkpoint_clock_is_not_the_original_newly_charged_work():
    task,original,policy,c=fixture();original['executed_updates']=5000
    policy=plan_extension(task,original,maximum_extra_updates=3000,maximum_wall_seconds=100,
        observation_extra_steps={phase:[r['extra_step'] for r in policy['append_observation_schedule'] if r['phase']==phase]
                                 for phase in policy['required_windows']},registered_at=policy['registered_at'])['policy']
    c['policy_sha256']=stable_hash(policy);c['prefix_verification']['policy_sha256']=stable_hash(policy)
    r=result((task,original,policy,c))
    assert r['checkpoint_step']==6000 and r['total_charged_training_updates']==5000+r['extra_executed_updates_including_retries']


def test_reused_original_group_is_not_charged_again_as_extension():
    task,original,policy,c=fixture();original['execution_group_ids']=['execution-fixture']
    policy=plan_extension(task,original,maximum_extra_updates=3000,maximum_wall_seconds=100,
        observation_extra_steps={phase:[r['extra_step'] for r in policy['append_observation_schedule'] if r['phase']==phase]
                                 for phase in policy['required_windows']},registered_at=policy['registered_at'])['policy']
    c['policy_sha256']=stable_hash(policy);c['prefix_verification']['policy_sha256']=stable_hash(policy)
    with pytest.raises(ValueError,match='shared execution'):result((task,original,policy,c))


def test_behavior_training_without_a_sampled_prior_is_still_blocked():
    task,original,_,_=fixture();task['adapter']='transfer_behavior';task['execution']['prior_applicability']='not_sampled'
    assert applicability(task,original)['status']=='BLOCKED'
