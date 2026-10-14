"""Offline provider/control loop with actual minimal CPU numerical gates.

These fixture tiers are not the ParticleGAN gauntlet and grant no qualification.
"""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json,subprocess
import pytest
import torch

from experiments.forge.grok_loop import (quota_check,grok_command,parse_envelope,all_pass,
    validate_proposal,semantic_candidate,CandidateBranch,Loop,ForgeExecutor,freeze_suite)
from experiments.forge.contracts import atomic_json,read_json,stable_hash
from experiments.forge.studies import scaffold

ROOT=Path(__file__).resolve().parents[1]


def template():
    s=scaffold('fixture-study','fixture-candidate','fixture-control','software-fixture-tiers',hypothesis='Training-noise response improves fixed target learning.')
    s.update(status='ready',competing_explanation='A rename or longer horizon explains nothing.',
        prior_evidence=[dict(path='fixture-evidence.json',selector=[],identity={'record_id':'fixed-source-fixture'},use='motivation_only')],
        prediction=dict(task_id='quality',metric='abs_error',op='<=',threshold=.3,phase='final'),
        falsifier=dict(task_id='quality',metric='abs_error',op='>',threshold=.3,phase='final'))
    s['scope'].update(through_tier=3,execution_backend='cpu');s['campaign']=dict(id='fixture-campaign',budget_seconds=1000,candidate_budget_seconds=1000)
    return s


def proposal(index,lr,family='formulation'):
    c=read_json(next((ROOT/'configs/forge/configurations').glob('bcap-dualnorm--7beb*.json')))
    for key in ('configuration_id','resolved_configuration_recipe'):c.pop(key,None)
    c.update(id=f'fixture-candidate-{index}',recipe_preset='k3p',recipe_overrides=dict(lr=lr,optimizer_family=family),
        changed_factors=[f'lr={lr}',f'optimizer_family={family}'],mechanism_rationale='Change the actual global optimizer, with frozen data and gates.')
    if family=='adam':
        c['recipe_preset']='bcap';c['recipe_overrides']['reg_arm']='b_cap'
    s=template();s.update(id=f'fixture-study-{index}',candidate=c['id'])
    return dict(candidate=c,study=s)


def quota():return dict(route='grok_build_subscription',credit_bucket='included_weekly',model='verified-grok-build-model',
    eligible_first_party_model=True,API_fallback_enabled=False,auto_topup_enabled=False,on_demand_enabled=False,
    provider_hard_limit_verified=True,integration_isolation_verified=True,hooks_enabled=False,enabled_mcp_servers=[],
    source='approved_account_UI_read',included_remaining_units=10,remaining_paid_additional_usage_allowance_usd=0,
    observed_at=datetime.now(timezone.utc).isoformat())


@pytest.mark.parametrize('key,value',[('route','api'),('credit_bucket','additional_prepaid'),('auto_topup_enabled',True),
    ('remaining_paid_additional_usage_allowance_usd',1),('included_remaining_units',None),('hooks_enabled',True),
    ('enabled_mcp_servers',['unknown']),('provider_hard_limit_verified',False)])
def test_paid_route_is_fail_closed(key,value):
    proof=quota();proof[key]=value
    with pytest.raises(ValueError):quota_check(proof,model='verified-grok-build-model')


def test_stale_quota_and_timeout_flags_do_not_prove_money_cap():
    proof=quota();proof['observed_at']=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError):quota_check(proof,model=proof['model'])
    cmd=grok_command('grok',cwd='/tmp/fixture',prompt_file='/tmp/prompt',model='verified-grok-build-model')
    assert '--max-turns' in cmd and '--always-approve' not in cmd and '--no-auto-update' not in cmd
    assert '--json-schema' in cmd and cmd[cmd.index('--sandbox')+1]=='read-only'


def test_envelope_and_candidate_schema_are_separate():
    p=proposal(1,.01)
    assert parse_envelope(dict(is_error=False,structured_output=p))==p
    assert parse_envelope(dict(is_error=False,result=json.dumps(p)))==p
    for bad in [dict(is_error=True,result=json.dumps(p)),dict(output='ambiguous'),dict(structured_output={'candidate':{},'study':{},'execute':'unsafe'})]:
        with pytest.raises(ValueError):parse_envelope(bad)
    validate_proposal(p,template())
    changed=deepcopy(p);changed['study']['scope']['through_tier']=1
    with pytest.raises(ValueError):validate_proposal(changed,template())


def test_cosmetic_names_and_labels_do_not_create_new_semantics():
    p=proposal(1,.01);q=deepcopy(p);q['candidate']['id']='another-name';q['candidate']['hypothesis']='Different words'
    assert semantic_candidate(p['candidate'])==semantic_candidate(q['candidate'])


def test_blocked_missing_wrong_source_and_mixed_candidates_are_not_passes():
    row=dict(task_id='a',gate_status='PASS',independent_grade_verified=True,complete_original_task=True,candidate_revision='r',source_digest='s',seed=0)
    assert all_pass({'a':row},['a'],'r','s')
    assert not all_pass({'a':row},['a','b'],'r','s')
    for key,value in [('gate_status','BLOCKED'),('candidate_revision','other'),('source_digest','other'),('seed',1),('complete_original_task',False)]:
        changed=deepcopy(row);changed[key]=value
        assert not all_pass({'a':changed},['a'],'r','s')


class MockBuild:
    def __init__(self,proposals):self.proposals=iter(proposals);self.feedback=[]
    def propose(self,feedback,worktree,output):
        self.feedback.append(deepcopy(feedback));Path(output).mkdir(parents=True)
        p=next(self.proposals);atomic_json(Path(output)/'grok.json',dict(is_error=False,structured_output=p,model='offline-mock',usage={'paid_calls':0}))
        return parse_envelope(read_json(Path(output)/'grok.json'))


class CPUGates:
    """Actual fixed-law scalar optimization, not mocked PASS stamps."""
    def __init__(self):self.calls=[];self.states={}
    def active(self):return False
    def prepare(self,directory,candidate,study,suite):
        revision=stable_hash(candidate);self.states[revision]=(deepcopy(candidate),{})
        return dict(candidate_revision=revision,source_digest='software-fixture-source')
    def run_tier(self,binding,tier,names):
        torch.set_num_threads(1);self.calls.append((binding['candidate_revision'],tier))
        candidate,states=self.states[binding['candidate_revision']];recipe=candidate['recipe_overrides'];rows={}
        for name in names:
            base=name.split('-')[0]
            if tier==1:
                p=torch.nn.Parameter(torch.zeros(()));optimizer=(torch.optim.SGD([p],lr=recipe['lr']) if recipe['optimizer_family']=='formulation' else torch.optim.Adam([p],lr=recipe['lr'],betas=(0.,.999)))
                generator=torch.Generator().manual_seed(0)
            else:
                p,optimizer,generator=states[base]
            target=2. if base=='a' else 3.
            if tier>1:target=-target
            trace=[];steps=20 if tier==1 else 160
            for step in range(1,steps+1):
                data=torch.full((32,),target) if tier==1 else target+4*torch.randn(32,generator=generator)
                optimizer.zero_grad();loss=(p-data).square().mean();loss.backward();optimizer.step()
                if tier==1 and step==steps or tier>1 and step>=70 and step%10==0:trace.append(abs(float(p.detach())-target))
            passing=all(value<=.3 for value in trace)
            states[base]=(p,optimizer,generator)
            rows[name]=dict(task_id=name,gate_status='PASS' if passing else 'FAIL',independent_grade_verified=True,
                complete_original_task=True,candidate_revision=binding['candidate_revision'],source_digest=binding['source_digest'],seed=0,
                metrics=dict(abs_error=max(trace),observation_checks=len(trace),actual_optimizer_updates=steps),software_fixture=True)
        return rows


def toy_repo(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'config','user.name','CPU fixture'],check=True)
    subprocess.run(['git','-C',str(repo),'config','user.email','fixture@example.invalid'],check=True)
    (repo/'README.md').write_text('Software fixture only; no real gauntlet qualification.\n')
    subprocess.run(['git','-C',str(repo),'add','README.md'],check=True)
    subprocess.run(['git','-C',str(repo),'commit','-qm','Initialize software fixture'],check=True)
    return repo,subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()


def run_fixture(tmp_path,*,test_only=False):
    repo,head=toy_repo(tmp_path)
    suite=dict(view=dict(id='software-fixture-tiers',revision=1,assignments=[dict(task=name,importance='required',qualification_tier=tier) for tier,names in [(1,['a-quality','b-quality']),(2,['a-shift','b-shift']),(3,['a-endurance','b-endurance'])] for name in names]),sha256='software-fixture-contract')
    runs=tmp_path/'runs';runs.mkdir();cfg=dict(max_candidates=3,maximum_loop_seconds=30,stop_file=str(runs/'STOP'),runs_dir=str(runs),study_template=template())
    provider=MockBuild([proposal(1,.001),proposal(2,.5),proposal(3,.3,'adam')]);executor=CPUGates()
    branches=lambda i:CandidateBranch(repo,head,tmp_path/f'candidate-{i}')
    records=Loop(cfg,suite,provider,executor,branches).run(test_only=test_only)
    return records,provider,executor,cfg


def test_actual_CPU_gate_feedback_named_branches_and_no_pooled_passes(tmp_path):
    records,provider,executor,cfg=run_fixture(tmp_path)
    assert records[0]['outcome']=='TIER1_FAILED_BLOCKED_OR_INCOMPLETE'
    assert len(provider.feedback)==3 and provider.feedback[1][0]['tiers']['1']['a-quality']['gate_status']=='FAIL'
    assert (records[0]['candidate_revision'],2) not in executor.calls
    assert records[1]['tiers']['1']['a-quality']['gate_status']=='PASS'
    assert records[1]['outcome']=='TIER2_FAILED_BLOCKED_OR_INCOMPLETE'
    assert (records[1]['candidate_revision'],3) not in executor.calls
    for row in records:
        assert row['candidate_branch'].startswith('codex/grok-candidate-')
        assert Path(cfg['runs_dir']).joinpath('loop-checkpoint.json').exists()
    atomic_json(Path(cfg['runs_dir'])/'CPU-FIXTURE-READOUT.json',dict(software_fixture=True,actual_rows=records,paid_model_calls=0,research_training_starts=0))


def test_one_complete_test_does_not_start_unattended_batch(tmp_path):
    records,provider,executor,cfg=run_fixture(tmp_path,test_only=True)
    assert len(records)==1 and len(provider.feedback)==1


def test_actual_evaluator_source_implementation_is_protected(tmp_path):
    task=dict(evaluation=dict(sources={'scorer.py':'expected'}))
    # Direct guard runs before resolve/admission; a lying declaration is not enough.
    (tmp_path/'scorer.py').write_text('modified numerical evaluator')
    suite=dict(view={'id':'fixture'},tasks={'a':task},sha256='frozen')
    from unittest.mock import patch
    with patch('experiments.forge.grok_loop.freeze_suite',return_value=suite):
        with pytest.raises(ValueError,match='evaluator/host implementation'):
            ForgeExecutor({},ROOT).prepare(tmp_path,{'id':'c'}, {},suite)


def test_resume_preserves_deadline_and_rejects_uncertain_or_changed_contract(tmp_path):
    records,provider,executor,cfg=run_fixture(tmp_path,test_only=True)
    checkpoint=Path(cfg['runs_dir'])/'loop-checkpoint.json';saved=read_json(checkpoint)
    saved['phase']='PROPOSAL_OR_DISPATCH_UNRECONCILED';atomic_json(checkpoint,saved)
    suite=dict(view=dict(assignments=[]),sha256='software-fixture-contract')
    with pytest.raises(ValueError,match='reconciliation'):
        Loop(cfg,suite,provider,executor,None).run(resume=True)
    saved['phase']='BETWEEN_CANDIDATES';atomic_json(checkpoint,saved)
    with pytest.raises(ValueError,match='contract changed'):
        Loop(cfg,dict(suite,sha256='changed'),provider,executor,None).run(resume=True)
    saved['overall_deadline_unix']=0;atomic_json(checkpoint,saved)
    assert Loop(cfg,suite,provider,executor,None).run(resume=True)==records
    assert cfg['overall_deadline_unix']==0


def test_user_confirmed_disabled_additional_is_valid_cutoff():
    proof=quota();proof['source']='explicit_user_account_confirmation'
    assert quota_check(proof,model=proof['model'])


def test_env_does_not_copy_keys_or_provider_overrides(monkeypatch):
    from experiments.forge.grok_loop import grok_environment
    monkeypatch.delenv('XAI_API_KEY',raising=False);monkeypatch.delenv('GROK_API_KEY',raising=False)
    monkeypatch.setenv('ARBITRARY_TOKEN','fixture-not-a-real-secret');monkeypatch.setenv('GROK_MODELS_BASE_URL','https://wrong.invalid')
    env=grok_environment()
    assert 'ARBITRARY_TOKEN' not in env and 'GROK_MODELS_BASE_URL' not in env
    assert env['GROK_CLAUDE_HOOKS_ENABLED']=='0' and env['GROK_CURSOR_MCPS_ENABLED']=='0'
    monkeypatch.setenv('XAI_API_KEY','fixture-not-a-real-secret')
    with pytest.raises(ValueError,match='API-key'):grok_environment()


def test_certificates_require_exact_admitted_job_and_frozen_raw_hash():
    from experiments.forge.grok_loop import certified_receipts
    raw=dict(attempt_status='completed',result={'measured':1})
    grade=dict(gate_status='PASS');raw['grading']=dict(raw_hash=stable_hash(raw['result']),source_digest='s',grades={'a':grade})
    definition=dict(compatibility_key='k',science={'candidate_revision':'r'})
    row=dict(task_id='a',gate_status='PASS')
    state=dict(jobs={'k':dict(definition=definition,result=dict(raw=raw,candidate_revision='r',attempt_id='attempt',task_results=[row]))})
    binding=dict(candidate_revision='r',source_digest='s',request=dict(jobs=[definition],protocol={'seed':0}))
    assert all_pass(certified_receipts(binding,state,'dispatch'),['a'],'r','s')
    for where,key,value in [(raw['grading'],'raw_hash','wrong'),(raw['grading'],'source_digest','wrong'),(raw,'attempt_status','timeout')]:
        old=where[key];where[key]=value
        assert not all_pass(certified_receipts(binding,state,'dispatch'),['a'],'r','s')
        where[key]=old
    state['jobs']['k']['definition']={'different':'source'}
    assert not all_pass(certified_receipts(binding,state,'dispatch'),['a'],'r','s')


def test_window_checks_each_full_job_and_reopens_after_bounded_interval():
    from experiments.forge.codex_runner import coordinated_window_blocks
    cfg=dict(blocked_UTC_windows=[dict(start='2026-10-14T14:00:00Z',end='2026-10-14T15:00:00Z')])
    epoch=lambda hour:datetime(2026,10,14,hour,tzinfo=timezone.utc).timestamp()
    assert not coordinated_window_blocks(cfg,3600,now=epoch(7))
    assert coordinated_window_blocks(cfg,3601,now=epoch(13))
    assert coordinated_window_blocks(cfg,1,now=epoch(14))
    assert not coordinated_window_blocks(cfg,7200,now=epoch(15))


def test_bounded_funding_appends_only_ready_full_horizon_and_never_resets(tmp_path):
    from experiments.forge.codex_runner import Spending
    ledger=tmp_path/'old.json';queue=tmp_path/'q';path=tmp_path/'budget.json'
    atomic_json(ledger,dict(remaining_queue_seconds=2,remaining_wall_seconds=3,reservations=0,historical_receipts=['negative']))
    spending=Spending(path);spending.initialize(ledger,queue_root=queue,agent_seconds=0,agent_sessions=0)
    before=read_json(path)
    request=dict(candidate_revision='r',source={'digest':'s'},tasks={'a':dict(execution={},resources={'timeout_seconds':10})})
    ForgeExecutor({'budget_state':str(path)},ROOT)._fund_ready_candidate(request,['a'])
    after=read_json(path)
    assert after['spent']==before['spent'] and after['reservations']=={}
    assert after['limits']['queue']==10 and after['limits']['supervisor']==30
    assert after['agent_session_limit']==0 and after['events'][:-1]==before['events']
    assert after['events'][-1]['kind']=='bounded_local_compute_allowance_increase'
    old=path.read_bytes();ForgeExecutor({'budget_state':str(path)},ROOT)._fund_ready_candidate(request,['a'])
    assert path.read_bytes()==old


def test_process_isolation_verifies_cli_discovery_before_model(tmp_path):
    from experiments.forge.grok_loop import verify_isolation
    from unittest.mock import patch
    result=lambda value:type('Result',(),{'stdout':json.dumps(value)})()
    with patch('experiments.forge.grok_loop.subprocess.run',return_value=result(dict(grokVersion='fixture',hooks=[{'disabled':True}],mcpServers=[],plugins=[]))),patch('experiments.forge.grok_loop.grok_environment',return_value={}):
        assert verify_isolation('fixture',tmp_path,tmp_path)
    for section in ('hooks','mcpServers','plugins'):
        value=dict(hooks=[],mcpServers=[],plugins=[]);value[section]=[{'name':'enabled-fixture'}]
        with patch('experiments.forge.grok_loop.subprocess.run',return_value=result(value)),patch('experiments.forge.grok_loop.grok_environment',return_value={}):
            with pytest.raises(ValueError,match='no live model call'):verify_isolation('fixture',tmp_path,tmp_path)


def test_model_cannot_invent_prior_evidence_for_admission():
    p=proposal(1,.1);p['study']['prior_evidence'][0]['path']='fabricated.json'
    with pytest.raises(ValueError,match='prior_evidence'):validate_proposal(p,template())
