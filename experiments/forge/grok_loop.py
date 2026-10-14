"""Bounded Grok Build proposal/Forge loop. Plan by default, no API fallback.

A candidate passes a tier only when every original required task has a certified
PASS at its one frozen source/revision. No cross-candidate success projection.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import math
import os
import signal
import subprocess
import time
import uuid

from .contracts import atomic_json,read_json,stable_hash,file_hash,identifier

PROPOSAL_SCHEMA={"type":"object","additionalProperties":False,"required":["candidate","study"],
    "properties":{"candidate":{"type":"object"},"study":{"type":"object"}}}


def quota_check(proof, *, model, now=None):
    """Nonsecret account facts must come from an approved account read, not a proposal."""
    required={'route':'grok_build_subscription','credit_bucket':'included_weekly','model':model,
              'eligible_first_party_model':True,'API_fallback_enabled':False,'auto_topup_enabled':False,
              'on_demand_enabled':False,'provider_hard_limit_verified':True,
              'integration_isolation_verified':True,'hooks_enabled':False,'enabled_mcp_servers':[]}
    for key,expected in required.items():
        if proof.get(key)!=expected:raise ValueError('paid Build test BLOCKED: unverified account/isolation fact '+key)
    if proof.get('source') not in {'approved_account_UI_read','approved_provider_quota_read','explicit_user_account_confirmation'}:
        raise ValueError('account limit must be provider/account evidence, not model content')
    remaining=proof.get('included_remaining_units')
    if type(remaining) not in (int,float) or not math.isfinite(remaining) or remaining<=0:
        raise ValueError('remaining included allowance is unknown/exhausted')
    extra=proof.get('remaining_paid_additional_usage_allowance_usd')
    if type(extra) not in (int,float) or extra!=0:
        raise ValueError('included-only use needs provider-enforced zero paid additional allowance')
    observed=datetime.fromisoformat(proof['observed_at'].replace('Z','+00:00'))
    if observed.tzinfo is None:raise ValueError('account read timestamp needs timezone')
    now=now or datetime.now(timezone.utc)
    if not 0<=(now-observed).total_seconds()<=3600:raise ValueError('fresh account allowance/cutoff evidence required')
    return stable_hash(proof)


def grok_command(binary, *, cwd, prompt_file, model, effort='high', schema=PROPOSAL_SCHEMA):
    return [str(binary),'--cwd',str(cwd),'-m',model,'--effort',effort,'--max-turns','1',
        '--no-plan','--no-subagents','--disable-web-search','--sandbox','read-only',
        '--tools','','--deny','Bash(*)','--deny','Write(*)','--deny','Edit(*)',
        '--deny','Read(*)','--deny','MCPTool(*)','--output-format','json',
        '--leader-socket',str(Path(prompt_file).parent/'isolated-leader.sock'),
        '--json-schema',json.dumps(schema,separators=(',',':')),'--prompt-file',str(prompt_file)]


def grok_environment():
    # Never inspect/copy credential contents or introduce API credentials.
    if any(name in os.environ for name in ('XAI_API_KEY','GROK_API_KEY')):
        raise ValueError('API-key environment present: Build-only route is ambiguous; no fallback permitted')
    env={name:os.environ[name] for name in ('HOME','USER','PATH','LANG','LC_ALL','TERM','TMPDIR','SSL_CERT_FILE','SSL_CERT_DIR') if name in os.environ}
    env.update(CUDA_VISIBLE_DEVICES='',GROK_DISABLE_AUTOUPDATER='1',GROK_MEMORY='0',GROK_SUBAGENTS='0',
        GROK_GOAL='0',GROK_TOOL_SEARCH='0',GROK_WRITE_FILE='0',GROK_WEB_FETCH='0',GROK_LSP_TOOLS='0')
    for family in ('CURSOR','CLAUDE'):
        for feature in ('SKILLS','RULES','AGENTS','MCPS','HOOKS'):env[f'GROK_{family}_{feature}_ENABLED']='0'
    return env


def verify_isolation(binary,cwd,output):
    """CLI discovery is a non-model read; retain counts, never server/auth contents."""
    result=subprocess.run([str(binary),'inspect','--json'],cwd=cwd,env=grok_environment(),
        capture_output=True,text=True,timeout=30,check=True)
    inspected=json.loads(result.stdout)
    hooks=[h for h in inspected.get('hooks',[]) if h.get('disabled') is not True and h.get('compatibilityStatus')!='disabled']
    mcps=[m for m in inspected.get('mcpServers',[]) if m.get('disabled') is not True]
    plugins=inspected.get('plugins',[])
    receipt=dict(grok_version=inspected.get('grokVersion'),enabled_hook_count=len(hooks),
        enabled_mcp_server_count=len(mcps),plugin_count=len(plugins),model_calls=0,
        observed_at=datetime.now(timezone.utc).isoformat(),credential_contents_read=False)
    atomic_json(Path(output)/'isolation-receipt.json',receipt)
    if hooks or mcps or plugins:raise ValueError('enabled hook/MCP/plugin found; no live model call')
    return stable_hash(receipt)


def parse_envelope(envelope):
    if not isinstance(envelope,dict) or envelope.get('is_error') is True or envelope.get('error'):
        raise ValueError('Grok CLI error or ambiguous envelope')
    # Installed live envelope is deliberately not guessed from arbitrary text.
    payload=envelope.get('structured_output')
    if payload is None:
        text=envelope.get('result')
        if not isinstance(text,str):raise ValueError('unsupported live Grok JSON envelope: retain and inspect before retry')
        payload=json.loads(text)
    if not isinstance(payload,dict) or set(payload)!={'candidate','study'}:
        raise ValueError('Grok proposal must contain exactly candidate and study objects')
    if not all(isinstance(payload[k],dict) for k in payload):raise ValueError('candidate/study must be objects')
    return payload


class GrokBuild:
    def __init__(self,cfg,quota_reader):self.cfg,self.quota_reader=cfg,quota_reader
    def propose(self,feedback,worktree,output):
        proof=self.quota_reader();proof_hash=quota_check(proof,model=self.cfg['model'])
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
        prompt=dict(instruction='Return one substantively new reusable candidate and READY study. Preserve seed0, all original tasks/evaluators/data/prior/init/gates and finite budgets. Use one global Recipe; no source edits or task-aware switches. Do not run tools, nested agents, training or network actions. Feed failures into a falsifiable training hypothesis. Do not rename/repeat a negative candidate.',
            schema=PROPOSAL_SCHEMA,study_template=self.cfg['study_template'],parent_candidate=self.cfg['parent_candidate'],
            previous_actual_failures=feedback,research_context=self.cfg.get('research_context',{}),
            allowed_recipe_fields=self.cfg.get('allowed_recipe_fields',[]),
            frozen_candidate_fields={k:v for k,v in self.cfg['parent_candidate'].items() if k not in {'id','hypothesis','changed_factors','mechanism_rationale','mechanism_class','recipe_overrides','prior_art'}})
        path=output/'prompt.json';atomic_json(path,prompt)
        isolation_hash=verify_isolation(self.cfg['grok_bin'],worktree,output)
        command=grok_command(self.cfg['grok_bin'],cwd=worktree,prompt_file=path,model=self.cfg['model'],effort=self.cfg.get('effort','high'))
        atomic_json(output/'command.json',dict(argv=command,account_proof_sha256=proof_hash,paid_additional_authorized_usd=0,API_fallback=False,isolation_receipt_sha256=isolation_hash))
        started=time.monotonic()
        with (output/'grok.json').open('wb') as handle:
            process=subprocess.Popen(command,cwd=worktree,stdout=handle,stderr=(output/'stderr.log').open('wb'),
                env=grok_environment(),start_new_session=True)
            try:
                while True:
                    if Path(self.cfg['stop_file']).exists() or time.monotonic()-started>=self.cfg['proposal_timeout_seconds']:
                        raise subprocess.TimeoutExpired(command,self.cfg['proposal_timeout_seconds'])
                    try:
                        process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                atomic_json(output/'terminal.json',dict(status='TIMEOUT',seconds=time.monotonic()-started,usage='UNKNOWN_PRESERVED',retry=False))
                raise RuntimeError('bounded Grok turn timed out; no automatic retry')
        atomic_json(output/'terminal.json',dict(status='COMPLETE' if process.returncode==0 else 'ERROR',exit_code=process.returncode,
            seconds=time.monotonic()-started,account_proof_sha256=proof_hash,output_sha256=file_hash(output/'grok.json'),stderr_sha256=file_hash(output/'stderr.log'),retry=False))
        if process.returncode:raise RuntimeError('Grok Build turn failed; retain original error; no automatic retry')
        envelope=read_json(output/'grok.json');proposal=parse_envelope(envelope)
        atomic_json(output/'provider-envelope-receipt.json',dict(session_id=envelope.get('session_id'),model=envelope.get('model'),usage=envelope.get('usage'),
            envelope_sha256=stable_hash(envelope),cost_source='Provider CLI receipt; not inferred from timeout/turn count'))
        return proposal


def freeze_suite(root,view_id='discriminator_stability'):
    from .views import load_view,load_tasks
    root=Path(root);view=load_view(root,view_id);tasks=load_tasks(root)
    selected={a['task']:deepcopy(tasks[a['task']]) for a in view['assignments']}
    return dict(view=deepcopy(view),tasks=selected,sha256=stable_hash(dict(view=view,tasks=selected)))


def required(suite,tier):return [a['task'] for a in suite['view']['assignments'] if a['importance']=='required' and a['qualification_tier']==tier]


def tier_allowance(suite,tier):
    groups={}
    for name in required(suite,tier):
        task=suite['tasks'][name];group=task['execution'].get('execution_group',name)
        groups[group]=max(groups.get(group,0),task['resources']['timeout_seconds'])
    return dict(queue=sum(groups.values()),supervisor=sum(groups.values())+20*len(groups),agent=0)


def all_pass(receipts,names,revision,source):
    return bool(names) and set(receipts)==set(names) and all(
        r.get('task_id')==name and r.get('gate_status')=='PASS' and r.get('independent_grade_verified') is True
        and r.get('candidate_revision')==revision and r.get('source_digest')==source
        and r.get('seed')==0 and r.get('complete_original_task') is True for name,r in receipts.items())


def semantic_candidate(candidate):
    from .api import resolve_public_recipe
    recipe=resolve_public_recipe(candidate)
    resolved=recipe.to_dict();resolved.pop('name',None)
    return stable_hash(dict(recipe=resolved,execution_path=candidate['execution_path'],extensions=candidate.get('extension_values',{})))


def validate_proposal(proposal,template):
    from .planning import validate_idea
    from .studies import validate_study
    candidate,study=deepcopy(proposal['candidate']),deepcopy(proposal['study'])
    validate_idea(candidate);validate_study(study)
    if candidate.get('decision_contract'):raise ValueError('new loop uses normal v3 study admission, not a legacy contract')
    if candidate.get('schema_version')!=3 or study['candidate']!=candidate['id'] or study['status']!='ready':
        raise ValueError('one reusable v3 candidate and matching READY study required')
    if not candidate.get('changed_factors') or not candidate.get('mechanism_rationale'):
        raise ValueError('substantive declared trainer delta and rationale required')
    for field in ('scope','campaign','control','terminal_rules','max_rounds','prior_evidence'):
        if study[field]!=template[field]:raise ValueError('proposal changed frozen study '+field)
    return candidate,study,semantic_candidate(candidate)


class CandidateBranch:
    def __init__(self,repo,base,directory):
        self.repo,self.base,self.directory=Path(repo),base,Path(directory)
    def create(self):
        if subprocess.check_output(['git','-C',str(self.repo),'rev-parse',self.base],text=True).strip()!=self.base:
            raise ValueError('candidate base must be exact full commit')
        branch='codex/grok-candidate-'+uuid.uuid4().hex[:12]
        subprocess.run(['git','-C',str(self.repo),'worktree','add','-b',branch,str(self.directory),self.base],check=True,capture_output=True)
        self.branch=branch;return self.directory
    def save(self,candidate,study):
        atomic_json(self.directory/'configs/forge/ideas'/f"{candidate['id']}.json",candidate)
        atomic_json(self.directory/'configs/forge/studies'/f"{study['id']}.json",study)
        subprocess.run(['git','-C',str(self.directory),'add',str(Path('configs/forge/ideas')/(candidate['id']+'.json')),str(Path('configs/forge/studies')/(study['id']+'.json'))],check=True)
        subprocess.run(['git','-C',str(self.directory),'commit','-m','Declare source-bound Grok candidate '+candidate['id']],check=True,capture_output=True)
        return subprocess.check_output(['git','-C',str(self.directory),'rev-parse','HEAD'],text=True).strip()


def write_board(directory,records,suite):
    path=Path(directory)/'reports/forge/grok-loop';path.mkdir(parents=True,exist_ok=True)
    atomic_json(path/'results.json',dict(schema_version=1,view=suite['view'],contract_sha256=suite['sha256'],rows=records,qualification_input=False))
    lines=['# Grok candidate loop','',f"Frozen view {suite['view']['id']} revision {suite['view']['revision']}. Every row is one candidate/source; no pooled passes.",'',
        '| Candidate / source | Tier1 | Tier2 | Full current gauntlet | Outcome |','| --- | --- | --- | --- | --- |']
    for row in records:
        counts=[]
        for tier in (1,2,3):
            receipts=row.get('tiers',{}).get(str(tier),{})
            counts.append(f"{sum(all_pass({name:receipt},[name],row['candidate_revision'],row['source_digest']) for name,receipt in receipts.items())}/{len(required(suite,tier))} required tasks; {len(receipts)} measured")
        full_pass=sum(all_pass({name:receipt},[name],row['candidate_revision'],row['source_digest']) for receipts in row.get('tiers',{}).values() for name,receipt in receipts.items())
        full_required=sum(len(required(suite,tier)) for tier in (1,2,3))
        lines.append(f"| {row['candidate_id']} / `{row['source_digest'][:12]}` | {counts[0]} | {counts[1]} | {full_pass}/{full_required} required tasks | {row['outcome']} |")
    (path/'README.md').write_text('\n'.join(lines)+'\n')
    subprocess.run(['git','-C',str(directory),'add','--sparse','-f','reports/forge/grok-loop/results.json','reports/forge/grok-loop/README.md'],check=True)
    subprocess.run(['git','-C',str(directory),'commit','-m','Retain exact candidate tier receipts and loop leaderboard'],check=True,capture_output=True)


class Loop:
    def __init__(self,cfg,suite,provider,executor,branches):
        self.cfg,self.suite,self.provider,self.executor,self.branches=cfg,suite,provider,executor,branches
    def run(self,*,test_only=True,resume=False):
        checkpoint=Path(self.cfg['runs_dir'])/'loop-checkpoint.json'
        if checkpoint.exists():
            if not resume:raise ValueError('existing loop evidence requires explicit continuation; never reset or retry an uncertain turn')
            saved=read_json(checkpoint)
            if saved.get('contract_sha256')!=self.suite['sha256'] or saved.get('phase')!='BETWEEN_CANDIDATES':
                raise ValueError('previous contract changed or proposal/dispatch needs explicit reconciliation')
            records=saved['records'];feedback=saved['feedback'];seen=set(saved['seen_semantics'])
            if any(row['outcome'] not in {'TIER1_FAILED_BLOCKED_OR_INCOMPLETE','TIER2_FAILED_BLOCKED_OR_INCOMPLETE','TIER3_FAILED_BLOCKED_OR_INCOMPLETE','FULL_CURRENT_GAUNTLET_PASS'} for row in records):
                raise ValueError('uncertain/partial previous candidate needs explicit reconciliation')
            if records and records[-1]['outcome']=='FULL_CURRENT_GAUNTLET_PASS':return records
        else:
            records=[];seen=set(self.cfg.get('negative_semantic_history',[]));feedback=[]
        began=time.monotonic();self.cfg['overall_deadline_unix']=saved['overall_deadline_unix'] if checkpoint.exists() else time.time()+self.cfg['maximum_loop_seconds']
        for index in range(len(records)+1,self.cfg['max_candidates']+1):
            if time.time()>=self.cfg['overall_deadline_unix'] or Path(self.cfg['stop_file']).exists():break
            if self.executor.active():raise ValueError('proposals only run after all owned training/graders are terminal')
            branch=self.branches(index);directory=branch.create()
            atomic_json(checkpoint,dict(phase='PROPOSAL_OR_DISPATCH_UNRECONCILED',records=records,feedback=feedback,seen_semantics=sorted(seen),contract_sha256=self.suite['sha256'],overall_deadline_unix=self.cfg['overall_deadline_unix'],next_index=index))
            proposal=self.provider.propose(deepcopy(feedback),directory,Path(self.cfg['runs_dir'])/f'proposal-{index}')
            candidate,study,semantic=validate_proposal(proposal,self.cfg['study_template'])
            if semantic in seen:raise ValueError('substantive duplicate/negative candidate: no rebranding retry')
            seen.add(semantic);code_head=branch.save(candidate,study)
            binding=self.executor.prepare(directory,candidate,study,self.suite)
            record=dict(candidate_id=candidate['id'],candidate_revision=binding['candidate_revision'],source_digest=binding['source_digest'],
                candidate_branch=branch.branch,code_head=code_head,semantic=semantic,tiers={},outcome='UNTESTED')
            records.append(record)
            for tier in (1,2,3):
                names=required(self.suite,tier)
                receipts=self.executor.run_tier(binding,tier,names)
                record['tiers'][str(tier)]=receipts
                write_board(directory,records,self.suite)
                if not all_pass(receipts,names,record['candidate_revision'],record['source_digest']):
                    record['outcome']=f'TIER{tier}_FAILED_BLOCKED_OR_INCOMPLETE';break
                record['outcome']='TIER1_PASS' if tier==1 else 'PROMISING_WHOLE_CANDIDATE_TIER1_AND_TIER2_PASS' if tier==2 else 'FULL_CURRENT_GAUNTLET_PASS'
                # The predefined interesting gate is the same whole candidate
                # PASSing every required Tier1 and Tier2 task; then Tier3 runs.
            write_board(directory,records,self.suite)
            feedback.append(deepcopy(record))
            atomic_json(Path(self.cfg['runs_dir'])/'loop-checkpoint.json',dict(records=records,feedback=feedback,seen_semantics=sorted(seen),test_only=test_only,phase='BETWEEN_CANDIDATES',overall_deadline_unix=self.cfg['overall_deadline_unix'],contract_sha256=self.suite['sha256']))
            if test_only or record['outcome']=='FULL_CURRENT_GAUNTLET_PASS':break
        return records


class ForgeExecutor:
    """Use the existing normal admission, queue, worker/fleet and measured ledger.

    Submit one frozen full candidate once; Forge's ordinary tier barriers prevent
    deeper execution after a required rejection. No stage-projector PASS enters
    these decisions. All worker monitoring completes before the next proposal.
    """
    def __init__(self,cfg,framework_root):
        self.cfg,self.root=cfg,Path(framework_root);self._active=False;self._results={}
    def active(self):return self._active
    def prepare(self,directory,candidate,study,suite):
        from .planning import resolve_idea
        from .sources import verify_snapshot
        directory=Path(directory)
        if freeze_suite(directory,suite['view']['id'])['sha256']!=suite['sha256']:
            raise ValueError('candidate changed frozen task/view/evaluator declarations')
        # Protect actual grader import closure, not merely declaration JSON.
        for name,task in suite['tasks'].items():
            for relative,expected in task['evaluation'].get('sources',{}).items():
                if file_hash(directory/relative)!=expected:
                    raise ValueError('protected evaluator/host implementation changed: '+relative)
        request=resolve_idea(directory,candidate['id'],queue_root=Path(self.cfg['queue_root']),freeze_source=True,study=study['id'])
        if request.get('preflight_blockers'):
            raise ValueError('normal admission preflight BLOCKED: '+'; '.join(request['preflight_blockers']))
        if request['protocol'].get('seed')!=0 or request['study_review']['status']!='READY':
            raise ValueError('normal seed0 reviewed study admission required')
        if request['view']!=suite['view']:
            raise ValueError('resolved current tier view differs from explicitly pinned loop suite')
        verify_snapshot(Path(request['source']['snapshot_path']),request['source'])
        self._results[request['candidate_revision']]=None
        return dict(candidate_revision=request['candidate_revision'],source_digest=request['source']['digest'],request=request,directory=str(directory),study=study)
    def _fund_ready_candidate(self,request,task_ids):
        from .codex_runner import Spending
        groups={}
        for name in task_ids:
            task=request['tasks'][name];group=task['execution'].get('execution_group',name)
            groups[group]=max(groups.get(group,0),task['resources']['timeout_seconds'])
        needed=dict(queue=sum(groups.values()),supervisor=sum(groups.values())+20*len(groups),agent=0)
        spending=Spending(self.cfg['budget_state'])
        with spending.state() as state:
            remaining=spending.remaining(state)
            delta={k:max(0,needed[k]-remaining[k]) for k in needed}
            if state['reservations']:raise ValueError('cannot increase allowance against an unreconciled active reservation')
            if any(delta.values()):
                before=deepcopy(state['limits'])
                for key in delta:state['limits'][key]+=delta[key]
                spending.event(state,'bounded_local_compute_allowance_increase',candidate_revision=request['candidate_revision'],
                    source_digest=request['source']['digest'],declared_full_horizon=needed,increase=delta,previous_limits=before,
                    authorization='Mikkel prior general local compute increase approval; parent clarification 2026-10-14. One ready candidate only, no reset or unlimited funding.',
                    paid_model_allowance_changed=False)

    def run_tier(self,binding,tier,names):
        from .codex_runner import Spending,contract,plan,execute,validate_request
        if self._results[binding['candidate_revision']] is None:
            request=binding['request'];directory=Path(binding['directory']);runs=Path(self.cfg['runs_dir'])/'science'
            path=Path(self.cfg['runs_dir'])/(request['candidate_revision']+'-request.json');atomic_json(path,request)
            study=dict(id=binding['study']['id'],hypothesis=binding['study']['hypothesis'],falsifier=json.dumps(binding['study']['falsifier']),
                cohort='screening-seed0-whole-candidate',task_ids=[a['task'] for a in request['view']['assignments']
                    if a['qualification_tier']<=request['through_tier']],request_path=str(path),
                semantic_factors=dict(recipe_semantics=semantic_candidate(request['candidate']),source_science=self.cfg['scientific_source_semantics']))
            cfg=dict(schema_version=1,source_commit=self.cfg['source_commit'],expected_view_revision=self.cfg['expected_view_revision'],slots=self.cfg.get('slots',6),max_studies=1,studies=[study],
                runs_dir=str(runs),queue_root=self.cfg['queue_root'],budget_state=self.cfg['budget_state'],lease_root=self.cfg['lease_root'],
                stop_files=[self.cfg['stop_file']],dispatch_window_seconds=self.cfg.get('dispatch_window_seconds',300),
                blocked_UTC_windows=self.cfg.get('blocked_UTC_windows',[]),overall_deadline_unix=self.cfg.get('overall_deadline_unix'))
            if self.cfg.get('gpu_allocation') is not None:cfg['gpu_allocation']=self.cfg['gpu_allocation']
            config=Path(self.cfg['runs_dir'])/(request['candidate_revision']+'-dispatch.json');atomic_json(config,cfg)
            validate_request(request,study,contract(directory))
            for name in ('queue.py','worker.py'):
                if request['source']['files'].get('experiments/forge/'+name)!=file_hash(self.root/'experiments/forge'/name):
                    raise ValueError('frozen source lacks current shared lease propagation')
            _,report=plan(config,root=directory)
            non_budget=[b for b in report['blockers'] if not b.startswith('aggregate ') or 'reservation exceeds remaining balance' not in b]
            if non_budget:raise ValueError('dispatch preflight BLOCKED before funding: '+'; '.join(non_budget))
            if self.cfg.get('allow_bounded_local_compute_increase') is True:
                self._fund_ready_candidate(request,study['task_ids'])
                _,report=plan(config,root=directory)
            if report['blockers']:
                return {name:dict(task_id=name,gate_status='BLOCKED',reason='; '.join(report['blockers'])) for name in names}
            self._active=True
            try:dispatch=Path(execute(cfg,report,directory))
            finally:self._active=False
            from .queue import Queue
            queue=Queue(Path(self.cfg['queue_root']),report_root=directory/'reports/forge');state=queue.inspect()
            receipts=certified_receipts(binding,state,dispatch)
            self._results[binding['candidate_revision']]=receipts
        all_rows=self._results[binding['candidate_revision']]
        return {name:all_rows.get(name,dict(task_id=name,gate_status='BLOCKED',reason='Original required task has no completed certified receipt')) for name in names}


def certified_receipts(binding,state,dispatch):
    """Bind queue rows to the exact admitted jobs and their independent certificates."""
    request=binding['request'];expected={j['compatibility_key']:j for j in request['jobs']};receipts={}
    for key,job in state['jobs'].items():
        result=job.get('result')
        if key not in expected or not result or result.get('candidate_revision')!=binding['candidate_revision']:continue
        raw=result.get('raw',{});grading=raw.get('grading',{})
        certified=(job.get('definition')==expected[key] and raw.get('attempt_status')=='completed'
            and grading.get('source_digest')==binding['source_digest']
            and grading.get('raw_hash')==stable_hash(raw.get('result',{})))
        for row in result.get('task_results',[]):
            name=row['task_id'];grade=grading.get('grades',{}).get(name,{})
            verified=certified and row.get('gate_status')==grade.get('gate_status',grade.get('status'))
            receipts[name]=dict(task_id=name,gate_status=row['gate_status'] if verified else 'INVALID',
                independent_grade_verified=verified,complete_original_task=verified and row['gate_status'] in {'PASS','FAIL'},
                candidate_revision=binding['candidate_revision'],source_digest=binding['source_digest'],seed=request['protocol']['seed'],
                attempt_id=result['attempt_id'],metrics=row.get('metrics'),evaluator_result=row.get('evaluator_result'),
                cost=row.get('cost'),dispatch_path=str(dispatch),original_task_result_sha256=stable_hash(row),
                independent_certificate_sha256=stable_hash(grading),reason=row.get('reason',''))
    return receipts


def configuration_check(cfg,root):
    if cfg.get('schema_version')!=1:raise ValueError('unknown Grok loop config')
    for field in ('max_candidates','proposal_timeout_seconds','maximum_loop_seconds'):
        value=cfg.get(field)
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('finite positive '+field+' required')
    if type(cfg['max_candidates']) is not int or cfg['max_candidates']>100:raise ValueError('finite candidate cap in 1..100 required')
    source=cfg['source_commit']
    view=json.loads(subprocess.check_output(['git','-C',str(root),'show',source+':configs/forge/views/'+cfg['view_id']+'.json'],text=True))
    if view['revision']!=cfg['expected_view_revision']:raise ValueError('explicit tier/source revision differs from selected base')
    if view['revision']==10 and cfg.get('acknowledge_unmerged_PR403') is not True:
        raise ValueError('revision10 contains unmerged PR403; explicit acknowledgement required, never silently substituted')
    if cfg['study_template']['scope']['through_tier']!=3:
        raise ValueError('full loop needs all current tiers declared before the one frozen candidate is admitted')
    return view


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('command',choices=['loop'])
    parser.add_argument('config',type=Path);parser.add_argument('--execute-test',action='store_true')
    parser.add_argument('--continue-included',action='store_true');args=parser.parse_args(argv)
    cfg=read_json(args.config);view=configuration_check(cfg,args.root)
    proof=read_json(cfg['account_receipt']) if Path(cfg.get('account_receipt','missing')).is_file() else {}
    blockers=[]
    try:quota_check(proof,model=cfg['model'])
    except (ValueError,KeyError) as exc:blockers.append(str(exc))
    print(json.dumps(dict(status='BLOCKED' if blockers else 'PLANNED_NOT_STARTED',provider='installed_grok_build',view_revision=view['revision'],
        proposed_full_gate='same candidate/source independent PASS of every required Tier1 and Tier2 task triggers every remaining current Tier3 task',
        blockers=blockers,paid_calls_started=0,training_starts=0),indent=2))
    if not args.execute_test and not args.continue_included:return 0
    if blockers:parser.error('; '.join(blockers))
    if args.continue_included:
        receipt=read_json(cfg['first_live_test_receipt'])
        if receipt.get('complete_test') is not True or receipt.get('outcome_reported_to_user') is not True or receipt.get('credit_bucket')!='included_weekly':
            parser.error('one completed and reported live Build test of the intended credit bucket is required before unattended use')
    # A worktree is created from the exact explicitly selected source. This
    # source must include the queue/worker FD forwarding used by the framework.
    bootstrap=CandidateBranch(args.root,cfg['source_commit'],Path(cfg['runs_dir'])/'suite-source')
    root=bootstrap.directory if args.continue_included and bootstrap.directory.exists() else bootstrap.create();suite=freeze_suite(root,cfg['view_id'])
    provider=GrokBuild(cfg,lambda:read_json(cfg['account_receipt']))
    executor=ForgeExecutor(cfg,args.root)
    branches=lambda i:CandidateBranch(args.root,cfg['source_commit'],Path(cfg['runs_dir'])/f'candidate-{i}')
    print(json.dumps(Loop(cfg,suite,provider,executor,branches).run(test_only=not args.continue_included,resume=args.continue_included),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
