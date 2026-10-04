"""Private request/coordinator controls. No learner, scorer or shared Queue runs."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'reports/forge/pr223-original-full-retest-20261004'


@pytest.fixture(scope='module')
def helper():
    spec=importlib.util.spec_from_file_location('_private_lifecycle_helper',OLD/'run_retest.py')
    value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
    return value


@pytest.fixture
def packet(helper,monkeypatch,tmp_path):
    scope=helper.native3_module()
    card=helper.protocol.load(ROOT)
    external={'/synthetic/source-only-host.py':'b'*64}
    definitions={}
    for declared in card['rows']:
        parent=deepcopy(declared['original_definition']);parent['external_inputs_sha256']=scope.digest(external)
        definition=deepcopy(parent);definition['id']=declared['id']
        definition['fresh_retest']=dict(parent_original_id=parent['id'],
            historical_case_sha256=declared['historical_case_sha256'],source_derived_parent_sha256=scope.digest(parent),
            goal_media_steps=declared['media_steps'],goal_observer_sha256=helper.sha(OLD/'goal_observer.py'),
            scope='fresh full original19 noisy selected law; no current26/default/speed credit')
        definitions[definition['id']]=definition
    base=dict(protocol=card,case_definitions=definitions,external_inputs=dict(files=external),
        recipe_overrides=dict(original_config_sha256=helper.legacy.CONFIG_SHA,original_options=deepcopy(helper.legacy.OPTIONS)),
        preflight=dict(status='PASS_METADATA_ONLY',cases=19,updates=48800,models=0,sampler_calls=0,scorer_calls=0))
    anchor=tmp_path/'private-closed-parent.json'
    anchor.write_bytes((HERE/'fixtures/closed-parent-metadata.json').read_bytes())
    with monkeypatch.context() as patcher:
        patcher.setattr(helper,'plan',lambda root:deepcopy(base))
        patcher.setattr(helper,'source_requirements',lambda root:dict(origin_commit='b'*40,files={}))
        value=scope.plan(ROOT,helper,dict(path=str(anchor),sha256=scope.ANCHOR_SHA,bytes=scope.ANCHOR_BYTES))

    # A PRIVATE metadata fixture, not an execution-source preparation. Scientific
    # external owners are inert source strings and are never imported.
    root=tmp_path/'private-source'
    names=[p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/forge').rglob('*.py')]
    names += [helper.SELF,helper.DIRECTORY+'/protocol.py',helper.protocol.LEGACY,
        helper.DIRECTORY+'/budget_ledger.py',helper.DIRECTORY+'/goal_observer.py',helper.protocol.PROTOCOL,
        helper.protocol.CONFIG,scope.DIRECTORY+'/native3_contract.py',scope.DIRECTORY+'/scorer_boundary_control.py',
        scope.DIRECTORY+'/request_boundary_control.py',*scope.PORTABLE_CONTROL_FILES,*scope.HISTORY_PINS]
    for name in set(names):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    target=root/scope.ANCHOR_RELATIVE;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(anchor.read_bytes())
    native=root/'atlas19-external/native_root'
    for name in ('lib/toy_models.py','particlegan/__init__.py','benchmarks/__init__.py',
                 'benchmarks/toy100/__init__.py','benchmarks/toy100/train.py',
                 'benchmarks/toy100/gate.py','benchmarks/toy100/accuracy_gate.py'):
        target=native/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('# SYNTHETIC: never imported\n')
    files={p.relative_to(root).as_posix():helper.sha(p) for p in root.rglob('*') if p.is_file()}
    source=dict(schema_version=1,origin_commit='b'*40,files=files,digest=helper.stable_hash(files))
    helper.atomic_json(root/'forge-source.json',source)
    output=tmp_path/'out'
    value.update(execution_source={**source,'snapshot_path':str(root)},
        source=dict(commit='b'*40,files_sha256=files,execution_digest=source['digest']),
        snapshot_locations=dict(package=str(root),native_root=str(native)),
        prepared_output=str(output),queue_root=str(tmp_path/'private-queue'),metadata_ledger_path=scope.CANONICAL_LEDGER,
        copied_preflight_receipt_path=str(scope.copied_preflight_path(output)))
    assert helper.validate_packet(value,source=True) is value
    return value


def control(helper):
    return helper.module('reports/forge/pr223-native3-continuation-20261004/request_boundary_control.py',
                         '_private_request_control')


def current_request(helper,packet,index=0):
    """Negative-only transport fixture; actual positive uses run/coordinator."""
    value=deepcopy(packet)
    value.update(lane_runtime={'device':'cuda:0','torch_threads':1},
        coordinator=dict(canonical_output=value['prepared_output'],queue_root=value['queue_root'],study_key='synthetic'))
    for old in value['rows'][:index]:
        old.update(status='PASS',full_protocol_complete=True,media={'synthetic':True},
            **helper.ledger_module().case_cost(old['allowance_seconds'],
                {'attempt_status':'completed','paid_wall_seconds':1.},certified=True))
    row=value['rows'][index];row.update(status='RUNNING',attempt_token='a'*32)
    row['attempt_key']=helper.PolicyCoordinator.attempt_key(None,value,
        {'family':'atlas','recipe_overrides':value['recipe_overrides']},row)
    class Ledger:
        def snapshot(self):return deepcopy(value['metadata_history']['predecessor_closed_state'])
    helper._save(Path(value['prepared_output'])/'negative-only-study.json',value,Ledger())
    target=Path(value['prepared_output'])/'native'/row['task']
    worker=dict(token=row['attempt_token'],lease_fd=3,lease_fds=[4,3],lease_path='/synthetic/execution.lock',
        started_monotonic=1.,deadline_monotonic=1.+row['allowance_seconds'],physical_gpu=1,device='cuda:0')
    command=[sys.executable,'-u','-B',str(Path(value['execution_source']['snapshot_path'])/helper.SELF),
             '--child',str(target/'request.json'),'--lease-fd','3']
    return helper.make_request(value,row,target,command,worker)


def test_actual_parent_coordinator_json_child_and_open_lease_boundary(helper,packet,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    before=set(sys.modules)
    proof=control(helper).run(packet,helper)
    assert proof['status']=='PASS_SYNTHETIC_MAINTAINED_REQUEST_BOUNDARY'
    assert proof['simulated_coordinator_requests']==proof['simulated_registrations']==proof['simulated_admissions']==1
    assert all(proof[k]==0 for k in ('actual_queue_calls','actual_admissions','models','sampler_calls','scorer_calls','observer_calls'))
    assert proof['outcomes']['maintained_current_request']=='PASS_STOPPED_BEFORE_OBSERVER'
    assert proof['outcomes']['foreign_lease']=='REFUSED_FOREIGN_LEASE'
    assert proof['outcomes']['expired_lease']=='REFUSED_EXPIRED_LEASE'
    assert not any(n.split('.')[0] in {'torch','numpy','particlegan','lib','benchmarks'} for n in set(sys.modules)-before)


@pytest.mark.parametrize('index',[0,1,2])
def test_saved_current_status_all_three_positions_and_lossless_wire(helper,packet,index):
    # This supplements the real first-parent construction. Later prefix values
    # are explicitly synthetic and cannot serve as attested outcomes.
    request=current_request(helper,packet,index)
    request=json.loads(json.dumps(request,sort_keys=True,allow_nan=False))
    assert helper.validate_request(request)[1]==request['row']
    assert request['packet']['status']==('READY' if index<2 else 'INCOMPLETE')
    assert request['packet']['protocol']==packet['protocol']
    assert request['packet']['case_definitions']==packet['case_definitions']


@pytest.mark.parametrize('change',['declared','not_run','historical','foreign_token','attempt_key','deadline',
    'fds','physical_gpu','physical_gpu_bool','canonical','tail_credit','prior_incomplete','host_steps','gate','seed','clocks',
    'full_recipe','source','command','retry','halted','unknown_field'])
def test_current_request_refuses_coherent_prior_credit_science_source_and_ownership(helper,packet,change):
    index=1 if change=='prior_incomplete' else 0
    value=current_request(helper,packet,index);scope=helper.native3_module();row=value['row'];p=value['packet']
    if change=='declared':p['status']='DECLARED'
    elif change=='not_run':row['status']='NOT_RUN'
    elif change=='historical':row['original_gate']='PASS'
    elif change=='foreign_token':value['worker']['token']='b'*32
    elif change=='attempt_key':row['attempt_key']='0'*64
    elif change=='deadline':value['worker']['deadline_monotonic']-=1
    elif change=='fds':value['worker']['lease_fds']=[3,3]
    elif change=='physical_gpu':value['worker']['physical_gpu']=0
    elif change=='physical_gpu_bool':value['worker']['physical_gpu']=True
    elif change=='canonical':p['coordinator']['canonical_output']+='/foreign'
    elif change=='tail_credit':p['rows'][2]['original_gate']='PASS'
    elif change=='prior_incomplete':p['rows'][0]['full_protocol_complete']=False
    elif change in {'host_steps','gate','seed','clocks','full_recipe'}:
        d=p['case_definitions'][row['id']]
        if change=='host_steps':d['original_host']['steps']=600
        elif change=='gate':d['original_requirements'][0][2]=0
        elif change=='seed':d['original_host']['seed']+=1
        elif change=='clocks':d['observation_steps'].pop()
        else:p['protocol']['rows'][0]['resolved_recipe']['lr']=.0053125
        row['case_sha256']=scope.digest(d)
    elif change=='source':p['source']['commit']='0'*40
    elif change=='command':value['command'][0]='/foreign/python'
    elif change=='retry':row['status']='FAIL'
    elif change=='halted':p['budget_accounting']['halt_required']=True
    else:row['unknown_scientific_field']=True
    p['rows'][index]=deepcopy(row)
    with pytest.raises(ValueError):helper.validate_request(value)


@pytest.mark.parametrize('status,paid',[('completed',1.),('completed',1471.),('error',1.),
    ('cancelled',1.),('timeout',1471.),('missing',0.),('error',1471.)])
def test_actual_maintained_recovery_equals_conservative_case_cost(helper,tmp_path,status,paid):
    policy=sys.modules[helper.PolicyCoordinator.__module__];budget=helper.ledger_module()
    lease=tmp_path/'private-attempt/execution.lock';lease.parent.mkdir()
    terminal=None if status=='missing' else dict(token='synthetic',attempt_status=status,paid_wall_seconds=paid,child_returncode=1)
    if terminal is not None:helper.atomic_json(lease.parent/'supervisor-terminal.json',terminal)
    entry=dict(token='synthetic',lease_path=str(lease),allowance_seconds=1470.,status='running')
    policy._released_attempt(entry)
    cost=budget.case_cost(1470.,terminal,certified=False)
    assert entry['charged_seconds']==cost['charged_seconds']==paid+cost['reserved_seconds']
    assert cost['paid_wall_seconds']==paid
    assert cost['reserved_seconds']==(0. if status=='completed' else max(0.,1470.-paid))
    assert cost['overrun_seconds']==max(0.,paid-1470.)
    assert entry['status']==('awaiting_certification' if status=='completed' else 'interrupted')


def test_current_attempt_cannot_spend_a_partial_or_reset_aggregate(helper,packet):
    scope=helper.native3_module();budget=helper.ledger_module();state=deepcopy(packet['metadata_history']['predecessor_closed_state'])
    with pytest.raises(ValueError):scope.require_next_reservation(packet,state,1469,helper)
    state['phases']=[];budget._update_totals(state)
    with pytest.raises(ValueError,match='lost or reset'):scope.require_next_reservation(packet,state,1470,helper)
    value=deepcopy(packet);row=value['rows'][0]
    row.update(status='INCOMPLETE',**budget.case_cost(row['allowance_seconds'],{'attempt_status':'error','paid_wall_seconds':1471.},certified=False))
    state=deepcopy(packet['metadata_history']['predecessor_closed_state'])
    assert scope.require_next_reservation(value,state,0,helper)['halt_required'] is True
    # The source loader reloads classes; use their stable base and message.
    with pytest.raises(RuntimeError,match='whole native allowance'):scope.require_next_reservation(value,state,1440,helper)


def test_source_proof_exact_binding_and_no_current_numeric_credit(helper,packet):
    current=control(helper)
    proof=dict(schema=current.SCHEMA,status='PASS_SYNTHETIC_MAINTAINED_REQUEST_BOUNDARY',
        binding=current.binding(packet,helper),outcomes=deepcopy(current.OUTCOMES),synthetic_only=True,
        actual_queue_calls=0,actual_admissions=0,models=0,sampler_calls=0,scorer_calls=0,observer_calls=0,
        simulated_coordinator_requests=1,simulated_registrations=1,simulated_admissions=1,
        numerical_credit=False,old_arrays_read=False)
    assert current.validate_proof(proof,packet,helper) is proof
    for key in ('models','actual_admissions','numerical_credit'):
        bad=deepcopy(proof);bad[key]=True
        with pytest.raises(ValueError):current.validate_proof(bad,packet,helper)


@pytest.mark.parametrize('change',['old_schema','missing_debit','reduced_debit','duplicate_debit','foreign_source',
    'old_grade_credit','missing_prefix','changed_prefix','twelve_only','reset_metadata','foreign_ledger',
    'source_pin_missing','source_bytes_changed','predecessor_origin','predecessor_digest'])
def test_additive_pretraining_debit_is_mandatory_exact_and_cost_only(helper,packet,change):
    value=deepcopy(packet);scope=helper.native3_module();state=value['metadata_history']['predecessor_closed_state']
    if change=='old_schema':value['schema']='pg_pr223_native3_continuation_v1'
    elif change=='missing_debit':value.pop('predecessor_cost')
    elif change=='reduced_debit':value['predecessor_cost']['pretraining_invalid_case_charged_seconds']=0.
    elif change=='duplicate_debit':value['spec']['prior_case_charged_seconds']+=scope.PRETRAINING_INVALID_SECONDS
    elif change=='foreign_source':value['predecessor_cost']['source']['digest']='0'*64
    elif change=='old_grade_credit':value['predecessor_cost']['numerical_credit']=True
    elif change=='missing_prefix':value['metadata_history'].pop('predecessor_closed_state')
    elif change=='changed_prefix':state['phases'][21]['name']='coherent foreign phase'
    elif change=='twelve_only':value['metadata_history']['predecessor_closed_state']=deepcopy(value['metadata_history']['closed_state'])
    elif change=='reset_metadata':value['metadata_ledger_path']='/synthetic/new180.json'
    elif change=='foreign_ledger':value['predecessor_cost']['canonical_metadata_ledger']='/synthetic/new180.json'
    elif change=='source_pin_missing':value['execution_source']['files'].pop(scope.PREDECESSOR_ANCHOR)
    elif change=='predecessor_origin':
        value['execution_source']['origin_commit']=value['source']['commit']=value['predecessor_cost']['source']['origin_commit']
        helper.atomic_json(Path(value['execution_source']['snapshot_path'])/'forge-source.json',
                           {k:v for k,v in value['execution_source'].items() if k!='snapshot_path'})
    elif change=='predecessor_digest':value['execution_source']['digest']=value['predecessor_cost']['source']['digest']
    else:(Path(value['execution_source']['snapshot_path'])/scope.PREDECESSOR_ANCHOR).write_text('changed\n')
    with pytest.raises((ValueError,KeyError)):helper.validate_packet(value,source=True)


def test_combined_cost_adds_startup_once_and_metadata_once_preserving_closed22(helper,packet):
    scope=helper.native3_module();budget=helper.ledger_module()
    state=deepcopy(packet['metadata_history']['predecessor_closed_state'])
    assert len(state['phases'])==22 and state['phases'][:12]==packet['metadata_history']['closed_state']['phases']
    state['phases'].append(dict(index=22,name='synthetic_additive_phase',status='COMPLETE',paid_wall_seconds=.1,paused_wall_seconds=0.))
    budget._update_totals(state)
    row=packet['rows'][0]
    row.update(status='INVALID',**budget.case_cost(row['allowance_seconds'],
               dict(attempt_status='completed',paid_wall_seconds=3.),certified=False))
    cost=scope.require_next_reservation(packet,state,1440,helper)
    assert cost['prior_case_charged_seconds']==3165.841891122982
    assert cost['pretraining_invalid_case_charged_seconds']==2.0496059330180287
    assert cost['current_case_paid_wall_seconds']==3. and cost['current_case_reserved_seconds']==0.
    assert cost['metadata_charged_seconds']==state['charged_seconds']
    assert cost['charged_seconds']==sum((3165.841891122982,2.0496059330180287,3.,state['charged_seconds']))
    assert cost['charged_seconds']!=cost['combined_prior_case_charged_seconds']+3.+state['charged_seconds']+scope.PRIOR_METADATA_SECONDS
    # Numerical INVALID halting remains the maintained executor's row guard;
    # this function projects exact costs without manufacturing a numeric grade.
    assert row['certified'] is False and 'original_gate' not in row


@pytest.mark.parametrize('target',['/ml2/hypergan/.pg-pr223-full-original-retest-20261004.pr223-full19-metadata-cost.json',
    '/ml2/hypergan/forge-pr223-native3/current/study.json','/ml2/hypergan/runs/forge/queue.json',
    '/home/martyn/dev/ParticleGAN/artifacts/old-model.pt'])
def test_audit_refuses_lexical_and_virtual_link_targets_before_original_metadata(target):
    auditor=sys.modules.get('__main__')
    if not hasattr(auditor,'canonical_allowed'):
        # Ordinary developer invocation can use the same pure audit source.
        spec=importlib.util.spec_from_file_location('_private_audit_policy',
            Path(os.environ['PARTICLEGAN_NATIVE3_SOFTWARE_OUTPUT'])/'run_controls.py')
        tree=__import__('ast').parse(Path(spec.origin).read_text())
        functions=[n for n in tree.body if isinstance(n,__import__('ast').FunctionDef) and n.name in {'permitted','canonical_allowed'}]
        namespace=dict(Path=Path,os=os,stat=__import__('stat'),allowed=(ROOT,Path(spec.origin).parent,Path(sys.prefix)))
        exec(compile(__import__('ast').Module(body=functions,type_ignores=[]),spec.origin,'exec'),namespace)
        auditor=type('Audit',(),namespace)
    calls=[]
    def no_original(path):calls.append(str(path));raise AssertionError('original metadata must not be probed')
    with pytest.raises(AssertionError):auditor.canonical_allowed(target,lstat=no_original,readlink=no_original)
    assert calls==[]
    link=Path('/tmp/private-link')
    def virtual_stat(path):
        calls.append(str(path))
        if Path(path)==link:return type('Stat',(),dict(st_mode=0o120777))()
        if Path(path).is_relative_to('/tmp'):return type('Stat',(),dict(st_mode=0o040777))()
        return type('Stat',(),dict(st_mode=0o040755))()
    with pytest.raises(AssertionError):auditor.canonical_allowed(link,lstat=virtual_stat,readlink=lambda path:target)
    assert not any(Path(p)==Path(target) or Path(p).is_relative_to(Path(target).parent) for p in calls)
