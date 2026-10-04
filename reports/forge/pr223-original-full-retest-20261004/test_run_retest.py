"""Source/metadata/lease controls only; no historical learner or queue is run."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


@pytest.fixture(scope='module')
def helper():
    spec=importlib.util.spec_from_file_location('_pr223_source_tests',HERE/'run_retest.py')
    result=importlib.util.module_from_spec(spec);sys.modules[spec.name]=result;spec.loader.exec_module(result)
    return result


@pytest.fixture(scope='module')
def planned(helper):
    # Source + compact JSON reads only. No snapshots/admission or paid output.
    return helper.plan(ROOT)


def test_plan_binds_full_original19_current_source_and_new_observer(helper,planned):
    assert planned['required']==19 and helper.validate_packet(planned) is planned
    assert planned['spec']['export_grace_seconds']==0 and planned['spec']['retries']==0
    assert sum(r['timeout_seconds'] for r in planned['rows'])==9810
    assert all(r['status']=='NOT_RUN' for r in planned['rows'])
    assert planned['preflight']['source_parity']['successful_commit']==helper.protocol.SUCCESSFUL_COMMIT
    assert len(planned['preflight']['location_only_external_fingerprints'])==19
    assert planned['source']['files_sha256'][helper.protocol.PROTOCOL]==helper.sha(ROOT/helper.protocol.PROTOCOL)
    assert planned['source']['files_sha256'][helper.SELF]==helper.sha(HERE/'run_retest.py')
    assert 'reports/toy_audit/catalog.json' in planned['source']['files_sha256']
    assert any(p.startswith('examples/') for p in planned['source']['files_sha256'])


@pytest.mark.parametrize('change',['missing','duplicate','budget','grace','gate','host','binding','borrow'])
def test_forged_or_partial_packet_rejected(helper,planned,change):
    packet=deepcopy(planned)
    if change=='missing': packet['rows'].pop()
    elif change=='duplicate': packet['rows'][-1]=deepcopy(packet['rows'][0])
    elif change=='budget': packet['rows'][0]['allowance_seconds']+=1
    elif change=='grace': packet['spec']['export_grace_seconds']=1
    elif change=='gate': packet['case_definitions'][packet['rows'][0]['id']]['original_requirements'][0][2]=0
    elif change=='host': packet['case_definitions'][packet['rows'][0]['id']]['original_host']['num_particles']=12
    elif change=='binding': packet['rows'][0]['case_sha256']='0'*64
    else: packet['recipe_overrides']={'lr':.0053125,'prior_lr_mult':1.5}
    with pytest.raises(ValueError): helper.validate_packet(packet)


def test_actual_standalone_metadata_preflight_imports_no_torch_or_model(helper):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B',str(HERE/'run_retest.py'),'--metadata-preflight'],
                          env=env,cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    record=json.loads(result.stdout)
    assert record['status']=='PASS_METADATA_ONLY'
    assert all(record[k]==0 for k in ('models','sampler_calls','scorer_calls','preparation_calls','queue_calls'))
    assert record['cases']==19 and record['updates']==48800


def source_fixture(tmp_path):
    root=tmp_path/'snapshot';(root/'particlegan').mkdir(parents=True)
    path=root/'particlegan/policy.py';path.write_text('synthetic source only\n')
    return root,path,{'snapshot_path':str(root),'files':{'particlegan/policy.py':hashlib.sha256(path.read_bytes()).hexdigest()}}


def test_actual_import_source_and_namespace_are_pinned(helper,tmp_path):
    root,path,source=source_fixture(tmp_path)
    helper.guard_imports(source,modules={'particlegan.policy':SimpleNamespace(__file__=str(path))})
    helper.guard_imports(source,modules={'particlegan':SimpleNamespace(__file__=None,__path__=[str(root/'particlegan')])})
    helper.guard_imports(source,modules={'torch.ops':SimpleNamespace(__file__='torch.ops')})
    helper.guard_imports(source,modules={'_frozen_ra11_initializer':SimpleNamespace(__file__=None,__path__=[str(root/'particlegan')])})


@pytest.mark.parametrize('change',['foreign','stale','empty_namespace','multiple_namespace','foreign_alias','pseudo_protected'])
def test_import_guard_refuses_foreign_stale_and_ambiguous_sources(helper,tmp_path,change):
    root,path,source=source_fixture(tmp_path)
    item=SimpleNamespace(__file__=str(path));name='particlegan.policy'
    if change=='foreign': item.__file__=str(tmp_path/'foreign.py')
    elif change=='stale': path.write_text('changed\n')
    elif change=='empty_namespace': item=SimpleNamespace(__file__=None,__path__=[])
    elif change=='multiple_namespace': item=SimpleNamespace(__file__=None,__path__=[str(root/'particlegan'),str(tmp_path)])
    elif change=='foreign_alias': name='current_api_fixtures';item.__file__=str(tmp_path/'foreign.py')
    else: item.__file__='particlegan.policy'
    with pytest.raises((ValueError,OSError)): helper.guard_imports(source,modules={name:item})


def test_derived_wrapper_needs_its_exact_explicit_pin(helper,tmp_path):
    root,path,source=source_fixture(tmp_path)
    derived=tmp_path/'wrapper.py';derived.write_text('synthetic observer wrapper\n')
    pin=helper.sha(derived)
    helper.guard_imports(source,modules={'_pr223_wrapper':SimpleNamespace(__file__=str(derived))},derived={derived.resolve():pin})
    derived.write_text('tampered\n')
    with pytest.raises(ValueError): helper.guard_imports(source,modules={},derived={derived.resolve():pin})


@pytest.fixture
def lease_request(helper,tmp_path):
    queue=tmp_path/'synthetic-queue';study=queue/'policy/studies'/'synthetic.lock'
    lease=queue/'policy/attempts'/'synthetic'/'execution.lock'
    study.parent.mkdir(parents=True);lease.parent.mkdir(parents=True)
    with study.open('w+') as s,lease.open('w+') as a:
        row={'allowance_seconds':150}
        source={'synthetic_only':True}
        command=['python','synthetic_only']
        worker={'lease_path':str(lease),'lease_fd':a.fileno(),'lease_fds':[s.fileno(),a.fileno()],
                'token':'synthetic','started_monotonic':10.,'deadline_monotonic':160.}
        request={'packet':{'execution_source':source,'queue_root':str(queue),'coordinator':{'study_key':'synthetic'}},
                 'worker':worker,'row':row,'command':command}
        durable={'token':'synthetic','source':source,'lease_fds':worker['lease_fds'],'command':command,
                 'started_monotonic':10.,'deadline_monotonic':160.}
        helper.atomic_json(lease.parent/'supervisor-request.json',durable)
        yield request,a.fileno(),lease.parent/'supervisor-request.json'


def test_two_real_inherited_descriptors_token_source_deadline(helper,lease_request):
    request,fd,_=lease_request
    assert helper.verify_lease(request,fd,now=159.)['token']=='synthetic'
    with pytest.raises(TimeoutError): helper.verify_lease(request,fd,now=160.)


@pytest.mark.parametrize('change',['token','source','command','one_fd','deadline','foreign_study','wrong_fd'])
def test_forged_durable_lease_cannot_authorize_child(helper,lease_request,change,tmp_path):
    request,fd,path=lease_request;durable=helper.read_json(path)
    if change=='token': durable['token']='foreign'
    elif change=='source': durable['source']={'foreign':True}
    elif change=='command': durable['command']=['foreign']
    elif change=='one_fd': durable['lease_fds']=[fd]
    elif change=='deadline': durable['deadline_monotonic']+=60
    elif change=='foreign_study': request['packet']['coordinator']['study_key']='other'
    else: request['worker']['lease_fd']+=1
    helper.atomic_json(path,durable)
    with pytest.raises((ValueError,OSError)): helper.verify_lease(request,fd,now=20.)


def attestation_fixture(helper,planned,tmp_path,status='FAIL'):
    packet=deepcopy(planned);row=deepcopy(packet['rows'][0]);row['attempt_token']='synthetic'
    target=tmp_path/'scientific-synthetic';target.mkdir()
    (target/'request.json').write_text('{}\n');(target/'result.json').write_text('{"synthetic_only":true}\n')
    (target/'goal.gif').write_bytes(b'synthetic-media-identity-only')
    declared=packet['protocol']['rows'][0]
    grade={'status':status,'original_gate':status,'full_protocol_complete':True,
           'completed_steps':600,'metric_observations':24,'artifacts':helper.artifacts(target),
           'result_path':str(target/'result.json'),'result_sha256':helper.sha(target/'result.json')}
    media={'file':'goal.gif','sha256':helper.sha(target/'goal.gif'),'bytes':(target/'goal.gif').stat().st_size,
           'actual_steps':declared['media_steps'],'frames':9,'metric_only':False}
    att={'schema':'pr223_full19_case_attestation_v1','status':'COMPLETE','original_gate':status,
         'case_id':row['id'],'case_sha256':row['case_sha256'],'source_digest':'synthetic',
         'protocol_sha256':helper.stable_hash(packet['protocol']),'request_sha256':helper.sha(target/'request.json'),
         'attempt_token':'synthetic','completed_before_deadline':True,'started_monotonic':0.,
         'attested_monotonic':149.,'deadline_monotonic':150.,'scientific_returncode':0,
         'expected_child_returncode':0 if status=='PASS' else 1,'grade':grade,'goal_media':media,
         'complete_recipe':declared['resolved_recipe']}
    packet['execution_source']={'digest':'synthetic'}
    terminal={'attempt_status':'completed','token':'synthetic','child_returncode':att['expected_child_returncode'],
              'paid_wall_seconds':149.25}
    helper.atomic_json(target/'case-attestation.json',att)
    return packet,row,target,terminal,att


@pytest.mark.parametrize('status',['PASS','FAIL'])
def test_numeric_grade_and_child_exit_agree_without_rescore(helper,planned,tmp_path,status):
    packet,row,target,terminal,att=attestation_fixture(helper,planned,tmp_path,status)
    assert helper.verify_attestation(packet,row,target,terminal)['original_gate']==status


@pytest.mark.parametrize('change',['exit','late','token','source','raw','media','clock','recipe','full','artifact_mapping','foreign_result'])
def test_partial_forged_or_changed_artifacts_never_get_credit(helper,planned,tmp_path,change):
    packet,row,target,terminal,att=attestation_fixture(helper,planned,tmp_path)
    if change=='exit': terminal['child_returncode']=0
    elif change=='late': att['attested_monotonic']=150.
    elif change=='token': terminal['token']='wrong'
    elif change=='source': att['source_digest']='old_source'
    elif change=='raw': (target/'result.json').write_text('tampered')
    elif change=='media': (target/'goal.gif').write_bytes(b'tampered')
    elif change=='clock': att['goal_media']['actual_steps'][-1]=599
    elif change=='recipe': att['complete_recipe']['prior_lr_mult']=1.5
    elif change=='artifact_mapping': att['grade']['artifacts'].pop('request.json')
    elif change=='foreign_result':
        outside=tmp_path/'foreign-result.json';outside.write_bytes((target/'result.json').read_bytes())
        att['grade']['result_path']=str(outside)
    else: att['grade']['full_protocol_complete']=False
    helper.atomic_json(target/'case-attestation.json',att)
    with pytest.raises(ValueError): helper.verify_attestation(packet,row,target,terminal)


def test_original_scorer_location_and_calls_are_unchanged(helper):
    screen=(ROOT/helper.legacy.ADAPTER/'screen_current.py').read_text()
    marker="capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')"
    assert screen.count(marker)==1
    assert screen.count("str(HARNESS / 'native100_score.py')")==1
    scorer=Path(helper.legacy.HARNESS)/'native100_score.py'
    source=scorer.read_text()
    assert source.count('    coverage = gate.score_run(run_dir, problem)')==1
    assert source.count('    accuracy = accuracy_gate.score_run(run_dir, problem, coverage)')==1


def test_all19_exact_compiled_wrappers_without_importing_models(helper,planned):
    packet=deepcopy(planned);inputs=packet['external_inputs']
    packet['snapshot_locations']={**inputs,'package':str(ROOT),'adapter':str(ROOT/helper.legacy.ADAPTER)}
    for row in packet['rows']:
        entries=helper.build_wrappers(packet,row,Path('/never-created-software-output')/row['id'],
                                      {'lease_fd':101,'lease_fds':[100,101]})
        for entry in entries:
            assert hashlib.sha256(entry['source'].encode()).hexdigest()==entry['executed_sha256']
            compile(entry['source'],entry['executed_path'],'exec')
        if row['group']!='moving':
            score=next(e['source'] for e in entries if e['label']=='native-score-observed-wrapper')
            assert score.count('coverage = gate.score_run(run_dir, problem)')==1
            assert score.count('accuracy = accuracy_gate.score_run(run_dir, problem, coverage)')==1
            fixture=Path(inputs['harness'])/'tasks/native100_fixture.json'
            assert f"FIXTURE = json.loads(Path({str(fixture)!r}).read_text())" in score
            assert "Path(__file__).resolve().parent / 'tasks'" not in score
            assert '_r.verify_lease(_q,101)' in score and 'derived=_pins' in score
            screen=entries[-1]['source']
            assert 'pass_fds=(100, 101)' in screen
            assert 'start_new_session' not in screen
    assert not Path('/never-created-software-output').exists()


def test_actual_copied_preflight_uses_retained_shared_metadata_cap(helper,monkeypatch,tmp_path):
    phases=[]
    class SyntheticLedger:
        def __init__(self,path): assert path=='synthetic_same_ledger'
        @contextmanager
        def phase(self,name):
            phases.append(name);yield
        def snapshot(self): return {'charged_seconds':.25,'synthetic_only':True}
    monkeypatch.setattr(helper,'ledger_module',lambda:SimpleNamespace(SharedMetadataLedger=SyntheticLedger))
    monkeypatch.setattr(helper,'copied_preflight',lambda packet:{'status':'PASS_COPIED_METADATA_ONLY'})
    packet={'metadata_ledger_path':'synthetic_same_ledger',
            'execution_source':{'snapshot_path':'synthetic_only','origin_commit':'synthetic'},
            'copied_preflight_receipt_path':str(tmp_path/'synthetic-preflight.json')}
    result=helper.bounded_copied_preflight(packet)
    assert phases==['copied_source_model_free_preflight']
    assert result['metadata_cost']['charged_seconds']==.25
    saved=helper.read_json(packet['copied_preflight_receipt_path'])
    assert saved['prepared_packet_sha256']==helper.stable_hash(packet)
    assert saved['helper_sha256']==helper.sha(HERE/'run_retest.py')


def test_raw_result_stays_unaccepted_when_owner_attestation_is_absent(helper,tmp_path):
    row={'group':'native'}
    assert helper.retained_result(row,tmp_path)=={'available':False,'accepted_numeric_credit':False}
    path=tmp_path/'result.json';path.write_text('{"status":"PASS","completed_steps":7000}')
    result=helper.retained_result(row,tmp_path)
    assert result['reported_status']=='PASS' and result['reported_completed_steps']==7000
    assert result['accepted_numeric_credit'] is False and result['sha256']==helper.sha(path)


def test_case_overrun_never_becomes_successful_retest_even_with_original_passes(helper,planned,tmp_path):
    packet=deepcopy(planned)
    for row in packet['rows']:
        row.update(status='PASS',full_protocol_complete=True,media={'synthetic_only':True},
                   **helper.ledger_module().case_cost(row['allowance_seconds'],
                    {'attempt_status':'completed','paid_wall_seconds':1.},certified=True))
    row=packet['rows'][-1]
    row.update(helper.ledger_module().case_cost(row['allowance_seconds'],
                {'attempt_status':'completed','paid_wall_seconds':row['allowance_seconds']+.001},certified=True))
    ledger=helper.ledger_module().SharedMetadataLedger(tmp_path/'synthetic-metadata.json')
    helper._save(tmp_path/'synthetic-study.json',packet,ledger)
    assert packet['scientific_status']=='PASS' and packet['status']=='INCOMPLETE'
    assert packet['budget_status']=='EXCEEDED_OR_INTERRUPTED' and packet['accepted_retest_complete'] is False
    assert all(row['status']=='PASS' for row in packet['rows'])


@pytest.mark.parametrize('change',['missing','foreign_source','foreign_protocol','foreign_helper','short'])
def test_missing_foreign_or_partial_preflight_never_reaches_registration(helper,planned,tmp_path,monkeypatch,change):
    packet=deepcopy(planned)
    packet.update(execution_source={'digest':'synthetic','snapshot_path':str(tmp_path/'snapshot'),
            'origin_commit':'synthetic','files':{helper.SELF:helper.sha(HERE/'run_retest.py')}},
        copied_preflight_receipt_path=str(tmp_path/'copied.json'))
    record={'schema':'pg_pr223_copied_source_preflight_v1','status':'PASS_COPIED_METADATA_ONLY',
            'prepared_packet_sha256':helper.stable_hash(packet),'source_digest':'synthetic',
            'protocol_sha256':helper.stable_hash(packet['protocol']),
            'snapshot_path':str(tmp_path/'snapshot'),'origin_commit':'synthetic',
            'helper_sha256':helper.sha(HERE/'run_retest.py'),'cases':19,'updates':48800,
            'compiled_original_wrappers':19,'models':0,'sampler_calls':0,'scorer_calls':0,'queue_calls':0,
            'numeric_credit':False}
    if change=='foreign_source': record['source_digest']='foreign'
    elif change=='foreign_protocol': record['protocol_sha256']='foreign'
    elif change=='foreign_helper': record['helper_sha256']='foreign'
    elif change=='short': record['compiled_original_wrappers']=18
    if change!='missing': helper.atomic_json(packet['copied_preflight_receipt_path'],record)
    class SyntheticLedger:
        def __init__(self,path): pass
        @contextmanager
        def phase(self,name): yield
    monkeypatch.setattr(helper,'ledger_module',lambda:SimpleNamespace(SharedMetadataLedger=SyntheticLedger))
    monkeypatch.setattr(helper,'prepare',lambda *a,**k:packet)
    monkeypatch.setattr(helper,'validate_packet',lambda *a,**k:None)
    def forbidden(*a,**k): raise AssertionError('preflight failure reached registration/admission')
    monkeypatch.setattr(helper,'PolicyCoordinator',forbidden)
    with pytest.raises(ValueError,match='copied-source preflight'):
        helper.run(tmp_path/'never-created',root=ROOT)


def test_matching_copied_preflight_is_accepted_without_any_model_or_queue_call(helper,planned,tmp_path):
    packet=deepcopy(planned)
    packet.update(execution_source={'digest':'synthetic','snapshot_path':'synthetic_only',
                  'origin_commit':'synthetic','files':{helper.SELF:helper.sha(HERE/'run_retest.py')}},
                  copied_preflight_receipt_path=str(tmp_path/'copied.json'))
    record={'schema':'pg_pr223_copied_source_preflight_v1','status':'PASS_COPIED_METADATA_ONLY',
            'prepared_packet_sha256':helper.stable_hash(packet),'source_digest':'synthetic',
            'protocol_sha256':helper.stable_hash(packet['protocol']),
            'snapshot_path':'synthetic_only','origin_commit':'synthetic',
            'helper_sha256':helper.sha(HERE/'run_retest.py'),'cases':19,'updates':48800,
            'compiled_original_wrappers':19,'models':0,'sampler_calls':0,'scorer_calls':0,'queue_calls':0,
            'numeric_credit':False}
    helper.atomic_json(packet['copied_preflight_receipt_path'],record)
    assert helper.require_copied_preflight(packet)==record


def test_resume_recertifies_completed_row_cost_against_durable_terminal(helper,planned,tmp_path):
    packet,row,target,terminal,att=attestation_fixture(helper,planned,tmp_path)
    terminal['paid_wall_seconds']=1.
    att['attested_monotonic']=.5
    helper.atomic_json(target/'case-attestation.json',att)
    row.update(helper.ledger_module().case_cost(150.,terminal,certified=True))
    assert helper.recertify_row(packet,row,target,terminal)['original_gate']=='FAIL'
    # A coherent lower spend still contradicts the actual completed terminal.
    row.update(helper.ledger_module().case_cost(150.,{**terminal,'paid_wall_seconds':.5},certified=True))
    with pytest.raises(ValueError,match='durable terminal'):
        helper.recertify_row(packet,row,target,terminal)


@pytest.mark.parametrize('paid',[0.,-1.,.01,float('nan'),float('inf')])
def test_completed_cost_cannot_drop_or_clip_final_attestation_time(helper,planned,tmp_path,paid):
    packet,row,target,terminal,att=attestation_fixture(helper,planned,tmp_path)
    terminal['paid_wall_seconds']=paid
    with pytest.raises(ValueError,match='paid time'):
        helper.verify_attestation(packet,row,target,terminal)


def test_fresh_pythonpath_snapshot_alias_is_canonical_before_protected_imports(helper):
    # Actual PEP420 namespace discovery, no hosts/models/snapshots or queue calls.
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                    OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT)+os.sep+'.')
    code=("import importlib.util,json,sys\n"
          f"spec=importlib.util.spec_from_file_location('_pr223_namespace_regression',{str(HERE/'run_retest.py')!r})\n"
          "helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)\n"
          "import experiments\n"
          "assert len(list(experiments.__path__))==1,list(experiments.__path__)\n"
          "source=helper.source_requirements(helper.ROOT);source['snapshot_path']=str(helper.ROOT)\n"
          "sys.path.insert(0,str(helper.ROOT)) # original moving/native insertion\n"
          "helper.guard_imports(source)\n"
          "assert len(list(experiments.__path__))==1,list(experiments.__path__)\n"
          "assert not any(n=='torch' or n=='particlegan' or n.startswith('particlegan.') for n in sys.modules)\n"
          "print(json.dumps({'namespace_paths':list(experiments.__path__),'models':0,'queue_calls':0}))\n")
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout)
    assert output['namespace_paths']==[str(ROOT/'experiments')]
    assert output['models']==output['queue_calls']==0


def test_canonical_normalization_retains_foreign_path_and_original_precedence(helper,tmp_path,monkeypatch):
    first=tmp_path/'first';foreign=tmp_path/'foreign';first.mkdir();foreign.mkdir()
    monkeypatch.setattr(sys,'path',[str(first),str(first)+os.sep+'.',str(foreign)])
    assert helper.normalize_paths()==[str(first),str(foreign)]
    assert helper.normalize_paths([foreign])==[str(foreign),str(first)]
