"""Synthetic metadata/bootstrap controls; no scientific fixture is instantiated."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'reports/forge/pr223-original-full-retest-20261004'
LIVE_LEDGER=Path('/ml2/hypergan/.pg-pr223-full-original-retest-20261004.pr223-full19-metadata-cost.json')
CONTROL_OUTPUT=os.environ.get('PARTICLEGAN_NATIVE3_SOFTWARE_OUTPUT')
ALLOWED_WORKSPACE=(ROOT.resolve(),)+(tuple([Path(CONTROL_OUTPUT).resolve()]) if CONTROL_OUTPUT else ())


def refuse_live_ledger_open(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
        path=Path(os.fsdecode(values[0])).resolve()
        ledger_family=(path.parent==LIVE_LEDGER.parent and
                       (path.name.startswith(LIVE_LEDGER.name) or path.name.startswith('.'+LIVE_LEDGER.name)))
        foreign_workspace=(path.is_relative_to('/ml2/hypergan') and
                           not any(path.is_relative_to(p) for p in ALLOWED_WORKSPACE))
        queue=path.is_relative_to(ROOT/'runs/forge')
        if ledger_family or foreign_workspace or queue or path.is_relative_to('/home/martyn/dev/ParticleGAN/artifacts'):
            raise AssertionError('software controls must never read or mutate LIVE ledgers, raw studies or queues')


# An isolation mistake must fail before opening the authoritative path even if
# a future synthetic fixture forgets to replace a freshly loaded ledger module.
sys.addaudithook(refuse_live_ledger_open)


@pytest.fixture(scope='module')
def helper():
    spec=importlib.util.spec_from_file_location('_pr223_native3_software_tests',OLD/'run_retest.py')
    loaded=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loaded;spec.loader.exec_module(loaded)
    return loaded


def test_live_metadata_open_guard_refuses_before_any_file_operation():
    with pytest.raises(AssertionError,match='never read or mutate'):
        refuse_live_ledger_open('open',(str(LIVE_LEDGER),'r',0))


@pytest.mark.parametrize('relative',['ledger_lock','ledger_temporary','old_raw','shared_queue','checkout_queue','alias'])
def test_software_isolation_also_refuses_lock_temporary_raw_queue_and_alias_paths(relative,tmp_path):
    if relative=='ledger_lock': path=LIVE_LEDGER.with_name(LIVE_LEDGER.name+'.lock')
    elif relative=='ledger_temporary': path=LIVE_LEDGER.with_name(LIVE_LEDGER.name+'.synthetic-tmp')
    elif relative=='old_raw': path=Path('/ml2/hypergan/pg-pr223-full-original-retest-20261004/native/grid100/result.json')
    elif relative=='shared_queue': path=Path('/ml2/hypergan/runs/forge/queue.json')
    elif relative=='checkout_queue': path=ROOT/'runs/forge/queue.json'
    else:
        path=tmp_path/'private-alias';path.symlink_to(LIVE_LEDGER)
    # Call the predicate directly: no real forbidden OS open is attempted.
    with pytest.raises(AssertionError,match='never read or mutate'):
        refuse_live_ledger_open('open',(str(path),'r',0))


@pytest.fixture
def scope(helper):
    return helper.native3_module()


def closed_parent_fixture(helper):
    # Exact authorized CLOSED parent metadata bytes; never reads the LIVE
    # ledger, constructs a ledger owner, or supplies a scientific verdict.
    return helper.read_json(HERE/'fixtures/closed-parent-metadata.json')


def synthetic_base(helper,scope):
    card=helper.protocol.load(ROOT)
    files={'/synthetic/source-only-host.py':'b'*64}
    definitions={};rows=[]
    for declared in card['rows']:
        parent=deepcopy(declared['original_definition']);parent['external_inputs_sha256']=scope.digest(files)
        definition=deepcopy(parent);definition['id']=declared['id']
        definition['fresh_retest']=dict(parent_original_id=parent['id'],
            historical_case_sha256=declared['historical_case_sha256'],source_derived_parent_sha256=scope.digest(parent),
            goal_media_steps=declared['media_steps'],goal_observer_sha256=helper.sha(OLD/'goal_observer.py'),
            scope='fresh full original19 noisy selected law; no current26/default/speed credit')
        definitions[definition['id']]=definition
        rows.append(dict(id=definition['id'],group=definition['group'],task=definition['task'],status='NOT_RUN',
            case_sha256=scope.digest(definition),timeout_seconds=declared['proposed_inclusive_allowance_seconds'],
            allowance_seconds=declared['proposed_inclusive_allowance_seconds']))
    source_files={helper.SELF:helper.sha(OLD/'run_retest.py'),helper.protocol.PROTOCOL:helper.sha(ROOT/helper.protocol.PROTOCOL),
                  helper.DIRECTORY+'/goal_observer.py':helper.sha(OLD/'goal_observer.py')}
    return dict(schema='pg_pr223_original_full_retest_v1',protocol=card,
        spec=dict(id='pr223-original-full19-retest-20261004',export_grace_seconds=0,retries=0,
            total_paid_cap_seconds=10800,case_caps_sum_seconds=9810,shared_metadata_and_finalization_seconds=180),
        family='atlas',required=19,rows=rows,case_definitions=definitions,
        recipe_overrides=dict(original_config_sha256=helper.legacy.CONFIG_SHA,original_options=deepcopy(helper.legacy.OPTIONS)),
        external_inputs=dict(files=files),source=dict(commit='b'*40,files_sha256=source_files),
        preflight=dict(status='PASS_METADATA_ONLY',cases=19,updates=48800,models=0,sampler_calls=0,scorer_calls=0))


@pytest.fixture
def packet(helper,scope,monkeypatch,tmp_path):
    anchor_path=tmp_path/'private-closed-parent-copy.json'
    anchor_path.write_bytes((HERE/'fixtures/closed-parent-metadata.json').read_bytes())
    anchor=dict(path=str(anchor_path),sha256=scope.ANCHOR_SHA,bytes=scope.ANCHOR_BYTES)
    base=synthetic_base(helper,scope)
    monkeypatch.setattr(helper,'plan',lambda root=ROOT:deepcopy(base))
    monkeypatch.setattr(helper,'source_requirements',lambda root:dict(origin_commit='b'*40,files=deepcopy(base['source']['files_sha256'])))
    return scope.plan(ROOT,helper,anchor)


def test_exact_closed_parent_fixture_and_later_same_ledger_phase_are_preserved(helper,scope,packet):
    source=HERE/'fixtures/closed-parent-metadata.json'
    assert source.stat().st_size==scope.ANCHOR_BYTES==2675
    assert helper.sha(source)==scope.ANCHOR_SHA=='ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369'
    anchor=dict(path=str(source),sha256=scope.ANCHOR_SHA,bytes=scope.ANCHOR_BYTES)
    closed=scope.validate_anchor(anchor,helper)
    assert len(closed['phases'])==12 and closed['current_phase'] is None
    assert packet['metadata_history']['closed_state']==closed
    later=deepcopy(closed)
    later['phases'].append(dict(index=12,name='native_three_parent_anchor_sealing',status='COMPLETE',
                               paid_wall_seconds=.002600732957944,paused_wall_seconds=0.))
    helper.ledger_module()._update_totals(later)
    assert scope.validate_live_history(packet,later,helper) is later
    assert math.isclose(later['charged_seconds'],44.374923426192254,rel_tol=0.,abs_tol=1e-9)


def test_actual_committed_history_is_bound_without_raw_or_ledger_reads(helper,scope):
    raw,cost=scope.history(ROOT,helper)
    assert raw['completed']==16 and cost['costs']['case_charged_seconds']==scope.PRIOR_CASE_SECONDS
    assert [r['execution_status'] for r in raw['rows'][-3:]]==['INVALID','NOT_RUN','NOT_RUN']
    assert cost['authoritative_final_metadata_ledger']['path']==scope.CANONICAL_LEDGER


def test_new_three_slots_preserve_full_original_native_law_and_no_old_grades(helper,scope,packet):
    assert helper.validate_packet(packet) is packet
    assert packet['required']==3 and packet['original_required']==19
    assert [r['id'] for r in packet['rows']]==list(scope.IDS)
    assert packet['protocol']['original19_catalog_sha256']==scope.ORIGINAL_PROTOCOL_DIGEST
    assert sum(r['allowance_seconds'] for r in packet['rows'])==4290
    assert all(r['status']=='NOT_RUN' and 'original_gate' not in r for r in packet['rows'])
    for active,parent in zip(packet['protocol']['rows'],packet['protocol']['original19_catalog']['rows'][-3:]):
        assert active['resolved_recipe']==parent['resolved_recipe']
        assert active['original_definition']==parent['original_definition']
        assert active['observer']==parent['observer']
        assert active['resolved_recipe']['lr']==.00425 and active['resolved_recipe']['prior_lr_mult']==2.
        assert active['resolved_recipe']['total_steps'] is None
        assert active['original_definition']['original_host']['steps']==7000
        assert active['observer']['observations']==34 and active['observer']['samples']==20000
        assert active['observer']['independent_holdout_samples']==100000


@pytest.mark.parametrize('change',['missing','duplicate','reordered','portability','full19','old_grade','old_id',
    'old_credit','retry','grace','reset_total','reset_prior','smaller_cap','extra_resources','catalog_gate',
    'law','seed','steps','cadence','recipe','definition_hash','external_source','ledger_path','open_anchor',
    'history_removed','history_reordered','wrong_anchor_pin'])
def test_wrong_scope_source_cost_and_old_credit_refused(helper,scope,packet,change):
    value=deepcopy(packet)
    if change=='missing': value['rows'].pop()
    elif change=='duplicate': value['rows'][1]=deepcopy(value['rows'][0])
    elif change=='reordered': value['rows'].reverse()
    elif change=='portability': value['rows'][0]['group']='portability'
    elif change=='full19': value['rows']=synthetic_base(helper,scope)['rows']
    elif change=='old_grade': value['rows'][0].update(status='PASS',original_gate='PASS')
    elif change=='old_id': value['rows'][0]['id']=value['protocol']['rows'][0]['parent_retest_id']
    elif change=='old_credit': value['old_results_are_current_credit']=True
    elif change=='retry': value['spec']['retries']=1
    elif change=='grace': value['spec']['export_grace_seconds']=60
    elif change=='reset_total': value['spec']['total_paid_cap_seconds']+=scope.PRIOR_CASE_SECONDS
    elif change=='reset_prior': value['spec']['prior_case_charged_seconds']=0
    elif change=='smaller_cap': value['rows'][0]['allowance_seconds']=900
    elif change=='extra_resources': value['spec']['resources']['gpus']=0
    elif change=='catalog_gate': value['protocol']['original19_catalog']['rows'][-3]['original_definition']['original_requirements'][0][2]=0
    elif change=='recipe': value['protocol']['rows'][0]['resolved_recipe']['lr']=.0053125
    elif change=='definition_hash': value['rows'][0]['case_sha256']='0'*64
    elif change=='external_source': value['external_inputs']['files']['/synthetic/source-only-host.py']='c'*64
    elif change=='ledger_path': value['metadata_history']['canonical_path']='/synthetic/new180.json'
    elif change=='open_anchor': value['metadata_history']['closed_state']['current_phase']=dict(index=12,name='synthetic_open',status='ACTIVE',paid_wall_seconds=0.,paused_wall_seconds=0.)
    elif change=='history_removed': value['metadata_history']['closed_state']['phases'].pop()
    elif change=='history_reordered': value['metadata_history']['closed_state']['phases'].reverse()
    elif change=='wrong_anchor_pin': value['metadata_history']['closed_anchor']['sha256']='0'*64
    else:
        d=value['case_definitions'][value['rows'][0]['id']]
        if change=='law': d['sampling']='clean-only replacement'
        elif change=='seed': d['original_host']['seed']=0
        elif change=='steps': d['original_host']['steps']=1000
        else: d['observation_steps']=d['observation_steps'][:-1]
        value['rows'][0]['case_sha256']=scope.digest(d) # Coherent hash forgery is still refused.
    with pytest.raises((ValueError,KeyError)): helper.validate_packet(value)


def extended_snapshot(packet,helper,paid=.2):
    state=deepcopy(packet['metadata_history']['closed_state'])
    state['phases'].append(dict(index=12,name='synthetic_new_metadata',status='COMPLETE',paid_wall_seconds=paid,paused_wall_seconds=0.))
    helper.ledger_module()._update_totals(state)
    return state


def test_same_metadata_once_prior_cost_once_and_full_next_native_reservation(helper,scope,packet):
    state=extended_snapshot(packet,helper)
    cost=scope.require_next_reservation(packet,state,1470,helper)
    assert math.isclose(cost['charged_seconds'],scope.PRIOR_CASE_SECONDS+scope.PRIOR_METADATA_SECONDS+.2)
    assert cost['prior_case_charged_seconds']==scope.PRIOR_CASE_SECONDS
    assert cost['current_case_charged_seconds']==0
    assert scope.PRIOR_CASE_SECONDS+4290+180==7635.841891122982
    assert cost['next_allowance_seconds']==1470


@pytest.mark.parametrize('change',['reset_history','changed_history','negative_paid','nonfinite_paid','reduced_paid',
                                  'changed_completion','partial_cost','wrong_cap','interrupted_reserve','overrun'])
def test_cost_history_or_terminal_substitution_cannot_admit(helper,scope,packet,change):
    value=deepcopy(packet);value['status']='READY';state=extended_snapshot(value,helper)
    row=value['rows'][0];row.update(status='FAIL',**helper.ledger_module().case_cost(1470,dict(attempt_status='completed',paid_wall_seconds=10.),certified=True))
    if change=='reset_history': state=helper.ledger_module()._initial()
    elif change=='changed_history': state['phases'][0]['name']='forged source setup'
    elif change=='negative_paid': row['paid_wall_seconds']=-1.
    elif change=='nonfinite_paid': row['paid_wall_seconds']=float('nan')
    elif change=='reduced_paid': row['charged_seconds']=0.
    elif change=='changed_completion': row['completed_terminal']=False
    elif change=='partial_cost': row.pop('certified')
    elif change=='wrong_cap': row['allowance_seconds']=1.
    elif change=='interrupted_reserve': row.update(helper.ledger_module().case_cost(1470,dict(attempt_status='error',paid_wall_seconds=10.),certified=False));row['reserved_seconds']=0.
    else: row.update(helper.ledger_module().case_cost(1470,dict(attempt_status='completed',paid_wall_seconds=1471.),certified=False))
    with pytest.raises((ValueError,RuntimeError)): scope.require_next_reservation(value,state,1440,helper)


def test_missing_terminal_reserves_full_new_cap_and_final_projection_preserves_overshoot(helper,scope,packet):
    value=deepcopy(packet);value['status']='INCOMPLETE';state=extended_snapshot(value,helper)
    value['rows'][0].update(status='INCOMPLETE',**helper.ledger_module().case_cost(1470,None,certified=False))
    assert scope.require_next_reservation(value,state,0,helper)['current_case_reserved_seconds']==1470
    value['rows'][0].update(helper.ledger_module().case_cost(1470,dict(attempt_status='completed',paid_wall_seconds=1471),certified=False))
    final=scope.require_next_reservation(value,state,0,helper)
    assert final['halt_required'] and final['overrun_seconds']==1 and final['current_case_paid_wall_seconds']==1471


def test_no_new_ledger_or_foreign_supplied_ledger(helper,scope,packet,monkeypatch,tmp_path):
    missing=tmp_path/'no-old-ledger.json'
    monkeypatch.setattr(scope,'CANONICAL_LEDGER',str(missing))
    with pytest.raises(ValueError,match='same canonical'): scope.require_existing_ledger()
    # No constructor can erase the canonical identity via the optional helper argument.
    ledger=SimpleNamespace(path=tmp_path/'new180.json')
    with pytest.raises(ValueError,match='exact existing parent'):
        helper.prepare(tmp_path/'unused',ledger=ledger,native3_anchor=packet['metadata_history']['closed_anchor'])
    assert not missing.exists() and not ledger.path.exists()


def pin_source(root,helper):
    files={p.relative_to(root).as_posix():helper.sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.json'}}
    return dict(snapshot_path=str(root),files=files,digest=helper.stable_hash(files),origin_commit='b'*40)


@pytest.fixture
def boundary_packet(helper,scope,tmp_path):
    root=tmp_path/'synthetic-boundary-source';root.mkdir()
    for path in (ROOT/'experiments/forge').rglob('*.py'):
        target=root/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    for name in (helper.SELF,helper.DIRECTORY+'/protocol.py',helper.protocol.LEGACY,
                 scope.DIRECTORY+'/scorer_boundary_control.py'):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    harness=root/'atlas19-external/harness';adapter=root/'atlas19-external/adapter';native=root/'atlas19-external/native_root'
    harness.mkdir(parents=True);adapter.mkdir(parents=True)
    fixture=(OLD/'fixtures/native100_score.py.txt').read_bytes()
    assert len(fixture)==1660 and hashlib.sha256(fixture).hexdigest()=='10cc14edfcd98ab34fd3768aaba2ee835dc2241dc1face2e18998c8f2b687feb'
    (harness/'native100_score.py').write_bytes(fixture)
    for name in ('screen_current.py','current_api_fixtures.py'):
        (adapter/name).write_bytes((ROOT/helper.legacy.ADAPTER/name).read_bytes())
    sources={'lib/toy_models.py':'# inert\n','particlegan/__init__.py':'# never imported\n',
        'benchmarks/__init__.py':'# inert\n','benchmarks/toy100/__init__.py':'# inert\n',
        'benchmarks/toy100/train.py':'# never imported\n','benchmarks/toy100/gate.py':'# never scored\n',
        'benchmarks/toy100/accuracy_gate.py':'# never scored\n'}
    for name,text in sources.items():
        target=native/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    return dict(schema=scope.SCHEMA,execution_source=pin_source(root,helper),
                snapshot_locations=dict(harness=str(harness),adapter=str(adapter),native_root=str(native)),
                protocol={'synthetic_only':True},copied_preflight_receipt_path=str(tmp_path/'synthetic-preflight.json'))


def boundary(helper,scope):
    return helper.module(scope.DIRECTORY+'/scorer_boundary_control.py','_pr223_native3_boundary')


def test_exact_source_generated_prefix_regression_is_fake_and_stops_before_score(helper,scope,boundary_packet,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    before=set(sys.modules)
    proof=boundary(helper,scope).run(boundary_packet,helper)
    assert proof['outcomes']=={'repaired':'PASS_STOPPED_BEFORE_SCORE','predecessor':'REFUSED_AMBIGUOUS_NAMESPACE',
                              'foreign':'REFUSED_FOREIGN_NAMESPACE','cached':'REFUSED_CACHED_CANDIDATE'}
    assert all(proof[k]==0 for k in ('models','sampler_calls','scorer_calls','queue_calls'))
    assert not any(n.split('.',1)[0] in {'torch','particlegan','lib','benchmarks'} for n in set(sys.modules)-before)


def preflight_receipt(helper,scope,packet):
    _,imports=helper.native_scorer_import_metadata(packet['execution_source'],packet['snapshot_locations']['native_root'],
                                                  paths=[packet['execution_source']['snapshot_path']])
    checker=boundary(helper,scope)
    proof=dict(schema=checker.SCHEMA,status='PASS_SYNTHETIC_SCORER_BOUNDARY',binding=checker.binding(packet,helper),
        outcomes=checker.EXPECTED,synthetic_only=True,models=0,sampler_calls=0,scorer_calls=0,queue_calls=0,
        numerical_credit=False,old_arrays_read=False)
    return dict(schema=scope.PREFLIGHT_SCHEMA,status='PASS_COPIED_METADATA_ONLY',
        prepared_packet_sha256=helper.stable_hash(packet),source_digest=packet['execution_source']['digest'],
        protocol_sha256=helper.stable_hash(packet['protocol']),snapshot_path=packet['execution_source']['snapshot_path'],
        origin_commit=packet['execution_source']['origin_commit'],helper_sha256=packet['execution_source']['files'][helper.SELF],
        cases=3,updates=21000,compiled_original_wrappers=3,native_scorer_imports=imports,
        native_scorer_boundary_control=proof,models=0,sampler_calls=0,scorer_calls=0,queue_calls=0,numeric_credit=False)


@pytest.mark.parametrize('change',['missing','wrong_schema','foreign_source','full19','wrong_helper','missing_boundary',
    'boundary_foreign','boundary_cached_missing','scorer_changed','model_calls'])
def test_missing_foreign_or_partial_copied_proof_never_satisfies_prerequisite(helper,scope,boundary_packet,change):
    path=Path(boundary_packet['copied_preflight_receipt_path'])
    record=preflight_receipt(helper,scope,boundary_packet)
    helper.atomic_json(path,record)
    assert helper.require_copied_preflight(boundary_packet)==record
    if change=='missing': path.unlink()
    elif change=='wrong_schema': record['schema']='pg_pr223_copied_source_preflight_v1'
    elif change=='foreign_source': record['source_digest']='old source'
    elif change=='full19': record.update(cases=19,updates=48800,compiled_original_wrappers=19)
    elif change=='wrong_helper': record['helper_sha256']='0'*64
    elif change=='missing_boundary': record.pop('native_scorer_boundary_control')
    elif change=='boundary_foreign': record['native_scorer_boundary_control']['binding']['source_digest']='old source'
    elif change=='boundary_cached_missing': record['native_scorer_boundary_control']['outcomes'].pop('cached')
    elif change=='scorer_changed': (Path(boundary_packet['snapshot_locations']['harness'])/'native100_score.py').write_text('changed\n')
    else: record['models']=1
    if change!='missing': helper.atomic_json(path,record)
    with pytest.raises(ValueError): helper.require_copied_preflight(boundary_packet)


def test_copied_source_validation_and_receipt_write_are_inside_same_metadata_phase(helper,scope,packet,monkeypatch,tmp_path):
    value=deepcopy(packet)
    value.update(metadata_ledger_path=scope.CANONICAL_LEDGER,
                 copied_preflight_receipt_path=str(tmp_path/'SYNTHETIC-copy-receipt.json'),
                 execution_source={'snapshot_path':'/synthetic/copied-source','origin_commit':'b'*40})
    events=[];active=[False]
    class SyntheticLedger:
        def __init__(self,path):
            assert path==scope.CANONICAL_LEDGER
            events.append('ledger_owner')
        @contextmanager
        def phase(self,name):
            assert name=='copied_source_model_free_preflight'
            active[0]=True;events.append('enter')
            try: yield
            finally: active[0]=False;events.append('exit')
        def snapshot(self): return extended_snapshot(value,helper)
    def validate(p,*,source=False):
        assert active[0] and p is value and source is True
        events.append('source_validation');return p
    def copied(p):
        assert active[0] and p is value
        events.append('fake_boundary');return {'status':'PASS_COPIED_METADATA_ONLY','synthetic_only':True}
    def write(path,result):
        assert active[0]
        events.append('receipt_write')
    budget=helper.ledger_module()
    monkeypatch.setattr(budget,'SharedMetadataLedger',SyntheticLedger)
    monkeypatch.setattr(helper,'ledger_module',lambda:budget)
    monkeypatch.setattr(scope,'require_existing_ledger',lambda:Path(scope.CANONICAL_LEDGER))
    monkeypatch.setattr(helper,'validate_packet',validate)
    monkeypatch.setattr(helper,'copied_preflight',copied)
    monkeypatch.setattr(helper,'atomic_json',write)
    result=helper.bounded_copied_preflight(value)
    assert events==['ledger_owner','enter','source_validation','fake_boundary','receipt_write','exit']
    assert result['schema']==scope.PREFLIGHT_SCHEMA


def test_foreign_preflight_ledger_refuses_before_owner_or_phase(helper,scope,packet,monkeypatch):
    value=deepcopy(packet);value['metadata_ledger_path']='/synthetic/new-180-ledger.json'
    def forbidden(*a,**kw): raise AssertionError('foreign ledger must not be read or owned')
    monkeypatch.setattr(helper,'ledger_module',forbidden)
    monkeypatch.setattr(scope,'require_existing_ledger',forbidden)
    with pytest.raises(ValueError,match='SAME canonical'):
        helper.bounded_copied_preflight(value)


def native_attestation(helper,scope,packet,tmp_path,status):
    value=deepcopy(packet);value['status']='READY';value['execution_source']={'digest':'SYNTHETIC'}
    row=deepcopy(value['rows'][0]);row['attempt_token']='synthetic_token';target=tmp_path/'synthetic-case';target.mkdir()
    (target/'request.json').write_text('{"synthetic_only":true}\n')
    (target/'result.json').write_text('{"synthetic_only":true}\n')
    (target/'goal.gif').write_bytes(b'SYNTHETIC identity only, never rendered')
    declared=value['protocol']['rows'][0]
    grade=dict(status=status,original_gate=status,full_protocol_complete=True,completed_steps=7000,
        metric_observations=34,result_path=str(target/'result.json'),result_sha256=helper.sha(target/'result.json'),
        artifacts=helper.artifacts(target),original_protocol_gate=status,reported_original_status=status,
        native_gates=dict(noisy=dict(coverage=status,accuracy=status),clean=dict(coverage='FAIL',accuracy='FAIL')),
        qualification_input=False)
    media=dict(file='goal.gif',sha256=helper.sha(target/'goal.gif'),bytes=(target/'goal.gif').stat().st_size,
               actual_steps=declared['media_steps'],frames=9,metric_only=False)
    att=dict(schema=scope.ATTESTATION_SCHEMA,status='COMPLETE',original_gate=status,case_id=row['id'],
        case_sha256=row['case_sha256'],source_digest='SYNTHETIC',protocol_sha256=helper.stable_hash(value['protocol']),
        request_sha256=helper.sha(target/'request.json'),attempt_token=row['attempt_token'],completed_before_deadline=True,
        started_monotonic=10.,attested_monotonic=20.,deadline_monotonic=1480.,scientific_returncode=0,
        expected_child_returncode=0 if status=='PASS' else 1,grade=grade,goal_media=media,
        complete_recipe=declared['resolved_recipe'],fresh_full_original19_only=False,
        full_original19_credit=False,execution_scope=scope.execution_scope())
    terminal=dict(attempt_status='completed',token=row['attempt_token'],paid_wall_seconds=10.1,
                  child_returncode=att['expected_child_returncode'])
    helper.atomic_json(target/'case-attestation.json',att)
    return value,row,target,terminal,att


@pytest.mark.parametrize('status',['PASS','FAIL'])
def test_native3_attestation_keeps_original_gate_authority_and_distinct_scope(helper,scope,packet,tmp_path,status):
    value,row,target,terminal,att=native_attestation(helper,scope,packet,tmp_path,status)
    assert helper.verify_attestation(value,row,target,terminal)==att


def test_raw_accuracy_pass_with_noisy_coverage_fail_remains_accepted_scientific_fail(helper,scope,packet,tmp_path):
    value,row,target,terminal,att=native_attestation(helper,scope,packet,tmp_path,'FAIL')
    att['grade']['native_gates']['noisy']=dict(coverage='FAIL',accuracy='PASS')
    att['grade']['reported_original_status']='PASS'
    helper.atomic_json(target/'case-attestation.json',att)
    accepted=helper.verify_attestation(value,row,target,terminal)
    assert accepted['grade']['reported_original_status']=='PASS'
    assert accepted['grade']['original_gate']==accepted['grade']['original_protocol_gate']=='FAIL'
    assert terminal['child_returncode']==1 and accepted['status']=='COMPLETE'


@pytest.mark.parametrize('change',['old_schema','full19_credit','borrowed_scope','exit','deadline','paid','overrun','source','recipe',
                                  'missing_full','observations','steps','media','raw','coverage','borrowed_clean','raw_label'])
def test_no_old_or_incomplete_native_attestation_credit(helper,scope,packet,tmp_path,change):
    value,row,target,terminal,att=native_attestation(helper,scope,packet,tmp_path,'FAIL')
    if change=='old_schema': att['schema']='pr223_full19_case_attestation_v1'
    elif change=='full19_credit': att['full_original19_credit']=True
    elif change=='borrowed_scope': att['execution_scope']['executed_required']=19
    elif change=='exit': terminal['child_returncode']=0
    elif change=='deadline': att['attested_monotonic']=1480.
    elif change=='paid': terminal['paid_wall_seconds']=0.
    elif change=='overrun': terminal['paid_wall_seconds']=1470.01
    elif change=='source': att['source_digest']='old cohort'
    elif change=='recipe': att['complete_recipe']['lr']=.0053125
    elif change=='missing_full': att['grade']['full_protocol_complete']=False
    elif change=='observations': att['grade']['metric_observations']=33
    elif change=='steps': att['grade']['completed_steps']=6999
    elif change=='media': (target/'goal.gif').write_bytes(b'changed')
    elif change=='coverage': att['grade']['native_gates']['noisy']['coverage']='UNKNOWN'
    elif change=='borrowed_clean': att['grade']['native_gates']['noisy']=dict(coverage='PASS',accuracy='PASS')
    elif change=='raw_label': att['grade']['reported_original_status']='PASS'
    else: (target/'result.json').write_text('changed')
    helper.atomic_json(target/'case-attestation.json',att)
    with pytest.raises(ValueError): helper.verify_attestation(value,row,target,terminal)


def fake_dispatch(helper,scope,packet,monkeypatch,tmp_path,*,missing_preflight=False,invalid=False):
    """Exercise ONLY real orchestration with a private inert coordinator/clock."""
    budget=helper.ledger_module();events=[];state=deepcopy(packet['metadata_history']['closed_state'])
    class Ledger:
        path=Path(scope.CANONICAL_LEDGER)
        def __init__(self,path): assert Path(path)==self.path
        @contextmanager
        def phase(self,name):
            state['current_phase']=dict(index=len(state['phases']),name=name,status='ACTIVE',paid_wall_seconds=.001,paused_wall_seconds=0.)
            budget._update_totals(state)
            try: yield self
            finally:
                phase=state.pop('current_phase');phase['status']='COMPLETE';state['phases'].append(phase)
                state['current_phase']=None;budget._update_totals(state)
        @contextmanager
        def pause(self): yield
        def snapshot(self): return deepcopy(budget._validate_snapshot(state))
    class Lease:
        def __init__(self,fd):self.fd=fd
        def fileno(self): return self.fd
    class Coordinator:
        def __init__(self,*a,**k): events.append('construct_coordinator')
        def register(self,prepared,output,family,runtime):
            events.append('register');output.mkdir()
            value=deepcopy(prepared);value['coordinator']=dict(canonical_output=str(output),study_key='synthetic',queue_root=str(tmp_path))
            helper.atomic_json(output/'study.json',value);return 'synthetic',output
        @contextmanager
        def study_lease(self,key): yield Lease(100)
        def recover(self): events.append('recover')
        def retained(self,key): return None
        def attempt_key(self,p,t,row): return row['id']
        @contextmanager
        def admit(self,key,p,row,device):
            assert row['id'] in scope.IDS and row['group']=='native'
            events.append(('admit',row['id']))
            folder=tmp_path/'synthetic-durable'/row['task'];folder.mkdir(parents=True)
            yield dict(status='running',lease_path=str(folder/'execution.lock'),token=row['task'],
                       started_monotonic=1.,deadline_monotonic=1.+row['allowance_seconds']),Lease(101)
        def launch(self,command,p,log,leases,allowance):
            events.append(('launch',allowance))
            assert command[3].endswith(helper.SELF) and '--child' in command
            request=helper.read_json(Path(log).parent/'request.json')
            assert request['worker']['lease_fds']==[100,101]
            row=request['row'];folder=tmp_path/'synthetic-durable'/row['task']
            helper.atomic_json(folder/'supervisor-terminal.json',dict(attempt_status='completed',token=row['task'],
                              paid_wall_seconds=2.,child_returncode=0))
            return SimpleNamespace(returncode=0,paid_wall_seconds=2.)
        def complete(self,key,result): events.append(('complete',key))
        def publish_attachment(self,key,canonical,output): return helper.read_json(canonical/'study.json')
    def prepared(output,**kwargs):
        assert kwargs['native3_anchor']==packet['metadata_history']['closed_anchor']
        value=deepcopy(packet);value['execution_source']=dict(snapshot_path=str(tmp_path/'SYNTHETIC-source'),digest='synthetic')
        value['queue_root']=str(tmp_path);return value
    def proof(p,row,target,terminal):
        events.append(('verify_current_attestation',row['id']))
        if invalid: raise ValueError('synthetic current attestation refused')
        return dict(grade=dict(status='PASS',original_gate='PASS',full_protocol_complete=True,completed_steps=7000,
                              metric_observations=34),goal_media=dict(synthetic_only=True))
    monkeypatch.setattr(scope,'require_existing_ledger',lambda:Path(scope.CANONICAL_LEDGER))
    monkeypatch.setattr(budget,'SharedMetadataLedger',Ledger)
    monkeypatch.setattr(helper,'ledger_module',lambda:budget)
    monkeypatch.setattr(helper,'prepare',prepared)
    # No real source snapshot, raw receipt, GPU, queue or scientific callback.
    monkeypatch.setattr(helper,'validate_packet',lambda p,**kwargs:p)
    def prerequisite(p):
        events.append('require_copied_preflight')
        if missing_preflight: raise ValueError('synthetic missing copied-source proof')
    monkeypatch.setattr(helper,'require_copied_preflight',prerequisite)
    monkeypatch.setattr(helper,'PolicyCoordinator',Coordinator)
    monkeypatch.setattr(helper,'runtime_metadata',lambda:dict(synthetic_only=True,physical_gpu=1,device='cuda:0',torch_threads=1))
    monkeypatch.setattr(helper.legacy,'gpu_readiness',lambda:dict(ready=True))
    monkeypatch.setattr(helper,'verify_attestation',proof)
    monkeypatch.setattr(helper,'retained_result',lambda *args:dict(available=False,accepted_numeric_credit=False))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1') # A string tested by metadata only; no device call.
    output=tmp_path/'SYNTHETIC-study'
    if missing_preflight:
        with pytest.raises(ValueError,match='missing copied-source'): helper.run(output,native3_anchor=packet['metadata_history']['closed_anchor'])
        return None,events
    return helper.run(output,native3_anchor=packet['metadata_history']['closed_anchor']),events


def test_missing_copied_proof_refuses_before_coordinator_registration_or_admission(helper,scope,packet,monkeypatch,tmp_path):
    result,events=fake_dispatch(helper,scope,packet,monkeypatch,tmp_path,missing_preflight=True)
    assert events==['require_copied_preflight'] and result is None


def test_only_three_new_cases_can_reach_maintained_dispatch(helper,scope,packet,monkeypatch,tmp_path):
    result,events=fake_dispatch(helper,scope,packet,monkeypatch,tmp_path)
    assert [e[1] for e in events if isinstance(e,tuple) and e[0]=='admit']==list(scope.IDS)
    assert [e[1] for e in events if isinstance(e,tuple) and e[0]=='launch']==list(scope.CAPS)
    assert len(result['rows'])==3 and result['completed']==3 and result['new_paid_seconds']==6.
    assert result['budget_accounting']['prior_case_charged_seconds']==scope.PRIOR_CASE_SECONDS
    assert result['prior_reference']['execution_counts']=={'PASS':16,'INVALID':1,'NOT_RUN':2}
    assert result['budget_accounting']['charged_seconds']==scope.PRIOR_CASE_SECONDS+6.+result['metadata_cost']['charged_seconds']


def test_first_invalid_attempt_stops_without_retry_or_old_grade_transfer(helper,scope,packet,monkeypatch,tmp_path):
    result,events=fake_dispatch(helper,scope,packet,monkeypatch,tmp_path,invalid=True)
    assert [e[1] for e in events if isinstance(e,tuple) and e[0]=='admit']==[scope.IDS[0]]
    assert [r['status'] for r in result['rows']]==['INVALID','NOT_RUN','NOT_RUN']
    assert result['new_paid_seconds']==2. and result['rows'][0]['reserved_seconds']==0.
    assert result['completed']==0 and result['accepted_retest_complete'] is False
    assert all('original_gate' not in r for r in result['rows'])


def test_scientific_functions_are_byte_exact_to_frozen_portability_source():
    original=subprocess.check_output(['git','show','f57bc3ebc5e2d9091b3700a2f770648635962b90:reports/forge/pr223-original-full-retest-20261004/run_retest.py'],cwd=ROOT,text=True)
    current=(OLD/'run_retest.py').read_text()
    wanted={'science','build_wrappers','guard_imports','native_scorer_import_metadata','normalize_native_scorer_paths',
            'verify_lease','validate_request','retained_result','recertify_row','runtime_metadata'}
    def slices(source):
        nodes=ast.parse(source).body;lines=source.splitlines(keepends=True)
        return {n.name:''.join(lines[n.lineno-1:n.end_lineno]) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in wanted}
    assert slices(current)==slices(original) and set(slices(current))==wanted
    for prefix in ('particlegan','benchmarks','configs','reports/forge/pr223-original-full-retest-20261004/protocol.py',
                   'reports/forge/pr223-original-full-retest-20261004/protocol.json','reports/forge/pr223-original-full-retest-20261004/goal_observer.py',
                   'reports/forge/pr223-original-full-retest-20261004/budget_ledger.py'):
        assert subprocess.check_output(['git','diff','f57bc3eb','--',prefix],cwd=ROOT)==b''
