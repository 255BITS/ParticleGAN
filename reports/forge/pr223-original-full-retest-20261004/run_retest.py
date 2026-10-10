"""Fresh full original PR223 Atlas19 retest with inclusive durable deadlines.

All numerical decisions come from the unchanged original scientific consumers.
The new source overlay captures already-computed tensors for goal media. Each
case's science, original certification, media and final attestations run inside
one maintained PolicyCoordinator child, with no export grace or retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import types

ROOT=Path(__file__).resolve().parents[3]
DIRECTORY='reports/forge/pr223-original-full-retest-20261004'
SELF=DIRECTORY+'/run_retest.py'


def normalize_paths(preferred=(),*,paths=None):
    """Deduplicate canonical aliases, retaining every distinct foreign path."""
    current=sys.path if paths is None else paths
    result=[];seen=set()
    for entry in [*preferred,*current]:
        path=str(Path(entry or os.getcwd()).resolve())
        if path not in seen: result.append(path);seen.add(path)
    sys.path[:]=result
    return result


normalize_paths([ROOT])
from experiments.forge.contracts import atomic_json,read_json,stable_hash
from experiments.forge.policy_execution import PolicyCoordinator,freeze_source
from experiments.forge.sources import snapshot_source,verify_snapshot
from experiments.forge.__main__ import queue_location


def module(relative,name):
    path=ROOT/relative
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


protocol=module(DIRECTORY+'/protocol.py','_pr223_retest_protocol')
legacy=module(protocol.LEGACY,'_pr223_retest_original_driver')
normalize_paths()  # The unchanged legacy driver also inserts its own ROOT.


def observer():
    found=sys.modules.get('_pr223_goal_observer')
    return found or module(DIRECTORY+'/goal_observer.py','_pr223_goal_observer')


def ledger_module():
    return module(DIRECTORY+'/budget_ledger.py','_pr223_retest_budget')


def native3_module():
    """Explicit metadata-only successor; never an alternate scientific loop."""
    found=sys.modules.get('_pr223_native3_contract')
    return found or module('reports/forge/pr223-native3-continuation-20261004/native3_contract.py',
                           '_pr223_native3_contract')


def scoped_module(packet):
    return native3_module() if packet.get('schema')=='pg_pr223_native3_continuation_v2' else None


def reservation(packet,snapshot,next_allowance):
    scope=scoped_module(packet)
    return (scope.require_next_reservation(packet,snapshot,next_allowance,sys.modules[__name__]) if scope
            else ledger_module().require_next_reservation(packet['rows'],snapshot,next_allowance))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar(output):
    output=Path(output).resolve()
    return output.parent/f'.{output.name}.pr223-full19-prepared.json'


def metadata_path(output):
    output=Path(output).resolve()
    return output.parent/f'.{output.name}.pr223-full19-metadata-cost.json'


def copied_preflight_path(output):
    output=Path(output).resolve()
    return output.parent/f'.{output.name}.pr223-full19-copied-preflight.json'


def source_requirements(root):
    """Full new helper closure, not the old inspection card or a three-knob preset."""
    root=Path(root)
    extras=[str(p.relative_to(root)) for directory in ('examples','reports') for p in (root/directory).rglob('*.py')]
    extras += [protocol.PROTOCOL,protocol.REFERENCE]
    catalog='reports/toy_audit/catalog.json'
    if (root/catalog).is_file(): extras.append(catalog)
    from experiments.forge.sources import inspect_source
    return inspect_source(root,extras)


def plan(root=ROOT):
    root=Path(root).resolve();card=protocol.load(root)
    proof=protocol.metadata_preflight(root,legacy,card)
    closure=source_requirements(root)
    definitions={};rows=[]
    for declared in card['rows']:
        d=deepcopy(proof['source_derived_definitions'][declared['id']])
        original_id=d['id'];d['id']=declared['id']
        d['fresh_retest']={'parent_original_id':original_id,
            'historical_case_sha256':declared['historical_case_sha256'],
            'source_derived_parent_sha256':stable_hash(proof['source_derived_definitions'][declared['id']]),
            'goal_media_steps':declared['media_steps'],'goal_observer_sha256':sha(root/DIRECTORY/'goal_observer.py'),
            'scope':'fresh full original19 noisy selected law; no current26/default/speed credit'}
        definitions[d['id']]=d
        cap=declared['proposed_inclusive_allowance_seconds']
        rows.append({'id':d['id'],'group':d['group'],'task':d['task'],'status':'NOT_RUN',
                     'case_sha256':stable_hash(d),'timeout_seconds':cap,'allowance_seconds':cap})
    return {'schema':'pg_pr223_original_full_retest_v1','spec':{
        'id':'pr223-original-full19-retest-20261004','representation_card':{'path':protocol.PROTOCOL,'sha256':sha(root/protocol.PROTOCOL)},
        'export_grace_seconds':0,'retries':0,'total_paid_cap_seconds':10800,
        'case_caps_sum_seconds':9810,'shared_metadata_and_finalization_seconds':180,
        'resources':{'host_memory_mb':2048},'diagnostic_independent_tests':True,
        'media_contract':'actual primary plus original reference;9existing clocks; moving4existing clocks'},
        'protocol':card,'source':{'commit':closure['origin_commit'],'files_sha256':closure['files']},
        'preflight':proof,'case_definitions':definitions,'external_inputs':proof['inputs'],
        'family':'atlas','recipe_overrides':{'original_config_sha256':legacy.CONFIG_SHA,'original_options':deepcopy(legacy.OPTIONS)},
        'rows':rows,'status':'DECLARED','required':19,'spent_seconds':0.,'new_paid_seconds':0.,
        'qualification_input':False,'default_adoption':False,'current_forge_mog_clean_qualification':False,
        'speed_ranking':False,'old_results_are_current_credit':False}


def validate_packet(packet,*,source=False):
    scope=scoped_module(packet)
    if scope: return scope.validate_packet(packet,sys.modules[__name__],source=source)
    card=protocol.validate(packet['protocol'])
    if (packet.get('schema')!='pg_pr223_original_full_retest_v1' or packet.get('family')!='atlas'
            or packet.get('required')!=19 or packet['spec'].get('export_grace_seconds')!=0
            or packet['spec'].get('retries')!=0 or packet['spec'].get('total_paid_cap_seconds')!=10800
            or packet['spec'].get('case_caps_sum_seconds')!=9810
            or packet['spec'].get('shared_metadata_and_finalization_seconds')!=180
            or packet['recipe_overrides']!={'original_config_sha256':legacy.CONFIG_SHA,'original_options':legacy.OPTIONS}):
        raise ValueError('fresh original full19 scope/config/caps changed')
    if len(packet['rows'])!=19 or [r['id'] for r in packet['rows']]!=[r['id'] for r in card['rows']]:
        raise ValueError('all19 exact ordered case slots required')
    if set(packet['case_definitions'])!={r['id'] for r in card['rows']}:
        raise ValueError('missing/extra original case definition')
    for row,declared in zip(packet['rows'],card['rows']):
        cap=declared['proposed_inclusive_allowance_seconds'];d=packet['case_definitions'][row['id']]
        if (row['group'],row['task'])!=(declared['group'],declared['task']) or row['timeout_seconds']!=cap or row['allowance_seconds']!=cap:
            raise ValueError('original task/whole inclusive cap changed')
        if stable_hash(d)!=row['case_sha256']: raise ValueError('requested case fingerprint changed')
        parent=deepcopy(d);retest=parent.pop('fresh_retest');parent['id']=retest['parent_original_id']
        historical=deepcopy(declared['original_definition']);historical['external_inputs_sha256']=parent['external_inputs_sha256']
        if parent!=historical or stable_hash(parent)!=retest['source_derived_parent_sha256']:
            raise ValueError('original law/model/gate/seed/horizon/cadence changed')
        if (retest['historical_case_sha256']!=declared['historical_case_sha256']
                or retest['goal_media_steps']!=declared['media_steps']):
            raise ValueError('old source or new retained goal clock differs')
    if source:
        execution=packet['execution_source'];snapshot=Path(execution['snapshot_path']).resolve()
        verify_snapshot(snapshot,execution)
        expected={k:v for k,v in execution.items() if k!='snapshot_path'}
        if read_json(snapshot/'forge-source.json')!=expected:
            raise ValueError('immutable source receipt metadata changed')
        if packet['source']['execution_digest']!=execution['digest'] or packet['source']['commit']!=execution['origin_commit']:
            raise ValueError('execution source/origin differs')
        if any(execution['files'].get(p)!=h for p,h in packet['source']['files_sha256'].items()):
            raise ValueError('prepared source not the complete planned closure')
        if sha(snapshot/protocol.PROTOCOL)!=packet['spec']['representation_card']['sha256'] or protocol.load(snapshot)!=card:
            raise ValueError('prepared protocol/config copy changed')
        if packet['snapshot_locations']['package']!=str(snapshot):
            raise ValueError('prepared current public package differs')
        output=Path(packet['prepared_output']).resolve()
        if (packet['metadata_ledger_path']!=str(metadata_path(output))
                or packet['copied_preflight_receipt_path']!=str(copied_preflight_path(output))):
            raise ValueError('durable metadata/preflight identity changed')
        for declared in card['rows']:
            if packet['case_definitions'][declared['id']]['fresh_retest']['goal_observer_sha256']!=execution['files'].get(DIRECTORY+'/goal_observer.py'):
                raise ValueError('new observer is not source-bound')
    return packet


def prepare(output,*,root=ROOT,queue_root=None,ledger=None,native3_anchor=None):
    """Root-only source preparation. It runs no models, samplers or scoring."""
    output=Path(output).resolve();root=Path(root).resolve()
    scope=native3_module() if native3_anchor is not None else None
    metadata=scope.sidecar(output) if scope else sidecar(output)
    # Invocations are charged to one durable180-second parent allowance, not
    # granted new export/preparation grace on each resume.
    ledger=ledger or ledger_module().SharedMetadataLedger(scope.require_existing_ledger() if scope else metadata_path(output))
    if scope and Path(ledger.path).resolve()!=Path(scope.CANONICAL_LEDGER):
        raise ValueError('native3 preparation must use the exact existing parent metadata ledger')
    with ledger.phase('original_source_preflight_and_snapshot'):
        if metadata.exists():
            saved=read_json(metadata);validate_packet(saved,source=True)
            if bool(scoped_module(saved))!=bool(scope): raise ValueError('prepared execution scope changed')
            if scope:
                if saved['metadata_history']['closed_anchor']!=native3_anchor: raise ValueError('closed parent input substituted')
                scope.validate_live_history(saved,ledger.snapshot(),sys.modules[__name__])
            return saved
        if output.exists() and any(output.iterdir()): raise ValueError('fresh nonempty output cannot be prepared')
        queue_root=queue_location(root,queue_root)
        packet=scope.plan(root,sys.modules[__name__],native3_anchor) if scope else plan(root)
        if scope: scope.validate_live_history(packet,ledger.snapshot(),sys.modules[__name__])
        expected=deepcopy(packet['source'])
        expected['files_sha256'].pop(protocol.REFERENCE)
        expected['files_sha256'].pop(protocol.PROTOCOL)
        if scope:
            for name in [*scope.HISTORY_PINS,*scope.PORTABLE_CONTROL_FILES]: expected['files_sha256'].pop(name)
        base=freeze_source(root,queue_root,expected)
        with tempfile.TemporaryDirectory(prefix='pr223-full19-source-',dir=queue_root) as tmp:
            staging=Path(tmp);files=dict(base['files'])
            for name in files:
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(Path(base['snapshot_path'])/name,target)
            for name in (protocol.REFERENCE,protocol.PROTOCOL):
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/name,target);files[name]=sha(root/name)
            if scope:
                for name in [*scope.HISTORY_PINS,*scope.PORTABLE_CONTROL_FILES]:
                    wanted=packet['source']['files_sha256'][name]
                    if sha(root/name)!=wanted: raise ValueError('committed carry source changed')
                    target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(root/name,target);files[name]=wanted
                anchor=packet['metadata_history']['closed_anchor']
                scope.validate_anchor(anchor,sys.modules[__name__])
                target=staging/scope.ANCHOR_RELATIVE;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(anchor['path'],target);files[scope.ANCHOR_RELATIVE]=anchor['sha256']
            for filename,wanted in packet['external_inputs']['files'].items():
                if sha(filename)!=wanted: raise ValueError('external original source/data changed during freeze')
                relative=legacy._external_relative(filename,packet).as_posix()
                target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(filename,target);files[relative]=wanted
            manifest={'schema_version':1,'digest':stable_hash(files),'files':files,'origin_commit':packet['source']['commit']}
            namespace='policy/pr223-native3-continuation' if scope else 'policy/pr223-full19-retest'
            snapshot=snapshot_source(staging,queue_root/namespace,manifest)
        packet['execution_source']={**manifest,'snapshot_path':str(snapshot)}
        packet['source']['execution_digest']=manifest['digest']
        packet['snapshot_locations']={k:str(snapshot/'atlas19-external'/k) for k in ('harness','initializer','native_root','adapter')}
        packet['snapshot_locations'].update(rotate=str(snapshot/'atlas19-external/rotate_gate.py'),package=str(snapshot))
        packet['queue_root']=str(queue_root);packet['prepared_output']=str(output)
        packet['metadata_ledger_path']=scope.CANONICAL_LEDGER if scope else str(metadata_path(output))
        packet['copied_preflight_receipt_path']=str(scope.copied_preflight_path(output) if scope else copied_preflight_path(output))
        validate_packet(packet,source=True)
        atomic_json(metadata,packet)
    return packet


def guard_imports(source,*,modules=None,derived=None):
    """Check actual imported local bytes, including protected aliases/namespaces."""
    if modules is None:
        # Original moving/native scripts insert their own source roots. Remove
        # repeated aliases without changing first-occurrence import precedence.
        normalize_paths()
    snapshot=Path(source['snapshot_path']).resolve();derived=derived or {}
    for name,value in tuple((sys.modules if modules is None else modules).items()):
        path=getattr(value,'__file__',None)
        protected=(name.split('.',1)[0] in {'experiments','particlegan','benchmarks','lib','native100','current_api_fixtures'}
                   or name.startswith(('_pr223_','_frozen_ra11_initializer','lrfree_')))
        if path is None:
            locations=list(getattr(value,'__path__',[]))
            if protected:
                if len(locations)!=1: raise ValueError('ambiguous/missing protected namespace '+name)
                location=Path(locations[0]).resolve()
                if not location.is_relative_to(snapshot): raise ValueError('foreign protected namespace '+name)
                relative=location.relative_to(snapshot).as_posix()
                if not any(p.startswith(relative+'/') for p in source['files']): raise ValueError('unpinned namespace '+name)
            continue
        original_path=Path(path)
        if not protected and not original_path.is_absolute() and not original_path.is_file():
            # torch.ops / torch.classes expose source-free pseudo __file__
            # strings. This exemption never applies to a protected namespace.
            continue
        path=original_path.resolve()
        if path in derived:
            if sha(path)!=derived[path]: raise ValueError('derived imported wrapper changed '+name)
            continue
        if path.is_relative_to(snapshot):
            relative=path.relative_to(snapshot).as_posix()
            if source['files'].get(relative)!=sha(path): raise ValueError('unpinned actual source import '+name)
        elif protected:
            raise ValueError('foreign protected import '+name)
        elif path.is_relative_to('/ml2/hypergan') and not path.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError('foreign workspace source import '+name)
        # Stdlib and installed third-party runtime modules retain their actual
        # runtime cohort. Pseudo-file Torch objects are not repository imports.
    for path,wanted in derived.items():
        if sha(path)!=wanted: raise ValueError('executed derived source mutated')


def native_scorer_import_metadata(source,native_root,*,paths):
    """Resolve the separate original scorer's imports without executing them.

    Its relocated native root owns the original scorer's package and hosts.
    The envelope root has already supplied the loaded, pinned control helpers;
    leaving that root searchable would merge two distinct PEP420 lib roots.
    Remove only that exact inherited root, retain every other distinct path,
    and refuse any foreign/ambiguous resolution before a host is imported.
    """
    snapshot=Path(source['snapshot_path']).resolve()
    native=Path(native_root).resolve()
    if native!=snapshot/'atlas19-external/native_root':
        raise ValueError('native scorer import root is not the pinned relocated owner')
    search=[];seen=set()
    for value in [str(native),*paths]:
        path=str(Path(value or os.getcwd()).resolve())
        if path==str(snapshot) or path in seen: continue
        search.append(path);seen.add(path)
    resolved={}
    specs={}
    for name in ('lib','lib.toy_models','particlegan','benchmarks',
                 'benchmarks.toy100','benchmarks.toy100.gate',
                 'benchmarks.toy100.accuracy_gate','benchmarks.toy100.train'):
        parent=name.rpartition('.')[0]
        scope=specs[parent].submodule_search_locations if parent else search
        spec=importlib.machinery.PathFinder.find_spec(name,scope)
        if spec is None: raise ValueError('missing native scorer import '+name)
        specs[name]=spec
        expected=native.joinpath(*name.split('.'))
        if name=='lib':
            locations=[str(Path(p).resolve()) for p in spec.submodule_search_locations or ()]
            if spec.origin is not None or spec.loader is not None or locations!=[str(expected)]:
                raise ValueError('ambiguous/foreign native scorer namespace lib')
            relative=expected.relative_to(snapshot).as_posix()
            if not any(p.startswith(relative+'/') for p in source['files']):
                raise ValueError('unpinned native scorer namespace lib')
            resolved[name]={'kind':'namespace','path':relative,'locations':1}
        else:
            expected=expected/'__init__.py' if spec.submodule_search_locations is not None else expected.with_suffix('.py')
            if spec.origin is None or Path(spec.origin).resolve()!=expected:
                raise ValueError('foreign native scorer import '+name)
            relative=expected.relative_to(snapshot).as_posix()
            wanted=source['files'].get(relative)
            if wanted is None or sha(expected)!=wanted:
                raise ValueError('unpinned/changed native scorer import '+name)
            if spec.submodule_search_locations is not None and [str(Path(p).resolve()) for p in spec.submodule_search_locations]!=[str(expected.parent)]:
                raise ValueError('ambiguous native scorer package '+name)
            resolved[name]={'kind':'source','path':relative,'sha256':wanted}
    proof={'status':'PASS_NATIVE_SCORER_IMPORT_METADATA_ONLY',
           'source_digest':source['digest'],'resolved_modules':resolved,
           'executed_modules':0,'models':0,'sampler_calls':0,'scorer_calls':0}
    return search,proof


def normalize_native_scorer_paths(source,native_root):
    """Bind the original scorer's isolated import owner; never relax the guard."""
    if any(name.split('.',1)[0] in {'lib','benchmarks','particlegan'} for name in sys.modules):
        raise ValueError('native scorer requires fresh original package imports')
    search,proof=native_scorer_import_metadata(source,native_root,paths=sys.path)
    sys.path[:]=search
    return proof


def verify_lease(request,fd,*,now=None):
    """Inherited exact descriptor + durable token/source/deadline, never a flag."""
    now=time.monotonic() if now is None else now
    actual=Path(os.readlink(f'/proc/self/fd/{fd}')).resolve()
    worker=request['worker'];wanted=Path(worker['lease_path']).resolve()
    if fd!=worker['lease_fd'] or actual!=wanted or os.fstat(fd).st_ino!=wanted.stat().st_ino:
        raise ValueError('missing or substituted inherited attempt descriptor')
    durable=read_json(wanted.parent/'supervisor-request.json')
    if (durable['token']!=worker['token'] or durable['source']!=request['packet']['execution_source']
            or fd not in durable['lease_fds'] or len(set(durable['lease_fds']))!=2
            or durable['lease_fds']!=worker['lease_fds']
            or durable['command']!=request['command']
            or durable['deadline_monotonic']!=worker['deadline_monotonic']
            or durable['started_monotonic']!=worker['started_monotonic']
            or worker['deadline_monotonic']-worker['started_monotonic']!=request['row']['allowance_seconds']):
        raise ValueError('durable source/command/two-lease/deadline binding changed')
    study=Path(request['packet']['queue_root'])/'policy/studies'/f"{request['packet']['coordinator']['study_key']}.lock"
    for inherited in durable['lease_fds']:
        identity=os.fstat(inherited)
        destination=Path(os.readlink(f'/proc/self/fd/{inherited}')).resolve()
        expected=wanted if inherited==fd else study.resolve()
        if destination!=expected or identity.st_ino!=expected.stat().st_ino:
            raise ValueError('both exact original study/attempt descriptors are required')
    if now>=durable['deadline_monotonic']: raise TimeoutError('inclusive original case deadline exhausted')
    return durable


def validate_request(request):
    packet=validate_packet(request['packet'],source=True)
    rows=[r for r in packet['rows'] if r['id']==request['row'].get('id')]
    if len(rows)!=1 or rows[0]!=request['row']:
        raise ValueError('child case substituted after reservation')
    row=rows[0]
    scope=scoped_module(packet)
    if scope: scope.validate_current_request(request,sys.modules[__name__])
    if row.get('status') not in {'NOT_RUN','RUNNING'}: raise ValueError('no unchanged scientific retry')
    target=Path(request['target']).resolve()
    canonical=Path(packet['coordinator']['canonical_output']).resolve()
    if target!=canonical/row['group']/row['task']: raise ValueError('wrong scientific output identity')
    if request['worker']['physical_gpu']!=1 or request['worker']['device']!='cuda:0': raise ValueError('GPU placement changed')
    expected=[sys.executable,'-u','-B',str(Path(packet['execution_source']['snapshot_path'])/SELF),
              '--child',str(target/'request.json'),'--lease-fd',str(request['worker']['lease_fd'])]
    if request['command']!=expected: raise ValueError('admitted child command changed')
    return packet,row,target


def make_request(packet,row,target,command,worker):
    """Copy current admission metadata only after its native3 state is saved."""
    request={'packet':deepcopy(packet),'row':deepcopy(row),'target':str(target),
             'command':list(command),'worker':deepcopy(worker)}
    scope=scoped_module(packet)
    if scope: scope.validate_current_request(request,sys.modules[__name__])
    return request


def _load_source(name,path,source):
    result=types.ModuleType(name);result.__file__=str(path);sys.modules[name]=result
    result.__compiled_source_sha256__=hashlib.sha256(source.encode()).hexdigest()
    exec(compile(source,str(path),'exec'),result.__dict__)
    return result


def build_wrappers(packet,row,target,worker):
    """Derive/compile exact wrappers using source bytes only, without imports."""
    target=Path(target);loc=packet['snapshot_locations'];obs=observer();entries=[]
    def retain(label,path,source,original,operations):
        compile(source,str(path),'exec')
        output=target/(label+'.py')
        entries.append({'label':label,'original_path':str(path),'original_file_sha256':sha(path),
                        'base_compiled_sha256':hashlib.sha256(original.encode()).hexdigest(),
                        'executed_path':str(output),'executed_sha256':hashlib.sha256(source.encode()).hexdigest(),
                        'operations':operations,'source':source})
        return output
    if row['group']=='moving':
        base=legacy.moving_source(packet,target);source,proof=obs.overlay(base,'moving')
        retain('moving-observed-wrapper',Path(loc['rotate']),source,base,proof)
        return entries
    fp=Path(loc['adapter'])/'current_api_fixtures.py';original=fp.read_text()
    fs=legacy._replace(original,f'REFERENCE = Path({str(legacy.INITIALIZER)!r})',f'REFERENCE = Path({loc["initializer"]!r})')
    retain('relocated-current-api-fixtures',fp,fs,original,{'kind':'initializer path only'})
    sp=Path(loc['adapter'])/'screen_current.py';base=sp.read_text()
    base=legacy._replace(base,f'HARNESS = Path({str(legacy.HARNESS)!r})',f'HARNESS = Path({loc["harness"]!r})')
    scorer=Path(loc['harness'])/'native100_score.py'
    score_source=legacy._replace(scorer.read_text(),"ROOT = Path(FIXTURE['frozen_repo'])",f'ROOT = Path({loc["native_root"]!r})')
    score_source=legacy._replace(score_source,
        "FIXTURE = json.loads((Path(__file__).resolve().parent / 'tasks' / 'native100_fixture.json').read_text())",
        f"FIXTURE = json.loads(Path({str(Path(loc['harness'])/'tasks/native100_fixture.json')!r}).read_text())")
    score_source=legacy._replace(score_source,'    sys.path.insert(0, str(ROOT))',
        '    sys.path.insert(0, str(ROOT))\n    _r.normalize_native_scorer_paths(_q[\'packet\'][\'execution_source\'],ROOT)')
    score_source=legacy._replace(score_source,'    coverage = gate.score_run(run_dir, problem)',
                                '    _retest_check()\n    coverage = gate.score_run(run_dir, problem)')
    score_source=legacy._replace(score_source,'    print(json.dumps(dict(coverage=coverage, accuracy=accuracy, sources=checked), default=str))',
                                '    _retest_check()\n    print(json.dumps(dict(coverage=coverage, accuracy=accuracy, sources=checked), default=str))')
    prefix=("import importlib.util as _u\nfrom pathlib import Path as _P\n"
        f"_s=_u.spec_from_file_location('_retest_child_checks',{str(ROOT/SELF)!r})\n"
        "_r=_u.module_from_spec(_s);_s.loader.exec_module(_r)\n"
        f"_q=_r.read_json({str(target/'request.json')!r})\n"
        f"_derivative=_r.read_json({str(target/'executed-source-overlays.json')!r})\n"
        "_pins={_P(x['executed_path']).resolve():x['executed_sha256'] for x in _derivative['records']}\n"
        f"def _retest_check():\n    _r.verify_lease(_q,{worker['lease_fd']})\n    _r.guard_imports(_q['packet']['execution_source'],derived=_pins)\n")
    score_script=retain('native-score-observed-wrapper',scorer,prefix+score_source,scorer.read_text(),
                         {'kind':'same original scorer calls; isolated relocated root + imported-source/deadline checks'})
    # The unchanged native scorer inherits both owners and remains in its
    # leader's fenced process group. No new score call or sibling session.
    base=legacy._replace(base,"str(HARNESS / 'native100_score.py')",repr(str(score_script)))
    base=legacy._replace(base,"capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')",
                         f"capture_output=True, text=True, pass_fds={tuple(worker['lease_fds'])!r}, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')")
    if row['task'].startswith('vector_'):
        vp=Path(loc['harness'])/'hosts/vector_host.py';vs,vproof=obs.overlay(vp.read_text(),'vector')
        vscript=retain('vector-host-observed-wrapper',vp,vs,vp.read_text(),vproof)
        base=legacy._replace(base,"host = load_module('lrfree_vector_host', HOSTS / 'vector_host.py')",
                             f"host = load_module('lrfree_vector_host', Path({str(vscript)!r}))")
    source,proof=obs.overlay(base,'screen')
    retain('screen-observed-wrapper',sp,source,base,proof)
    return entries


def science(request,obs,recorder,check):
    """Exact old construction/updates/scorers, plus explicit reversible captures."""
    packet,row,target=validate_request(request);loc=packet['snapshot_locations']
    for name in ('ABSENT','ABSENT_START','ABSENT_END','LRFREE_NATIVE_TEST_STEPS'): os.environ.pop(name,None)
    normalize_paths([ROOT,loc['adapter'],loc['harness']],paths=[
        p for p in sys.path if p and not str(Path(p).resolve()).startswith('/ml2/hypergan/ParticleGAN')])
    if any(n=='particlegan' or n.startswith('particlegan.') for n in sys.modules):
        raise ValueError('fresh scientific package process required')
    entries=build_wrappers(packet,row,target,request['worker']);derived={}
    for entry in entries:
        path=Path(entry['executed_path']);path.write_text(entry['source'])
        if sha(path)!=entry['executed_sha256']: raise ValueError('derived source changed before execution')
        derived[path.resolve()]=entry['executed_sha256']
    write_proof(target,[{k:v for k,v in e.items() if k!='source'} for e in entries])
    check(derived)
    if row['group']=='moving':
        entry=entries[-1];script=Path(entry['executed_path']);source=entry['source']
        sys.argv=[str(script),row['task'],str(target/'frames.npz'),'--every','500','--points','4096',
                  '--steps','1500','--gate','--rotate-every','500','--rotate-deg','30']
        exec(compile(source,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
    else:
        import torch
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
        fixture=entries[0]
        _load_source('current_api_fixtures',Path(fixture['executed_path']),fixture['source'])
        screen=entries[-1];script=Path(screen['executed_path'])
        child_module=_load_source('_pr223_scientific_screen',script,screen['source'])
        sys.argv=[str(script),'--package-root',loc['package'],'--overrides',str(Path(loc['package'])/protocol.CONFIG),
                  '--task',row['task'],'--output',str(target),'--device','cuda:0',
                  '--candidate-options',json.dumps(protocol.OPTIONS),'--cand','pr223-original-full19-retest-20261004']
        child_module.main()
    check(derived)
    return derived


def write_proof(target,records):
    atomic_json(Path(target)/'executed-source-overlays.json',{'schema':'pr223_original_readonly_source_overlays_v1',
                'records':records,'algorithm_changes':False,'new_draws':0,'new_forwards':0,'new_scores':0,'new_updates':0})


def artifacts(target):
    target=Path(target)
    found={}
    for path in sorted(target.rglob('*')):
        if path.is_symlink(): raise ValueError('scientific artifact symlink refused')
        if path.is_file() and path.name not in {'case-attestation.json','run.log'}:
            found[path.relative_to(target).as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size}
    return found


def child(path,lease_fd):
    request=read_json(path);packet,row,target=validate_request(request)
    if any(os.environ.get(k)!=v for k,v in protocol.ENVIRONMENT.items()): raise ValueError('exact declared environment required')
    derived={}
    def check(extra=None):
        if extra is not None: derived.update(extra)
        verify_lease(request,lease_fd)
        guard_imports(packet['execution_source'],derived=derived)
    obs=observer();case=next(c for c in packet['protocol']['rows'] if c['id']==row['id'])
    recorder=obs.Recorder(target,case,source_guard=check,deadline_guard=lambda:verify_lease(request,lease_fd))
    obs.configure(recorder);check()
    scientific_returncode=0
    try:
        science(request,obs,recorder,check)
        check()
        grade=legacy.certify(packet,row,target,scientific_returncode)
        if grade.get('status') not in {'PASS','FAIL'} or grade.get('full_protocol_complete') is not True:
            raise ValueError('original numeric verdict/full protocol unavailable: '+str(grade.get('reason')))
        if row['group']!='moving' and read_json(target/'result.json').get('recipe')!=case['resolved_recipe']:
            raise ValueError('actual full original resolved Recipe differs')
        recorder.finish()
        media=obs.render(target,case,grade['status'],guard_callback=check)
        check();validate_packet(packet,source=True);check()
        grade['artifacts']=artifacts(target)
        scope=scoped_module(packet)
        result={'schema':scope.ATTESTATION_SCHEMA if scope else 'pr223_full19_case_attestation_v1','status':'COMPLETE','original_gate':grade['status'],
                'case_id':row['id'],'case_sha256':row['case_sha256'],'source_digest':packet['execution_source']['digest'],
                'protocol_sha256':stable_hash(packet['protocol']),'request_sha256':sha(path),
                'attempt_token':request['worker']['token'],'completed_before_deadline':True,
                'started_monotonic':request['worker']['started_monotonic'],
                'attested_monotonic':time.monotonic(),'deadline_monotonic':request['worker']['deadline_monotonic'],
                'scientific_returncode':scientific_returncode,'expected_child_returncode':0 if grade['status']=='PASS' else 1,
                'grade':grade,'goal_media':media,'complete_recipe':case['resolved_recipe'],
                'fresh_full_original19_only':not bool(scope),'qualification_input':False,'speed_ranking':False}
        if scope: result.update(execution_scope=scope.execution_scope(),full_original19_credit=False)
        atomic_json(target/'case-attestation.json',result);check()
        return result['expected_child_returncode']
    except BaseException as error:
        # A cap may kill the process before this write. Missing attestation then
        # remains incomplete; original ERROR/FAIL data never gets relabelled.
        try:
            atomic_json(target/'case-error.json',{'schema':'pr223_full19_case_error_v1','status':'INVALID',
                        'case_id':row['id'],'error':f'{type(error).__name__}: {error}',
                        'new_original_grade':None,'no_retry':True})
        except OSError: pass
        raise


def verify_attestation(packet,row,target,terminal):
    """Pure retained-byte validation. No regrade, checkpoint load or model call."""
    target=Path(target);attestation=read_json(target/'case-attestation.json')
    paid=terminal.get('paid_wall_seconds')
    if type(paid) not in (int,float) or not math.isfinite(paid) or paid<=0:
        raise ValueError('durable completed terminal paid time must be finite and positive')
    for key in ('started_monotonic','attested_monotonic','deadline_monotonic'):
        if type(attestation.get(key)) not in (int,float) or not math.isfinite(attestation[key]):
            raise ValueError('nonfinite final attestation time')
    if paid+1e-6<attestation['attested_monotonic']-attestation['started_monotonic']:
        raise ValueError('durable paid time does not cover final attestation')
    scope=scoped_module(packet)
    if scope and paid>row['allowance_seconds']:
        raise ValueError('native3 supervisor paid time exceeded the whole inclusive case cap')
    if scope and (attestation.get('execution_scope')!=scope.execution_scope()
                  or attestation.get('full_original19_credit') is not False
                  or attestation.get('fresh_full_original19_only') is not False):
        raise ValueError('native3 attestation cannot borrow full19 credit')
    if (terminal.get('attempt_status')!='completed' or terminal.get('token')!=row['attempt_token']
            or attestation.get('schema')!=(scope.ATTESTATION_SCHEMA if scope else 'pr223_full19_case_attestation_v1')
            or attestation.get('status')!='COMPLETE' or attestation['attempt_token']!=terminal['token']
            or attestation['case_id']!=row['id'] or attestation['case_sha256']!=row['case_sha256']
            or attestation['source_digest']!=packet['execution_source']['digest']
            or attestation['protocol_sha256']!=stable_hash(packet['protocol'])
            or attestation['request_sha256']!=sha(target/'request.json')
            or attestation['completed_before_deadline'] is not True
            or not attestation['started_monotonic']<=attestation['attested_monotonic']<attestation['deadline_monotonic']
            or attestation['scientific_returncode']!=0):
        raise ValueError('missing/late/substituted full case attestation')
    declared=next(c for c in packet['protocol']['rows'] if c['id']==row['id'])
    grade=attestation['grade'];status=attestation['original_gate']
    if scope: scope.validate_grade_metadata(grade)
    if (status not in {'PASS','FAIL'} or grade['status']!=status or grade['original_gate']!=status
            or grade.get('full_protocol_complete') is not True
            or grade.get('completed_steps')!=declared['original_definition']['original_host']['steps']
            or grade.get('metric_observations')!=len(declared['original_definition']['observation_steps'])
            or attestation['complete_recipe']!=declared['resolved_recipe']
            or attestation['expected_child_returncode']!=(0 if status=='PASS' else 1)
            or terminal['child_returncode']!=attestation['expected_child_returncode']):
        raise ValueError('child exit/original numeric/full protocol/Recipe disagree')
    expected_result=target/('frames.npz.verdict.json' if row['group']=='moving' else 'result.json')
    if Path(grade['result_path']).resolve()!=expected_result.resolve():
        raise ValueError('original result is outside the exact admitted case')
    if grade['artifacts']!=artifacts(target):
        raise ValueError('original full artifact mapping changed')
    for name,pin in grade['artifacts'].items():
        path=(target/name).resolve()
        if not path.is_relative_to(target.resolve()) or not path.is_file() or sha(path)!=pin['sha256'] or path.stat().st_size!=pin['bytes']:
            raise ValueError('original retained artifact changed: '+name)
    media=attestation['goal_media']
    if media['actual_steps']!=declared['media_steps'] or media['frames']!=len(declared['media_steps']) or media['metric_only'] is not False:
        raise ValueError('goal media is not actual original observation state')
    if media['sha256']!=sha(target/media['file']) or media['bytes']!=(target/media['file']).stat().st_size:
        raise ValueError('accepted goal GIF bytes changed')
    if grade['result_sha256']!=sha(grade['result_path']): raise ValueError('original result changed')
    return attestation


def retained_result(row,target):
    """Preserve reported raw status even if execution attestation is missing."""
    target=Path(target)
    path=target/('frames.npz.verdict.json' if row['group']=='moving' else 'result.json')
    if not path.is_file(): return {'available':False,'accepted_numeric_credit':False}
    before=sha(path);raw=read_json(path)
    if sha(path)!=before: raise ValueError('retained raw result changed during publication')
    return {'available':True,'path':str(path),'sha256':before,'bytes':path.stat().st_size,
            'reported_status':raw.get('status'),'reported_completed_steps':raw.get('completed_steps'),
            'accepted_numeric_credit':False}


def recertify_row(packet,row,target,terminal):
    """Resume validates recorded costs against their exact durable terminal."""
    proof=verify_attestation(packet,row,target,terminal)
    expected=ledger_module().case_cost(row['allowance_seconds'],terminal,certified=True)
    if any(row.get(key)!=value for key,value in expected.items()):
        raise ValueError('saved original case cost contradicts its durable terminal')
    return proof


def runtime_metadata():
    """Placement/runtime inspection without initializing a CUDA context."""
    row=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=name,driver_version,compute_cap',
                                  '--format=csv,noheader,nounits'],text=True).strip().split(',')
    if len(row)!=3: raise ValueError('one physical GPU1 runtime required')
    return {'python':platform.python_version(),'torch':importlib.metadata.version('torch'),
            'numpy':importlib.metadata.version('numpy'),'device':'cuda:0','physical_gpu':1,
            'cuda_device_model':row[0].strip(),'driver':row[1].strip(),'compute_capability':row[2].strip(),
            'torch_threads':1,'memory_fraction':.2,'deterministic':True,'tf32':False}


def _save(path,packet,ledger):
    packet['metadata_cost']=ledger.snapshot()
    accounting=reservation(packet,packet['metadata_cost'],0)
    packet['budget_accounting']=accounting
    packet['spent_seconds']=accounting['charged_seconds']
    packet['new_paid_seconds']=accounting.get('current_case_paid_wall_seconds',accounting.get('case_paid_wall_seconds'))
    packet['completed']=sum(r.get('full_protocol_complete') is True for r in packet['rows'])
    packet['media_completed']=sum(bool(r.get('media')) for r in packet['rows'])
    terminal=all(r['status']!='NOT_RUN' for r in packet['rows'])
    packet['scientific_status']=('PASS' if terminal and all(r['status']=='PASS' for r in packet['rows'])
                      else 'FAIL' if terminal and all(r['status'] in {'PASS','FAIL'} for r in packet['rows'])
                      else 'INCOMPLETE' if terminal else 'READY')
    overruns=[r['id'] for r in packet['rows'] if r.get('overrun_seconds',0)>0]
    if packet['metadata_cost']['charged_seconds']>180: overruns.append('shared_metadata')
    if packet['spent_seconds']>10800: overruns.append('aggregate')
    if accounting['halt_required'] and not overruns: overruns.append('metadata_interrupted')
    packet['budget_status']='EXCEEDED_OR_INTERRUPTED' if overruns else 'WITHIN_DECLARED_CAPS'
    packet['budget_overruns']=overruns
    packet['status']='INCOMPLETE' if overruns else packet['scientific_status']
    packet['required_evidence_complete']=packet['completed']==packet['required'] and packet['media_completed']==packet['required']
    packet['accepted_retest_complete']=packet['required_evidence_complete'] and not overruns
    atomic_json(path,packet)


def run(output,*,root=ROOT,queue_root=None,max_new_attempts=None,native3_anchor=None):
    output=Path(output).resolve();scope=native3_module() if native3_anchor is not None else None
    ledger=ledger_module().SharedMetadataLedger(scope.require_existing_ledger() if scope else metadata_path(output))
    prepared=prepare(output,root=root,queue_root=queue_root,ledger=ledger,native3_anchor=native3_anchor)
    with ledger.phase('registration_parent_validation_and_finalization'):
        validate_packet(prepared,source=True)
        if scope: scope.validate_live_history(prepared,ledger.snapshot(),sys.modules[__name__])
        require_copied_preflight(prepared)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('physical GPU1 numeric placement required')
        prepared['lane_runtime']=runtime_metadata()
        coordinator=PolicyCoordinator(prepared['queue_root'],report_root=Path(root)/'reports/forge')
        key,canonical=coordinator.register(prepared,output,'atlas',prepared['lane_runtime'])
        if canonical.resolve()!=output.resolve():
            raise ValueError('use the original canonical output and retained metadata budget; no alias budget reset')
        launched=0
        with coordinator.study_lease(key) as study_lease:
            if study_lease is None: return coordinator.publish_attachment(key,canonical,output)
            packet=read_json(canonical/'study.json');validate_packet(packet,source=True)
            coordinator.recover()
            for row in packet['rows']:
                if row['status'] in {'PASS','FAIL'}:
                    terminal=read_json(row['terminal_path'])
                    recertify_row(packet,row,canonical/row['group']/row['task'],terminal)
                    continue
                if row['status'] not in {'NOT_RUN','RUNNING'}:
                    packet['waiting_reason']='preserved incomplete/invalid first attempt; no automatic retry';break
                trial={'family':'atlas','recipe_overrides':packet['recipe_overrides']}
                attempt_key=coordinator.attempt_key(packet,trial,row);row['attempt_key']=attempt_key
                retained=coordinator.retained(attempt_key)
                reservation(packet,ledger.snapshot(),0 if retained else row['allowance_seconds'])
                if not retained:
                    if max_new_attempts is not None and launched>=max_new_attempts: break
                    readiness=legacy.gpu_readiness()
                    if not readiness['ready']:
                        packet['waiting_reason']=readiness['reason'];break
                with coordinator.admit(attempt_key,packet,row,'cuda:0') as (admission,lease):
                    if admission['status']=='busy': packet['waiting_reason']=admission['reason'];break
                    target=canonical/row['group']/row['task']
                    terminal_path=Path(admission['lease_path']).parent/'supervisor-terminal.json'
                    if admission['status'] in {'completed','awaiting_certification'}:
                        terminal=read_json(terminal_path)
                        row['attempt_token']=admission['token']
                        try:
                            proof=verify_attestation(packet,row,target,terminal)
                            result={**proof['grade'],'media':proof['goal_media']}
                        except (OSError,ValueError,KeyError) as error:
                            result={'status':'INVALID','reason':'retained attestation unavailable: '+str(error),'full_protocol_complete':False}
                        charged_terminal=terminal if terminal.get('token')==admission['token'] else None
                        cost=ledger_module().case_cost(row['allowance_seconds'],charged_terminal,certified=result['status'] in {'PASS','FAIL'})
                        result.update(cost,terminal_path=str(terminal_path),attempt_token=admission['token'],reused_physical_attempt=True)
                        if admission['status']=='awaiting_certification': coordinator.complete(attempt_key,result)
                        row.update(result)
                    elif admission['status']=='interrupted':
                        terminal=read_json(terminal_path) if terminal_path.exists() else None
                        charged_terminal=terminal if terminal and terminal.get('token')==admission.get('token') else None
                        row.update(status='INCOMPLETE',reason=admission['reason'],full_protocol_complete=False,
                                   **ledger_module().case_cost(row['allowance_seconds'],charged_terminal,certified=False))
                    elif admission['status']=='running':
                        if target.exists():
                            result={'status':'INVALID','reason':'orphan case retained; no retry','full_protocol_complete':False,
                                    **ledger_module().case_cost(row['allowance_seconds'],None,certified=False)}
                        else:
                            target.mkdir(parents=True);request_path=target/'request.json'
                            command=[sys.executable,'-u','-B',str(Path(packet['execution_source']['snapshot_path'])/SELF),
                                     '--child',str(request_path),'--lease-fd',str(lease.fileno())]
                            row.update(status='RUNNING',attempt_token=admission['token'])
                            # DECLARED remains a pristine preparation contract.
                            # Persist native3's current admission before copying
                            # the request, so its RUNNING row has current status.
                            if scoped_module(packet): _save(canonical/'study.json',packet,ledger)
                            request=make_request(packet,row,target,command,
                                {'lease_fd':lease.fileno(),'lease_fds':[study_lease.fileno(),lease.fileno()],
                                    'lease_path':admission['lease_path'],'token':admission['token'],
                                    'started_monotonic':admission['started_monotonic'],
                                    'deadline_monotonic':admission['deadline_monotonic'],'physical_gpu':1,'device':'cuda:0'})
                            atomic_json(request_path,request)
                            if not scoped_module(packet): _save(canonical/'study.json',packet,ledger)
                            print(json.dumps({'event':'start','case':row['id'],'attempt_key':attempt_key,'log':str(target/'run.log')}),flush=True)
                            try:
                                # The durable case supervisor owns all child
                                # phases. Parent metadata time is suspended
                                # only for this already-paid wait, never media.
                                with ledger.pause():
                                    done=coordinator.launch(command,packet,target/'run.log',(study_lease,lease),row['allowance_seconds'])
                                terminal=read_json(terminal_path)
                                proof=verify_attestation(packet,row,target,terminal)
                                result={**proof['grade'],'media':proof['goal_media']}
                            except Exception as error:
                                terminal=read_json(terminal_path) if terminal_path.exists() else None
                                result={'status':'INCOMPLETE' if isinstance(error,subprocess.TimeoutExpired) else 'INVALID',
                                        'reason':f'{type(error).__name__}: {error}','full_protocol_complete':False}
                            charged_terminal=terminal if terminal and terminal.get('token')==admission['token'] else None
                            cost=ledger_module().case_cost(row['allowance_seconds'],charged_terminal,certified=result['status'] in {'PASS','FAIL'})
                            result.update(cost,terminal_path=str(terminal_path),attempt_token=admission['token'])
                            launched+=1
                        coordinator.complete(attempt_key,result);row.update(result)
                    else: raise ValueError('unknown shared admission state')
                row['raw_result']=retained_result(row,target)
                _save(canonical/'study.json',packet,ledger)
                print(json.dumps({'event':'complete','case':row['id'],'status':row['status'],
                                  'completed':packet['completed'],'required':packet['required'],'charged_seconds':row['charged_seconds']}),flush=True)
                if row['status'] not in {'PASS','FAIL'} or row.get('overrun_seconds',0)>0:
                    packet['waiting_reason']='first invalid/interrupted/overrun retained; no retry';break
            _save(canonical/'study.json',packet,ledger)
        packet=coordinator.publish_attachment(key,canonical,output)
    # Phase completion records actual parent time. The final compact rewrite
    # is performed in another bounded phase; remaining180 never resets.
    with ledger.phase('final_metadata_receipt'):
        _save(canonical/'study.json',packet,ledger)
    return packet


def copied_preflight(prepared):
    """CPU-hidden copied-source metadata proof, without science or queue calls."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('preflight must hide CUDA')
    validate_packet(prepared,source=True)
    if ROOT!=Path(prepared['execution_source']['snapshot_path']).resolve():
        raise ValueError('execute preflight from the actual immutable copied helper')
    guard_imports(prepared['execution_source'])
    obs=observer();loc=prepared['snapshot_locations']
    for kind,path in [('screen',Path(loc['adapter'])/'screen_current.py'),
                      ('vector',Path(loc['harness'])/'hosts/vector_host.py')]:
        original=path.read_text();derived,proof=obs.overlay(original,kind)
        compile(derived,str(path),'exec')
        if obs.remove_overlay(derived,kind)!=original: raise ValueError('copied overlay not reversible')
    moving=legacy.moving_source(prepared,Path('/unused-metadata-only-output'))
    derived,proof=obs.overlay(moving,'moving');compile(derived,'moving-model-free-preflight','exec')
    if obs.remove_overlay(derived,'moving')!=moving: raise ValueError('moving copied overlay changed')
    for row in prepared['rows']:
        entries=build_wrappers(prepared,row,Path('/unused-metadata-only-output')/row['id'],
                               {'lease_fd':101,'lease_fds':[100,101]})
        if not entries: raise ValueError('missing original compiled child wrapper')
    _,native_imports=native_scorer_import_metadata(prepared['execution_source'],
        loc['native_root'],paths=[str(ROOT),*sys.path])
    scope=scoped_module(prepared)
    boundary=(module(scope.DIRECTORY+'/scorer_boundary_control.py','_pr223_native3_boundary').run(prepared,sys.modules[__name__])
              if scope else None)
    request_boundary=(module(scope.DIRECTORY+'/request_boundary_control.py','_pr223_native3_request_boundary').run(prepared,sys.modules[__name__])
                      if scope else None)
    guard_imports(prepared['execution_source'])
    if any(n=='torch' or n=='particlegan' or n.startswith('particlegan.') for n in sys.modules):
        raise ValueError('model-free preflight imported a model package')
    result={'status':'PASS_COPIED_METADATA_ONLY','source_digest':prepared['execution_source']['digest'],
            'protocol_sha256':stable_hash(prepared['protocol']),'cases':3 if scope else 19,'updates':21000 if scope else 48800,
            'compiled_original_wrappers':3 if scope else 19,'native_scorer_imports':native_imports,
            'models':0,'sampler_calls':0,'scorer_calls':0,'queue_calls':0,'numeric_credit':False}
    if scope:
        result['native_scorer_boundary_control']=boundary
        result['native_request_boundary_control']=request_boundary
    return result


def bounded_copied_preflight(prepared):
    """The actual copied-source proof spends the same retained180-second cap."""
    scope=scoped_module(prepared)
    if scope:
        if prepared.get('metadata_ledger_path')!=scope.CANONICAL_LEDGER:
            raise ValueError('copied native3 proof must use the SAME canonical parent ledger')
        scope.require_existing_ledger()
    ledger=ledger_module().SharedMetadataLedger(prepared['metadata_ledger_path'])
    with ledger.phase('copied_source_model_free_preflight'):
        if scope:
            validate_packet(prepared,source=True)
            scope.validate_live_history(prepared,ledger.snapshot(),sys.modules[__name__])
        result=copied_preflight(prepared)
        result.update(schema=scope.PREFLIGHT_SCHEMA if scope else 'pg_pr223_copied_source_preflight_v1',
                      prepared_packet_sha256=stable_hash(prepared),
                      snapshot_path=prepared['execution_source']['snapshot_path'],
                      origin_commit=prepared['execution_source']['origin_commit'],
                      helper_sha256=sha(ROOT/SELF))
        atomic_json(prepared['copied_preflight_receipt_path'],result)
    return {**result,'metadata_cost':ledger.snapshot()}


def require_copied_preflight(prepared):
    """Persisted exact copied-source proof is mandatory before registration."""
    path=Path(prepared['copied_preflight_receipt_path'])
    if not path.is_file(): raise ValueError('copied-source preflight receipt is missing')
    result=read_json(path)
    scope=scoped_module(prepared)
    if (result.get('schema')!=(scope.PREFLIGHT_SCHEMA if scope else 'pg_pr223_copied_source_preflight_v1')
            or result.get('status')!='PASS_COPIED_METADATA_ONLY'
            or result.get('prepared_packet_sha256')!=stable_hash(prepared)
            or result.get('source_digest')!=prepared['execution_source']['digest']
            or result.get('protocol_sha256')!=stable_hash(prepared['protocol'])
            or result.get('snapshot_path')!=prepared['execution_source']['snapshot_path']
            or result.get('origin_commit')!=prepared['execution_source']['origin_commit']
            or result.get('helper_sha256')!=prepared['execution_source']['files'].get(SELF)
            or result.get('cases')!=(3 if scope else 19) or result.get('updates')!=(21000 if scope else 48800)
            or result.get('compiled_original_wrappers')!=(3 if scope else 19)
            or any(result.get(k)!=0 for k in ('models','sampler_calls','scorer_calls','queue_calls'))
            or result.get('numeric_credit') is not False):
        raise ValueError('copied-source preflight receipt is stale, foreign or incomplete')
    # The copied helper's native import plan is checked without importing the
    # scorer or a model package. Missing/foreign proof cannot reach admission.
    _,native_imports=native_scorer_import_metadata(prepared['execution_source'],
        prepared['snapshot_locations']['native_root'],paths=[prepared['execution_source']['snapshot_path']])
    if result.get('native_scorer_imports')!=native_imports:
        raise ValueError('copied-source preflight native import proof is stale or missing')
    if scope:
        boundary=module(scope.DIRECTORY+'/scorer_boundary_control.py','_pr223_native3_boundary')
        boundary.validate_proof(result.get('native_scorer_boundary_control'),prepared,sys.modules[__name__])
        request_boundary=module(scope.DIRECTORY+'/request_boundary_control.py','_pr223_native3_request_boundary')
        request_boundary.validate_proof(result.get('native_request_boundary_control'),prepared,sys.modules[__name__])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--queue-root',type=Path)
    parser.add_argument('--metadata-preflight',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--copied-preflight',type=Path);parser.add_argument('--child',type=Path)
    parser.add_argument('--lease-fd',type=int);parser.add_argument('--max-new-attempts',type=int)
    args=parser.parse_args(argv)
    if args.child:
        if args.lease_fd is None: parser.error('--child requires inherited --lease-fd')
        return child(args.child,args.lease_fd)
    if args.copied_preflight:
        print(json.dumps(bounded_copied_preflight(read_json(args.copied_preflight)),sort_keys=True));return 0
    if args.metadata_preflight:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='': parser.error('metadata preflight must hide CUDA')
        result=protocol.metadata_preflight(ROOT,legacy)
        if any(n=='torch' or n=='particlegan' or n.startswith('particlegan.') for n in sys.modules):
            raise ValueError('metadata-only source preflight imported a model package')
        print(json.dumps({k:v for k,v in result.items() if k not in {'inputs','source_derived_definitions'}},sort_keys=True));return 0
    if args.output is None: parser.error('--output required')
    if args.max_new_attempts is not None and args.max_new_attempts<1: parser.error('positive dispatch limit required')
    if not args.prepare_only:
        for name,value in protocol.ENVIRONMENT.items():
            if name=='CUDA_VISIBLE_DEVICES' and os.environ.get(name) not in (None,'1'): parser.error('physical GPU1 numeric mask required')
            os.environ[name]=value
    packet=prepare(args.output,queue_root=args.queue_root) if args.prepare_only else run(args.output,queue_root=args.queue_root,max_new_attempts=args.max_new_attempts)
    print(json.dumps({'status':packet['status'],'required':19,'completed':packet.get('completed',0),
                      'paid_seconds':packet.get('new_paid_seconds'),'waiting_reason':packet.get('waiting_reason')}))
    return 0 if args.prepare_only or packet['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
