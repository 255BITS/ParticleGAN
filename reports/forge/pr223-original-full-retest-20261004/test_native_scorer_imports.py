"""Synthetic import controls only: no original scorer/model/evidence execution."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCORER_FIXTURE=HERE/'fixtures/native100_score.py.txt'
SCORER_FIXTURE_SHA256='10cc14edfcd98ab34fd3768aaba2ee835dc2241dc1face2e18998c8f2b687feb'


def scorer_fixture_bytes(path=SCORER_FIXTURE):
    data=Path(path).read_bytes()
    if len(data)!=1660 or hashlib.sha256(data).hexdigest()!=SCORER_FIXTURE_SHA256:
        raise ValueError('inert original scorer source fixture changed')
    return data


def test_original_scorer_source_fixture_is_exact_and_inert():
    data=scorer_fixture_bytes()
    assert SCORER_FIXTURE.suffix=='.txt'
    assert importlib.util.spec_from_file_location('_never_imported_original_scorer',SCORER_FIXTURE) is None
    assert data.count(b'coverage = gate.score_run(run_dir, problem)')==1
    assert data.count(b'accuracy = accuracy_gate.score_run(run_dir, problem, coverage)')==1


@pytest.mark.parametrize('change',['length','same_length_hash'])
def test_tampered_scorer_source_fixture_is_refused(tmp_path,change):
    data=scorer_fixture_bytes()
    changed=data+b'\n' if change=='length' else bytes([data[0]^1])+data[1:]
    path=tmp_path/'synthetic-tampered-source.py.txt';path.write_bytes(changed)
    with pytest.raises(ValueError,match='source fixture changed'): scorer_fixture_bytes(path)


@pytest.fixture(scope='module')
def helper():
    spec=importlib.util.spec_from_file_location('_pr223_native_import_tests',HERE/'run_retest.py')
    result=importlib.util.module_from_spec(spec);sys.modules[spec.name]=result
    spec.loader.exec_module(result)
    return result


def pin_source(root):
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.json'}}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'snapshot_path':str(root),'files':files,'digest':digest}


@pytest.fixture
def synthetic_native(tmp_path):
    root=tmp_path/'synthetic-snapshot'
    native=root/'atlas19-external/native_root'
    sources={
        'lib/toy_models.py':"raise AssertionError('envelope lib must not be imported')\n",
        'atlas19-external/native_root/lib/toy_models.py':"OWNER='synthetic-original-native'\n",
        'atlas19-external/native_root/particlegan/__init__.py':"# Synthetic metadata only, never imported.\n",
        'atlas19-external/native_root/benchmarks/__init__.py':"# Synthetic original benchmark owner.\n",
        'atlas19-external/native_root/benchmarks/toy100/__init__.py':"# Synthetic original host owner.\n",
        'atlas19-external/native_root/benchmarks/toy100/train.py':"from lib import toy_models\nOWNER=toy_models.OWNER\n",
        'atlas19-external/native_root/benchmarks/toy100/gate.py':
            "from . import train\ndef score_run(*args):\n    raise AssertionError('no scorer call allowed')\n",
        'atlas19-external/native_root/benchmarks/toy100/accuracy_gate.py':
            "from . import gate\ndef score_run(*args):\n    raise AssertionError('no scorer call allowed')\n",
    }
    for relative,text in sources.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    return root,native,pin_source(root)


def test_original_distinct_lib_collision_is_not_an_alias(helper,synthetic_native,monkeypatch):
    root,native,source=synthetic_native
    from importlib.machinery import PathFinder
    monkeypatch.setattr(sys,'path',list(sys.path))
    normalized=helper.normalize_paths(paths=[str(native),str(root),str(root)+os.sep+'.'])
    spec=PathFinder.find_spec('lib',normalized)
    assert list(spec.submodule_search_locations)==[str(native/'lib'),str(root/'lib')]
    # Actual guard, with synthetic import metadata; no original host executes.
    with pytest.raises(ValueError,match='ambiguous/missing protected namespace lib'):
        helper.guard_imports(source,modules={'lib':type('SyntheticNamespace',(),{
            '__file__':None,'__path__':spec.submodule_search_locations})()})


def test_isolated_metadata_resolution_preserves_original_host_and_foreign_paths(helper,synthetic_native,tmp_path):
    root,native,source=synthetic_native
    foreign=tmp_path/'different-but-empty';foreign.mkdir()
    before=list(sys.path)
    paths,proof=helper.native_scorer_import_metadata(source,native,
        paths=[str(root),str(root)+os.sep+'.',str(native),str(foreign)])
    assert sys.path==before  # Copied preflight must not change parent imports.
    assert paths==[str(native),str(foreign)]
    assert proof['resolved_modules']['lib']=={
        'kind':'namespace','path':'atlas19-external/native_root/lib','locations':1}
    assert len(proof['resolved_modules'])==8
    assert all(proof[k]==0 for k in ('executed_modules','models','sampler_calls','scorer_calls'))


@pytest.mark.parametrize('change',['foreign_namespace','foreign_regular_package','missing_lib',
    'changed_toy_models','missing_pin','missing_gate','changed_train','foreign_native_root'])
def test_missing_changed_or_foreign_native_resolution_refused(helper,synthetic_native,tmp_path,change):
    root,native,source=synthetic_native
    paths=[str(root)]
    if change.startswith('foreign_') and change!='foreign_native_root':
        foreign=tmp_path/'foreign';(foreign/'lib').mkdir(parents=True)
        (foreign/'lib/foreign.py').write_text('# no execution\n')
        if change=='foreign_regular_package': (foreign/'lib/__init__.py').write_text("raise AssertionError('never execute')\n")
        paths.append(str(foreign))
    elif change=='missing_lib': shutil.rmtree(native/'lib')
    elif change=='changed_toy_models': (native/'lib/toy_models.py').write_text('changed\n')
    elif change=='missing_pin': source['files'].pop('atlas19-external/native_root/lib/toy_models.py')
    elif change=='missing_gate': (native/'benchmarks/toy100/gate.py').unlink()
    elif change=='changed_train': (native/'benchmarks/toy100/train.py').write_text('changed\n')
    else: native=tmp_path/'different-root'
    with pytest.raises(ValueError): helper.native_scorer_import_metadata(source,native,paths=paths)


@pytest.mark.parametrize('name',['lib','lib.toy_models','benchmarks','benchmarks.toy100.train',
                                 'particlegan','particlegan.training'])
def test_no_loaded_candidate_package_can_be_reused(helper,synthetic_native,monkeypatch,name):
    _,native,source=synthetic_native
    monkeypatch.setitem(sys.modules,name,type('SyntheticAlreadyLoaded',(),{})())
    with pytest.raises(ValueError,match='fresh original package'):
        helper.normalize_native_scorer_paths(source,native)


def test_helper_bootstrap_eager_import_closure_is_metadata_only(helper):
    paths=[HERE/'run_retest.py',HERE/'protocol.py',ROOT/helper.protocol.LEGACY]
    paths += [ROOT/'experiments/forge'/name for name in (
        '__init__.py','contracts.py','policy_execution.py','queue.py','sources.py','__main__.py')]
    blocked={'torch','numpy','particlegan','lib','benchmarks','native100','current_api_fixtures'}
    def eager(nodes):
        for node in nodes:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): continue
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert all(name.split('.',1)[0] not in blocked for name in names)
            # Top-level conditional/try bodies still execute during bootstrap.
            for field in ('body','orelse','finalbody'):
                body=getattr(node,field,None)
                if isinstance(body,list): eager(body)
            for handler in getattr(node,'handlers',[]): eager(handler.body)
    for path in paths: eager(ast.parse(path.read_text(),filename=str(path)).body)
    source=(HERE/'run_retest.py').read_text()
    assert "protocol=module(DIRECTORY+'/protocol.py'" in source
    assert 'legacy=module(protocol.LEGACY' in source


def test_copied_preflight_checks_native_resolution_without_executing_imports(
        helper,synthetic_native,monkeypatch):
    root,native,_=synthetic_native
    adapter=root/'atlas19-external/adapter';harness=root/'atlas19-external/harness'
    for relative in ('screen_current.py','current_api_fixtures.py'):
        target=adapter/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('# synthetic overlay source\n')
    target=harness/'hosts/vector_host.py';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('# synthetic vector overlay source\n')
    source=pin_source(root)
    packet={'execution_source':source,'snapshot_locations':{
        'native_root':str(native),'adapter':str(adapter),'harness':str(harness)},
        'protocol':{'synthetic_only':True},'rows':[{'id':f'synthetic-{i}'} for i in range(19)]}
    guards=[];compiled=[]
    monkeypatch.setattr(helper,'ROOT',root)
    monkeypatch.setattr(helper,'validate_packet',lambda p,**k:None)
    monkeypatch.setattr(helper,'guard_imports',lambda p:guards.append(p))
    monkeypatch.setattr(helper,'observer',lambda:SimpleNamespace(
        overlay=lambda text,kind:(text,{'synthetic_only':True}),remove_overlay=lambda text,kind:text))
    monkeypatch.setattr(helper.legacy,'moving_source',lambda *a:'# synthetic moving source\n')
    def wrapper(*args):
        compiled.append(args[1]['id']);return [{'synthetic_only':True}]
    monkeypatch.setattr(helper,'build_wrappers',wrapper)
    # Avoid only the real enclosing checkout path; all namespace resolution
    # and source pin checks still execute on the synthetic copied layout.
    monkeypatch.setattr(sys,'path',[str(root)])
    before=set(sys.modules)
    record=helper.copied_preflight(packet)
    assert compiled==[r['id'] for r in packet['rows']] and guards==[source,source]
    assert record['native_scorer_imports']['status']=='PASS_NATIVE_SCORER_IMPORT_METADATA_ONLY'
    assert not any(name.split('.',1)[0] in {'lib','benchmarks','particlegan','torch'}
                   for name in set(sys.modules)-before)
    assert all(record[k]==0 for k in ('models','sampler_calls','scorer_calls','queue_calls'))


def copy_control_sources(root,helper):
    # Only current source files, for the actual generated wrapper bootstrap.
    paths=[p for p in (ROOT/'experiments/forge').rglob('*.py')]
    paths += [HERE/'run_retest.py',HERE/'protocol.py',
              ROOT/helper.protocol.LEGACY]
    for original in paths:
        target=root/original.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(original.read_bytes())


@pytest.mark.parametrize('repair',['new_owner','predecessor','foreign_namespace'])
def test_exact_generated_scorer_bootstrap_with_real_pythonpath_and_fake_hosts(
        helper,synthetic_native,tmp_path,monkeypatch,repair):
    root,native,_=synthetic_native
    copy_control_sources(root,helper)
    harness=root/'atlas19-external/harness';adapter=root/'atlas19-external/adapter'
    harness.mkdir();adapter.mkdir()
    # Copy the SHA-pinned inert fixture, never a local archived scorer. The
    # generated wrapper imports fake hosts and aborts before any score call.
    (harness/'native100_score.py').write_bytes(scorer_fixture_bytes())
    for name in ('current_api_fixtures.py','screen_current.py'):
        (adapter/name).write_bytes((ROOT/helper.legacy.ADAPTER/name).read_bytes())
    fixture={'frozen_repo':str(native),'host_source_sha256':{
        'lib/toy_models.py':helper.sha(native/'lib/toy_models.py')}}
    (harness/'tasks').mkdir();(harness/'tasks/native100_fixture.json').write_text(json.dumps(fixture))
    target=tmp_path/'synthetic-outputs';target.mkdir()
    packet={'snapshot_locations':{'native_root':str(native),'harness':str(harness),
        'adapter':str(adapter),'initializer':str(root/'atlas19-external/initializer')}}
    obs=helper.observer()  # Source transforms only; no capture/render/model.
    monkeypatch.setattr(helper,'observer',lambda:obs)
    monkeypatch.setattr(helper,'ROOT',root)
    entries=helper.build_wrappers(packet,{'group':'native','task':'grid100'},target,
                                 {'lease_fd':101,'lease_fds':[100,101]})
    scorer=next(e for e in entries if e['label']=='native-score-observed-wrapper')
    source=scorer['source']
    marker="    _r.normalize_native_scorer_paths(_q['packet']['execution_source'],ROOT)\n"
    assert source.count(marker)==1
    assert source.index(marker)<source.index('    from benchmarks.toy100 import gate, accuracy_gate')
    if repair=='predecessor': source=source.replace(marker,'',1)
    Path(scorer['executed_path']).write_text(source)
    scorer['executed_sha256']=hashlib.sha256(source.encode()).hexdigest()
    helper.atomic_json(target/'executed-source-overlays.json',{'records':[
        {k:v for k,v in e.items() if k!='source'} for e in entries if e is scorer]})
    pinned=pin_source(root)
    helper.atomic_json(target/'request.json',{'packet':{'execution_source':pinned}})
    foreign=tmp_path/'synthetic-foreign';(foreign/'lib').mkdir(parents=True)
    (foreign/'lib/unexpected.py').write_text('# synthetic foreign namespace, no execution\n')
    pythonpath=[str(root),str(root)+os.sep+'.']
    if repair=='foreign_namespace': pythonpath.append(str(foreign))
    code=("import importlib.util,json,sys\nfrom pathlib import Path as _AuditPath\n"
        f"_original_scorer=_AuditPath({str(Path(helper.legacy.HARNESS)/'native100_score.py')!r}).resolve()\n"
        "def _no_original_source_read(event,args):\n"
        "    if event=='open' and isinstance(args[0],(str,bytes)):\n"
        "        path=_AuditPath(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()\n"
        "        if path==_original_scorer:\n"
        "            raise AssertionError('portable control attempted original external scorer read')\n"
        "sys.addaudithook(_no_original_source_read)\n"
        f"s=importlib.util.spec_from_file_location('_synthetic_native_wrapper',{scorer['executed_path']!r})\n"
        "w=importlib.util.module_from_spec(s);sys.modules[s.name]=w;s.loader.exec_module(w)\n"
        "class StoppedBeforeScore(Exception): pass\n"
        "def stop():\n"
        "    w._r.guard_imports(w._q['packet']['execution_source'],derived=w._pins)\n"
        "    raise StoppedBeforeScore()\n"
        "w._retest_check=stop # this control never executes lease, score, or model calls\n"
        "sys.argv=['synthetic_wrapper','/never-read-evidence','grid100']\n"
        "try:\n    w.main()\n"
        "except StoppedBeforeScore:\n"
        "    import lib\n"
        "    assert len(list(lib.__path__))==1\n"
        "    assert not any(n=='torch' or n=='particlegan' or n.startswith('particlegan.') for n in sys.modules)\n"
        "    print(json.dumps({'status':'PASS_SYNTHETIC_IMPORT_ONLY','lib_paths':list(lib.__path__),"
        "'models':0,'scorer_calls':0,'sampler_calls':0,'queue_calls':0}))\n")
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(pythonpath))
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=root,env=env,
                          capture_output=True,text=True,timeout=15)
    if repair=='new_owner':
        assert result.returncode==0,result.stderr
        record=json.loads(result.stdout)
        assert record['lib_paths']==[str(native/'lib')]
        assert all(record[k]==0 for k in ('models','scorer_calls','sampler_calls','queue_calls'))
    else:
        assert result.returncode!=0
        expected=('ambiguous/missing protected namespace lib' if repair=='predecessor' else
                  'ambiguous/foreign native scorer namespace lib')
        assert expected in result.stderr,result.stderr


@pytest.mark.parametrize('change',['missing_proof','foreign_digest','wrong_lib_owner','changed_module_pin'])
def test_copied_native_import_receipt_cannot_be_substituted_before_admission(
        helper,synthetic_native,tmp_path,change):
    root,native,source=synthetic_native
    source['files'][helper.SELF]=helper.sha(HERE/'run_retest.py')
    source['digest']=helper.stable_hash(source['files'])
    packet={'protocol':{'synthetic_only':True},'execution_source':{**source,'origin_commit':'synthetic'},
        'snapshot_locations':{'native_root':str(native)},
        'copied_preflight_receipt_path':str(tmp_path/'synthetic-preflight.json')}
    _,proof=helper.native_scorer_import_metadata(packet['execution_source'],native,paths=[str(root)])
    record={'schema':'pg_pr223_copied_source_preflight_v1','status':'PASS_COPIED_METADATA_ONLY',
        'prepared_packet_sha256':helper.stable_hash(packet),'source_digest':source['digest'],
        'protocol_sha256':helper.stable_hash(packet['protocol']),'snapshot_path':str(root),
        'origin_commit':'synthetic','helper_sha256':helper.sha(HERE/'run_retest.py'),'cases':19,
        'updates':48800,'compiled_original_wrappers':19,'models':0,'sampler_calls':0,
        'scorer_calls':0,'queue_calls':0,'numeric_credit':False,'native_scorer_imports':deepcopy(proof)}
    helper.atomic_json(packet['copied_preflight_receipt_path'],record)
    assert helper.require_copied_preflight(packet)==record
    if change=='missing_proof': record.pop('native_scorer_imports')
    elif change=='foreign_digest': record['native_scorer_imports']['source_digest']='foreign'
    elif change=='wrong_lib_owner': record['native_scorer_imports']['resolved_modules']['lib']['path']='lib'
    else: record['native_scorer_imports']['resolved_modules']['lib.toy_models']['sha256']='0'*64
    helper.atomic_json(packet['copied_preflight_receipt_path'],record)
    with pytest.raises(ValueError,match='copied-source preflight native import'):
        helper.require_copied_preflight(packet)
