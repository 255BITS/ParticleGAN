"""Model-free generated-scorer import regression for the copied native3 helper.

Real scorer bytes are read as source only. A private layout uses the EXACT
copied helper/scorer with inert benchmark hosts and stops before either score
call. Nothing is hydrated, trained, sampled, or measured numerically.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


DIRECTORY = 'reports/forge/pr223-native3-continuation-20261004'
SELF = DIRECTORY + '/scorer_boundary_control.py'
SCORER_SHA = '10cc14edfcd98ab34fd3768aaba2ee835dc2241dc1face2e18998c8f2b687feb'
SCORER_BYTES = 1660
SCHEMA = 'pg_pr223_native3_scorer_boundary_control_v1'
EXPECTED = {'repaired': 'PASS_STOPPED_BEFORE_SCORE', 'predecessor': 'REFUSED_AMBIGUOUS_NAMESPACE',
            'foreign': 'REFUSED_FOREIGN_NAMESPACE', 'cached': 'REFUSED_CACHED_CANDIDATE'}


def binding(packet, helper):
    source = packet['execution_source'];snapshot = Path(source['snapshot_path']).resolve()
    scorer = Path(packet['snapshot_locations']['harness'])/'native100_score.py'
    relative = scorer.resolve().relative_to(snapshot).as_posix()
    if source['files'].get(relative) != SCORER_SHA or helper.sha(scorer) != SCORER_SHA or scorer.stat().st_size != SCORER_BYTES:
        raise ValueError('exact source-only original native scorer required')
    for relative_path in (helper.SELF, SELF):
        if helper.sha(snapshot/relative_path) != source['files'].get(relative_path):
            raise ValueError('copied helper/control bytes changed')
    return dict(source_digest=source['digest'], helper_sha256=source['files'][helper.SELF],
                control_sha256=source['files'][SELF], scorer_source_sha256=SCORER_SHA,
                scorer_source_bytes=SCORER_BYTES, protected_root='atlas19-external/native_root')


def validate_proof(proof, packet, helper):
    expected = dict(schema=SCHEMA, status='PASS_SYNTHETIC_SCORER_BOUNDARY', binding=binding(packet,helper),
                    outcomes=EXPECTED, synthetic_only=True, models=0, sampler_calls=0, scorer_calls=0,
                    queue_calls=0, numerical_credit=False, old_arrays_read=False)
    if proof != expected:
        raise ValueError('copied scorer-boundary regression proof is missing, foreign or incomplete')
    return proof


def _pin(root, helper):
    files = {p.relative_to(root).as_posix():helper.sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in {'.py','.json'}}
    return dict(snapshot_path=str(root), files=files, digest=helper.stable_hash(files))


def run(packet, helper):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('scorer-boundary structural control must hide CUDA')
    pins = binding(packet, helper)
    source = packet['execution_source'];snapshot = Path(source['snapshot_path']).resolve()
    # Copy only the eagerly imported stdlib Forge helper closure, never models.
    names = [p for p in source['files'] if p.startswith('experiments/forge/') and p.endswith('.py')]
    names += [helper.SELF, helper.DIRECTORY+'/protocol.py', helper.protocol.LEGACY]
    if not all(p in source['files'] for p in names):
        raise ValueError('incomplete copied helper bootstrap source')
    with tempfile.TemporaryDirectory(prefix='pr223-native3-inert-bootstrap-') as directory:
        root = Path(directory)/'synthetic-snapshot';root.mkdir()
        for name in names:
            data = (snapshot/name).read_bytes()
            if hashlib.sha256(data).hexdigest() != source['files'][name]:
                raise ValueError('copied bootstrap source changed')
            destination = root/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        native = root/'atlas19-external/native_root'
        inert = {
            'lib/toy_models.py': "raise AssertionError('candidate lib must never execute')\n",
            'atlas19-external/native_root/lib/toy_models.py': "OWNER='synthetic-native-only'\n",
            'atlas19-external/native_root/particlegan/__init__.py': "raise AssertionError('model import forbidden')\n",
            'atlas19-external/native_root/benchmarks/__init__.py': '# inert host\n',
            'atlas19-external/native_root/benchmarks/toy100/__init__.py': '# inert host\n',
            'atlas19-external/native_root/benchmarks/toy100/train.py': 'from lib import toy_models\n',
            'atlas19-external/native_root/benchmarks/toy100/gate.py':
                "from . import train\ndef score_run(*args):\n    raise AssertionError('scorer call forbidden')\n",
            'atlas19-external/native_root/benchmarks/toy100/accuracy_gate.py':
                "from . import gate\ndef score_run(*args):\n    raise AssertionError('scorer call forbidden')\n",
        }
        for name, text in inert.items():
            destination = root/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(text)
        harness = root/'atlas19-external/harness';adapter = root/'atlas19-external/adapter'
        harness.mkdir();adapter.mkdir()
        scorer = Path(packet['snapshot_locations']['harness'])/'native100_score.py'
        (harness/'native100_score.py').write_bytes(scorer.read_bytes())
        for name in ('current_api_fixtures.py','screen_current.py'):
            (adapter/name).write_bytes((Path(packet['snapshot_locations']['adapter'])/name).read_bytes())
        (harness/'tasks').mkdir()
        fixture = dict(frozen_repo=str(native),host_source_sha256={'lib/toy_models.py':helper.sha(native/'lib/toy_models.py')})
        helper.atomic_json(harness/'tasks/native100_fixture.json',fixture)
        target = Path(directory)/'synthetic-output';target.mkdir()
        synthetic = dict(snapshot_locations=dict(native_root=str(native),harness=str(harness),adapter=str(adapter),
                                                initializer=str(root/'atlas19-external/initializer')))
        entries = helper.build_wrappers(synthetic,dict(group='native',task='grid100'),target,
                                        dict(lease_fd=101,lease_fds=[100,101]))
        entry = next(e for e in entries if e['label']=='native-score-observed-wrapper')
        prefix = f"_s=_u.spec_from_file_location('_retest_child_checks',{str(helper.ROOT/helper.SELF)!r})"
        if entry['source'].count(prefix) != 1:
            raise ValueError('generated original scorer prefix changed')
        exact_wrapper = entry['source'].replace(prefix,
            f"_s=_u.spec_from_file_location('_retest_child_checks',{str(root/helper.SELF)!r})",1)
        marker = "    _r.normalize_native_scorer_paths(_q['packet']['execution_source'],ROOT)\n"
        if exact_wrapper.count(marker) != 1 or exact_wrapper.index(marker) >= exact_wrapper.index('    from benchmarks.toy100 import gate, accuracy_gate'):
            raise ValueError('native owner guard must precede exact original scorer imports')
        pinned = _pin(root, helper)
        helper.atomic_json(target/'request.json',dict(packet=dict(execution_source=pinned)))
        foreign = Path(directory)/'synthetic-foreign';(foreign/'lib').mkdir(parents=True)
        (foreign/'lib/unexpected.py').write_text('# inert foreign namespace\n')
        outcomes = {}
        for scenario in EXPECTED:
            wrapper = exact_wrapper.replace(marker,'',1) if scenario=='predecessor' else exact_wrapper
            path = Path(entry['executed_path']);path.write_text(wrapper)
            record = {k:v for k,v in entry.items() if k!='source'}
            record['executed_sha256'] = helper.sha(path)
            helper.atomic_json(target/'executed-source-overlays.json',{'records':[record]})
            pythonpath = [str(root),str(root)+os.sep+'.']
            if scenario=='foreign': pythonpath.append(str(foreign))
            code = ("import importlib.util,json,os,sys,types\n"
                "from pathlib import Path\n"
                "def no_live_inputs(event,values):\n"
                "    if event=='open' and isinstance(values[0],(str,bytes)):\n"
                "        p=Path(os.fsdecode(values[0])).resolve()\n"
                "        if p.is_relative_to('/ml2/hypergan') or p.is_relative_to('/home/martyn/dev/ParticleGAN/artifacts'):\n"
                "            raise AssertionError('inert scorer control must never open original live/raw inputs')\n"
                "sys.addaudithook(no_live_inputs)\n"
                f"s=importlib.util.spec_from_file_location('_synthetic_generated_scorer',{str(path)!r})\n"
                "w=importlib.util.module_from_spec(s);sys.modules[s.name]=w;s.loader.exec_module(w)\n"
                + ("sys.modules['particlegan.training']=types.ModuleType('synthetic_cached_candidate')\n" if scenario=='cached' else '')
                + "class StopBeforeScore(Exception): pass\n"
                "def stop():\n"
                "    w._r.guard_imports(w._q['packet']['execution_source'],derived=w._pins)\n"
                "    raise StopBeforeScore()\n"
                "w._retest_check=stop\n"
                "sys.argv=['synthetic_scorer','/never-read-evidence','grid100']\n"
                "try:\n    w.main()\n"
                "except StopBeforeScore:\n"
                "    import lib\n"
                f"    assert list(lib.__path__)==[{str(native/'lib')!r}]\n"
                "    assert not any(n.split('.',1)[0] in {'torch','particlegan'} for n in sys.modules)\n"
                "    print('PASS_STOPPED_BEFORE_SCORE')\n")
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(pythonpath))
            done = subprocess.run([sys.executable,'-B','-c',code],cwd=root,env=env,
                                  capture_output=True,text=True,timeout=15)
            if scenario=='repaired':
                if done.returncode != 0 or done.stdout.strip() != EXPECTED[scenario]:
                    raise ValueError('exact copied scorer bootstrap failed: '+done.stderr)
            else:
                wanted = {'predecessor':'ambiguous/missing protected namespace lib',
                          'foreign':'ambiguous/foreign native scorer namespace lib',
                          'cached':'native scorer requires fresh original package imports'}[scenario]
                if done.returncode == 0 or wanted not in done.stderr:
                    raise ValueError('negative native scorer owner control failed: '+scenario+' '+done.stderr)
            outcomes[scenario] = EXPECTED[scenario]
    result = dict(schema=SCHEMA,status='PASS_SYNTHETIC_SCORER_BOUNDARY',binding=pins,outcomes=outcomes,
                  synthetic_only=True,models=0,sampler_calls=0,scorer_calls=0,queue_calls=0,
                  numerical_credit=False,old_arrays_read=False)
    return validate_proof(result,packet,helper)
