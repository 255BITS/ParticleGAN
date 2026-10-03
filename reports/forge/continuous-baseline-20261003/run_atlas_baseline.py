"""Finite original Atlas19 diagnostic baseline through Forge's shared owner.

Bulk traces/checkpoints stay in --output. No sweep, changed seed, new gate,
policy-hold requirement, automatic scientific retry or promotion is provided.
The optional GIFs display actual original metric observations, not model images.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.policy_execution import PolicyCoordinator, freeze_source
from experiments.forge.sources import snapshot_source, verify_snapshot
from experiments.forge.__main__ import queue_location

RELATIVE_SELF = 'reports/forge/continuous-baseline-20261003/run_atlas_baseline.py'
REFERENCE = 'reports/develop-gates-20261001/atlas19-replay.json'
REFERENCE_SHA = '1e1b536bfd66e20a240436b417d429059d22ea55c2b6ecfff31e796b96c470fa'
CONFIG = 'configs/100gaussians/atlas.json'
CONFIG_SHA = 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
ADAPTER = 'reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/validation-ra15'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
INITIALIZER = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/pkg-CB64-RA11/particlegan')
ROTATE = Path('/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py')
NATIVE = ('grid100', 'rotated100', 'staggered100')
PORTS = ('img_intensity2', 'mode_hold', 'img_blobs4', 'img_bars4', 'img_stripes2',
         'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width',
         'vector_anisotropic', 'vector_overlap', 'vector_spiral', 'stationary', 'ring_shift')
OPTIONS = dict(eval_output_noise=True, save_final_state=True, strict_streams=True,
               diagnostics=True, evaluation_generate='indexed', serial_backward_argument=True,
               initialization='batch_feature_zero', image_prior_perturb=False, ring_frozen_control=False)
TOTAL_CAP = 10800.
GRACE = 60.
FRAME_CONTRACT = {'kind': 'original_metric_observations_only', 'maximum_frames': 9,
                  'selection': 'inclusive equally spaced indices of retained observations',
                  'no_model_images_or_new_draws': True}
ENVIRONMENT = dict(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONUNBUFFERED='1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_digest(package_root):
    root=Path(package_root).resolve()/'particlegan';digest=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode()+b'\x00'+path.read_bytes()+b'\x00')
    return digest.hexdigest()


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'original wrapper relocation marker changed: {old}')
    return source.replace(old, new, 1)


def ordered_cases():
    return [('portability', task) for task in PORTS] + [('moving', task) for task in NATIVE] + [('native', task) for task in NATIVE]


def original_inputs(root):
    """Resolve and verify original receipts before any reservation or learner."""
    root = Path(root)
    if sha(root / REFERENCE) != REFERENCE_SHA or sha(root / CONFIG) != CONFIG_SHA:
        raise ValueError('original Atlas reference/config bytes changed')
    reference = read_json(root / REFERENCE)
    pins = reference['original_protocol_sources']['verified_source_sha256']
    files = {}
    for filename, expected in pins.items():
        path = Path(filename)
        if path.name in ('screen_current.py', 'current_api_fixtures.py') and 'validation-ra15' in filename:
            path = root / ADAPTER / path.name
        if sha(path) != expected:
            raise ValueError(f'original external protocol changed: {path}')
        files[str(path)] = expected
    # Include all source/data dependencies, not only the historical 60-file subset.
    for base in (HARNESS, INITIALIZER):
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json', '.gz'} and '__pycache__' not in path.parts:
                files[str(path)] = sha(path)
    files[str(ROTATE)] = sha(ROTATE)
    fixture = read_json(HARNESS / 'tasks/native100_fixture.json')
    native_root = Path(fixture['frozen_repo'])
    for name, expected in fixture['host_source_sha256'].items():
        if sha(native_root / name) != expected:
            raise ValueError(f'original native scorer/host changed: {name}')
    for directory in ('particlegan', 'benchmarks', 'lib', 'configs'):
        for path in sorted((native_root / directory).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.json'} and '__pycache__' not in path.parts:
                files[str(path)] = sha(path)
    return {'reference': reference, 'files': files, 'native_root': str(native_root),
            'harness': str(HARNESS), 'initializer': str(INITIALIZER), 'rotate': str(ROTATE)}


def native_requirements(inputs):
    source=next(Path(p) for p in inputs['files'] if Path(p).name=='screen_current.py')
    wanted=('NATIVE_COVERAGE_THRESHOLDS','NATIVE_ACCURACY_THRESHOLDS')
    found={}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in wanted:
                    found[target.id]=ast.literal_eval(node.value)
    if set(found)!=set(wanted):raise ValueError('original literal native thresholds unavailable')
    return [list(t) for name in wanted for t in found[name]]


def task_definition(group, task, inputs):
    reference = next(r for r in inputs['reference']['results'] if (r['group'], r['task']) == (group, task))
    harness = Path(inputs['harness'])
    if group in {'moving', 'native'}:
        definition = dict(steps=1500 if group == 'moving' else 7000, seed=1234,
                          num_particles=20000, z_dim=2, batch_size=2048)
        observations = [500, 1000, 1500] if group == 'moving' else [0, 1, 10, 25, 50, 100] + list(range(250, 7001, 250))
        if group == 'moving':
            thresholds = [['modes', '>=', 95], ['hq', '>=', '0.9 * observed pre_turn_hq']]
            definition.update(period_steps=500, rotate_degrees=30, turns=2, captured_points=4096,
                              evaluation_samples=20000)
        else:
            thresholds = native_requirements(inputs)
            definition.update(evaluation_samples=20000, terminal_steps=observations[-5:],
                              holdout_samples=100000, laws=['noisy primary', 'clean diagnostic'])
    elif task.startswith('img_'):
        spec = read_json(harness / 'tasks/image_task_specs.json')[task]
        definition = {k:spec[k] for k in ('steps','z_dim','batch_size')}
        definition.update(num_particles=spec['particles'], seed=0, original_spec=spec)
        observations = list(range(25, 601, 25))
        thresholds = reference['portability']['requirements']
    elif task.startswith('vector_'):
        spec = read_json(harness / 'tasks/vector_task_specs.json')[task]['spec']
        definition = dict(steps=spec['steps'], z_dim=spec['z_dim'], batch_size=spec['batch'],
                          num_particles=spec['particles'], seed=0, original_spec=spec)
        observations = [(i*spec['steps']+23)//24 for i in range(1,25)]
        thresholds = reference['portability']['requirements']
    else:
        mode = task == 'mode_hold'
        definition = dict(steps=1200 if mode else 4600 if task=='ring_shift' else 7500,
                          num_particles=12 if mode else 20000, z_dim=4 if mode else 2,
                          batch_size=128 if mode else 2048, seed=0)
        observations = list(range(50,1201,50)) if mode else list(range(10,definition['steps']+1,10))
        thresholds = reference['portability']['requirements']
    if definition['steps'] != reference['completed_steps'] or len(observations) != reference['observations']:
        raise ValueError('original Atlas task resource/cadence changed')
    return {'id':f'atlas-original19-{group}-{task}', 'group':group, 'task':task,
            'original_host':definition, 'observation_steps':observations, 'original_requirements':thresholds,
            'original_terminal_gate':True, 'added_policy_hold_gate':False,
            'original_options':deepcopy(OPTIONS), 'config_sha256':CONFIG_SHA,
            'sampling':'original noisy state-selected primary; clean diagnostics separate',
            'reference_sha256':REFERENCE_SHA, 'external_inputs_sha256':stable_hash(inputs['files']),
            'frame_contract':deepcopy(FRAME_CONTRACT)}


def plan(root=ROOT, *, inputs=None):
    root = Path(root).resolve()
    inputs = original_inputs(root) if inputs is None else inputs
    definitions = {d['id']:d for d in (task_definition(g,t,inputs) for g,t in ordered_cases())}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    sources = {str(p.relative_to(root)):sha(p) for p in (root/'particlegan').glob('*.py')}
    for relative in (RELATIVE_SELF, CONFIG, REFERENCE): sources[relative]=sha(root/relative)
    rows = [{'id':d['id'], 'group':d['group'], 'task':d['task'], 'status':'UNKNOWN',
             'timeout_seconds':2400. if d['group']=='native' else 1800.,
             'allowance_seconds':(2400. if d['group']=='native' else 1800.)+GRACE,
             'case_sha256':stable_hash(d)} for d in definitions.values()]
    if sum(d['original_host']['steps'] for d in definitions.values())!=48800:
        raise ValueError('original19 total update budget changed')
    return {'schema':'atlas_original19_baseline_v1', 'spec':{'id':'atlas-original19-baseline-20261003',
             'representation_card':{'path':REFERENCE,'sha256':REFERENCE_SHA},'export_grace_seconds':GRACE,
             'resources':{'host_memory_mb':2048},'total_paid_cap_seconds':TOTAL_CAP,
             'diagnostic_independent_tests':True,'frame_contract':FRAME_CONTRACT},
            'source':{'commit':commit,'files_sha256':sources}, 'external_inputs':inputs,
            'case_definitions':definitions, 'family':'atlas',
            'recipe_overrides':{'original_config_sha256':CONFIG_SHA,'original_options':deepcopy(OPTIONS)},
            'rows':rows,'status':'READY','required':19,'spent_seconds':0.,'new_paid_seconds':0.,
            'qualification_input':False,'default_adoption':False,'current_forge_mog_clean_qualification':False}


def _external_relative(filename, packet):
    p=Path(filename); inp=packet['external_inputs']
    for label in ('harness','initializer','native_root'):
        base=Path(inp[label])
        if p.is_relative_to(base): return Path('atlas19-external')/label/p.relative_to(base)
    if p==Path(inp['rotate']): return Path('atlas19-external/rotate_gate.py')
    if p.name in ('screen_current.py','current_api_fixtures.py'):return Path('atlas19-external/adapter')/p.name
    raise ValueError(f'undeclared external source layout: {p}')


def preparation_path(output):
    output=Path(output).resolve()
    return output.parent/f'.{output.name}.atlas19-preparation.json'


def prepare(output, *, root=ROOT, queue_root=None):
    output=Path(output).resolve(); queue_root=queue_location(Path(root),queue_root)
    metadata=preparation_path(output)
    if metadata.exists():
        saved=read_json(metadata)
        verify_snapshot(Path(saved['execution_source']['snapshot_path']),saved['execution_source'])
        return saved
    packet=plan(root)
    # freeze_source owns executable/config discovery; the historical JSON card
    # is an explicit input added below, rather than an implicit executable.
    expected=deepcopy(packet['source']);expected['files_sha256'].pop(REFERENCE)
    base=freeze_source(root,queue_root,expected)
    # Reuse the maintained immutable snapshot builder with one complete source
    # manifest. Location-only wrappers are derived at child time and recorded.
    with tempfile.TemporaryDirectory(prefix='atlas19-source-',dir=queue_root) as tmp:
        staging=Path(tmp); files=dict(base['files'])
        for relative in files:
            target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(base['snapshot_path'])/relative,target)
        reference=Path(root)/REFERENCE
        if sha(reference)!=packet['source']['files_sha256'][REFERENCE]:
            raise ValueError('original reference changed during freeze')
        target=staging/REFERENCE;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(reference,target);files[REFERENCE]=sha(reference)
        for filename,expected in packet['external_inputs']['files'].items():
            if sha(filename)!=expected:raise ValueError('external source changed during freeze')
            relative=_external_relative(filename,packet).as_posix()
            target=staging/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(filename,target);files[relative]=expected
        manifest={'schema_version':1,'digest':stable_hash(files),'files':files,'origin_commit':packet['source']['commit']}
        snapshot=snapshot_source(staging,queue_root/'policy/atlas19',manifest)
    packet['execution_source']={**manifest,'snapshot_path':str(snapshot)}
    packet['source']['execution_digest']=manifest['digest']
    packet['snapshot_locations']={k:str(snapshot/'atlas19-external'/k) for k in ('harness','initializer','native_root','adapter')}
    packet['snapshot_locations']['rotate']=str(snapshot/'atlas19-external/rotate_gate.py')
    packet['snapshot_locations']['package']=str(snapshot)
    packet['queue_root']=str(queue_root)
    # The maintained coordinator rejects unregistered nonempty archives.
    # Preparation is durable beside the fresh canonical archive, not inside it.
    atomic_json(metadata,packet)
    return packet


def runtime():
    import torch
    return dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                device='cuda:0',cuda_device_model=torch.cuda.get_device_name(0),torch_threads=1)


def gpu_readiness():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        return {'ready':False,'reason':'requires numeric CUDA_VISIBLE_DEVICES=1 and logical cuda:0'}
    try:
        values=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free,temperature.gpu',
                  '--format=csv,noheader,nounits'],text=True,stderr=subprocess.STDOUT).strip().split(',')
        free,temp=[float(v.strip()) for v in values]
        if not math.isfinite(free) or not math.isfinite(temp):raise ValueError('nonfinite GPU telemetry')
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        return {'ready':False,'reason':f'GPU1 telemetry unavailable: {error}'}
    return {'ready':free>=12288 and temp<=82,'free_mib':free,'temperature_c':temp,
            'reason':None if free>=12288 and temp<=82 else 'GPU1 requires free >=12288 MiB and temperature <=82 C'}


def attempt_identity(packet,row):
    # Existing coordinator key binds case law/options/seed/cadence/frame contract,
    # complete source digest, family/config, scientific hardware and full cap.
    coordinator=PolicyCoordinator(packet['queue_root'])
    return coordinator.attempt_key(packet,{'family':'atlas','recipe_overrides':packet['recipe_overrides']},row)


def _load_module(name,path,source=None):
    import types
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(Path(path).read_text() if source is None else source,str(path),'exec'),module.__dict__)
    return module


def moving_source(packet, target):
    locations=packet['snapshot_locations']; source=Path(locations['rotate']).read_text()
    source=_replace(source,"REPO = '/ml2/hypergan/ParticleGAN-pr155-merge'",f"REPO = {locations['package']!r}")
    source=_replace(source,"HOSTS = '/ml2/hypergan/lrfree-20260926/harness/hosts'",f"HOSTS = {str(Path(locations['harness'])/'hosts')!r}")
    source=_replace(source,"options = json.load(open(f'{REPO}/configs/100gaussians/e22-noout.json'))",f"options = json.load(open({str(Path(locations['package'])/CONFIG)!r}))")
    source=_replace(source,'            latent, _ = table.sample(n, generator=latent_stream)','            latent, indices = table.sample(n, generator=latent_stream)')
    source=_replace(source,'            clean = trainer._generate(model, latent, 0., latent_stream)','            clean = trainer._generate(model, latent, 0., latent_stream, indices=indices)')
    source=_replace(source,'torch.cuda.set_device(0); torch.set_num_threads(1); torch.set_num_interop_threads(1)',
                           'torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.2, 0); torch.set_num_threads(1); torch.set_num_interop_threads(1)')
    source=_replace(source,"print('GATE ' + json.dumps(row), flush=True)","print('GATE ' + json.dumps(row), flush=True)\n        torch.save(trainer.state_dict(), args.out + f'.checkpoint-{step:06d}.pt')")
    source=_replace(source,"    json.dump(verdict, open(args.out + '.verdict.json', 'w'), indent=1)","    assert len(gate_rows) == 3 and len(ok) == 2\n    json.dump(verdict, open(args.out + '.verdict.json', 'w'), indent=1)")
    return source


def child(request_path):
    request=read_json(request_path);packet=request['packet'];row=request['row'];target=Path(request['target'])
    verify_snapshot(Path(packet['execution_source']['snapshot_path']),packet['execution_source'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('child GPU placement changed')
    for name in ('ABSENT','ABSENT_START','ABSENT_END','LRFREE_NATIVE_TEST_STEPS'):os.environ.pop(name,None)
    target.mkdir(parents=True,exist_ok=True)
    loc=packet['snapshot_locations'];sys.path.insert(0,loc['package'])
    if row['group']=='moving':
        source=moving_source(packet,target);script=target/'runner.py';script.write_text(source)
        sys.argv=[str(script),row['task'],str(target/'frames.npz'),'--every','500','--points','4096',
                  '--steps','1500','--gate','--rotate-every','500','--rotate-deg','30']
        def moving_cap(signum,frame):raise TimeoutError('original acquisition wall cap exhausted')
        signal.signal(signal.SIGALRM,moving_cap);signal.setitimer(signal.ITIMER_REAL,row['timeout_seconds'])
        try:exec(compile(source,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    else:
        import torch
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
        # Only filesystem locations change. Mathematical source bytes are kept
        # in the immutable manifest; the executed wrapper bytes are retained.
        sys.path.insert(0,loc['harness']);sys.path.insert(0,loc['adapter'])
        fp=Path(loc['adapter'])/'current_api_fixtures.py';fs=fp.read_text()
        fs=_replace(fs,f"REFERENCE = Path({str(INITIALIZER)!r})",f"REFERENCE = Path({loc['initializer']!r})")
        _load_module('current_api_fixtures',fp,fs)
        sp=Path(loc['adapter'])/'screen_current.py';ss=sp.read_text()
        ss=_replace(ss,f"HARNESS = Path({str(HARNESS)!r})",f"HARNESS = Path({loc['harness']!r})")
        # The original scorer still executes separately against its original
        # package bytes, now copied into the same immutable source snapshot.
        scorer=Path(loc['harness'])/'native100_score.py'
        scored_source=_replace(scorer.read_text(),"ROOT = Path(FIXTURE['frozen_repo'])",f"ROOT = Path({loc['native_root']!r})")
        score_wrapper=target/'native-score-wrapper.py'
        score_wrapper.write_text("exec(compile("+repr(scored_source)+","+repr(str(scorer))+",'exec'),{'__name__':'__main__','__file__':"+repr(str(scorer))+"})\n")
        ss=_replace(ss,"str(HARNESS / 'native100_score.py')",f"{str(score_wrapper)!r}")
        (target/'adapter-relocation.json').write_text(json.dumps({'screen_source_sha256':hashlib.sha256(ss.encode()).hexdigest(),'fixture_source_sha256':hashlib.sha256(fs.encode()).hexdigest(),'native_scorer_source_sha256':hashlib.sha256(scored_source.encode()).hexdigest(),'native_scorer_wrapper_sha256':sha(score_wrapper),'scope':'filesystem-only relocation; original native scorer package remains byte-bound'}))
        module=_load_module('_atlas19_screen',sp,ss)
        def cap(signum,frame):
            raise TimeoutError('original acquisition wall cap exhausted')
        signal.signal(signal.SIGALRM,cap)
        signal.setitimer(signal.ITIMER_REAL,row['timeout_seconds'])
        close=module.Context.close
        def close_after_acquisition(self):
            signal.setitimer(signal.ITIMER_REAL,0)
            return close(self)
        module.Context.close=close_after_acquisition
        sys.argv=[str(sp),'--package-root',loc['package'],'--overrides',str(Path(loc['package'])/CONFIG),
                  '--task',row['task'],'--output',str(target),'--device','cuda:0',
                  '--candidate-options',json.dumps(OPTIONS),'--cand','atlas-original19-baseline-20261003']
        try:module.main()
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    verify_snapshot(Path(packet['execution_source']['snapshot_path']),packet['execution_source'])


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def certify(packet,row,target,returncode):
    """Validate complete original resources; JSON verdict, never exit0, grades."""
    target=Path(target);definition=packet['case_definitions'][row['id']]
    filename='frames.npz.verdict.json' if row['group']=='moving' else 'result.json'
    path=target/filename
    if not path.is_file():return {'status':'INCOMPLETE','reason':'original result unavailable; no retry'}
    result=read_json(path)
    if result.get('task')!=row['task'] or result.get('status') not in {'PASS','FAIL'}:
        return {'status':'INCOMPLETE' if result.get('status')!='ERROR' or 'original acquisition wall cap exhausted' in result.get('error','') else 'ERROR','reason':'original result has no complete scientific verdict','result_path':str(path),'result_sha256':sha(path)}
    if returncode!=0:raise ValueError('completed original verdict has unsuccessful child exit')
    host=definition['original_host']; scientific_status=result['status']; native_gates={}
    if row['group']=='moving':
        observations=result.get('periods',[])
        if result.get('turns')!=2 or [o.get('period_end') for o in observations]!=definition['observation_steps'] or [o.get('target_deg') for o in observations]!=[0,30,60]:
            raise ValueError('moving original periods/turns incomplete')
        required=[target/f'frames.npz.checkpoint-{s:06d}.pt' for s in definition['observation_steps']]
        if not all(p.is_file() for p in required):raise ValueError('moving checkpoints incomplete')
        if not (target/'frames.npz').is_file():raise ValueError('actual moving arrays unavailable')
    else:
        if result.get('completed_steps')!=host['steps']:
            return {'status':'INCOMPLETE','reason':'original full update budget unavailable','completed_steps':result.get('completed_steps'),'result_path':str(path),'result_sha256':sha(path)}
        if result.get('stream_deviations')!=0 or result.get('header',{}).get('options')!=OPTIONS:
            raise ValueError('original RNG/options contract changed')
        if result['header'].get('device')!='cuda:0' or result['header'].get('cuda_visible_devices')!='1':raise ValueError('GPU placement differs')
        package_root=Path(packet['snapshot_locations']['package']).resolve()
        if result['header'].get('package_root')!=str(package_root) or result['header'].get('package_sha256')!=package_digest(package_root):
            raise ValueError('executed package identity differs from immutable source')
        if result['header'].get('overrides')!={**read_json(Path(packet['snapshot_locations']['package'])/CONFIG),'initialization':'batch_feature_zero'}:
            raise ValueError('original configuration changed')
        observations=_jsonl(target/'metrics.jsonl')
        if [o.get('step') for o in observations]!=definition['observation_steps']:raise ValueError('original full observation cadence unavailable')
        if not (target/'final-state.pt').is_file():raise ValueError('complete public state unavailable')
        if row['group']=='native':
            import numpy as np
            for law in ('noisy','clean'):
                folder=target/f'native-{law}';summary=read_json(folder/'summary.json');verdict=read_json(folder/'verdict.json')
                if summary.get('completed_steps')!=7000 or summary.get('eval_steps')!=definition['observation_steps'] or summary.get('accuracy',{}).get('holdout_samples')!=100000:
                    raise ValueError('native original clean/noisy/100k protocol incomplete')
                if law=='noisy' and result['status']!=verdict['accuracy']['status']:raise ValueError('native primary verdict changed')
                if verdict['coverage']['status'] not in {'PASS','FAIL'} or verdict['accuracy']['status'] not in {'PASS','FAIL'}:
                    raise ValueError('native original gates have no complete verdict')
                native_gates[law]={k:verdict[k]['status'] for k in ('coverage','accuracy')}
                for name,count in [('final_samples.npz',20000),('holdout_samples.npz',100000)]+[(f'quality_checks/step_{s:06d}.npz',20000) for s in host['terminal_steps']]:
                    with np.load(folder/name,allow_pickle=False) as cloud:
                        if any(cloud[k].shape!=(count,2) or not np.isfinite(cloud[k]).all() for k in ('live','ema','target')):
                            raise ValueError('native original saved draw law/count invalid')
            scientific_status='PASS' if all(v=='PASS' for v in native_gates['noisy'].values()) else 'FAIL'
    artifacts={str(p.relative_to(target)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in target.rglob('*') if p.is_file()}
    return {'status':scientific_status,'original_gate':scientific_status,
            'reported_original_status':result['status'],'original_protocol_gate':scientific_status,
            'native_gates':native_gates,'completed_steps':host['steps'],
            'result_path':str(path),'result_sha256':sha(path),'artifacts':artifacts,
            'metric_observations':len(observations),'full_protocol_complete':True,
            'final_metrics':result.get('final',observations[-1] if observations else None),
            'clean_diagnostic_status':result.get('clean_status'),
            'sampling':definition['sampling'],'qualification_input':False}


def render_metric_gif(packet,row,target):
    """Plot existing training metrics only; no host/model/scorer or new draws."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    target=Path(target);definition=packet['case_definitions'][row['id']]
    data=read_json(target/'frames.npz.verdict.json')['periods'] if row['group']=='moving' else _jsonl(target/'metrics.jsonl')
    if not data:raise ValueError('no retained training metric observations')
    steps=[r.get('step',r.get('period_end')) for r in data]
    if steps!=definition['observation_steps']:raise ValueError('metric GIF requires complete actual cadence')
    requirements=definition['original_requirements']
    if row['group']=='native':
        shown={'modes','precision','mass_tv','acc_center_rms_sigma','acc_abs_cov_trace_bias','acc_radial_ks'}
        requirements=[x for x in requirements if x[0] in shown]
    elif row['group']=='moving':requirements=[['modes','>=',95],['hq','>=',.9*read_json(target/'frames.npz.verdict.json')['pre_turn_hq']]]
    requirements=[x for x in requirements if x[0] in data[-1] and isinstance(x[2],(float,int))][:6]
    if not requirements:raise ValueError('no original metric available for illustrative GIF')
    indices=np.unique(np.linspace(0,len(data)-1,min(9,len(data)),dtype=int)).tolist();frames=[]
    missing={name:[step for step,r in zip(steps,data) if r.get(name) is None]
             for name,_,_ in requirements}
    for index in indices:
        fig,axes=plt.subplots(len(requirements),1,figsize=(8,2.1*len(requirements)+1.3),squeeze=False)
        for ax,(name,op,bound) in zip(axes[:,0],requirements):
            values=[r.get(name) for r in data]
            # Original native accuracy records legitimately use None before
            # those diagnostics are available. Preserve gaps visibly; absence
            # earns no numerical pass and is never replaced with zero.
            if any(v is not None and (type(v) not in (int,float) or not math.isfinite(v)) for v in values):raise ValueError('nonfinite retained plot metric')
            numeric=[v for v in values if v is not None]
            ax.plot(steps[:index+1],values[:index+1],color='#d44a6b',marker='.',markersize=3);ax.axhline(bound,color='#577588',linestyle='--',label=f'{name} {op} {bound:g}')
            if values[index] is None:
                ax.text(.98,.82,'Not available at this check',transform=ax.transAxes,ha='right',fontsize=8)
            ax.set_xlim(0,definition['original_host']['steps']);lo=min(numeric+[bound]);hi=max(numeric+[bound]);pad=max((hi-lo)*.15,.01)
            ax.set_ylim(lo-pad,hi+pad);ax.set_ylabel(name);ax.legend(loc='best',fontsize=8)
        axes[-1,0].set_xlabel('Actual completed updates')
        fig.suptitle(f'Atlas original19: {row["group"]}/{row["task"]}\nActual metric traces only; generated-image frames were not retained',fontsize=11)
        scope=('Shown subset; full coverage + final-five fidelity + independent100k required.' if row['group']=='native' else 'Original noisy primary; clean diagnostic separate. No Forge MoG/default qualification.')
        fig.text(.06,.025,f'Original full test {row["status"]} | actual checkpoint {steps[index]}/{definition["original_host"]["steps"]}\n{scope}',fontsize=9)
        fig.tight_layout(rect=(0,.09,1,.91));fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'));plt.close(fig)
    destination=target/'goal-metrics.gif';frames[0].save(destination,save_all=True,append_images=frames[1:],duration=700,loop=0)
    with Image.open(destination) as gif:count=gif.n_frames
    result={'path':str(destination),'sha256':sha(destination),'bytes':destination.stat().st_size,'frames':count,
            'actual_steps':[steps[i] for i in indices],'metric_only':True,'new_draws':False,'training_updates':0,
            'unavailable_observations':{k:v for k,v in missing.items() if v},
            'renderer_source':{'path':RELATIVE_SELF,'sha256':sha(__file__),
                               'scope':'offline visualization only; frozen scientific child and gates unchanged',
                               'scientific_source_digest':packet['source'].get('execution_digest')}}
    atomic_json(target/'media-receipt.json',result)
    return result


def _save(path,packet):
    packet['spent_seconds']=sum(r.get('charged_seconds',0.) for r in packet['rows'])
    packet['new_paid_seconds']=sum(r.get('new_paid_seconds',0.) for r in packet['rows'])
    terminal=all(r['status'] not in {'UNKNOWN','RUNNING'} for r in packet['rows'])
    packet['status']='PASS' if terminal and all(r['status']=='PASS' for r in packet['rows']) else 'FAIL' if terminal and any(r['status']=='FAIL' for r in packet['rows']) else 'INCOMPLETE' if terminal else 'READY'
    packet['completed']=sum(r.get('full_protocol_complete',False) for r in packet['rows'])
    packet['media_completed']=sum(bool(r.get('media')) for r in packet['rows'])
    packet['required_evidence_complete']=packet['completed']==19 and packet['media_completed']==19
    packet['media_status']='COMPLETE' if packet['media_completed']==19 else 'INCOMPLETE'
    if packet['status']=='PASS' and not packet['required_evidence_complete']:packet['status']='INCOMPLETE'
    atomic_json(path,packet)


def _verify_retained(row):
    if row.get('result_path') and sha(row['result_path'])!=row['result_sha256']:raise ValueError('retained original result changed; never reexecute')
    root=Path(row['result_path']).parent if row.get('result_path') else None
    if row.get('media'):
        m=row['media']
        if sha(m['path'])!=m['sha256'] or Path(m['path']).stat().st_size!=m['bytes']:raise ValueError('retained metric GIF changed')
    for rel,meta in row.get('artifacts',{}).items():
        if sha(root/rel)!=meta['sha256'] or (root/rel).stat().st_size!=meta['bytes']:raise ValueError('retained raw artifact changed')


def run(output, *, root=ROOT, queue_root=None, max_new_attempts=None):
    output=Path(output).resolve();preparation=preparation_path(output)
    prepared=read_json(preparation) if preparation.exists() else prepare(output,root=root,queue_root=queue_root)
    verify_snapshot(Path(prepared['execution_source']['snapshot_path']),prepared['execution_source'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        prepared['waiting_reason']='requires numeric CUDA_VISIBLE_DEVICES=1 and logical cuda:0'
        atomic_json(output/'registration.json',prepared);return prepared
    prepared['lane_runtime']=runtime();coordinator=PolicyCoordinator(prepared['queue_root'],report_root=Path(root)/'reports/forge')
    key,canonical=coordinator.register(prepared,output,'atlas',prepared['lane_runtime'])
    launched=0
    with coordinator.study_lease(key) as study_lease:
        if study_lease is None:
            return coordinator.publish_attachment(key,canonical,output)
        packet=read_json(canonical/'study.json');packet.pop('waiting_reason',None)
        coordinator.recover()
        for row in packet['rows']:
            if row['status'] not in {'UNKNOWN','RUNNING'}:_verify_retained(row);continue
            attempt_key=attempt_identity(packet,row);row['attempt_key']=attempt_key
            retained=coordinator.retained(attempt_key)
            # Reuse/certify originals before checking whether new work can fit.
            if not retained and packet['spent_seconds']+row['allowance_seconds']>TOTAL_CAP:
                packet['waiting_reason']='remaining paid cap cannot reserve complete next original task';break
            if not retained and max_new_attempts is not None and launched>=max_new_attempts:
                packet['waiting_reason']='explicit new-attempt dispatch limit reached';break
            if not retained:
                readiness=gpu_readiness()
                if not readiness['ready']:packet['waiting_reason']=readiness['reason'];break
            with coordinator.admit(attempt_key,packet,row,'cuda:0') as (admission,lease):
                if admission['status']=='busy':packet['waiting_reason']=admission['reason'];break
                if admission['status']=='completed':
                    result=deepcopy(admission['result']);_verify_retained(result)
                    row.update(result,reused_physical_attempt=True,new_paid_seconds=0.)
                elif admission['status']=='interrupted':
                    row.update(status='INCOMPLETE',reason=admission['reason'],charged_seconds=admission['charged_seconds'],new_paid_seconds=0.,reused_physical_attempt=True)
                elif admission['status']=='awaiting_certification':
                    target=Path(admission['command'][-1]).parent;request=read_json(admission['command'][-1]);target=Path(request['target'])
                    result=certify(packet,row,target,admission['terminal']['child_returncode'])
                    result.update(paid_wall_seconds=admission['terminal']['paid_wall_seconds'],charged_seconds=admission['terminal']['paid_wall_seconds'])
                    coordinator.complete(attempt_key,result);row.update(result,reused_physical_attempt=True,new_paid_seconds=0.)
                else:
                    target=canonical/row['group']/row['task']
                    if target.exists():
                        result={'status':'INCOMPLETE','reason':'orphan original artifacts retained; no automatic retry','paid_wall_seconds':0.,'charged_seconds':row['allowance_seconds'],'unmeasured_interrupt_reserved_seconds':row['allowance_seconds']}
                    else:
                        target.mkdir(parents=True);request=target/'request.json'
                        atomic_json(request,{'packet':packet,'row':row,'target':str(target)})
                        command=[sys.executable,'-u','-B',str(Path(packet['execution_source']['snapshot_path'])/RELATIVE_SELF),'--child',str(request)]
                        row.update(status='RUNNING',command=command,log_path=str(target/'run.log'));_save(canonical/'study.json',packet)
                        print(json.dumps({'event':'start','case':row['id'],'attempt_key':attempt_key,'log':row['log_path']}),flush=True)
                        done=None
                        try:
                            done=coordinator.launch(command,packet,target/'run.log',(study_lease,lease),row['allowance_seconds'])
                            result=certify(packet,row,target,done.returncode)
                            result.update(child_returncode=done.returncode,paid_wall_seconds=done.paid_wall_seconds,charged_seconds=done.paid_wall_seconds,new_paid_seconds=done.paid_wall_seconds)
                        except subprocess.TimeoutExpired as error:
                            paid=getattr(error,'paid_wall_seconds',None)
                            reserved=row['allowance_seconds'] if paid is None else 0.
                            result={'status':'INCOMPLETE','reason':'supervised original full-task cap exhausted; no retry','paid_wall_seconds':paid or 0.,'charged_seconds':(paid or 0.)+reserved,'unmeasured_interrupt_reserved_seconds':reserved,'new_paid_seconds':paid or 0.}
                        except Exception as error:
                            terminal_path=Path(admission['lease_path']).parent/'supervisor-terminal.json'
                            terminal=read_json(terminal_path) if terminal_path.exists() else None
                            paid=done.paid_wall_seconds if done is not None else terminal['paid_wall_seconds'] if terminal and terminal.get('token')==admission['token'] else None
                            reserved=row['allowance_seconds'] if paid is None else 0.
                            result={'status':'ERROR','reason':f'{type(error).__name__}: {error}','paid_wall_seconds':paid or 0.,'charged_seconds':(paid or 0.)+reserved,'unmeasured_interrupt_reserved_seconds':reserved,'new_paid_seconds':paid or 0.}
                        launched+=1
                    coordinator.complete(attempt_key,result);row.update(result)
                _save(canonical/'study.json',packet)
                if row.get('full_protocol_complete') and not row.get('media'):
                    target=Path(row['result_path']).parent
                    try:row['media']=render_metric_gif(packet,row,target)
                    except Exception as error:row['media_error']=f'{type(error).__name__}: {error}'
                    _save(canonical/'study.json',packet)
                print(json.dumps({'event':'complete','case':row['id'],'status':row['status'],'completed':packet['completed'],'required':19,'spent_seconds':packet['spent_seconds']}),flush=True)
        _save(canonical/'study.json',packet)
    return coordinator.publish_attachment(key,canonical,output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--queue-root',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--max-new-attempts',type=int)
    parser.add_argument('--child',type=Path)
    args=parser.parse_args(argv)
    if args.child:child(args.child);return 0
    if args.output is None:parser.error('--output required')
    if args.max_new_attempts is not None and args.max_new_attempts<1:parser.error('dispatch limit must be positive')
    for key,value in ENVIRONMENT.items():
        if key=='CUDA_VISIBLE_DEVICES' and os.environ.get(key) not in (None,'1'):parser.error('physical GPU1 numeric mask required')
        os.environ[key]=value
    packet=prepare(args.output,queue_root=args.queue_root) if args.prepare_only else run(args.output,queue_root=args.queue_root,max_new_attempts=args.max_new_attempts)
    print(json.dumps({'status':packet['status'],'required':19,'completed':packet.get('completed',0),
                      'waiting_reason':packet.get('waiting_reason'),'registration':str(args.output/'study.json')}),flush=True)
    return 0 if packet['status']=='PASS' or args.prepare_only else 1


if __name__=='__main__':raise SystemExit(main())
