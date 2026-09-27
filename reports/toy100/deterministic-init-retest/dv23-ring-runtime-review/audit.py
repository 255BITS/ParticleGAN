"""Independent terminal ring audit; standard library only, no model execution."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import math
import struct
import sys
import zipfile

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
from audit_public3_runtime import Reader, Tensor

BASE = Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/20260927T031740Z/dv23-new-init-ring/20260927T031740Z-1983204/repo/reports/reviewed-probe-output/fixed-batch-0ii7edgr')
OLD = P.parent / 'continuous-api-search/evidence'
sha = lambda data: hashlib.sha256(data).hexdigest()
read = lambda path: json.loads(path.read_bytes())


def lines(path):
    data = path.read_bytes()
    if path.suffix == '.gz': data = gzip.decompress(data)
    return [json.loads(s) for s in data.splitlines()]


class RawCheckpoint:
    def __init__(self, path):
        with zipfile.ZipFile(path) as z:
            name, = [n for n in z.namelist() if n.endswith('/data.pkl')]
            prefix = name[:-8]
            assert z.read(prefix + 'byteorder') == b'little'
            self.value = Reader(io.BytesIO(z.read(name))).load()
            self.storages = {n[len(prefix+'data/'):]: z.read(n) for n in z.namelist() if n.startswith(prefix+'data/')}

    def raw(self, t):
        kind, key, device, count = t.storage
        width, dtype = {'FloatStorage': (4, 'torch.float32'), 'ByteStorage': (1, 'torch.uint8')}[kind]
        data = self.storages[key]
        assert len(data) == count * width
        expected = 1
        for length, stride in reversed(list(zip(t.shape, t.stride))):
            assert length <= 1 or stride == expected
            expected *= length
        return data[t.offset*width:(t.offset+math.prod(t.shape))*width], dtype

    def material(self, x):
        if isinstance(x, Tensor):
            data, dtype = self.raw(x)
            return dict(shape=list(x.shape), dtype=dtype, sha256=sha(data))
        if isinstance(x, dict): return {str(k): self.material(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [self.material(v) for v in x]
        assert x is None or isinstance(x, (str,int,float,bool))
        return x

    def digest(self, x):
        h = hashlib.sha256()
        def visit(v):
            if isinstance(v, Tensor):
                data,dtype = self.raw(v)
                h.update(json.dumps(['tensor',dtype,list(v.shape)]).encode()); h.update(data)
            elif isinstance(v, dict):
                h.update(b'dict[')
                for k in sorted(v,key=lambda k:(type(k).__name__,str(k))): visit(k);visit(v[k])
                h.update(b']')
            elif isinstance(v,(list,tuple)):
                h.update(type(v).__name__.encode()+b'[')
                for item in v: visit(item)
                h.update(b']')
            else: h.update(json.dumps([type(v).__name__,v],allow_nan=False).encode())
        visit(x)
        return h.hexdigest()

    def receipt(self):
        e = self.value; s = e['trainer']; d = self.digest
        return dict(step=s['completed_steps'],trainer_sha256=d(s),
            models_sha256={k:d(v) for k,v in s['models'].items()},
            optimizers_sha256=[d(v) for v in s['optimizers']],
            streams_sha256={k:d(v) for k,v in s['streams'].items()},
            cpu_rng_sha256=d(s['cpu_rng']),cuda_rng_sha256=d(s['cuda_rng']),
            real_stream_sha256=d(e['data_rng']),means_sha256=d(e['means']))

    def clocks(self, step):
        opts = self.value['trainer']['optimizers']
        assert [len(o['state']) for o in opts] == ([0,0] if not step else [9,8])
        for opt in opts:
            for g in opt['param_groups']:
                assert g['capturable'] is g['fused'] is g['foreach'] is False
            for state in opt['state'].values():
                t = state['step']; data,dtype = self.raw(t)
                assert t.storage[2] == 'cpu' and t.shape == () and dtype == 'torch.float32'
                assert struct.unpack('<f',data)[0] == step
                for key in ('exp_avg','exp_avg_sq'):
                    assert state[key].storage[2] == 'cuda:0' and self.raw(state[key])[1] == 'torch.float32'


def segment(points, start, end):
    xs = [p for p in points if start < p['step'] <= end]
    good = lambda p: p['modes'] == 8 and p['hq'] >= .9
    first = next((p['step'] for p in xs if good(p)), None)
    after = [p for p in xs if first is not None and p['step'] >= first]
    suffix = []
    for p in reversed(xs):
        if not good(p): break
        suffix.append(p)
    return dict(start=start,end=end,first_arrival=first,delay=None if first is None else first-start,
        passing_since_arrival=sum(map(good,after)),checks_since_arrival=len(after),
        every_failing_observation_since_arrival=[p['step'] for p in after if not good(p)],
        departures=[p['step'] for i,p in enumerate(after) if i and good(after[i-1]) and not good(p)],
        stable_suffix_start=suffix[-1]['step'] if suffix else None,stable_suffix_checks=len(suffix),
        minimum_hq_since_arrival=min((p['hq'] for p in after),default=None),
        minimum_modes_since_arrival=min((p['modes'] for p in after),default=None),
        minimum_hq_entire_segment=min(p['hq'] for p in xs),final=xs[-1])


def archive(source, target):
    target.mkdir(parents=True,exist_ok=True)
    receipt = {}
    for f in sorted(source.iterdir()):
        assert f.is_file()
        data=f.read_bytes(); compressed=f.suffix in ('.pt','.jsonl')
        name=f.name+('.gz' if compressed else '')
        payload=gzip.compress(data,mtime=0) if compressed else data
        dest=target/name
        if dest.exists(): assert dest.read_bytes()==payload
        else: dest.write_bytes(payload)
        assert (gzip.decompress(dest.read_bytes()) if compressed else dest.read_bytes())==data
        receipt[f.name]=dict(archived_file=name,original_sha256=sha(data),archive_sha256=sha(payload),bytes=len(data))
    return receipt


def audit(n):
    alias=f'api-dv{n}'; d=BASE/(alias+'-single-shift'); prep=P/'dv23-single-shift-preparation'/alias
    old=OLD/(alias+'-single'); manifest=read(d/'artifact-sha256.json')
    assert {f.name:sha(f.read_bytes()) for f in d.iterdir() if f.name!='artifact-sha256.json'}==manifest
    with zipfile.ZipFile(d/'source.zip') as z:
        contents={name:z.read(name) for name in z.namelist()}
    seal=read(prep/'bundle-sha256.json'); seal_hash=sha((prep/'bundle-sha256.json').read_bytes())
    assert contents['bundle-sha256.json']==(prep/'bundle-sha256.json').read_bytes()
    assert set(contents)==set(seal['files'])|{'bundle-sha256.json'}
    for name,digest in seal['files'].items(): assert sha(contents[name])==digest
    declared=json.loads(contents['declaration.json']); actual=read(d/'declaration.json')
    assert {k:v for k,v in actual.items() if k not in ('started_utc','source_sha256')}=={**declared,'status':'DECLARED_BEFORE_EXECUTION'}
    assert actual['source_sha256']=={**seal['files'],'bundle-sha256.json':seal_hash}
    assert {k.removeprefix('source/'):sha(v) for k,v in contents.items() if k.startswith('source/particlegan/')}==declared['package_sha256']
    cpu=read(d/'reviewed-cpu-proof.json'); peer=read(P/'dv23-single-shift-preparation/reviews'/alias/'source-review.json')
    assert cpu==read(Path(peer['cpu_receipt'])) and sha((d/'reviewed-cpu-proof.json').read_bytes())==peer['cpu_receipt_sha256']
    assert cpu['status']=='PASS' and cpu['manifest_sha256']==seal_hash==peer['manifest_sha256']
    assert read(d/'source-preflight.json')==dict(status='PASS_SOURCE_ONLY',manifest_sha256=seal_hash,package_files=24,quality='NOT_RUN')
    runtime=read(d/'runtime.json'); expected=declared['runtime_expected']
    for key in ('torch','cuda','gpu','threads','interop_threads','deterministic','tf32'): assert runtime[key]==expected[key]
    assert runtime['executable']==expected['interpreter'] and runtime['default_factory_device']=='cpu'
    assert runtime['execution']==dict(serial_backward=True,scope='full GANTrainer.step')
    for module,item in runtime['imported_package'].items():
        name='source/'+('particlegan/__init__.py' if module=='particlegan' else module.replace('.','/')+'.py')
        assert item['sha256']==sha(contents[name])==sha(Path(item['path']).read_bytes())
        assert '/reports/reviewed-probe-inputs/'+alias+'/source/' in item['path']
    for key,item in runtime['pinned_torch_sources'].items(): assert item['sha256']==expected['source_hashes'][key]
    assert runtime['environment']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): assert runtime['environment'][name]=='1'
    points=lines(d/'metrics.jsonl'); rates=lines(d/'learning-rates.jsonl'); states=lines(d/'state-hashes.jsonl')
    assert [p['step'] for p in points]==list(range(10,4601,10))
    assert [p['step'] for p in rates]==list(range(1,4601))
    assert [p['step'] for p in states]==list(range(100,4601,100))
    result=read(d/'result.json'); summary=read(d/'summary.json')
    assert result['status']=='COMPLETE' and result['metrics']==summary
    derived=[segment(points,0,2400),segment(points,2400,4600)]
    assert derived==summary['segments'] and segment(points,2400,3600)==summary['comparison3600']
    assert summary['observations']==460 and summary['completed_updates']==summary['rate_rows']==4600
    assert summary['final']==points[-1] and summary['source_and_runtime_assertions_passed'] is True
    controls=[p['frozen'] for p in points if p['step']>2400]
    assert len(controls)==220 and all('frozen' not in p for p in points if p['step']<=2400)
    assert summary['frozen_control']==dict(checkpoint=2400,updates_after_copy=0,passing=sum(p['modes']==8 and p['hq']>=.9 for p in controls),observations=220,maximum_hq=max(p['hq'] for p in controls))
    recipe=declared['recipe']; assert recipe['total_steps'] is None and recipe['initialization']=='batch_feature_zero'
    mobility=1.; alignment=0.; memory=0.; previous_reopens=0; events=[]
    for r in rates:
        policy=r['policy']; drive=policy['data_drive']
        target=max(drive,min(1.,max(0.,alignment/.2)))
        mobility+=(.05 if target>mobility else .005)*(target-mobility)
        memory+=(.05 if drive>memory else .005)*(drive-memory)
        assert policy['mobility']==mobility and policy['data_memory']==memory
        expected_rates=dict(generator_0=recipe['lr']*(.01+.99*mobility),
            prior_1=recipe['lr']*recipe['prior_lr_mult']*(.05+.95*mobility),
            critic_0=recipe['lr']*recipe['d_lr_mult']*(.01+.99*mobility))
        for key,val in expected_rates.items(): assert r[key]==val
        assert r['step']==r['accepted_updates']==policy['updates']
        assert policy['variant']==f'dv{n}' and r['input_noise']==0 and r['output_noise']==.029
        assert drive==min(1.,max(0.,(policy['data_score']-3.)/3.))
        if policy['reopens']!=previous_reopens:
            events.append(dict(step=r['step'],reopens=policy['reopens'],data_score=policy['data_score'],mobility=mobility))
        previous_reopens=policy['reopens']; alignment=policy['alignment']
    for p in points:
        r=rates[p['step']-1]
        assert p['learning_rates']=={k:r[k] for k in ('generator_0','prior_1','critic_0')}
        assert p['policy']==r['policy'] and p['penalty']==r['penalty']
        assert p['input_noise']==0 and p['output_noise']==p['evaluation_output_noise']==.029
        for s in [p,p['ema']]+([p['frozen']] if 'frozen' in p else []):
            assert 0<=s['modes']<=8 and 0<=s['hq']<=1 and s['cover']==s['modes']/8 and s['n_modes']==8
            assert s['hq']*4096==int(s['hq']*4096)
    raw=[]
    for name,step in [('initial-state.pt',0),('change-2400-state.pt',2400),('final-state.pt',4600)]:
        ck=RawCheckpoint(d/name); e=ck.value; t=e['trainer']; raw.append(ck)
        assert e['schema']==1 and e['execution']==runtime['execution']
        assert e['identity']==dict(bundle_manifest_sha256=seal_hash,recipe=recipe,host=declared['host'],evaluation=declared['evaluation'])
        assert t['schema']==4 and t['completed_steps']==step and ck.material(t['recipe'])==recipe
        assert t['controller']['updates']==step and t['controller']['variant']==f'dv{n}'
        ck.clocks(step)
        assert ck.receipt()==(read(d/'initial.json') if not step else next(s for s in states if s['step']==step))
        if step:
            r=rates[step-1]
            for k,v in r['policy'].items(): assert t['controller'][k]==v
            assert [g['lr'] for o in t['optimizers'] for g in o['param_groups']]==[r['generator_0'],r['prior_1'],r['critic_0']]
    initial=raw[0]; t=initial.value['trainer']
    material=initial.material({k:v for k,v in t.items() if k not in ('streams','cpu_rng','cuda_rng','device')})
    assert material==read(d/'initial-material.json')['state']==cpu['all_initial_material']['state']
    assert read(d/'initial-material.json')==cpu['all_initial_material']
    cpu_models=RawCheckpoint(d/'initial-models-cpu.pt')
    assert cpu_models.material(cpu_models.value)=={k:initial.material(t['models'][k]) for k in ('G','D','prior')}
    assert all(read(d/'initialization-audit.json')['matches'].values())
    assert read(d/'initialization-audit.json')['adam_state_empty'] is True
    assert read(d/'initial-model-cpu-cuda-proof.json')==dict(status='PASS',all_initial_non_rng_state_and_named_parameters_and_buffers_equal=True)
    for label,step in [('main-0001',1),('frozen-2400',2400),('main-4600',4600)]:
        proof=read(d/f'optimizer-device-proof-{label}.json')
        assert [len(proof[k]) for k in ('generator','critic')]==[9,8]
        for entries in proof.values():
            for e in entries:
                assert e['step']==step and e['step_device']=='cpu' and e['step_shape']==[]
                assert e['parameter_device']==e['exp_avg_device']==e['exp_avg_sq_device']=='cuda:0'
                assert e['dtype']==e['step_dtype']==e['exp_avg_dtype']==e['exp_avg_sq_dtype']=='torch.float32'
    oldpoints=lines(old/'metrics.jsonl.gz'); oldstates=lines(old/'state-hashes.jsonl.gz'); oldrates=lines(old/'learning-rates.jsonl.gz')
    assert sha((old/'source.zip').read_bytes())==declared['historical_source_zip_sha256']
    assert sha((old/'declaration.json').read_bytes())==declared['historical_declaration_sha256']
    oldinitial=read(old/'initial.json'); newinitial=read(d/'initial.json')
    sampling_keys=('streams_sha256','cpu_rng_sha256','cuda_rng_sha256','real_stream_sha256','means_sha256')
    for key in sampling_keys: assert newinitial[key]==oldinitial[key]==declared['expected_initial_fixture'][key]
    assert newinitial['models_sha256']!=oldinitial['models_sha256']
    for x,y in zip(states,oldstates):
        assert x['step']==y['step']
        for key in sampling_keys: assert x[key]==y[key]
    for x,y in zip(rates,oldrates):
        assert x['step']==y['step']
        for key in ('data_score','data_drive','data_memory'): assert x['policy'][key]==y['policy'][key]
    oldderived=[segment(oldpoints,0,2400),segment(oldpoints,2400,4600)]
    dest=P/'supplementary-evidence'/(alias+'-single-shift-new-init')
    artifacts=archive(d,dest)
    row=dict(candidate=declared['candidate'],audit='PASS_ARTIFACT_SOURCE_RUNTIME_AND_DERIVED_METRICS',status='COMPLETE',
        assessment='Initial acquisition not observed within 2400; shifted arrival followed by uninterrupted observed retention through 4600. No automatic winner or irreversible nonconvergence claim.',
        source=str(d),archive=str(dest.relative_to(P)),source_zip_sha256=manifest['source.zip'],bundle_manifest_sha256=seal_hash,
        own_cpu_proof_sha256=peer['cpu_receipt_sha256'],old_source_zip_sha256=sha((old/'source.zip').read_bytes()),
        old_evidence=str(old),seconds=result['seconds'],segments=derived,old_segments=oldderived,
        comparison3600=summary['comparison3600'],frozen_control=summary['frozen_control'],controller_reopen_events=events,
        rate_ranges={k:[min(r[k] for r in rates),max(r[k] for r in rates)] for k in ('generator_0','prior_1','critic_0')},
        constant_applied_rates=False,horizon_driven_decay=False,constant_input_noise=0.,constant_output_noise=.029,
        verified=dict(package_files=24,observations=460,frozen_observations=220,rate_rows=4600,state_receipts=46,
            original_sampling_boundary_matches=46,real_detector_trace_matches_old_rows=4600,
            raw_checkpoints=[0,2400,4600],native_cpu_scalar_clocks_per_populated_checkpoint=17,
            actual_initial_non_rng_state_equals_own_cpu=True,actual_initial_models_differ_from_old=True,
            initial_actual_cpu_copy_equals_cuda=True,uninterrupted_main_run_source=True),
        limits=['No model, Torch, CUDA or training execution by this audit.',
            'Scores derived from all recorded frozen-evaluator observations; generated sample tensors were not retained for rescoring.',
            'Sampling proof covers initial and 46 actual state boundaries plus all 4600 detector statistics matching own old host; no per-update real/index batch receipts were retained.',
            'Frozen control has source assertions for unchanged main/caller RNG and unchanged frozen state at every observation; no separate final frozen checkpoint or fresh-process continuation was executed.',
            'No new-init stationary 7500 / long 30000 / full 22 qualification is supplied; tiny mode-hold continuation is a different task. Old quality is not inherited.',
            'Adaptive controller varies applied rates using real-data evidence and gradient alignment; this is not literal constant LR.'],
        artifact_manifest=artifacts)
    (dest/'archive-manifest.json').write_text(json.dumps(dict(source=str(d),artifacts=artifacts),indent=2)+'\n')
    return row


def main():
    rows=[audit(n) for n in (2,3)]
    output=dict(schema=1,scope='Standalone new-initialization ring followups; does not replace any original strict screen or historical ring score.',audit_status='PASS',results=rows)
    (P/'ring-followup-results.json').write_text(json.dumps(output,indent=2)+'\n')
    (Path(__file__).parent/'audit.json').write_text(json.dumps(output,indent=2)+'\n')
    table=[]
    for row in rows:
        for label,segs in [('Historical initialization',row['old_segments']),('New deterministic initialization',row['segments'])]:
            a,b=segs
            table.append(f"| {row['candidate']} | {label} | {a['first_arrival'] or 'not reached'} | {a['passing_since_arrival']}/{a['checks_since_arrival']} | {b['delay']} | {b['passing_since_arrival']}/{b['checks_since_arrival']} | {b['minimum_hq_since_arrival']:.6f} |")
    note='''# DV2 / DV3 completed ring followups\n\nBoth new-init runs finish all 4,600 updates. Neither reaches eight modes and HQ≥.90 before the target changes at 2,400. After the change, DV2 arrives in 380 updates and retains 183/183 observations; DV3 arrives in 500 and retains 171/171. No post-arrival departure occurs. These finite observations establish later adaptation and retention, but leave original-distribution acquisition unverified. No 81-observation deadline or claim of impossible eventual convergence is used.\n\n| Candidate | Initialization | Original arrival | Original retention since arrival | Shifted arrival delay | Shifted retention | Minimum shifted HQ |\n|---|---|---:|---:|---:|---:|---:|\n'''+ '\n'.join(table)+'''\n\nEach own historical run arrived on the original distribution at 790: DV2 then missed 1390, 1420, 1430; DV3 missed 1390–1440 and 1470–1490. Their old shifted arrivals were 400/520 updates, each followed by uninterrupted retained quality. New initialization preserves the sampled streams while changing the initial learned weights; the 20-update earlier shifted arrivals do not establish an overall win because initial acquisition regressed. Both frozen-at-change controls fail all 220 shifted observations.\n\nAll 24 isolated package files and the reviewed worker seal match. Actual raw initial model/EMA/prior and controller/optimizer state matches each own CPU proof; all 17 raw Adam clocks at 2,400/4,600 remain CPU scalars with CUDA moments. The main run is never reloaded; the separate frozen copy restores constructor-modified global RNG and is guarded against main/caller state changes. All 460 scores,4,600 adaptive-rate/noise rows, 46 original sampling boundaries and initial/change/final raw state receipts verify. Detector evidence traces also match each own historical run at all 4,600 steps.\n\nRates vary with data evidence and gradient alignment; total_steps=None and constant output noise. These are horizon-free adaptive rates, not literal constant rates. The source-only audit does not regenerate samples, invent missing per-update batch receipts or prove fresh-process continuation. Stationary/long/full22 qualification remains absent on this initialization. No additional runs are authorized by this report.\n\nAll terminal files, including the three raw checkpoints, are archived losslessly under supplementary-evidence/api-dv{2,3}-single-shift-new-init. [Standalone manifest](../ring-followup-results.json) and [full audit](audit.json) retain exact hashes and limits. Existing score manifests and original results remain unchanged.\n'''
    (Path(__file__).parent/'audit.md').write_text(note)
    print(json.dumps({r['candidate']:{'audit':r['audit'],'segments':[{k:v for k,v in s.items() if k!='final'} for s in r['segments']]} for r in rows},indent=2))


if __name__=='__main__': main()
