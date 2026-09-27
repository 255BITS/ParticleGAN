"""Independent stdlib image artifact/state audit. No Torch or model execution."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,io,json,math,struct,zipfile
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('raw',ROOT/'audit_public3_runtime.py')
raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
lines=lambda p:[json.loads(line) for line in p.read_text().splitlines()]

def load(path):
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')];assert len(names)==1
        prefix=names[0][:-8];assert archive.read(prefix+'byteorder')==b'little'
        d=raw.Reader(io.BytesIO(archive.read(names[0]))).load()
        def binary(t):
            kind,key,device,count=t.storage;width={'FloatStorage':4,'ByteStorage':1}[kind]
            b=archive.read(prefix+'data/'+key);assert len(b)==count*width
            stride=1
            for length,value in reversed(list(zip(t.shape,t.stride))):
                assert length<=1 or value==stride;stride*=length
            return b[t.offset*width:(t.offset+math.prod(t.shape))*width]
        def material(v):
            if isinstance(v,raw.Tensor):return dict(shape=list(v.shape),dtype={'FloatStorage':'torch.float32','ByteStorage':'torch.uint8'}[v.storage[0]],device=v.storage[2],sha256=sha(binary(v)))
            if isinstance(v,dict):return {str(k):material(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [material(x) for x in v]
            assert v is None or isinstance(v,(str,int,float,bool)),type(v)
            return v
        clocks=[]
        for o in d['trainer']['optimizers']:
            ids=[p for g in o['param_groups'] for p in g['params']];assert set(ids)==set(o['state'])
            role=[]
            for i in ids:
                v=o['state'][i];t=v['step'];assert t.shape==() and t.storage[0]=='FloatStorage'
                assert v['exp_avg'].storage[2]==v['exp_avg_sq'].storage[2]
                role.append(dict(step=struct.unpack('<f',binary(t))[0],device=t.storage[2],moment_device=v['exp_avg'].storage[2]))
            clocks.append(role)
        return material(d),clocks

def no_devices(v):
    if isinstance(v,dict):
        tensor={'shape','dtype','device','sha256'}<=v.keys()
        return {k:no_devices(x) for k,x in v.items() if not (tensor and k=='device')}
    if isinstance(v,list):return [no_devices(x) for x in v]
    return v

def summarize(rows,spec):
    good=lambda r:r['modes']>=spec['thresholds']['modes'] and r['hq']>=spec['thresholds']['hq_min']
    first=next((r['step'] for r in rows if good(r)),None);after=[r for r in rows if first is not None and r['step']>=first]
    suffix=[]
    for r in reversed(rows):
        if not good(r):break
        suffix.append(r)
    return dict(observations=len(rows),passing_observations=sum(map(good,rows)),passing_suffix=len(suffix),first_arrival=first,departures=[r['step'] for r in after if not good(r)],passing_since_arrival=sum(map(good,after)),observations_since_arrival=len(after),minimum_modes_since_arrival=min((r['modes'] for r in after),default=None),minimum_hq_since_arrival=min((r['hq'] for r in after),default=None),final_suffix_start=suffix[-1]['step'] if suffix else None)

def audit(source,name,task):
    artifact=read(source/'artifact-sha256.json')
    assert {p.name:sha(p.read_bytes()) for p in source.iterdir() if p.is_file() and p.name!='artifact-sha256.json'}==artifact
    declaration=read(source/'declaration.json');candidate=read(ROOT/'port-source'/name/'candidate-declaration.json')
    assert declaration['candidate']==candidate
    harness=ROOT/'port-source/new-init-image-screen-logging-v2';seal=read(harness/'bundle-sha256.json')
    expected={**candidate['package_sha256'],**seal,'bundle-sha256.json':sha((harness/'bundle-sha256.json').read_bytes())}
    assert declaration['source_sha256']==expected
    with zipfile.ZipFile(source/'source.zip') as z:
        assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(expected)|{'candidate-declaration.json'}
        assert {n:sha(z.read(n)) for n in expected}==expected
        assert json.loads(z.read('candidate-declaration.json'))==candidate
    cpu_path=ROOT/'image-screen-review'/(name+'-'+('intensity2' if task=='img_intensity2' else task)+'-cpu')/'cpu-preflight.json';cpu=read(cpu_path)
    assert cpu['status']=='PASS_CPU_ZERO_STEP' and cpu['cuda_initialized'] is False
    assert cpu['source_sha256']==expected and cpu['task']==task
    assert cpu['candidate_declaration_sha256']==sha((ROOT/'port-source'/name/'candidate-declaration.json').read_bytes())
    result=read(source/'result.json');spec=read(harness/'task-specs.json')[task]
    assert declaration['task']==spec and result['status'] in ('PASS','FAIL') and result['completed_steps']==spec['steps']==600
    assert result['cpu_receipt_sha256']==sha(cpu_path.read_bytes())
    recipe=dict(candidate['resolved_recipe']);recipe.update(z_dim=8,num_particles=32,batch_size=32)
    assert result['recipe']==recipe
    for mod,entry in result['imports'].items():
        path=Path(entry['path']);key='particlegan/'+path.name
        assert entry['sha256']==candidate['package_sha256'][key]==sha(path.read_bytes()),mod
    initial,clocks0=load(source/'initial-state.pt');final,clocksN=load(source/'final-state.pt')
    assert initial==read(source/'initial.json')
    assert initial['trainer']['completed_steps']==0 and final['trainer']['completed_steps']==600
    assert initial['trainer']['recipe']==final['trainer']['recipe']==recipe
    proof=read(source/'initial-model-cpu-cuda-proof.json');assert proof['status']=='EXACT_ALL_MODEL_BYTES'
    assert initial['trainer']['models']==proof['models']
    assert no_devices(proof['models'])==no_devices(cpu['initial_material']['models'])
    for c in (clocks0,clocksN):assert len(c)==2 and [len(x) for x in c]==[len(cpu['optimizer'][x]) for x in ('G','D')]
    for clocks,step in ((clocks0,0),(clocksN,600)):
        assert all(c==dict(step=float(step),device='cuda:0',moment_device='cuda:0') for role in clocks for c in role)
    for file,step in [('optimizer-initial.json',0),('optimizer-final.json',600)]:
        op=read(source/file)
        assert [len(op[x]) for x in ('G','D')]==[len(c) for c in clocksN]
        for role in op.values():
            for r in role:assert r['step']==step and r['step_device']==r['parameter_device']=='cuda:0'
    batches=read(source/'batch-receipts.json');assert [r['step'] for r in batches]==list(range(1,601))
    assert final['data_rng']==batches[-1]['accepted_cursor']
    assert final['global_cuda_rng']==[final['data_rng']]
    assert initial['global_cuda_rng']==[initial['data_rng']]
    assert final['trainer']['streams']['latent_generator']==final['data_rng']
    rates=lines(source/'learning-rates.jsonl');assert [r['step'] for r in rates]==list(range(1,601))
    for row in rates:
        assert row['input_noise']==0. and row['output_noise']==row['evaluation_output_noise']==.029
        assert row['game_stats']['accepted_updates']==row['step']
        assert row['game_stats']['policy']==recipe['game_update']
        assert row['game_stats']['field_evaluations']==2
        assert row['precision']['updates']==row['step']
        for r in row['rates']:
            assert all(math.isfinite(v) and v>0 for v in r)
    assert final['trainer']['precision']['state']==rates[-1]['precision']
    observations=lines(source/'metrics.jsonl');assert [r['step'] for r in observations]==list(range(25,601,25))
    assert result['final']==observations[-1]
    summary=summarize(observations,spec);conv=result['convergence']
    for k in ('observations','passing_observations','passing_suffix'):assert conv[k]==summary[k]
    assert conv['complete'] is True and conv['minimum_stable_checks']==5 and conv['first_pass_step']==summary['first_arrival']
    assert result['status']==('PASS' if summary['passing_suffix']>=5 else 'FAIL')
    historical=ROOT.parent/'continuous-api-search/evidence'/(name+'-'+task)/'result.json'
    old={'status':'NOT_MEASURED_ON_THIS_EXACT_IMAGE'}
    if historical.exists():
        old_result=read(historical);old=dict(status=old_result['status'],source=str(historical),source_sha256=sha(historical.read_bytes()),old_initialization=True)
    events=[dict(step=r['step'],event=r['precision']['event'],open=r['precision']['open']) for r in rates if r['precision']['event'] is not None]
    return dict(status='PASS',scope='Independent stdlib source, raw-state and scoring audit; no Torch or training',quality_status=result['status'],candidate=result['candidate'],task=task,source=str(source),**summary,final=result['final'],seconds=result['seconds'],old_result=old,source_zip_sha256=artifact['source.zip'],cpu_receipt_sha256=result['cpu_receipt_sha256'],optimizer_clocks=clocksN,actual_CUDA_model_bytes_equal_own_CPU_proof=True,initial_raw_state_matches_full_initial_receipt=True,full_dry_sampling_receipts=600,final_shared_sampling_cursor_matches=True,applied_rate_and_controller_rows=600,field_evaluations=sum(r['game_stats']['field_evaluations'] for r in rates),precision_events=events,final_precision=rates[-1]['precision'],artifacts=artifact,limits=['The sealed worker checked all actual real/latent draws and accepted shared cursors against retained600 dry receipts. Audit verifies source assertions, expected receipts and final cursor; it does not rerun sampling.','Each accepted public update includes two game fields; cost is retained without equal-compute speed claims.','This finite image score does not establish ring recovery, indefinite retention, checkpoint replay or full22 qualification.'],audit_source_sha256=sha(Path(__file__).read_bytes()))

def archive(source,out,audit):
    out.mkdir(exist_ok=False,parents=True);copied={};checkpoints={}
    for p in sorted(source.iterdir()):
        if not p.is_file():continue
        b=p.read_bytes()
        if p.suffix=='.pt':checkpoints[p.name]=dict(path=str(p),sha256=sha(b),bytes=len(b));continue
        compress=p.suffix=='.jsonl' or (p.suffix=='.json' and len(b)>100000)
        payload=gzip.compress(b,mtime=0) if compress else b;target=out/(p.name+'.gz' if compress else p.name);target.write_bytes(payload)
        assert (gzip.decompress(payload) if compress else payload)==b
        copied[p.name]=dict(path=str(target.relative_to(ROOT)),sha256=sha(payload),original_sha256=sha(b),bytes=len(payload))
    receipt=dict(candidate=audit['candidate'],task=audit['task'],quality_status=audit['quality_status'],source_directory=str(source),artifacts=copied,raw_checkpoints=checkpoints,audit_sha256=sha(json.dumps(audit,indent=2).encode()+b'\n'))
    (out/'archive-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return str((out/'archive-manifest.json').relative_to(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--candidate',choices=['api-rp12','api-rp14','api-rp15'],required=True);p.add_argument('--task',required=True,choices=['img_intensity2','img_bars4','img_blobs4','img_stripes2']);p.add_argument('--archive',action='store_true');a=p.parse_args()
    result=audit(a.source.resolve(),a.candidate,a.task);out=Path(__file__).parent/(a.candidate+'-'+a.task+'-runtime-audit.json');out.write_text(json.dumps(result,indent=2)+'\n')
    if a.archive:
        tablepath=ROOT/'followup-results.json';table=read(tablepath);assert not any(r['candidate']==result['candidate'] and r['task']==result['task'] for r in table['results'])
        manifest=archive(a.source.resolve(),ROOT/'followup-evidence'/(a.candidate+'-'+a.task.replace('_','-')+'-new-init'),result)
        table['results'].append({**result,'archive_manifest':manifest,'audit':str(out),'audit_sha256':sha(out.read_bytes())});tablepath.write_text(json.dumps(table,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','candidate','quality_status','passing_observations','passing_suffix','first_arrival')}))

if __name__=='__main__':main()
