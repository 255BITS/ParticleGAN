"""Restore certified round-four Gaussian states; inspect saved curves, add no updates."""
from pathlib import Path
import json,time
import torch
from experiments.forge.contracts import atomic_json,read_json,file_hash
from experiments.forge.gaussian_tasks import build,bounds,grade
from experiments.forge.state import state_digest

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
ARCHIVE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_transport/queue')
REQUESTS={'transport':'574010f5dd3494e037142738','finite':'e435328afce2e594bf7e9622'}

def find_stats(value):
    if isinstance(value,dict):
        return ({k:value[k]['stats'] for k in ('constraint_geometry','strict_progress') if k in value}
                or {k:v for child in value.values() for k,v in find_stats(child).items()})
    if isinstance(value,(list,tuple)):
        return {k:v for child in value for k,v in find_stats(child).items()}
    return {}

def main():
    started=time.monotonic();torch.set_num_threads(1)
    state=read_json(ARCHIVE/'queue/state.json');rows=[]
    for role,rid in REQUESTS.items():
        request=state['submissions'][rid]['request']
        for job in state['jobs'].values():
            task_id=job['definition']['task_id']
            if rid not in job['subscribers'] or task_id not in ('gaussian1d_smoke','gaussian1d_stability'):continue
            local=Path(job['attempts'][-1]['path']);result=read_json(local/'result.json');row=result['task_results'][0];task=request['tasks'][task_id]
            certificate=read_json(Path('/home/martyn/dev/ParticleGAN-bcap-r4-projection_transport/reports/forge/attempts')/result['attempt_id']/'evidence.json')
            assert certificate['source']==request['source']
            descriptor=row['evidence']['provenance_checkpoint'];path=Path(descriptor['artifact_root'])/descriptor['path']
            assert file_hash(path)==descriptor['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False)
            with torch.random.fork_rng(devices=[0]):
                context,trainer,_=build(request,task,'cuda:0',max_steps=6000 if task_id.endswith('stability') else 1000)
                context.load_state_dict(saved)
                assert state_digest(context.state_dict())==state_digest(saved)
            evidence=row['evidence'];assert grade(task,evidence)['gate_status']==row['gate_status']
            curve=evidence['observations']; chunks=[curve[i:i+24] for i in range(0,len(curve),24)]
            item=dict(role=role,task_id=task_id,attempt_id=result['attempt_id'],source={k:request['source'][k] for k in ('origin_commit','digest')},
                certificates={n:file_hash(local/(n+'.json')) for n in ('request','result')},
                checkpoint_path=str(path),checkpoint_sha256=descriptor['sha256'],exact_restore=True,
                optimizer_steps=trainer.completed_steps,optimizer_stats=find_stats(saved),
                gate_status=row['gate_status'],grade=grade(task,evidence),continuity=evidence.get('continuity'),
                passing_per_1000=[sum(not bounds(r) for r in chunk) for chunk in chunks],
                failing_bounds={name:sum(any(f.startswith(name+' ') for f in bounds(r)) for r in curve) for name in ('mean_error_sigma','std_ratio','cdf_ks')},
                endpoint_metrics=row['metrics'],
                temporal_activity_available=False,note='Archive retains endpoint aggregate optimizer counters and scheduled quality curve, not per-update conflict/scale events. No event-to-quality causal attribution.')
            if task_id.endswith('stability'):
                # Smoke state restored by the original protocol is its completed 1000-update state.
                smoke=next(j for j in state['jobs'].values() if rid in j['subscribers'] and j['definition']['task_id']=='gaussian1d_smoke')
                sr=read_json(Path(smoke['attempts'][-1]['path'])/'result.json')['task_results'][0]
                item['own_smoke_first_confirmation']=grade(request['tasks']['gaussian1d_smoke'],sr['evidence'])['evaluator_result']['first_confirmed_step']
                item['own_smoke_endpoint']=sr['metrics']
            rows.append(item)
    receipt=dict(schema_version=1,qualification_input=False,scope='saved-state Gaussian diagnostic',
        optimizer_updates_added=0,sampling_draws_added=0,allowance_seconds=120,elapsed_seconds=time.monotonic()-started,rows=rows)
    assert receipt['elapsed_seconds']<120
    atomic_json(OUT/'prior-evidence.json',receipt)
    print(json.dumps([{k:r[k] for k in ('role','task_id','gate_status','optimizer_stats','passing_per_1000')} for r in rows],indent=2))
if __name__=='__main__':main()
