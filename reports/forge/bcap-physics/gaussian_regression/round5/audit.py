"""Read-only source, certificates, original controls and actual stream proof."""
from pathlib import Path
import json,hashlib,subprocess,time
import torch
from experiments.forge.contracts import atomic_json,read_json,file_hash,stable_hash
from experiments.forge.state import state_digest
from experiments.forge.tier1_media import _scored_outputs
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/queue')
PROGRESS=Path('/tmp/bcap-physics-round5-20261009/gaussian_regression/progress.json')
OLD=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_transport/queue')
OLD_REQUESTS={'local':'574010f5dd3494e037142738','finite':'e435328afce2e594bf7e9622'}

def load(local,task):
    result=read_json(local/'result.json');row={**next(r for r in result['task_results'] if r['task_id']==task['id']),'attempt_id':result['attempt_id']}
    desc=row['evidence']['provenance_checkpoint'];path=Path(desc['artifact_root'])/desc['path']
    assert file_hash(path)==desc['sha256'];saved=torch.load(path,map_location='cpu',weights_only=False)
    arrays,_=_scored_outputs(task,row['evidence'],local)
    return row,saved,arrays,desc

def models_streams(saved):
    return dict(models=saved['trainer']['models'],optimizers=saved['trainer']['optimizers'],
                streams=saved['streams'],trainer_streams=saved['trainer']['streams'])

def compact_rng_audits(audits):
    assert isinstance(audits,list) and all(a['unintended_rng_deviations']==0 and not a['unintended_streams'] for a in audits)
    return dict(checks=len(audits),original_sha256=stable_hash(audits),unintended_rng_deviations=0,changed_streams=sorted({k for a in audits for k in a['changed_streams']}))

def numerical_records(value):
    if isinstance(value,dict):return {k:numerical_records(v) for k,v in value.items() if k not in ('training_state_sha256','primary_state_sha256','confirmed_state_sha256')}
    if isinstance(value,(tuple,list)):return type(value)(numerical_records(v) for v in value)
    return value

def data_proof(task,row,saved):
    recorded=row['evidence'].get('data_sha256')
    if recorded is not None:return dict(sha256=recorded,method='recorded actual Gaussian batches',replay_batches=0)
    from benchmarks.transfer_suite.vector_tasks import sample_target
    from experiments.forge.vectorprofiles import resolve_vector_spec
    key='["data","target","training","cpu"]'
    binding=row['rng']['bindings'][key]
    generator=torch.Generator(device='cpu').manual_seed(binding['seed'])
    assert hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()==binding['initial_state_sha256']
    digest=hashlib.sha256();spec=resolve_vector_spec(task)
    for step in range(task['execution']['steps']):
        real=sample_target(spec,spec['batch'],generator,step)
        digest.update(real.numpy().tobytes())
    assert torch.equal(generator.get_state(),saved['streams']['states'][key])
    return dict(sha256=digest.hexdigest(),method='disposable replay of frozen one-real-batch-per-update law; exact initial/final data stream proof',replay_batches=task['execution']['steps'])

def main():
    start=time.monotonic();torch.set_num_threads(1)
    state=read_json(QUEUE/'queue/state.json');requests=read_json(PROGRESS)['requests'];proofs=[];current={};old=read_json(OLD/'queue/state.json')
    allsources=[];source_checks=[]
    for role,rid in requests.items():
        request=state['submissions'][rid]['request'];source=request['source'];allsources.append(source)
        assert state['submissions'][rid]['status'] not in ('queued','running','paused')
        for relative,expected in source['files'].items():
            snapshot=Path(source['snapshot_path'])/relative;assert file_hash(snapshot)==expected
            if relative.startswith(('particlegan/','benchmarks/','experiments/','configs/forge/')):
                assert file_hash(ROOT/relative)==expected,(relative,'scientific source changed')
                source_checks.append(relative)
        for job in state['jobs'].values():
            if rid not in job['subscribers']:continue
            assert job['status']=='terminal' and job.get('result'),job['definition']['task_id']
            local=Path(job['attempts'][-1]['path']);task=request['tasks'][job['definition']['task_id']]
            row,saved,arrays,desc=load(local,task);current[(role,task['id'])]=(row,saved,arrays,desc)
            aid=job['result']['attempt_id'];durable=ROOT/'reports/forge/attempts'/aid
            certificate=read_json(durable/'evidence.json');result=read_json(durable/'result.json');envelope=read_json(durable/'request.json')
            assert certificate['result_hash']==stable_hash(result) and certificate['source']==source
            assert read_json(local/'result.json')==result
            streams=dict(context=state_digest(saved['streams']),trainer=state_digest(saved['trainer']['streams']))
            assert saved['initialization']==row.get('initialization',row.get('applied',{}).get('initialization'))
            proofs.append(dict(role=role,task_id=task['id'],attempt_id=aid,gate_status=row['gate_status'],
                initial_model_hashes=saved['initialization'],data_sha256=data_proof(task,row,saved),
                actual_consumed_stream_sha256=streams,checkpoint_path=str(Path(desc['artifact_root'])/desc['path']),checkpoint_sha256=desc['sha256'],
                receipt_files={n:dict(path=str((durable/(n+'.json')).relative_to(ROOT)),sha256=file_hash(durable/(n+'.json'))) for n in ('request','result','evidence')},
                rng_audits=compact_rng_audits(row['evidence']['rng_audits']),same_real_tensor_D_G=True))
    assert all(s==allsources[0] for s in allsources)
    matched=[]
    for task_id in sorted({p['task_id'] for p in proofs}):
        group=[p for p in proofs if p['task_id']==task_id]
        assert len(group)==3
        for key in ('initial_model_hashes','data_sha256','actual_consumed_stream_sha256'):
            assert all(p[key]==group[0][key] for p in group), (task_id,key)
        matched.append(dict(task_id=task_id,arms=[p['role'] for p in group],initial_models_data_and_consumed_streams_equal=True))
    parity=[]
    for role,rid in OLD_REQUESTS.items():
        request=old['submissions'][rid]['request']
        for job in old['jobs'].values():
            tid=job['definition']['task_id']
            if rid not in job['subscribers'] or (role,tid) not in current:continue
            row,saved,arrays,desc=load(Path(job['attempts'][-1]['path']),request['tasks'][tid]);new=current[(role,tid)]
            equal=dict(metrics=new[0]['metrics']==row['metrics'],observations=new[0]['evidence']['observations']==row['evidence']['observations'],
                samples=state_digest(numerical_records(new[2]))==state_digest(numerical_records(arrays)),model_optimizer_stream_tensors=state_digest(models_streams(new[1]))==state_digest(models_streams(saved)))
            assert all(equal.values()),(role,tid,equal)
            parity.append(dict(role=role,task_id=tid,archived_attempt_id=job['result']['attempt_id'],new_attempt_id=new[0]['attempt_id'],
                archived_source_digest=request['source']['digest'],archived_checkpoint_sha256=desc['sha256'],equal=equal,qualification_credit=False))
    preserved=read_json(OUT/'preserved-inputs.json')
    assert all(file_hash(ROOT/p)==sha for p,sha in preserved.items())
    gifs=read_json(OUT/'media/index.json')['media']
    assert len(gifs)==15
    for item in gifs:assert file_hash(OUT/item['gif'])==item['gif_sha256']
    for file,commit in [('particlegan/optim/direction_blend.py','bfb06c2083cbc41c301797350cfb429776da59ee'),('particlegan/optim/strict_progress.py','60944075e353834df313c74f1adf6324d94f2811'),('particlegan/kinetic_transport.py','60944075e353834df313c74f1adf6324d94f2811')]:
        assert (ROOT/file).read_bytes()==subprocess.check_output(['git','show',commit+':'+file])
    atomic_json(OUT/'audit.json',dict(schema_version=1,qualification_input=False,source_commit=allsources[0]['origin_commit'],source_digest=allsources[0]['digest'],
        source_snapshot_files_verified=len(allsources[0]['files']),current_scientific_files_verified=len(set(source_checks)),
        proofs=proofs,matched_conditions=matched,archived_control_parity=parity,excluded_unused_ambient_state=['trainer.cpu_rng','trainer.cuda_rng'],excluded_opaque_trace_fields=['training_state_sha256','primary_state_sha256','confirmed_state_sha256'],exclusion_reason='Opaque hashes include unused ambient states; all numeric outputs/metrics and actual model/optimizer/consumed streams compared' ,original_task_and_qualification_snapshots_preserved=len(preserved),
        actual_training_gifs=15,optimizer_updates_added=0,served_model_sampling_draws_added=0,disposable_data_replay_batches=sum(p['data_sha256']['replay_batches'] for p in proofs),elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(proofs=len(proofs),matched=len(matched),archived_bitwise_controls=len(parity),preserved=len(preserved))))
if __name__=='__main__':main()
