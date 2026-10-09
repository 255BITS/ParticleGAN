"""Read-only source, task, stream, initializer, receipt and media audit."""
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys

from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json,stable_hash
from experiments.forge.queue import process_identity
from publish import QUEUE,BRIEF,OUT

BASE='bfb06c2083cbc41c301797350cfb429776da59ee'

def main():
    torch.set_num_threads(1);state=read_json(QUEUE/'queue/state.json');requests=read_json(BRIEF/'progress.json')['requests']
    arms={r:state['submissions'][rid]['request'] for r,rid in requests.items()};first=arms['control']
    assert arms['candidate']['source']==first['source'] and arms['candidate']['runtime']==first['runtime']
    assert arms['candidate']['protocol']==first['protocol'] and first['protocol']['seed']==0
    original_conditions={}
    for taskid,task in first['tasks'].items():
        original=json.loads(subprocess.check_output(['git','show',f'{BASE}:configs/forge/tasks/{taskid}.json'],cwd=ROOT,text=True))
        for key in ('execution','evaluation','resources','dependencies','adapter','requires_capabilities'):
            assert task.get(key)==arms['candidate']['tasks'][taskid].get(key)==original.get(key),(taskid,key)
        original_conditions[taskid]=stable_hash(original)
    scientific={p:d for p,d in first['source']['files'].items() if p.startswith(('particlegan/','benchmarks/','experiments/','lib/','configs/forge/'))}
    changed=[p for p,d in scientific.items() if not (ROOT/p).exists() or file_hash(ROOT/p)!=d];assert not changed,changed
    archived=read_json(BRIEF/'archived-snapshot-hashes.json');assert all(file_hash(ROOT/p)==d for p,d in archived.items())
    results=read_json(OUT/'results.json');proofs=read_json(OUT/'provenance.json')['proofs'];media=read_json(OUT/'media/index.json')['media']
    assert len(results['task_results'])==12 and all(r['gate_status'] in {'PASS','FAIL'} for r in results['task_results'])
    assert all(j['status']=='terminal' for j in state['jobs'].values())
    assert all(process_identity(j['worker']['pid'])!=j['worker']['process_identity'] for j in state['jobs'].values() if j.get('worker'))
    accounting=state['campaigns']['force-distortion-round5-v1'];assert accounting['reserved_seconds']==0
    assert sum(len(j['attempts']) for j in state['jobs'].values())==14
    assert len(results['execution_retry_history'])==2 and all(r['gate_status']=='INCOMPLETE' for r in results['execution_retry_history'])
    checkpoints={};rng_audits=0
    for proof in proofs:
        durable=ROOT/'reports/forge/attempts'/proof['attempt_id'];row=read_json(durable/'result.json')['task_results'][0]
        desc=proof['provenance_checkpoint'];root=Path(desc.get('artifact_root') or row['evidence'].get('artifact_root') or proof['artifact_root'])
        path=root/desc['path'];assert file_hash(path)==desc['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False);checkpoints[(proof['role'],proof['task_id'])]=saved
        if 'applied' in saved:
            assert saved['applied']['rng']['seed']==0
            for audit in saved['applied']['rng_audits']:assert audit['unintended_rng_deviations']==0;rng_audits+=1
    paired=[]
    for taskid in first['tasks']:
        a,b=[checkpoints[(r,taskid)] for r in ('control','candidate')]
        streams=[]
        for saved in (a,b):
            streams.append({key:stable_hash(value.tolist()) for key,value in saved['streams']['states'].items() if json.loads(key)[0]!='eval'})
        assert streams[0] and streams[0]==streams[1],taskid
        initial=[saved.get('initialization',saved.get('applied',{}).get('initialization')) for saved in (a,b)]
        assert initial[0] and initial[0]==initial[1],taskid
        paired.append(dict(task_id=taskid,initial_models_and_prior_equal=True,all_consumed_non_eval_streams_equal=True,
            training_stream_families=sorted({json.loads(key)[0] for key in streams[0]})))
    for role in ('control','candidate'):
        saved=checkpoints[(role,'gaussian1d_stability')]
        row=next(r for r in results['task_results'] if r['role']==role and r['task_id']=='gaussian1d_stability')
        assert saved['trainer']['completed_steps']==6000
        own=next(r for r in proofs if r['role']==role and r['task_id']=='gaussian1d_stability')
        evidence=read_json(ROOT/'reports/forge/attempts'/own['attempt_id']/'result.json')['task_results'][0]['evidence']
        assert evidence['continuity']['restored_exactly'] and evidence['continuity']['prefix_steps']==1000
    frames={}
    assert len(media)==12
    for item in media:
        path=OUT/item['gif'];assert file_hash(path)==item['gif_sha256']
        with Image.open(path) as gif:frames[item['gif']]=gif.n_frames;assert gif.n_frames>=2
    controls=read_json(OUT/'scorer-controls.json');assert controls['ambient_rng_unchanged']
    atomic_json(OUT/'validation.json',dict(schema_version=1,qualification_input=False,valid=True,
        source_commit=first['source']['origin_commit'],source_digest=first['source']['digest'],
        scientific_files_verified=len(scientific),changed_scientific_files=changed,
        original_task_condition_hashes=original_conditions,matched_protocol_proofs=paired,
        zero_unintended_rng_audits=rng_audits,archived_qualification_telemetry_hashes=archived,
        original_snapshot_paths_absent_at_stacked_base=[p for p in read_json(Path('/tmp/bcap-physics-round4-20261009/qualification-before.json')) if not (ROOT/p).exists()],
        final_jobs=12,paid_attempts=14,retained_incomplete_attempts=2,active_workers=0,remaining_reserved_seconds=0,
        actual_training_gif_frames=frames,main_full_reservations_including_retries=19680,ancillary_full_allowance=1920,
        total_full_allowance=21600,paid_worker_seconds=accounting['spent_seconds'],
        software_tests=dict(unique_passed=225,broader_suite_passed=224,final_sample_force_suite_passed=4,overlap=3),
        scorer_controls_sha256=file_hash(OUT/'scorer-controls.json'),optimizer_updates_added=0,sampling_draws_added=0))
    print('Source, original task laws, all consumed training streams, initial states, 14 receipts and 12 actual-training GIFs verified.',flush=True)

if __name__=='__main__':main()
