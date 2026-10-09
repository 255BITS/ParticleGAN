"""Reconstruct frozen data draws and verify BOTH consumed data RNG states.

No model construction, optimizer updates, or mutation of trained streams.
This explicit diagnostic supplies sequence hashes where original adapters
retained only named stream states, not a per-batch tensor trace.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.rng import NamedStreams
from experiments.forge.contracts import atomic_json,read_json,file_hash
from benchmarks.toy100.problems import sample_real
from benchmarks.transfer_suite.vector_tasks import sample_target

OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/queue')

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=read_json(QUEUE/'queue/state.json');requests=read_json('/tmp/bcap-physics-round4-20261009/native_overshoot/progress.json')['requests']
    proofs=[]
    for taskid in ('grid100','vector_two_broad'):
        started=time.monotonic();final=[];request=None
        for role,rid in requests.items():
            request=state['submissions'][rid]['request']
            job=next(j for j in state['jobs'].values() if rid in j['subscribers'] and j['definition']['task_id']==taskid)
            descriptor=job['result']['task_results'][0]['evidence']['provenance_checkpoint']
            path=Path(descriptor['artifact_root'])/descriptor['path'];assert file_hash(path)==descriptor['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            device=saved['trainer']['device'] if taskid=='grid100' else 'cpu'
            key=json.dumps(['data','target','training',device],separators=(',',':'))
            final.append(dict(role=role,state=saved['streams']['states'][key],checkpoint=descriptor['sha256']))
        for source in ('experiments/forge/rng.py','benchmarks/toy100/problems.py','benchmarks/transfer_suite/vector_tasks.py'):
            assert file_hash(ROOT/source)==request['source']['files'][source]
        stream=NamedStreams(0,device=device).generator('data',component='target',purpose='training')
        initial=stream.get_state().clone();task=request['tasks'][taskid]
        n=task['execution']['resources']['batch_size'] if taskid=='grid100' else task['execution']['host_definition']['batch']
        digest=hashlib.sha256()
        for step in range(task['execution']['steps']):
            batch=(sample_real(taskid,n,device=device,generator=stream) if taskid=='grid100'
                   else sample_target(task['execution']['host_definition'],n,stream,step))
            digest.update(batch.detach().cpu().contiguous().numpy().tobytes())
        assert all(torch.equal(stream.get_state(),f['state']) for f in final)
        proofs.append(dict(task_id=taskid,steps=task['execution']['steps'],batch_size=n,
            reconstructed_batch_sequence_sha256=digest.hexdigest(),both_actual_data_stream_final_states_match=True,
            initial_rng_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
            final_rng_sha256=hashlib.sha256(stream.get_state().numpy().tobytes()).hexdigest(),
            stream=key,checkpoint_sha256={f['role']:f['checkpoint'] for f in final},elapsed_seconds=time.monotonic()-started))
    atomic_json(OUT/'batch-sequence-replay.json',dict(schema_version=1,qualification_input=False,
        scope='Deterministic reconstruction from unchanged frozen samplers, schedules and named seeds; checked against both actual consumed RNG states; original per-batch tensors were not retained.',
        source_digest=request['source']['digest'],optimizer_updates_added=0,trained_rng_mutations=0,reservation_seconds=120,proofs=proofs))
    print(json.dumps(proofs),flush=True)

if __name__=='__main__':main()
