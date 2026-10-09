"""Freeze two ready arms before bounded, shared-GPU public Queue/drain execution."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import sys
from threading import Event,Thread

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import particlegan
from experiments.forge.planning import resolve_idea,plan_summary
from experiments.forge.queue import Queue,drain
from experiments.forge.contracts import atomic_json,read_json

QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/force_distortion/queue')
BRIEF=Path('/tmp/bcap-physics-round5-20261009/force_distortion')
HANDOFF=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/handoff/force_distortion')
CAMPAIGN='force-distortion-round5-v1'

def log(**values):print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),**values)),flush=True)

def main():
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT)
    queue=Queue(QUEUE,report_root=ROOT/'reports/forge',on_completion=None);requests={};sources=[];reservation=0
    for role in ('control','candidate'):
        request=resolve_idea(ROOT,f'force-distortion-round5-{role}-v1',queue_root=QUEUE,freeze_source=True,
            study=f'force-distortion-round5-{role}-study-v1')
        plan=plan_summary(request,queue.inspect(),include_ownership=True)
        atomic_json(BRIEF/f'{role}-plan.json',plan)
        reservation+=plan['worst_case_seconds'];assert reservation<=19440
        entry=queue.submit(request,request['study']['campaign']);requests[role]=entry['request']['request_id']
        sources.append(request['source'])
        log(phase='enqueued',role=role,request_id=requests[role],source_digest=request['source']['digest'],reservation=plan['worst_case_seconds'])
    assert sources[0]==sources[1]
    progress=read_json(BRIEF/'progress.json');progress.update(phase='training',requests=requests,
        sourcecommit=sources[0]['origin_commit'],source_commit=sources[0]['origin_commit'],digest=sources[0]['digest'],source_digest=sources[0]['digest'])
    atomic_json(BRIEF/'progress.json',progress);atomic_json(HANDOFF/'progress.json',progress)
    log(phase='draining',pid=os.getpid(),python=sys.version.split()[0],module=particlegan.__file__,gpus=['0','1'],sharing=True)
    stopped=Event()
    def heartbeat():
        while not stopped.wait(30):
            state=queue.inspect()
            jobs=[j for j in state['jobs'].values() if any(rid in j['subscribers'] for rid in requests.values())]
            log(phase='progress',jobs=[dict(task=j['definition']['task_id'],status=j['status'],
                candidate=j['definition'].get('candidate_id'),attempt=j.get('worker',{}).get('attempt_id')) for j in jobs],
                accounting=state['campaigns'].get(CAMPAIGN))
    reporter=Thread(target=heartbeat,daemon=True);reporter.start()
    try:drain(queue,['0','1'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN)
    finally:stopped.set();reporter.join(timeout=1)
    state=queue.inspect();progress['phase']='trained'
    atomic_json(BRIEF/'progress.json',progress);atomic_json(HANDOFF/'progress.json',progress)
    log(phase='drained',campaign=state['campaigns'][CAMPAIGN],requests={r:state['submissions'][rid]['status'] for r,rid in requests.items()})

if __name__=='__main__':main()
