"""Freeze all three ready arms, then run their bounded shared GPU worker."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
import particlegan
from experiments.forge.planning import resolve_idea, plan_summary
from experiments.forge.queue import Queue, drain
from experiments.forge.contracts import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[5]
QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_ablation/queue')
PROGRESS = Path('/tmp/bcap-physics-round4-20261009/projection_ablation/progress.json')
ROLES = ('nonascent', 'direction_blend', 'strict_progress')
CAMPAIGN = 'projection-ablation-round4-v1'


def log(**values):
    print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(), **values)),flush=True)


def main():
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT)
    queue = Queue(QUEUE, report_root=ROOT/'reports/forge', on_completion=None)
    requests={}
    for role in ROLES:
        candidate=f'projection-ablation-round4-{role}-v1'
        study=f'projection-ablation-round4-{role}-study-v1'
        request=resolve_idea(ROOT,candidate,queue_root=QUEUE,freeze_source=True,study=study)
        plan=plan_summary(request,queue.inspect(),include_ownership=True)
        atomic_json(PROGRESS.parent/f'{role}-plan.json',plan)
        entry=queue.submit(request,request['study']['campaign'])
        requests[role]=entry['request']['request_id']
        atomic_json(PROGRESS.parent/f'{role}-enqueue.json',dict(request_id=requests[role],status=entry['status']))
        log(phase='enqueued',role=role,request_id=requests[role],source_digest=request['source']['digest'],reservation=plan['worst_case_seconds'])
    state=queue.inspect()
    sources=[state['submissions'][rid]['request']['source'] for rid in requests.values()]
    assert all(source==sources[0] for source in sources)
    progress=read_json(PROGRESS)
    progress.update(phase='training',requests=requests,source_commit=sources[0]['origin_commit'],source_digest=sources[0]['digest'])
    atomic_json(PROGRESS,progress)
    log(phase='draining',pid=os.getpid(),module=particlegan.__file__,gpu='0',workers=1,sharing=True)
    drain(queue,['0'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN)
    state=queue.inspect()
    progress['phase']='trained';atomic_json(PROGRESS,progress)
    log(phase='drained',campaign=state['campaigns'][CAMPAIGN],requests={r:state['submissions'][rid]['status'] for r,rid in requests.items()})

if __name__=='__main__':main()
