"""Submit the frozen four-arm campaign, then run one bounded shared GPU worker."""
from pathlib import Path
import json
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue, drain
from experiments.forge.contracts import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[5]
QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_transport/queue')
PROGRESS = Path('/tmp/bcap-physics-round4-20261009/projection_transport/progress.json')
ROLES = ('winner', 'projection', 'transport', 'both')
CAMPAIGN = 'projection_transport-round4-v1'

def main():
    queue = Queue(QUEUE, report_root=ROOT/'reports/forge', on_completion=None)
    progress = json.loads(PROGRESS.read_text())
    requests = {}
    for role in ROLES:
        study = f'projection_transport-round4-{role}-study-v1'
        request = resolve_idea(ROOT, f'projection_transport-round4-{role}-v1', study=study,
                               queue_root=QUEUE, freeze_source=True)
        receipt = queue.submit(request, request['study']['campaign'])
        requests[role] = receipt['request']['request_id']
        progress.update(phase='submitted',requests=requests,studies=[f'projection_transport-round4-{x}-study-v1' for x in ROLES],
                        source_commit=request['source']['origin_commit'],source_digest=request['source']['digest'])
        atomic_json(PROGRESS,progress)
        print(json.dumps(dict(time=utc_now(),event='submitted',role=role,request_id=requests[role],status=receipt['status'])),flush=True)
    progress.update(phase='running');atomic_json(PROGRESS,progress)
    print(json.dumps(dict(time=utc_now(),event='drain-start',requests=requests)),flush=True)
    drain(queue,['0'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN)
    progress.update(phase='trained');atomic_json(PROGRESS,progress)
    print(json.dumps(dict(time=utc_now(),event='drain-complete',requests=requests)),flush=True)

if __name__=='__main__':main()
