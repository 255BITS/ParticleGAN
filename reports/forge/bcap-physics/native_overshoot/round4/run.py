"""Public Forge Queue/drain, one frozen global recipe per arm; no full compile."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.planning import resolve_idea,plan_summary
from experiments.forge.queue import Queue,drain
from experiments.forge.contracts import atomic_json

QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/native_overshoot/queue')
STATUS=Path('/tmp/bcap-physics-round4-20261009/native_overshoot/progress.json')
IDS={'control':'native-overshoot-round4-control-v1','candidate':'native-overshoot-round4-armijo-v1'}

def main():
    queue=Queue(QUEUE,report_root=ROOT/'reports/forge',on_completion=None)
    requests={};plans={}
    for role,identifier in IDS.items():
        request=resolve_idea(ROOT,identifier,study=f'native-overshoot-round4-{role}-study-v1',queue_root=QUEUE,freeze_source=True)
        plans[role]=plan_summary(request)
        assert not request['preflight_blockers']
        campaign=request['study']['campaign']
        submitted=queue.submit(request,campaign)
        requests[role]=submitted['request']['request_id']
        print(json.dumps(dict(event='submitted',role=role,request_id=requests[role],status=submitted['status'])),flush=True)
    queue.flush_events()
    atomic_json(Path(__file__).parent/'plan.json',plans)
    progress=json.loads(STATUS.read_text())
    progress.update(phase='training',requests=requests,source_commit=request['source']['origin_commit'],source_digest=request['source']['digest'])
    atomic_json(STATUS,progress)
    drain(queue,['1'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=campaign['id'])
    progress['phase']='publishing';atomic_json(STATUS,progress)
    print(json.dumps(dict(event='bounded_drain_finished',requests=requests)),flush=True)

if __name__=='__main__':main()
