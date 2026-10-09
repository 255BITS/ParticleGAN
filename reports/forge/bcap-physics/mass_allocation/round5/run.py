"""Submit both ready arms before a bounded, independent shared-GPU drain."""
from pathlib import Path
import json
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue,drain
from experiments.forge.contracts import atomic_json,read_json,utc_now
ROOT=Path(__file__).resolve().parents[5]
ARCHIVE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009')
QUEUE=ARCHIVE/'mass_allocation/queue';PROGRESS=Path('/tmp/bcap-physics-round5-20261009/mass_allocation/progress.json')
CAMPAIGN='mass-allocation-round5-v1'
def save(p):
 atomic_json(PROGRESS,p);atomic_json(ARCHIVE/'handoff/mass_allocation/progress.json',p)
def main():
 q=Queue(QUEUE,report_root=ROOT/'reports/forge',on_completion=None);p=read_json(PROGRESS);requests={}
 for role in ('control','candidate'):
  req=resolve_idea(ROOT,f'mass-allocation-round5-{role}-v1',study=f'mass-allocation-round5-{role}-study-v1',queue_root=QUEUE,freeze_source=True)
  receipt=q.submit(req,req['study']['campaign']);requests[role]=receipt['request']['request_id']
  p.update(phase='submitted',requests=requests,source_commit=req['source']['origin_commit'],source_digest=req['source']['digest']);save(p)
  print(json.dumps(dict(time=utc_now(),event='submitted',role=role,request_id=requests[role],status=receipt['status'])),flush=True)
 p['phase']='running';save(p)
 drain(q,['0'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN)
 p['phase']='trained';save(p);print(json.dumps(dict(time=utc_now(),event='drain-complete',requests=requests)),flush=True)
if __name__=='__main__':main()
