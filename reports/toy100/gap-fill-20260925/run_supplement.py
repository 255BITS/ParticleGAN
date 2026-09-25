"""Add P1's own ring checks as slots free at the end of the initial queue."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/ml2/hypergan/gan-attempts/gap-fill-20260925')
jobs=json.loads((ROOT/'supplemental-jobs.json').read_text())
env=os.environ.copy()
for key in ['LD_PRELOAD','PYTHONPATH','CODEX_THREAD_ID']: env.pop(key,None)
env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',ATEN_CPU_CAPABILITY='avx2',MKL_ENABLE_INSTRUCTIONS='AVX2',ONEDNN_MAX_CPU_ISA='AVX2',DNNL_MAX_CPU_ISA='AVX2')
active=[]
completed=[]
def log(event,**kw):
    row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,queue='supplement',**kw)
    with (ROOT/'progress.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
while jobs or active:
    for item in active[:]:
        rc=item['p'].poll()
        if rc is None: continue
        item['f'].close();job=item['job'];path=Path(job['output'])/'result.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        row=dict(id=job['id'],status=result.get('status','ERROR'),returncode=rc,gpu=item['gpu'],result=str(path))
        completed.append(row);active.remove(item);log('DONE',**row)
        (ROOT/'supplemental-completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    try: state=json.loads((ROOT/'status.json').read_text())
    except json.JSONDecodeError: time.sleep(1);continue
    if not state['remaining'] and jobs:
        for gpu,resource in state['resources'].items():
            count=sum(x['gpu']==gpu for x in state['active'])+sum(x['gpu']==gpu for x in active)
            if jobs and count<6 and resource[0]>4096 and state['ram_available_mib']>8192:
                job=jobs.pop(0)
                assert not Path(job['output']).exists()
                f=open(job['log'],'w');p=subprocess.Popen(job['command'],cwd=job['cwd'],env={**env,'CUDA_VISIBLE_DEVICES':gpu},stdout=f,stderr=subprocess.STDOUT)
                active.append(dict(job=job,p=p,f=f,gpu=gpu));log('LAUNCH',id=job['id'],gpu=gpu,pid=p.pid)
    (ROOT/'supplemental-status.json').write_text(json.dumps(dict(remaining=[j['id'] for j in jobs],active=[dict(id=x['job']['id'],pid=x['p'].pid,gpu=x['gpu']) for x in active],completed=len(completed)),indent=2)+'\n')
    if jobs or active: time.sleep(10)
log('COMPLETE',jobs=len(completed))
