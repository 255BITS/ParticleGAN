"""Bounded, memory-aware parallel runner. Immutable commands and one log per job."""
import collections
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/ml2/hypergan/gan-attempts/gap-fill-20260925')
jobs = collections.deque(json.loads((ROOT/'manifest.json').read_text())['jobs'])
GPUS = ['GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69','GPU-72c1b506-891d-b8bc-b353-e020585e1c47']
env = os.environ.copy()
for key in ['LD_PRELOAD','PYTHONPATH','CODEX_THREAD_ID']: env.pop(key,None)
env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',ATEN_CPU_CAPABILITY='avx2',MKL_ENABLE_INSTRUCTIONS='AVX2',ONEDNN_MAX_CPU_ISA='AVX2',DNNL_MAX_CPU_ISA='AVX2')
active, completed = [], []
started = time.monotonic()
def log(event, **data):
    row = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**data)
    with (ROOT/'progress.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def resources():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu={p[0].strip():(int(p[1]),int(p[2])) for line in raw.splitlines() if (p:=line.split(','))}
    memory={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    return gpu,memory['MemAvailable']//1024
log('START',jobs=len(jobs),max_workers_per_gpu=6,gpus=GPUS)
while jobs or active:
    for item in active[:]:
        rc=item['process'].poll()
        if rc is None: continue
        item['handle'].close()
        job=item['job']
        result_path=Path(job['output'])/'result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        row=dict(id=job['id'],status=result.get('status','ERROR'),returncode=rc,gpu=item['gpu'],wall_seconds=round(time.monotonic()-item['start'],2),result=str(result_path),error=result.get('error'))
        completed.append(row); active.remove(item)
        log('DONE',**row)
        (ROOT/'completed.json').write_text(json.dumps(completed,indent=2)+'\n')
    gpu,ram=resources()
    cap=min(6, 2+int((time.monotonic()-started)//45))
    for device in GPUS:
        count=sum(x['gpu']==device for x in active)
        # At most one fresh launch per device per loop; startup allocation settles before the next.
        if jobs and count<cap and gpu[device][0]>4096 and ram>8192 and not (ROOT/'STOP').exists():
            job=jobs.popleft()
            if Path(job['output']).exists(): raise RuntimeError('Refusing to overwrite '+job['output'])
            handle=open(job['log'],'w')
            process=subprocess.Popen(job['command'],cwd=job['cwd'],env={**env,'CUDA_VISIBLE_DEVICES':device},stdout=handle,stderr=subprocess.STDOUT)
            active.append(dict(job=job,process=process,handle=handle,gpu=device,start=time.monotonic()))
            log('LAUNCH',id=job['id'],gpu=device,pid=process.pid,remaining=len(jobs),active=len(active))
    (ROOT/'status.json').write_text(json.dumps(dict(remaining=[j['id'] for j in jobs],active=[dict(id=x['job']['id'],pid=x['process'].pid,gpu=x['gpu']) for x in active],completed=len(completed),resources=gpu,ram_available_mib=ram),indent=2)+'\n')
    if jobs or active: time.sleep(10)
log('COMPLETE',jobs=len(completed),statuses=dict(collections.Counter(x['status'] for x in completed)))
