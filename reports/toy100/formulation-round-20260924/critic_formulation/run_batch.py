"""Run exactly one GPU worker, preserving frozen probe receipts and concise ledger."""
import argparse, datetime, hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('tasks',nargs='+');a=p.parse_args()
root=Path(__file__).resolve().parent;checkout=root.parents[2];attempt=checkout.parent
supervisor=attempt/'supervisor.md'
if supervisor.exists():
 steering=supervisor.read_text();print('SUPERVISOR '+steering.strip(),flush=True)
 if 'STOP' in steering: raise SystemExit('Supervisor STOP')
candidate=root/'candidates'/a.candidate
d=json.loads((candidate/'declaration.json').read_text())
for name in ('probe','config'):
 assert hashlib.sha256((candidate/(name+('.py' if name=='probe' else '.json'))).read_bytes()).hexdigest()==d[name+'_sha256']
env=os.environ.copy();env.pop('LD_PRELOAD',None);env.pop('PYTHONPATH',None)
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ATEN_CPU_CAPABILITY='avx2',MKL_ENABLE_INSTRUCTIONS='AVX2',ONEDNN_MAX_CPU_ISA='AVX2',DNNL_MAX_CPU_ISA='AVX2',PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for task in a.tasks:
 output=candidate/'runs'/task;output.parent.mkdir(exist_ok=True)
 fixture=checkout/'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures'/task/'initial-values.pt'
 assert fixture.exists(),fixture
 command=[sys.executable,str(candidate/'probe.py'),'--repo',str(root/'prepared/repos/cuda'),'--config',str(candidate/'config.json'),'--task',task,'--backend','cuda','--initial-state',str(fixture),'--output',str(output)]
 with (candidate/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,env={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','ATEN_CPU_CAPABILITY','MKL_ENABLE_INSTRUCTIONS','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA','PYTHONHASHSEED')},started=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
 print('START '+a.candidate+' '+task,flush=True);started=time.monotonic()
 with (candidate/(task+'.log')).open('w') as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 path=output/'result.json'
 record=json.loads(path.read_text()) if path.exists() else dict(status='ERROR',error='worker exited '+str(r.returncode))
 verdict=record.get('verdict',{});result=record.get('result',{})
 row=dict(candidate=a.candidate,gate=task,status=record['status'],seconds=record.get('seconds',time.monotonic()-started),metrics=result.get('live',{}),artifact=str(path),convergence=verdict.get('convergence'),shortfall=verdict.get('shortfall'),adam_calls=record.get('proof',{}).get('adam_calls'),update_counts=result.get('update_counts'),error=record.get('error'))
 with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 print(json.dumps({k:row[k] for k in ('candidate','gate','status','seconds','metrics','adam_calls','error')}),flush=True)
