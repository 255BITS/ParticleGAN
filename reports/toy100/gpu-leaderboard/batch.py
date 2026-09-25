"""Resumable complete GPU matrix; all supported toys run despite quality failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json,os
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--retry-errors',action='store_true');a=p.parse_args()
rows=json.loads((ROOT/'candidates.json').read_text())
toys=['two_pole','mode_hold','unipolar','mid_scale_identity','cover_leftover','trajectory','residual_student',
 'img_stripes2','img_bars4','vector_overlap','img_blobs4','img_intensity2','vector_unequal_mass','vector_unequal_width',
 'vector_two_broad','vector_anisotropic','vector_spiral','ae_gan_hold','unused_token_hold','grid100','rotated100','staggered100']
all_tasks=toys+['convergence']
ledger=ROOT/'ledger.jsonl';(ROOT/'runs').mkdir(exist_ok=True)
profile=dict(name='cuda_fp32_v1',gpu_visible='1',workers=3,torch='2.13.0+cu126',cuda='12.6',tf32=False,
 deterministic=True,workspace=':4096:8',cpu_scores_reused=False,all_supported_toys_run=True,
 native_budget=7000,legacy_budgets='unchanged 19-task protocol',rank='completed GPU toy pass count; post-convergence stability shown independently')
(ROOT/'protocol.json').write_text(json.dumps(dict(profile=profile,toys=toys,candidates=[r['name'] for r in rows],
 convergence=dict(confirm=200,settle_max_step=6000,hold=1200,no_rearm=True)),indent=2)+'\n')
latest={}
if ledger.exists():
 for line in ledger.read_text().splitlines():
  event=json.loads(line);latest[(event['candidate'],event['task'])]=event

def record(event):
 with ledger.open('a') as stream:stream.write(json.dumps(event,default=str)+'\n')
 latest[(event['candidate'],event['task'])]=event
 print(json.dumps(dict(event='COMPLETE',candidate=event['candidate'],task=event['task'],status=event['status'],seconds=round(event.get('seconds',0),2))),flush=True)
 publish()

def publish():
 lines=['# GPU leaderboard — cuda_fp32_v1','','All supported tests use the same RTX A6000, FP32/CUDA stack. CPU results are historical only.',
 'The matrix remains provisional until the batch finishes. Unsupported is not a pass or a measured failure.','',
 '| Candidate | GPU toy PASS | FAIL | ERROR | Unsupported | Pending | Post-convergence |','|---|---:|---:|---:|---:|---:|---|']
 stats=[]
 for r in rows:
  values=[latest.get((r['name'],t),{}).get('status','PENDING') for t in toys]
  counts={s:values.count(s) for s in ['PASS','FAIL','ERROR','UNSUPPORTED','PENDING']}
  c=latest.get((r['name'],'convergence'),{})
  conv=c.get('convergence',{});hold=conv.get('hold_checks',0)-(1 if conv.get('first_hold_failure') else 0)
  desc=c.get('status','PENDING')
  if conv:desc+=f"; confirm {conv.get('converged_step')}; {hold}/1200 good hold checks"
  stats.append((counts['PASS'],r['name'],counts,desc))
 for _,name,c,desc in sorted(stats,key=lambda x:(-x[0],x[1])):
  lines.append(f"| {name} | {c['PASS']}/22 | {c['FAIL']} | {c['ERROR']} | {c['UNSUPPORTED']} | {c['PENDING']} | {desc} |")
 lines+=['','## All 22 toys','','| Toy | '+' | '.join(r['name'] for r in rows)+' |','|---|'+'---|'*len(rows)]
 for task in toys:
  lines.append('| '+task+' | '+' | '.join(latest.get((r['name'],task),{}).get('status','PENDING') for r in rows)+' |')
 lines+=['','Updated '+datetime.now(timezone.utc).isoformat(),'',
 'No release winner is implied by rank. Required full-toy coverage and post-convergence hold must both pass.']
 (ROOT/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
 (ROOT/'latest.json').write_text(json.dumps(list(latest.values()),indent=2,default=str)+'\n')

jobs=[]
for task in all_tasks:
 for row in rows:
  key=row['name'],task
  if key in latest and (latest[key]['status']!='ERROR' or not a.retry_errors):continue
  supported=(row['supported']=='all' or task in ('convergence','trajectory','mode_hold') and row['supported']=='two'
    or row['supported']=='three' and task in ('two_pole','mode_hold','unipolar','convergence'))
  if not supported:
   record(dict(candidate=row['name'],task=task,status='UNSUPPORTED',executed=False,seconds=0,
               reason='Published candidate has no adapter for this host; no replacement policy credited'))
  else:jobs.append((row['name'],task))
publish()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
env.pop('LD_PRELOAD',None);env.pop('PYTHONPATH',None)

def execute(job):
 name,task=job
 base=ROOT/'runs'/name/task;out=base;attempt=1
 while out.exists():
  attempt+=1;out=base.with_name(task+f'-attempt{attempt}')
 out.parent.mkdir(parents=True,exist_ok=True)
 command=[sys.executable,'-u',str(ROOT/'worker.py'),'--candidate',name,'--task',task,'--output',str(out)]
 started=time.perf_counter()
 with out.with_suffix('.log').open('w',buffering=1) as log:
  print(json.dumps(dict(event='START',candidate=name,task=task,log=str(out.with_suffix('.log')))),flush=True)
  completed=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 file=out/'status.json'
 result=json.loads(file.read_text()) if file.exists() else dict(candidate=name,task=task,status='ERROR',error='Worker failed before status; inspect '+str(out.with_suffix('.log')))
 result.update(command=command,exit_code=completed.returncode,wall_seconds=time.perf_counter()-started,attempt=attempt)
 return result

halted=False
with ThreadPoolExecutor(max_workers=3) as pool:
 active={};remaining=iter(jobs)
 for _ in range(3):
  job=next(remaining,None)
  if job:active[pool.submit(execute,job)]=job
 while active:
  future=next(as_completed(active));event=future.result();record(event);del active[future]
  if event['status']=='ERROR':halted=True
  if not halted:
   job=next(remaining,None)
   if job:active[pool.submit(execute,job)]=job
(ROOT/'batch-status.json').write_text(json.dumps(dict(status='HALTED_ON_ERROR' if halted else 'COMPLETE',
 completed=len(latest),planned=len(rows)*len(all_tasks),finished=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
print(json.dumps(dict(event='BATCH_DONE',halted=halted)),flush=True)
