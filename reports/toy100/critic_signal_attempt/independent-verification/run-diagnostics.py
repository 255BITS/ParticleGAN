"""Authorized one-time baseline audit after first strict failure; no search."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, os, subprocess, time
ROOT=Path(__file__).parent
TASKS=['two_pole','cover_leftover','mid_scale_identity','vector_anisotropic','vector_two_broad','vector_spiral','unused_token_hold','ae_gan_hold']
PYTHON='/tmp/pr38-default-env/bin/python'
out=ROOT/'diagnostics';out.mkdir(exist_ok=False)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ATEN_CPU_CAPABILITY='avx2')
commands={task:[PYTHON,'-u',str(ROOT/'repo/reports/toy100/selected_h_remaining.py'),'--declaration',str(ROOT/'one.json'),'--output',str(out/task),'--ledger',str(ROOT/'diagnostic-tests.jsonl'),'--tasks',task,'--workers','1'] for task in TASKS}
(out/'declaration.json').write_text(json.dumps({'purpose':'Explicitly authorized diagnostic exception after strict unipolar FAIL; one run per remaining older host, no tuning','tasks':TASKS,'workers':4,'commands':commands,'native':'SKIPPED after older-host failure'},indent=2)+'\n')
def run(task):
    started=time.perf_counter()
    print(json.dumps({'event':'START','task':task}),flush=True)
    with (out/(task+'.log')).open('w') as log:
        process=subprocess.run(commands[task],cwd=ROOT/'repo',env=env,stdout=log,stderr=subprocess.STDOUT)
    status_file=out/task/'h_n05r06_mixup_c0p01_lr15/status.json'
    status=json.loads(status_file.read_text()) if status_file.exists() else {'status':'ERROR'}
    row={'task':task,'returncode':process.returncode,'status':status['status'],'seconds':time.perf_counter()-started,'artifact':str(status_file)}
    print(json.dumps({'event':'DONE',**row}),flush=True)
    return row
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(run,task) for task in TASKS]
    for future in as_completed(futures):
        rows.append(future.result())
        (out/'summary.json').write_text(json.dumps({'purpose':'diagnostic only','rows':rows},indent=2)+'\n')
