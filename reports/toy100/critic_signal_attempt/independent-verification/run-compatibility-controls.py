from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).parent
OUT=ROOT/'compatibility-controls';OUT.mkdir(exist_ok=False)
TASKS=['ae_gan_hold','unused_token_hold']
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ATEN_CPU_CAPABILITY='avx2')
commands={task:['/tmp/pr38-default-env/bin/python','-u',str(ROOT/'repo/reports/toy100/selected_h_compatibility.py'),'--declaration',str(ROOT/'one.json'),'--output',str(OUT/task),'--ledger',str(ROOT/'compatibility-control-tests.jsonl'),'--tasks',task,'--workers','1'] for task in TASKS}
(OUT/'declaration.json').write_text(json.dumps({'purpose':'Original required auxiliary-host benchmark compatibility; explicitly NOT pure GAN and cannot replace strict failures','commands':commands,'workers':2},indent=2)+'\n')
def run(task):
    started=time.perf_counter()
    with (OUT/(task+'.log')).open('w') as log:
        r=subprocess.run(commands[task],env=env,cwd=ROOT/'repo',stdout=log,stderr=subprocess.STDOUT)
    p=OUT/task/'h_n05r06_mixup_c0p01_lr15/status.json'
    v=json.loads(p.read_text())
    row={'task':task,'status':v['status'],'returncode':r.returncode,'seconds':time.perf_counter()-started,'artifact':str(p),'exclusively_adversarial':False}
    print(json.dumps(row),flush=True)
    return row
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    for future in as_completed([pool.submit(run,t) for t in TASKS]):
        rows.append(future.result())
        (OUT/'summary.json').write_text(json.dumps({'track':'benchmark compatibility only; not pure GAN','rows':rows},indent=2)+'\n')
