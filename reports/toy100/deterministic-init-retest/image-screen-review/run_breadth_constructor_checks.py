from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import itertools,json,os,subprocess,sys
R=Path(__file__).resolve().parent
rows=list(itertools.product(('api-rp12','api-rp14','api-rp15'),('img_bars4','img_blobs4','img_stripes2')))
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
def run(row):
 candidate,task=row
 result=subprocess.run([sys.executable,str(R/'breadth_constructor_check.py'),'--candidate',candidate,'--task',task],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 log=R/(candidate+'-'+task+'-cpu.log');log.write_text(result.stdout)
 report=dict(candidate=candidate,task=task,returncode=result.returncode,log=str(log))
 print(json.dumps(report),flush=True)
 return report
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,rows))
(R/'breadth-cpu-results.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['returncode']==0 for r in results)
