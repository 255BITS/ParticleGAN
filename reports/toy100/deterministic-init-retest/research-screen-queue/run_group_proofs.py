from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys
Q=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--selection',required=True);a=p.parse_args();rows=json.loads((Q/a.selection).read_text())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def run(r):
 out=Path(r['required_review']).parent;log=Q/'review-logs'/(out.name+'.log');log.parent.mkdir(exist_ok=True)
 with log.open('w') as f:p=subprocess.run([sys.executable,str(Q/'constructor_preflight.py'),'--bundle',r['directory'],'--output',str(out)],stdout=f,stderr=subprocess.STDOUT,env=env)
 result={'candidate':r['candidate'],'returncode':p.returncode,'proof':r['required_review'],'log':str(log)};print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,rows))
(Q/(Path(a.selection).stem+'-cpu-results.json')).write_text(json.dumps(results,indent=2)+'\n')
assert all(x['returncode']==0 for x in results)
