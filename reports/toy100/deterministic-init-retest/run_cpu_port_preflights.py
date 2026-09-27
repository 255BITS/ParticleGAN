"""Bounded constructor-only CPU subprocess batch; never runs a learner step."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os, subprocess, sys
ROOT=Path(__file__).resolve().parent
INVENTORY=ROOT.parent/'continuous-api-search/fixed-init-retest-inventory/api-screening-candidates.json'
rows=json.loads(INVENTORY.read_text())['candidate_rows']
LOG=ROOT/'cpu-preflight-logs';LOG.mkdir(exist_ok=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
def run(row):
 alias=row['candidate'].lower();port=ROOT/'port-source'/alias;decl=port/'candidate-declaration.json';receipt=ROOT/(alias+'-cpu-preflight.json')
 if not decl.exists():return dict(candidate=alias,status='NO_RUNNABLE_DECLARATION')
 if receipt.exists():return dict(candidate=alias,status='EXISTING_RECEIPT',receipt=str(receipt))
 cmd=[sys.executable,str(ROOT/'preflight.py'),'--package-root',str(port/'package'),'--declaration',str(decl),'--cpu-init','--output',str(receipt)]
 proc=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (LOG/(alias+'.log')).write_text(proc.stdout)
 return dict(candidate=alias,status='PASS' if proc.returncode==0 else 'ERROR',exit_code=proc.returncode,receipt=str(receipt),log=str(LOG/(alias+'.log')))
results=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 for future in as_completed([pool.submit(run,row) for row in rows]):
  value=future.result();results.append(value);print(json.dumps(value),flush=True)
  (ROOT/'cpu-port-preflight-batch.json').write_text(json.dumps(dict(scope='CPU constructors only; 3 concurrent subprocesses; CUDA_VISIBLE_DEVICES empty',results=sorted(results,key=lambda x:x['candidate'])),indent=2)+'\n')
