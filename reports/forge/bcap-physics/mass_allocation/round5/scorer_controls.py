"""Independent oracle/destructive controls; evaluation only, no training."""
from pathlib import Path
import sys,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,read_json,file_hash
from benchmarks.toy100.problems import sample_real,evaluation_geometry
from benchmarks.toy100.metrics import evaluate_samples
from benchmarks.toy100.accuracy import evaluate_accuracy
from benchmarks.transfer_suite.vector_tasks import sample_target,score_samples
from benchmarks.toy_audit.gaussian1d_quality import sample_target as gaussian_target,score_samples as gaussian_score

def passed(m,t):return all(m[k]<=b if op=='<=' else m[k]>=b if op=='>=' else m[k]==b for k,op,b in t)
def main():
 torch.set_num_threads(1);rows={};g=torch.Generator().manual_seed(0);x=sample_real('grid100',100000,generator=g);c,_=evaluation_geometry('grid100')
 rows['grid100']={name:dict(coverage=evaluate_samples(p,'grid100'),accuracy=evaluate_accuracy(p,'grid100')) for name,p in [('oracle',x),('balanced_point_collapse',c.repeat_interleave(1000,0)),('one_mode',x*0+c[0])]}
 assert rows['grid100']['oracle']['coverage']['passed'] and rows['grid100']['oracle']['accuracy']['passed']
 assert all(not rows['grid100'][name]['coverage']['passed'] for name in ['balanced_point_collapse','one_mode'])
 for tid in ['gaussian1d_smoke','vector_unequal_mass','vector_unequal_width','vector_two_broad']:
  t=read_json(ROOT/f'configs/forge/tasks/{tid}.json');spec=t['execution']['host_definition'];gaussian=tid.startswith('gaussian');scorer=gaussian_score if gaussian else score_samples;x=(gaussian_target if gaussian else sample_target)(spec,20000,torch.Generator().manual_seed(0),spec['steps']);collapse=x*0+torch.tensor(spec['means'][0]);centers=torch.tensor(spec['means'],dtype=x.dtype)
  count=torch.tensor(spec['masses'])*len(x);repeats=count.round().long();repeats[-1]+=len(x)-int(repeats.sum());balanced=centers.repeat_interleave(repeats,dim=0)
  rows[tid]={}
  for name,p in [('oracle',x),('one_mode',collapse),('balanced_point_collapse',balanced)]:
   m=scorer(p,spec,spec['steps']);rows[tid][name]=dict(metrics=m,passed=passed(m,t['evaluation']['thresholds']))
  assert rows[tid]['oracle']['passed'] and not rows[tid]['one_mode']['passed'] and not rows[tid]['balanced_point_collapse']['passed'],tid
 atomic_json(Path(__file__).parent/'scorer-controls.json',dict(schema_version=1,qualification_input=False,scope='Target-informed evaluation-only controls; never learned baseline or training supervision.',optimizer_updates_added=0,controls=rows,sources={p:file_hash(ROOT/p) for p in ['benchmarks/toy100/metrics.py','benchmarks/toy100/accuracy.py','benchmarks/transfer_suite/vector_tasks.py','benchmarks/toy_audit/gaussian1d_quality.py']}));print(json.dumps({k:'PASS' for k in rows}))
if __name__=='__main__':main()
