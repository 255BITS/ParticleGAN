"""Independent oracle/destructive scoring witnesses; no trainer or paid probe."""
from pathlib import Path
import time,torch
from experiments.forge.contracts import read_json,atomic_json
from benchmarks.transfer_suite import vector_tasks as vector
from benchmarks.toy_audit import gaussian1d_quality as scalar
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent

def main():
    start=time.monotonic();torch.set_num_threads(1);rows=[]
    for task_id in ('gaussian1d_smoke','vector_unequal_mass','vector_two_broad','vector_unequal_width'):
        task=read_json(ROOT/f'configs/forge/tasks/{task_id}.json');spec={**task['execution']['host_definition'],'thresholds':task['evaluation']['thresholds']};api=scalar if task_id.startswith('gaussian') else vector
        oracle=api.sample_target(spec,16384,torch.Generator().manual_seed(0),0)
        mean=torch.tensor(spec['means'][0]);collapse=mean.expand_as(oracle).clone()
        fixtures={'oracle':oracle,'point_collapse':collapse,'shift':oracle+3.}
        if api is scalar:fixtures['wrong_width']=mean+3*(oracle-mean)
        for name,points in fixtures.items():
            metrics=api.score_samples(points,spec,0)
            failed=[(k,op,b) for k,op,b in task['evaluation']['thresholds'] if not (metrics[k]>=b if op=='>=' else metrics[k]<=b if op=='<=' else metrics[k]==b)]
            assert bool(failed)==(name!='oracle'),(task_id,name,failed)
            rows.append(dict(task_id=task_id,control=name,status='FAIL' if failed else 'PASS',metrics=metrics,failed_bounds=failed))
    atomic_json(OUT/'scorer-controls.json',dict(schema_version=1,qualification_input=False,controls=rows,optimizer_updates_added=0,elapsed_seconds=time.monotonic()-start,information_access='target-informed scoring only, never training or initialization'))
if __name__=='__main__':main()
