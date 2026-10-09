"""Frozen numerical scorer oracle/destructive controls without training."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import torch
from experiments.forge.contracts import read_json,atomic_json,file_hash
from benchmarks.transfer_suite.vector_tasks import sample_target,score_samples
from benchmarks.toy_audit.gaussian1d_quality import sample_target as gaussian_target,score_samples as gaussian_score
from benchmarks.toy100.problems import sample_real,evaluation_geometry
from benchmarks.toy100.accuracy import evaluate_accuracy
from benchmarks.toy_audit.api_vectors import _bounds

def main():
    began=time.monotonic();torch.set_num_threads(1);rows=[]
    for name in ['gaussian1d_smoke','vector_unequal_width','vector_two_broad','grid100']:
        task=read_json(ROOT/f'configs/forge/tasks/{name}.json');rng=torch.Generator().manual_seed(0)
        if name=='grid100':
            points=sample_real(name,100000,generator=rng);centers,_=evaluation_geometry(name)
            collapsed=centers[torch.cdist(points,centers).argmin(1)]
        else:
            spec=task['execution']['host_definition'];fn=gaussian_target if name=='gaussian1d_smoke' else sample_target
            points=fn(spec,4096,rng,task['execution']['steps'])
            means=torch.tensor(spec['means']);collapsed=means[torch.cdist(points,means).argmin(1)]
        for law,values in [('oracle',points),('center_collapse',collapsed)]:
            if name=='grid100':
                metrics=evaluate_accuracy(values,name);passed=metrics['passed'];failed=[] if passed else ['full accuracy gate']
            else:
                fn=gaussian_score if name=='gaussian1d_smoke' else score_samples
                metrics=fn(values,spec,task['execution']['steps']);failed=_bounds(metrics,task['evaluation']['thresholds']);passed=not failed
            assert passed==(law=='oracle'),(name,law,failed)
            rows.append(dict(task_id=name,control=law,status='PASS' if passed else 'FAIL',metrics=metrics,failed_bounds=failed))
    directory=Path(__file__).parent
    atomic_json(directory/'scorer-controls.json',dict(schema_version=1,qualification_input=False,
        scope='independent_oracle_and_destructive_scorer_controls',seed=0,optimizer_updates_added=0,
        controls=rows,scorer_sources={name:file_hash(ROOT/name) for name in
        ['benchmarks/toy_audit/gaussian1d_quality.py','benchmarks/transfer_suite/vector_tasks.py','benchmarks/toy100/accuracy.py']},
        seconds=time.monotonic()-began))
    print([(r['task_id'],r['control'],r['status']) for r in rows],flush=True)

if __name__=='__main__':main()
