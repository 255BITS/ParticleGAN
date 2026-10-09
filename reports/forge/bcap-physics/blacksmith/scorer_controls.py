"""Separate seed0 scorer fixtures, zero model calls/training; unchanged full gates."""
import json
import math
from pathlib import Path
import time

import torch
from benchmarks.toy_audit.gaussian1d_quality import sample_target as gaussian_target, score_samples as gaussian_score
from benchmarks.transfer_suite.vector_tasks import sample_target as vector_target, score_samples as vector_score
from benchmarks.locked_shared.mode_hold import ring_means, sample_ring, diversity, SIGMA

ROOT = Path(__file__).resolve().parents[4]


def failed(metrics, thresholds):
    checks = {'>=': lambda a,b: a>=b, '<=': lambda a,b: a<=b, '==': lambda a,b: a==b}
    return [f'{k} {op} {bound}' for k,op,bound in thresholds
            if not isinstance(metrics.get(k), (int,float)) or not math.isfinite(metrics[k])
            or not checks[op](metrics[k],bound)]


def main():
    started=time.monotonic();torch.set_num_threads(1)
    rows=[]
    for name, target, scorer in [('gaussian1d_smoke',gaussian_target,gaussian_score),('vector_two_broad',vector_target,vector_score)]:
        task=json.loads((ROOT/f'configs/forge/tasks/{name}.json').read_text());spec=task['execution']['host_definition']
        oracle=target(spec,4096,torch.Generator().manual_seed(0),task['execution']['steps'])
        if name.startswith('gaussian'):
            controls={'oracle':oracle,'collapsed':torch.full_like(oracle,2.),'shifted':oracle+.5,
                      'same_moment_atoms':torch.tensor([1.5,2.5]).repeat(2048)[:,None],
                      'too_wide':2+2*(oracle-2)}
        else:
            means=torch.tensor(spec['means']);assignment=torch.cdist(oracle,means).argmin(1)
            controls={'oracle':oracle,'collapsed_local_shape':means[assignment],
                      'shifted':oracle+1.,'missing_component':oracle[assignment==0].repeat(3,1)[:4096]}
        for control, points in controls.items():
            metrics=scorer(points,spec,task['execution']['steps']); failures=failed(metrics,task['evaluation']['thresholds'])
            assert bool(failures)==(control!='oracle'),(name,control,failures)
            rows.append(dict(task=name,fixture=control,metric_status='FAIL' if failures else 'PASS',failed_bounds=failures,metrics=metrics))
    task=json.loads((ROOT/'configs/forge/tasks/mode_hold.json').read_text());means=ring_means()
    oracle=sample_ring(means,4096,SIGMA,torch.Generator().manual_seed(0))
    for control,points in {'oracle':oracle,'missing_modes':means[0].repeat(4096,1),'off_support':oracle+1.}.items():
        metrics=diversity(points,means); failures=failed(metrics,task['evaluation']['thresholds'])
        assert bool(failures)==(control!='oracle')
        rows.append(dict(task='mode_hold',fixture=control,metric_status='FAIL' if failures else 'PASS',failed_bounds=failures,metrics=metrics))
    task=json.loads((ROOT/'configs/forge/tasks/two_pole.json').read_text())
    for control,metrics in {'metric_oracle':{'mean_abs':.5,'grad_med':.5},'inactive':{'mean_abs':0.,'grad_med':.5},'uncapped':{'mean_abs':.5,'grad_med':2.}}.items():
        failures=failed(metrics,task['evaluation']['thresholds']);assert bool(failures)==(control!='metric_oracle')
        rows.append(dict(task='two_pole',fixture=control,metric_status='FAIL' if failures else 'PASS',failed_bounds=failures,metrics=metrics,scope='metric_only_fixture'))
    out=dict(qualification_input=False,scope='separate_scorer_fixture_cohort',seed=0,optimizer_updates_added=0,model_calls_added=0,
             elapsed_cpu_wall_seconds=time.monotonic()-started,controls=rows)
    (Path(__file__).parent/'scorer-controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'verified_controls':len(rows),'cpu_wall_seconds':out['elapsed_cpu_wall_seconds']}))

if __name__=='__main__':main()
