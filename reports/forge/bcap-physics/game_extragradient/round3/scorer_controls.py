"""Independent scorer controls for unchanged Gaussian/vector gates; no training."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtri
import torch

from benchmarks.toy_audit.gaussian1d_quality import score_samples as gaussian_score
from benchmarks.transfer_suite import vector_tasks

ROOT = Path(__file__).resolve().parents[5]


def failures(task, metrics):
    checks = {'<=':lambda a,b:a<=b, '>=':lambda a,b:a>=b, '==':lambda a,b:a==b}
    return [f'{k} {op} {v}' for k,op,v in task['evaluation']['thresholds']
            if metrics.get(k) is None or not checks[op](metrics[k],v)]


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    rows = {}
    for name in ('gaussian1d_smoke','vector_two_broad','vector_unequal_width'):
        task = json.loads((ROOT / 'configs/forge/tasks' / (name+'.json')).read_text())
        spec = task['execution']['host_definition']
        step = task['execution']['steps']
        spec = {**spec, 'thresholds':task['evaluation']['thresholds']}
        if name.startswith('gaussian'):
            oracle = torch.tensor(2.+.5*ndtri((np.arange(4096)+.5)/4096))[:,None]
            controls = dict(oracle=oracle, collapsed=torch.full_like(oracle,2.),
                            shifted=oracle+1., too_wide=2.+2.*(oracle-2.))
            scorer = gaussian_score
        else:
            oracle = vector_tasks.sample_target(spec,4096,torch.Generator().manual_seed(0),step)
            centers = torch.tensor(spec['means'])
            ids = torch.cdist(oracle,centers).argmin(1)
            first = oracle[ids==0]
            controls = dict(oracle=oracle, collapsed=oracle.mean(0).expand_as(oracle),
                            no_width=centers[ids], one_mode=first[torch.arange(4096)%len(first)])
            scorer = vector_tasks.score_samples
        rows[name] = {}
        for control, points in controls.items():
            metrics = scorer(points,spec,step)
            failed = failures(task,metrics)
            rows[name][control] = dict(passed=not failed,failed_bounds=failed,metrics=metrics)
            assert (not failed) == (control=='oracle'), (name,control,failed)
    from benchmarks.locked_shared.mode_hold import ring_means, sample_ring, diversity, SIGMA
    task = json.loads((ROOT / 'configs/forge/tasks/mode_hold.json').read_text())
    means = ring_means()
    oracle = sample_ring(means,4096,SIGMA,torch.Generator().manual_seed(0))
    rows['mode_hold'] = {}
    for name,points in dict(oracle=oracle,collapsed=torch.zeros_like(oracle),
                             one_mode=means[:1].expand_as(oracle)).items():
        metrics = diversity(points,means)
        failed = failures(task,metrics)
        rows['mode_hold'][name] = dict(passed=not failed,failed_bounds=failed,metrics=metrics)
        assert (not failed) == (name=='oracle')
    result = dict(schema_version=1, qualification_input=False, optimizer_updates_added=0,
                  scope=__doc__, seconds=time.monotonic()-started, tasks=rows,
                  note='Gaussian oracle quantiles; vector oracle draws use an isolated seed0 control stream. These are scorer controls, not trainer seed experiments or profile calibration.')
    Path(__file__).with_name('scorer-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{k:v['passed'] for k,v in controls.items()} for name,controls in rows.items()}))


if __name__=='__main__':
    main()
