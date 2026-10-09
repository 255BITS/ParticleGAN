"""Oracle and destructive numerical scorer checks, never training fixtures."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from benchmarks.toy100.problems import sample_real,evaluation_geometry
from benchmarks.toy100.metrics import evaluate_samples
from benchmarks.toy100.accuracy import evaluate_accuracy
from benchmarks.toy_audit.gaussian1d_quality import score_samples as gaussian_score, sample_target as gaussian_target
from benchmarks.transfer_suite.vector_tasks import sample_target,score_samples
from experiments.forge.contracts import atomic_json,read_json

def passed(values,thresholds):
    return all(values[name]<=bound if op=='<=' else values[name]>=bound if op=='>=' else values[name]==bound
               for name,op,bound in thresholds)

def main():
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(0)
    target=sample_real('grid100',100000,generator=generator)
    centers,sigma=evaluation_geometry('grid100')
    collapsed=centers.repeat_interleave(1000,0)
    native={name:dict(coverage=evaluate_samples(points,'grid100'),accuracy=evaluate_accuracy(points,'grid100'))
            for name,points in [('oracle',target),('balanced_point_collapse',collapsed),('one_mode',target[:10000]*0+centers[0])]}
    assert native['oracle']['coverage']['passed'] and native['oracle']['accuracy']['passed']
    assert not native['balanced_point_collapse']['coverage']['passed']
    assert not native['one_mode']['coverage']['passed']
    results=dict(grid100=native)
    for taskid in ('gaussian1d_smoke','vector_two_broad'):
        task=read_json(ROOT/f'configs/forge/tasks/{taskid}.json');spec=task['execution']['host_definition']
        thresholds=task['evaluation']['thresholds']
        rng=torch.Generator().manual_seed(0)
        oracle=(gaussian_target if taskid=='gaussian1d_smoke' else sample_target)(spec,20000,rng,spec['steps'])
        centers=np.asarray(spec['means'])
        scorer=gaussian_score if taskid=='gaussian1d_smoke' else score_samples
        table={}
        for name,points in [('oracle',oracle),('point_collapse',np.broadcast_to(centers[0],oracle.shape).copy())]:
            metrics=scorer(torch.as_tensor(points,dtype=oracle.dtype),spec,spec['steps'])
            table[name]=dict(metrics=metrics,passed=passed(metrics,thresholds))
        assert table['oracle']['passed'] and not table['point_collapse']['passed']
        results[taskid]=table
    atomic_json(Path(__file__).parent/'scorer-controls.json',dict(schema_version=1,qualification_input=False,
        scope='Evaluation-only oracle and destructive controls, separate from seed0 learned baseline',
        optimizer_updates_added=0,training_information_access='none',controls=results))
    print(json.dumps({name:'PASS' for name in results}),flush=True)

if __name__=='__main__':main()
