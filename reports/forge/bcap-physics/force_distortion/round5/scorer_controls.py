"""Evaluation-only oracle and destructive witnesses; no training or new gates."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtri
import torch

ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,read_json
from experiments.forge.gaussian_tasks import bounds
from benchmarks.toy100.problems import sample_real,evaluation_geometry
from benchmarks.toy100.metrics import evaluate_samples
from benchmarks.toy100.accuracy import evaluate_accuracy
from benchmarks.toy_audit.gaussian1d_quality import score_samples
from benchmarks.locked_shared.trajectory import trajectories,identity_mse
from benchmarks.locked_shared.hosts.residual_student import landing_stats
from benchmarks.locked_shared.hosts.mid_scale_identity import smile_teacher,score_hold

OUT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1);ambient=torch.get_rng_state().clone();controls={}
    generator=torch.Generator().manual_seed(0);target=sample_real('grid100',100000,generator=generator)
    centers,_=evaluation_geometry('grid100')
    native={name:dict(coverage=evaluate_samples(points,'grid100'),accuracy=evaluate_accuracy(points,'grid100'))
        for name,points in [('oracle',target),('balanced_point_collapse',centers.repeat_interleave(1000,0)),
                           ('one_mode',target*0+centers[0])]}
    assert native['oracle']['coverage']['passed'] and native['oracle']['accuracy']['passed']
    assert not native['balanced_point_collapse']['coverage']['passed'] and not native['one_mode']['coverage']['passed']
    controls['grid100']=native
    for mean in (2.,3.):
        spec=dict(kind='gaussian_mixture',means=[[mean]],covariances=[[[.25]]],masses=[1.])
        quantiles=ndtri((np.arange(4096)+.5)/4096);rows={}
        for name,values in [('oracle',mean+.5*quantiles),('point_collapse',np.full(4096,mean)),('wrong_mean',mean+1+.5*quantiles)]:
            metrics=score_samples(values[:,None],spec);failed=bounds(metrics)
            rows[name]=dict(metrics=metrics,failed_bounds=failed,passed=not failed)
        assert rows['oracle']['passed'] and not rows['point_collapse']['passed'] and not rows['wrong_mean']['passed']
        controls[f'gaussian_mean_{mean}']=rows
    _,fast=trajectories();wrong=fast.roll(1,0)
    for taskid in ('trajectory','residual_student'):
        task=read_json(ROOT/f'configs/forge/tasks/{taskid}.json');rows={}
        for name,values in [('oracle',fast),('wrong_identity',wrong)]:
            metrics=dict(identity_mse=identity_mse(values,fast))
            if taskid=='residual_student':metrics.update(landing_stats(values,fast))
            passed=all(metrics[k]<=v if op=='<=' else metrics[k]>=v if op=='>=' else metrics[k]==v
                       for k,op,v in task['evaluation']['thresholds'])
            rows[name]=dict(metrics=metrics,passed=passed)
        assert rows['oracle']['passed'] and not rows['wrong_identity']['passed'];controls[taskid]=rows
    teacher=smile_teacher()
    class Witness:
        def __init__(self,corrupt):self.corrupt=corrupt
        def state(self,scale):
            identity=teacher.stranger if self.corrupt and scale==.5 else teacher.identity
            return identity+scale*teacher.concept
    mid={label:score_hold(Witness(corrupt),teacher=teacher) for label,corrupt in [('oracle',False),('mid_identity_swap',True)]}
    assert mid['oracle']['pass'] and not mid['mid_identity_swap']['pass'];controls['mid_scale_identity']=mid
    assert torch.equal(ambient,torch.get_rng_state())
    atomic_json(OUT/'scorer-controls.json',dict(schema_version=1,qualification_input=False,controls=controls,
        information_access='Target-informed scorer witnesses only; never candidate training or alternate initializers.',
        optimizer_updates_added=0,training_rng_draws_added=0,ambient_rng_unchanged=True))
    print('All six task gates distinguish their numerical oracle and destructive witnesses.',flush=True)

if __name__=='__main__':main()
