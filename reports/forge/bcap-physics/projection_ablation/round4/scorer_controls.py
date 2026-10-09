"""Target-informed scorer witnesses; no learned arm, initialization or training."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtri
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,read_json
from experiments.forge.gaussian_tasks import bounds
from benchmarks.toy_audit.gaussian1d_quality import score_samples
from benchmarks.locked_shared.hosts.mid_scale_identity import smile_teacher,score_hold
from benchmarks.locked_shared.two_pole import HostCritic,real_batch,_grad_median
OUT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1);rng=torch.get_rng_state().clone();rows=[]
    spec={'kind':'gaussian_mixture','means':[[2.]],'covariances':[[[.25]]],'masses':[1.]}
    quantiles=ndtri((np.arange(4096)+.5)/4096)
    for label,points,expected in [('oracle',2.+.5*quantiles,True),('collapsed',np.full(4096,2.),False),('wrong_mean',3.+.5*quantiles,False)]:
        metrics=score_samples(points[:,None],spec)
        passed=not bounds(metrics);assert passed==expected
        rows.append(dict(task='both_gaussian_gates',control=label,metrics=metrics,failed_bounds=bounds(metrics),passed=passed))
    teacher=smile_teacher()
    class Witness:
        def __init__(self,corrupt):self.corrupt=corrupt
        def state(self,s):
            identity=teacher.stranger if self.corrupt and s==.5 else teacher.identity
            return identity+s*teacher.concept
    for label,corrupt in [('oracle',False),('mid_identity_swap',True)]:
        result=score_hold(Witness(corrupt),teacher=teacher)
        assert result['pass']==(not corrupt)
        rows.append(dict(task='mid_scale_identity',control=label,metrics={k:v for k,v in result.items() if type(v) in (int,float,bool,str)},passed=result['pass']))
    with torch.random.fork_rng(devices=[]):critic=HostCritic()
    real=real_batch(512)
    for label,points,expected in [('oracle',real,True),('collapsed',torch.zeros_like(real),False)]:
        metrics=dict(mean_abs=float(points.abs().mean()),grad_med=float(_grad_median(critic,real,points)))
        passed=metrics['mean_abs']>=.3 and metrics['grad_med']<=1
        assert passed==expected
        rows.append(dict(task='two_pole',control=label,metrics=metrics,passed=passed,scope='Stored-weight fixture critic, separately scoped scoring witness'))
    identity_controls=[dict(task=r['task_id'],role=r['role'],controls=r['scorer_controls']) for r in read_json(OUT/'results.json')['task_results'] if 'scorer_controls' in r]
    assert torch.equal(rng,torch.get_rng_state())
    atomic_json(OUT/'scorer-controls.json',dict(schema_version=1,qualification_input=False,information_access='Target-informed analytic/fixture scoring witnesses, not training arms or alternate initialized models',rows=rows,conditional_controls=identity_controls,optimizer_updates_added=0,sampling_draws_added=0,ambient_rng_unchanged=True))
    print('Numerical oracle and destructive controls verified for every task.')

if __name__=='__main__':main()
