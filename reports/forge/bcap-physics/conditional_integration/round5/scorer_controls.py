"""Target-informed numerical controls, separate from all trained arms."""
from pathlib import Path
import numpy as np
from scipy.special import ndtri
import torch
from experiments.forge.contracts import atomic_json,read_json
from experiments.forge.gaussian_tasks import bounds
from benchmarks.toy_audit.gaussian1d_quality import score_samples
from benchmarks.transfer_suite import vector_tasks
from benchmarks.locked_shared.hosts.mid_scale_identity import smile_teacher,score_hold
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]

def main():
    torch.set_num_threads(1);rng=torch.get_rng_state().clone();rows=[]
    spec={'kind':'gaussian_mixture','means':[[2.]],'covariances':[[[.25]]],'masses':[1.]}
    quantiles=ndtri((np.arange(4096)+.5)/4096)
    for label,points,expected in [('oracle',2.+.5*quantiles,True),('collapsed',np.full(4096,2.),False),('wrong_mean',3.+.5*quantiles,False)]:
        metrics=score_samples(points[:,None],spec);passed=not bounds(metrics);assert passed==expected
        rows.append(dict(task='gaussian_smoke_and_retention',control=label,metrics=metrics,failed_bounds=bounds(metrics),passed=passed))
    teacher=smile_teacher()
    class Witness:
        def __init__(self,corrupt):self.corrupt=corrupt
        def state(self,s):return (teacher.stranger if self.corrupt and s==.5 else teacher.identity)+s*teacher.concept
    for label,corrupt in [('oracle',False),('mid_identity_swap',True)]:
        result=score_hold(Witness(corrupt),teacher=teacher);assert result['pass']==(not corrupt)
        rows.append(dict(task='mid_scale_identity',control=label,metrics={k:v for k,v in result.items() if type(v) in (int,float,bool,str)},passed=result['pass']))
    for taskid in ('vector_unequal_mass','vector_two_broad'):
        task=read_json(ROOT/f'configs/forge/tasks/{taskid}.json')
        spec={**task['execution']['host_definition'],'thresholds':task['evaluation']['thresholds']}
        # Explicit scoring-only stream, seed993 inherited from scorer contract;
        # not an experiment seed, model, initializer or training stream.
        points=vector_tasks.sample_target(spec,4096,torch.Generator().manual_seed(993),spec['steps'])
        for label,panel,expected in [('oracle',points,True),('point_collapse',torch.zeros_like(points)+points.mean(0),False)]:
            metrics=vector_tasks.score_samples(panel,spec,spec['steps'])
            passed=vector_tasks.passes(metrics,task['evaluation']['thresholds']);assert passed==expected,(taskid,label,metrics)
            rows.append(dict(task=taskid,control=label,metrics=metrics,passed=passed))
    identities=[dict(task=r['task_id'],role=r['role'],controls=r['scorer_controls']) for r in read_json(OUT/'results.json')['task_results'] if 'scorer_controls' in r]
    assert len(identities)==8 and torch.equal(rng,torch.get_rng_state())
    atomic_json(OUT/'scorer-controls.json',dict(schema_version=1,qualification_input=False,rows=rows,conditional_controls=identities,
        information_access='Target-informed scoring witnesses only; no trained arm, model, alternate initialization or qualification.',
        scoring_only_target_draws=8192,optimizer_updates_added=0,training_draws_added=0,ambient_rng_unchanged=True))
    print('Verified original identity, Gaussian, mid-scale and vector destructive scorer controls.')
if __name__=='__main__':main()
