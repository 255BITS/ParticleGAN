"""Cold acquisition gate for the alternating Adam with critic-gated G own-curvature bound."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100.alternating_curvature_scratch import alternating_curvature,METHOD


def main():
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--curvature-bound',type=float,default=1.)
    parser.add_argument('--advantage-gate',type=float)
    parser.add_argument('--bound-d',action='store_true')
    parser.add_argument('--d-curvature-bound',type=float)
    parser.add_argument('--ratio-reference',type=float)
    parser.add_argument('--per-group',action='store_true')
    parser.add_argument('--per-particle',action='store_true')
    parser.add_argument('--critic-average',type=float)
    parser.add_argument('--smooth-critic',type=float);args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='alternating_curvature_response',lr_floor=1.,lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
    recipe,noise,_=declared_recipe(config);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    declaration=dict(seed=0,task_order=['trajectory','mode_hold'],budgets={'mode_hold':1200,'trajectory':400},
        stop_at_first_failed_host=True,shared_gate_eligible=False,scratch_optimizer_policy=METHOD,
        solver=dict(curvature_bound=args.curvature_bound,advantage_gate=args.advantage_gate,bound_d=args.bound_d,d_curvature_bound=args.d_curvature_bound,ratio_reference=args.ratio_reference,ratio_span=1.5,ratio_decay=.9,per_group=args.per_group,per_particle=args.per_particle,critic_average=args.critic_average,smooth_critic=args.smooth_critic,smooth_samples=8,update_order='alternating'),
        source={name:hashlib.sha256((ROOT/'reports/toy100'/name).read_bytes()).hexdigest() for name in
                ('alternating_curvature_scratch.py','extra_adam_scratch.py','alternating_curvature_probe.py')})
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    stages=[];started=time.perf_counter()
    for task in declaration['task_order']:
        spec=next(job['spec'] for job in plan() if job['spec']['name']==task)
        try:
            with alternating_curvature(task=task,curvature_bound=args.curvature_bound,**({'bound_d':True,'d_curvature_bound':args.d_curvature_bound,'ratio_reference':args.ratio_reference,'per_group':args.per_group,'per_particle':args.per_particle,'critic_average':args.critic_average,'smooth_critic':args.smooth_critic} if args.bound_d else {'advantage_gate':args.advantage_gate})) as (recorder,source):
                result,context=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
            verdict=test_verdict(spec,result)
            data=dict(result=result,applied=context['applied'],noise=context['noise_receipt'],
                dynamics=recorder.receipt(),spec=spec,verdict=verdict,shared_gate_eligible=False,
                scratch_optimizer_policy=METHOD)
            (args.output/(task+'.json')).write_text(json.dumps(data,allow_nan=False)+'\n')
            stages.append(dict(task=task,verdict=verdict,live=result['live'],seconds=result['seconds']))
            print(json.dumps(dict(event='STAGE_DONE',**stages[-1])),flush=True)
            if not verdict['passed']:break
        except Exception as error:
            import traceback
            failure=dict(task=task,error=repr(error),traceback=traceback.format_exc(),
                completed_outer_steps=recorder.outer_steps,records=recorder.records)
            (args.output/(task+'.error.json')).write_text(json.dumps(failure,indent=2)+'\n')
            stages.append(dict(task=task,error=repr(error)));break
    status=dict(stages=stages,seconds=time.perf_counter()-started)
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(dict(event='DONE',**status)),flush=True)


if __name__=='__main__':main()

