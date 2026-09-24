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
    parser.add_argument('--ratio-loosen',type=float,default=None)
    parser.add_argument('--ratio-tighten',type=float,default=None)
    parser.add_argument('--acq-ratio',type=float,default=1.5)
    parser.add_argument('--rest-ratio',type=float,default=4.)
    parser.add_argument('--mode-loosen',type=float,default=None)
    parser.add_argument('--boost-steps',type=int,default=0)
    parser.add_argument('--boost-cap',type=float,default=None)
    parser.add_argument('--latent-nudge',action='store_true')
    parser.add_argument('--latent-step',type=float,default=.02)
    parser.add_argument('--center-critic',action='store_true')
    parser.add_argument('--stray-step',type=float,default=0.)
    parser.add_argument('--smooth-critic',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='alternating_curvature_response',lr_floor=1.,lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
    recipe,noise,_=declared_recipe(config);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    declaration=dict(seed=0,task_order=['trajectory','mode_hold'],budgets={'mode_hold':1200,'trajectory':400},
        stop_at_first_failed_host=True,shared_gate_eligible=False,scratch_optimizer_policy=METHOD,
        solver=dict(curvature_bound=args.curvature_bound,advantage_gate=args.advantage_gate,bound_d=args.bound_d,d_curvature_bound=args.d_curvature_bound,update_order='alternating'),
        source={name:hashlib.sha256((ROOT/'reports/toy100'/name).read_bytes()).hexdigest() for name in
                ('alternating_curvature_scratch.py','extra_adam_scratch.py','alternating_curvature_probe.py')})
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    stages=[];started=time.perf_counter()
    for task in declaration['task_order']:
        spec=next(job['spec'] for job in plan() if job['spec']['name']==task)
        try:
            extra=dict(bound_d=True,d_curvature_bound=args.d_curvature_bound,ratio_loosen=args.ratio_loosen,ratio_tighten=args.ratio_tighten,acq_ratio=args.acq_ratio,rest_ratio=args.rest_ratio,mode_loosen=args.mode_loosen,boost_steps=args.boost_steps,boost_cap=args.boost_cap,latent_nudge=args.latent_nudge,latent_step=args.latent_step,stray_step=args.stray_step) if args.bound_d else dict(advantage_gate=args.advantage_gate)
            with alternating_curvature(task=task,curvature_bound=args.curvature_bound,center_critic=args.center_critic,smooth_critic=args.smooth_critic,**extra) as (recorder,source):
                result,context=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
            verdict=test_verdict(spec,result)
            data=dict(result=result,applied=context['applied'],noise=context['noise_receipt'],
                dynamics=recorder.receipt(),spec=spec,verdict=verdict,shared_gate_eligible=False,
                scratch_optimizer_policy=METHOD)
            (args.output/(task+'.json')).write_text(json.dumps(data,allow_nan=False)+'\n')
            stages.append(dict(task=task,verdict=verdict,live=result['live'],seconds=result['seconds']))
            print(json.dumps(dict(event='STAGE_DONE',**stages[-1],early_acquisition=recorder.early_acquisition() if hasattr(recorder,'early_acquisition') else None)),flush=True)
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

