"""Frozen cold acquisition filter, cheap trajectory before mode hold."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100.secant_extra_scratch import secant_extra


def run(row):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy
    torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name=row['name'],lr_floor=1.,lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
    recipe,noise,_=declared_recipe(config)
    output=Path(row['output'])/row['name'];output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    stages=[]
    started=time.perf_counter()
    for task in ('trajectory','mode_hold'):
        spec=next(job['spec'] for job in plan() if job['spec']['name']==task)
        try:
            with secant_extra(task=task,c=row['c']) as (recorder,source):
                result,context=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
            verdict=test_verdict(spec,result)
            data=dict(result=result,applied=context['applied'],noise=context['noise_receipt'],
                dynamics=recorder.receipt(),spec=spec,verdict=verdict,production_gate_eligible=False,
                adapter_sha256=hashlib.sha256((ROOT/'reports/toy100/secant_extra_scratch.py').read_bytes()).hexdigest())
            (output/(task+'.json')).write_text(json.dumps(data,allow_nan=False)+'\n')
            stages.append(dict(task=task,verdict=verdict,live=result['live'],seconds=result['seconds']))
            print(json.dumps(dict(event='STAGE_DONE',name=row['name'],**stages[-1])),flush=True)
            if not verdict['passed']:break
        except Exception as error:
            import traceback
            failure=dict(task=task,error=repr(error),traceback=traceback.format_exc())
            (output/(task+'.error.json')).write_text(json.dumps(failure,indent=2)+'\n')
            stages.append(failure)
            break
    status=dict(name=row['name'],c=row['c'],stages=stages,seconds=time.perf_counter()-started)
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[dict(name=f'secant_c{int(c*100):02}',c=c,output=str(args.output)) for c in (.25,.5,.9)]
    (args.output/'declaration.json').write_text(json.dumps(dict(rows=rows,seed=0,task_order=['trajectory','mode_hold'],
        budgets={'mode_hold':1200,'trajectory':400},stop_at_first_failed_host=True,
        source={name:hashlib.sha256((ROOT/'reports/toy100'/name).read_bytes()).hexdigest() for name in
                ('secant_extra_scratch.py','fixed_metric_extra_scratch.py','extra_adam_scratch.py','secant_extra_probe.py')}),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in as_completed([pool.submit(run,row) for row in rows]):
            print(json.dumps(dict(event='DONE',**result.result())),flush=True)


if __name__=='__main__':main()
