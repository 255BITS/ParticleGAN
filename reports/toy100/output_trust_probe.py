"""Three finite geometric output bounds, frozen full-budget hosts."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import nullcontext
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100.output_trust_scratch import output_trust


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
    for task in ('mode_hold','trajectory'):
        spec=next(job['spec'] for job in plan() if job['spec']['name']==task)
        dense=[]
        if task=='mode_hold':
            from benchmarks.locked_shared import mode_hold
            original_checkpoint=mode_hold.checkpoint
            def checkpoint(completed,measure):
                original_checkpoint(completed,measure)
                if completed>=1000 and completed%10==0:
                    with torch.random.fork_rng(devices=[]):
                        dense.append(dict(step=completed,**measure()))
            observer=patch.object(mode_hold,'checkpoint',checkpoint)
        else:
            observer=nullcontext()
        trust=(output_trust(row['multiplier']*noise['output_noise_std'])
               if row['multiplier'] is not None else nullcontext([]))
        with trust as receipt,observer:
            result,context=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
        verdict=test_verdict(spec,result)
        data=dict(result=result,applied=context['applied'],noise=context['noise_receipt'],
            trust=receipt,spec=spec,verdict=verdict,dense_tail=dense,
            dense_tail_pass=all(p['modes']==8 and p['hq']>=.9 for p in dense),production_gate_eligible=False,
            adapter_sha256=hashlib.sha256((ROOT/'reports/toy100/output_trust_scratch.py').read_bytes()).hexdigest())
        (output/(task+'.json')).write_text(json.dumps(data,indent=2)+'\n')
        stages.append(dict(task=task,verdict=verdict,live=result['live'],seconds=result['seconds']))
        print(json.dumps(dict(event='STAGE_DONE',name=row['name'],**stages[-1])),flush=True)
        if not verdict['passed']:
            break
    status=dict(name=row['name'],multiplier=row['multiplier'],stages=stages,seconds=time.perf_counter()-started)
    (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[dict(name=f'trust_{i}',multiplier=m,output=str(args.output)) for i,m in enumerate((.5,1.,2.,None))]
    (args.output/'declaration.json').write_text(json.dumps(dict(rows=rows,seed=0,task_order=['mode_hold','trajectory'],
        budgets={'mode_hold':1200,'trajectory':400},rule='Fixed output-noise-relative RMS radius, current training input batch; G-only Adam proposal interpolation; moments/prior retained',
        source={name:hashlib.sha256((ROOT/'reports/toy100'/name).read_bytes()).hexdigest() for name in ('output_trust_scratch.py','output_trust_probe.py')}),indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in as_completed([pool.submit(run,row) for row in rows]):
            print(json.dumps(dict(event='DONE',**result.result())),flush=True)


if __name__=='__main__':main()
