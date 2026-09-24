"""Cold ring (mode-hold) run of the alternating adapter with a per-update trace
of the twelve clean particle outputs. Diagnostic only; the gate verdict is the
unchanged host verdict."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reports.toy100.alternating_curvature_scratch import alternating_curvature


def main():
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--options',default='{}',help='JSON keyword options for alternating_curvature')
    args=parser.parse_args();torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='alternating_curvature_response',lr_floor=1.,lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
    recipe,noise,_=declared_recipe(config);options=json.loads(args.options)
    spec=next(job['spec'] for job in plan() if job['spec']['name']=='mode_hold')
    with alternating_curvature(task='mode_hold',trace_outputs=True,**options) as (recorder,_):
        result,_=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
    verdict=test_verdict(spec,result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(options=options,verdict=verdict,observations=result['observations'],
        receipt=recorder.receipt(),trace=recorder.trace,means=recorder.trace_means,diagnostic_only=True))+'\n')
    print(json.dumps(dict(status=verdict['status'],hq=[round(o['hq'],3) for o in result['observations']])),flush=True)


if __name__=='__main__':main()
