"""Cold acquisition for the declared skew arm, with explicit warm provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.toy100.projected_skew_scratch import projected_skew


def main():
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm-evidence',type=Path,required=True)
    parser.add_argument('--allow-transplant-diagnostic',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    warm=json.loads(args.warm_evidence.read_text());warm_arm=warm['variants']['projected_skew']
    exception=None
    if warm_arm['status']!='PASS':
        grade=warm_arm['local_stability']
        if not (args.allow_transplant_diagnostic and warm['identity_cold_parity']
            and grade['failing_steps']==[1001] and grade['passing_suffix']==199
            and warm_arm['stationary']['pass_all']):
            raise RuntimeError('warm failure does not meet the explicitly authorized transplant-only exception')
        exception='single transplant update failure followed by 199/199 stable; tests cold own dynamics'
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='rank_two_projected_skew_response',lr_floor=1.,lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
    recipe,noise,_=declared_recipe(config);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    files=[Path(__file__),ROOT/'reports/toy100/projected_skew_scratch.py',ROOT/'reports/toy100/implicit_extra_scratch.py',
        ROOT/'reports/toy100/fixed_metric_extra_scratch.py',ROOT/'reports/toy100/extra_adam_scratch.py']
    declaration=dict(seed=0,task_order=['trajectory','mode_hold'],budgets={'trajectory':400,'mode_hold':1200},
        stop_at_first_failed_host=True,shared_gate_eligible=False,
        scratch_optimizer_policy='rank_two_projected_implicit_skew_response',
        solver=dict(fd_relative=1e-4,rank=2,implicit_skew_coefficient=1.,no_line_search=True,
            fd_scaling='independent per-player central finite difference'),
        warm_status=warm_arm['status'],warm_evidence=str(args.warm_evidence),
        warm_evidence_sha256=hashlib.sha256(args.warm_evidence.read_bytes()).hexdigest(),
        intentional_exception=exception,gate_credit='Cold acquisition only; warm failure remains recorded',
        source={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    archive=args.output/'source_archive';archive.mkdir()
    for path in files:(archive/path.name).write_bytes(path.read_bytes())
    (archive/'warm-summary.json').write_bytes(args.warm_evidence.read_bytes())
    stages=[];started=time.perf_counter()
    for task in declaration['task_order']:
        spec=next(job['spec'] for job in plan() if job['spec']['name']==task)
        try:
            with projected_skew(task=task) as (recorder,source):
                (archive/(task+'_transformed.py')).write_text(source)
                result,context=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
            verdict=test_verdict(spec,result)
            data=dict(result=result,applied=context['applied'],noise=context['noise_receipt'],
                dynamics=recorder.receipt(),spec=spec,verdict=verdict,shared_gate_eligible=False,
                scratch_optimizer_policy='rank_two_projected_implicit_skew_response',
                intentional_exception=exception)
            (args.output/(task+'.json')).write_text(json.dumps(data,allow_nan=False)+'\n')
            stages.append(dict(task=task,verdict=verdict,live=result['live'],seconds=result['seconds']))
            print(json.dumps(dict(event='STAGE_DONE',**stages[-1])),flush=True)
            if not verdict['passed']:break
        except Exception as error:
            import traceback
            failure=dict(task=task,error=repr(error),traceback=traceback.format_exc(),
                partial_skew_steps=recorder.skew_steps,completed_outer_steps=recorder.outer_steps,queries=recorder.queries)
            (args.output/(task+'.error.json')).write_text(json.dumps(failure,indent=2)+'\n')
            stages.append(dict(task=task,error=repr(error)));break
    status=dict(stages=stages,seconds=time.perf_counter()-started,intentional_exception=exception)
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(dict(event='DONE',**status)),flush=True)


if __name__=='__main__':main()
