"""One declared projected-skew arm; the warm gate must pass before cold use."""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context,run_warm_variants
from reports.toy100.projected_skew_scratch import projected_skew


def variants():
    def factory(method):
        @contextmanager
        def activate(state,prefix):
            recorder,_=prefix
            recorder.enabled=method in ('projected_skew','joint_gradient')
            recorder.correction=method=='projected_skew'
            completed=state['completed_steps'];target=state['target_steps']
            def accounting(calls,outer):
                state['declare_optimizer_accounting'](
                    calls=completed+calls+(target-completed-outer),moment_updates=target)
            recorder.accounting=accounting
            receipt=dict(method=method,shared_gate_eligible=False,
                scratch_optimizer_policy='rank_two_projected_implicit_skew_response' if recorder.enabled else 'control')
            if method=='identity':yield receipt
            else:
                with constant_rate_context(state) as rates:
                    receipt.update(rates);yield receipt
                if recorder.enabled:receipt.update(recorder.receipt())
        return activate
    return {name:factory(name) for name in ('identity','constant','joint_gradient','projected_skew')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    files=[Path(__file__),ROOT/'reports/toy100/projected_skew_scratch.py',ROOT/'reports/toy100/implicit_extra_scratch.py',
        ROOT/'reports/toy100/fixed_metric_extra_scratch.py',ROOT/'reports/toy100/extra_adam_scratch.py',
        ROOT/'benchmarks/toy100/warm_equilibrium_probe.py',ROOT/'benchmarks/toy100/continuous_probe.py',
        ROOT/'configs/toy100/constraints_simple_regularization.json']
    declaration=dict(methods=list(variants()),prefix_steps=1000,total_steps=1200,seed=0,
        shared_gate_eligible=False,scratch_optimizer_policy='rank_two_projected_implicit_skew_response',
        solver=dict(fd_relative=1e-4,rank=2,implicit_skew_coefficient=1.,no_line_search=True),
        filter_order='Consume warm projected_skew verdict PASS before any new cold trajectory run',
        source={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    declaration_path=args.output.with_suffix('.declaration.json')
    if args.output.exists() or declaration_path.exists():raise FileExistsError('refusing to overwrite evidence')
    declaration_path.write_text(json.dumps(declaration,indent=2)+'\n')
    archive=args.output.with_name(args.output.name+'-source');archive.mkdir()
    for path in files:
        target=archive/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    with projected_skew(start_step=1000) as (_,source):
        (archive/'generated_mode_hold.py').write_text(source)
    result=run_warm_variants(config,variants(),output_dir=args.output,
        prefix_context=lambda:projected_skew(start_step=1000))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
