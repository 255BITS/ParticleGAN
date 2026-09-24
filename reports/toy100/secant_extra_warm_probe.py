"""Predeclared three-condition local secant screen at the matched warm state."""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context,run_warm_variants
from reports.toy100.secant_extra_scratch import secant_extra


def variants():
    def factory(method,c=None):
        @contextmanager
        def activate(state,prefix):
            recorder,_=prefix
            recorder.enabled=c is not None
            if c is not None:
                recorder.c=c
                completed=state['completed_steps'];target=state['target_steps']
                def accounting(calls,outer):
                    state['declare_optimizer_accounting'](
                        calls=completed+calls+2*(target-completed-outer),moment_updates=target)
                recorder.accounting=accounting
            receipt=dict(method=method)
            if method=='identity':
                yield receipt
            else:
                with constant_rate_context(state) as rates:
                    receipt.update(rates)
                    yield receipt
                if recorder.enabled:receipt.update(recorder.receipt())
        return activate
    return {name:factory(name,c) for name,c in
            [('identity',None),('constant',None),('c025',.25),('c05',.5),('c09',.9)]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1200);args=parser.parse_args()
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    declaration=dict(methods=['identity','constant','c025','c05','c09'],prefix_steps=1000,
        total_steps=args.steps,seed=0,source={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__),ROOT/'reports/toy100/secant_extra_scratch.py',ROOT/'reports/toy100/fixed_metric_extra_scratch.py',
         ROOT/'reports/toy100/extra_adam_scratch.py',ROOT/'benchmarks/toy100/warm_equilibrium_probe.py',
         ROOT/'benchmarks/toy100/continuous_probe.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    result=run_warm_variants(config,variants(),output_dir=args.output,steps=args.steps,
        prefix_context=lambda:secant_extra(start_step=1000))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
