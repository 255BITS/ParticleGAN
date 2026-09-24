"""One declared coupled implicit response, tested from a shared passing state."""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import benchmarks.toy100
PROBE_ROOT=Path('/ml2/hypergan/ParticleGAN-continuous-probe')
benchmarks.toy100.__path__.append(str(PROBE_ROOT/'benchmarks/toy100'))
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context,run_warm_variants
from reports.toy100.implicit_extra_scratch import implicit_extra


def variants():
    def factory(method):
        @contextmanager
        def activate(state,prefix):
            recorder,_=prefix;recorder.enabled=method=='implicit'
            completed=state['completed_steps'];target=state['target_steps']
            def accounting(calls,outer):
                state['declare_optimizer_accounting'](
                    calls=completed+calls+(target-completed-outer),moment_updates=target)
            recorder.accounting=accounting
            receipt=dict(method=method,shared_gate_eligible=False,
                scratch_optimizer_policy='same_sample_linearized_implicit_response' if recorder.enabled else 'control')
            if method=='identity':yield receipt
            else:
                with constant_rate_context(state) as rates:
                    receipt.update(rates)
                    yield receipt
                if recorder.enabled:receipt.update(recorder.receipt())
        return activate
    return {name:factory(name) for name in ('identity','constant','implicit')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1200);args=parser.parse_args()
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    declaration=dict(methods=['identity','constant','implicit'],prefix_steps=1000,total_steps=args.steps,
        seed=0,shared_gate_eligible=False,scratch_optimizer_policy='same_sample_linearized_implicit_response',
        solver=dict(krylov_dim=8,linear_tolerance=.1,nonlinear_tolerance=.5,fd_relative=1e-4,
                    correction_limit=2.,max_backtracks=8),
        source={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__),ROOT/'reports/toy100/implicit_extra_scratch.py',ROOT/'reports/toy100/fixed_metric_extra_scratch.py',
         ROOT/'reports/toy100/extra_adam_scratch.py',PROBE_ROOT/'benchmarks/toy100/warm_equilibrium_probe.py',
         PROBE_ROOT/'benchmarks/toy100/continuous_probe.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    result=run_warm_variants(config,variants(),output_dir=args.output,steps=args.steps,
        prefix_context=lambda:implicit_extra(start_step=1000))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
