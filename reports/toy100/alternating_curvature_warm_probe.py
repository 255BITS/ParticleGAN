"""One bounded cross-only response at the matched passing state."""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context,run_warm_variants
from reports.toy100.alternating_curvature_scratch import alternating_curvature,METHOD


def variants():
    def factory(method):
        @contextmanager
        def activate(state,prefix):
            recorder,_=prefix;recorder.enabled=method=='alternating_curvature'
            completed=state['completed_steps'];target=state['target_steps']
            def accounting(calls,outer):
                state['declare_optimizer_accounting'](
                    calls=completed+calls+(target-completed-outer),moment_updates=target)
            recorder.accounting=accounting
            receipt=dict(method=method,shared_gate_eligible=False,
                scratch_optimizer_policy=METHOD if recorder.enabled else 'control')
            if method=='identity':yield receipt
            else:
                with constant_rate_context(state) as rates:
                    receipt.update(rates);yield receipt
                if recorder.enabled:receipt.update(recorder.receipt())
        return activate
    return {name:factory(name) for name in ('identity','constant','alternating_curvature')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--curvature-bound',type=float,default=1.)
    parser.add_argument('--advantage-gate',type=float)
    parser.add_argument('--bound-d',action='store_true')
    parser.add_argument('--d-curvature-bound',type=float)
    parser.add_argument('--ratio-reference',type=float)
    parser.add_argument('--per-group',action='store_true')
    parser.add_argument('--per-particle',action='store_true')
    parser.add_argument('--critic-average',type=float)
    parser.add_argument('--smooth-critic',type=float)
    parser.add_argument('--stencil-critic',action='store_true')
    parser.add_argument('--plain-curvature',action='store_true')
    parser.add_argument('--slope-reference',type=float)
    parser.add_argument('--slope-step-reference',type=float)
    parser.add_argument('--slope-gate',type=float);args=parser.parse_args()
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    declaration=dict(methods=['identity','constant','alternating_curvature'],prefix_steps=1000,total_steps=1200,
        seed=0,shared_gate_eligible=False,scratch_optimizer_policy=METHOD,
        solver=dict(curvature_bound=args.curvature_bound,advantage_gate=args.advantage_gate,bound_d=args.bound_d,d_curvature_bound=args.d_curvature_bound,ratio_reference=args.ratio_reference,ratio_span=1.5,ratio_decay=.9,per_group=args.per_group,per_particle=args.per_particle,critic_average=args.critic_average,smooth_critic=args.smooth_critic,smooth_samples=8,stencil_critic=args.stencil_critic,plain_curvature=args.plain_curvature,slope_reference=args.slope_reference,slope_step_reference=args.slope_step_reference,slope_gate=args.slope_gate,update_order='alternating'),
        source={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__),ROOT/'reports/toy100/alternating_curvature_scratch.py',ROOT/'reports/toy100/extra_adam_scratch.py',
         ROOT/'benchmarks/toy100/warm_equilibrium_probe.py',ROOT/'benchmarks/toy100/continuous_probe.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    result=run_warm_variants(config,variants(),output_dir=args.output,
        prefix_context=lambda:alternating_curvature(start_step=1000,curvature_bound=args.curvature_bound,**({'bound_d':True,'d_curvature_bound':args.d_curvature_bound,'ratio_reference':args.ratio_reference,'per_group':args.per_group,'per_particle':args.per_particle,'critic_average':args.critic_average,'smooth_critic':args.smooth_critic,'stencil_critic':args.stencil_critic,'plain_curvature':args.plain_curvature,'slope_reference':args.slope_reference,'slope_step_reference':args.slope_step_reference,'slope_gate':args.slope_gate} if args.bound_d else {'advantage_gate':args.advantage_gate})))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
