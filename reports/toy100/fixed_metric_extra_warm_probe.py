"""Compare sample reuse from one byte-identical scheduled passing state."""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import benchmarks.toy100
PROBE_ROOT=Path('/ml2/hypergan/ParticleGAN-continuous-probe')
benchmarks.toy100.__path__.append(str(PROBE_ROOT/'benchmarks/toy100'))
from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants
from benchmarks.toy100 import warm_equilibrium_probe as warm_module
from reports.toy100.fixed_metric_extra_scratch import fixed_metric_extra


def variants(config):
    def factory(method):
        @contextmanager
        def activate(state,prefix):
            recorder,_=prefix
            recorder.enabled=method in ('same_sample','independent_sample')
            recorder.same_sample=method=='same_sample'
            receipt=dict(method=method,constant_rates_after_passing_state=method!='identity')
            if recorder.enabled:
                state['declare_optimizer_accounting'](calls=1400,moment_updates=1200)
            if method=='identity':
                yield receipt
                return
            control=state['control']
            original_control=control.step
            def step(optimizer,completed_updates,role):
                original_control(optimizer,completed_updates,role)
                for group in optimizer.param_groups:
                    kind='d' if role=='d' else 'prior' if group['_comparison_prior'] else 'g'
                    group['lr']=config['lr']*dict(g=1.,d=config['d_lr_mult'],prior=config['prior_lr_mult'])[kind]
            with patch.object(control,'step',step):
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        return activate
    return {name:factory(name) for name in ('identity','constant','same_sample','independent_sample')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    declaration=dict(methods=['identity','constant','same_sample','independent_sample'],prefix_steps=1000,
        continuation_updates=200,seed=0,source={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__),ROOT/'reports/toy100/fixed_metric_extra_scratch.py',ROOT/'reports/toy100/extra_adam_scratch.py',
         PROBE_ROOT/'benchmarks/toy100/warm_equilibrium_probe.py',PROBE_ROOT/'benchmarks/toy100/continuous_probe.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    ordinary_probe=warm_module.run_probe
    def explicit_nonfinite(value,path='',found=None):
        if found is None:found=[]
        if isinstance(value,float) and not math.isfinite(value):
            found.append(dict(path=path,value=str(value)))
            return str(value)
        if isinstance(value,dict):
            return {key:explicit_nonfinite(item,f'{path}.{key}',found) for key,item in value.items()}
        if isinstance(value,(list,tuple)):
            return [explicit_nonfinite(item,f'{path}[{index}]',found) for index,item in enumerate(value)]
        return value
    def probe(*a,**kw):
        value=ordinary_probe(*a,**kw)
        found=[]
        value=explicit_nonfinite(value,found=found)
        value['nonfinite_host_values']=found
        return value
    original_receipt=None
    @contextmanager
    def prefix():
        with fixed_metric_extra(start_step=1000) as value:
            recorder,_=value
            original_receipt=recorder.receipt
            def receipt():
                found=[]
                result=explicit_nonfinite(original_receipt(),found=found)
                result['nonfinite_dynamics_values']=found
                return result
            recorder.receipt=receipt
            yield value
    with patch.object(warm_module,'run_probe',probe):
        result=run_warm_variants(config,variants(config),output_dir=args.output,
            prefix_context=prefix)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
