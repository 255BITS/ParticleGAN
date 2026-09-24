"""Read-only accepted-point diagnostics for the frozen PR82 G.25/D3 arm."""
from contextlib import contextmanager
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reports.toy100 import alternating_curvature_scratch as adapter


def loss_in_host(role):
    frame=inspect.currentframe().f_back
    while frame is not None and not all(k in frame.f_locals for k in ('opt_d','opt_g','d_loss','gan')):
        frame=frame.f_back
    if frame is None:raise RuntimeError('host loss unavailable')
    value=float(frame.f_locals[role+'_loss'].detach());del frame
    return value


class AcceptedPointObserver(adapter.BothBoundRecorder):
    def __init__(self,**options):
        super().__init__(**options);self.accepted_points=[];self.observing=False

    def step(self,optimizer,ordinary_step,closure=None):
        if self.passthrough:return super().step(optimizer,ordinary_step,closure)
        role='d' if optimizer is self.optimizers[0] else 'g'
        if self.observing:
            self.observed[role]=dict(loss=loss_in_host(role),grads=[p.grad.detach().clone() for p in self._params(optimizer)])
            return None
        if self.phase==0 and role=='d':self.loss_d0=loss_in_host(role)
        if self.phase==1 and role=='d':self.loss_d1=loss_in_host(role)
        if self.phase==1 and role=='g':self.loss_g0=loss_in_host(role)
        if self.phase==2 and role=='g':self.loss_g1=loss_in_host(role)
        return super().step(optimizer,ordinary_step,closure)

    def phases(self,step,opt_d,opt_g,local):
        streams=[x for x in local.values() if isinstance(x,torch.Generator)]
        policy=local.get('noise_policy')
        if policy is not None:streams.extend(x for name in ('input_stream','output_stream')
            if isinstance((x:=getattr(policy,name,None)),torch.Generator))
        streams=list({id(x):x for x in streams}.values());before=self._rng(streams)
        buffers=[(b,b.detach().clone()) for name in ('generator','critic','prior')
                 if isinstance((m:=local.get(name)),torch.nn.Module) for b in m.buffers()]
        yield from super().phases(step,opt_d,opt_g,local)
        after=self._rng(streams);accepted_g=[p.detach().clone() for p in self._params(opt_g)]
        self.observing=True
        observations=[]
        for point in (self.g_base,accepted_g):
            with torch.no_grad():
                for p,value in zip(self._params(opt_g),point):p.copy_(value)
                for b,value in buffers:b.copy_(value)
            self.observed={};self._set_rng(streams,before)
            yield 1
            if not all(torch.equal(a,b) for a,b in zip(after,self._rng(streams))):
                raise RuntimeError('accepted-point observation changed training RNG progression')
            observations.append(self.observed)
        self.observing=False
        rows={}
        for role,base,new,accepted,g0,metric,l0,l1,observation in (
            ('d',self.d0,self.d1,self.d_star,self.gd0,self.metric_d,self.loss_d0,self.loss_d1,observations[0]['d']),
            ('g',self.g_base,self.g1,accepted_g,self.gg0,self.metric_g,self.loss_g0,self.loss_g1,observations[1]['g'])):
            predicted=work=delta_metric=change_metric=0.
            for b,n,g,a,m in zip(base,accepted,g0,observation['grads'],metric):
                delta=(n-b).double();change=(a-g).double()
                predicted-=float((g.double()*delta).sum());work+=float((change*delta).sum())
                delta_metric+=float((delta.square()/m).sum());change_metric+=float((m*change.square()).sum())
            gain=l0-observation['loss']
            rows[role]=dict(loss_base=l0,loss_full_proposal=l1,loss_accepted=observation['loss'],
                actual_improvement=gain,predicted_improvement=predicted,
                armijo_efficiency=gain/predicted if predicted else None,
                directional_secant=work/predicted if predicted else None,
                accepted_norm_curvature=math.sqrt(change_metric/delta_metric) if delta_metric else 0.,
                original_bound=self.records[-1][role])
        self.accepted_points.append(dict(step=step+1,d=rows['d'],g=rows['g']))

    def receipt(self):
        return dict(**super().receipt(),accepted_point_diagnostic=self.accepted_points,
                    extra_observer_field_evaluations_per_outer=2,observer_preserves_rng=True)


def main():
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--observe',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    config.update(name='pr82_accepted_point_diagnosis',lr_floor=1.,lr_anneal_start=0.)
    config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
    recipe,noise,_=declared_recipe(config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    declaration=dict(method='existing PR82 G.25/D3 accepted-point diagnosis',observe=args.observe,
        shared_gate_eligible=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),seed=0,
        new_candidate=False,steps=1200,noise_horizon=1200)
    args.output.with_suffix('.declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    spec=next(job['spec'] for job in plan() if job['spec']['name']=='mode_hold')
    recorder_type=AcceptedPointObserver if args.observe else adapter.BothBoundRecorder
    with patch.object(adapter,'BothBoundRecorder',recorder_type):
        with adapter.alternating_curvature(curvature_bound=.25,bound_d=True,d_curvature_bound=3.,trace_outputs=True) as (recorder,source):
            result,context=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
    raw=dict(result=result,verdict=test_verdict(spec,result),receipt=recorder.receipt(),noise=context['noise_receipt'],
        trace=recorder.trace,trace_means=recorder.trace_means,applied=context['applied'],declaration=declaration)
    args.output.write_text(json.dumps(raw,allow_nan=False)+'\n')
    print(json.dumps(dict(verdict=raw['verdict'],live=result['live'])),flush=True)


if __name__=='__main__':main()
