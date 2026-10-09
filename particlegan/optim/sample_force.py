"""Damped empirical output-kernel force filtering, without target geometry.

Four matrix-free conjugate-gradient iterations approximate
u = lambda (J J.T + lambda I)^-1 f, lambda = ||J.T f||² / ||f||².
Replace only the output pullback J.T f by J.T u in the existing loss gradient.
Direct parameter losses, FullDualNorm, rates and protected losses are retained.
This is a local preconditioner; it certifies neither finite descent nor fidelity.
"""
from copy import deepcopy
import math

import torch

from .direction_blend import DirectionBlendOptimizer


def sample_force_filter(loss, outputs, parameters):
    """Return a pullback correction and numerical diagnostics; draw no RNG."""
    outputs, parameters = tuple(outputs), tuple(parameters)
    if not outputs or any(not x.requires_grad for x in outputs):
        raise ValueError('sample-force filtering requires active public host outputs')
    force = torch.autograd.grad(loss, outputs, retain_graph=True, allow_unused=True)
    force = tuple(torch.zeros_like(x) if f is None else f.detach() for x,f in zip(outputs,force))
    def dot(a,b):return sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    def pullback(vector):
        values = torch.autograd.grad(outputs,parameters,vector,retain_graph=True,allow_unused=True)
        return tuple(torch.zeros_like(p) if v is None else v.detach() for p,v in zip(parameters,values))
    original = pullback(force)
    energy = dot(force,force)
    ridge = float(dot(original,original)/energy) if bool(energy>0) else 0.
    if not math.isfinite(ridge):raise ValueError('nonfinite sample-force ridge')
    if ridge == 0:
        return tuple(torch.zeros_like(p) for p in parameters),dict(ridge=0.,iterations=0,relative_residual=0.,
            correction_ratio=0.,filtered_force_ratio=1.)
    cotangent = tuple(torch.zeros_like(x,requires_grad=True) for x in outputs)
    transpose = torch.autograd.grad(outputs,parameters,cotangent,create_graph=True,
                                    retain_graph=True,allow_unused=True)
    def pushforward(vector):
        product = sum((g*v).sum() for g,v in zip(transpose,vector) if g is not None and g.requires_grad)
        if not isinstance(product,torch.Tensor):return tuple(torch.zeros_like(x) for x in outputs)
        values = torch.autograd.grad(product,cotangent,retain_graph=True,allow_unused=True)
        return tuple(torch.zeros_like(x) if v is None else v.detach() for x,v in zip(outputs,values))
    def operator(vector):return tuple(j+ridge*v for j,v in zip(pushforward(pullback(vector)),vector))
    rhs=tuple(ridge*f for f in force);solution=tuple(torch.zeros_like(f) for f in force)
    residual=rhs;direction=rhs;initial=dot(rhs,rhs);rr=initial;iterations=0
    for _ in range(4):
        if bool(rr<=initial*1e-12):break
        applied=operator(direction);denominator=dot(direction,applied)
        if not bool(torch.isfinite(denominator)) or not bool(denominator>0):
            raise ValueError('sample-force CG requires a finite positive operator')
        alpha=float(rr/denominator)
        solution=tuple(x+alpha*p for x,p in zip(solution,direction))
        residual=tuple(r-alpha*a for r,a in zip(residual,applied));next_rr=dot(residual,residual)
        beta=float(next_rr/rr);direction=tuple(r+beta*p for r,p in zip(residual,direction))
        rr=next_rr;iterations+=1
    corrected=pullback(solution)
    correction=tuple(c-o for c,o in zip(corrected,original))
    if any(not bool(torch.isfinite(c).all()) for c in correction):
        raise ValueError('nonfinite sample-force correction')
    return correction,dict(ridge=ridge,iterations=iterations,
        relative_residual=float((rr/initial).sqrt()),
        correction_ratio=float((dot(correction,correction)/dot(original,original)).sqrt()),
        filtered_force_ratio=float((dot(solution,solution)/energy).sqrt()))


class SampleForceOptimizer(DirectionBlendOptimizer):
    """Filter the real batch's output forces, then apply retained direction blend."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._force_correction=None
        self.sample_force_stats=dict(steps=0,filtered_steps=0,cg_iterations=0,
            max_relative_residual=0.,ridge_sum=0.,correction_ratio_sum=0.,filtered_force_ratio_sum=0.)

    def bind_sample_force(self,loss,outputs):
        if self._force_correction is not None:raise ValueError('sample-force loss already pending')
        if outputs is None:raise ValueError('host omitted sample-force output consumer')
        correction,stats=sample_force_filter(loss,outputs,self._parameters())
        self._force_correction=correction
        self._force_measure=stats

    @torch.no_grad()
    def step(self,closure=None):
        if self._force_correction is None:raise ValueError('sample-force backward is required')
        correction,measure=self._force_correction,self._force_measure
        try:
            for p,c in zip(self._parameters(),correction):
                if p.grad is None:p.grad=c.clone()
                else:p.grad.add_(c)
            result=super().step(closure)
            stats=self.sample_force_stats;stats['steps']+=1
            stats['filtered_steps']+=int(measure['correction_ratio']>0)
            stats['cg_iterations']+=measure['iterations']
            stats['max_relative_residual']=max(stats['max_relative_residual'],measure['relative_residual'])
            for key in ('ridge','correction_ratio','filtered_force_ratio'):stats[key+'_sum']+=measure[key]
            return result
        finally:self._force_correction=None;self._force_measure=None

    def state_dict(self):
        if self._force_correction is not None:raise ValueError('checkpoint only between complete sample-force updates')
        result=super().state_dict()
        result['sample_force']=dict(schema=1,mode='kernel_ridge_cg4',stats=deepcopy(self.sample_force_stats))
        return result

    def validate_state_dict(self,saved):
        base=dict(saved);meta=base.pop('sample_force',None)
        if (not isinstance(meta,dict) or set(meta)!={'schema','mode','stats'} or meta['schema']!=1
                or meta['mode']!='kernel_ridge_cg4' or set(meta['stats'])!=set(self.sample_force_stats)):
            raise ValueError('invalid sample-force checkpoint')
        for key,value in meta['stats'].items():
            if key in ('steps','filtered_steps','cg_iterations'):
                if type(value) is not int or value<0:raise ValueError('invalid sample-force count')
            elif type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('invalid sample-force statistic')
        super().validate_state_dict(base)

    def load_state_dict(self,saved):
        super().load_state_dict(saved)
        self.sample_force_stats=deepcopy(saved['sample_force']['stats'])
        self._force_correction=None;self._force_measure=None
