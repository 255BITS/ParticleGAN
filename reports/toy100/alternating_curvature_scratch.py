"""Alternating Adam with a critic-gated same-sample G own-curvature bound.

Every recorder-based game update tested before this (extragradient, implicit,
cross-only, error-relative guards) evaluates D and G at one joint point: a
simultaneous game. With its own bound disabled, that simultaneous scaffold
already fails cold trajectory (MSE .228), while the host's alternating
constant Adam passes (.003). This adapter keeps the host's alternation: D
takes its ordinary Adam step, then G takes its ordinary Adam step against the
new D. Nothing else changes while the critic is clearly winning.

When the critic's base-point advantage log 2 - L_D (raw relativistic D loss,
before penalties) is below ``advantage_gate``, the host block is replayed once
with identical data/noise at (D_new, G_new) with D's replayed step discarded.
With P = lr / Adam denominator and Delta the G+prior Adam step,

    rho = ||sqrt(P) (F_G(D_new, G_new) - F_G(D_new, G_base))|| / ||Delta / sqrt(P)||

is the effective preconditioned step times own curvature along the step, and
G+prior is placed at G_base + min(1, c / rho) Delta. Moments advance once. The
target is never read, and there is no elapsed-time schedule.
"""
from contextlib import ExitStack,contextmanager
import ast
import math
from unittest.mock import patch

import torch

from particlegan.gan_loss import GANLoss
from reports.toy100.extra_adam_scratch import HOSTS,sha,transformed_function

METHOD='alternating_adam_with_critic_gated_own_curvature_bound'


class AlternatingCurvatureRecorder:
    def __init__(self,start_step=0,curvature_bound=.25,advantage_gate=.1,trace_outputs=False):
        if not math.isfinite(curvature_bound) or curvature_bound<=0:raise ValueError('invalid curvature bound')
        if advantage_gate is not None and (not math.isfinite(advantage_gate) or advantage_gate<=0):
            raise ValueError('invalid critic-advantage gate')
        self.start_step=start_step;self.curvature_bound=curvature_bound;self.advantage_gate=advantage_gate
        self.enabled=True;self.passthrough=False;self.phase=None;self.optimizers=None
        self.rows={};self.records=[];self.outer_steps=0;self.rng_replay_verified=0
        self.advantage=None;self.accounting=None;self.host_source=None
        self.trace_outputs=trace_outputs;self.trace=[]

    @staticmethod
    def _rng(streams):
        return [torch.get_rng_state().clone()]+[s.get_state().clone() for s in streams]

    @staticmethod
    def _set_rng(streams,states):
        torch.set_rng_state(states[0])
        for stream,state in zip(streams,states[1:]):stream.set_state(state)

    @torch.no_grad()
    def _record_trace(self,local):
        if not self.trace_outputs or 'means' not in local:return
        model=local['generator'];clean=getattr(model,'model',model)
        self.trace.append([[round(float(v),5) for v in row] for row in clean(local['prior'].z).tolist()])
        if not hasattr(self,'trace_means'):self.trace_means=local['means'].tolist()

    def _params(self,opt):
        return [p for group in opt.param_groups for p in group['params']]

    def phases(self,step,opt_d,opt_g,local):
        if not self.enabled or step<self.start_step:
            self.passthrough=True
            try:yield 0
            finally:self.passthrough=False
            return
        if self.optimizers is None:
            self.optimizers=(opt_d,opt_g)
            self.rows={opt:dict(role=role,calls=0) for role,opt in zip(('d','g'),self.optimizers)}
        if self.optimizers!=(opt_d,opt_g):raise RuntimeError('game optimizers changed')
        streams=[v for v in local.values() if isinstance(v,torch.Generator)]
        policy=local.get('noise_policy')
        if policy is not None:streams.extend(v for name in ('input_stream','output_stream')
            if isinstance((v:=getattr(policy,name,None)),torch.Generator))
        streams=list({id(s):s for s in streams}.values())
        buffers=[(b,b.detach().clone()) for name in ('generator','critic','prior')
                 if isinstance((m:=local.get(name)),torch.nn.Module) for b in m.buffers()]
        self.g_base=[p.detach().clone() for p in self._params(opt_g)]
        rng_before=self._rng(streams)
        self.advantage=None;self.phase=0;self.pending=None
        yield 0
        rng_after=self._rng(streams)
        row=dict(outer_step=self.outer_steps+1,critic_advantage=self.advantage)
        gate_open=(self.advantage_gate is not None and self.advantage is not None
                   and self.advantage>=self.advantage_gate)
        row['gate_open']=gate_open
        if not gate_open:
            if self.advantage_gate is not None and self.advantage is None:
                raise RuntimeError('critic advantage was not observed at the base point')
            g_new=[p.detach().clone() for p in self._params(opt_g)]
            d_new=[p.detach().clone() for p in self._params(opt_d)]
            self._set_rng(streams,rng_before);self.phase=1
            yield 1
            if not all(torch.equal(a,b) for a,b in zip(rng_after,self._rng(streams))):
                raise RuntimeError('replayed block consumed a different RNG pattern')
            self.rng_replay_verified+=1
            if any(not torch.equal(p,v) for p,v in zip(self._params(opt_d),d_new)):
                raise RuntimeError('D moved during G curvature replay')
            with torch.no_grad():
                num=den=0.
                for p,base,new,g0,g1,metric in zip(self._params(opt_g),self.g_base,g_new,self.g0,self.g1,self.metric):
                    delta=(new-base).double()
                    num+=float((metric*(g1-g0).double().square()).sum())
                    den+=float((delta.square()/metric).sum())
                rho=math.sqrt(num/den) if den>0 else 0.
                if not math.isfinite(rho):raise FloatingPointError('nonfinite own-curvature ratio')
                factor=min(1.,self.curvature_bound/rho) if rho>0 else 1.
                for p,base,new in zip(self._params(opt_g),self.g_base,g_new):
                    p.copy_(torch.lerp(base,new,factor) if factor<1 else new)
                for b,saved in buffers:b.copy_(saved)
            row.update(rho=rho,factor=factor)
        self.records.append(row);self._record_trace(local)
        self.phase=None;self.outer_steps+=1
        if self.accounting is not None:self.accounting(self.rows[opt_d]['calls'],self.outer_steps)

    @torch.no_grad()
    def step(self,optimizer,ordinary_step,closure=None):
        if self.passthrough:return ordinary_step(optimizer,closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError('optimizer call outside the declared game update')
        self.rows[optimizer]['calls']+=1
        opt_d,opt_g=self.optimizers
        if self.phase==0:
            if optimizer is opt_g:
                self.g0=[p.grad.detach().clone() for p in self._params(opt_g)]
            result=ordinary_step(optimizer)
            if optimizer is opt_g:
                self.metric=[]
                for group in opt_g.param_groups:
                    for p in group['params']:
                        state=opt_g.state[p]
                        denominator=(state['exp_avg_sq']/(1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']
                        self.metric.append(group['lr']/denominator.double())
            return result
        if optimizer is opt_g:
            self.g1=[p.grad.detach().clone() for p in self._params(opt_g)]
        return None

    def receipt(self):
        closed=[r for r in self.records if not r['gate_open']]
        def stats(values):
            values=[v for v in values if v is not None]
            return dict(min=min(values),mean=sum(values)/len(values),max=max(values)) if values else None
        return dict(method=METHOD,scratch_optimizer_policy=METHOD,shared_gate_eligible=False,
            curvature_bound=self.curvature_bound,d_curvature_bound=getattr(self,'d_curvature_bound',None),advantage_gate=self.advantage_gate,
            outer_steps=self.outer_steps,gate_open=len(self.records)-len(closed),bound_evaluated=len(closed),
            bound_active=sum(r['factor']<1 for r in closed),rho=stats([r['rho'] for r in closed]),
            factor=stats([r['factor'] for r in closed]),
            d_rho=stats([r.get('d',{}).get('rho') for r in closed]),d_factor=stats([r.get('d',{}).get('factor') for r in closed]),
            d_bound_active=sum(r.get('d',{}).get('factor',1.)<1 for r in closed),
            ratio_reference=getattr(self,'ratio_reference',None),ratio_span=getattr(self,'ratio_span',None),
            ratio_decay=getattr(self,'ratio_decay',None),g_bound=stats([r.get('g_bound') for r in closed]),
            smoothed_ratio=stats([r.get('smoothed_ratio') for r in closed]),critic_advantage=stats([r['critic_advantage'] for r in self.records]),
            gradient_evaluations_per_outer_step=(sum(3 if 'd' in r else 1 if r['gate_open'] else 2 for r in self.records)/self.outer_steps) if self.outer_steps else None,
            moment_updates_per_outer_step=1,rng_replay_verified=self.rng_replay_verified,
            update_order='alternating: D Adam step, then G+prior Adam step against the new D',
            host_source=self.host_source,records=self.records)


def _metric(opt):
    values=[]
    for group in opt.param_groups:
        for p in group['params']:
            state=opt.state[p]
            denominator=(state['exp_avg_sq']/(1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']
            values.append(group['lr']/denominator.double())
    return values


def _rho(base,new,g0,g1,metric):
    num=den=0.
    for b,n,a,c,m in zip(base,new,g0,g1,metric):
        num+=float((m*(c-a).double().square()).sum());den+=float(((n-b).double().square()/m).sum())
    rho=math.sqrt(num/den) if den>0 else 0.
    if not math.isfinite(rho):raise FloatingPointError('nonfinite own-curvature ratio')
    return rho


class BothBoundRecorder(AlternatingCurvatureRecorder):
    """Alternating Adam with the same-sample own-curvature bound on D and G.

    Pass 0: D takes its ordinary Adam step (D0 -> D1); G's gradient is ignored.
    Pass 1 (same data/noise) at (D1, G0): D's own field change gives rho_D and
    D* = D0 + min(1, c/rho_D)(D1 - D0); G's gradient at (D*, G0) then takes
    G's ordinary Adam step (G0 -> G1). Pass 2 at (D*, G1) gives rho_G and
    G = G0 + min(1, c/rho_G)(G1 - G0). Each player's moments advance once and
    G still responds to the D that actually materializes.

    With ``ratio_reference`` the G bound is c * clip(r_ref / r, 1/span, span),
    where r is an exponential moving average (decay ``ratio_decay``) of
    log(rho_G / rho_D). The measured ratio is about 1-2 while either host is
    acquiring and 6-18 once matched, so the bound loosens during acquisition
    and tightens at rest. It is a function of the current state only.
    """

    def __init__(self,start_step=0,curvature_bound=.25,d_curvature_bound=None,trace_outputs=False,
                 ratio_reference=None,ratio_span=1.5,ratio_decay=.9):
        super().__init__(start_step=start_step,curvature_bound=curvature_bound,advantage_gate=None,trace_outputs=trace_outputs)
        if ratio_reference is not None and (not math.isfinite(ratio_reference) or ratio_reference<=0
                                            or not ratio_span>=1 or not 0<=ratio_decay<1):
            raise ValueError('invalid curvature-ratio controller')
        self.ratio_reference=ratio_reference;self.ratio_span=ratio_span;self.ratio_decay=ratio_decay
        self.log_ratio=None
        self.d_curvature_bound=curvature_bound if d_curvature_bound is None else d_curvature_bound
        if not math.isfinite(self.d_curvature_bound) or self.d_curvature_bound<=0:raise ValueError('invalid D curvature bound')

    def phases(self,step,opt_d,opt_g,local):
        if not self.enabled or step<self.start_step:
            self.passthrough=True
            try:yield 0
            finally:self.passthrough=False
            return
        if self.optimizers is None:
            self.optimizers=(opt_d,opt_g)
            self.rows={opt:dict(role=role,calls=0) for role,opt in zip(('d','g'),self.optimizers)}
        if self.optimizers!=(opt_d,opt_g):raise RuntimeError('game optimizers changed')
        streams=[v for v in local.values() if isinstance(v,torch.Generator)]
        policy=local.get('noise_policy')
        if policy is not None:streams.extend(v for name in ('input_stream','output_stream')
            if isinstance((v:=getattr(policy,name,None)),torch.Generator))
        streams=list({id(s):s for s in streams}.values())
        buffers=[(b,b.detach().clone()) for name in ('generator','critic','prior')
                 if isinstance((m:=local.get(name)),torch.nn.Module) for b in m.buffers()]
        self.d0=[p.detach().clone() for p in self._params(opt_d)]
        self.g_base=[p.detach().clone() for p in self._params(opt_g)]
        rng_before=self._rng(streams);self.advantage=None;self.row=dict(outer_step=self.outer_steps+1)
        rng_after=None
        for phase in range(3):
            if phase:
                self._set_rng(streams,rng_before)
                with torch.no_grad():
                    for b,saved in buffers:b.copy_(saved)
            self.phase=phase
            yield phase
            state=self._rng(streams)
            if rng_after is None:rng_after=state
            elif not all(torch.equal(a,b) for a,b in zip(rng_after,state)):
                raise RuntimeError('replayed block consumed a different RNG pattern')
            else:self.rng_replay_verified+=1
        self.row['critic_advantage']=self.advantage;self.row['gate_open']=False
        self.row['rho'],self.row['factor']=self.row['g']['rho'],self.row['g']['factor']
        self.records.append(self.row);self._record_trace(local)
        self.phase=None;self.outer_steps+=1
        if self.accounting is not None:self.accounting(self.rows[opt_d]['calls'],self.outer_steps)

    @torch.no_grad()
    def step(self,optimizer,ordinary_step,closure=None):
        if self.passthrough:return ordinary_step(optimizer,closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError('optimizer call outside the declared game update')
        self.rows[optimizer]['calls']+=1
        opt_d,opt_g=self.optimizers
        grads=lambda opt:[p.grad.detach().clone() for p in self._params(opt)]
        if optimizer is opt_d:
            if self.phase==0:
                self.gd0=grads(opt_d);ordinary_step(opt_d)
                self.metric_d=_metric(opt_d);self.d1=[p.detach().clone() for p in self._params(opt_d)]
                for p,v in zip(self._params(opt_g),self.g_base):p.copy_(v)
            elif self.phase==1:
                rho=_rho(self.d0,self.d1,self.gd0,grads(opt_d),self.metric_d)
                factor=min(1.,self.d_curvature_bound/rho) if rho>0 else 1.
                for p,b,n in zip(self._params(opt_d),self.d0,self.d1):p.copy_(torch.lerp(b,n,factor) if factor<1 else n)
                self.d_star=[p.detach().clone() for p in self._params(opt_d)]
                self.row['d']=dict(rho=rho,factor=factor)
            else:
                for p,v in zip(self._params(opt_d),self.d_star):p.copy_(v)
            return None
        if self.phase==0:
            for p,v in zip(self._params(opt_g),self.g_base):p.copy_(v)
            for p,v in zip(self._params(opt_d),self.d1):p.copy_(v)
        elif self.phase==1:
            self.gg0=grads(opt_g);ordinary_step(opt_g)
            self.metric_g=_metric(opt_g);self.g1=[p.detach().clone() for p in self._params(opt_g)]
        else:
            rho=_rho(self.g_base,self.g1,self.gg0,grads(opt_g),self.metric_g)
            bound=self.curvature_bound
            if self.ratio_reference is not None:
                rho_d=self.row['d']['rho']
                if rho>0 and rho_d>0:
                    current=math.log(rho/rho_d)
                    self.log_ratio=current if self.log_ratio is None else self.ratio_decay*self.log_ratio+(1-self.ratio_decay)*current
                if self.log_ratio is not None:
                    scale=min(self.ratio_span,max(1/self.ratio_span,self.ratio_reference/math.exp(self.log_ratio)))
                    bound=self.curvature_bound*scale
                self.row['g_bound']=bound;self.row['smoothed_ratio']=None if self.log_ratio is None else math.exp(self.log_ratio)
            factor=min(1.,bound/rho) if rho>0 else 1.
            for p,b,n in zip(self._params(opt_g),self.g_base,self.g1):p.copy_(torch.lerp(b,n,factor) if factor<1 else n)
            self.row['g']=dict(rho=rho,factor=factor)
        return None


@contextmanager
def alternating_curvature(task='mode_hold',bound_d=False,**options):
    from benchmarks.locked_shared import mode_hold,trajectory
    module={'mode_hold':mode_hold,'trajectory':trajectory}[task]
    tree,_,original_sha=transformed_function(module,task)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='phases']
    if len(calls)!=1:raise RuntimeError('expected one phase iterator')
    calls[0].args.append(ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]));ast.fix_missing_locations(tree)
    source=ast.unparse(tree)+'\n';recorder=(BothBoundRecorder if bound_d else AlternatingCurvatureRecorder)(**options)
    recorder.host_source=dict(task=task,original_function_sha256=original_sha,generated_function_sha256=sha(source.encode()))
    ordinary_step=torch.optim.Adam.step
    original_d_loss=GANLoss.d_loss
    def observed_d_loss(gan,real_logits,fake_logits):
        value=original_d_loss(gan,real_logits,fake_logits)
        if recorder.phase==0 and recorder.advantage is None:
            recorder.advantage=math.log(2)-float(value.detach())
        return value
    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__,{'_extra_state':recorder}));namespace={}
        exec(compile(tree,f'<alternating-curvature-{task}>','exec'),module.__dict__,namespace)
        stack.enter_context(patch.object(module,HOSTS[task],namespace[HOSTS[task]]))
        stack.enter_context(patch.object(torch.optim.Adam,'step',
            lambda optimizer,closure=None:recorder.step(optimizer,ordinary_step,closure)))
        stack.enter_context(patch.object(GANLoss,'d_loss',observed_d_loss))
        yield recorder,source
