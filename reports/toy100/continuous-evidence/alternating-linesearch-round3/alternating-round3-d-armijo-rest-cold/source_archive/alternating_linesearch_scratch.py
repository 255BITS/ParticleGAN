"""Alternating PR82 G bound with verified same-sample D Armijo backtracking.

Only the D safeguard changes: the original G norm-curvature rule stays .25.
D advances Adam moments once, and trial steps interpolate that fixed proposal.
G's first gradient and moment update occur after D's actually accepted point.
Every replay executes the actual host block with the original minibatch/noise.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from reports.toy100 import alternating_curvature_scratch as adapter
from reports.toy100.alternating_linesearch_diagnosis import loss_in_host

METHOD='alternating_D_armijo_preserved_G_curvature'


class DLineSearchRecorder(adapter.AlternatingCurvatureRecorder):
    def __init__(self,start_step=0,curvature_bound=.25,d_armijo=.1,max_retries=12,
                 search_enabled=True,reject_exhausted=True,loss_reader=None):
        super().__init__(start_step=start_step,curvature_bound=curvature_bound,advantage_gate=None)
        if not 0<d_armijo<1:raise ValueError('invalid Armijo fraction')
        if type(max_retries) is not int or max_retries<0:raise ValueError('invalid retry budget')
        self.d_armijo=d_armijo;self.max_retries=max_retries;self.search_enabled=search_enabled
        self.reject_exhausted=reject_exhausted
        self.loss_reader=loss_in_host if loss_reader is None else loss_reader
        self.stage=None

    def _copy(self,opt):return [p.detach().clone() for p in self._params(opt)]

    @torch.no_grad()
    def _place(self,opt,values):
        for p,value in zip(self._params(opt),values):
            p.copy_(value)
            if not torch.isfinite(p).all():raise FloatingPointError('nonfinite trial point')

    def _check_rng(self):
        if not all(torch.equal(a,b) for a,b in zip(self.rng_after,self._rng(self.streams))):
            raise RuntimeError('same-sample line search changed training RNG progression')
        self.rng_replay_verified+=1

    def _prepare_replay(self):
        self._set_rng(self.streams,self.rng_before)
        with torch.no_grad():
            for buffer,value in self.buffers:buffer.copy_(value)
        self.phase=1

    def phases(self,step,opt_d,opt_g,local):
        if not self.enabled or step<self.start_step:
            self.passthrough=True
            try:yield 0
            finally:self.passthrough=False
            return
        if self.phase is not None or opt_d is opt_g:raise RuntimeError('invalid nested update')
        if self.optimizers is None:
            self.optimizers=(opt_d,opt_g)
            self.rows={opt:dict(role=role,calls=0,rates=[]) for role,opt in zip(('d','g'),self.optimizers)}
        if self.optimizers!=(opt_d,opt_g):raise RuntimeError('optimizers changed')
        self.streams=[x for x in local.values() if isinstance(x,torch.Generator)]
        policy=local.get('noise_policy')
        if policy is not None:self.streams.extend(x for name in ('input_stream','output_stream')
            if isinstance((x:=getattr(policy,name,None)),torch.Generator))
        self.streams=list({id(x):x for x in self.streams}.values())
        unique_buffers={id(b):b for name in ('generator','critic','prior')
                        if isinstance((m:=local.get(name)),torch.nn.Module) for b in m.buffers()}
        self.buffers=[(b,b.detach().clone()) for b in unique_buffers.values()]
        self.d0=self._copy(opt_d);self.g_base=self._copy(opt_g)
        self.rng_before=self._rng(self.streams);self.advantage=None
        self.row=dict(outer_step=self.outer_steps+1,d_trials=[])
        self.stage='d_base';self.phase=0
        yield 0
        self.rng_after=self._rng(self.streams)
        self.d_alpha=1.
        for retry in range(self.max_retries+1):
            self._place(opt_g,self.g_base)
            self._place(opt_d,[n if self.d_alpha==1 else torch.lerp(b,n,self.d_alpha) for b,n in zip(self.d0,self.d1)])
            self.stage='d_trial';self.d_accepted=False;self.retry=retry
            self._prepare_replay()
            yield 1
            self._check_rng()
            if self.d_accepted:break
            if retry==self.max_retries:
                if not self.reject_exhausted:
                    raise RuntimeError('D Armijo search exhausted declared retry budget')
                self.d_alpha=0.;self.stage='d_trial';self.retry=retry+1
                self._place(opt_d,self.d0);self._place(opt_g,self.g_base)
                self._prepare_replay()
                yield 1
                self._check_rng()
                if not self.d_accepted:raise RuntimeError('zero-step D replay was not accepted')
                break
            self.d_alpha*=.5
        # D was accepted inside its callback, so this same pass evaluated G
        # and advanced G's moments against precisely that accepted critic.
        if self.stage!='g_trial':raise RuntimeError('G did not respond to accepted D')
        self._prepare_replay()
        yield 1
        self._check_rng()
        if self.stage!='done':raise RuntimeError('G safeguard did not finish')
        self.row.update(critic_advantage=self.advantage,gate_open=False,
                        rho=self.row['g']['rho'],factor=self.row['g']['factor'])
        self.records.append(self.row)
        self.phase=None;self.stage=None;self.outer_steps+=1
        if self.accounting is not None:self.accounting(self.rows[opt_d]['calls'],self.outer_steps)

    @torch.no_grad()
    def step(self,optimizer,ordinary_step,closure=None):
        if self.passthrough:return ordinary_step(optimizer,closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError('optimizer callback outside declared update')
        opt_d,opt_g=self.optimizers;row=self.rows[optimizer];row['calls']+=1
        rates=[float(group['lr']) for group in optimizer.param_groups]
        if row['rates'] and rates!=row['rates'][0]:raise RuntimeError('nominal learning rate changed')
        if any(not math.isfinite(rate) or rate<=0 for rate in rates):raise ValueError('invalid rate')
        row['rates'].append(rates)
        for group in optimizer.param_groups:
            if group['betas'][0]!=0 or group.get('weight_decay',0) or any(group.get(k,False) for k in
                ('amsgrad','maximize','capturable','differentiable','decoupled_weight_decay','fused')):
                raise ValueError('requires ordinary CPU Adam with beta1=0')
        gradients=lambda:[p.grad.detach().clone() for p in self._params(optimizer)]
        if optimizer is opt_d:
            if self.stage=='d_base':
                self.gd0=gradients();self.ld0=self.loss_reader('d')
                if not math.isfinite(self.ld0) or any(not torch.isfinite(g).all() for g in self.gd0):
                    raise FloatingPointError('invalid base critic loss/gradient')
                ordinary_step(opt_d);self.d1=self._copy(opt_d)
                self.predicted_d=-sum(float((g.double()*(n-b).double()).sum()) for g,b,n in zip(self.gd0,self.d0,self.d1))
                if not math.isfinite(self.predicted_d) or self.predicted_d<0:
                    raise FloatingPointError('Adam proposal is not a descent direction')
                self.stage='d_proposed'
            elif self.stage=='d_trial':
                loss=self.loss_reader('d')
                if self.d_alpha==0 and (loss!=self.ld0 or any(not torch.equal(p.grad,g)
                    for p,g in zip(self._params(opt_d),self.gd0))):
                    raise RuntimeError('zero-step D replay changed base loss or gradient')
                tolerance=4*torch.finfo(self._params(opt_d)[0].dtype).eps*max(1.,abs(self.ld0),abs(loss))
                threshold=self.ld0-self.d_armijo*self.d_alpha*self.predicted_d
                accepted=math.isfinite(loss) and (not self.search_enabled or loss<=threshold+tolerance)
                self.row['d_trials'].append(dict(retry=self.retry,alpha=self.d_alpha,loss=loss,
                    loss_base=self.ld0,predicted_improvement=self.d_alpha*self.predicted_d,
                    armijo_threshold=threshold,rounding_tolerance=tolerance,accepted=accepted))
                if accepted:
                    self.d_accepted=True;self.d_star=self._copy(opt_d);self.stage='g_base'
                    self.row['d']=dict(alpha=self.d_alpha,backtracks=self.retry,
                        loss_before=self.ld0,loss_after=loss,predicted_improvement=self.d_alpha*self.predicted_d,
                        actual_improvement=self.ld0-loss,rejected_proposal=self.d_alpha==0,
                        zero_step_identity_verified=self.d_alpha==0)
            elif self.stage=='g_trial':
                if any(not torch.equal(p,v) for p,v in zip(self._params(opt_d),self.d_star)):
                    raise RuntimeError('critic moved during G curvature replay')
            return None
        if self.stage=='g_base':
            self.gg0=gradients()
            if any(not torch.isfinite(g).all() for g in self.gg0):raise FloatingPointError('nonfinite G base gradient')
            ordinary_step(opt_g);self.metric_g=adapter._metric(opt_g);self.g1=self._copy(opt_g)
            if any(not torch.isfinite(m).all() or not (m>0).all() for m in self.metric_g):
                raise FloatingPointError('invalid G Adam metric')
            self.stage='g_trial'
        elif self.stage=='g_trial':
            rho=adapter._rho(self.g_base,self.g1,self.gg0,gradients(),self.metric_g)
            factor=min(1.,self.curvature_bound/rho) if rho>0 else 1.
            self._place(opt_g,[n if factor==1 else torch.lerp(b,n,factor) for b,n in zip(self.g_base,self.g1)])
            self.row['g']=dict(rho=rho,factor=factor)
            self.stage='done'
        return None

    def receipt(self):
        optimizers=[]
        for opt,row in self.rows.items():
            optimizers.append(dict(role=row['role'],calls=row['calls'],rates=row['rates'][0] if row['rates'] else [],
                moment_steps=[[int(opt.state[p]['step']) for p in group['params']] for group in opt.param_groups]))
        return dict(method=METHOD,shared_gate_eligible=False,scratch_optimizer_policy=METHOD,
            outer_steps=self.outer_steps,records=self.records,optimizers=optimizers,host_source=self.host_source,
            d_armijo=self.d_armijo,max_retries=self.max_retries,curvature_bound=self.curvature_bound,
            search_enabled=self.search_enabled,reject_exhausted=self.reject_exhausted,rng_replay_verified=self.rng_replay_verified,
            moment_updates_per_outer_step=1,constant_nominal_rates=True,
            update_order='D accepted Armijo point, then G first gradient and Adam step, then unchanged PR82 G bound',
            scope='same-minibatch own-loss sufficient decrease, not a nonlinear game convergence guarantee')


@contextmanager
def alternating_linesearch(task='mode_hold',**options):
    with patch.object(adapter,'BothBoundRecorder',DLineSearchRecorder):
        with adapter.alternating_curvature(task=task,bound_d=True,**options) as value:
            yield value
