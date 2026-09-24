"""Same-sample joint EG with a reversible, local operator secant test.

One first-gradient Adam metric update per outer step. With P=diag(lr/denom),
try x_trial=x-alpha*P*F(x) and require
    ||F(x_trial)-F(x)||_P <= c ||F(x)||_P.
An accepted update is x-alpha*P*F(x_trial). Rejected trials halve alpha;
the next outer step tries twice the previous accepted alpha, capped at one.
This is state-dependent line search, not a training-age schedule. The host
game can be nonmonotone, so no convergence guarantee is claimed.
"""
from contextlib import ExitStack, contextmanager
import ast
import math
from unittest.mock import patch

import torch

from reports.toy100.fixed_metric_extra_scratch import FixedMetricExtraRecorder
from reports.toy100.extra_adam_scratch import HOSTS, sha, transformed_function


class SecantExtraRecorder(FixedMetricExtraRecorder):
    def __init__(self, c=.5, start_step=0, max_backtracks=20):
        super().__init__(same_sample=True,start_step=start_step)
        if not math.isfinite(c) or not 0 < c < 1:
            raise ValueError('secant c must lie strictly between zero and one')
        if type(max_backtracks) is not int or max_backtracks < 0:
            raise ValueError('max_backtracks must be a nonnegative integer')
        self.c=c
        self.max_backtracks=max_backtracks
        self.last_scale=1.
        self.trials=[]
        self.accounting=None

    @torch.no_grad()
    def _apply(self, gradients):
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.copy_(self.base[p])
                    p.addcdiv_(gradients[optimizer][p],self.metric[p],value=-self.scale*group['lr'])
                    if not torch.isfinite(p).all():
                        raise FloatingPointError('nonfinite secant EG parameters')

    def phases(self,step,opt_d,opt_g,local):
        if not self.enabled or step < self.start_step:
            self.passthrough=True
            try:yield 0
            finally:self.passthrough=False
            return
        if self.phase is not None or opt_d is opt_g:
            raise RuntimeError('invalid nested game update')
        if self.optimizers is None:
            self.optimizers=(opt_d,opt_g)
            parameters=[p for opt in self.optimizers for group in opt.param_groups for p in group['params']]
            if len(parameters)!=len(set(parameters)):
                raise RuntimeError('game players share parameters')
            self.rows={opt:dict(role=role,calls=0,rates=[],diagnostics=[]) for role,opt in zip(('d','g'),self.optimizers)}
        if self.optimizers!=(opt_d,opt_g):raise RuntimeError('game optimizer changed')
        streams=[value for value in local.values() if isinstance(value,torch.Generator)]
        policy=local.get('noise_policy')
        if policy is not None:
            streams.extend(value for name in ('input_stream','output_stream')
                           if isinstance((value:=getattr(policy,name,None)),torch.Generator))
        streams=list({id(stream):stream for stream in streams}.values())
        rng_before=self._rng(streams)
        self.base=self._copy_parameters()
        self.scale=min(1.,2*self.last_scale)
        self.pending={}
        self.phase=0
        self.point=self._copy_parameters()
        model=local.get('generator');clean=getattr(model,'model',model);prior=local.get('prior')
        measure='means' in local and prior is not None
        if measure:
            with torch.no_grad():
                original_z=prior.z.detach().clone()
                original_output=clean(original_z).double()
        yield 0
        if self.pending:raise RuntimeError('incomplete base gradient')
        rng_after=self._rng(streams)
        for retry in range(self.max_backtracks+1):
            self.retry=retry
            self.phase=1
            self.point=self._copy_parameters()
            self._set_rng(streams,rng_before)
            yield 1
            if self.pending:raise RuntimeError('incomplete trial gradient')
            if not all(torch.equal(a,b) for a,b in zip(rng_after,self._rng(streams))):
                raise RuntimeError('secant trial consumed a different RNG pattern')
            self.rng_replay_verified+=1
            if self.accepted:break
            if retry==self.max_backtracks:
                raise RuntimeError('secant backtracking exhausted its declared budget')
            self.scale*=.5
            self._apply(self.first_gradients)
        if measure:
            with torch.no_grad():
                network=clean(original_z).double();both=clean(prior.z).double()
                self.output_motion.append(dict(outer_step=self.outer_steps+1,phase='accepted',
                    g_output_rms=float((network-original_output).square().sum(1).mean().sqrt()),
                    prior_output_rms=float((both-network).square().sum(1).mean().sqrt()),
                    total_output_rms=float((both-original_output).square().sum(1).mean().sqrt())))
        self.last_scale=self.scale
        self.phase=None
        self.base,self.point={},{}
        self.outer_steps+=1
        if self.accounting is not None:self.accounting(self.rows[opt_d]['calls'],self.outer_steps)

    @torch.no_grad()
    def step(self,optimizer,ordinary_step,closure=None):
        if self.passthrough:return ordinary_step(optimizer,closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError('gradient outside declared game point')
        if optimizer in self.pending or optimizer is not self.optimizers[len(self.pending)]:
            raise RuntimeError('joint gradient order changed')
        gradients={}
        for group in optimizer.param_groups:
            if (group['betas'][0]!=0 or group.get('weight_decay',0) or any(group.get(key,False)
                    for key in ('amsgrad','maximize','capturable','differentiable','decoupled_weight_decay','fused'))):
                raise ValueError('secant EG requires ordinary CPU Adam, beta1=0')
            if not math.isfinite(group['lr']) or group['lr']<=0:raise ValueError('invalid constant rate')
            for p in group['params']:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise FloatingPointError('missing or nonfinite game gradient')
                gradients[p]=p.grad.detach().clone()
        self.pending[optimizer]=gradients
        if optimizer is self.optimizers[0]:return
        if any(not torch.equal(p,saved) for p,saved in self.point.items()):
            raise RuntimeError('player moved before joint gradient capture')
        self.joint_points_verified+=1
        if self.phase==0:
            self.first_gradients={opt:{p:g.clone() for p,g in values.items()} for opt,values in self.pending.items()}
            for target in self.optimizers:
                for p,g in self.pending[target].items():p.grad=g
                ordinary_step(target)
                for group in target.param_groups:
                    for p in group['params']:
                        state=target.state[p]
                        denominator=(state['exp_avg_sq']/(1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']
                        if not torch.isfinite(denominator).all() or not (denominator>0).all():
                            raise FloatingPointError('invalid first-gradient Adam metric')
                        self.metric[p]=denominator
            self._apply(self.first_gradients)
        else:
            first_sq=0.;difference_sq=0.
            for target in self.optimizers:
                for group in target.param_groups:
                    for p in group['params']:
                        first=self.first_gradients[target][p].double()
                        difference=self.pending[target][p].double()-first
                        weight=group['lr']/self.metric[p].double()
                        first_sq+=float((first.square()*weight).sum())
                        difference_sq+=float((difference.square()*weight).sum())
            ratio=math.sqrt(difference_sq/first_sq) if first_sq else (0. if not difference_sq else math.inf)
            self.accepted=math.isfinite(ratio) and ratio<=self.c
            self.trials.append(dict(outer_step=self.outer_steps+1,retry=self.retry,scale=self.scale,
                secant_ratio=ratio if math.isfinite(ratio) else str(ratio),base_field_metric_norm=math.sqrt(first_sq),
                accepted=self.accepted))
            if self.accepted:
                self._apply(self.pending)
                self.base_restores_verified+=1
        for target in self.optimizers:
            row=self.rows[target];row['calls']+=1
            rates=[float(group['lr']) for group in target.param_groups]
            if row['rates'] and rates!=row['rates'][0]:raise RuntimeError('constant rate changed')
            row['rates'].append(rates)
            row['diagnostics'].append([dict(parameters=sum(p.numel() for p in group['params']),
                max_scaled_gradient=max(float((self.pending[target][p].double()/self.metric[p].double()).abs().max()) for p in group['params']),
                denominator_min=min(float(self.metric[p].min()) for p in group['params']),
                denominator_max=max(float(self.metric[p].max()) for p in group['params']),
                gradient_rms=math.sqrt(sum(float(self.pending[target][p].double().square().sum()) for p in group['params'])/sum(p.numel() for p in group['params']))) for group in target.param_groups])
        self.pending={}

    def receipt(self):
        value=super().receipt()
        value.update(method='same_sample_metric_secant_eg',c=self.c,max_backtracks=self.max_backtracks,
            trials=self.trials,step_adaptation='halve rejected trial; double last accepted scale next update, cap one',
            secant_condition='norm(Ftrial-Fbase,P) <= c*norm(Fbase,P), P=diag(group_lr/first_Adam_denominator)',
            gradient_evaluations_per_player_per_outer_step=None,
            extra_gradient_evaluations_per_player_per_outer_step=None,
            accepted_scales=[row['scale'] for row in self.trials if row['accepted']],
            rejected_trials=sum(not row['accepted'] for row in self.trials))
        return value


@contextmanager
def secant_extra(task='mode_hold',c=.5,start_step=0,max_backtracks=20):
    from benchmarks.locked_shared import mode_hold,trajectory
    module={'mode_hold':mode_hold,'trajectory':trajectory}[task]
    tree,_,original_sha=transformed_function(module,task)
    phase_calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                 and isinstance(node.func,ast.Attribute) and node.func.attr=='phases']
    if len(phase_calls)!=1:raise RuntimeError('expected one phase iterator')
    phase_calls[0].args.append(ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))
    ast.fix_missing_locations(tree)
    source=ast.unparse(tree)+'\n'
    recorder=SecantExtraRecorder(c,start_step,max_backtracks)
    recorder.host_source=dict(task=task,original_function_sha256=original_sha,generated_function_sha256=sha(source.encode()),source_transform='verified gradient-loop transform plus locals for RNG binding')
    original_step=torch.optim.Adam.step
    def patched_step(optimizer,closure=None):return recorder.step(optimizer,original_step,closure)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__,{'_extra_state':recorder}))
        namespace={}
        exec(compile(tree,f'<secant-extra-{task}>','exec'),module.__dict__,namespace)
        stack.enter_context(patch.object(module,HOSTS[task],namespace[HOSTS[task]]))
        stack.enter_context(patch.object(torch.optim.Adam,'step',patched_step))
        yield recorder,source
