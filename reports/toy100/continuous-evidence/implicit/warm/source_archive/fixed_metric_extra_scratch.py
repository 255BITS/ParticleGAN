"""Joint extragradient with one Adam metric update and optional RNG replay.

For beta1=0, phase 0 computes v_t and a joint lookahead with ordinary Adam.
Phase 1 returns to the original weights and uses its fresh gradient with
the same v_t denominator. Moments advance once, gradients twice. This is
an explicitly declared practical variant, distinct from Gidel ExtraAdam.
"""
from contextlib import ExitStack,contextmanager
import ast
import math
from unittest.mock import patch
import torch
from reports.toy100.extra_adam_scratch import ExtraAdamRecorder,HOSTS,sha,transformed_function


class FixedMetricExtraRecorder(ExtraAdamRecorder):
    def __init__(self,same_sample=True,start_step=0):
        super().__init__('extra_adam')
        self.same_sample=same_sample
        self.metric={}
        self.rng_replay_verified=0
        self.start_step=start_step
        self.enabled=True
        self.passthrough=False
        self.output_motion=[]

    def _rng(self,streams):
        return [torch.get_rng_state().clone()]+[s.get_state().clone() for s in streams]

    def _set_rng(self,streams,states):
        torch.set_rng_state(states[0])
        for stream,state in zip(streams,states[1:]):stream.set_state(state)

    def phases(self,step,opt_d,opt_g,local):
        if not self.enabled or step<self.start_step:
            self.passthrough=True
            yield 0
            self.passthrough=False
            return
        streams=[value for value in local.values() if isinstance(value,torch.Generator)]
        policy=local.get('noise_policy')
        if policy is not None:
            streams.extend(value for name in ('input_stream','output_stream')
                           if isinstance((value:=getattr(policy,name,None)),torch.Generator))
        streams=list({id(stream):stream for stream in streams}.values())
        model=local.get('generator')
        clean=getattr(model,'model',model)
        prior=local.get('prior')
        measure_output='means' in local and prior is not None
        if measure_output:
            with torch.no_grad():
                original_z=prior.z.detach().clone()
                original_output=clean(original_z)
        rng_before=self._rng(streams)
        rng_after=None
        for phase in super().phases(step,opt_d,opt_g):
            if phase==1 and self.same_sample:
                self._set_rng(streams,rng_before)
            yield phase
            if measure_output:
                with torch.no_grad():
                    after_network=clean(original_z)
                    after_all=clean(prior.z)
                    self.output_motion.append(dict(outer_step=self.outer_steps+1,phase=phase,
                        g_output_rms=float((after_network.double()-original_output.double()).square().sum(1).mean().sqrt()),
                        prior_output_rms=float((after_all.double()-after_network.double()).square().sum(1).mean().sqrt()),
                        total_output_rms=float((after_all.double()-original_output.double()).square().sum(1).mean().sqrt())))
            if phase==0:
                rng_after=self._rng(streams)
            elif self.same_sample:
                if not all(torch.equal(a,b) for a,b in zip(rng_after,self._rng(streams))):
                    raise RuntimeError('Second evaluation consumed a different RNG pattern')
                self.rng_replay_verified+=1

    @torch.no_grad()
    def step(self,optimizer,ordinary_step,closure=None):
        if self.passthrough:
            return ordinary_step(optimizer,closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError('Gradient arrived outside the declared game point')
        if optimizer in self.pending or optimizer is not self.optimizers[len(self.pending)]:
            raise RuntimeError('Joint gradient order changed')
        gradients={}
        for group in optimizer.param_groups:
            if group['betas'][0]!=0 or group.get('weight_decay',0) or group.get('amsgrad',False):
                raise ValueError('Fixed-metric EG requires beta1=0 ordinary Adam without weight decay')
            for p in group['params']:
                if p.grad is None:raise RuntimeError('Missing game gradient')
                gradients[p]=p.grad.detach().clone()
        self.pending[optimizer]=gradients
        if optimizer is self.optimizers[0]:return
        if any(not torch.equal(p,saved) for p,saved in self.point.items()):
            raise RuntimeError('Player moved before joint gradient capture')
        self.joint_points_verified+=1
        if self.phase==1:
            for p,base in self.base.items():p.copy_(base)
            self.base_restores_verified+=1
        for target in self.optimizers:
            if self.phase==0:
                for p,g in self.pending[target].items():p.grad=g
                ordinary_step(target)
                for group in target.param_groups:
                    for p in group['params']:
                        moment=target.state[p]
                        denominator=(moment['exp_avg_sq']/(1-group['betas'][1]**float(moment['step']))).sqrt()+group['eps']
                        self.metric[p]=denominator
            else:
                for group in target.param_groups:
                    for p in group['params']:
                        p.grad=self.pending[target][p]
                        p.addcdiv_(p.grad,self.metric[p],value=-group['lr'])
            row=self.rows[target]
            rates=[float(group['lr']) for group in target.param_groups]
            row['rates'].append(rates);row['calls']+=1
            row['diagnostics'].append([dict(parameters=sum(p.numel() for p in group['params']),
                max_scaled_gradient=max(float((self.pending[target][p]/self.metric[p]).abs().max()) for p in group['params']),
                denominator_min=min(float(self.metric[p].min()) for p in group['params']),
                denominator_max=max(float(self.metric[p].max()) for p in group['params']),
                gradient_rms=math.sqrt(sum(float(self.pending[target][p].double().square().sum()) for p in group['params'])/sum(p.numel() for p in group['params'])),
                move_from_outer_base_rms=math.sqrt(sum(float((p-self.base[p]).double().square().sum()) for p in group['params'])/sum(p.numel() for p in group['params']))) for group in target.param_groups])
        self.pending={}

    def receipt(self):
        value=super().receipt()
        value.update(method='fixed_metric_same_sample_eg' if self.same_sample else 'fixed_metric_independent_sample_eg',
            paper='https://proceedings.mlr.press/v108/mishchenko20a.html',
            theoretical_scope='Practical Adam-metric variant, no convergence theorem claimed for this nonlinear game',
            moments_updated_at_every_gradient_evaluation=False,
            moment_updates_per_outer_step=1,metric='first-gradient bias-corrected Adam denominator held for both evaluations',
            same_sample=self.same_sample,rng_replay_verified=self.rng_replay_verified,
            clean_output_motion=self.output_motion)
        return value


@contextmanager
def fixed_metric_extra(task='mode_hold',same_sample=True,start_step=0):
    from benchmarks.locked_shared import mode_hold,trajectory
    module={'mode_hold':mode_hold,'trajectory':trajectory}[task]
    tree,_,original_sha=transformed_function(module,task)
    phase_calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                 and isinstance(node.func,ast.Attribute) and node.func.attr=='phases']
    if len(phase_calls)!=1:raise RuntimeError('Expected one declared phase iterator')
    phase_calls[0].args.append(ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))
    ast.fix_missing_locations(tree)
    source=ast.unparse(tree)+'\n'
    recorder=FixedMetricExtraRecorder(same_sample,start_step)
    recorder.host_source=dict(task=task,original_function_sha256=original_sha,
        generated_function_sha256=sha(source.encode()),source_transform='existing verified gradient-loop transform plus locals argument for RNG binding')
    original_step=torch.optim.Adam.step
    def patched_step(optimizer,closure=None):return recorder.step(optimizer,original_step,closure)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__,{'_extra_state':recorder}))
        namespace={}
        exec(compile(tree,f'<fixed-metric-extra-{task}>','exec'),module.__dict__,namespace)
        stack.enter_context(patch.object(module,HOSTS[task],namespace[HOSTS[task]]))
        stack.enter_context(patch.object(torch.optim.Adam,'step',patched_step))
        yield recorder,source
