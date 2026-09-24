"""Matrix-free linearized implicit joint game response with measured residuals.

P is the first-gradient Adam metric including each role's constant LR.
In u=P^(-1/2) delta coordinates, GMRES solves
    (I + alpha sqrt(P) J_F sqrt(P)) u = -alpha sqrt(P) F0.
Finite-difference field evaluations replay the original host block/data/RNG.
The accepted point must also satisfy the actual nonlinear implicit residual.
This is a practical safeguarded linearization, not exact implicit Euler and
not a convergence theorem for the nonlinear, potentially nonmonotone host.
"""
from contextlib import ExitStack,contextmanager
import ast
import math
from unittest.mock import patch

import torch

from reports.toy100.fixed_metric_extra_scratch import FixedMetricExtraRecorder
from reports.toy100.extra_adam_scratch import HOSTS,sha,transformed_function


class ImplicitExtraRecorder(FixedMetricExtraRecorder):
    def __init__(self,start_step=0,krylov_dim=8,linear_tolerance=.1,
                 nonlinear_tolerance=.5,fd_relative=1e-4,correction_limit=2.,max_backtracks=8):
        super().__init__(same_sample=True,start_step=start_step)
        if type(krylov_dim) is not int or krylov_dim<1:raise ValueError('invalid Krylov dimension')
        if type(max_backtracks) is not int or max_backtracks<0:raise ValueError('invalid retry budget')
        if not 0<linear_tolerance<nonlinear_tolerance<1:raise ValueError('invalid residual tolerances')
        if not math.isfinite(fd_relative) or fd_relative<=0:raise ValueError('invalid finite difference scale')
        if not math.isfinite(correction_limit) or correction_limit<1:raise ValueError('invalid correction bound')
        self.krylov_dim=krylov_dim;self.linear_tolerance=linear_tolerance
        self.nonlinear_tolerance=nonlinear_tolerance;self.fd_relative=fd_relative
        self.correction_limit=correction_limit;self.max_backtracks=max_backtracks
        self.last_scale=1.;self.solves=[];self.queries=[];self.accounting=None

    def _flat(self,values):return torch.cat([values[p].detach().flatten().double() for p in self.parameters])

    @torch.no_grad()
    def _set_point(self,u):
        displacement=self.root_metric*u
        offset=0
        for p in self.parameters:
            p.copy_((self.base[p].double()+displacement[offset:offset+p.numel()].reshape_as(p)).to(p.dtype))
            if not torch.isfinite(p).all():raise FloatingPointError('nonfinite implicit trial parameters')
            offset+=p.numel()
        for buffer,saved in self.buffers:buffer.copy_(saved)
        self.point=self._copy_parameters()
        return (self._flat(self.point)-self.base_flat)/self.root_metric

    def _evaluate(self,u,kind):
        actual_u=self._set_point(u)
        self._set_rng(self.streams,self.rng_before)
        self.phase=1
        yield 1
        if self.pending:raise RuntimeError('incomplete implicit field evaluation')
        if not all(torch.equal(a,b) for a,b in zip(self.rng_after,self._rng(self.streams))):
            raise RuntimeError('implicit field consumed a different RNG pattern')
        self.rng_replay_verified+=1
        self.queries.append(dict(outer_step=self.outer_steps+1,kind=kind,
            parameter_displacement_norm=float(((self._flat(self.point)-self.base_flat).square().sum()).sqrt())))
        return self.root_metric*self._flat(self.field),actual_u

    def _solve(self,scale,q0):
        rhs=-scale*q0;beta=float(torch.linalg.vector_norm(rhs))
        vectors=[rhs/beta]
        h=torch.zeros((self.krylov_dim+1,self.krylov_dim),dtype=torch.float64)
        target=torch.zeros(self.krylov_dim+1,dtype=torch.float64);target[0]=beta
        fd=[]
        for column in range(self.krylov_dim):
            direction=self.root_metric*vectors[column]
            desired=self.fd_relative*(1+float(torch.linalg.vector_norm(self.base_flat)))
            epsilon=desired/float(torch.linalg.vector_norm(direction))
            q,actual_u=yield from self._evaluate(epsilon*vectors[column],'jvp')
            actual_delta=self.root_metric*actual_u
            error=float(torch.linalg.vector_norm(actual_delta-epsilon*direction))/desired
            if not math.isfinite(error) or error>.05:
                raise FloatingPointError('float32 finite-difference perturbation lost direction/scale')
            fd.append(dict(epsilon=epsilon,desired_parameter_norm=desired,
                achieved_parameter_norm=float(torch.linalg.vector_norm(actual_delta)),rounding_relative_error=error))
            image=vectors[column]+scale*(q-q0)/epsilon
            # Two-pass modified Gram-Schmidt avoids a false small residual
            # caused by loss of basis orthogonality.
            for _ in range(2):
                for index,vector in enumerate(vectors):
                    coefficient=torch.dot(vector,image)
                    h[index,column]+=coefficient
                    image-=coefficient*vector
            norm=float(torch.linalg.vector_norm(image));h[column+1,column]=norm
            active=h[:column+2,:column+1]
            coefficients=torch.linalg.lstsq(active,target[:column+2],driver='gelsd').solution
            solution=torch.stack(vectors,dim=1)@coefficients
            residual=float(torch.linalg.vector_norm(active@coefficients-target[:column+2]))/beta
            if residual<=self.linear_tolerance or norm<1e-12:break
            vectors.append(image/norm)
        return solution,dict(krylov_iterations=column+1,linear_relative_residual=residual,finite_differences=fd)

    def phases(self,step,opt_d,opt_g,local):
        if not self.enabled or step<self.start_step:
            self.passthrough=True
            try:yield 0
            finally:self.passthrough=False
            return
        if self.phase is not None or opt_d is opt_g:raise RuntimeError('invalid nested game update')
        if self.optimizers is None:
            self.optimizers=(opt_d,opt_g)
            self.parameters=[p for opt in self.optimizers for group in opt.param_groups for p in group['params']]
            if len(self.parameters)!=len(set(self.parameters)):raise RuntimeError('game players share parameters')
            self.rows={opt:dict(role=role,calls=0,rates=[],diagnostics=[]) for role,opt in zip(('d','g'),self.optimizers)}
        if self.optimizers!=(opt_d,opt_g):raise RuntimeError('game optimizers changed')
        self.streams=[value for value in local.values() if isinstance(value,torch.Generator)]
        policy=local.get('noise_policy')
        if policy is not None:self.streams.extend(value for name in ('input_stream','output_stream')
            if isinstance((value:=getattr(policy,name,None)),torch.Generator))
        self.streams=list({id(stream):stream for stream in self.streams}.values())
        unique_buffers={id(buffer):buffer for name in ('generator','critic','prior')
                        if isinstance((module:=local.get(name)),torch.nn.Module) for buffer in module.buffers()}
        self.buffers=[(buffer,buffer.detach().clone()) for buffer in unique_buffers.values()]
        self.base=self._copy_parameters();self.point=self._copy_parameters();self.base_flat=self._flat(self.base)
        self.rng_before=self._rng(self.streams);self.phase=0;self.pending={}
        model=local.get('generator');clean=getattr(model,'model',model);prior=local.get('prior')
        measure='means' in local and prior is not None
        if measure:
            with torch.no_grad():
                original_z=prior.z.detach().clone();original_output=clean(original_z).double()
        yield 0
        if self.pending:raise RuntimeError('incomplete first game gradient')
        self.rng_after=self._rng(self.streams)
        q0=self.root_metric*self._flat(self.field)
        qnorm=float(torch.linalg.vector_norm(q0));scale=min(1.,2*self.last_scale)
        if qnorm==0:
            self._set_point(torch.zeros_like(q0))
            self.solves.append(dict(outer_step=self.outer_steps+1,scale=1.,accepted=True,zero_field=True,
                krylov_iterations=0,linear_relative_residual=0.,nonlinear_relative_residual=0.))
            scale=1.
        else:
            for retry in range(self.max_backtracks+1):
                solution,diagnostic=yield from self._solve(scale,q0)
                diagnostic.update(outer_step=self.outer_steps+1,retry=retry,scale=scale,accepted=False)
                correction_ratio=float(torch.linalg.vector_norm(solution))/(scale*qnorm)
                diagnostic['correction_to_explicit_norm_ratio']=correction_ratio
                if diagnostic['linear_relative_residual']>self.linear_tolerance:
                    diagnostic['rejection']='linear_residual'
                elif not math.isfinite(correction_ratio) or correction_ratio>self.correction_limit:
                    diagnostic['rejection']='correction_bound'
                else:
                    final_q,actual_u=yield from self._evaluate(solution,'nonlinear_residual')
                    residual=float(torch.linalg.vector_norm(actual_u+scale*final_q))/(scale*qnorm)
                    diagnostic['nonlinear_relative_residual']=residual
                    if math.isfinite(residual) and residual<=self.nonlinear_tolerance:
                        diagnostic['accepted']=True
                    else:diagnostic['rejection']='nonlinear_residual'
                self.solves.append(diagnostic)
                if diagnostic['accepted']:break
                if retry==self.max_backtracks:
                    raise RuntimeError('implicit response exhausted declared residual/backtracking budget')
                scale*=.5
        self.last_scale=scale;self.base_restores_verified+=1
        if measure:
            with torch.no_grad():
                network=clean(original_z).double();both=clean(prior.z).double()
                row=dict(outer_step=self.outer_steps+1,
                    g_output_rms=float((network-original_output).square().sum(1).mean().sqrt()),
                    prior_output_rms=float((both-network).square().sum(1).mean().sqrt()),
                    total_output_rms=float((both-original_output).square().sum(1).mean().sqrt()))
                if not all(math.isfinite(value) for value in row.values()):raise FloatingPointError('nonfinite clean output motion')
                self.output_motion.append(row)
        self.phase=None;self.base,self.point={},{};self.outer_steps+=1
        if self.accounting is not None:self.accounting(self.rows[opt_d]['calls'],self.outer_steps)

    @torch.no_grad()
    def step(self,optimizer,ordinary_step,closure=None):
        if self.passthrough:return ordinary_step(optimizer,closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:raise RuntimeError('field outside declared point')
        if optimizer in self.pending or optimizer is not self.optimizers[len(self.pending)]:raise RuntimeError('joint field order changed')
        gradients={}
        for group in optimizer.param_groups:
            if (group['betas'][0]!=0 or group.get('weight_decay',0) or any(group.get(key,False)
                    for key in ('amsgrad','maximize','capturable','differentiable','decoupled_weight_decay','fused'))):
                raise ValueError('implicit response requires ordinary CPU Adam, beta1=0')
            if not math.isfinite(group['lr']) or group['lr']<=0:raise ValueError('invalid constant rate')
            for p in group['params']:
                if p.grad is None or not torch.isfinite(p.grad).all():raise FloatingPointError('missing/nonfinite game gradient')
                gradients[p]=p.grad.detach().clone()
        self.pending[optimizer]=gradients
        if optimizer is self.optimizers[0]:return
        if any(not torch.equal(p,saved) for p,saved in self.point.items()):raise RuntimeError('player moved before joint field capture')
        self.joint_points_verified+=1
        self.field={p:g for values in self.pending.values() for p,g in values.items()}
        if self.phase==0:
            metric={}
            for target in self.optimizers:
                for p,g in self.pending[target].items():p.grad=g
                ordinary_step(target)
                for group in target.param_groups:
                    for p in group['params']:
                        state=target.state[p]
                        denominator=(state['exp_avg_sq']/(1-group['betas'][1]**float(state['step']))).sqrt()+group['eps']
                        if not torch.isfinite(denominator).all() or not (denominator>0).all():raise FloatingPointError('invalid Adam metric')
                        self.metric[p]=denominator;metric[p]=group['lr']/denominator.double()
            self.root_metric=self._flat(metric).sqrt()
            for p,saved in self.base.items():p.copy_(saved)
        for target in self.optimizers:
            row=self.rows[target];row['calls']+=1;rates=[float(group['lr']) for group in target.param_groups]
            if row['rates'] and rates!=row['rates'][0]:raise RuntimeError('constant rate changed')
            row['rates'].append(rates)
            row['diagnostics'].append([dict(parameters=sum(p.numel() for p in group['params']),
                max_scaled_gradient=max(float((self.pending[target][p].double()/self.metric[p].double()).abs().max()) for p in group['params'])) for group in target.param_groups])
        self.pending={}

    def receipt(self):
        value=super().receipt()
        value.update(method='same_sample_linearized_implicit_response',krylov_dim=self.krylov_dim,
            linear_tolerance=self.linear_tolerance,nonlinear_tolerance=self.nonlinear_tolerance,
            fd_relative=self.fd_relative,correction_limit=self.correction_limit,max_backtracks=self.max_backtracks,
            solves=self.solves,queries=self.queries,accepted_scales=[row['scale'] for row in self.solves if row['accepted']],
            gradient_evaluations_per_player_per_outer_step=None,extra_gradient_evaluations_per_player_per_outer_step=None,
            output_weights='accepted nonlinear-residual-checked joint implicit proposal',
            shared_gate_eligible=False,scratch_optimizer_policy='same_sample_linearized_implicit_response',
            paper='https://f-t-s.github.io/projects/cgd/',
            reference_scope='Full-Jacobian regularized Newton comparison in author explanation; this is not cross-only CGD',
            correction_bound_scope='GMRES proposed solution before dtype rounding; nonlinear residual uses actual rounded parameters',
            theoretical_scope='Matrix-free finite-difference linearization with measured residuals; no convergence theorem claimed')
        return value


@contextmanager
def implicit_extra(task='mode_hold',**options):
    from benchmarks.locked_shared import mode_hold,trajectory
    module={'mode_hold':mode_hold,'trajectory':trajectory}[task]
    tree,_,original_sha=transformed_function(module,task)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='phases']
    if len(calls)!=1:raise RuntimeError('expected one phase iterator')
    calls[0].args.append(ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]));ast.fix_missing_locations(tree)
    source=ast.unparse(tree)+'\n';recorder=ImplicitExtraRecorder(**options)
    recorder.host_source=dict(task=task,original_function_sha256=original_sha,generated_function_sha256=sha(source.encode()),source_transform='verified gradient-loop transform plus locals for RNG binding')
    ordinary_step=torch.optim.Adam.step
    def patched_step(optimizer,closure=None):return recorder.step(optimizer,ordinary_step,closure)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__,{'_extra_state':recorder}));namespace={}
        exec(compile(tree,f'<implicit-extra-{task}>','exec'),module.__dict__,namespace)
        stack.enter_context(patch.object(module,HOSTS[task],namespace[HOSTS[task]]))
        stack.enter_context(patch.object(torch.optim.Adam,'step',patched_step))
        yield recorder,source
