"""Scratch rank-two implicit skew response in a fixed per-update Adam metric.

This is inspired by SGA, not the published LRSGA recurrence. With q=sqrt(P)F,
v=q/||q|| and w=orth(J_cross v), two central finite differences determine
a=(w.T J_cross v - v.T J_cross w)/2. The rank-two antisymmetric operator is
A=a*(w v.T-v w.T). Solve (I+A)u=-q, then apply theta += sqrt(P)u.

Own-player Hessians do not enter this approximation. Within the measured
plane the symmetric cross response cancels exactly, up to finite differences.
The inverse skew response cannot amplify the metric step norm. No nonlinear
convergence guarantee, objective oracle, acceptance gate or time schedule is
claimed. Every field query replays the unchanged detached host gradient block.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from reports.toy100 import implicit_extra_scratch as implicit_module
from reports.toy100.fixed_metric_extra_scratch import FixedMetricExtraRecorder
from reports.toy100.implicit_extra_scratch import ImplicitExtraRecorder


class ProjectedSkewRecorder(ImplicitExtraRecorder):
    def __init__(self,start_step=0,fd_relative=1e-4,correction=True):
        super().__init__(start_step=start_step,fd_relative=fd_relative)
        self.correction=correction
        self.skew_steps=[]

    def _cross_product(self,direction,label,q0):
        physical=self.root_metric*direction
        desired=self.fd_relative*(1+float(torch.linalg.vector_norm(self.base_flat)))
        epsilon=desired/float(torch.linalg.vector_norm(physical))
        n_d=sum(p.numel() for group in self.optimizers[0].param_groups for p in group['params'])
        diagnostics=[];values=[]
        for sign in (1.,-1.):
            parts=[]
            for role,section in (('d',slice(None,n_d)),('g',slice(n_d,None))):
                point=torch.zeros_like(direction)
                point[section]=sign*epsilon*direction[section]
                if not torch.count_nonzero(point):
                    parts.append(q0)
                    continue
                field,actual=yield from super()._evaluate(point,label+'_'+role+('_plus' if sign>0 else '_minus'))
                planned=self.root_metric*point;observed=self.root_metric*actual
                norm=float(torch.linalg.vector_norm(planned))
                error=float(torch.linalg.vector_norm(observed-planned))/norm
                if not math.isfinite(error) or error>.05:
                    raise FloatingPointError('finite-difference perturbation lost its actual dtype direction')
                diagnostics.append(dict(role=role,sign=sign,epsilon=epsilon,
                    desired_parameter_norm=norm,achieved_parameter_norm=float(torch.linalg.vector_norm(observed)),
                    rounding_relative_error=error))
                parts.append(field)
            values.append(torch.cat((parts[1][:n_d],parts[0][n_d:])))
        image=(values[0]-values[1])/(2*epsilon)
        if not torch.isfinite(image).all():raise FloatingPointError('nonfinite cross-field finite difference')
        return image,diagnostics

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
        self.rng_before=self._rng(self.streams)
        unique_buffers={id(buffer):buffer for name in ('generator','critic','prior')
                        if isinstance((module:=local.get(name)),torch.nn.Module) for buffer in module.buffers()}
        self.buffers=[(buffer,buffer.detach().clone()) for buffer in unique_buffers.values()]
        self.base=self._copy_parameters();self.point=self._copy_parameters();self.base_flat=self._flat(self.base)
        self.phase=0;self.pending={}
        model=local.get('generator');clean=getattr(model,'model',model);prior=local.get('prior')
        measure='means' in local and prior is not None
        if measure:
            with torch.no_grad():
                original_z=prior.z.detach().clone();original_output=clean(original_z).double()
            if not all(torch.equal(a,b) for a,b in zip(self.rng_before,self._rng(self.streams))):
                raise RuntimeError('clean diagnostic consumed training RNG')
            if any(not torch.equal(buffer,saved) for buffer,saved in self.buffers):
                raise RuntimeError('clean diagnostic mutated module buffers')
        yield 0
        if self.pending:raise RuntimeError('incomplete first game field')
        self.rng_after=self._rng(self.streams)
        q0=self.root_metric*self._flat(self.field);qnorm=float(torch.linalg.vector_norm(q0))
        if not math.isfinite(qnorm):raise FloatingPointError('nonfinite metric field norm')
        record=dict(outer_step=self.outer_steps+1,zero_field=qnorm==0,field_metric_norm=qnorm,
                    correction_enabled=self.correction,finite_differences=[])
        solution=-q0
        if qnorm and self.correction:
            v=q0/qnorm
            jv,fd=yield from self._cross_product(v,'cross_v',q0)
            record['finite_differences'].extend(fd)
            orth=jv-torch.dot(v,jv)*v
            # Repeat the subtraction to keep the measured plane orthogonal.
            orth-=torch.dot(v,orth)*v
            orth_norm=float(torch.linalg.vector_norm(orth))
            record.update(cross_v_norm=float(torch.linalg.vector_norm(jv)),orthogonal_cross_v_norm=orth_norm)
            if orth_norm>1e-12*max(1.,float(torch.linalg.vector_norm(jv))):
                w=orth/orth_norm
                jw,fd=yield from self._cross_product(w,'cross_w',q0)
                record['finite_differences'].extend(fd)
                b21=float(torch.dot(w,jv));b12=float(torch.dot(v,jw));a=(b21-b12)/2
                if not math.isfinite(a):raise FloatingPointError('nonfinite projected skew coefficient')
                solution=(-qnorm/(1+a*a))*v+(a*qnorm/(1+a*a))*w
                skew_solution=a*(w*torch.dot(v,solution)-v*torch.dot(w,solution))
                record.update(b21=b21,b12=b12,skew_coefficient=a,
                    symmetric_offdiagonal=(b21+b12)/2,
                    projected_linear_relative_residual=float(torch.linalg.vector_norm(solution+skew_solution+q0))/qnorm,
                    basis_dot=float(torch.dot(v,w)))
            else:record.update(skew_coefficient=0.,degenerate_plane=True,projected_linear_relative_residual=0.)
        actual=self._set_point(solution)
        proposed_ratio=float(torch.linalg.vector_norm(solution))/qnorm if qnorm else 0.
        actual_ratio=float(torch.linalg.vector_norm(actual))/qnorm if qnorm else 0.
        if not math.isfinite(actual_ratio) or actual_ratio>1.0001:
            raise FloatingPointError('rounded skew proposal amplified the metric step')
        record.update(proposed_metric_norm_ratio=proposed_ratio,actual_metric_norm_ratio=actual_ratio,
            actual_metric_rounding_error=float(torch.linalg.vector_norm(actual-solution))/qnorm if qnorm else 0.)
        if qnorm and self.correction:
            final_q,actual=yield from super()._evaluate(solution,'joint_field_observation')
            record.update(final_metric_field_norm=float(torch.linalg.vector_norm(final_q)),
                full_joint_implicit_residual_observational=float(torch.linalg.vector_norm(actual+final_q))/qnorm)
        self.base_restores_verified+=1
        if measure:
            with torch.no_grad():
                network=clean(original_z).double();both=clean(prior.z).double()
                motion=dict(outer_step=self.outer_steps+1,
                    g_output_rms=float((network-original_output).square().sum(1).mean().sqrt()),
                    prior_output_rms=float((both-network).square().sum(1).mean().sqrt()),
                    total_output_rms=float((both-original_output).square().sum(1).mean().sqrt()))
                if not all(math.isfinite(value) for value in motion.values()):raise FloatingPointError('nonfinite output motion')
                self.output_motion.append(motion)
            if not all(torch.equal(a,b) for a,b in zip(self.rng_after,self._rng(self.streams))):
                raise RuntimeError('clean diagnostic consumed training RNG')
        self.skew_steps.append(record)
        self.phase=None;self.base,self.point={},{};self.outer_steps+=1
        if self.accounting is not None:self.accounting(self.rows[opt_d]['calls'],self.outer_steps)

    def receipt(self):
        value=FixedMetricExtraRecorder.receipt(self)
        value.update(method='rank_two_projected_implicit_skew_response',
            scratch_optimizer_policy='rank_two_projected_implicit_skew_response',shared_gate_eligible=False,
            fd_relative=self.fd_relative,correction_enabled=self.correction,skew_steps=self.skew_steps,queries=self.queries,
            gradient_evaluations_per_player_per_outer_step=None,extra_gradient_evaluations_per_player_per_outer_step=None,
            output_weights='rank-two inverse-skew response of the joint first-gradient Adam metric field',
            paper='https://arxiv.org/abs/2510.25716v2',
            reference_scope='SGA skew decomposition inspiration; this is not the published LRSGA recurrence',
            theoretical_scope='Exact projected 2x2 skew solve and metric norm bound; no nonlinear game convergence guarantee',
            full_joint_residual_scope='Observational only; does not control the step',
            nominal_rate_policy='Constant role rates; no training-age attenuation; one moment update per outer step')
        return value


@contextmanager
def projected_skew(task='mode_hold',**options):
    with patch.object(implicit_module,'ImplicitExtraRecorder',ProjectedSkewRecorder):
        with implicit_module.implicit_extra(task=task,**options) as value:
            yield value
