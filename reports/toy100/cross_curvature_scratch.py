"""Cross-player-only competitive response plus a per-player own-curvature bound.

Replay of the stock cross-only response (``cross_drift_replay``) showed its
live warm failure is own-player G curvature: G's own field change from its
own step was about twice the step, twenty times the cross response, while the
critic pointed particles toward their correct modes. Its cold failure instead
came from backtracking alpha to .008-.03. This variant therefore drops the
nonlinear-residual backtracking and bounds each player's accepted step by
its measured own-curvature ratio

    rho_p = ||alpha (F(u_p only) - F0)_p|| / ||u_p||,   u_p <- min(1, c / rho_p) u_p,

with two extra same-sample evaluations (only D moved, only G+prior moved).
rho_p is the effective preconditioned step times curvature along the step
(GD on a quadratic is monotone for rho<1 and unstable beyond 2). It depends
on the current state, not on training age.

With ``explicit=True`` there is no cross solve: the proposal is the ordinary
beta1=0 Adam step u=-sqrt(P) F0 (constant Adam, which acquires the cold
trajectory host unaided), and only the per-player own-curvature bound is
applied. That costs two extra same-sample evaluations per update.

The underlying solve:

P is the first-gradient Adam metric including each role's constant LR, as in
``implicit_extra_scratch``. With players D (x) and G+prior (y), the joint
field F=(F_x, F_y) and only the cross Jacobian blocks A=dF_x/dy, B=dF_y/dx,
the update delta=sqrt(P) u solves

    (I + alpha sqrt(P) K sqrt(P)) u = -alpha sqrt(P) F0,   K=[[0, A], [B, 0]].

Each player's own curvature is replaced by the Adam proximal term, as in
competitive gradient descent (Schaefer & Anandkumar 2019), whose linearized
local game is the Nash equilibrium of the bilinear approximation. K v is
measured by two same-sample finite-difference field evaluations, one moving
only D and one moving only G+prior; each keeps only the other player's
response. There is deliberately no nonlinear implicit residual: the CGD step
is not an implicit Euler solution, and that check drove alpha to .0077 in the
full-Jacobian cold failure. Retries halve alpha only when the linear residual
or proposed-correction bound fails. No convergence theorem is claimed for
this nonlinear, non-zero-sum, Adam-preconditioned host.
"""
from contextlib import ExitStack,contextmanager
import ast
import math
from unittest.mock import patch

import torch

from reports.toy100.implicit_extra_scratch import ImplicitExtraRecorder
from reports.toy100.extra_adam_scratch import HOSTS,sha,transformed_function

METHOD='same_sample_cross_only_response_with_own_curvature_step_bound'
EXPLICIT_METHOD='adam_with_same_sample_own_curvature_step_bound'


class CrossCurvatureRecorder(ImplicitExtraRecorder):
    def __init__(self,start_step=0,krylov_dim=8,linear_tolerance=.1,fd_relative=1e-4,
                 correction_limit=2.,max_backtracks=8,curvature_bound=1.,amplification_bound=None,explicit=False):
        super().__init__(start_step=start_step,krylov_dim=krylov_dim,linear_tolerance=linear_tolerance,
                         nonlinear_tolerance=.5,fd_relative=fd_relative,correction_limit=correction_limit,
                         max_backtracks=max_backtracks)
        if not math.isfinite(curvature_bound) or curvature_bound<=0:raise ValueError('invalid curvature bound')
        if amplification_bound is not None and (not math.isfinite(amplification_bound) or amplification_bound<=0):
            raise ValueError('invalid amplification bound')
        self.curvature_bound=curvature_bound;self.amplification_bound=amplification_bound
        self.explicit=explicit
        self.split=None;self.curvature=[]

    def _own_curvature_bound(self,solution,q0,scale):
        """Scale each player's step so its own-curvature ratio is at most the bound."""
        factors={};row=dict(outer_step=self.outer_steps+1)
        solution=solution.clone()
        for role,block in (('d',slice(0,self.split)),('g',slice(self.split,len(solution)))):
            if self.amplification_bound is not None:
                explicit=scale*float(torch.linalg.vector_norm(q0[block]))
                proposed=float(torch.linalg.vector_norm(solution[block]))
                if proposed>self.amplification_bound*explicit:
                    solution[block]*=self.amplification_bound*explicit/proposed
                row[role+'_amplification']=dict(proposed_to_explicit=proposed/explicit if explicit else None,
                    clamped=bool(proposed>self.amplification_bound*explicit))
            step=torch.zeros_like(solution);step[block]=solution[block]
            size=float(torch.linalg.vector_norm(step))
            if size==0:
                factors[role]=1.;row[role]=dict(rho=0.,factor=1.);continue
            q,_=yield from self._evaluate(step,'own_curvature_'+role)
            rho=scale*float(torch.linalg.vector_norm((q-q0)[block]))/size
            if not math.isfinite(rho):raise FloatingPointError('nonfinite own-curvature ratio')
            factors[role]=min(1.,self.curvature_bound/rho) if rho>0 else 1.
            row[role]=dict(rho=rho,factor=factors[role],step_norm=size)
        bounded=solution
        bounded[:self.split]*=factors['d'];bounded[self.split:]*=factors['g']
        self.curvature.append(row)
        return bounded

    def _cross_image(self,vector,q0):
        """Return sqrt(P) K sqrt(P) vector using one evaluation per moving player."""
        image=torch.zeros_like(vector);records=[]
        for moving,(lo,hi),(rlo,rhi) in (('d',(0,self.split),(self.split,len(vector))),
                                          ('g',(self.split,len(vector)),(0,self.split))):
            block=torch.zeros_like(vector);block[lo:hi]=vector[lo:hi]
            direction=self.root_metric*block;norm=float(torch.linalg.vector_norm(direction))
            if norm==0:continue
            desired=self.fd_relative*(1+float(torch.linalg.vector_norm(self.base_flat)))
            epsilon=desired/norm
            q,actual_u=yield from self._evaluate(epsilon*block,'cross_jvp_'+moving)
            actual_delta=self.root_metric*actual_u
            error=float(torch.linalg.vector_norm(actual_delta-epsilon*direction))/desired
            if not math.isfinite(error) or error>.05:
                raise FloatingPointError('float32 finite-difference perturbation lost direction/scale')
            image[rlo:rhi]=(q-q0)[rlo:rhi]/epsilon
            records.append(dict(moving=moving,epsilon=epsilon,rounding_relative_error=error,
                own_block_response_norm=float(torch.linalg.vector_norm((q-q0)[lo:hi]/epsilon)),
                cross_block_response_norm=float(torch.linalg.vector_norm(image[rlo:rhi]))))
        return image,records

    def _solve(self,scale,q0):
        rhs=-scale*q0;beta=float(torch.linalg.vector_norm(rhs))
        vectors=[rhs/beta]
        h=torch.zeros((self.krylov_dim+1,self.krylov_dim),dtype=torch.float64)
        target=torch.zeros(self.krylov_dim+1,dtype=torch.float64);target[0]=beta
        fd=[]
        for column in range(self.krylov_dim):
            cross,records=yield from self._cross_image(vectors[column],q0)
            fd.extend(records)
            image=vectors[column]+scale*cross
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
        return solution,dict(krylov_iterations=column+1,linear_relative_residual=residual,
                             finite_differences=fd[-2:],finite_difference_evaluations=len(fd))

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
            self.split=sum(p.numel() for group in opt_d.param_groups for p in group['params'])
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
                krylov_iterations=0,linear_relative_residual=0.,finite_difference_evaluations=0))
            scale=1.
        else:
            for retry in range(self.max_backtracks+1):
                if self.explicit:
                    solution=-scale*q0
                    diagnostic=dict(krylov_iterations=0,linear_relative_residual=0.,finite_difference_evaluations=0)
                else:solution,diagnostic=yield from self._solve(scale,q0)
                diagnostic.update(outer_step=self.outer_steps+1,retry=retry,scale=scale,accepted=False)
                correction_ratio=float(torch.linalg.vector_norm(solution))/(scale*qnorm)
                diagnostic['correction_to_explicit_norm_ratio']=correction_ratio
                d_part=float(torch.linalg.vector_norm(solution[:self.split]))
                g_part=float(torch.linalg.vector_norm(solution[self.split:]))
                d_explicit=scale*float(torch.linalg.vector_norm(q0[:self.split]))
                g_explicit=scale*float(torch.linalg.vector_norm(q0[self.split:]))
                diagnostic['d_step_to_explicit']=d_part/d_explicit if d_explicit else None
                diagnostic['g_step_to_explicit']=g_part/g_explicit if g_explicit else None
                if diagnostic['linear_relative_residual']>self.linear_tolerance:
                    diagnostic['rejection']='linear_residual'
                elif not math.isfinite(correction_ratio) or correction_ratio>self.correction_limit:
                    diagnostic['rejection']='correction_bound'
                else:diagnostic['accepted']=True
                self.solves.append(diagnostic)
                if diagnostic['accepted']:break
                if retry==self.max_backtracks:
                    raise RuntimeError('cross response exhausted declared residual/backtracking budget')
                scale*=.5
            solution=yield from self._own_curvature_bound(solution,q0,scale)
            self._set_point(solution)
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

    def summary(self):
        accepted=[row for row in self.solves if row['accepted']]
        scales=[row['scale'] for row in accepted]
        evaluations=self.rows[self.optimizers[0]]['calls'] if self.rows else 0
        def stats(values):
            values=[v for v in values if v is not None]
            return dict(min=min(values),mean=sum(values)/len(values),max=max(values)) if values else None
        motion=[row['total_output_rms'] for row in self.output_motion]
        return dict(outer_steps=self.outer_steps,gradient_evaluations_per_player=evaluations,
            gradient_evaluations_per_player_per_outer_step=evaluations/self.outer_steps if self.outer_steps else None,
            moment_updates_per_outer_step=1,queries=len(self.queries),rng_replay_verified=self.rng_replay_verified,
            solves=len(self.solves),rejections={kind:sum(row.get('rejection')==kind for row in self.solves)
                for kind in ('linear_residual','correction_bound')},
            accepted_scale=stats(scales),last50_scale=stats(scales[-50:]),
            g_step_to_explicit=stats([row['g_step_to_explicit'] for row in accepted]),
            d_step_to_explicit=stats([row['d_step_to_explicit'] for row in accepted]),
            krylov_iterations=stats([row['krylov_iterations'] for row in accepted]),
            clean_output_motion=stats(motion),late_clean_output_motion=stats(motion[-200:]),
            curvature_bound=self.curvature_bound,amplification_bound=self.amplification_bound,
            **{f'{role}_amplification_clamped':sum(bool(row.get(role+'_amplification',{}).get('clamped')) for row in self.curvature) for role in ('d','g')},
            **{f'{role}_rho':stats([row[role]['rho'] for row in self.curvature]) for role in ('d','g')},
            **{f'{role}_curvature_factor':stats([row[role]['factor'] for row in self.curvature]) for role in ('d','g')},
            **{f'{role}_curvature_bound_active':sum(row[role]['factor']<1 for row in self.curvature) for role in ('d','g')})

    def receipt(self):
        value=super().receipt()
        value.update(method=EXPLICIT_METHOD if self.explicit else METHOD,nonlinear_tolerance=None,
            output_weights='accepted linear-residual-checked cross-only competitive proposal',
            scratch_optimizer_policy=METHOD,summary=self.summary(),
            jacobian_blocks='cross-player only in the solve; own-player blocks measured along the accepted step and used only for the per-player curvature bound',
            curvature=self.curvature,own_curvature_evaluations_per_outer_step=2,
            finite_difference_evaluations_per_krylov_iteration=2,
            reference_scope='Cross-only CGD linearized local game with Adam-metric proximal terms; not the full-Jacobian comparison',
            theoretical_scope='Matrix-free finite-difference cross blocks with measured linear residuals; no convergence theorem claimed')
        return value


@contextmanager
def cross_curvature(task='mode_hold',**options):
    from benchmarks.locked_shared import mode_hold,trajectory
    module={'mode_hold':mode_hold,'trajectory':trajectory}[task]
    tree,_,original_sha=transformed_function(module,task)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='phases']
    if len(calls)!=1:raise RuntimeError('expected one phase iterator')
    calls[0].args.append(ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]));ast.fix_missing_locations(tree)
    source=ast.unparse(tree)+'\n';recorder=CrossCurvatureRecorder(**options)
    recorder.host_source=dict(task=task,original_function_sha256=original_sha,generated_function_sha256=sha(source.encode()),source_transform='verified gradient-loop transform plus locals for RNG binding')
    ordinary_step=torch.optim.Adam.step
    def patched_step(optimizer,closure=None):return recorder.step(optimizer,ordinary_step,closure)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__,{'_extra_state':recorder}));namespace={}
        exec(compile(tree,f'<cross-curvature-{task}>','exec'),module.__dict__,namespace)
        stack.enter_context(patch.object(module,HOSTS[task],namespace[HOSTS[task]]))
        stack.enter_context(patch.object(torch.optim.Adam,'step',patched_step))
        yield recorder,source
