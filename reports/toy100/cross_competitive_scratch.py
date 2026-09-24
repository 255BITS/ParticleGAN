"""CGD-style cross-player response using the bounded implicit solver.

Only the off-diagonal Jacobian blocks enter the local solve. G network and
prior form one player; D is the other. Each product uses two separate actual
host reevaluations so detach/temporary requires_grad flags do not erase
strategic cross-dependence. The nonlinear acceptance condition is also
cross-only, and the full joint residual is retained as an observation.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch

from reports.toy100 import implicit_extra_scratch as implicit_module
from reports.toy100.implicit_extra_scratch import ImplicitExtraRecorder


class CrossCompetitiveRecorder(ImplicitExtraRecorder):
    def __init__(self,matched_output_rms_limit=None,matched_nearest_limit=.12,
                 error_relative_gain=None,**options):
        super().__init__(**options)
        self.cross_products=[]
        self.joint_residuals=[]
        self.matched_output_rms_limit=matched_output_rms_limit
        self.matched_nearest_limit=matched_nearest_limit
        self.error_relative_gain=error_relative_gain
        self.guard_rejections=[]
        self._geometry_local=None
        self._base_clean=None
        self._matched=False
        self._base_error=None
        self._error_kind=None

    def phases(self,step,opt_d,opt_g,local):
        self._geometry_local=local
        self._base_clean=None
        self._matched=False
        self._base_error=None
        self._error_kind=None
        yield from super().phases(step,opt_d,opt_g,local)

    def _capture_match(self):
        local=self._geometry_local or {}
        means,prior,generator=local.get('means'),local.get('prior'),local.get('generator')
        if means is None or prior is None or generator is None or self.matched_output_rms_limit is None:
            self._base_clean=None
            self._matched=False
            return
        clean=getattr(generator,'model',generator)
        with torch.no_grad():
            points=clean(prior.z).detach()
        nearest=torch.cdist(points,means).min(1).values
        self._base_clean=points
        self._matched=bool(float(nearest.max())<=3*.07 and float(nearest.mean())<=self.matched_nearest_limit)

    def _clean_points(self):
        local=self._geometry_local or {}
        generator,prior=local.get('generator'),local.get('prior')
        if generator is None or prior is None:return None
        clean=getattr(generator,'model',generator)
        means,fast,slow=local.get('means'),local.get('fast'),local.get('slow')
        with torch.no_grad():
            if means is not None:
                points=clean(prior.z).detach()
                error=float(torch.cdist(points,means).min(1).values.mean())
                return points,error,'modes'
            if fast is not None and slow is not None:
                points=clean(slow,prior.z).detach()
                error=float((points-fast).pow(2).mean().sqrt())
                return points,error,'identity'
        return None

    def _capture_error(self):
        if self.error_relative_gain is None:return
        measured=self._clean_points()
        if measured is None:
            self._base_clean=None
            self._base_error=None
            return
        points,error,kind=measured
        self._base_clean=points
        self._base_error=error
        self._error_kind=kind

    def _solve(self,scale,q0):
        self.solve_scale=scale;self.solve_q0=q0
        if self._base_clean is None and self.matched_output_rms_limit is not None:
            self._capture_match()
        if self._base_error is None:
            self._capture_error()
        return (yield from super()._solve(scale,q0))

    def _evaluate(self,u,kind):
        n_d=sum(p.numel() for group in self.optimizers[0].param_groups for p in group['params'])
        d_only=torch.zeros_like(u);d_only[:n_d]=u[:n_d]
        g_only=torch.zeros_like(u);g_only[n_d:]=u[n_d:]
        q_d,_=yield from super()._evaluate(d_only,kind+'_d_only')
        q_g,_=yield from super()._evaluate(g_only,kind+'_g_only')
        cross_q=torch.cat((q_g[:n_d],q_d[n_d:]))
        actual_u=self._set_point(u)
        if kind=='jvp':
            own_q=torch.cat((q_d[:n_d],q_g[n_d:]))
            self.cross_products.append(dict(outer_step=self.outer_steps+1,
                cross_field_difference_norm=float(torch.linalg.vector_norm(cross_q-self.solve_q0)),
                own_field_difference_norm=float(torch.linalg.vector_norm(own_q-self.solve_q0))))
        elif kind=='nonlinear_residual':
            joint_q,actual_u=yield from super()._evaluate(u,'joint_residual_diagnostic')
            denominator=self.solve_scale*float(torch.linalg.vector_norm(self.solve_q0))
            cross_residual=float(torch.linalg.vector_norm(actual_u+self.solve_scale*cross_q))/denominator
            self.joint_residuals.append(dict(outer_step=self.outer_steps+1,scale=self.solve_scale,
                full_joint_relative_residual=float(torch.linalg.vector_norm(actual_u+self.solve_scale*joint_q))/denominator,
                cross_relative_residual=cross_residual))
            if self._matched and self._base_clean is not None:
                current=getattr(generator,'model',generator) if (generator:=(self._geometry_local or {}).get('generator')) is not None else None
                if current is not None:
                    with torch.no_grad():
                        moved=current((self._geometry_local['prior']).z).detach()
                    rms=float((moved-self._base_clean).square().sum(1).mean().sqrt())
                    self.joint_residuals[-1]['clean_output_rms']=rms
                    if rms>self.matched_output_rms_limit:
                        self.joint_residuals[-1]['guard']='matched_output_motion'
                        self.guard_rejections.append(dict(outer_step=self.outer_steps+1,scale=self.solve_scale,
                            clean_output_rms=rms,cross_relative_residual=cross_residual))
                        cross_q=torch.zeros_like(cross_q)
            if self.error_relative_gain is not None and self._base_clean is not None and self._base_error is not None:
                moved=self._clean_points()
                if moved is not None:
                    points,new_error,_=moved
                    delta=points-self._base_clean
                    rms=float(delta.square().sum(-1).mean().sqrt() if self._error_kind=='modes' else delta.pow(2).mean().sqrt())
                    allow=self.error_relative_gain*max(self._base_error,1e-6)
                    self.joint_residuals[-1].update(target_error=self._base_error,clean_output_rms=rms,
                        error_after=new_error,error_kind=self._error_kind,allowed_output_rms=allow)
                    if rms>allow:
                        reason='error_relative_motion'
                        self.joint_residuals[-1]['guard']=reason
                        self.guard_rejections.append(dict(outer_step=self.outer_steps+1,scale=self.solve_scale,
                            clean_output_rms=rms,target_error=self._base_error,error_after=new_error,
                            cross_relative_residual=cross_residual,guard=reason))
                        cross_q=torch.zeros_like(cross_q)
        return cross_q,actual_u

    def receipt(self):
        value=super().receipt()
        value.update(method='same_sample_cross_only_competitive_response',
            scratch_optimizer_policy='same_sample_cross_only_competitive_response',
            reference_scope='CGD cross-player block system with current Adam metric, finite-difference products and nonlinear safeguards; not original published optimizer code',
            nonlinear_residual_scope='D field at (Dbase,Gnew), G+prior field at (Dnew,Gbase)',
            cross_products=self.cross_products,joint_residual_diagnostics=self.joint_residuals,
            matched_output_rms_limit=self.matched_output_rms_limit,
            matched_nearest_limit=self.matched_nearest_limit,
            error_relative_gain=self.error_relative_gain,
            guard_rejections=self.guard_rejections)
        return value


@contextmanager
def cross_competitive(task='mode_hold',**options):
    with patch.object(implicit_module,'ImplicitExtraRecorder',CrossCompetitiveRecorder):
        with implicit_module.implicit_extra(task=task,**options) as value:
            yield value
