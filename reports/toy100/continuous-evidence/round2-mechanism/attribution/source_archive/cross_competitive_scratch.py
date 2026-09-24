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
    def __init__(self,**options):
        super().__init__(**options)
        self.cross_products=[]
        self.joint_residuals=[]

    def _solve(self,scale,q0):
        self.solve_scale=scale;self.solve_q0=q0
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
            self.joint_residuals.append(dict(outer_step=self.outer_steps+1,scale=self.solve_scale,
                full_joint_relative_residual=float(torch.linalg.vector_norm(actual_u+self.solve_scale*joint_q))/denominator,
                cross_relative_residual=float(torch.linalg.vector_norm(actual_u+self.solve_scale*cross_q))/denominator))
        return cross_q,actual_u

    def receipt(self):
        value=super().receipt()
        value.update(method='same_sample_cross_only_competitive_response',
            scratch_optimizer_policy='same_sample_cross_only_competitive_response',
            reference_scope='CGD cross-player block system with current Adam metric, finite-difference products and nonlinear safeguards; not original published optimizer code',
            nonlinear_residual_scope='D field at (Dbase,Gnew), G+prior field at (Dnew,Gbase)',
            cross_products=self.cross_products,joint_residual_diagnostics=self.joint_residuals)
        return value


@contextmanager
def cross_competitive(task='mode_hold',**options):
    with patch.object(implicit_module,'ImplicitExtraRecorder',CrossCompetitiveRecorder):
        with implicit_module.implicit_extra(task=task,**options) as value:
            yield value
