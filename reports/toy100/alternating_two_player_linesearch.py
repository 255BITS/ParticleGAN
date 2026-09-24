"""Second bounded arm: verify G's actual sufficient decrease as well as D's.

The G Armijo fraction7/8 matches the old .25 positive-curvature margin on a
scalar quadratic: accepted curvature <=2*(1-c)=.25. Unlike a norm bound,
negative directional curvature is allowed to improve acquisition. This is an
own-loss check, not a game convergence theorem or a quality-metric oracle.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from reports.toy100 import alternating_linesearch_scratch as d_adapter
from reports.toy100.alternating_linesearch_scratch import DLineSearchRecorder
from reports.toy100.alternating_curvature_scratch import _rho

METHOD='alternating_verified_two_player_armijo'


class TwoPlayerLineSearchRecorder(DLineSearchRecorder):
    def __init__(self,g_armijo=.875,**options):
        super().__init__(**options)
        if not 0<g_armijo<1:raise ValueError('invalid G Armijo fraction')
        self.g_armijo=g_armijo;self.g_retry_needed=False

    def phases(self,step,opt_d,opt_g,local):
        for phase in super().phases(step,opt_d,opt_g,local):
            yield phase
            while self.g_retry_needed:
                # Parent verifies the final replay; verify each rejected one
                # here before adding a further same-RNG pass.
                self._check_rng()
                self.g_retry+=1
                self.g_alpha=0. if self.g_retry>self.max_retries else self.g_alpha*.5
                self._place(opt_g,[b if self.g_alpha==0 else torch.lerp(b,n,self.g_alpha)
                    for b,n in zip(self.g_base,self.g1)])
                self._prepare_replay()
                yield 1

    @torch.no_grad()
    def step(self,opt,ordinary_step,closure=None):
        if self.passthrough or self.optimizers is None or opt is self.optimizers[0]:
            return super().step(opt,ordinary_step,closure)
        if self.stage=='g_base':
            self.lg0=self.loss_reader('g')
            result=super().step(opt,ordinary_step,closure)
            self.predicted_g=-sum(float((g.double()*(n-b).double()).sum()) for g,b,n in zip(self.gg0,self.g_base,self.g1))
            if not math.isfinite(self.lg0) or not math.isfinite(self.predicted_g) or self.predicted_g<0:
                raise FloatingPointError('invalid G base loss or descent direction')
            self.row['g_trials']=[];self.g_alpha=1.;self.g_retry=0;self.g_retry_needed=False
            return result
        if self.stage!='g_trial':return super().step(opt,ordinary_step,closure)
        # Reuse the parent's optimizer/rate audit, without invoking its G
        # norm-curvature update. The only additional state is this line search.
        self.stage='g_probe'
        try:super().step(opt,ordinary_step,closure)
        finally:self.stage='g_trial'
        loss=self.loss_reader('g');grads=[p.grad.detach().clone() for p in self._params(opt)]
        if self.g_alpha==0 and (loss!=self.lg0 or any(not torch.equal(a,b) for a,b in zip(grads,self.gg0))):
            raise RuntimeError('zero-step G replay changed base loss or gradient')
        tolerance=4*torch.finfo(self._params(opt)[0].dtype).eps*max(1.,abs(self.lg0),abs(loss))
        threshold=self.lg0-self.g_armijo*self.g_alpha*self.predicted_g
        accepted=math.isfinite(loss) and all(torch.isfinite(g).all() for g in grads) and loss<=threshold+tolerance
        self.row['g_trials'].append(dict(retry=self.g_retry,alpha=self.g_alpha,loss=loss,loss_base=self.lg0,
            predicted_improvement=self.g_alpha*self.predicted_g,armijo_threshold=threshold,
            rounding_tolerance=tolerance,accepted=accepted))
        self.g_retry_needed=not accepted
        if accepted:
            current=self._copy(opt)
            rho=_rho(self.g_base,current,self.gg0,grads,self.metric_g)
            self.row['g']=dict(rho=rho,factor=self.g_alpha,accepted_effective_norm_curvature=self.g_alpha*rho,
                loss_before=self.lg0,loss_after=loss,actual_improvement=self.lg0-loss,
                predicted_improvement=self.g_alpha*self.predicted_g,backtracks=self.g_retry,
                rejected_proposal=self.g_alpha==0,zero_step_identity_verified=self.g_alpha==0)
            self.stage='done'
        elif self.g_alpha==0:raise RuntimeError('zero-step G replay rejected')
        return None

    def receipt(self):
        value=super().receipt()
        value.update(method=METHOD,scratch_optimizer_policy=METHOD,g_armijo=self.g_armijo,
            update_order='verified D point first, then G first gradient/moment update and verified G point',
            g_curvature_scope='old norm rule replaced by actual own-loss Armijo; recorded rho is observational',
            scalar_quadratic_positive_curvature_margin=2*(1-self.g_armijo))
        return value


@contextmanager
def alternating_two_player_linesearch(task='mode_hold',**options):
    with patch.object(d_adapter,'DLineSearchRecorder',TwoPlayerLineSearchRecorder):
        with d_adapter.alternating_linesearch(task=task,**options) as value:yield value
