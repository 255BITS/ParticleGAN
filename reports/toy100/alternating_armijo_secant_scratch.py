"""Frozen D Armijo update combined with positive own-secant G response.

The D side is exactly the archived same-sample total-loss Armijo policy,
including its verified zero-step rejection after exhausted trials. Only the
G side changes from PR #82's scalar own-curvature bound to the positive
rank-one secant resolvent, with that bound as nonpositive-curvature fallback.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from reports.toy100 import alternating_linesearch_scratch as linesearch
from reports.toy100.alternating_positive_secant_scratch import positive_secant_direction


METHOD = "alternating_D_Armijo_G_positive_own_secant"


class ArmijoSecantRecorder(linesearch.DLineSearchRecorder):
    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if (self.passthrough or self.optimizers is None or optimizer is not self.optimizers[1]
                or self.stage != "g_trial"):
            return super().step(optimizer, ordinary_step, closure)
        params = self._params(optimizer)
        g1 = [p.grad.detach().clone() for p in params]
        result = super().step(optimizer, ordinary_step, closure)
        s = torch.cat([(new - base).flatten().double()
                       for base, new in zip(self.g_base, self.g1)])
        y = torch.cat([(later - g0).flatten().double()
                       for g0, later in zip(self.gg0, g1)])
        metric = torch.cat([value.flatten().double() for value in self.metric_g])
        rho = self.row["g"]["rho"]
        direction, diagnostic = positive_secant_direction(
            s, y, metric, rho, self.curvature_bound)
        self._place(optimizer, [(base.double() + block.reshape_as(parameter)).to(parameter.dtype)
                                for parameter, base, block in zip(
                                    params, self.g_base, direction.split([p.numel() for p in params]))])
        self.row["g"] = dict(rho=rho, **diagnostic)
        return result

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     positive_secant_updates=sum(row["g"]["rule"] == "positive_rank_one"
                                                 for row in self.records),
                     g_fallback_updates=sum(row["g"]["rule"] == "scalar_curvature_fallback"
                                            for row in self.records),
                     d_policy="archived same-sample total-D-loss Armijo, reject exhausted at verified alpha0")
        return value


@contextmanager
def alternating_armijo_secant(task="mode_hold", **options):
    with patch.object(linesearch, "DLineSearchRecorder", ArmijoSecantRecorder):
        with linesearch.alternating_linesearch(task=task, **options) as value:
            yield value
