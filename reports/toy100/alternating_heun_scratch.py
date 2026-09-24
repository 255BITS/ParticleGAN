"""Alternating, same-sample trapezoid correction to the G/prior Adam proposal.

The D update retains PR #82's own-curvature bound. G first makes its ordinary
Adam proposal against that realized D. The same sampled host block is replayed
at the proposed G point, without advancing optimizer moments or RNG, and the
G proposal receives the vector correction ``-P (g_proposal - g_base) / 2``.
This is one explicit trapezoid step in the current Adam metric, not a scalar
curvature cap or an oracle quality gate. Both optimizers advance moments once.
"""

from contextlib import ExitStack, contextmanager
import ast
import math
from unittest.mock import patch

import torch

from benchmarks.locked_shared import mode_hold, trajectory
from particlegan.gan_loss import GANLoss
from reports.toy100.alternating_curvature_scratch import BothBoundRecorder, _rho
from reports.toy100.extra_adam_scratch import HOSTS, sha, transformed_function


METHOD = "alternating_same_sample_g_trapezoid_d_curvature_bound"


class AlternatingHeunRecorder(BothBoundRecorder):
    def __init__(self, start_step=0, d_curvature_bound=2.0, heun_weight=0.5,
                 scope="joint"):
        if not math.isfinite(heun_weight) or not 0 <= heun_weight <= 1:
            raise ValueError("invalid trapezoid weight")
        if scope not in {"network", "joint"}:
            raise ValueError("invalid trapezoid scope")
        super().__init__(start_step=start_step, curvature_bound=1e9,
                         d_curvature_bound=d_curvature_bound)
        self.heun_weight = heun_weight
        self.scope = scope

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self.phase != 2 or self.optimizers is None or optimizer is not self.optimizers[1]:
            return super().step(optimizer, ordinary_step, closure)
        if closure is not None:
            raise RuntimeError("closure outside declared game update")
        self.rows[optimizer]["calls"] += 1
        params = self._params(optimizer)
        g1 = [p.grad.detach().clone() for p in params]
        rho = _rho(self.g_base, self.g1, self.gg0, g1, self.metric_g)
        delta_sq = correction_sq = dot = 0.0
        selected = [self.scope == "joint" or not group.get("_comparison_prior", False)
                    for group in optimizer.param_groups for _ in group["params"]]
        roles = ["prior" if group.get("_comparison_prior", False) else "network"
                 for group in optimizer.param_groups for _ in group["params"]]
        if self.scope == "network" and not any(group.get("_comparison_prior", False)
                                                for group in optimizer.param_groups):
            raise RuntimeError("separate learned-prior optimizer group required")
        group_norms = {"network": [0.0, 0.0], "prior": [0.0, 0.0]}
        for p, base, proposal, g0, later, metric, chosen, role in zip(
                params, self.g_base, self.g1, self.gg0, g1, self.metric_g,
                selected, roles):
            delta = (proposal - base).double()
            correction = (-self.heun_weight * metric * (later - g0).double()
                          if chosen else torch.zeros_like(delta))
            delta_sq += float(delta.square().sum())
            correction_sq += float(correction.square().sum())
            dot += float((delta * correction).sum())
            group_norms[role][0] += float(delta.square().sum())
            group_norms[role][1] += float(correction.square().sum())
            p.copy_((proposal.double() + correction).to(p.dtype))
        self.row["g"] = dict(
            rho=rho, factor=1.0, vector_correction=True,
            correction_norm_ratio=math.sqrt(correction_sq / delta_sq) if delta_sq else 0.0,
            correction_cosine=dot / math.sqrt(delta_sq * correction_sq)
            if delta_sq and correction_sq else 0.0,
            group_norms={name: dict(proposal=math.sqrt(value[0]),
                                    correction=math.sqrt(value[1]))
                         for name, value in group_norms.items()})
        return None

    def receipt(self):
        receipt = super().receipt()
        receipt.update(method=METHOD, scratch_optimizer_policy=METHOD,
                       heun_weight=self.heun_weight,
                       scope=self.scope,
                       g_scalar_bound_disabled=True,
                       vector_corrected_updates=self.outer_steps)
        return receipt


@contextmanager
def alternating_heun(task="mode_hold", **options):
    module = {"mode_hold": mode_hold, "trajectory": trajectory}[task]
    tree, _, original_sha = transformed_function(module, task)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "phases"]
    if len(calls) != 1:
        raise RuntimeError("expected one phase iterator")
    calls[0].args.append(ast.Call(func=ast.Name(id="locals", ctx=ast.Load()),
                               args=[], keywords=[]))
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    recorder = AlternatingHeunRecorder(**options)
    recorder.host_source = dict(task=task, original_function_sha256=original_sha,
                                generated_function_sha256=sha(source.encode()))
    ordinary_step = torch.optim.Adam.step
    original_d_loss = GANLoss.d_loss

    def observed_d_loss(gan, real_logits, fake_logits):
        value = original_d_loss(gan, real_logits, fake_logits)
        if recorder.phase == 0 and recorder.advantage is None:
            recorder.advantage = math.log(2) - float(value.detach())
        return value

    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<alternating-heun-{task}>", "exec"),
             module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        yield recorder, source
