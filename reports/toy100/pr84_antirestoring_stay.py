"""PR84 plus a mild G shrink when own-curvature is anti-restoring.

The magnitude bound min(1, 0.25/rho) accepts a full generator step whenever
rho is below 0.25. Negative directional curvature along that step means the
move is concave in the Adam metric. That sign alone also trips on a hold the
pin already keeps, so the shrink stays idle unless the base-point critic
advantage is negative. Restoring steps, already-bounded steps, and
anti-restoring steps with a non-negative critic advantage stay exactly PR84.
When the shrink does fire it is still max(0.5, 1+alignment).

No coverage term, likelihood, target center, HQ clip, or rest gate.
"""

from contextlib import ExitStack, contextmanager
import ast
import math

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from particlegan.gan_loss import GANLoss
from reports.toy100.alternating_curvature_scratch import _metric, _rho
from reports.toy100.extra_adam_scratch import HOSTS, sha, transformed_function
from reports.toy100.pr84_smoothed_candidate import (
    METHOD as PR84_METHOD, SmoothedBothBoundRecorder,
)


METHOD = "pr84_mild_antirestoring_g_step"
MILD_FLOOR = 0.5


def signed_alignment(base, new, g0, g1, metric):
    """Return (kappa, rho, alignment) in the PR84 preconditioned metric."""
    num = den = mag = 0.
    for b, n, a, c, m in zip(base, new, g0, g1, metric):
        delta = (n - b).double()
        change = (c - a).double()
        num += float((change * delta).sum())
        den += float((delta.square() / m).sum())
        mag += float((m * change.square()).sum())
    if den <= 0.:
        return 0., 0., 0.
    kappa = num / den
    rho = math.sqrt(mag / den) if mag > 0. else 0.
    if not math.isfinite(kappa) or not math.isfinite(rho):
        raise FloatingPointError("nonfinite signed own-curvature")
    alignment = 0. if rho <= 0. else max(-1., min(1., kappa / rho))
    return kappa, rho, alignment


def mild_factor(rho, bound, alignment, *, mild=True, floor=MILD_FLOOR, advantage=None):
    """PR84 magnitude factor, then a mild shrink on an accepted anti-restoring step.

    Negative alignment alone is common on a hold the pin already keeps. The
    shrink stays idle unless the base-point critic advantage is also negative
    (discriminator loss above log 2: the critic is not confirming the step).
    The floor and ``max(0.5, 1+alignment)`` shape are unchanged.
    """
    factor = min(1., bound / rho) if rho > 0. else 1.
    applied = False
    critic_losing = advantage is not None and advantage < 0.
    if mild and factor >= 1. and alignment < 0. and critic_losing:
        shrunk = max(floor, 1. + alignment)
        if shrunk < factor:
            factor = shrunk
            applied = True
    return factor, applied


class AntiRestoringStayRecorder(SmoothedBothBoundRecorder):
    def __init__(self, *, mild=True, mild_floor=MILD_FLOOR, **kwargs):
        super().__init__(**kwargs)
        if not 0. < mild_floor <= 1.:
            raise ValueError("mild floor must lie in (0, 1]")
        self.mild = mild
        self.mild_floor = float(mild_floor)

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self.passthrough:
            return ordinary_step(optimizer, closure=closure)
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError("optimizer call outside the declared game update")
        self.rows[optimizer]["calls"] += 1
        opt_d, opt_g = self.optimizers
        grads = lambda opt: [p.grad.detach().clone() for p in self._params(opt)]
        if optimizer is opt_d:
            if self.phase == 0:
                self.gd0 = grads(opt_d)
                ordinary_step(opt_d)
                self.metric_d = _metric(opt_d)
                self.d1 = [p.detach().clone() for p in self._params(opt_d)]
                for p, v in zip(self._params(opt_g), self.g_base):
                    p.copy_(v)
            elif self.phase == 1:
                rho = _rho(self.d0, self.d1, self.gd0, grads(opt_d), self.metric_d)
                factor = min(1., self.d_curvature_bound / rho) if rho > 0 else 1.
                for p, b, n in zip(self._params(opt_d), self.d0, self.d1):
                    p.copy_(torch.lerp(b, n, factor) if factor < 1 else n)
                self.d_star = [p.detach().clone() for p in self._params(opt_d)]
                self.row["d"] = dict(rho=rho, factor=factor)
            else:
                for p, v in zip(self._params(opt_d), self.d_star):
                    p.copy_(v)
            self._arm_smoothed_critic()
            return None
        if self.phase == 0:
            for p, v in zip(self._params(opt_g), self.g_base):
                p.copy_(v)
            for p, v in zip(self._params(opt_d), self.d1):
                p.copy_(v)
        elif self.phase == 1:
            self.gg0 = grads(opt_g)
            ordinary_step(opt_g)
            self.metric_g = _metric(opt_g)
            self.g1 = [p.detach().clone() for p in self._params(opt_g)]
        else:
            g1 = grads(opt_g)
            rho = _rho(self.g_base, self.g1, self.gg0, g1, self.metric_g)
            kappa, rho_signed, alignment = signed_alignment(
                self.g_base, self.g1, self.gg0, g1, self.metric_g)
            if abs(rho - rho_signed) > 1e-6:
                raise RuntimeError("signed curvature magnitude disagrees with rho")
            factor, applied = mild_factor(
                rho, self.curvature_bound, alignment,
                mild=self.mild, floor=self.mild_floor, advantage=self.advantage)
            for p, b, n in zip(self._params(opt_g), self.g_base, self.g1):
                p.copy_(torch.lerp(b, n, factor) if factor < 1 else n)
            self.row["g"] = dict(rho=rho, factor=factor, kappa=kappa,
                                 alignment=alignment, mild=applied,
                                 critic_advantage=self.advantage)
        return None

    def receipt(self):
        value = super().receipt()
        closed = [r for r in self.records if not r["gate_open"]]
        value.update(
            method=METHOD, scratch_optimizer_policy=METHOD, smooth_critic=True,
            base_method=PR84_METHOD, mild=self.mild, mild_floor=self.mild_floor,
            mild_updates=sum(bool(r.get("g", {}).get("mild")) for r in closed),
            purity="GAN dynamics only — no coverage/likelihood term",
        )
        return value


@contextmanager
def pr84_antirestoring_stay(*, task="mode_hold", start_step=0, mild=True):
    """Yield ``(recorder, generated_host_source)``. ``mild=False`` matches PR84 steps."""
    from benchmarks.locked_shared import mode_hold, trajectory

    module = {"mode_hold": mode_hold, "trajectory": trajectory}[task]
    tree, _, original_sha = transformed_function(module, task)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "phases"]
    if len(calls) != 1:
        raise RuntimeError("expected one phase iterator")
    calls[0].args.append(ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    recorder = AntiRestoringStayRecorder(start_step=start_step, mild=mild)
    recorder.host_source = dict(task=task, original_function_sha256=original_sha,
                                generated_function_sha256=sha(source.encode()))
    ordinary_step = torch.optim.Adam.step
    original_d_loss = GANLoss.d_loss
    original_forward = SimpleMLPDiscriminator.forward

    def observed_d_loss(gan, real_logits, fake_logits):
        value = original_d_loss(gan, real_logits, fake_logits)
        if recorder.phase == 0 and recorder.advantage is None:
            recorder.advantage = math.log(2) - float(value.detach())
        return value

    def smoothed_forward(self, x):
        if (recorder._smooth_on and recorder.enabled and not recorder.passthrough
                and x.ndim >= 2 and x.shape[-1] == 2 and recorder._smooth_width > 0):
            width = recorder._smooth_width
            vals = [original_forward(self, x)]
            for dim in range(x.shape[-1]):
                shift = torch.zeros_like(x)
                shift[..., dim] = width
                vals.append(original_forward(self, x + shift))
                vals.append(original_forward(self, x - shift))
            return torch.stack(vals, 0).mean(0)
        return original_forward(self, x)

    with ExitStack() as stack:
        stack.enter_context(patch_dict(module, recorder))
        namespace = {}
        exec(compile(tree, f"<pr84-antirestoring-{task}>", "exec"), module.__dict__, namespace)
        from unittest.mock import patch
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(
            torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(SimpleMLPDiscriminator, "forward", smoothed_forward))
        yield recorder, source


def patch_dict(module, recorder):
    from unittest.mock import patch
    return patch.dict(module.__dict__, {"_extra_state": recorder})
