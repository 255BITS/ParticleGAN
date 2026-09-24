"""PR84 plus mode-exit projection on the generator step.

For fakes that sit in a high local D region (acquired mode), the Adam/G
update direction may push them out.  The directional derivative of D
along the proposed G step tells us:

    dD_step = sum_i (dD/dx_i) . (dx_i / dtheta) . delta_theta

If dD_step >= 0 the step is not exiting; keep the full PR84 update.
If dD_step < 0 the step moves particles toward lower D: shrink the G
parameter step by projecting out the mode-exit component.

Implementation: after the PR84 curvature-bounded G step (g_base -> g_new),
evaluate fakes at both g_base and g_new.  For each fake, compute D and its
input gradient.  The per-sample directional derivative is
    dot(grad_D(x_new), x_new - x_base).
If the mean directional derivative is negative (net mode exit), scale the
step by max(projection_floor, 1 + mean_dd / ||dx||_mean / ||grad_D||_mean)
to remove the exiting component.

Purity: GAN dynamics only.  Uses D and its input gradient.  No target
centers, coverage loss, HQ ball clip, or bank screens.
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


METHOD = "pr84_mode_exit_projection"
PROJECTION_FLOOR = 0.3


def directional_derivative_stats(critic, generator, prior, g_base_all,
                                 g_new_all, *, original_forward):
    """Directional derivative of D along the G step for HIGH-D particles only.

    g_base_all / g_new_all are the full opt_g param list (generator + prior).
    Only the generator-parameter prefix is used to produce fakes; the prior
    latents (z) stay fixed.

    Particles in a "high local D region" are those whose D value at
    x_base is above the per-batch median.  The directional derivative
    is computed only for those particles.

    Returns (mean_dd, dx_norm, fired, n_high).
    """
    clean_gen = getattr(generator, "model", generator)
    z = prior.z.detach()

    gen_params = list(clean_gen.parameters())
    n_gen = len(gen_params)
    saved = [p.detach().clone() for p in gen_params]

    with torch.no_grad():
        for p, v in zip(gen_params, g_base_all[:n_gen]):
            p.copy_(v)
        x_base = clean_gen(z).detach()

        for p, v in zip(gen_params, g_new_all[:n_gen]):
            p.copy_(v)
        x_new = clean_gen(z).detach()

        for p, v in zip(gen_params, saved):
            p.copy_(v)

    dx = x_new - x_base
    dx_norm = dx.norm(dim=-1).mean()
    if float(dx_norm) < 1e-10:
        return 0.0, 0.0, False, 0

    critic_module = critic
    while not isinstance(critic_module, SimpleMLPDiscriminator) and hasattr(critic_module, "model"):
        critic_module = critic_module.model
    if not isinstance(critic_module, SimpleMLPDiscriminator):
        return 0.0, 0.0, False, 0

    x_base_grad = x_base.detach().requires_grad_(True)
    d_base = original_forward(critic_module, x_base_grad)
    grad_d_base = torch.autograd.grad(
        d_base.sum(), x_base_grad, create_graph=False)[0].detach()

    d_vals = d_base.detach().squeeze()
    median_d = float(d_vals.median())
    high_mask = d_vals >= median_d
    n_high = int(high_mask.sum())
    if n_high == 0:
        return 0.0, float(dx_norm), False, 0

    dd_per_sample = (grad_d_base * dx).sum(dim=-1)
    high_dd = dd_per_sample[high_mask]
    mean_dd = float(high_dd.mean())

    return mean_dd, float(dx_norm), mean_dd < 0, n_high


def mode_exit_factor(pr84_factor, mean_dd, dx_norm, fired,
                     *, floor=PROJECTION_FLOOR):
    """Compute the final G step scale after mode-exit projection.

    If fired is False or pr84_factor already bounded the step below 1,
    return pr84_factor unchanged.  Otherwise shrink proportionally to
    how much of the step is mode-exiting.
    """
    if not fired or pr84_factor < 1.0:
        return pr84_factor, False
    if dx_norm < 1e-10:
        return pr84_factor, False
    shrink = max(floor, 1.0 + mean_dd / max(abs(mean_dd), dx_norm * 0.01))
    shrink = min(shrink, 1.0)
    if shrink < pr84_factor:
        return shrink, True
    return pr84_factor, False


class ModeExitProjectionRecorder(SmoothedBothBoundRecorder):
    def __init__(self, *, projection=True, projection_floor=PROJECTION_FLOOR, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 < projection_floor <= 1.0:
            raise ValueError("projection floor must lie in (0, 1]")
        self.projection = projection
        self.projection_floor = float(projection_floor)
        self._original_forward = None
        self._critic_ref = None
        self._generator_ref = None
        self._prior_ref = None

    def _arm_smoothed_critic(self):
        super()._arm_smoothed_critic()
        local = self._local or {}
        self._critic_ref = local.get("critic")
        self._generator_ref = local.get("generator")
        self._prior_ref = local.get("prior")

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
            g1_grads = grads(opt_g)
            rho = _rho(self.g_base, self.g1, self.gg0, g1_grads, self.metric_g)
            pr84_factor = min(1., self.curvature_bound / rho) if rho > 0 else 1.

            for p, b, n in zip(self._params(opt_g), self.g_base, self.g1):
                p.copy_(torch.lerp(b, n, pr84_factor) if pr84_factor < 1 else n)

            g_after_pr84 = [p.detach().clone() for p in self._params(opt_g)]

            mean_dd = 0.0
            dx_norm = 0.0
            fired = False
            projected = False
            factor = pr84_factor

            n_high = 0
            if (self.projection and pr84_factor >= 1.0
                    and self._critic_ref is not None
                    and self._generator_ref is not None
                    and self._prior_ref is not None
                    and self._original_forward is not None):
                with torch.enable_grad():
                    mean_dd, dx_norm, fired, n_high = directional_derivative_stats(
                        self._critic_ref, self._generator_ref, self._prior_ref,
                        self.g_base, g_after_pr84,
                        original_forward=self._original_forward)
                if fired:
                    factor, projected = mode_exit_factor(
                        pr84_factor, mean_dd, dx_norm,
                        fired, floor=self.projection_floor)
                    if projected:
                        for p, b, n in zip(self._params(opt_g), self.g_base, g_after_pr84):
                            p.copy_(torch.lerp(b, n, factor / pr84_factor)
                                    if factor < pr84_factor else n)

            self.row["g"] = dict(
                rho=rho, factor=factor, pr84_factor=pr84_factor,
                mean_dd=mean_dd, dx_norm=dx_norm, n_high=n_high,
                projected=projected)
        return None

    def receipt(self):
        value = super().receipt()
        closed = [r for r in self.records if not r["gate_open"]]
        projected_count = sum(bool(r.get("g", {}).get("projected")) for r in closed)
        value.update(
            method=METHOD, scratch_optimizer_policy=METHOD, smooth_critic=True,
            base_method=PR84_METHOD, projection=self.projection,
            projection_floor=self.projection_floor,
            projected_updates=projected_count,
            purity="GAN dynamics only — D directional derivative, no coverage/likelihood",
        )
        return value


@contextmanager
def pr84_mode_exit_projection(*, task="mode_hold", start_step=0, projection=True):
    """Yield ``(recorder, generated_host_source)``. ``projection=False`` matches PR84."""
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
    recorder = ModeExitProjectionRecorder(start_step=start_step, projection=projection)
    recorder.host_source = dict(task=task, original_function_sha256=original_sha,
                                generated_function_sha256=sha(source.encode()))
    ordinary_step = torch.optim.Adam.step
    original_d_loss = GANLoss.d_loss
    original_forward = SimpleMLPDiscriminator.forward
    recorder._original_forward = original_forward

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
        exec(compile(tree, f"<pr84-mode-exit-{task}>", "exec"), module.__dict__, namespace)
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
