"""Particle-local exit clip on the frozen PR84 alternating stencil.

The curvature cap stays .25 and is unchanged. After that cap is applied, a
particle's output step is shortened only when both of these hold:

1. Its distance to the nearest real in the current minibatch already exceeds
   the real cloud's own nearest-neighbor fence (median + 3 robust standard
   deviations). That fence is the margin proxy: a particle farther from every
   observed real than reals are from each other is already at the edge of
   observed support.
2. The proposed displacement increases that nearest-real distance.

Interior particles, and edge particles whose step moves toward the reals,
keep the PR84 step. No mode center, HQ radius, or critic-slope rest damp
enters the rule.
"""

from contextlib import ExitStack, contextmanager
import ast
import math
from unittest.mock import patch

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from particlegan.gan_loss import GANLoss
from reports.toy100.extra_adam_scratch import HOSTS, sha, transformed_function
from reports.toy100.pr84_smoothed_candidate import (
    METHOD as PR84_METHOD, SmoothedBothBoundRecorder,
)


METHOD = "pr84_stencil_with_particle_exit_clip"
FENCE_MAD_SCALE = 3 * 1.4826


def support_fence(reals: torch.Tensor) -> torch.Tensor:
    """Robust upper fence on nearest-neighbor distances inside the real batch."""
    if reals.ndim != 2 or reals.shape[0] < 2 or reals.shape[1] != 2:
        raise ValueError("reals must have shape (N, 2), N >= 2")
    distance = torch.cdist(reals, reals)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.min(dim=1).values
    median = nearest.median()
    mad = (nearest - median).abs().median().clamp_min(1e-8)
    return median + FENCE_MAD_SCALE * mad


def exit_scales(before: torch.Tensor, after: torch.Tensor, reals: torch.Tensor,
                *, steps: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest per-particle scale in [0, 1] that does not worsen a thin margin.

    Scale stays 1 unless the particle is already outside the support fence and
    the full step would increase its distance to the real batch.
    """
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 2:
        raise ValueError("before and after must both have shape (P, 2)")
    fence = support_fence(reals)
    d0 = torch.cdist(before, reals).min(dim=1).values
    d1 = torch.cdist(after, reals).min(dim=1).values
    need = (d0 >= fence) & (d1 > d0 + 1e-8)
    scale = torch.ones(before.shape[0], dtype=before.dtype, device=before.device)
    if not bool(need.any()):
        return scale, fence
    step = after - before
    lo = torch.zeros_like(scale)
    hi = torch.ones_like(scale)
    for _ in range(steps):
        mid = (lo + hi) / 2
        moved = before + mid[:, None] * step
        distance = torch.cdist(moved, reals).min(dim=1).values
        fits = distance <= d0 + 1e-8
        lo = torch.where(need & fits, mid, lo)
        hi = torch.where(need & ~fits, mid, hi)
    return torch.where(need, lo, scale), fence


def realize_output_scales(clean, latents: torch.Tensor, before: torch.Tensor,
                          after: torch.Tensor, scales: torch.Tensor,
                          *, iterations: int = 4) -> float:
    """Land scaled particles on the clipped output step by moving only their latents.

    Shared generator weights stay at the curvature-scaled point. A latent row
    affects only its own output, so unscaled particles are left untouched.
    """
    active = scales < 1 - 1e-6
    if not bool(active.any()):
        return 0.0
    target = before + scales[:, None] * (after - before)
    z = latents.detach()
    residual = float("inf")
    index = active.nonzero(as_tuple=False).flatten().tolist()
    for _ in range(iterations):
        z_var = z.detach().requires_grad_(True)
        with torch.enable_grad():
            out = clean(z_var)
            err = out - target
            residual = float(err[active].detach().norm())
            if residual < 1e-5:
                break
            correction = torch.zeros_like(z)
            for i in index:
                rows = []
                for dim in range(out.shape[-1]):
                    grad, = torch.autograd.grad(out[i, dim], z_var, retain_graph=True)
                    rows.append(grad[i])
                jacobian = torch.stack(rows, 0)
                delta = torch.linalg.lstsq(jacobian, err[i].detach()).solution.reshape(-1)
                correction[i] = delta
        z = z_var.detach() - correction
    latents.copy_(z)
    return residual


class ExitAwareRecorder(SmoothedBothBoundRecorder):
    """Frozen PR84 update, plus the particle-local exit clip when enabled."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.clip_enabled = False
        self._reals = None
        self.clip_count = 0
        self.clipped_particles = 0

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        result = super().step(optimizer, ordinary_step, closure)
        if not self._should_clip(optimizer):
            return result
        self._clip_exits()
        return result

    def _should_clip(self, optimizer) -> bool:
        return bool(
            self.clip_enabled and not self.passthrough and self.phase == 2
            and self.optimizers is not None and optimizer is self.optimizers[1]
            and self._reals is not None and self._local)

    def _clean_positions(self, weights):
        local = self._local
        generator, prior = local["generator"], local["prior"]
        saved = [p.detach().clone() for p in self._params(self.optimizers[1])]
        for p, value in zip(self._params(self.optimizers[1]), weights):
            p.copy_(value)
        clean = getattr(generator, "model", generator)
        positions = clean(prior.z).detach()
        for p, value in zip(self._params(self.optimizers[1]), saved):
            p.copy_(value)
        return positions

    def _clip_exits(self):
        local = self._local
        generator, prior = local["generator"], local["prior"]
        opt_g = self.optimizers[1]
        proposed = [p.detach().clone() for p in self._params(opt_g)]
        before = self._clean_positions(self.g_base)
        after = self._clean_positions(proposed)
        scales, fence = exit_scales(before, after, self._reals)
        shrunk = int((scales < 1 - 1e-6).sum())
        self.row["exit_fence"] = float(fence)
        self.row["exit_clipped"] = shrunk
        self.row["exit_min_scale"] = float(scales.min())
        if shrunk == 0:
            return
        clean = getattr(generator, "model", generator)
        for p, value in zip(self._params(opt_g), proposed):
            p.copy_(value)
        residual = realize_output_scales(clean, prior.z, before, after, scales)
        self.clip_count += 1
        self.clipped_particles += shrunk
        self.row["exit_residual"] = residual
        host_step = self.start_step + int(self.row.get("outer_step", 0))
        print(
            f"event=EXIT_CLIP update={host_step} clipped={shrunk} "
            f"min_scale={float(scales.min()):.4f} fence={float(fence):.4f} "
            f"residual={residual:.3e}",
            flush=True,
        )

    def note_reals(self, batch: torch.Tensor):
        if self.clip_enabled and self.phase == 2 and batch.ndim == 2 and batch.shape[-1] == 2:
            self._reals = batch.detach()

    def receipt(self):
        value = super().receipt()
        value.update(
            method=METHOD, scratch_optimizer_policy=METHOD,
            base_method=PR84_METHOD, exit_clip=self.clip_enabled,
            exit_clip_updates=self.clip_count,
            exit_clipped_particles=self.clipped_particles,
            margin_proxy="nearest real in the current minibatch",
            support_fence="median NN + 3 * 1.4826 * MAD of real nearest neighbors",
        )
        return value


@contextmanager
def exit_aware_candidate(*, task="mode_hold", start_step=0):
    """PR84 smoothed alternating host with an optional particle exit clip."""
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
    recorder = ExitAwareRecorder(start_step=start_step)
    recorder.host_source = dict(task=task, original_function_sha256=original_sha,
                                generated_function_sha256=sha(source.encode()))
    ordinary_step = torch.optim.Adam.step
    original_d_loss = GANLoss.d_loss
    original_forward = SimpleMLPDiscriminator.forward
    original_sample = module.sample_ring

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

    def observing_sample(*args, **kwargs):
        batch = original_sample(*args, **kwargs)
        recorder.note_reals(batch)
        return batch

    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<exit-aware-{task}>", "exec"), module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(
            torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(SimpleMLPDiscriminator, "forward", smoothed_forward))
        stack.enter_context(patch.object(module, "sample_ring", observing_sample))
        yield recorder, source
