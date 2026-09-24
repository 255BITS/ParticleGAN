"""Stall-reach G step, then #113's mode-exit shrink on that bounded step.

PR #107 stall reach is unchanged: slope/stall stencil, G curvature bound
.25, D curvature bound 3. The curvature bound is applied first and is never
loosened. PR #113 measured the critic directional derivative of the G step
on high-D particles and shrank the step, but only when the bound had
accepted the full Adam step. Here that same shrink is applied to the
displacement the bound already kept, including when the bound clipped it.

Purity: adversarial dynamics only. The signal is D and its input gradient
along the accepted particle step. No coverage, anchor, assignment,
likelihood, mode quota, or clip ladder.
"""

from contextlib import contextmanager
import inspect

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100 import pr84_reach_candidate as reach


METHOD = "stall_reach_dd_exit_on_bounded_step"
PROJECTION_FLOOR = 0.3


def directional_derivative_stats(critic, generator, prior, g_base_all, g_new_all, *,
                                 original_forward):
    """#113 diagnostic: mean grad_D · Δx on particles with D(x_base) above the median.

    Returns ``(mean_dd, dx_norm, exiting, n_high)``. ``exiting`` is true when
    that mean is negative: the step lowers D on the high-D half of the cloud.
    """
    critic_module = critic
    while not isinstance(critic_module, SimpleMLPDiscriminator) and hasattr(critic_module, "model"):
        critic_module = critic_module.model
    if not isinstance(critic_module, SimpleMLPDiscriminator):
        return 0.0, 0.0, False, 0
    clean_gen = getattr(generator, "model", generator)
    positional = [p for p in inspect.signature(clean_gen.forward).parameters.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                  and p.default is inspect.Parameter.empty]
    if len(positional) != 1 or not hasattr(prior, "z"):
        return 0.0, 0.0, False, 0
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

    if x_base.ndim != 2 or x_base.shape[-1] != 2 or x_new.shape != x_base.shape:
        return 0.0, 0.0, False, 0
    dx = x_new - x_base
    dx_norm = dx.norm(dim=-1).mean()
    if float(dx_norm) < 1e-10:
        return 0.0, 0.0, False, 0

    x_base_grad = x_base.detach().requires_grad_(True)
    d_base = original_forward(critic_module, x_base_grad)
    grad_d = torch.autograd.grad(d_base.sum(), x_base_grad, create_graph=False)[0].detach()
    d_vals = d_base.detach().reshape(-1)
    high_mask = d_vals >= d_vals.median()
    n_high = int(high_mask.sum())
    if n_high == 0:
        return 0.0, float(dx_norm), False, 0
    mean_dd = float((grad_d * dx).sum(dim=-1)[high_mask].mean())
    return mean_dd, float(dx_norm), mean_dd < 0, n_high


def bounded_exit_shrink(mean_dd, dx_norm, *, floor=PROJECTION_FLOOR):
    """Scale applied to the already curvature-bounded displacement.

    This is #113's shrink with the ``factor == 1`` gate removed. The result
    is in ``[floor, 1]``. It never exceeds 1, so it cannot loosen the bound.
    A non-negative directional derivative leaves the bounded step alone.
    """
    if mean_dd >= 0 or dx_norm < 1e-10:
        return 1.0, False
    if not 0.0 < floor <= 1.0:
        raise ValueError("projection floor must lie in (0, 1]")
    shrink = max(floor, 1.0 + mean_dd / max(abs(mean_dd), dx_norm * 0.01))
    shrink = min(float(shrink), 1.0)
    return shrink, shrink < 1.0


class DDExitBoundedRecorder(reach.ReachRecorder):
    """Stall reach, then project the accepted G step along the mode-exit signal."""

    ramp = "stall"
    game_bound = False

    def __init__(self, *, start_step=0, projection=True):
        super().__init__(start_step=start_step)
        if self.curvature_bound != base.G_CURVATURE_BOUND:
            raise RuntimeError("G curvature bound must stay at the stall-reach value")
        if self.d_curvature_bound != base.D_CURVATURE_BOUND:
            raise RuntimeError("D curvature bound must stay at the stall-reach value")
        self.projection = bool(projection)
        self.projection_floor = PROJECTION_FLOOR
        self._original_forward = None

    def step(self, optimizer, ordinary_step, closure=None):
        result = super().step(optimizer, ordinary_step, closure)
        if self.projection:
            self._project_bounded_step(optimizer)
        return result

    def _project_bounded_step(self, optimizer):
        if (self.passthrough or self.phase != 2 or self.optimizers is None
                or optimizer is not self.optimizers[1] or "g" not in self.row):
            return
        local = self._local or {}
        critic, generator, prior = local.get("critic"), local.get("generator"), local.get("prior")
        if critic is None or generator is None or prior is None or self._original_forward is None:
            raise RuntimeError("DD-exit projection missing the critic field")
        opt_g = self.optimizers[1]
        bounded = [p.detach().clone() for p in self._params(opt_g)]
        curv_factor = float(self.row["g"]["factor"])
        with torch.enable_grad():
            mean_dd, dx_norm, exiting, n_high = directional_derivative_stats(
                critic, generator, prior, self.g_base, bounded,
                original_forward=self._original_forward)
        shrink, projected = bounded_exit_shrink(
            mean_dd, dx_norm, floor=self.projection_floor) if exiting else (1.0, False)
        final = curv_factor * shrink
        if final > curv_factor + 1e-12:
            raise RuntimeError("DD-exit projection loosened the curvature bound")
        if projected:
            with torch.no_grad():
                for p, start, accepted in zip(self._params(opt_g), self.g_base, bounded):
                    p.copy_(torch.lerp(start, accepted, shrink))
        self.row["g"].update(
            factor=final, curv_factor=curv_factor, exit_shrink=shrink,
            mean_dd=mean_dd, dx_norm=dx_norm, n_high=n_high,
            projected=projected, negative_dd=bool(exiting),
            clipped_exit=bool(projected and curv_factor < 1.0),
            legacy_full_step_exit=bool(exiting and curv_factor >= 1.0))
        update = self.outer_steps + 1
        if update % 50 == 0:
            g = self.row["g"]
            print(
                '{"event":"DD_EXIT","update":%d,"curv_factor":%.6g,"factor":%.6g,'
                '"mean_dd":%.6g,"projected":%s,"clipped_exit":%s,"width":%s}' % (
                    update, curv_factor, final, mean_dd,
                    "true" if projected else "false",
                    "true" if g["clipped_exit"] else "false",
                    "null" if self.row.get("critic_width") is None else "%.6g" % self.row["critic_width"]),
                flush=True)

    def receipt(self):
        value = super().receipt()
        closed = [r for r in self.records if not r.get("gate_open")]

        def count(flag):
            return int(sum(bool(r.get("g", {}).get(flag)) for r in closed))

        curv = [r["g"]["curv_factor"] for r in closed if "curv_factor" in r.get("g", {})]
        shrinks = [r["g"]["exit_shrink"] for r in closed if r.get("g", {}).get("projected")]
        first = next((r for r in closed if r.get("g", {}).get("clipped_exit")), None)
        value.update(
            method=METHOD, scratch_optimizer_policy=METHOD,
            projection=self.projection, projection_floor=self.projection_floor,
            g_curvature_bound=self.curvature_bound, d_curvature_bound=self.d_curvature_bound,
            projected_updates=count("projected"),
            negative_dd_updates=count("negative_dd"),
            clipped_exit_updates=count("clipped_exit"),
            legacy_full_step_exit_updates=count("legacy_full_step_exit"),
            curv_factor_min=min(curv) if curv else None,
            curv_factor_mean=(sum(curv) / len(curv)) if curv else None,
            exit_shrink_min=min(shrinks) if shrinks else None,
            exit_shrink_mean=(sum(shrinks) / len(shrinks)) if shrinks else None,
            first_clipped_update=None if first is None else first["outer_step"],
            first_clipped_curv_factor=None if first is None else first["g"]["curv_factor"],
            first_clipped_mean_dd=None if first is None else first["g"]["mean_dd"],
            purity="GAN dynamics only: D directional derivative on the curvature-bounded G step",
        )
        return value


@contextmanager
def pr84_dd_exit_bounded(*, task="mode_hold", start_step=0, projection=True):
    """Stall-reach host. ``projection=False`` is stall reach with no DD shrink."""
    original_cls = base.SmoothedBothBoundRecorder
    original_forward = SimpleMLPDiscriminator.forward

    def init(self, *, start_step=0):
        DDExitBoundedRecorder.__init__(self, start_step=start_step, projection=projection)

    base.SmoothedBothBoundRecorder = type(
        "DDExitBoundedRecorder", (DDExitBoundedRecorder,),
        dict(reach=reach.REACH, ramp="stall", game_bound=False, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
            if recorder.curvature_bound != base.G_CURVATURE_BOUND or recorder.d_curvature_bound != base.D_CURVATURE_BOUND:
                raise RuntimeError("stall-reach curvature bounds changed")
            if recorder.ramp != "stall" or recorder.game_bound:
                raise RuntimeError("stall reach was not left on its stall stencil")
            recorder._original_forward = original_forward
            yield recorder, source
    finally:
        base.SmoothedBothBoundRecorder = original_cls


__all__ = [
    "METHOD", "DDExitBoundedRecorder", "bounded_exit_shrink",
    "directional_derivative_stats", "pr84_dd_exit_bounded",
]
