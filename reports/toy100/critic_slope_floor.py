"""D-only occupied slope floor on the PR #84 smoothed-critic candidate.

G still takes its step from the five-point stencil, and both curvature bounds
stay .25 / 3. The width rule is unchanged. This does not scale or reject G
steps. It adds the flatness half of the published ``d_asym`` penalty, measured
on that same stencil at the real and fake samples D already visits, so a
matched critic cannot sit at zero smoothed slope. Empty-space interpolates are
not penalized, so unoccupied basins are not pinned to the floor.
"""

from contextlib import contextmanager
import math

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from particlegan.grad_regularizers import GradientPenalty
from reports.toy100.pr84_smoothed_candidate import SMOOTH_WIDTH_CAP, pr84_smoothed_candidate
from unittest.mock import patch


METHOD = "pr84_stencil_with_occupied_smoothed_slope_floor"
# Flatness half of GradRegularizer d_asym: 0.25 * relu(1 - ||grad||)^2,
# then the same 1/2 real/fake average used by GradientPenalty.penalty.
FLATNESS_WEIGHT = .25
SLOPE_FLOOR = 1.


def _mlp(module):
    while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
        module = module.model
    return module if isinstance(module, SimpleMLPDiscriminator) else None


def stencil_values(module, points, width, forward):
    vals = [forward(module, points)]
    for dim in range(points.shape[-1]):
        shift = torch.zeros_like(points)
        shift[..., dim] = width
        vals.append(forward(module, points + shift))
        vals.append(forward(module, points - shift))
    return torch.stack(vals, 0).mean(0)


def smoothed_slope(module, points, width, forward):
    """Per-sample input slope of the five-point field, with a D-parameter graph."""
    lone = points.detach().requires_grad_(True)
    score = stencil_values(module, lone, width, forward)
    if not score.requires_grad:
        return torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
    grad, = torch.autograd.grad(score.sum(), lone, create_graph=True)
    reduce = tuple(range(1, grad.ndim))
    return grad.square().sum(reduce).sqrt()


def occupied_floor(critic, real, fake, width, forward):
    module = _mlp(critic)
    if (module is None or width <= 0 or real.shape[-1] != 2 or fake.shape[-1] != 2):
        return real.new_zeros(())
    slopes = [smoothed_slope(module, points, width, forward) for points in (real, fake)]
    phi = [FLATNESS_WEIGHT * torch.relu(SLOPE_FLOOR - slope).square().mean() for slope in slopes]
    return .5 * (phi[0] + phi[1])


@contextmanager
def critic_slope_floor(*, task="mode_hold", start_step=0):
    """Yield ``(recorder, source)`` with the D floor wrapped around PR #84."""
    original_forward = SimpleMLPDiscriminator.forward
    original_call = GradientPenalty.__call__
    with pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
        def call(self, discriminator, real, fake, step=1, generator=None):
            base = original_call(self, discriminator, real, fake, step, generator)
            if not recorder.enabled or recorder.passthrough or recorder.phase is None:
                return base
            width = recorder._smooth_width if recorder._smooth_width > 0 else SMOOTH_WIDTH_CAP
            extra = occupied_floor(discriminator, real, fake, width, original_forward)
            if recorder.phase == 0 and extra.requires_grad:
                slope = smoothed_slope(_mlp(discriminator), fake, width, original_forward)
                recorder.row["occupied_smooth_slope"] = float(slope.mean().detach())
                recorder.row["occupied_floor"] = float(extra.detach())
                recorder.row["occupied_floor_width"] = width
                print(
                    f"floor step={recorder.row.get('outer_step')} "
                    f"smooth_slope={recorder.row['occupied_smooth_slope']:.6f} "
                    f"pen={recorder.row['occupied_floor']:.6f} width={width:.4f}",
                    flush=True)
            return base + extra

        with patch.object(GradientPenalty, "__call__", call):
            yield recorder, source


def floor_receipt(recorder):
    value = recorder.receipt()
    slopes = [row.get("occupied_smooth_slope") for row in recorder.records
              if "occupied_smooth_slope" in row]
    value.update(method=METHOD, scratch_optimizer_policy=METHOD, shared_gate_eligible=False,
                 occupied_slope_floor=SLOPE_FLOOR, flatness_weight=FLATNESS_WEIGHT,
                 g_step_unchanged=True, slope_floor_samples=len(slopes),
                 occupied_smooth_slope_mean=(sum(slopes) / len(slopes) if slopes else None))
    if not math.isfinite(value["occupied_smooth_slope_mean"] or 0.) and slopes:
        raise FloatingPointError("nonfinite occupied slope")
    return value
