"""GAN-5: real-sample Laplacian basin on the PR84 critic.

PR84 is unchanged for G: five-point spatial stencil and alternating curvature
bounds .25 / 3. The only addition is a discriminator penalty on the
finite-difference Laplacian of the sharp critic at real ring points. Positive
Laplacian means the score is locally convex, so a relativistic generator is
not restored toward that sample. Penalizing relu(Laplacian) pushes real
points toward local maxima of D. The term is part of the adversarial critic
loss. It does not assign particles, match centers, or add a likelihood.

The stencil is defined for planar critic inputs. Higher-dimensional hosts
(the trajectory fast vector) leave the term at zero, so that host stays the
PR84 update.
"""

from contextlib import ExitStack, contextmanager

import torch

from particlegan.grad_regularizers import GradRegularizer
from reports.toy100.pr84_smoothed_candidate import pr84_smoothed_candidate


METHOD = "pr84_real_laplacian_basin"
BASIN_COEFF = 0.1
BASIN_EPS = 0.1


def real_laplacian(critic, points, eps=BASIN_EPS):
    """Finite-difference Laplacian of a scalar critic at planar points."""
    if points.ndim != 2 or points.shape[-1] != 2:
        return None
    center = points.detach()
    score = critic(center)
    stencil = -4.0 * score
    for dim in range(center.shape[-1]):
        shift = torch.zeros_like(center)
        shift[:, dim] = eps
        stencil = stencil + critic(center + shift) + critic(center - shift)
    return stencil / (eps * eps)


@contextmanager
def gan5_spectral_basin(*, task="mode_hold", start_step=0, basin=True):
    """PR84 smoothed candidate plus the real-sample basin penalty."""
    original_call = GradRegularizer.__call__

    def penalized_call(self, D, x_real, x_fake, step=1, generator=None):
        penalty = original_call(self, D, x_real, x_fake, step, generator)
        recorder = penalized_call.recorder
        if (recorder is None or not getattr(recorder, "_basin", False)
                or not recorder.enabled or recorder.passthrough
                or recorder.phase not in (0, 1, 2)):
            return penalty
        lap = real_laplacian(D, x_real)
        if lap is None:
            return penalty
        extra = BASIN_COEFF * torch.relu(lap).mean()
        recorder.row["basin_penalty"] = float(extra.detach())
        recorder.row["basin_lap_mean"] = float(lap.detach().mean())
        return penalty + extra

    penalized_call.recorder = None
    with pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
        recorder._basin = bool(basin)
        penalized_call.recorder = recorder
        with ExitStack() as stack:
            stack.enter_context(patch_call(original_call, penalized_call))
            yield recorder, source


@contextmanager
def patch_call(original, replacement):
    from unittest.mock import patch
    with patch.object(GradRegularizer, "__call__", replacement):
        yield original
