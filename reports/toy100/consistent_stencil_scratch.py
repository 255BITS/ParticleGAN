"""One PR84 ablation: both players use the same frozen spatial critic stencil.

Research scope is the two hosts of the alternating replay adapter. The stencil
is supported only for a 2D SimpleMLPDiscriminator (as in PR84); trajectory keeps
the original bounded alternating update. This is not a production optimizer.
The width policy and bounds are inherited from PR84, without a slope rest gate.
Width is computed once at the outer base point and frozen across every replay.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100 import alternating_curvature_scratch as parent


METHOD = "alternating_consistent_frozen_critic_stencil"


def stencil(forward, model, x, width):
    if width == 0:
        return forward(model, x)
    values = [forward(model, x)]
    for dim in range(x.shape[-1]):
        shift = torch.zeros_like(x)
        shift[..., dim] = width
        values.extend((forward(model, x + shift), forward(model, x - shift)))
    return torch.stack(values).mean(0)


class ConsistentStencilRecorder(parent.BothBoundRecorder):
    def __init__(self, *args, smooth_cap=.15, **kwargs):
        if not math.isfinite(smooth_cap) or smooth_cap < 0:
            raise ValueError("invalid stencil cap")
        super().__init__(*args, **kwargs)
        self.smooth_cap = smooth_cap
        self.width = 0.
        self.smoothing_active = False
        self.smoothing_calls = 0

    @torch.no_grad()
    def set_width(self, local):
        self.width = 0.
        if self.smooth_cap == 0 or local.get("slow") is not None:
            return
        critic, generator, prior = (local.get(k) for k in ("critic", "generator", "prior"))
        if critic is None or generator is None or prior is None:
            return
        while not isinstance(critic, SimpleMLPDiscriminator) and hasattr(critic, "model"):
            critic = critic.model
        if not isinstance(critic, SimpleMLPDiscriminator):
            return
        clean = getattr(generator, "model", generator)
        points = clean(prior.z).detach()
        if points.ndim != 2 or points.shape[-1] != 2:
            return
        squared = 0.
        for dim in range(points.shape[-1]):
            shift = torch.zeros_like(points)
            shift[:, dim] = .001
            squared = squared + ((critic(points + shift) - critic(points - shift)) / .002).square()
        sharpness = float(squared.mean().sqrt())
        if not math.isfinite(sharpness):
            raise FloatingPointError("nonfinite stencil sharpness")
        # PR84's flat-critic convention is retained for this isolated ablation.
        self.width = min(self.smooth_cap, .5 / sharpness) if sharpness > 1e-6 else 0.
        self.row.update(critic_sharpness=sharpness, critic_width=self.width)

    def phases(self, step, opt_d, opt_g, local):
        self.smoothing_active = False
        try:
            for phase in super().phases(step, opt_d, opt_g, local):
                if not self.passthrough and phase == 0:
                    self.set_width(local)
                self.smoothing_active = not self.passthrough
                yield phase
                self.smoothing_active = False
        finally:
            self.smoothing_active = False

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     smooth_cap=self.smooth_cap, smoothing_calls=self.smoothing_calls,
                     smoothing_scope="D and G, including D cap; 2D SimpleMLPDiscriminator only",
                     width_scope="frozen at outer base across D, G and all replays",
                     shared_gate_eligible=False)
        return value


@contextmanager
def consistent_stencil(task="mode_hold", smooth_cap=.15, **options):
    original_forward = SimpleMLPDiscriminator.forward
    def factory(**kwargs):
        return ConsistentStencilRecorder(smooth_cap=smooth_cap, **kwargs)
    with patch.object(parent, "BothBoundRecorder", factory):
        with parent.alternating_curvature(task=task, bound_d=True, **options) as (recorder, source):
            def forward(model, x):
                if recorder.smoothing_active and recorder.width and x.ndim == 2 and x.shape[-1] == 2:
                    recorder.smoothing_calls += 1
                    return stencil(original_forward, model, x, recorder.width)
                return original_forward(model, x)
            with patch.object(SimpleMLPDiscriminator, "forward", forward):
                yield recorder, source
