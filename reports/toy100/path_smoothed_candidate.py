"""PR84 smoothed G critic plus a path-crossing direction change.

The alternating Adam update, curvature bounds, and five-point stencil are the
selected adapter. During G's critic evaluation only, samples that face a score
dip in front of an empty higher basin have their output loss-gradient rotated
onto that ray. Gradient norms, Adam moments, and the curvature cap are unchanged.
"""
from contextlib import contextmanager
import json
from unittest.mock import patch

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100 import pr84_smoothed_candidate as frozen
from reports.toy100.path_acquisition import path_crossing_directions, redirect_loss_grad


METHOD = "pr84_smoothed_g_with_path_crossing_direction"


class PathSmoothedRecorder(frozen.SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, path=True):
        super().__init__(start_step=start_step)
        self.path = bool(path)
        self._path_hits = 0
        self.path_forwards = 0

    def phases(self, step, opt_d, opt_g, local):
        for phase in super().phases(step, opt_d, opt_g, local):
            yield phase
        if self.records:
            self.records[-1]["path_redirects"] = self._path_hits
        if self.outer_steps % 50 == 0 and self.path:
            print(json.dumps(dict(event="PATH_STEP", step=self.outer_steps,
                                  redirects=self._path_hits)), flush=True)
        self._path_hits = 0

    def note_redirects(self, count):
        self._path_hits += int(count)
        self.path_forwards += 1

    def receipt(self):
        result = super().receipt()
        redirects = [row.get("path_redirects", 0) for row in self.records]
        result.update(method=METHOD, scratch_optimizer_policy=METHOD,
                      path_crossing=self.path, path_forwards=self.path_forwards,
                      path_redirects_total=sum(redirects),
                      path_steps_with_redirect=sum(v > 0 for v in redirects),
                      acquisition_signal="path-crossing critic ray; direction only")
        return result


def _support(recorder):
    local = recorder._local or {}
    generator, prior = local.get("generator"), local.get("prior")
    if generator is None or prior is None or not hasattr(prior, "z"):
        return None
    clean = getattr(generator, "model", generator)
    return clean(prior.z).detach()


@contextmanager
def path_smoothed_candidate(*, task="mode_hold", start_step=0, path=True):
    def factory(*, start_step=0):
        return PathSmoothedRecorder(start_step=start_step, path=path)

    with patch.object(frozen, "SmoothedBothBoundRecorder", factory):
        with frozen.pr84_smoothed_candidate(task=task, start_step=start_step) as (recorder, source):
            installed = SimpleMLPDiscriminator.forward

            def path_forward(self, x):
                value = installed(self, x)
                if (recorder.path and recorder.enabled and not recorder.passthrough
                        and recorder._smooth_on and torch.is_tensor(x) and x.requires_grad
                        and x.ndim == 2 and x.shape[-1] == 2):
                    support = _support(recorder)
                    if support is not None:
                        def score(points):
                            return installed(self, points).reshape(-1)

                        direction = path_crossing_directions(score, x.detach(), support)
                        hits = int((direction.norm(dim=1) > 0).sum())
                        recorder.note_redirects(hits)
                        if hits:
                            x.register_hook(lambda grad, direction=direction: redirect_loss_grad(grad, direction))
                return value

            with patch.object(SimpleMLPDiscriminator, "forward", path_forward):
                yield recorder, source
