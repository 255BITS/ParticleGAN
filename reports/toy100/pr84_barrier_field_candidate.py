"""PR84 G critic plus one barrier-crossing field probe.

D stays on the sharp critic. G still scores the PR84 five-point stencil.
When that stencil sits in a local moat, the same width supplies a unit
direction from the stencil's central difference, and G reads the sharp
critic two and four widths opposite that gradient. If either probe beats
the stencil mean, G uses that probe value with the offset detached, so
the step follows the critic past the lip. Otherwise the score is the
stencil and the update matches PR84. No data-coverage term, assignment,
or step clip.
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
    SMOOTH_WIDTH_CAP, SmoothedBothBoundRecorder, pr84_smoothed_candidate,
)


METHOD = "pr84_stencil_plus_antigradient_barrier_field"
PROBE_SCALES = (2.0, 4.0)


class BarrierFieldRecorder(SmoothedBothBoundRecorder):
    """PR84 recorder that counts barrier-probe wins on G's proposal."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self._barrier_uses = 0
        self._barrier_seen = 0

    def step(self, optimizer, ordinary_step, closure=None):
        self._barrier_uses = 0
        self._barrier_seen = 0
        result = super().step(optimizer, ordinary_step, closure)
        if (not self.passthrough and self.optimizers is not None
                and optimizer is self.optimizers[1] and self.phase == 1):
            self.row["barrier_uses"] = self._barrier_uses
            self.row["barrier_seen"] = self._barrier_seen
        return result

    def receipt(self):
        value = super().receipt()
        uses = [row.get("barrier_uses", 0) for row in self.records]
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     shared_gate_eligible=False, barrier_field=True,
                     barrier_probe_scales=list(PROBE_SCALES),
                     barrier_uses_total=int(sum(uses)),
                     barrier_steps_fired=int(sum(v > 0 for v in uses)))
        return value


def barrier_score(original_forward, module, x, width):
    """Stencil mean, or a detached anti-gradient probe when that probe is higher."""
    center = original_forward(module, x)
    vals = [center]
    comps = []
    for dim in range(x.shape[-1]):
        shift = torch.zeros_like(x)
        shift[..., dim] = width
        plus = original_forward(module, x + shift)
        minus = original_forward(module, x - shift)
        vals.extend((plus, minus))
        comps.append((plus - minus) / (2 * width))
    local = torch.stack(vals, 0).mean(0)
    grad = torch.stack(comps, -1)
    norm = grad.norm(dim=-1, keepdim=True)
    usable = norm.squeeze(-1) > 1e-6
    if not bool(usable.detach().any()):
        return local, usable, usable
    direction = (grad / norm.clamp_min(1e-12)).detach()
    best = None
    best_score = None
    for scale in PROBE_SCALES:
        probed = original_forward(module, x - (scale * width) * direction)
        score = probed.detach()
        if best is None:
            best, best_score = probed, score
        else:
            take = score > best_score
            best = torch.where(take, probed, best)
            best_score = torch.where(take, score, best_score)
    use = usable & (best_score > local.detach())
    return torch.where(use, best, local), use, usable


@contextmanager
def pr84_barrier_field_candidate(*, task="mode_hold", start_step=0):
    """Yield ``(recorder, generated_host_source)`` for one declared toy host."""
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
    recorder = BarrierFieldRecorder(start_step=start_step)
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

    def field_forward(self, x):
        if (recorder._smooth_on and recorder.enabled and not recorder.passthrough
                and x.ndim >= 2 and x.shape[-1] == 2 and recorder._smooth_width > 0):
            score, use, seen = barrier_score(original_forward, self, x, recorder._smooth_width)
            if recorder.phase == 1:
                recorder._barrier_uses += int(use.detach().sum())
                recorder._barrier_seen += int(seen.detach().sum())
            return score
        return original_forward(self, x)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<pr84-barrier-field-{task}>", "exec"), module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(
            torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(SimpleMLPDiscriminator, "forward", field_forward))
        yield recorder, source


__all__ = ["METHOD", "PROBE_SCALES", "barrier_score", "pr84_barrier_field_candidate",
           "pr84_smoothed_candidate", "SMOOTH_WIDTH_CAP"]
