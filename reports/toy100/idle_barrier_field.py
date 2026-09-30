"""PR84 plus one particle-local critic field, idle when the cloud is covered.

G still minimizes the relativistic adversarial loss on the smoothed critic.
During that G evaluation only, each fake particle looks against its own critic
slope. If that look never rises, or the highest critic on the look does not
beat every current fake logit, the added term is exactly zero. A rising look
that beats the fake cloud adds a detached push toward that critic peak. There
is no coverage, Chamfer, likelihood, or data-assignment loss.
"""

from contextlib import ExitStack, contextmanager
import ast
import math
from unittest.mock import patch

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from particlegan.gan_loss import GANLoss
from reports.toy100.extra_adam_scratch import HOSTS, sha, transformed_function
from reports.toy100.pr84_smoothed_candidate import SmoothedBothBoundRecorder


METHOD = "pr84_idle_anti_gradient_critic_field"
LOOK_STEP = 0.5
LOOK_COUNT = 4
SLOPE_EPS = 1e-3


class IdleBarrierRecorder(SmoothedBothBoundRecorder):
    """PR84 alternating update with an idle critic-peak correction on G."""

    def __init__(self, *, start_step=0, field=True):
        super().__init__(start_step=start_step)
        self.field = bool(field)
        self._step_fires = 0
        self._step_queries = 0
        self.field_fires = 0
        self.field_queries = 0

    def phases(self, step, opt_d, opt_g, local):
        self._step_fires = 0
        self._step_queries = 0
        for phase in super().phases(step, opt_d, opt_g, local):
            yield phase
        if self.records and not self.passthrough:
            self.records[-1]["field_fires"] = self._step_fires
            self.records[-1]["field_queries"] = self._step_queries
            self.field_fires += self._step_fires
            self.field_queries += self._step_queries

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     field=self.field, look_step=LOOK_STEP, look_count=LOOK_COUNT,
                     field_fires=self.field_fires, field_queries=self.field_queries,
                     purity="GAN dynamics only — no coverage/likelihood term")
        return value


def anti_gradient_peak(forward, module, points):
    """Return a detached unit push and slope toward a higher critic, or zeros.

    ``points`` is ``[N, 2]``. The walk steps against the local critic slope and
    stops at the first step that does not improve. Particles whose peak does
    not strictly beat the best current fake logit get a zero push.
    """
    with torch.no_grad():
        return _anti_gradient_peak(forward, module, points)


def _anti_gradient_peak(forward, module, points):
    base = forward(module, points)
    grad = torch.zeros_like(points)
    for dim in range(points.shape[-1]):
        shift = torch.zeros_like(points)
        shift[:, dim] = SLOPE_EPS
        grad[:, dim] = (forward(module, points + shift) - forward(module, points - shift)) / (2 * SLOPE_EPS)
    norm = grad.norm(dim=-1, keepdim=True)
    direction = grad / norm.clamp_min(1e-8)
    active = norm.squeeze(-1) > 1e-6
    best = base.clone()
    displacement = torch.zeros_like(points)
    for k in range(1, LOOK_COUNT + 1):
        if not bool(active.any()):
            break
        candidate = points - (LOOK_STEP * k) * direction
        value = forward(module, candidate)
        improved = active & (value > best)
        displacement = torch.where(improved.unsqueeze(-1), candidate - points, displacement)
        best = torch.where(improved, value, best)
        active = improved
    distance = displacement.norm(dim=-1)
    fire = (distance > 0) & (best > base) & (best > base.max())
    unit = torch.zeros_like(points)
    unit = torch.where(fire.unsqueeze(-1), displacement / distance.clamp_min(1e-8).unsqueeze(-1), unit)
    slope = torch.where(fire, (best - base) / distance.clamp_min(1e-8), torch.zeros_like(base))
    return unit, slope, int(fire.sum())


@contextmanager
def idle_barrier_field(*, task="mode_hold", start_step=0, field=True):
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
    recorder = IdleBarrierRecorder(start_step=start_step, field=field)
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
        smoothed = original_forward(self, x)
        if (recorder._smooth_on and recorder.enabled and not recorder.passthrough
                and x.ndim >= 2 and x.shape[-1] == 2 and recorder._smooth_width > 0):
            width = recorder._smooth_width
            vals = [smoothed]
            for dim in range(x.shape[-1]):
                shift = torch.zeros_like(x)
                shift[..., dim] = width
                vals.append(original_forward(self, x + shift))
                vals.append(original_forward(self, x - shift))
            smoothed = torch.stack(vals, 0).mean(0)
            if recorder.field and x.requires_grad:
                unit, slope, fires = anti_gradient_peak(original_forward, self, x.detach())
                recorder._step_queries += int(x.shape[0])
                recorder._step_fires += fires
                smoothed = smoothed + slope * (unit * x).sum(-1)
        return smoothed

    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<idle-barrier-{task}>", "exec"), module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(
            torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(SimpleMLPDiscriminator, "forward", smoothed_forward))
        yield recorder, source
