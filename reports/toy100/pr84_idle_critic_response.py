"""PR84 G critic, plus one idle-when-covered response.

D stays on the sharp critic. G scores the PR84 five-point stencil. After
each D step the current real batch is compared with the clean particles.
The response arms only when three critic facts hold together: some real
point's sharp critic beats every clean particle by a fixed margin, that
point lies farther from the particle cloud than a fixed gap, and the
nearest particle's sharp central difference points away from it. Query
points near that particle and still oriented away then read the sharp
critic at the first segment sample where that critic points toward the
peak. The offset is detached.

If any of the three facts fails, every G phase uses the PR84 stencil and
nothing else. No coverage loss, assignment, likelihood, or step clip.
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
    SMOOTH_WIDTH_CAP, SmoothedBothBoundRecorder,
)


METHOD = "pr84_stencil_plus_idle_when_covered_critic_response"
CRITIC_MARGIN = 0.5
POCKET_GAP = 1.0
NEAR_RADIUS = 0.75
SEGMENT_PROBES = 8


def stencil_mean(original_forward, module, x, width):
    vals = [original_forward(module, x)]
    for dim in range(x.shape[-1]):
        shift = torch.zeros_like(x)
        shift[..., dim] = width
        vals.append(original_forward(module, x + shift))
        vals.append(original_forward(module, x - shift))
    return torch.stack(vals, 0).mean(0)


def sharp_gradient(original_forward, module, x, width):
    comps = []
    for dim in range(x.shape[-1]):
        shift = torch.zeros_like(x)
        shift[..., dim] = width
        plus = original_forward(module, x + shift)
        minus = original_forward(module, x - shift)
        comps.append((plus - minus) / (2 * width))
    return torch.stack(comps, -1)


def _crossing_travel(original_forward, module, origin, pocket, width):
    """Distance from ``origin`` toward ``pocket`` where the sharp critic first points that way."""
    direction = pocket - origin
    length = float(direction.norm())
    if length <= width:
        return None
    unit = direction / length
    slots = torch.linspace(width, length, SEGMENT_PROBES)
    points = origin + slots.unsqueeze(-1) * unit
    grad = sharp_gradient(original_forward, module, points, width)
    toward = (grad * unit).sum(-1)
    positive = toward > 0.
    if not bool(positive.any()):
        return None
    return float(slots[int(positive.nonzero()[0])])


def idle_gate(original_forward, module, real, particles, width):
    """Return ``(respond, pocket, anchor, travel)``. ``respond`` is false on a covered cloud."""
    real_score = original_forward(module, real).detach().reshape(-1)
    particle_score = original_forward(module, particles).detach().reshape(-1)
    peak_i = int(real_score.argmax())
    peak_score = float(real_score[peak_i])
    if peak_score <= float(particle_score.max()) + CRITIC_MARGIN:
        return False, None, None, None, None
    pocket = real[peak_i].detach()
    dist = (particles - pocket).norm(dim=-1)
    nearest = int(dist.argmin())
    if float(dist[nearest]) <= POCKET_GAP:
        return False, None, None, None, None
    origin = particles[nearest].detach()
    grad = sharp_gradient(original_forward, module, origin.unsqueeze(0), width)
    toward = pocket - origin
    if float((grad.reshape(-1) * toward).sum()) >= 0.:
        return False, None, None, None, None
    travel = _crossing_travel(original_forward, module, origin, pocket, width)
    if travel is None:
        return False, None, None, None, None
    return True, pocket, origin, travel, peak_score


def response_score(original_forward, module, x, width, pocket, anchor, peak_score, travel):
    """PR84 stencil, except rows near an away-facing anchor read the critic past the lip."""
    local = stencil_mean(original_forward, module, x, width)
    near = (x - anchor).norm(dim=-1) <= NEAR_RADIUS
    if not bool(near.detach().any()):
        return local, torch.zeros(x.shape[0], dtype=torch.bool)
    grad = sharp_gradient(original_forward, module, x, width)
    toward = (pocket - x).detach()
    away = (grad * toward).sum(-1) < 0.
    low = local.detach().reshape(-1) < float(peak_score) - CRITIC_MARGIN
    use = near & away & low
    if not bool(use.detach().any()):
        return local, use
    step = toward / toward.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    probed = original_forward(module, x + (travel * step))
    return torch.where(use, probed, local), use


class IdleCriticRecorder(SmoothedBothBoundRecorder):
    """PR84 recorder that arms the response once per D step and counts G-phase uses."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self._respond = False
        self._pocket = None
        self._anchor = None
        self._peak_score = None
        self._travel = None
        self._phase_forwards = 0
        self._real_batch = None
        self._response_uses = 0

    def phases(self, step, opt_d, opt_g, local):
        self._real_batch = None
        self._phase_forwards = 0
        yield from super().phases(step, opt_d, opt_g, local)

    def step(self, optimizer, ordinary_step, closure=None):
        self._response_uses = 0
        result = super().step(optimizer, ordinary_step, closure)
        if (not self.passthrough and self.optimizers is not None
                and optimizer is self.optimizers[1] and self.phase == 1):
            self.row["response_armed"] = bool(self._respond)
            self.row["response_uses"] = self._response_uses
        return result

    def _arm_smoothed_critic(self):
        frozen = (self._respond, self._pocket, self._anchor, self._peak_score, self._travel)
        super()._arm_smoothed_critic()
        if self.phase != 0:
            self._respond, self._pocket, self._anchor, self._peak_score, self._travel = frozen
            return
        self._respond = False
        self._pocket = None
        self._anchor = None
        self._peak_score = None
        self._travel = None
        if not self._smooth_on or self._real_batch is None:
            return
        local = self._local or {}
        critic = local.get("critic")
        generator = local.get("generator")
        prior = local.get("prior")
        module = critic
        while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
            module = module.model
        clean = getattr(generator, "model", generator)
        points = clean(prior.z).detach()
        real = self._real_batch.detach()
        if real.ndim != 2 or real.shape[-1] != 2 or real.shape[0] == 0:
            return
        respond, pocket, anchor, travel, peak_score = idle_gate(
            self._sharp, module, real, points, self._smooth_width)
        self._peak_score = peak_score
        self._travel = travel
        self._pocket = pocket
        self._anchor = anchor
        self._respond = respond

    def note_forward(self, x):
        if self.phase == 0 and not self._smooth_on and self._phase_forwards == 0:
            if x.ndim == 2 and x.shape[-1] == 2 and not x.requires_grad:
                self._real_batch = x.detach()
        if self.phase == 0 and not self._smooth_on:
            self._phase_forwards += 1

    def receipt(self):
        value = super().receipt()
        armed = [bool(row.get("response_armed", False)) for row in self.records]
        uses = [row.get("response_uses", 0) for row in self.records]
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     shared_gate_eligible=False, idle_critic_response=True,
                     critic_margin=CRITIC_MARGIN, pocket_gap=POCKET_GAP,
                     near_radius=NEAR_RADIUS,
                     response_steps_armed=int(sum(armed)),
                     response_uses_total=int(sum(uses)))
        return value


@contextmanager
def pr84_idle_critic_response(*, task="mode_hold", start_step=0):
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
    recorder = IdleCriticRecorder(start_step=start_step)
    recorder._sharp = SimpleMLPDiscriminator.forward
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
        recorder.note_forward(x)
        if (recorder._smooth_on and recorder.enabled and not recorder.passthrough
                and x.ndim >= 2 and x.shape[-1] == 2 and recorder._smooth_width > 0):
            if recorder._respond and recorder._pocket is not None:
                score, use = response_score(
                    original_forward, self, x, recorder._smooth_width,
                    recorder._pocket, recorder._anchor, recorder._peak_score,
                    recorder._travel)
                if recorder.phase == 1:
                    recorder._response_uses += int(use.detach().sum())
                return score
            return stencil_mean(original_forward, self, x, recorder._smooth_width)
        return original_forward(self, x)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<pr84-idle-critic-{task}>", "exec"), module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(
            torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(SimpleMLPDiscriminator, "forward", field_forward))
        yield recorder, source
