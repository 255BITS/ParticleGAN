"""Scratch PR #84 smoothed-critic candidate on the PR #82 alternating adapter.

The D and G/prior Adam base rates stay constant. D uses the sharp critic;
only G's two-dimensional critic evaluation uses a five-point spatial stencil.
The PR #82 same-sample own-curvature bounds remain fixed at .25 for G and 3
for D. This module has no target-center, elapsed-time, or rest-gate controller.
It is limited to the deterministic mode-hold and trajectory diagnostic hosts;
its current evidence does not qualify it for the shared production gate.
"""

from contextlib import ExitStack, contextmanager
import ast
import math
from unittest.mock import patch

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from particlegan.gan_loss import GANLoss
from reports.toy100.alternating_curvature_scratch import BothBoundRecorder
from reports.toy100.extra_adam_scratch import HOSTS, sha, transformed_function


METHOD = "alternating_adam_with_spatially_smoothed_g_critic"
G_CURVATURE_BOUND = .25
D_CURVATURE_BOUND = 3.
SMOOTH_WIDTH_CAP = .15


class SmoothedBothBoundRecorder(BothBoundRecorder):
    """Add only PR #84's G-side spatial stencil to PR #82's D-then-G update."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step, curvature_bound=G_CURVATURE_BOUND,
                         d_curvature_bound=D_CURVATURE_BOUND)
        self._smooth_on = False
        self._smooth_width = 0.
        self._local = None

    def phases(self, step, opt_d, opt_g, local):
        self._local = local
        for phase in super().phases(step, opt_d, opt_g, local):
            # As in the original PR #84 source, each D replay begins sharp.
            self._smooth_on = False
            yield phase

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        result = super().step(optimizer, ordinary_step, closure)
        if not self.passthrough and optimizer is self.optimizers[0]:
            # G sees the critic after this D phase has materialized.
            self._arm_smoothed_critic()
        return result

    @torch.no_grad()
    def _arm_smoothed_critic(self):
        self._smooth_on = False
        self._smooth_width = 0.
        local = self._local or {}
        if local.get("slow") is not None:
            return
        critic = local.get("critic")
        generator = local.get("generator")
        prior = local.get("prior")
        if critic is None or generator is None or prior is None or not hasattr(prior, "z"):
            return
        module = critic
        while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
            module = module.model
        if not isinstance(module, SimpleMLPDiscriminator):
            return
        clean = getattr(generator, "model", generator)
        points = clean(prior.z).detach()
        if points.ndim != 2 or points.shape[-1] != 2:
            return
        eps = 1e-3
        acc = 0.
        for dim in range(points.shape[-1]):
            shift = torch.zeros_like(points)
            shift[:, dim] = eps
            acc = acc + ((module(points + shift) - module(points - shift)) / (2 * eps)).square()
        sharp = float(acc.mean().sqrt())
        if not math.isfinite(sharp) or sharp <= 1e-6:
            return
        width = min(SMOOTH_WIDTH_CAP, .5 / sharp)
        self._smooth_width = width
        self._smooth_on = True
        self.row["critic_sharpness"] = sharp
        self.row["critic_width"] = width

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     shared_gate_eligible=False, smooth_critic=True,
                     smooth_width_cap=SMOOTH_WIDTH_CAP)
        return value


@contextmanager
def pr84_smoothed_candidate(*, task="mode_hold", start_step=0, prepare_tree=None):
    """Yield ``(recorder, generated_host_source)`` for one declared toy host.

    ``prepare_tree`` may mutate the transformed function after the phase
    adapter is installed. The default leaves the PR84 / stall-reach host
    unchanged.
    """
    from benchmarks.locked_shared import mode_hold, trajectory

    module = {"mode_hold": mode_hold, "trajectory": trajectory}[task]
    tree, _, original_sha = transformed_function(module, task)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "phases"]
    if len(calls) != 1:
        raise RuntimeError("expected one phase iterator")
    calls[0].args.append(ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))
    if prepare_tree is not None:
        prepare_tree(tree)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    recorder = SmoothedBothBoundRecorder(start_step=start_step)
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
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<pr84-smoothed-{task}>", "exec"), module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(
            torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(SimpleMLPDiscriminator, "forward", smoothed_forward))
        yield recorder, source
