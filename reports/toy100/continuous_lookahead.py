"""Joint Lookahead-Minmax on live Adam iterates, with fixed coefficients.

Algorithm 1 of https://arxiv.org/html/2006.14567v3. Both players (including
the learned prior) are interpolated together after k complete D/G rounds.
Adam moments are retained. Inner iterates remain live and must be evaluated;
only scoring interpolation boundaries would conceal possible excursions.
"""
from contextlib import ExitStack, contextmanager
import math
from unittest.mock import patch

import torch

from benchmarks import learned_lr_evaluation as bridge
from particlegan.training import GANTrainer


class JointLookahead:
    def __init__(self, k, alpha):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0,1]")
        self.k, self.alpha = k, float(alpha)
        self.optimizers = {}
        self.roles = {}
        self.counts = {"g": 0, "d": 0}
        self.slow = {}
        self.receipt = dict(policy="joint_lookahead_minmax_v1", k=k, alpha=self.alpha,
                            shared_gate_eligible=False, additional_gradient_evaluations=0,
                            updates=[], synchronizations=[])

    def register(self, optimizer, role):
        if role not in ("g", "d"):
            raise ValueError("only one G and one D optimizer are supported")
        if role in self.optimizers and self.optimizers[role] is not optimizer:
            raise RuntimeError("multiple optimizers for one player are unsupported")
        if id(optimizer) in self.roles and self.roles[id(optimizer)] != role:
            raise RuntimeError("optimizer changed role")
        self.optimizers[role] = optimizer
        self.roles[id(optimizer)] = role

    def step(self, optimizer, original_step, closure=None):
        if closure is not None:
            raise ValueError("explicit host gradients are required")
        role = self.roles.get(id(optimizer))
        if role is None:
            raise RuntimeError("optimizer player role is unknown")
        if ((role == "d" and self.counts["d"] != self.counts["g"])
                or (role == "g" and self.counts["d"] != self.counts["g"] + 1)):
            raise RuntimeError("Lookahead requires one D then one G update per round")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter not in self.slow:
                    self.slow[parameter] = parameter.detach().clone()
        result = original_step(optimizer)
        self.counts[role] += 1
        self.receipt["updates"].append(dict(role=role, step=self.counts[role],
            rates=[float(group["lr"]) for group in optimizer.param_groups]))
        if role == "g" and self.counts["g"] % self.k == 0:
            self.synchronize()
        return result

    @torch.no_grad()
    def synchronize(self):
        if set(self.optimizers) != {"g", "d"} or self.counts["g"] != self.counts["d"]:
            raise RuntimeError("cannot interpolate an incomplete game round")
        changes = {}
        for role, optimizer in self.optimizers.items():
            square_sum, count = 0.0, 0
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    slow = self.slow[parameter]
                    if self.alpha == 1:
                        slow.copy_(parameter)
                        continue
                    before = parameter.detach().clone()
                    slow.lerp_(parameter, self.alpha)
                    parameter.copy_(slow)
                    square_sum += float((parameter - before).double().square().sum())
                    count += parameter.numel()
            changes[role] = math.sqrt(square_sum / count) if count else 0.0
        self.receipt["synchronizations"].append(dict(step=self.counts["g"],
                                                     correction_rms=changes))


@contextmanager
def lookahead(k=5, alpha=0.5):
    """Scope the fixed joint update to one complete host episode."""
    controller = JointLookahead(k, alpha)
    original_role = bridge.optimizer_role
    original_init = GANTrainer.__init__
    original_step = torch.optim.Adam.step

    def role(optimizer, local_variables):
        result = original_role(optimizer, local_variables)
        controller.register(optimizer, result)
        return result

    def initialize(trainer, *args, **kwargs):
        original_init(trainer, *args, **kwargs)
        controller.register(trainer.opt_g, "g")
        controller.register(trainer.opt_d, "d")

    def step(optimizer, closure=None):
        return controller.step(optimizer, original_step, closure)

    with ExitStack() as stack:
        stack.enter_context(patch.object(bridge, "optimizer_role", role))
        stack.enter_context(patch.object(GANTrainer, "__init__", initialize))
        stack.enter_context(patch.object(torch.optim.Adam, "step", step))
        yield controller.receipt
