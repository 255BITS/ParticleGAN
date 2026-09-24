"""Diagnostic G-proposal controller based on current paired D confidence.

The ordinary Adam moments and prior proposal are retained. Only the generator
network proposal is interpolated by a positive, time-independent function of
the training batch's mean sigmoid(real-fake). This is an experimental dynamics
rule, not a convergence claim or an additional loss regularizer.
"""
from contextlib import ExitStack, contextmanager
import math
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch
from particlegan.gan_loss import GANLoss
from benchmarks import learned_lr_evaluation as bridge
from particlegan.training import GANTrainer


class ConfidenceDynamics:
    def __init__(self, threshold=1.0, floor=.02, observe_only=False):
        if not 0 < threshold <= 1 or not 0 < floor <= 1:
            raise ValueError("threshold and positive floor must be in (0,1]")
        self.threshold, self.floor = threshold, floor
        self.observe_only = observe_only
        self.pending = None
        self.receipt = dict(policy="confidence_generator_proposal_v1",
            shared_gate_eligible=False, threshold=threshold, floor=floor,
            observe_only=observe_only, updates=[],
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def observe(self, fake_logits, real_logits):
        if real_logits is None:
            raise ValueError("paired relativistic logits are required")
        with torch.no_grad():
            gap = real_logits - fake_logits
            advantage = float(2 * gap.sigmoid().mean() - 1)
            if not math.isfinite(advantage):
                raise ValueError("nonfinite discriminator confidence")
            self.pending = dict(advantage=advantage, mean_gap=float(gap.mean()),
                                gap_std=float(gap.std(unbiased=False)))

    def step(self, optimizer, original_step, closure=None):
        if closure is not None or self.pending is None:
            raise RuntimeError("G update requires exactly one observed loss")
        parameters = [p for group in optimizer.param_groups
                      if not group.get("_comparison_prior", False)
                      for p in group["params"]]
        if len(parameters) == sum(len(g["params"]) for g in optimizer.param_groups):
            raise RuntimeError("explicit separate learned-prior group required")
        scale = (1.0 if self.observe_only else
                 max(self.floor, min(1.0, self.pending["advantage"] / self.threshold)))
        before = [p.detach().clone() for p in parameters]
        result = original_step(optimizer)
        with torch.no_grad():
            if scale != 1.0:
                for p, base in zip(parameters, before):
                    p.copy_(torch.lerp(base, p, scale))
        self.receipt["updates"].append(dict(self.pending, scale=scale,
            nominal_rates=[float(g["lr"]) for g in optimizer.param_groups]))
        self.pending = None
        return result


@contextmanager
def warm_confidence(state, threshold=1., floor=.02, observe_only=False):
    controller = ConfidenceDynamics(threshold, floor, observe_only)
    original_loss = GANLoss.g_loss
    original_step = state["base_adam_step"]
    def loss(gan, fake_logits, real_logits=None):
        if gan.mode != "rp":
            raise ValueError("scratch confidence rule supports RpGAN only")
        controller.observe(fake_logits, real_logits)
        return original_loss(gan, fake_logits, real_logits)
    def step(optimizer, closure=None):
        if optimizer is state["opt_g"]:
            return controller.step(optimizer, original_step, closure)
        return original_step(optimizer, closure=closure)
    state["set_step_delegate"](step)
    try:
        with patch.object(GANLoss, "g_loss", loss):
            yield controller.receipt
    finally:
        state["set_step_delegate"](original_step)


@contextmanager
def confidence_dynamics(threshold=1., floor=.02, observe_only=False):
    controller = ConfidenceDynamics(threshold, floor, observe_only)
    roles = {}
    original_loss, original_step = GANLoss.g_loss, torch.optim.Adam.step
    original_role, original_init = bridge.optimizer_role, GANTrainer.__init__
    def loss(gan, fake_logits, real_logits=None):
        if gan.mode != "rp":
            raise ValueError("scratch confidence rule supports RpGAN only")
        controller.observe(fake_logits, real_logits)
        return original_loss(gan, fake_logits, real_logits)
    def role(optimizer, local):
        value = original_role(optimizer, local)
        roles[id(optimizer)] = value
        return value
    def initialize(trainer, *args, **kwargs):
        original_init(trainer, *args, **kwargs)
        roles[id(trainer.opt_g)], roles[id(trainer.opt_d)] = "g", "d"
    def step(optimizer, closure=None):
        value = roles.get(id(optimizer))
        if value == "g":
            return controller.step(optimizer, original_step, closure)
        if value != "d":
            raise RuntimeError("unknown optimizer role")
        return original_step(optimizer, closure=closure)
    with ExitStack() as stack:
        stack.enter_context(patch.object(GANLoss, "g_loss", loss))
        stack.enter_context(patch.object(bridge, "optimizer_role", role))
        stack.enter_context(patch.object(GANTrainer, "__init__", initialize))
        stack.enter_context(patch.object(torch.optim.Adam, "step", step))
        yield controller.receipt
