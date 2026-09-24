"""State-based G loss-budget experiment, without an elapsed-time schedule.

Use the positive Rp logistic loss excess over log(2) as a budget for the
linearized improvement of an Adam proposal. This is inspired by preconditioned
Polyak steps, but log(2) is an equilibrium reference, NOT a sample-loss lower
bound for a fixed, potentially stale critic. No convergence claim is made.
"""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from reports.toy100 import confidence_dynamics_scratch as confidence


class LossBudget:
    def __init__(self, scope="network", observe_only=False):
        if scope not in ("network", "joint"):
            raise ValueError("scope must be network or joint")
        self.scope, self.observe_only, self.pending = scope, observe_only, None
        self.receipt = dict(policy="equilibrium_reference_loss_budget_v1",
            shared_gate_eligible=False, scope=scope, observe_only=observe_only,
            reference=math.log(2), additional_gradient_evaluations=0,
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), updates=[])

    def observe(self, fake_logits, real_logits):
        if real_logits is None:
            raise ValueError("paired logistic logits required")
        with torch.no_grad():
            loss = float(F.softplus(real_logits - fake_logits).mean())
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite observed loss")
        self.pending = dict(loss=loss, budget=max(0., loss - math.log(2)))

    def step(self, optimizer, original_step, closure=None):
        if closure is not None or self.pending is None:
            raise RuntimeError("requires exactly one observed G loss")
        if not any(g.get("_comparison_prior", False) for g in optimizer.param_groups):
            raise RuntimeError("explicit separate learned-prior group required")
        if any(g["betas"][0] != 0 or g.get("weight_decay", 0) != 0 for g in optimizer.param_groups):
            raise ValueError("requires zero first momentum and weight decay")
        parameters = [p for group in optimizer.param_groups
            if self.scope == "joint" or not group.get("_comparison_prior", False)
            for p in group["params"] if p.grad is not None]
        before = [p.detach().clone() for p in parameters]
        gradients = [p.grad.detach().clone() for p in parameters]
        answer = original_step(optimizer)
        with torch.no_grad():
            predicted = -sum(float((g.double() * (p.double()-base.double())).sum())
                             for p, base, g in zip(parameters, before, gradients))
            if not math.isfinite(predicted) or predicted < -1e-12:
                raise FloatingPointError("invalid proposed descent")
            scale = (1. if self.observe_only else
                     min(1., self.pending["budget"] / predicted) if predicted > 0 else 0.)
            if scale != 1.:
                for p, base in zip(parameters, before):
                    p.copy_(torch.lerp(base, p, scale))
        self.receipt["updates"].append(dict(self.pending, predicted_improvement=predicted,
            scale=scale, nominal_rates=[float(g["lr"]) for g in optimizer.param_groups],
            moment_steps=[sorted({int(optimizer.state[p]["step"]) for p in g["params"]
                                 if p in optimizer.state}) for g in optimizer.param_groups]))
        self.pending = None
        return answer


@contextmanager
def loss_budget(*, scope="network", observe_only=False, state=None):
    controller = LossBudget(scope, observe_only)
    context = confidence.confidence_dynamics if state is None else confidence.warm_confidence
    args = () if state is None else (state,)
    with patch.object(confidence, "ConfidenceDynamics", lambda *a, **kw: controller):
        with context(*args) as receipt:
            yield receipt
