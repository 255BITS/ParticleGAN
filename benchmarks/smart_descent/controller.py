"""Causal gradient observations drive LR and next-loss regularization actions."""
from contextlib import contextmanager
import math
import time
from unittest.mock import patch

import torch

from particlegan import GradientPenalty, ParticleRegularizer, learning_rate_scale

FEATURES = ("bias", "log_gradient_ratio", "gradient_alignment",
            "log_gradient_innovation", "opponent_log_gradient_ratio")


class GradientFeedback:
    """Two roles × two actions × five generic features; no Adam-state input.

    LR actions use the current backward pass. Regularization actions are held
    for the following loss computation, so they cannot alter past gradients.
    This serial research bridge supports dense gradients with fixed layouts.
    """
    def __init__(self, policy, total_steps, *, ablation="none"):
        self.weights = torch.as_tensor(policy["weights"], dtype=torch.float64)
        if self.weights.shape != (2, 2, len(FEATURES)) or not torch.isfinite(self.weights).all():
            raise ValueError("weights must be finite [2, 2, 5]")
        if policy.get("features", list(FEATURES)) != list(FEATURES):
            raise ValueError("feature schema mismatch")
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("positive integer budget required")
        if ablation not in ("none", "bias_only", "lr_only", "reg_only"):
            raise ValueError("unknown ablation")
        self.total_steps, self.ablation = total_steps, ablation
        self.interval = policy.get("interval", 5)
        if type(self.interval) is not int or self.interval < 1:
            raise ValueError("positive observation interval required")
        self.schedule = policy.get("schedule", "cosine")
        if self.schedule not in ("constant", "cosine"):
            raise ValueError("invalid base schedule")
        self.states, self.latest, self.trace = {}, {}, []
        self.controller_seconds = 0.

    def regularization_scale(self, role):
        return math.exp(self.latest.get(role, {}).get("reg_log", 0.))

    @torch.no_grad()
    def step(self, optimizer, completed_updates, *, role):
        started = time.perf_counter()
        if role not in ("g", "d"):
            raise ValueError("role must be g or d")
        if type(completed_updates) is not int or not 0 <= completed_updates < self.total_steps:
            raise ValueError("update outside budget")
        state = self.states.get(optimizer)
        if state is None:
            rates = [float(group["lr"]) for group in optimizer.param_groups]
            if not all(math.isfinite(rate) and rate > 0 for rate in rates):
                raise ValueError("positive finite initial learning rates required")
            state = dict(rates=rates, role=role, last_step=-1, lr_log=0., reg_log=0.)
            self.states[optimizer] = state
        if role != state["role"] or completed_updates <= state["last_step"]:
            raise ValueError("role changed or non-increasing update")
        state["last_step"] = completed_updates
        if completed_updates % self.interval == 0:
            grads = [p.grad.detach().flatten() for group in optimizer.param_groups
                     for p in group["params"] if p.grad is not None]
            if not grads or any(g.is_sparse for g in grads):
                raise ValueError("dense gradients required after backward")
            grad = torch.cat(grads)
            rms = float(grad.square().mean().sqrt())
            if not math.isfinite(rms):
                raise FloatingPointError("nonfinite gradient")
            if "mean_gradient" not in state:
                state["mean_gradient"] = grad.clone()
                state["initial_rms"] = max(rms, 1e-12)
            mean = state["mean_gradient"]
            if mean.shape != grad.shape:
                raise ValueError("gradient layout changed")
            clip_log = lambda value: max(-2., min(2., math.log(max(value, 1e-12))))
            ratio = clip_log(rms / state["initial_rms"])
            alignment = float(torch.nn.functional.cosine_similarity(grad[None], mean[None], eps=1e-12))
            innovation = clip_log(rms / max(float(mean.square().mean().sqrt()), 1e-12))
            opponent = self.latest.get("d" if role == "g" else "g", {}).get("gradient_ratio", 0.)
            features = [1., ratio, alignment, innovation, opponent]
            if self.ablation == "bias_only":
                features[1:] = [0.] * 4
            values = self.weights[0 if role == "g" else 1] @ torch.tensor(features, dtype=torch.float64)
            lr_target = max(math.log(.25), min(math.log(2.), float(values[0])))
            reg_target = max(math.log(.5), min(math.log(2.), float(values[1])))
            if self.ablation == "lr_only":
                reg_target = 0.
            if self.ablation == "reg_only":
                lr_target = 0.
            state["lr_log"] = .5 * state["lr_log"] + .5 * lr_target
            state["reg_log"] = .5 * state["reg_log"] + .5 * reg_target
            state["gradient_ratio"] = ratio
            mean.mul_(.9).add_(grad, alpha=.1)
            self.latest[role] = state
            self.trace.append(dict(step=completed_updates, role=role, features=features,
                                   lr_multiplier=math.exp(state["lr_log"]),
                                   next_regularization_multiplier=math.exp(state["reg_log"])))
        scale = learning_rate_scale(completed_updates, self.total_steps, .6, .05) if self.schedule == "cosine" else 1.
        for group, rate in zip(optimizer.param_groups, state["rates"]):
            group["lr"] = rate * scale * math.exp(state["lr_log"])
        self.controller_seconds += time.perf_counter() - started


@contextmanager
def control_regularization(controller):
    """Apply the last causal action while constructing differentiable losses.

    Penalty statistics still describe the original penalty. The action trace
    separately records its multiplier. Only ParticleRegularizer is scaled on G;
    arbitrary reconstruction/content loss coefficients are not controlled.
    """
    original_penalty = GradientPenalty.penalty
    original_prior = ParticleRegularizer.forward

    def penalty(instance, *args, **kwargs):
        value, stats = original_penalty(instance, *args, **kwargs)
        return value * controller.regularization_scale("d"), stats

    def prior(instance, *args, **kwargs):
        return original_prior(instance, *args, **kwargs) * controller.regularization_scale("g")

    with patch.object(GradientPenalty, "penalty", penalty), patch.object(ParticleRegularizer, "forward", prior):
        yield
