"""A small causal optimizer-feedback policy learned by outer-loop search.

Only optimizer tensors and normalized update progress enter the policy. There
are no task identities, evaluation metrics, or formulation names in this API.
"""
from __future__ import annotations

import math
import time

import torch

FEATURES = ("bias", "progress", "log_gradient_ratio", "gradient_alignment",
            "log_adam_direction_ratio", "log_parameter_ratio")


class OptimizerLRAdapter:
    """Bounded learned log-LR equation, one coefficient row for G and D.

    Call immediately before optimizer.step(), after backward(). All parameter
    groups retain their original LR ratios. Optimizers must be ordinary Adam
    without weight decay or AMSGrad. This research version does not checkpoint
    its online state and is deliberately outside the supported trainer API.
    """
    def __init__(self, policy_dict, total_steps, *, ablation="none", interval=20):
        self.weights = torch.as_tensor(policy_dict["weights"], dtype=torch.float64)
        if self.weights.shape != (2, len(FEATURES)) or not torch.isfinite(self.weights).all():
            raise ValueError("policy weights must be finite, shape [2, 6]")
        if policy_dict.get("features", list(FEATURES)) != list(FEATURES):
            raise ValueError("policy feature schema mismatch")
        if type(total_steps) is not int or total_steps <= 0 or type(interval) is not int or interval <= 0:
            raise ValueError("total_steps and interval must be positive integers")
        if ablation not in ("none", "time_only"):
            raise ValueError("unknown ablation")
        self.total_steps, self.interval, self.ablation = total_steps, interval, ablation
        self.state, self.trace = {}, []
        self.controller_seconds = 0.

    @staticmethod
    @torch.no_grad()
    def _attributes(optimizer):
        gradients, parameters, directions = [], [], []
        for group in optimizer.param_groups:
            if group.get("amsgrad", False) or group.get("weight_decay", 0) or group.get("maximize", False):
                raise ValueError("research adapter supports Adam without decay, AMSGrad, or maximize")
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise ValueError("sparse gradients are unsupported")
                state = optimizer.state[p]
                update = int(state.get("step", 0)) + 1
                m = state.get("exp_avg", torch.zeros_like(p)) * beta1 + grad * (1 - beta1)
                v = state.get("exp_avg_sq", torch.zeros_like(p)) * beta2 + grad.square() * (1 - beta2)
                direction = (m / (1 - beta1 ** update)) / ((v / (1 - beta2 ** update)).sqrt() + group["eps"])
                gradients.append(grad.flatten())
                parameters.append(p.detach().flatten())
                directions.append(direction.flatten())
        if not gradients:
            raise ValueError("adapter needs gradients from backward()")
        gradient, parameter, direction = map(torch.cat, (gradients, parameters, directions))
        # Device-to-host scalar conversions are intentionally infrequent.
        values = [float(x.square().mean().sqrt()) for x in (gradient, parameter, direction)]
        if not all(math.isfinite(x) for x in values):
            raise FloatingPointError("nonfinite optimizer attributes")
        return gradient, values

    @torch.no_grad()
    def step(self, optimizer, completed_updates, *, role):
        started = time.perf_counter()
        if role not in ("g", "d"):
            raise ValueError("optimizer role must be g or d")
        if type(completed_updates) is not int or not 0 <= completed_updates < self.total_steps:
            raise ValueError("completed_updates outside policy budget")
        if not isinstance(optimizer, torch.optim.Adam):
            raise ValueError("research adapter expects torch.optim.Adam")
        if optimizer not in self.state:
            rates = [group["lr"] for group in optimizer.param_groups]
            if not all(math.isfinite(x) and x > 0 for x in rates):
                raise ValueError("initial learning rates must be finite and positive")
            self.state[optimizer] = dict(rates=rates, log_scale=0., role=role, last_step=-1)
        state = self.state[optimizer]
        if role != state["role"] or completed_updates <= state["last_step"]:
            raise ValueError("optimizer role changed or update was repeated")
        state["last_step"] = completed_updates
        if completed_updates % self.interval == 0:
            gradient, (grad_rms, param_rms, direction_rms) = self._attributes(optimizer)
            if "baseline" not in state:
                state["baseline"] = [max(x, 1e-12) for x in (grad_rms, param_rms, direction_rms)]
            base_grad, base_param, base_direction = state["baseline"]
            previous = state.get("previous_gradient")
            alignment = 0. if previous is None else float(torch.nn.functional.cosine_similarity(
                gradient.unsqueeze(0), previous.unsqueeze(0), eps=1e-12))
            state["previous_gradient"] = gradient.clone()
            ratio = lambda value, reference: max(-2., min(2., math.log(max(value, 1e-12) / reference)))
            features = [1., completed_updates / self.total_steps,
                        ratio(grad_rms, base_grad), alignment,
                        ratio(direction_rms, base_direction), ratio(param_rms, base_param)]
            if self.ablation == "time_only":
                features[2:] = [0.] * (len(features) - 2)
            row = self.weights[0 if role == "g" else 1]
            target = float(row @ torch.tensor(features, dtype=torch.float64))
            target = max(math.log(.05), min(math.log(2.), target))
            state["log_scale"] = .5 * state["log_scale"] + .5 * target
            self.trace.append(dict(step=completed_updates, role=role, features=features,
                                   multiplier=math.exp(state["log_scale"])))
        multiplier = math.exp(state["log_scale"])
        for group, rate in zip(optimizer.param_groups, state["rates"]):
            group["lr"] = rate * multiplier
        self.controller_seconds += time.perf_counter() - started
        return multiplier
