"""Scoped, auditable Adam epsilon probe for the frozen toy hosts.

This research adapter changes only Adam's fixed denominator epsilon.  It uses
the existing optimizer steps, gradients, moments, and learning-rate policy.
The receipt is scratch evidence and is not eligible for the production common
gate until that gate explicitly understands and verifies the new policy.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math
from unittest.mock import patch

import torch

from benchmarks import learned_lr_evaluation as bridge
from particlegan.particle_prior import ParticlePrior
from particlegan.training import GANTrainer


def _positive_eps(value, name: str) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


class _EpsilonObserver:
    def __init__(self, options: dict):
        required = {"network_eps", "prior_eps"}
        allowed = required | {"g_eps", "d_eps"}
        if (not isinstance(options, dict) or not required <= set(options)
                or not set(options) <= allowed):
            raise ValueError("declare network_eps and prior_eps; g_eps and d_eps are optional")
        self.network_eps = _positive_eps(options["network_eps"], "network_eps")
        self.prior_eps = _positive_eps(options["prior_eps"], "prior_eps")
        self.g_eps = _positive_eps(options.get("g_eps", self.network_eps), "g_eps")
        self.d_eps = _positive_eps(options.get("d_eps", self.network_eps), "d_eps")
        self.prior_ids: set[int] = set()
        self.optimizer_roles: dict[int, str] = {}
        self.optimizer_counts: dict[int, int] = {}
        self.receipt = {
            "policy": "fixed_adam_denominator_floor_v1",
            "options": {name: float(value) for name, value in options.items()},
            "effective_eps": {"g": self.g_eps, "d": self.d_eps,
                              "prior": self.prior_eps},
            "shared_gate_eligible": False,
            "additional_gradient_evaluations": 0,
            "diagnostic_interval": 50,
            "updates": [],
        }

    def register(self, optimizer, role: str):
        if role not in ("g", "d"):
            raise ValueError("optimizer role must be g or d")
        identity = id(optimizer)
        previous = self.optimizer_roles.setdefault(identity, role)
        if previous != role:
            raise RuntimeError("optimizer changed player role")

    def _group_role(self, group: dict, optimizer_role: str) -> str:
        parameters = group["params"]
        if not parameters:
            raise RuntimeError("Adam group has no parameters")
        if optimizer_role == "d":
            if group.get("_comparison_prior", False):
                raise RuntimeError("critic optimizer owns prior parameters")
            return "d"
        marker = group.get("_comparison_prior")
        owned = [id(parameter) in self.prior_ids for parameter in parameters]
        if marker is True:
            if any(owned) and not all(owned):
                raise RuntimeError("Adam group mixes prior and network parameters")
            return "prior"
        if marker is False:
            if any(owned):
                raise RuntimeError("network group contains tracked prior parameters")
            return "g"
        if all(owned):
            return "prior"
        if any(owned):
            raise RuntimeError("Adam group mixes prior and network parameters")
        return "g"

    @staticmethod
    def _rms(values, count: int) -> float:
        return math.sqrt(sum(float(value.detach().double().square().sum())
                             for value in values) / count)

    def _eps(self, kind: str) -> float:
        return {"g": self.g_eps, "d": self.d_eps,
                "prior": self.prior_eps}[kind]

    def step(self, optimizer: torch.optim.Adam, original_step, closure=None):
        if closure is not None:
            raise ValueError("epsilon probe requires explicit frozen-host gradients")
        identity = id(optimizer)
        role = self.optimizer_roles.get(identity)
        if role is None:
            raise RuntimeError("Adam optimizer has no declared G/D role")
        completed = self.optimizer_counts.get(identity, 0) + 1
        diagnostic = completed == 1 or completed % 50 == 0
        groups = []
        before = {}
        for group in optimizer.param_groups:
            if (group.get("weight_decay", 0) != 0
                    or any(group.get(key, False) for key in
                           ("amsgrad", "maximize", "differentiable", "capturable",
                            "decoupled_weight_decay"))):
                raise ValueError("epsilon probe requires ordinary zero-decay Adam")
            kind = self._group_role(group, role)
            group["eps"] = self._eps(kind)
            parameters = [p for p in group["params"] if p.grad is not None]
            if not parameters:
                raise RuntimeError("Adam group has no applied gradient")
            count = sum(p.numel() for p in parameters)
            row = {"role": kind, "parameters": count,
                   "lr": float(group["lr"]), "eps": float(group["eps"]),
                   "betas": [float(beta) for beta in group["betas"]]}
            if diagnostic:
                row["gradient_rms"] = self._rms((p.grad for p in parameters), count)
                before.update({p: p.detach().clone() for p in parameters})
            groups.append(row)
        result = original_step(optimizer)
        for group, row in zip(optimizer.param_groups, groups):
            # Read the groups after Adam's own pre-step hooks have run.
            row["lr"] = float(group["lr"])
            row["eps"] = float(group["eps"])
            expected = self._eps(row["role"])
            if row["eps"] != expected:
                raise RuntimeError("Adam epsilon changed during the optimizer update")
            if diagnostic:
                parameters = [p for p in group["params"] if p in before]
                row["update_rms"] = self._rms((p.detach() - before[p]
                                               for p in parameters), row["parameters"])
                denominator_min = math.inf
                for parameter in parameters:
                    state = optimizer.state[parameter]
                    beta2 = group["betas"][1]
                    second = state["exp_avg_sq"] / (1 - beta2 ** int(state["step"]))
                    denominator_min = min(denominator_min,
                                          float(second.sqrt().min()) + row["eps"])
                row["denominator_min"] = denominator_min
        self.optimizer_counts[identity] = completed
        self.receipt["updates"].append({"optimizer_role": role,
                                         "optimizer_step": completed,
                                         "groups": groups})
        return result


@contextmanager
def candidate_update(options: dict):
    """Apply a fixed Adam epsilon by network/prior ownership for one episode.

    Enter this context around the entire native or transfer episode, including
    optimizer construction.  Legacy hosts identify direct particle groups in
    their existing schedule bridge; native/vector/image trainers register both
    optimizers when constructed.  The yielded dict is JSON serializable after
    updates and retains every actual group rate and epsilon.
    """
    observer = _EpsilonObserver(options)
    original_prior_init = ParticlePrior.__init__
    original_trainer_init = GANTrainer.__init__
    original_role = bridge.optimizer_role
    original_step = torch.optim.Adam.step

    def prior_init(prior, *args, **kwargs):
        original_prior_init(prior, *args, **kwargs)
        observer.prior_ids.update(id(p) for p in prior.parameters())

    def trainer_init(trainer, *args, **kwargs):
        original_trainer_init(trainer, *args, **kwargs)
        observer.register(trainer.opt_g, "g")
        observer.register(trainer.opt_d, "d")

    def optimizer_role(optimizer, local_variables):
        role = original_role(optimizer, local_variables)
        observer.register(optimizer, role)
        return role

    def adam_step(optimizer, closure=None):
        return observer.step(optimizer, original_step, closure)

    with ExitStack() as stack:
        stack.enter_context(patch.object(ParticlePrior, "__init__", prior_init))
        stack.enter_context(patch.object(GANTrainer, "__init__", trainer_init))
        stack.enter_context(patch.object(bridge, "optimizer_role", optimizer_role))
        stack.enter_context(patch.object(torch.optim.Adam, "step", adam_step))
        yield observer.receipt
