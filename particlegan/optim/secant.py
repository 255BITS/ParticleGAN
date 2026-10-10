"""Bounded secant control of actual rounded FullDualNorm proposal lengths.

This is a stochastic game heuristic inspired by Malitsky/Mishchenko's local
smoothness rule, not their convex gradient-descent algorithm or guarantee.
There are no additional forwards, samples, or gradient evaluations.
"""
from copy import deepcopy
import math

import torch

from .dualnorm import NormalizedOptimizer


class SecantOptimizer(NormalizedOptimizer):
    """Scale each parameter proposal, or each owned prior row, by [1/16, 1]."""

    FLOOR = 1. / 16.
    FIELDS = {"secant_x", "secant_g", "secant_eta", "secant_theta",
              "secant_valid", "secant_clock"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.family != "dualnorm" or self.momentum:
            raise ValueError("secant requires zero-momentum full DualNorm")
        self.secant_stats = dict(steps=0, observations=0, history_uses=0,
            damped=0, floor_hits=0, zero_proposals=0, scale_sum=0., minimum_scale=1.,
            proposal_length_sum=0., applied_length_sum=0.)
        self.secant_role_stats = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("secant consumes the caller's existing gradient only")
        pending = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()) or not bool(torch.isfinite(parameter).all()):
                    raise ValueError("secant requires finite parameters and gradients")
                rows = None
                if group["algorithm"] == "rownorm":
                    rows = (self.sampled_rows_for(parameter) if group["sampled_rows_required"]
                            else torch.arange(len(parameter), device=parameter.device))
                    if rows is None:
                        raise ValueError("secant prior history requires actual sampled rows")
                pending.append((group, parameter, rows, parameter.detach().clone(), parameter.grad.detach().clone()))
        result = super().step()
        for group, parameter, rows, original, gradient in pending:
            state, rate = self.state[parameter], group["lr"]
            rowwise = rows is not None
            shape = (len(parameter),) if rowwise else ()
            if "secant_x" not in state:
                state.update(secant_x=torch.zeros_like(parameter), secant_g=torch.zeros_like(parameter),
                    secant_eta=parameter.new_zeros(shape), secant_theta=parameter.new_ones(shape),
                    secant_valid=torch.zeros(shape, dtype=torch.bool, device=parameter.device),
                    secant_clock=torch.zeros(shape, dtype=torch.long, device=parameter.device))
            old = original[rows] if rowwise else original
            grad = gradient[rows] if rowwise else gradient
            proposed = parameter[rows].clone() if rowwise else parameter.detach().clone()
            delta = proposed - old
            previous_x = state["secant_x"][rows] if rowwise else state["secant_x"]
            previous_g = state["secant_g"][rows] if rowwise else state["secant_g"]
            previous_eta = state["secant_eta"][rows] if rowwise else state["secant_eta"]
            theta = state["secant_theta"][rows] if rowwise else state["secant_theta"]
            valid = state["secant_valid"][rows] if rowwise else state["secant_valid"].clone()
            norm = (lambda value: value.norm(dim=1)) if rowwise else (lambda value: value.norm())
            s, y, g, proposal = norm(old - previous_x), norm(grad - previous_g), norm(grad), norm(delta)
            # alpha_curvature = ||s|| ||g|| / (2 ||y|| ||actual proposal||).
            # Zero y/proposal gives no restriction. Growth uses effective rate
            # history, independent of changes to the caller's nominal schedule.
            denominator = 2 * y * proposal
            curvature = torch.where(denominator > 0, s * g / denominator.clamp_min(torch.finfo(parameter.dtype).tiny),
                                    torch.full_like(s, float("inf")))
            growth = ((1 + theta).sqrt() * previous_eta / rate if rate > 0
                      else torch.ones_like(s))
            bound = torch.minimum(torch.ones_like(s), torch.minimum(curvature, growth))
            scale = torch.where(valid, bound.clamp(min=self.FLOOR, max=1.), torch.ones_like(s))
            # Preserve a full rounded base proposal exactly. At smaller scale
            # the same displacement ray is used, with parameter dtype rounding.
            effective = scale.unsqueeze(1) if rowwise else scale
            accepted = torch.where(effective == 1, proposed, old + effective * delta)
            if rowwise:
                parameter[rows] = accepted
            else:
                parameter.copy_(accepted)
            eta = scale * rate
            new_theta = torch.where(valid & (previous_eta > 0), eta / previous_eta.clamp_min(torch.finfo(parameter.dtype).tiny),
                                    torch.ones_like(eta))
            for key, value in (("secant_x", old), ("secant_g", grad), ("secant_eta", eta),
                               ("secant_theta", new_theta), ("secant_valid", torch.ones_like(valid))):
                if rowwise:
                    state[key][rows] = value
                else:
                    state[key].copy_(value)
            if rowwise:
                state["secant_clock"][rows] += 1
            else:
                state["secant_clock"] += 1
            values = dict(observations=scale.numel(), history_uses=int(valid.sum()),
                damped=int((scale < 1).sum()), floor_hits=int((scale == self.FLOOR).sum()),
                scale_sum=float(scale.sum()), zero_proposals=int((proposal == 0).sum()),
                proposal_length_sum=float(proposal.sum()), applied_length_sum=float(norm(accepted - old).sum()))
            role_stats = self.secant_role_stats.setdefault(group["role"],
                {key: (1. if key == "minimum_scale" else type(value)(0)) for key, value in self.secant_stats.items()})
            for stats in (self.secant_stats, role_stats):
                for key, value in values.items():
                    stats[key] += value
                if scale.numel():
                    stats["minimum_scale"] = min(stats["minimum_scale"], float(scale.min()))
        self.secant_stats["steps"] += 1
        for role in {group["role"] for group, *_ in pending}:
            self.secant_role_stats[role]["steps"] += 1
        return result

    def state_dict(self):
        saved = super().state_dict()
        saved["secant"] = dict(schema=1, mode="bounded", floor=self.FLOOR, stats=deepcopy(self.secant_stats), roles=deepcopy(self.secant_role_stats))
        return saved

    def validate_state_dict(self, saved):
        base = dict(saved)
        meta = base.pop("secant", None)
        if (not isinstance(meta, dict) or set(meta) != {"schema", "mode", "floor", "stats", "roles"}
                or meta["schema"] != 1 or meta["mode"] != "bounded" or meta["floor"] != self.FLOOR
                or not isinstance(meta["stats"], dict) or set(meta["stats"]) != set(self.secant_stats)):
            raise ValueError("invalid secant checkpoint metadata")
        if (not isinstance(meta["roles"], dict)
                or not set(meta["roles"]) <= {group["role"] for group in self.param_groups}):
            raise ValueError("invalid secant role counters")
        for stats in (meta["stats"], *meta["roles"].values()):
            if not isinstance(stats, dict) or set(stats) != set(self.secant_stats):
                raise ValueError("invalid secant role statistic fields")
            for key, value in stats.items():
                if type(self.secant_stats[key]) is int:
                    if type(value) is not int or value < 0:
                        raise ValueError("invalid secant counter")
                elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid secant statistic")
            if not self.FLOOR <= stats["minimum_scale"] <= 1:
                raise ValueError("invalid secant minimum scale")
        base["state"] = {key: {name: value for name, value in state.items() if name not in self.FIELDS}
                         for key, state in saved["state"].items()}
        super().validate_state_dict(base)
        for actual, group in zip(self.param_groups, saved["param_groups"]):
            for identifier, parameter in zip(group["params"], actual["params"]):
                history = saved["state"].get(identifier, {})
                own = set(history) & self.FIELDS
                if not history:
                    continue
                if own != self.FIELDS:
                    raise ValueError("missing secant parameter history")
                rowwise = actual["algorithm"] == "rownorm"
                scalar_shape = (len(parameter),) if rowwise else ()
                for key in self.FIELDS:
                    value = history[key]
                    shape = parameter.shape if key in {"secant_x", "secant_g"} else scalar_shape
                    dtype = torch.bool if key == "secant_valid" else (torch.long if key == "secant_clock" else parameter.dtype)
                    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or not bool(torch.isfinite(value).all()):
                        raise ValueError(f"invalid {key} history")
                    if key in {"secant_eta", "secant_theta", "secant_clock"} and bool((value < 0).any()):
                        raise ValueError(f"negative {key} history")
                if not torch.equal(history["secant_valid"], history["secant_clock"] > 0):
                    raise ValueError("secant row clock/validity mismatch")

    def load_state_dict(self, saved):
        self.validate_state_dict(saved)
        super().load_state_dict(saved)
        # Optimizer's generic caster turns boolean/integer state into parameter
        # dtype. Restore exact owned-row validity and visit clocks explicitly.
        for actual, group in zip(self.param_groups, saved["param_groups"]):
            for identifier, parameter in zip(group["params"], actual["params"]):
                for key in ("secant_valid", "secant_clock"):
                    if key in saved["state"].get(identifier, {}):
                        self.state[parameter][key] = saved["state"][identifier][key].to(device=parameter.device).clone()
        self.secant_stats = deepcopy(saved["secant"]["stats"])
        self.secant_role_stats = deepcopy(saved["secant"]["roles"])
