"""Bounded joint output travel along an existing zero-momentum GAN update.

This is a sampled function-space trust bound, not a convergence guarantee.
No targets beyond the consumed training batch, RNG draws or objective changes.
"""
import math
from copy import deepcopy

import torch


class HydraulicTravel:
    def __init__(self, fraction):
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("hydraulic_travel_fraction must be finite and positive")
        self.fraction = float(fraction)
        self.summary = {"updates": 0, "limited": 0, "rejected": 0,
                        "radius_sum": 0., "proposed_rms_sum": 0., "accepted_rms_sum": 0.,
                        "network_rms_sum": 0., "prior_rms_sum": 0.,
                        "shared_fraction_sum": 0., "scale_sum": 0., "work_sum": 0.,
                        "max_accepted_radius_ratio": 0., "probe_calls": 0}

    @staticmethod
    def rms(value):
        return value.flatten(1).square().sum(1).mean().sqrt()

    @torch.no_grad()
    def step(self, optimizer, real, probe):
        points = real.detach().flatten(1)
        if len(points) < 2:
            raise ValueError("hydraulic travel needs at least two real training samples")
        distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
        distances.fill_diagonal_(float("inf"))
        radius = self.fraction * float(distances.min(1).values.median())
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("hydraulic travel requires positive finite batch spacing")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in parameters]
        old = probe()[1]
        optimizer.step()
        delta = [p.detach() - value for p, value in zip(parameters, before)]
        network, joint = probe()
        proposed = float(self.rms(joint - old))
        network_rms = float(self.rms(network - old))
        prior_rms = float(self.rms(joint - network))
        travel = (joint - old).flatten(1)
        shared = float(travel.mean(0).square().sum() / travel.square().sum(1).mean().clamp_min(1e-30))
        scale, accepted, calls = 1., proposed, 2
        if not math.isfinite(proposed):
            scale = 0.
        elif proposed > radius:
            scale = radius / proposed
        if scale < 1:
            # One ratio estimate plus at most six halvings. A finite exact
            # replay checks nonlinear output motion; exhaustion rejects motion.
            for _ in range(7):
                for p, value, movement in zip(parameters, before, delta):
                    p.copy_(value + scale * movement)
                _, joint = probe()
                calls += 1
                accepted = float(self.rms(joint - old))
                if math.isfinite(accepted) and accepted <= radius:
                    break
                scale *= .5
            else:
                scale = 0.
                for p, value in zip(parameters, before):
                    p.copy_(value)
                accepted = 0.
        work = -sum(float((p.grad * (p.detach() - value)).sum())
                    for p, value in zip(parameters, before) if p.grad is not None)
        row = self.summary
        row["updates"] += 1
        row["limited"] += int(scale < 1)
        row["rejected"] += int(scale == 0)
        row["probe_calls"] += calls
        for key, value in (("radius", radius), ("proposed_rms", proposed), ("accepted_rms", accepted),
                           ("network_rms", network_rms), ("prior_rms", prior_rms),
                           ("shared_fraction", shared), ("scale", scale), ("work", work)):
            row[key + "_sum"] += value
        row["max_accepted_radius_ratio"] = max(row["max_accepted_radius_ratio"], accepted / radius)

    def state_dict(self):
        return {"fraction": self.fraction, "summary": deepcopy(self.summary)}

    def validate_state_dict(self, state, steps):
        if not isinstance(state, dict) or set(state) != {"fraction", "summary"} or state["fraction"] != self.fraction:
            raise ValueError("hydraulic checkpoint setting differs")
        row = state["summary"]
        if (not isinstance(row, dict) or row.keys() != self.summary.keys()
                or row["updates"] != steps or any(type(value) not in (int, float)
                or not math.isfinite(value) for value in row.values())
                or not 0 <= row["rejected"] <= row["limited"] <= steps
                or not 0 <= row["max_accepted_radius_ratio"] <= 1):
            raise ValueError("invalid hydraulic checkpoint counters")

    def load_state_dict(self, state):
        self.summary = deepcopy(state["summary"])
