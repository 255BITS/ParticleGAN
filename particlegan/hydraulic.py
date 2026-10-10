"""Bounded joint output travel along an existing zero-momentum GAN update.

Ported from the PR360 v1 study (``research/bcap-physics-hydraulic`` commit
13ef22dbb). The normal optimizer (including direction blending) proposes its
joint generator/prior update; this rule then scales the accepted parameter
delta along that same direction so the exactly replayed RMS output travel on
the consumed generator batch is at most ``fraction * radius``.

``real_spacing`` (v1): radius is the median nearest-neighbour distance within
the consumed real batch. ``gap_adaptive``: radius is the larger of that and
the median distance from each current generated sample to its nearest real
sample, so the bound relaxes while the generator is far from the data and
returns to the v1 bound once the batches overlap.

This is a sampled function-space trust bound, not a convergence guarantee.
It draws no RNG, evaluates no targets beyond the consumed training batch and
changes no objective.
"""
import math
from copy import deepcopy

import torch
from torch import nn


STATELESS_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                     nn.LeakyReLU, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Identity, nn.Flatten,
                     nn.Unflatten, nn.Upsample)


def require_stateless_generator(generator):
    if any(not tuple(module.children()) and not isinstance(module, STATELESS_MODULES)
           for module in generator.modules()) or any(True for _ in generator.buffers()):
        raise ValueError("hydraulic travel requires a deterministic stateless generator")


class HydraulicTravel:
    RADII = ("real_spacing", "gap_adaptive")

    def __init__(self, fraction, *, radius="real_spacing"):
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("hydraulic_travel_fraction must be finite and positive")
        if radius not in self.RADII:
            raise ValueError("hydraulic_travel_radius must be real_spacing or gap_adaptive")
        self.fraction, self.radius_mode = float(fraction), radius
        self.summary = {"updates": 0, "limited": 0, "rejected": 0,
                        "radius_sum": 0., "spacing_sum": 0., "gap_sum": 0., "gap_wider": 0,
                        "proposed_rms_sum": 0., "accepted_rms_sum": 0.,
                        "network_rms_sum": 0., "prior_rms_sum": 0.,
                        "shared_fraction_sum": 0., "scale_sum": 0., "work_sum": 0.,
                        "max_accepted_radius_ratio": 0., "probe_calls": 0}

    @staticmethod
    def rms(value):
        return value.flatten(1).square().sum(1).mean().sqrt()

    @staticmethod
    def _nearest(rows, cols, *, exclude_self):
        distances = torch.cdist(rows, cols, compute_mode="donot_use_mm_for_euclid_dist")
        if exclude_self:
            distances.fill_diagonal_(float("inf"))
        return float(distances.min(1).values.median())

    @torch.no_grad()
    def step(self, optimizer, real, probe):
        """Step ``optimizer`` once, then bound its replayed output travel.

        ``probe()`` returns ``(network_only, joint)`` outputs for the fixed
        consumed latents: network-only keeps sampled prior rows at their
        pre-step locations; joint uses the current rows.
        """
        points = real.detach().flatten(1)
        if len(points) < 2:
            raise ValueError("hydraulic travel needs at least two real training samples")
        spacing = self._nearest(points, points, exclude_self=True)
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("hydraulic travel requires positive finite batch spacing")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in parameters]
        old = probe()[1]
        gap = 0.
        if self.radius_mode == "gap_adaptive":
            gap = self._nearest(old.flatten(1), points, exclude_self=False)
            if not math.isfinite(gap):
                raise ValueError("hydraulic travel requires finite generated samples")
        radius = self.fraction * max(spacing, gap)
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
        row["gap_wider"] += int(gap > spacing)
        row["probe_calls"] += calls
        for key, value in (("radius", radius), ("spacing", spacing), ("gap", gap),
                           ("proposed_rms", proposed), ("accepted_rms", accepted),
                           ("network_rms", network_rms), ("prior_rms", prior_rms),
                           ("shared_fraction", shared), ("scale", scale), ("work", work)):
            row[key + "_sum"] += value
        row["max_accepted_radius_ratio"] = max(row["max_accepted_radius_ratio"], accepted / radius)

    def state_dict(self):
        return {"fraction": self.fraction, "radius": self.radius_mode, "summary": deepcopy(self.summary)}

    def validate_state_dict(self, state, steps):
        if (not isinstance(state, dict) or set(state) != {"fraction", "radius", "summary"}
                or state["fraction"] != self.fraction or state["radius"] != self.radius_mode):
            raise ValueError("hydraulic checkpoint setting differs")
        row = state["summary"]
        if (not isinstance(row, dict) or row.keys() != self.summary.keys()
                or row["updates"] != steps or any(type(value) not in (int, float)
                or not math.isfinite(value) for value in row.values())
                or not 0 <= row["rejected"] <= row["limited"] <= steps
                or not 0 <= row["gap_wider"] <= steps
                or not 0 <= row["max_accepted_radius_ratio"] <= 1):
            raise ValueError("invalid hydraulic checkpoint counters")

    def load_state_dict(self, state):
        self.summary = deepcopy(state["summary"])
