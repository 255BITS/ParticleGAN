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
    def radius(self, real):
        points = real.detach().flatten(1)
        if len(points) < 2:
            raise ValueError("hydraulic travel needs at least two real training samples")
        distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
        distances.fill_diagonal_(float("inf"))
        radius = self.fraction * float(distances.min(1).values.median())
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("hydraulic travel requires positive finite batch spacing")
        return radius

    @torch.no_grad()
    def step(self, optimizer, real, probe, *, radius=None):
        radius = self.radius(real) if radius is None else radius
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
        if radius == 0:
            scale, accepted = 0., 0.
            for p, value in zip(parameters, before):
                p.copy_(value)
        elif not math.isfinite(proposed):
            scale = 0.
        elif proposed > radius:
            scale = radius / proposed
        if scale < 1 and radius > 0:
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
        row["max_accepted_radius_ratio"] = max(row["max_accepted_radius_ratio"], accepted / radius if radius else 0.)

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


class HydraulicDeformationTravel(HydraulicTravel):
    """Travel plus a network-only antithetic finite-response penalty.

    Uses the already sampled MoG jitter, never another random draw. This is a
    contraction preference, not a target-covariance estimator or guarantee.
    """
    def __init__(self, fraction, weight):
        super().__init__(fraction)
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("hydraulic_deformation_weight must be finite and positive")
        self.weight = float(weight)
        self.summary.update(zero_spacing=0, secant_energy_sum=0., real_variance_sum=0.,
                            deformation_loss_sum=0.)

    @torch.no_grad()
    def radius(self, real):
        points = real.detach().flatten(1)
        if not torch.isfinite(points).all():
            raise ValueError("hydraulic deformation needs finite real training samples")
        variance = (points - points.mean(0)).square().sum(1).mean()
        if not torch.isfinite(variance):
            raise ValueError("hydraulic deformation needs finite real batch variance")
        if len(points) < 2:
            return 0.
        distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
        # Duplicates supply no distinct spacing. Use nearest positive distance;
        # a constant finite batch supplies zero, hence a zero G/prior ray.
        distances.masked_fill_(distances == 0, float("inf"))
        nearest = distances.min(1).values
        if torch.isinf(nearest).all() and variance == 0:
            return 0.
        radius = self.fraction * float(nearest.median())
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("hydraulic deformation needs finite distinct batch spacing")
        return radius

    def regularizer(self, generator, latent, centers, real, loss_gan):
        points = real.detach().flatten(1)
        variance = (points - points.mean(0)).square().sum(1).mean()
        # Detach both latent coordinates: this term owns network gradients only.
        plus, minus = generator(torch.cat((latent.detach(),
                            2 * centers.detach() - latent.detach()), 0)).chunk(2)
        energy = ((plus - minus) * .5).flatten(1).square().sum(1).mean()
        loss = (self.weight * loss_gan.detach().abs() * energy / variance
                if variance > 0 else energy * 0.)
        self.summary['secant_energy_sum'] += float(energy.detach())
        self.summary['real_variance_sum'] += float(variance)
        self.summary['deformation_loss_sum'] += float(loss.detach())
        return loss

    @torch.no_grad()
    def step(self, optimizer, real, probe, *, radius=None):
        radius = self.radius(real) if radius is None else radius
        self.summary['zero_spacing'] += int(radius == 0)
        super().step(optimizer, real, probe, radius=radius)

    def state_dict(self):
        return {**super().state_dict(), 'deformation_weight': self.weight,
                'zero_spacing_policy': 'nearest_distinct_or_zero_ray_v1'}

    def validate_state_dict(self, state, steps):
        if (not isinstance(state, dict) or state.get('deformation_weight') != self.weight
                or state.get('zero_spacing_policy') != 'nearest_distinct_or_zero_ray_v1'):
            raise ValueError('hydraulic deformation checkpoint setting differs')
        base = {k: v for k, v in state.items() if k not in ('deformation_weight','zero_spacing_policy')}
        super().validate_state_dict(base, steps)
        if not 0 <= state['summary']['zero_spacing'] <= steps:
            raise ValueError('invalid hydraulic zero-spacing counters')
