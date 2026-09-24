"""Monge targets split a doubled mode instead of averaging into the valley."""

import math

import pytest
import torch

from reports.toy100.monge_pullback import OUTPUT_CAP, farthest_representatives, hungarian, monge_pullback


def test_hungarian_is_the_minimum_cost_bijection():
    cost = torch.tensor([[9., 2., 7.], [6., 4., 3.], [5., 8., 1.]])
    assignment = hungarian(cost)
    # Optimal bijection cost is 2+6+1=9; the other column-1 start costs 10.
    assert assignment.tolist() == [1, 0, 2]
    assert float(cost[torch.arange(3), assignment].sum()) == 9.


def test_farthest_points_cover_every_ring_mode():
    theta = torch.arange(8) * (2 * math.pi / 8)
    centers = 3 * torch.stack((theta.cos(), theta.sin()), 1)
    real = centers.repeat(16, 1)
    picked = farthest_representatives(real, 12)
    nearest = torch.cdist(picked, centers).min(dim=0).values
    assert int((nearest < 1e-6).sum()) == 8
    assert picked.shape == (12, 2)


def test_missing_mode_is_assigned_not_averaged():
    theta = torch.arange(8, dtype=torch.float64) * (2 * math.pi / 8)
    centers = 3 * torch.stack((theta.cos(), theta.sin()), 1)
    # Mode 6 is empty. Two particles sit on mode 5; the rest cover 0..4 and 7, plus one extra on 0.
    slots = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5, 7, 7, 0, 1])
    particles = centers[slots]
    real = centers.repeat(16, 1)
    targets = farthest_representatives(real, 12)
    assignment = hungarian(torch.cdist(particles, targets).square())
    matched = targets[assignment]
    # The two mode-5 particles must not share a midpoint target in the valley.
    mode6 = centers[6]
    # Some particle is sent to the empty mode itself, not to the valley midpoint.
    assert float(torch.cdist(matched, mode6[None]).min()) < 1e-6
    traveler = int(torch.cdist(matched, mode6[None]).argmin())
    assert int(slots[traveler]) != 6
    midpoint = 0.5 * (centers[5] + centers[6])
    assert float((matched - midpoint).norm(dim=1).min()) > 1.0


class Identity(torch.nn.Module):
    def forward(self, z):
        return z


def test_capped_identity_step_walks_toward_the_hole_and_rests_when_matched():
    theta = torch.arange(8, dtype=torch.float64) * (2 * math.pi / 8)
    centers = 3 * torch.stack((theta.cos(), theta.sin()), 1)
    slots = torch.tensor([0, 0, 1, 2, 3, 4, 5, 5, 7, 7, 0, 1])
    z = torch.nn.Parameter(centers[slots].clone())
    real = centers.repeat(16, 1)
    before = z.detach().clone()
    row = monge_pullback(Identity(), z, real, torch.ones_like(z), output_cap=OUTPUT_CAP)
    assert row["accepted"] and row["alpha"] == 1.
    move = (z.detach() - before).norm(dim=1)
    assert float(move.max()) == pytest.approx(OUTPUT_CAP, abs=1e-8)
    # At least one particle reduces its distance to mode 6.
    hole = centers[6]
    assert float((z.detach() - hole).norm(dim=1).min()) < float((before - hole).norm(dim=1).min()) - 0.1

    targets = farthest_representatives(real, 12)
    rested_z = torch.nn.Parameter(targets.clone())
    rested = monge_pullback(Identity(), rested_z, real, torch.ones_like(rested_z), output_cap=OUTPUT_CAP)
    assert not rested["accepted"]
    assert torch.equal(rested_z, targets)
