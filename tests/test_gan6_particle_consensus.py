"""Critic-basin consensus decisions. No training and no data assignment."""

import torch

from reports.toy100.gan6_particle_consensus import consensus_delta, shell_probes


def _mids(points, score_fn):
    n = points.shape[0]
    mid = points.new_zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            value = score_fn(0.5 * (points[i] + points[j]))
            mid[i, j] = mid[j, i] = value
    return mid


def test_separate_redundant_particle_toward_empty_high_probe():
    points = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0]])
    scores = torch.tensor([1.0, 0.2, 1.1])
    probes = torch.tensor([[0.0, 3.0]])
    probe_scores = torch.tensor([1.4])
    dx, info = consensus_delta(points, scores, _mids(points, lambda p: torch.tensor(1.0)),
                               probes, probe_scores)
    assert info["action"] == "separate"
    assert info["donor"] == 1
    assert torch.allclose(dx[0], torch.zeros(2))
    assert dx[1, 1] > 0


def test_shrink_when_shared_basin_has_no_hole():
    points = torch.tensor([[0.0, 0.0], [0.2, 0.0]])
    scores = torch.tensor([1.0, 1.0])
    probes = torch.tensor([[5.0, 0.0]])
    probe_scores = torch.tensor([-2.0])
    dx, info = consensus_delta(points, scores, _mids(points, lambda p: torch.tensor(1.0)),
                               probes, probe_scores)
    assert info["action"] == "shrink"
    assert info["holes"] == 0
    assert dx[0, 0] > 0 and dx[1, 0] < 0


def test_idle_when_basins_are_already_distinct():
    points = torch.tensor([[0.0, 0.0], [3.0, 0.0]])
    scores = torch.tensor([1.0, 1.0])
    probes = torch.tensor([[1.5, 2.0]])
    probe_scores = torch.tensor([-1.0])

    def valley(p):
        return torch.tensor(-1.0 if abs(float(p[0]) - 1.5) < 0.2 else 1.0)

    dx, info = consensus_delta(points, scores, _mids(points, valley), probes, probe_scores)
    assert info["action"] == "idle"
    assert info["basins"] == 2
    assert float(dx.norm()) == 0.0


def test_shell_follows_particle_cloud_not_a_fixed_ring():
    points = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
    probes = shell_probes(points)
    center = probes.mean(0)
    assert torch.allclose(center, torch.zeros(2), atol=1e-5)
    radius = (probes - center).norm(dim=1)
    assert torch.allclose(radius, torch.full_like(radius, 2.0), atol=1e-4)
