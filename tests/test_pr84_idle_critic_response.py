"""The idle response matches the PR84 stencil unless a far high-critic peak faces away."""

import torch

from reports.toy100.pr84_idle_critic_response import (
    CRITIC_MARGIN, NEAR_RADIUS, POCKET_GAP, SEGMENT_PROBES, idle_gate, response_score,
    stencil_mean,
)


class _Field:
    """Sharp critic with a peak at ``pocket`` and a negative slope through the origin."""

    def __call__(self, module, x):
        pocket = torch.tensor([3.0, 0.0], dtype=x.dtype)
        center = 4.0 * torch.exp(-((x - pocket) ** 2).sum(-1))
        slope = -0.4 * x[..., 0]
        return center + slope


def test_constants_are_a_single_gate():
    assert CRITIC_MARGIN == 0.5
    assert POCKET_GAP == 1.0
    assert NEAR_RADIUS == 0.75
    assert SEGMENT_PROBES == 8


def test_covered_cloud_stays_idle():
    field = _Field()
    particles = torch.tensor([[3.05, 0.02], [-3.0, 0.0]])
    real = torch.tensor([[3.0, 0.0], [-2.9, 0.1]])
    respond, pocket, anchor, travel, peak = idle_gate(field, None, real, particles, width=0.15)
    assert respond is False and pocket is None and anchor is None and travel is None and peak is None


def test_uncovered_away_peak_arms_and_rewrites_only_the_near_row():
    field = _Field()
    particles = torch.tensor([[0.0, 0.0], [-3.0, 0.1]])
    real = torch.tensor([[3.0, 0.0], [-3.0, 0.0]])
    respond, pocket, anchor, travel, _peak = idle_gate(field, None, real, particles, width=0.15)
    assert respond is True and travel > 0.15
    assert torch.allclose(pocket, torch.tensor([3.0, 0.0]))
    assert torch.allclose(anchor, particles[0])
    rows = torch.tensor([[0.05, 0.0], [-3.0, 0.0]], requires_grad=True)
    local = stencil_mean(field, None, rows, 0.15)
    score, use = response_score(field, None, rows, 0.15, pocket, anchor, peak_score=4.0, travel=travel)
    assert bool(use[0]) and not bool(use[1])
    assert torch.allclose(score[1], local[1])
    score[0].backward()
    assert rows.grad[0, 0] > 0.
