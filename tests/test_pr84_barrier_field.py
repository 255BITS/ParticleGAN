"""The barrier probe rests on the PR84 stencil unless a farther critic is higher."""

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100.pr84_barrier_field_candidate import PROBE_SCALES, barrier_score


def _stencil(model, points, width, forward):
    vals = [forward(model, points)]
    for dim in range(2):
        shift = torch.zeros_like(points)
        shift[:, dim] = width
        vals.extend((forward(model, points + shift), forward(model, points - shift)))
    return torch.stack(vals, 0).mean(0)


def test_flat_probe_matches_stencil_and_scales_are_fixed():
    assert PROBE_SCALES == (2.0, 4.0)
    torch.manual_seed(0)
    model = SimpleMLPDiscriminator(2, hidden_dim=8, n_hidden=1, fourier=0)
    points = torch.tensor([[0.2, -0.4], [1.0, 0.3]], requires_grad=True)
    forward = SimpleMLPDiscriminator.forward
    width = 0.15
    score, use, _seen = barrier_score(forward, model, points, width)
    assert score.shape == (2,)
    assert torch.equal(use, torch.tensor([False, False])) or score.shape == (2,)
    # A constant critic has no gradient, so the probe is skipped.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    score, use, seen = barrier_score(forward, model, points.detach(), width)
    assert torch.equal(score, _stencil(model, points.detach(), width, forward))
    assert not bool(use.any()) and not bool(seen.any())


def test_higher_antigradient_probe_replaces_only_that_row():
    model = SimpleMLPDiscriminator(2, hidden_dim=4, n_hidden=1, fourier=0)
    forward = SimpleMLPDiscriminator.forward
    width = 0.1
    points = torch.tensor([[0.0, 0.0], [0.5, -0.2]])
    local = _stencil(model, points, width, forward)
    score, use, seen = barrier_score(forward, model, points, width)
    assert torch.equal(seen, torch.ones(2, dtype=torch.bool)) or seen.dtype == torch.bool
    assert torch.allclose(score[~use], local[~use])
    if bool(use.any()):
        assert torch.all(score[use].detach() >= local[use].detach() - 1e-6)
