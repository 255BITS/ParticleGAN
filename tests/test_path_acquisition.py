"""Path-crossing direction: barrier flip, owner gate, and exact rest."""

import torch

from reports.toy100.path_acquisition import path_crossing_directions, redirect_loss_grad


def _field(points):
    out = torch.full((points.shape[0],), -0.2)
    out[points.norm(dim=1) < 0.05] = 0.0
    out[(points - torch.tensor([1.0, 0.0])).norm(dim=1) < 0.2] = -1.0
    out[(points - torch.tensor([2.5, 0.0])).norm(dim=1) < 0.3] = 2.0
    return out


def test_ray_crosses_dip_toward_empty_basin_for_the_owning_particle_only():
    support = torch.tensor([[0.0, 0.0], [-3.0, 0.0]])
    direction = path_crossing_directions(_field, support, support)
    assert torch.allclose(direction[0], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.equal(direction[1], torch.zeros(2))


def test_flat_field_and_zero_gradient_stay_at_rest():
    support = torch.tensor([[0.0, 0.0], [1.5, 0.0]])
    direction = path_crossing_directions(lambda points: torch.zeros(points.shape[0]), support, support)
    assert torch.equal(direction, torch.zeros_like(support))
    grad = torch.zeros(2, 2)
    assert torch.equal(redirect_loss_grad(grad, torch.tensor([[1.0, 0.0], [0.0, 1.0]])), grad)


def test_redirect_flips_only_an_opposing_loss_gradient_and_keeps_its_norm():
    grad = torch.tensor([[0.3, 0.4], [-0.3, -0.4]])
    direction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    out = redirect_loss_grad(grad, direction)
    assert torch.allclose(out[0], torch.tensor([-0.5, 0.0]))
    assert torch.equal(out[1], grad[1])
