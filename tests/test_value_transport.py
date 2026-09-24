"""Critic-value spare-particle rule: rest, barrier walk, and rejected pullback."""
import torch
from torch import nn

from reports.toy100.value_transport import select_particle, spare_particle_pull


def test_matched_scores_do_not_fire():
    points = torch.tensor([[0., 0.], [0.02, 0.]])
    real = torch.tensor([[0.01, 0.0], [0.0, 0.02], [-0.01, 0.]])
    scores = torch.tensor([0.1, 0.0, 0.12])
    choice, info = select_particle(points, real, torch.tensor([0.08, 0.1]), scores)
    assert choice is None and info["fired"] is False


def test_spare_walks_toward_high_scoring_reals_and_a_singleton_does_not():
    points = torch.tensor([[-2., -2.1], [-2.05, -2.12], [2.1, -2.2]])
    real = torch.tensor([[0., -3.], [0.05, -2.95], [-2., -2.1], [2.1, -2.2]])
    fake_scores = torch.tensor([-0.9, -0.8, -0.7])
    real_scores = torch.tensor([1.8, 1.7, -0.5, -0.4])
    choice, info = select_particle(points, real, fake_scores, real_scores)
    assert info["fired"] and info["twin"] and choice[0] == 0
    assert abs(choice[1][1].item() + 3) < 0.1

    alone = torch.tensor([[-2., -2.1], [2.1, -2.2], [0., 3.]])
    choice, info = select_particle(alone, real, torch.tensor([-0.9, -0.7, 0.2]), real_scores)
    assert choice is None and info.get("blocked") is True


def test_pullback_moves_only_the_selected_row_and_restores_on_a_dead_map():
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    z = torch.tensor([[-2., -2.1], [-2.04, -2.08], [2.1, -2.2]], requires_grad=False)
    real = torch.tensor([[0., -3.], [0.02, -2.98], [-2., -2.], [2., -2.]])
    fake_scores = torch.tensor([-1., -0.9, -0.8])
    real_scores = torch.tensor([2., 1.9, -0.2, -0.2])
    metric = torch.ones_like(z)
    before = z.clone()
    info = spare_particle_pull(layer, z, real, fake_scores, real_scores, metric)
    assert info["accepted"] and info["nearest"] == 0
    assert info["output_error_after"] < info["output_error_before"]
    assert torch.equal(z[1:], before[1:])
    assert not torch.equal(z[0], before[0])
    assert info["latent_norm"] <= 0.1 + 1e-5

    dead = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        dead.weight.zero_()
    z2 = before.clone()
    info = spare_particle_pull(dead, z2, real, fake_scores, real_scores, metric)
    assert info["fired"] and not info["accepted"]
    assert torch.equal(z2, before)
