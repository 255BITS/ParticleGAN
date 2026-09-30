"""Refreshed Chamfer targets and prior-only nonlinear landing."""

import torch

from reports.toy100.chamfer_pullback import chamfer_targets, chamfer_terms
from reports.toy100.chamfer_refreshed_pullback import (
    ROUNDS, nonlinear_refreshed_pullback, refreshed_chamfer_target,
)


def test_four_refreshes_move_a_stranded_particle_onto_uncovered_real_mass():
    real = torch.tensor([[0., 0.], [0.1, 0.], [5., 0.], [5.1, 0.]])
    points = torch.tensor([[0., 0.], [0.2, 0.], [0.4, 0.], [6., 4.]])
    four, _, _, _ = refreshed_chamfer_target(real, points, ROUNDS)
    assert float((four[-1] - torch.tensor([5., 0.])).norm()) < 0.5


def test_identity_landing_reaches_refreshed_target_and_restores_on_flat_map():
    real = torch.tensor([[0., 0.], [0.2, 0.], [4., 1.], [4.2, 1.1]])
    z = torch.nn.Parameter(torch.tensor([[0., 0.], [0.1, 0.], [0.3, 0.2], [1., 3.]]))
    row = nonlinear_refreshed_pullback(torch.nn.Identity(), z, real)
    assert row["accepted"] and row["rounds"] == 4
    assert row["final_target_error"] < 1e-3
    assert row["objective_after"] < row["objective_before"]
    flat = torch.nn.Parameter(torch.zeros(2, 2))
    before = flat.detach().clone()
    rejected = nonlinear_refreshed_pullback(lambda rows: rows * 0, flat, real[:2])
    assert not rejected["accepted"]
    assert torch.equal(flat, before)


def test_terms_of_landed_identity_match_the_receipt():
    real = torch.tensor([[-1., 0.], [1., 0.], [0., 3.]])
    z = torch.nn.Parameter(torch.tensor([[0., 0.], [0.2, 0.], [2., 2.]]))
    row = nonlinear_refreshed_pullback(torch.nn.Identity(), z, real)
    coverage, backward = chamfer_terms(real, z.detach())
    assert abs(float(coverage + backward) - row["objective_after"]) < 1e-5
