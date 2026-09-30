"""Rest, energy decrease, latent cap, and rejected-trial restoration."""

import torch

from reports.toy100.chi_mixture_pullback import drift, mixture_pullback


def test_identical_clouds_have_zero_drift_and_do_not_move():
    cloud = torch.tensor([[0., 0.], [1., -1.], [2., 0.5]], dtype=torch.float64)
    assert float(drift(cloud, cloud, torch.tensor(1.)).abs().max()) == 0.
    z = torch.nn.Parameter(cloud.clone())
    before = z.detach().clone()
    row = mixture_pullback(torch.nn.Identity(), z, cloud, torch.ones_like(z))
    assert row["accepted"] and row["alpha"] == 0.
    assert torch.equal(z, before)


def test_missing_cluster_step_lowers_energy_without_exceeding_caps():
    real = torch.tensor([[0., 0.], [0.1, 0.], [3., 0.], [3.1, 0.]], dtype=torch.float64)
    points = torch.tensor([[0., 0.05], [0.05, -0.05]], dtype=torch.float64)
    z = torch.nn.Parameter(points.clone())
    row = mixture_pullback(torch.nn.Identity(), z, real, torch.ones_like(z))
    assert row["accepted"] and row["alpha"] > 0.
    assert row["energy_after"] < row["energy_before"]
    assert row["max_output_displacement"] <= 0.1 + 1e-8
    assert row["actual_latent_displacement_norm"] <= 2. ** 0.5 + 1e-6
    assert float((z.detach() - points)[:, 0].sum()) > 0.


def test_rejected_trial_restores_prior_when_energy_cannot_fall():
    real = torch.tensor([[3., 0.], [3.1, 0.]], dtype=torch.float64)
    # A constant map has a zero Jacobian, so no latent step can change energy.
    z = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64))
    before = z.detach().clone()
    row = mixture_pullback(lambda latent: torch.zeros(len(latent), 2, dtype=latent.dtype),
                           z, real, torch.ones_like(z))
    assert not row["accepted"]
    assert torch.equal(z, before)
