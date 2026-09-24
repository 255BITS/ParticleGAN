"""Analytic checks for the intervention's declared distribution metric."""

import torch

from reports.toy100.matched_population_diagnostic import gaussian_mixture_mmd2


def test_identical_gaussian_mixtures_have_zero_mmd():
    centers = torch.tensor([[0., 0.], [1., 1.], [-1., 1.]])
    assert gaussian_mixture_mmd2(centers, centers) < 1e-12


def test_mmd_uses_distribution_not_particle_index_order():
    centers = torch.tensor([[0., 0.], [1., 1.], [-1., 1.]])
    assert gaussian_mixture_mmd2(centers, centers[[2, 0, 1]]) < 1e-12
    assert gaussian_mixture_mmd2(centers, centers + torch.tensor([.35, 0.])) > .01
