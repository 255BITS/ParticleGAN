"""Residual curvature is the stencil leftover, not the plain critic."""

import torch

from reports.toy100.alternating_curvature_scratch import _rho
from reports.toy100.residual_curvature import residual_grads


def test_zero_residual_when_the_stencil_changes_nothing():
    plain = [torch.tensor([1., 0.]), torch.tensor([2.])]
    assert all(torch.equal(a, torch.zeros_like(a)) for a in residual_grads(plain, plain))


def test_residual_rho_drops_curvature_shared_with_the_smoothed_field():
    base = [torch.zeros(2)]
    new = [torch.ones(2)]
    metric = [torch.ones(2)]
    plain0, plain1 = [torch.zeros(2)], [torch.tensor([4., 0.])]
    smooth0, smooth1 = [torch.zeros(2)], [torch.tensor([4., 0.])]
    plain_rho = _rho(base, new, plain0, plain1, metric)
    leftover = _rho(base, new, residual_grads(plain0, smooth0), residual_grads(plain1, smooth1), metric)
    assert plain_rho > 0
    assert leftover == 0
