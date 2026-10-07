"""The two response rules compose in one public optimizer and cached field."""
import pytest
import torch
from torch import nn

from particlegan.extrapolation import stateless_directions
from particlegan.optim.dualnorm import NormalizedOptimizer
from experiments.forge.state import state_digest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_both_caps_preview_exact_correction_with_sampled_row_ownership():
    weight = nn.Parameter(torch.zeros(2, 2, device="cuda:0", dtype=torch.float64))
    prior = nn.Parameter(torch.zeros(4, 2, device="cuda:0", dtype=torch.float64))
    optimizer = NormalizedOptimizer([
        dict(params=[weight], role="generator"),
        dict(params=[prior], role="prior", row_gradient_scale=.001),
    ], lr=.03, network_update="spectral_capped", network_gradient_scale=.1)
    weight.grad = torch.diag(torch.tensor([.001, 1.], device="cuda:0", dtype=torch.float64))
    prior.grad = torch.tensor([[.0003, .0004], [.003, .004], [1., 1.], [1., 1.]],
                              device="cuda:0", dtype=torch.float64)
    optimizer.set_sampled_rows(prior, torch.tensor([0, 1], device="cuda:0"))
    before = state_digest(optimizer.state_dict())
    field = stateless_directions(optimizer)
    assert state_digest(optimizer.state_dict()) == before
    assert torch.equal(field[weight], torch.diag(torch.tensor([.01, 1.], device="cuda:0", dtype=torch.float64)))
    assert torch.allclose(field[prior][:2], torch.tensor([[.3, .4], [.6, .8]], device="cuda:0", dtype=torch.float64))
    assert torch.count_nonzero(field[prior][2:]) == 0
    optimizer.step()
    assert torch.equal(weight, -.03 * field[weight])
    assert torch.equal(prior, -.03 * field[prior])
