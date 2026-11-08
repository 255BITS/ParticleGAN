"""A nonfinite gradient must not be hidden by the zero-gradient skip mask."""
import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.optim.convolution_frobenius import frobenius_direction
from particlegan.optim.convolution_quintic import quintic_direction


@pytest.mark.parametrize("function", [frobenius_direction, quintic_direction])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_matrix_is_explicitly_rejected(function, value):
    matrix = torch.ones(2, 3, 4)
    matrix[1, 2, 3] = value
    with pytest.raises(ValueError, match="finite matrix"):
        function(matrix, smoothing=.001)


@pytest.mark.cuda
@pytest.mark.parametrize("rule", ["frobenius", "quintic"])
def test_optimizer_does_not_silently_skip_nan_gradient(rule):
    if not torch.cuda.is_available():
        pytest.skip("accelerated convolution dispatch requires CUDA")
    module = nn.Conv2d(2, 64, 1, bias=False, device="cuda:0")
    optimizer = get_recipe("bcap", constraint_geometry_mode="none",
                           optimizer_convolution_rule=rule).make_generator_optimizer(module)
    before = module.weight.detach().clone()
    module.weight.grad = torch.full_like(module.weight, float("nan"))
    with pytest.raises(ValueError, match="finite matrix"):
        optimizer.step()
    assert torch.equal(module.weight, before)
