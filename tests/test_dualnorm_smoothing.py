"""CUDA analytic and checkpoint checks for fixed-scale smoothed DualNorm."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from particlegan.optim.dualnorm import NormalizedOptimizer, polar_factor


@pytest.fixture(autouse=True)
def cuda_contract():
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for smoothing verification")
    with torch.device("cuda:0"), torch.autograd.set_multithreading_enabled(False):
        yield


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("truncate", [False, True])
def test_diagonal_paper_operator_and_null_direction(dtype, truncate):
    matrix = torch.diag(torch.tensor([3., .004, 0.], dtype=dtype))
    result = polar_factor(matrix, smoothing=.003, truncate=truncate)
    expected = torch.diag(matrix.new_tensor([3 / math.sqrt(9 + .003**2), .8, 0.]))
    torch.testing.assert_close(result, expected)
    assert result.device.type == "cuda" and result.dtype == dtype
    assert torch.isfinite(polar_factor(torch.zeros_like(matrix), smoothing=.003)).all()


def test_rounding_perturbation_is_damped_in_null_space():
    left = torch.diag(torch.tensor([1., 1e-8, 0.], dtype=torch.float64))
    right = left.clone()
    right[-1, -1] = -1e-10
    # The full polar completion has an O(1) change from this tiny perturbation.
    difference = (polar_factor(left, truncate=False) - polar_factor(right, truncate=False)).norm()
    smooth_difference = (polar_factor(left, smoothing=1e-4, truncate=False)
                         - polar_factor(right, smoothing=1e-4, truncate=False)).norm()
    assert difference >= 1 and smooth_difference < 1.01e-6


def test_sampled_prior_rows_and_biases_follow_analytic_vector_rule():
    prior = nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    bias = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = NormalizedOptimizer([dict(params=[prior], role="prior"),
                                     dict(params=[bias], role="generator")], lr=.2, smoothing=3.)
    prior.grad = prior.new_tensor([[0., 4.], [9., 12.], [0., 0.]])
    bias.grad = bias.new_tensor([0., 4.])
    optimizer.set_sampled_rows(prior, torch.tensor([0, 2], dtype=torch.long))
    optimizer.step()
    torch.testing.assert_close(prior, prior.new_tensor([[0., -.16], [0., 0.], [0., 0.]]))
    torch.testing.assert_close(bias, bias.new_tensor([0., -.16]))


def test_optimizer_checkpoint_preserves_rule_and_rejects_mismatch_atomically():
    parameter = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    optimizer = NormalizedOptimizer([parameter], smoothing=1e-4)
    parameter.grad = torch.eye(2, dtype=torch.float64)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    other = NormalizedOptimizer([nn.Parameter(parameter.clone())], smoothing=1e-4)
    other.load_state_dict(saved)
    assert other.state_dict()["dualnorm"]["smoothing"] == 1e-4
    before = deepcopy(optimizer.state_dict())
    invalid = deepcopy(saved)
    invalid["dualnorm"]["smoothing"] = .1
    with pytest.raises(ValueError, match="smoothing"):
        optimizer.load_state_dict(invalid)
    assert optimizer.state_dict()["dualnorm"] == before["dualnorm"]
    with pytest.raises(ValueError):
        NormalizedOptimizer([nn.Parameter(parameter.clone())]).load_state_dict(saved)


def test_public_recipe_consumes_scale_for_every_role_and_preserves_zero_packet():
    base = get_recipe("bcap", optimizer_family="dualnorm", num_particles=5, z_dim=2)
    assert "optimizer_smoothing" not in base.to_dict()
    smooth = base.replace(optimizer_smoothing=1e-4)
    prior = smooth.make_prior()
    generator, critic = nn.Linear(2, 1), nn.Linear(1, 1)
    optimizers = smooth.make_optimizers(generator, critic, prior)
    assert all(optimizer.smoothing == 1e-4 for optimizer in optimizers)
    assert smooth.to_dict()["optimizer_smoothing"] == 1e-4
    from experiments.forge.techniques import validate_same_technique
    with pytest.raises(ValueError, match="smoothed_dualnorm"):
        validate_same_technique(base, smooth)


def test_public_trainer_restores_exact_next_smoothed_update():
    def build():
        recipe = get_recipe("bcap", optimizer_family="dualnorm", optimizer_smoothing=1e-4,
                            num_particles=8, z_dim=2, batch_size=4, total_steps=8,
                            standardize=False, prior_kind="mog", sigma_rel=.1)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        critic = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for index, module in enumerate((generator, critic, prior)):
            deterministic_orthogonal_(module, seed=index)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator(device="cuda:0").manual_seed(0))
    trainer = build()
    batch = torch.tensor([[-1.], [-.2], [.4], [1.]])
    trainer.step(batch)
    saved = deepcopy(trainer.state_dict())
    trainer.step(batch)
    expected = trainer.state_dict()
    restored = build()
    restored.load_state_dict(saved)
    restored.step(batch)
    from experiments.forge.state import state_digest
    assert state_digest(restored.state_dict()) == state_digest(expected)


@pytest.mark.parametrize("value", [True, -1., float("nan"), float("inf")])
def test_invalid_scales_rejected(value):
    with pytest.raises(ValueError):
        get_recipe("bcap", optimizer_family="dualnorm", optimizer_smoothing=value)
    with pytest.raises(ValueError):
        polar_factor(torch.eye(2), smoothing=value)
