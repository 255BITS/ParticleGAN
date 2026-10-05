"""Scalar paper objectives and their score derivatives; no quality claims."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import GANLoss, GANTrainer, Recipe, get_recipe


REAL = (-2., .25, 3.)
FAKE = (.75, -.4, 1.4)


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def expected_values(loss):
    """Independent scalar definitions (natural logarithms, raw scores)."""
    if loss == "relativistic":
        d = sum(math.log1p(math.exp(f - r)) for r, f in zip(REAL, FAKE)) / 3
        g = sum(math.log1p(math.exp(r - f)) for r, f in zip(REAL, FAKE)) / 3
    elif loss == "non_saturating":
        d = sum(math.log1p(math.exp(-r)) for r in REAL) / 3
        d += sum(math.log1p(math.exp(f)) for f in FAKE) / 3
        g = sum(math.log1p(math.exp(-f)) for f in FAKE) / 3
    elif loss == "hinge":
        d = (sum(max(0., 1 - r) for r in REAL) + sum(max(0., 1 + f) for f in FAKE)) / 3
        g = -sum(FAKE) / 3
    elif loss == "wasserstein":
        d = (sum(FAKE) - sum(REAL)) / 3
        g = -sum(FAKE) / 3
    else:
        d = (sum((r - 1) ** 2 for r in REAL) + sum(f ** 2 for f in FAKE)) / 6
        g = sum((f - 1) ** 2 for f in FAKE) / 6
    return d, g


@pytest.mark.parametrize("loss", GANLoss.LOSSES)
@pytest.mark.parametrize("shape", [(3,), (3, 1)])
def test_paper_scalar_values_on_raw_scores(loss, shape):
    real = torch.tensor(REAL, dtype=torch.float64).reshape(shape)
    fake = torch.tensor(FAKE, dtype=torch.float64).reshape(shape)
    actual = GANLoss(loss)
    d, g = actual.d_loss(real, fake), actual.g_loss(fake, real)
    assert d.shape == g.shape == torch.Size([])
    assert (d.item(), g.item()) == pytest.approx(expected_values(loss), rel=1e-14)


@pytest.mark.parametrize("loss", GANLoss.LOSSES)
def test_analytic_score_derivatives_and_generator_real_ownership(loss):
    real = torch.tensor(REAL, dtype=torch.float64, requires_grad=True)
    fake = torch.tensor(FAKE, dtype=torch.float64, requires_grad=True)
    objective = GANLoss(loss)
    d_real, d_fake = torch.autograd.grad(objective.d_loss(real, fake), (real, fake))
    g_real, g_fake = torch.autograd.grad(objective.g_loss(fake, real), (real, fake), allow_unused=True)
    if loss == "relativistic":
        expected_d_real = [-sigmoid(f - r) / 3 for r, f in zip(REAL, FAKE)]
        expected_d_fake = [sigmoid(f - r) / 3 for r, f in zip(REAL, FAKE)]
        expected_g_real = [sigmoid(r - f) / 3 for r, f in zip(REAL, FAKE)]
        expected_g_fake = [-sigmoid(r - f) / 3 for r, f in zip(REAL, FAKE)]
        assert g_real.tolist() == pytest.approx(expected_g_real)
    elif loss == "non_saturating":
        expected_d_real = [-sigmoid(-r) / 3 for r in REAL]
        expected_d_fake = [sigmoid(f) / 3 for f in FAKE]
        expected_g_fake = [-sigmoid(-f) / 3 for f in FAKE]
    elif loss == "hinge":
        expected_d_real = [-1 / 3 if r < 1 else 0. for r in REAL]
        expected_d_fake = [1 / 3 if f > -1 else 0. for f in FAKE]
        expected_g_fake = [-1 / 3] * 3
    elif loss == "wasserstein":
        expected_d_real, expected_d_fake = [-1 / 3] * 3, [1 / 3] * 3
        expected_g_fake = [-1 / 3] * 3
    else:
        expected_d_real = [(r - 1) / 3 for r in REAL]
        expected_d_fake = [f / 3 for f in FAKE]
        expected_g_fake = [(f - 1) / 3 for f in FAKE]
    assert d_real.tolist() == pytest.approx(expected_d_real)
    assert d_fake.tolist() == pytest.approx(expected_d_fake)
    assert g_fake.tolist() == pytest.approx(expected_g_fake)
    if loss != "relativistic":
        assert g_real is None
        assert torch.equal(objective.g_loss(fake), objective.g_loss(fake, real))


@pytest.mark.parametrize("loss", ["relativistic", "non_saturating", "least_squares"])
def test_nonlinear_losses_support_second_order_autograd(loss):
    real = torch.tensor(REAL, dtype=torch.float64, requires_grad=True)
    fake = torch.tensor(FAKE, dtype=torch.float64, requires_grad=True)
    objective = GANLoss(loss)
    assert torch.autograd.gradcheck(objective.d_loss, (real, fake))
    assert torch.autograd.gradgradcheck(objective.d_loss, (real, fake))
    assert torch.autograd.gradcheck(objective.g_loss, (fake, real))
    assert torch.autograd.gradgradcheck(objective.g_loss, (fake, real))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_retains_original_exact_relativistic_arithmetic(dtype):
    real = torch.tensor(REAL, dtype=dtype)
    fake = torch.tensor(FAKE, dtype=dtype)
    objective = GANLoss()
    assert torch.equal(objective.d_loss(real, fake), F.softplus(-(real - fake)).mean())
    assert torch.equal(objective.g_loss(fake, real), F.softplus(-(fake - real)).mean())
    with pytest.raises(ValueError, match="RpGAN requires real_logits"):
        objective.g_loss(fake)


@pytest.mark.parametrize("loss", GANLoss.LOSSES[1:])
def test_unpaired_discriminator_batches_reduce_separately(loss):
    real = torch.ones(2, dtype=torch.float64)
    fake = torch.zeros(7, dtype=torch.float64)
    d = GANLoss(loss).d_loss(real, fake)
    expected = {"non_saturating": math.log1p(math.exp(-1)) + math.log(2),
                "hinge": 1., "wasserstein": -1., "least_squares": 0.}
    assert d.item() == pytest.approx(expected[loss])


@pytest.mark.parametrize("loss", ["unknown", "logistic", "rp", "", None, True, ["hinge"]])
def test_invalid_loss_is_rejected_at_public_configuration_boundary(loss):
    with pytest.raises(ValueError, match="adversarial loss"):
        GANLoss(loss)
    with pytest.raises(ValueError, match="adversarial loss"):
        Recipe(loss=loss)
    with pytest.raises(ValueError, match="adversarial loss"):
        get_recipe("bcap", loss=loss)


@pytest.mark.parametrize("loss", GANLoss.LOSSES)
def test_public_recipe_selects_objective_and_persists_alternatives(loss):
    recipe = get_recipe("bcap", loss=loss, num_particles=8, batch_size=4)
    assert recipe.loss == recipe.make_loss().loss == loss
    packet = recipe.to_dict()
    assert Recipe(**packet) == recipe
    if loss == "relativistic":
        assert "loss" not in packet  # original checkpoint/config identities
    else:
        assert packet["loss"] == loss
    trainer = GANTrainer(recipe, nn.Linear(2, 2), nn.Linear(2, 1))
    assert trainer.loss.loss == loss
    assert trainer.completed_steps == 0


def test_historical_default_packet_keeps_fixed_loss_implicit():
    for name in ("gan", "ka2", "k3p", "e22", "mog"):
        recipe = get_recipe(name)
        assert recipe.loss == "relativistic"
        assert "loss" not in recipe.to_dict()

