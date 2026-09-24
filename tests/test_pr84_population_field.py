import torch

from reports.toy100.pr84_population_field import ratio_score


def test_identical_mixtures_have_zero_population_score_and_field():
    means = torch.tensor([[-1., .5], [2., 1.]], requires_grad=True)
    x = torch.tensor([[0., .2], [1., 1.]], requires_grad=True)
    score = ratio_score(x, means, .07, means, .07)
    gradient = torch.autograd.grad(score.sum(), x)[0]
    assert torch.equal(score, torch.zeros_like(score))
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert means.grad is None


def test_fixed_equal_variance_gaussian_ratio_points_toward_real_mean():
    # If q's mean were differentiated through the sampled x, this derivative
    # would change. The virtual critic must hold its fitted distribution fixed.
    real = torch.tensor([[.3, -.1]], dtype=torch.float64)
    fake = torch.tensor([[.1, .2]], dtype=torch.float64, requires_grad=True)
    x = fake + torch.tensor([[.02, -.01]], dtype=torch.float64)
    score = ratio_score(x, real, .2, fake, .2)
    derivative = torch.autograd.grad(score.sum(), fake)[0]
    assert torch.allclose(derivative, (real - fake.detach()) / .2**2, atol=1e-12, rtol=0)
