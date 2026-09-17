"""Retrieval gradients, legacy sampling parity, and study metric checks."""

import math

import pytest
import torch

from lib.particle_prior import HopfieldRead, ParticlePrior
from lib.hopfield_metrics import (evaluate_read, measure_sampling_floor,
                                  steps_to_tv, weight_metrics)
from lib.toy_models import make_100gaussian_weights, sample_100gaussians


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_retrieval_matches_formula_and_has_dense_gradients():
    prior = ParticlePrior(7, 3, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(31))
    read = HopfieldRead(prior, beta=2, learn_beta=True)
    q = torch.randn(5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    actual, p = read.retrieve(q)
    expected_p = (2 * q @ prior.z.T).softmax(-1)
    torch.testing.assert_close(p, expected_p)
    torch.testing.assert_close(actual, expected_p @ prior.z)
    actual.square().sum().backward()
    assert (prior.z.grad.abs().sum(dim=1) > 0).all()
    assert read.log_beta.grad.abs() > 0
    assert torch.autograd.gradcheck(lambda table, log_beta: ((log_beta.exp() * q @ table.T).softmax(-1) @ table),
                                   (prior.z, read.log_beta))


def test_beta_clamp_and_sampling_rng():
    prior = ParticlePrior(8, 2)
    read = HopfieldRead(prior, learn_beta=True)
    with torch.no_grad():
        read.log_beta.fill_(100)
    read.clamp_beta_()
    assert float(read.log_beta.detach()) == pytest.approx(math.log(256))
    with torch.no_grad():
        read.log_beta.fill_(-5)
    read.clamp_beta_()
    assert float(read.log_beta.detach()) == 0
    rng = torch.Generator().manual_seed(28)
    expected_q = torch.randn(10, 2, generator=torch.Generator().manual_seed(28))
    torch.testing.assert_close(read(10, generator=rng)[0], read.retrieve(expected_q)[0])
    fixed = HopfieldRead(prior, beta=0.5)
    fixed.clamp_beta_()
    assert not fixed.log_beta.requires_grad
    assert float(fixed.log_beta.exp()) == 0.5


def test_uniform_sampler_bitwise_legacy_and_weighted_grid_order():
    rng = torch.Generator().manual_seed(19)
    ix = torch.randint(0, 10, (40,), generator=rng)
    iy = torch.randint(0, 10, (40,), generator=rng)
    expected = torch.stack((ix - 4.5, iy - 4.5), dim=1) + torch.randn(40, 2, generator=rng) * 0.03
    actual = sample_100gaussians(40, torch.device("cpu"), generator=torch.Generator().manual_seed(19))
    assert torch.equal(expected, actual)
    weights = torch.zeros(100)
    weights[27] = 1
    points = sample_100gaussians(40, torch.device("cpu"), weights=weights, std=0)
    assert torch.equal(points, torch.tensor([-2.5, 2.5]).expand(40, 2))


def test_weights_are_seed_independent_and_noise_floor_is_measured():
    state = torch.random.get_rng_state()
    weights = make_100gaussian_weights()
    assert torch.equal(state, torch.random.get_rng_state())
    expected = torch.logspace(0, 2, 100, dtype=torch.float64)
    expected /= expected.sum()
    torch.testing.assert_close(weights.sort().values, expected)
    assert float(weights.max() / weights.min()) == pytest.approx(100)
    floor = measure_sampling_floor(weights, n_eval=4000, repeats=3)
    assert len(floor["values"]) == 3
    assert 0 < floor["mean"] < 0.1
    assert floor["sd"] > 0


def test_weight_metrics_ignore_stragglers_and_detect_delta_width():
    weights = torch.zeros(100)
    weights[0], weights[99] = 0.25, 0.75
    points = torch.cat((torch.full((100, 2), -4.5), torch.full((300, 2), 4.5), torch.zeros(100, 2)))
    metrics = weight_metrics(points, weights)
    assert metrics["tv"] == 0
    assert metrics["kl"] == 0
    assert metrics["hq"] == pytest.approx(0.8)
    assert metrics["modes"] == 2
    prior = ParticlePrior(2, 2)
    with torch.no_grad():
        prior.z.copy_(torch.tensor([[-4.5, -4.5], [4.5, 4.5]]))
    g = torch.nn.Identity()
    result, fake = evaluate_read(g, prior, torch.ones(100) / 100, n_eval=300)
    assert result["sigma_ratio"] == 0
    assert result["max_w"] is None
    assert fake.shape == (300, 2)
    assert g.training
    read = HopfieldRead(prior, beta=16)
    result, _ = evaluate_read(g, prior, torch.ones(100) / 100, read=read, n_eval=300, batch_size=37,
                              sample_generator=torch.Generator().manual_seed(31))
    assert 0 <= result["interp_hq"] <= 1
    assert 0 <= result["dead_frac"] <= 1
    assert 1 <= result["eff_n"] <= 2


def test_convergence_uses_final_uninterrupted_suffix():
    rows = [{"step": 100 * i, "tv": value} for i, value in enumerate([0.1, 0.02, 0.04, 0.025, 0.01], 1)]
    assert steps_to_tv(rows) == 400
    assert steps_to_tv(rows[:3]) is None
    assert steps_to_tv([]) is None
