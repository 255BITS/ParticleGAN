"""Public-loss analytic identities; these tests do not train a model."""
import json
import math

import pytest
import torch
from torch.nn import functional as F

from particlegan import init
from experiments import paired_residual_oracle as oracle


def test_public_fixture_init_preserves_constants_and_global_rng():
    before = torch.get_rng_state().clone()
    critic = oracle.PolynomialCritic(-.125, .0625, .25)
    assert all(value is init.KEEP for value in init.declarations(critic).values())
    assert [float(p.detach()) for p in critic.parameters()] == [-.125, .0625, .25]
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("linear", [0., .0625, -.4])
def test_zero_residual_antithetic_quadratic_cancellation_and_odd_force(linear):
    critic = oracle.PolynomialCritic(linear=linear)
    delta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    noise = torch.tensor([-1.7, .3, 2.1], dtype=torch.float64)
    value = oracle.generator_loss(critic, delta, noise)
    actual = torch.autograd.grad(value, delta)[0]
    torch.testing.assert_close(value.detach(), torch.tensor(math.log(2), dtype=torch.float64))
    torch.testing.assert_close(actual, delta.new_tensor(-linear / 2), rtol=0, atol=1e-15)
    # Nonzero odd force atzero rules out a hidden squared-residual objective.
    assert all(p.grad is None for p in critic.parameters())


def test_D_adversarial_parameter_gradients_zero_at_zero_residual():
    critic = oracle.PolynomialCritic(-.2, .4, .7)
    noise = torch.tensor([-2., .1, 1.5], dtype=torch.float64)
    for antithetic in (False, True):
        value = oracle.discriminator_loss(critic, noise.new_zeros(()), noise, antithetic=antithetic)
        gradients = torch.autograd.grad(value, tuple(critic.parameters()))
        assert all(torch.equal(g, torch.zeros_like(g)) for g in gradients)
    assert all(p.grad is None for p in critic.parameters())


@pytest.mark.parametrize("antithetic", [False, True])
def test_nonzero_public_G_loss_and_gradient_match_original_logistic_loss(antithetic):
    critic = oracle.PolynomialCritic(-.3, .12)
    noise = torch.tensor([-1.2, .4, 2.], dtype=torch.float64)
    delta = noise.new_tensor(.35).requires_grad_(True)
    actual_loss = oracle.generator_loss(critic, delta, noise, antithetic=antithetic)
    expected = F.softplus(-(critic(noise + delta) - critic(noise).detach())).mean()
    if antithetic:
        expected = (expected + F.softplus(-(critic(-noise + delta) - critic(-noise).detach())).mean()) / 2
    torch.testing.assert_close(actual_loss, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(actual_loss, delta)[0]
    predicted = oracle.analytic_g_gradient(delta.detach(), noise, -.3, .12, antithetic=antithetic).mean()
    torch.testing.assert_close(gradient, predicted, rtol=1e-14, atol=1e-14)
    assert not torch.isclose(gradient, 2 * delta.detach())


def test_G_references_detach_but_fake_sigma_graph_remains():
    critic = oracle.PolynomialCritic(-.125)
    sigma = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    epsilon = torch.tensor([-1.4, .2, .8], dtype=torch.float64)
    delta = sigma.new_tensor(0.)
    actual = torch.autograd.grad(oracle.generator_loss(critic, delta, sigma * epsilon), sigma)[0]
    expected = .125 * sigma.detach() * epsilon.square().mean()
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual > 0  # Differentiating both equal score references would give0.


def test_initial_D_energy_noise_and_antithetic_gradient_identity():
    noise = torch.tensor([-2., -.3, .7, 1.5], dtype=torch.float64)
    delta = noise.new_tensor(.25)
    one = oracle.initial_energy_gradients(delta, noise)
    paired = oracle.initial_energy_gradients(delta, noise, antithetic=True)
    torch.testing.assert_close(one, .5 * delta.square() + delta * noise, rtol=0, atol=1e-15)
    torch.testing.assert_close(paired, torch.full_like(noise, .5 * .25 ** 2), rtol=0, atol=1e-15)
    assert one.var(unbiased=False) > 0
    assert paired.var(unbiased=False) < 1e-30
    # Also verify the shared-coefficient D gradient, not only per-row formulas.
    critic = oracle.PolynomialCritic(0., 0.)
    actual = torch.autograd.grad(oracle.discriminator_loss(critic, delta, noise), critic.quadratic)[0]
    torch.testing.assert_close(actual, one.mean(), rtol=0, atol=1e-15)


def test_gaussian_population_parity_restoring_sign_and_quadratic_optimum():
    nodes, weights = oracle.normal_quadrature()
    sigma = oracle.SIGMA
    for amplitude in oracle.AMPLITUDES:
        delta = nodes.new_tensor(amplitude)
        one = oracle.analytic_g_gradient(delta, sigma * nodes, -.125, antithetic=False)
        paired = oracle.analytic_g_gradient(delta, sigma * nodes, -.125)
        torch.testing.assert_close((weights * one).sum(), (weights * paired).sum(), atol=1e-14, rtol=1e-14)
        assert (weights * paired).sum() > 0
        # Unregularized free quadratic optimum; this is not a KA2 claim.
        statistic = delta.square() + 2 * sigma * nodes * delta
        derivative = (weights * statistic * torch.sigmoid(-statistic / (2 * sigma ** 2))).sum()
        assert abs(float(derivative)) < 1e-13


@pytest.mark.parametrize("dimension", [1, 16, 128])
def test_actual_public_e22_KA2_A_mean_linear_dimension_law(dimension):
    row = oracle.ka2_mean_dimension_check(dimension)
    assert row["phase"] == "a" and row["penalty_calls"] == 1 and row["optimizer_steps"] == 0
    torch.testing.assert_close(torch.tensor(row["penalty"], dtype=torch.float64),
                               torch.tensor(1.5 / dimension ** 2, dtype=torch.float64), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(torch.tensor(row["slope_gradient"], dtype=torch.float64),
                               torch.tensor(3. / dimension ** 2, dtype=torch.float64), rtol=2e-12, atol=2e-12)


def test_bounded_fixed_stream_report_and_CLI(tmp_path, capsys):
    before = torch.get_rng_state().clone()
    report = oracle.main(["--samples", "512", "--out", str(tmp_path / "receipt.json")])
    stored = json.loads((tmp_path / "receipt.json").read_text())
    assert report == stored and report["status"] == "passed"
    assert report["optimizer_updates"] == 0 and not report["reconstruction_objective"]
    assert report["global_CPU_RNG_unchanged"] and torch.equal(before, torch.get_rng_state())
    assert all(row["MC_parity_within_6SE"] for row in report["nonzero_G_parity"])
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_sample_budget_cannot_be_unbounded():
    with pytest.raises(ValueError, match="Sample count"):
        oracle.calibrate(samples=1_000_000)
