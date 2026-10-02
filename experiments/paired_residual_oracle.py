"""CPU float64 calibration of paired-residual RpGAN gradients, without training.

Run from the repository root:
    python -m experiments.paired_residual_oracle

The explicit polynomial coefficients are analytic fixtures declared through
public initialization. They are not proposed fixed energies for a generator.
This component oracle neither runs the full E22 lifecycle nor establishes
decoded image quality, controller correctness, or Forge qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from particlegan import get_recipe, init
from particlegan.gan_loss import GANLoss

PROBE_SEED = 1143
SIGMA = 1.3
AMPLITUDES = (.05, .2, .8)
LOSS = GANLoss()


class PolynomialCritic(nn.Module):
    """D(x)=w*x²+v*x+b; all coefficients are explicit analytic fixtures."""

    def __init__(self, quadratic=-.125, linear=0., bias=0.):
        super().__init__()
        for name, value in (("quadratic", quadratic), ("linear", linear), ("bias", bias)):
            if not math.isfinite(value):
                raise ValueError("Analytic coefficients must be finite")
            setattr(self, name, nn.Parameter(torch.tensor(value, dtype=torch.float64)))
        init.deterministic_orthogonal_(self, seed=1)

    def forward(self, x):
        return self.quadratic * x.square() + self.linear * x + self.bias


class MeanLinearCritic(nn.Module):
    """D(x)=v*mean_i(x_i), used only for the KA2 pure-A dimension identity."""

    def __init__(self, slope=1.):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope, dtype=torch.float64))
        init.deterministic_orthogonal_(self, seed=1)

    def forward(self, x):
        return self.slope * x.flatten(1).mean(1)


init.register(PolynomialCritic, {name: init.KEEP for name in ("quadratic", "linear", "bias")})
init.register(MeanLinearCritic, {"slope": init.KEEP})


def generator_loss(critic, residual, noise, *, antithetic=True):
    """Original public RpGAN G loss; only each real reference is detached.

    One residual/prediction is reused for ±noise. In particular, fake noise
    keeps its graph when sigma is differentiable. There is no reconstruction
    term and no optimizer or backward/step operation in this module.
    """
    positive = LOSS.g_loss(critic(noise + residual), critic(noise).detach())
    if not antithetic:
        return positive
    negative = LOSS.g_loss(critic(-noise + residual), critic(-noise).detach())
    return (positive + negative) * .5


def discriminator_loss(critic, residual, noise, *, antithetic=False):
    """Both real and fake retain critic parameter graphs in each D loss."""
    positive = LOSS.d_loss(critic(noise), critic(noise + residual))
    if not antithetic:
        return positive
    negative = LOSS.d_loss(critic(-noise), critic(-noise + residual))
    return (positive + negative) * .5


def analytic_g_gradient(residual, noise, quadratic, linear=0., *, antithetic=True):
    """Per-observation derivative, before the public loss's batch mean."""
    def one(z):
        gap = quadratic * (residual.square() + 2 * z * residual) + linear * residual
        return -torch.sigmoid(-gap) * (2 * quadratic * (z + residual) + linear)
    positive = one(noise)
    return (positive + one(-noise)) * .5 if antithetic else positive


def sampled_g_gradients(critic, residual, noise, *, antithetic=True):
    """Differentiate the public loss per observation without accumulating .grad."""
    def one(delta, z):
        return generator_loss(critic, delta, z, antithetic=antithetic)
    return torch.func.vmap(torch.func.grad(one), in_dims=(None, 0))(residual, noise).detach()


def initial_energy_gradients(residual, noise, *, antithetic=False):
    """Public D gradients at the explicit all-zero critic, not a trained D.

    At w=v=b=0, dL_D/dw=.5*(delta²+2*z*delta). Averaging the two original
    D losses cancels that cross term. This identity does not assert the same
    exact cancellation for a nonzero nonlinear critic.
    """
    def one(weight, z):
        positive = LOSS.d_loss(weight * z.square(), weight * (z + residual).square())
        if not antithetic:
            return positive
        negative = LOSS.d_loss(weight * z.square(), weight * (-z + residual).square())
        return (positive + negative) * .5
    zero = noise.new_zeros(())
    return torch.func.vmap(torch.func.grad(one), in_dims=(None, 0))(zero, noise)


def normal_quadrature(order=64):
    """Deterministic standard-normal quadrature; no SciPy, random state or data."""
    if type(order) is not int or not 8 <= order <= 128:
        raise ValueError("Quadrature order must be between 8 and 128")
    off = torch.arange(1, order, dtype=torch.float64).sqrt()
    jacobi = torch.diag(off, 1) + torch.diag(off, -1)
    nodes, vectors = torch.linalg.eigh(jacobi)
    return nodes, vectors[0].square()


def ka2_mean_dimension_check(dimension, *, slope=1.):
    """Measure the real public first-call penalty, not just its algebra.

    For a mean-linear critic with |v|/dimension below the fake cap,
    A=(coeff/2)*v²/dimension². At v=1,d=1 the cap is at its boundary;
    the kernel's norm epsilon can produce a negligible positive cap.
    This is pure-A only: not a CNN normalization law or blended KA2 claim.
    """
    if type(dimension) is not int or not 1 <= dimension <= 65536:
        raise ValueError("Dimension must be between 1 and 65536")
    critic = MeanLinearCritic(slope)
    recipe = get_recipe("e22_routed", batch_size=4, num_particles=16,
                        z_dim=2, output_noise_std=SIGMA)
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
    real = torch.zeros(4, dimension, dtype=torch.float64)
    value = penalty(critic, real, real + .25)
    derivative = torch.autograd.grad(value, critic.slope)[0]
    expected = recipe.reg_coeff * slope ** 2 / (2 * dimension ** 2)
    expected_derivative = recipe.reg_coeff * slope / dimension ** 2
    if penalty.last_stats["phase"] != "a" or optimizer.record.calls != 1:
        raise AssertionError("Expected the native first pure-A call")
    if optimizer.record.observed_steps != 0 or optimizer.state:
        raise AssertionError("Calibration must not perform optimizer steps")
    if penalty.regularizer.record is not optimizer.record or penalty.ema_critic is not optimizer.ema_critic:
        raise AssertionError("Penalty/optimizer EMA ownership differs")
    return dict(dimension=dimension, slope=slope, reg_coeff=recipe.reg_coeff,
                penalty=float(value.detach()), expected_penalty=expected,
                slope_gradient=float(derivative), expected_slope_gradient=expected_derivative,
                phase="a", penalty_calls=1, optimizer_steps=0)


def _require_close(actual, expected, *, atol=2e-12):
    torch.testing.assert_close(torch.as_tensor(actual, dtype=torch.float64),
                               torch.as_tensor(expected, dtype=torch.float64), rtol=2e-12, atol=atol)


def calibrate(*, samples=8192):
    """Bounded fixed-stream calibration, not a seed study or training run."""
    if type(samples) is not int or not 128 <= samples <= 65536:
        raise ValueError("Sample count must be between 128 and 65536")
    global_before = torch.get_rng_state().clone()
    stream = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    epsilon = torch.randn(samples, generator=stream, dtype=torch.float64)
    noise = SIGMA * epsilon
    zero = noise.new_zeros(())
    at_zero = []
    for linear in (0., .0625):
        critic = PolynomialCritic(linear=linear)
        gradients = sampled_g_gradients(critic, zero, noise)
        _require_close(gradients, torch.full_like(gradients, -linear / 2))
        d_value = discriminator_loss(critic, zero, noise)
        d_grads = torch.autograd.grad(d_value, tuple(critic.parameters()))
        for grad in d_grads:
            _require_close(grad, 0., atol=0.)
        at_zero.append(dict(linear=linear, G_antithetic_gradient_mean=float(gradients.mean()),
                            G_antithetic_gradient_variance=float(gradients.var(unbiased=False)),
                            expected_G_force=-linear / 2, D_loss=float(d_value.detach()),
                            D_parameter_gradients=[float(x) for x in d_grads]))
    critic = PolynomialCritic()
    single_zero = sampled_g_gradients(critic, zero, noise, antithetic=False)
    _require_close(single_zero, -float(critic.quadratic.detach()) * noise)
    initial = []
    nodes, weights = normal_quadrature()
    nonzero = []
    for amplitude in AMPLITUDES:
        delta = noise.new_tensor(amplitude)
        one = initial_energy_gradients(delta, noise)
        paired = initial_energy_gradients(delta, noise, antithetic=True)
        _require_close(one, .5 * delta.square() + delta * noise)
        _require_close(paired, torch.full_like(paired, .5 * amplitude ** 2))
        initial.append(dict(amplitude=amplitude, single_mean=float(one.mean()),
                            single_variance=float(one.var(unbiased=False)),
                            expected_mean=.5 * amplitude ** 2,
                            expected_single_variance=SIGMA ** 2 * amplitude ** 2,
                            antithetic_mean=float(paired.mean()),
                            antithetic_variance=float(paired.var(unbiased=False))))
        one_g = sampled_g_gradients(critic, delta, noise, antithetic=False)
        pair_g = sampled_g_gradients(critic, delta, noise)
        _require_close(pair_g, analytic_g_gradient(delta, noise, -.125))
        quadrature_one = (weights * analytic_g_gradient(delta, SIGMA * nodes, -.125, antithetic=False)).sum()
        quadrature_pair = (weights * analytic_g_gradient(delta, SIGMA * nodes, -.125)).sum()
        _require_close(quadrature_one, quadrature_pair)
        if quadrature_pair <= 0:
            raise AssertionError("Even-critic population direction should restore positive residuals")
        difference = one_g - pair_g
        se = difference.std(unbiased=True) / math.sqrt(samples)
        nonzero.append(dict(amplitude=amplitude, G_single_mean=float(one_g.mean()),
                            G_antithetic_mean=float(pair_g.mean()),
                            quadrature_population_gradient=float(quadrature_pair),
                            common_stream_mean_difference=float(difference.mean()),
                            common_stream_difference_SE=float(se),
                            MC_parity_within_6SE=bool(difference.mean().abs() <= 6 * se)))
    sigma = noise.new_tensor(SIGMA).requires_grad_(True)
    sigma_grad = torch.autograd.grad(generator_loss(critic, zero, sigma * epsilon), sigma)[0]
    expected_sigma_grad = .125 * SIGMA * epsilon.square().mean()
    _require_close(sigma_grad, expected_sigma_grad)
    dimensions = [ka2_mean_dimension_check(d) for d in (1, 16, 128)]
    for row in dimensions:
        _require_close(row["penalty"], row["expected_penalty"])
        _require_close(row["slope_gradient"], row["expected_slope_gradient"])
    if not torch.equal(global_before, torch.get_rng_state()):
        raise AssertionError("Calibration consumed the global CPU RNG")
    root = Path(__file__).resolve().parents[1]
    sources = (Path(__file__), root / "particlegan/gan_loss.py", root / "particlegan/init.py",
               root / "particlegan/recipes.py", root / "particlegan/ka2.py")
    return dict(format="paired_residual_analytic_calibration_v1", status="passed",
                scope="CPU float64 component oracle; no trained E22 or decoded-quality claim",
                sigma=SIGMA, samples=samples, fixed_probe_seed=PROBE_SEED,
                initialization="Public init.KEEP analytic coefficients; deterministic_orthogonal_(seed=1)",
                zero_residual=at_zero,
                quadratic_single_zero_variance=dict(measured=float(single_zero.var(unbiased=False)),
                    expected=.125 ** 2 * SIGMA ** 2),
                initial_D_energy=initial, nonzero_G_parity=nonzero,
                fake_only_sigma_chain=dict(measured=float(sigma_grad), expected=float(expected_sigma_grad),
                    note="Detached real references; sigma keeps its fake-branch graph even at zero residual"),
                KA2_A_mean_linear_dimensions=dimensions,
                limitations=["D energy cancellation is at an all-zero critic, not an arbitrary trained nonlinear D",
                    "Unconditional residual subtraction retains pairing; finite critic capacity can still limit optimization",
                    "The mean-linear d^-2 law is not a universal CNN or blended-KA2 scaling law",
                    "Monte Carlo parity is approximate; deterministic quadrature establishes symmetric population parity"],
                optimizer_updates=0, reconstruction_objective=False, global_CPU_RNG_unchanged=True,
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=Path("artifacts/paired-residual-oracle/receipt.json"))
    args = parser.parse_args(argv)
    report = calibrate(samples=args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], out=str(args.out), samples=args.samples,
                         optimizer_updates=0, scope=report["scope"]), allow_nan=False), flush=True)
    return report


if __name__ == "__main__":
    main()
