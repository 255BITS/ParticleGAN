"""Initial function parity and effective step of the scratch fan-in probe."""

from copy import deepcopy
import math

import torch
from torch import nn

from benchmarks.toy100.train import _init_linear, make_trainer, resolve_config
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from reports.toy100.accuracy_fanin_probe import (
    FanInLinear, make_equalized_trainer, parameterize_linears, parity_receipt,
)


def test_initial_xavier_functions_and_input_gradients_match_without_rng_use():
    torch.manual_seed(1234)
    generator = SimpleMLPGenerator(z_dim=4, hidden_dim=128, n_hidden=3)
    discriminator = SimpleMLPDiscriminator(
        in_dim=2, hidden_dim=128, n_hidden=3, fourier=2,
    )
    _init_linear(generator)
    _init_linear(discriminator)
    originals = (deepcopy(generator), deepcopy(discriminator))
    rng_before = torch.random.get_rng_state().clone()
    parameterize_linears(generator)
    parameterize_linears(discriminator)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    for original, candidate, inputs in (
        (originals[0], generator, torch.linspace(-1, 1, 124).reshape(31, 4)),
        (originals[1], discriminator, torch.linspace(-1, 1, 62).reshape(31, 2)),
    ):
        row = parity_receipt(original, candidate, inputs)
        assert row["layers"] == 4
        assert row["max_initial_bias_error"] == 0
        assert row["max_initial_weight_error"] <= 3e-8
        assert row["max_initial_output_error"] <= 2e-6
        assert row["max_initial_input_gradient_error"] <= 2e-6


def test_declared_optimizer_rates_stay_identical_and_weight_step_is_fanin_scaled():
    config, recipe = resolve_config({
        "steps": 3, "seed": 23, "batch_size": 16, "num_particles": 64,
        "g_hidden": 16, "d_hidden": 16, "n_hidden": 1,
        "lr": .00425, "d_lr_mult": 1.0, "prior_lr_mult": 2.0,
        "betas": [0.0, .99], "output_noise_std": .029,
    })
    baseline = make_trainer(config, recipe)
    for change_d in (False, True):
        candidate = make_equalized_trainer(config, recipe, parameterize_d=change_d)
        assert candidate.initial_lrs == baseline.initial_lrs == [
            [.00425, .0085], [.00425],
        ]
        inputs = torch.linspace(-1, 1, 64).reshape(16, 4)
        torch.testing.assert_close(
            candidate.G.model(inputs), baseline.G.model(inputs), atol=2e-6, rtol=0,
        )
        assert torch.equal(candidate.prior.z, baseline.prior.z)

    linear = nn.Linear(128, 1)
    nn.init.xavier_uniform_(linear.weight)
    direct = deepcopy(linear)
    scaled = FanInLinear(linear)
    before_direct, before_scaled = direct.weight.detach().clone(), scaled.weight.detach().clone()
    opt_direct = torch.optim.Adam(direct.parameters(), lr=.00425, betas=(0.0, .99))
    opt_scaled = torch.optim.Adam(scaled.parameters(), lr=.00425, betas=(0.0, .99))
    direct.weight.sum().backward()
    scaled.weight.sum().backward()
    opt_direct.step()
    opt_scaled.step()
    movement_direct = (direct.weight - before_direct).abs().mean().item()
    movement_scaled = (scaled.weight - before_scaled).abs().mean().item()
    assert math.isclose(movement_scaled / movement_direct, 1 / math.sqrt(128), rel_tol=1e-5)
