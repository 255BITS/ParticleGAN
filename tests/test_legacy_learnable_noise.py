"""The optional noise scalar is a real G parameter on every custom host."""

import math

import pytest
import torch
from torch import nn

from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, wrap_output


def test_scalar_gradient_belongs_to_generator_only_and_eval_preserves_rng():
    torch.manual_seed(9)
    policy = NoisePolicy(0.029, 0.5, 0.5, 4, output_noise_learnable=True)
    generator = wrap_output(nn.Linear(2, 2), policy)
    discriminator = nn.Linear(2, 1)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    policy.register_generator_optimizer(opt_g, opt_d)
    scalar = policy.output_scale.raw_scale
    policy.set_step(0)
    with policy.discriminator():
        fake_d = generator(torch.ones(16, 2))
    discriminator(fake_d).sum().backward()
    assert scalar.grad is None
    opt_g.zero_grad(set_to_none=True)
    fake_g = generator(torch.ones(16, 2))
    fake_g.sum().backward()
    assert scalar.grad is not None and scalar.grad.abs() > 0
    before = float(policy.output_scale().detach())
    opt_g.step()
    assert float(policy.output_scale().detach()) != before
    torch.manual_seed(23)
    global_state = torch.random.get_rng_state().clone()
    input_state = policy.input_stream.get_state().clone()
    with policy.evaluation(2):
        first = generator(torch.ones(16, 2))
        policy.input(torch.ones(16, 2))
    with policy.evaluation(2):
        second = generator(torch.ones(16, 2))
    assert torch.equal(first, second)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert torch.equal(input_state, policy.input_stream.get_state())


def test_learnable_scalar_multiplies_the_shared_warmup_without_rng_at_zero():
    policy = NoisePolicy(
        0.029, 0.5, 0.5, 10, output_noise_warmup=0.5,
        output_noise_learnable=True,
    )
    base = torch.zeros(8, 2)
    torch.manual_seed(29)
    saved_rng = torch.random.get_rng_state().clone()
    policy.set_step(0)
    assert policy.output(base) is base
    assert torch.equal(saved_rng, torch.random.get_rng_state())
    with torch.no_grad():
        policy.output_scale.raw_scale.add_(0.2)
    learned = float(policy.output_scale().detach())
    policy.set_step(1)
    expected = 0.029 * learned / 0.029 / 5
    assert math.isclose(policy.receipt()["output_sigma_effective_step_trace"][1],
                        expected, rel_tol=1e-6)
    policy.set_step(5)
    assert math.isclose(policy.receipt()["output_sigma_effective_step_trace"][2],
                        learned, rel_tol=1e-6)


@pytest.mark.parametrize("learnable,std", [(True, 0.0), (1, 0.029)])
def test_invalid_legacy_learnable_policy_is_rejected(learnable, std):
    with pytest.raises(ValueError, match="output_noise_learnable"):
        NoisePolicy(std, 0.5, 0.5, 2, output_noise_learnable=learnable)
