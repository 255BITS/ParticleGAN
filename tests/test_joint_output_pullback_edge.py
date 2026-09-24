"""Independent numerical edge check for the frozen joint output pullback."""

import torch

from reports.toy100.joint_output_pullback import fit_output_targets


class ExponentialGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, latent):
        return torch.exp(self.weight).expand_as(latent)


def test_nonfinite_full_trial_halves_to_finite_descent_without_state_leak():
    generator = ExponentialGenerator()
    prior = torch.nn.Parameter(torch.zeros(1, 1))
    generator.weight.grad = torch.tensor([3.])
    prior.grad = torch.tensor([[4.]])
    random_before = torch.get_rng_state().clone()
    result = fit_output_targets(generator, prior, torch.tensor([[100.]]))
    first = result['records'][0]
    assert first['trials'][0]['alpha'] == 1.
    assert first['trials'][0]['finite'] is False
    assert first['trials'][0]['squared_error'] is None
    assert first['accepted'] is True
    assert first['after'] < first['before']
    assert result['final_squared_error'] < result['initial_squared_error']
    assert torch.isfinite(generator.weight).all()
    assert torch.equal(prior, torch.zeros_like(prior))
    assert torch.equal(generator.weight.grad, torch.tensor([3.]))
    assert torch.equal(prior.grad, torch.tensor([[4.]]))
    assert torch.equal(torch.get_rng_state(), random_before)
