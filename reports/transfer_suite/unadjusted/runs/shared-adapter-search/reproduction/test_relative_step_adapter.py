from copy import deepcopy

import pytest
import torch

from benchmarks.transfer_suite.relative_step_adapter import adapted_steps, mechanism


def test_identity_preserves_parameters_and_adam_state_exactly():
    a = torch.nn.Parameter(torch.tensor([.2, -.4, 0.]))
    b = torch.nn.Parameter(a.detach().clone())
    plain = torch.optim.Adam([a], lr=.00425, betas=(0., .99))
    wrapped = torch.optim.Adam([b], lr=.00425, betas=(0., .99))
    evidence = {}
    for gradient in ([.1, 4., 0.], [0., -2., 1.], [.2, .3, -.5]):
        a.grad = torch.tensor(gradient)
        b.grad = a.grad.clone()
        plain.step()
        with adapted_steps(mechanism(None), evidence):
            wrapped.step()
        assert torch.equal(a, b)
        for key, value in plain.state[a].items():
            assert torch.equal(value, wrapped.state[b][key])


def test_cap_uses_common_relative_bound_and_leaves_moments_unchanged():
    old = [torch.zeros(4), torch.tensor([2., -3.])]
    plain_parameters = [torch.nn.Parameter(x.clone()) for x in old]
    capped_parameters = [torch.nn.Parameter(x.clone()) for x in old]
    plain = torch.optim.Adam(plain_parameters, lr=.2, betas=(0., .99))
    capped = torch.optim.Adam(capped_parameters, lr=.2, betas=(0., .99))
    for a, b in zip(plain_parameters, capped_parameters):
        a.grad = torch.ones_like(a)
        b.grad = a.grad.clone()
    plain.step()
    evidence = {}
    with adapted_steps(mechanism(.01), evidence):
        capped.step()
    for original, a, b in zip(old, plain_parameters, capped_parameters):
        proposal = a.detach() - original
        bound = .01 * max(float(original.square().mean().sqrt()), .1)
        factor = min(1., bound / (float(proposal.square().mean().sqrt()) + 1e-12))
        assert torch.equal(b, original + proposal * factor)
        assert float((b.detach() - original).square().mean().sqrt()) <= bound + 1e-7
        for key, value in plain.state[a].items():
            assert torch.equal(value, capped.state[b][key])


def test_reporting_role_cannot_change_updates():
    outputs = []
    for role in ('g', 'd', 'prior', 'arbitrary'):
        p = torch.nn.Parameter(torch.tensor([.1, -.3]))
        opt = torch.optim.Adam([p], lr=.1, betas=(0., .99))
        opt._relative_step_report_role = role
        p.grad = torch.tensor([1., 2.])
        with adapted_steps(mechanism(.025), {}):
            opt.step()
        outputs.append(p.detach().clone())
    assert all(torch.equal(outputs[0], p) for p in outputs[1:])


def test_rejects_undeclared_mechanism_and_invalid_fraction():
    with pytest.raises(ValueError):
        mechanism(float('nan'))
    card = deepcopy(mechanism(.01))
    card['parameter_rms_floor'] = 2.
    with pytest.raises(ValueError), adapted_steps(card, {}):
        pass
