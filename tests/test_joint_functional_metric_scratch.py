import torch
import pytest
from reports.toy100.joint_functional_metric_scratch import JointFunctionalMetric


def setup(observe=False):
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.4, -.2]], dtype=torch.float64))
    z = torch.nn.Parameter(torch.tensor([[.2, .3]], dtype=torch.float64))
    opt = torch.optim.Adam([dict(params=list(model.parameters()), _comparison_prior=False),
                            dict(params=[z], _comparison_prior=True)], lr=.1, betas=(0., .9))
    controller = JointFunctionalMetric(observe_only=observe)
    controller.register(model)
    return model, z, opt, controller


def test_joint_step_matches_independent_primal_system():
    model, z, opt, controller = setup()
    before = torch.cat([model.weight.detach().flatten(), z.detach().flatten()])
    model(z).sum().backward()
    grad = torch.cat([model.weight.grad.flatten(), z.grad.flatten()])
    p = .1/(grad.abs()+1e-8)
    j = torch.cat([z.detach().flatten(), model.weight.detach().flatten()])[None, :]
    expected = torch.linalg.solve(torch.diag(1/p)+j.T@j/.029, -grad)
    rng = torch.get_rng_state().clone()
    controller.step(opt, torch.optim.Adam.step)
    actual = torch.cat([model.weight.detach().flatten(), z.detach().flatten()])-before
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert all(int(s['step']) == 1 for s in opt.state.values())
    assert torch.equal(rng, torch.get_rng_state())
    controller.close()


def test_observation_only_matches_adam_for_multiple_updates():
    a = setup(True)
    b = setup(True)
    for _ in range(3):
        for model, z, opt, controller in (a, b):
            opt.zero_grad()
            model(z).square().sum().backward()
        a[3].step(a[2], torch.optim.Adam.step)
        b[2].step()
        assert torch.equal(a[0].weight, b[0].weight)
        assert torch.equal(a[1], b[1])
    a[3].close(); b[3].close()


def test_latents_that_do_not_identify_a_particle_are_rejected_before_update():
    model, z, opt, controller = setup()
    model(z + .1).sum().backward()
    with pytest.raises(ValueError, match='exactly identify'):
        controller.step(opt, torch.optim.Adam.step)
    assert len(opt.state) == 0
    controller.close()
