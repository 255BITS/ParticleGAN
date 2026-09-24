from copy import deepcopy

import pytest
import torch

from reports.toy100.continuous_lookahead import JointLookahead, lookahead
from benchmarks import learned_lr_evaluation as bridge


def test_joint_formula_retains_moments_and_preserves_rng():
    d, g = (torch.nn.Parameter(torch.tensor([x], dtype=torch.float64)) for x in (1., 2.))
    od, og = (torch.optim.Adam([p], lr=.1, betas=(0., .9)) for p in (d, g))
    dref, gref = (torch.nn.Parameter(p.detach().clone()) for p in (d, g))
    rd, rg = (torch.optim.Adam([p], lr=.1, betas=(0., .9)) for p in (dref, gref))
    ordinary = torch.optim.Adam.step
    rng = torch.get_rng_state().clone()
    with lookahead(2, .5) as receipt:
        bridge.optimizer_role(od, {"opt_d": od})
        bridge.optimizer_role(og, {"opt_g": og})
        for _ in range(2):
            for p, ref, opt, plain in ((d,dref,od,rd), (g,gref,og,rg)):
                p.grad = torch.ones_like(p)
                ref.grad = p.grad.clone()
                ordinary(plain)
                opt.step()
    assert d.item() == pytest.approx((1+dref.item())/2)
    assert g.item() == pytest.approx((2+gref.item())/2)
    for opt, plain, p, ref in ((od,rd,d,dref), (og,rg,g,gref)):
        assert torch.equal(opt.state[p]["exp_avg_sq"],plain.state[ref]["exp_avg_sq"])
        assert opt.state[p]["step"] == 2
    assert torch.equal(rng,torch.get_rng_state())
    assert len(receipt["synchronizations"]) == 1


def test_alpha_one_is_bitwise_identity():
    parameters = [torch.nn.Parameter(torch.tensor([1.,-2.])) for _ in range(2)]
    optimizers = [torch.optim.Adam([p],lr=.01) for p in parameters]
    controller = JointLookahead(1,1.)
    for role,opt in zip(("d","g"),optimizers):
        controller.register(opt,role)
    before=[]
    def ordinary(opt):
        result=torch.optim.Adam.step(opt)
        before.append(opt.param_groups[0]["params"][0].detach().clone())
        return result
    for p,opt in zip(parameters,optimizers):
        p.grad=torch.ones_like(p)
        controller.step(opt,ordinary)
    assert all(torch.equal(p,b) for p,b in zip(parameters,before))


def test_bilinear_joint_interpolation_contracts_without_changing_equilibrium():
    # Alternating GDA on min_x max_y xy has a unit-circle orbit. The joint
    # interpolation changes that orbit while leaving its zero fixed.
    matrix=torch.tensor([[1-.3**2,-.3],[.3,1.]],dtype=torch.float64)
    joint=.5*torch.eye(2,dtype=torch.float64)+.5*torch.linalg.matrix_power(matrix,5)
    assert torch.linalg.eigvals(matrix).abs().max().item() == pytest.approx(1.)
    assert torch.linalg.eigvals(joint).abs().max().item() < .8
    assert torch.equal(joint@torch.zeros(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64))
