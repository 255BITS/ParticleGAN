"""Exact chain term and rest for the declared single virtual D update."""
import torch

from reports.toy100.pr84_one_step_unroll import virtual_parameters


def test_general_sum_quadratic_full_chain_differs_from_same_point_partial():
    dtype=torch.float64
    g=torch.tensor(1.3,dtype=dtype,requires_grad=True)
    d=torch.tensor(-.7,dtype=dtype,requires_grad=True)
    p=torch.tensor(.2,dtype=dtype,requires_grad=True)
    point,_=virtual_parameters({'d':d},[p],d*g+.5*d*d)
    v=point['d'];c=.4
    total=torch.autograd.grad(.5*c*g*g-v*g,g,retain_graph=True)[0]
    partial=torch.autograd.grad(.5*c*g*g-v.detach()*g,g)[0]
    assert torch.allclose(total,(c+2*p.detach())*g-(1-p.detach())*d)
    assert torch.allclose(partial,c*g-v.detach())
    assert torch.allclose(total-partial,p.detach()*g)
    assert p.grad is None and d.grad is None
    assert g.grad is None


def test_exact_zero_field_rests_without_persistent_parameter_or_moment_change():
    g=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    d=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    point,gradient=virtual_parameters({'d':d},[torch.tensor(.2)],d*g+.5*d*d)
    loss=-point['d']*g+.5*g*g
    assert gradient[0].item()==0 and point['d'].item()==0
    assert torch.autograd.grad(loss,g)[0].item()==0
    assert g.item()==0 and d.item()==0 and g.grad is None and d.grad is None
