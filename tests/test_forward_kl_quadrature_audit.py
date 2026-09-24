"""Independent checks of delta integration and the Gaussian tail envelope."""

import math

import torch

from reports.toy100.forward_kl_free_filter import cross_entropy, quadrature
from reports.toy100.forward_kl_quadrature_audit import (
    delta_quadrature, outside_square_bound,
)


def test_independent_log_ratio_delta_matches_original_objectives():
    real=torch.tensor([[0.,1.],[-1.2,.3],[1.,-.6]],dtype=torch.float64)
    old=torch.tensor([[.1,.2],[-.8,.2],[1.1,-.5]],dtype=torch.float64)
    new=torch.tensor([[.2,.3],[-.7,.2],[1.,-.45]],dtype=torch.float64)
    h=.13;v=h*h+.029**2
    for order in (5,9):
        target=quadrature(real,h,order)
        expected=float(cross_entropy(*target,new,v)-cross_entropy(*target,old,v))
        actual=delta_quadrature(real,old,new,h,v,order)
        assert abs(actual-expected)<1e-12


def test_tail_bound_dominates_sampled_pointwise_log_ratio_envelope():
    real=torch.tensor([[3.1,.2],[-2.5,1.]],dtype=torch.float64)
    old=torch.tensor([[2.,1.],[1.,-1.]],dtype=torch.float64)
    new=torch.tensor([[2.2,1.1],[.7,-1.2]],dtype=torch.float64)
    h=.031286;v=h*h+.029**2
    row=outside_square_bound(real,old,new,h,v)
    assert 0<row['absolute_contribution_upper_bound']<1e-8
    slope=row['slope_bound'];offset=row['offset_bound']
    for z in (torch.tensor([8.5,0.]),torch.tensor([-9.,10.])):
        for center in real:
            x=center+h*z
            old_logits=(x@old.T-.5*old.square().sum(1))/v
            new_logits=(x@new.T-.5*new.square().sum(1))/v
            log_ratio=float(torch.logsumexp(old_logits,0)-
                            torch.logsumexp(new_logits,0))
            assert abs(log_ratio)<=slope*float(x.norm())+offset+1e-10
