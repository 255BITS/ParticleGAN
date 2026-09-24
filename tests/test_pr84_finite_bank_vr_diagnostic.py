"""Variance cancellation and geometry are logically distinct, with fixed gains."""
import torch

from reports.toy100.pr84_finite_bank_vr_diagnostic import control_variate


def test_finite_bank_root_rests_despite_nonzero_component_fields_and_responds():
    a = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)
    offsets = torch.tensor([[2., -1.], [-2., 1.]], dtype=torch.float64)
    reference = torch.zeros(2, dtype=torch.float64)
    base = reference@a.T+offsets
    assert not torch.equal(base, torch.zeros_like(base))
    corrected = control_variate(base, base, base.mean(0))
    assert torch.equal(corrected, torch.zeros_like(corrected))
    changed = torch.tensor([.3, -.2], dtype=torch.float64)
    response = control_variate(changed@a.T+offsets, base, base.mean(0))
    assert torch.allclose(response, (a@changed).expand_as(response))
    eta = .5
    # Exact deterministic EG for a bilinear rotation, with unchanged gain.
    transition = torch.eye(2, dtype=a.dtype)-eta*a+eta*eta*a@a
    assert torch.allclose((transition@changed).norm()/changed.norm(),
                          torch.tensor((1-eta**2+eta**4)**.5, dtype=a.dtype))
    assert (transition@changed).norm() < changed.norm()


def test_control_variate_is_unbiased_but_does_not_cure_mean_antimonotonicity():
    matrices = torch.tensor([[[-1.]], [[-3.]]], dtype=torch.float64)
    reference, current = torch.tensor([.4]), torch.tensor([.7])
    old = torch.einsum('nij,j->ni', matrices, reference.double())
    new = torch.einsum('nij,j->ni', matrices, current.double())
    corrected = control_variate(new, old, old.mean(0))
    assert torch.allclose(corrected.mean(0), new.mean(0))
    # Mean A=-2: exact EG has gain 1+2 eta+4 eta^2 > 1 for every eta>0.
    eta = .2
    assert 1+2*eta+4*eta*eta > 1


def test_sampled_snapshot_mean_keeps_its_error_even_at_reference():
    reference_components = torch.tensor([[1., 2.], [-1., -2.]], dtype=torch.float64)
    biased_snapshot = torch.tensor([.1, -.2], dtype=torch.float64)
    result = control_variate(reference_components, reference_components, biased_snapshot)
    assert torch.equal(result, biased_snapshot.expand_as(result))
    assert torch.count_nonzero(result) == result.numel()
