"""Pure-data checks for held-out discovery geometry and error spending."""

import hashlib
import math

import pytest
import torch

from reports.toy100.sample_group_mass_discovery import (
    AnytimeMassCertificate, Region, propose_region)


def test_freezes_a_disjoint_region_and_uses_only_future_full_banks():
    refs = torch.tensor([[-2., 0.], [2., 0.]], dtype=torch.float64)
    proposal = torch.tensor([[-.07, 3.01], [.04, 3.], [.01, 2.96],
                             [-.02, 3.06], [.03, 3.02], [-.01, 2.98]],
                            dtype=torch.float32)
    region, row = propose_region(proposal, refs, 2., bank_id=5)
    assert row['status'] == 'REGION_FROZEN'
    assert region.radius > 0
    assert all(math.dist(region.center, tuple(x.tolist())) > 2 + region.radius
               for x in refs)
    test = AnytimeMassCertificate(region, candidate_index=1,
                                  n_particles=12, delta=.01)
    with pytest.raises(RuntimeError, match='reused'):
        test.observe(proposal, bank_id=6)
    with pytest.raises(RuntimeError, match='consecutive'):
        test.observe(torch.zeros(128, 2), bank_id=7)
    bank = torch.tensor(region.center, dtype=torch.float32).repeat(128, 1)
    result = test.observe(bank, bank_id=6)
    assert result['admitted'] and result['validation_samples'] == 128
    assert result['candidate_error_allocation'] == .005
    assert result['lower_mass_bound'] > 1/12


def test_singleton_tail_and_interior_mass_have_no_rejected_region():
    refs = torch.tensor([[3., 0.], [-3., 0.]], dtype=torch.float64)
    interior = torch.tensor([[2.12, 2.12]], dtype=torch.float32).repeat(128, 1)
    region, row = propose_region(interior, refs, 3., bank_id=1)
    assert region is None and row['rejected_count'] == 0
    assert row['status'] == 'TOO_FEW_REJECTED_TO_GROUP'

    closer = torch.tensor([[3.01, .02]], dtype=torch.float32).repeat(127, 1)
    singleton = torch.cat((closer, torch.tensor([[7., 0.]])), dim=0)
    region, row = propose_region(singleton, refs, 3., bank_id=2)
    assert region is None and row['rejected_count'] == 1
    assert row['status'] == 'TOO_FEW_REJECTED_TO_GROUP'


def test_countable_allocation_and_nonadmission_of_zero_mass():
    digest = hashlib.sha256(b'fixed').hexdigest()
    region = Region((0., 0.), .1, 1, digest, digest, 5, 5)
    assert sum(.01/(j*(j+1)) for j in range(1,10001)) < .01
    test = AnytimeMassCertificate(region, candidate_index=2,
                                  n_particles=12, delta=.01)
    assert test.alpha == pytest.approx(.01/6)
    for bank_id in range(2, 30):
        result = test.observe(torch.full((128, 2), float(bank_id)), bank_id=bank_id)
        assert not result['admitted']
        assert result['hits'] == 0
        assert result['lower_mass_bound'] == 0
