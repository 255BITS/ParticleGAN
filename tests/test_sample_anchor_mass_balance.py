"""Free-output check: balanced quotas beat the pinned surplus collapse."""
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.sample_anchor_mass_balance import (
    HQ_RADIUS, SIGMA_OUT, assign_balanced, balanced_mm_step, balanced_targets)
from reports.toy100.sample_group_anchor import mst_groups


def _bank(n=128):
    means = mode_hold.ring_means()
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(means), (n,), generator=gen)
    return means[idx] + mode_hold.SIGMA * torch.randn(n, 2, generator=gen)


def test_twelve_on_eight_is_four_doubles_and_lower_tv():
    real = _bank()
    centers, grouping = mst_groups(real)
    assert len(centers) == 8
    means = mode_hold.ring_means()
    # Pinned defect: five particles on mode 0, one on each other mode.
    support = torch.cat((means[0].repeat(5, 1), means[1:]), 0).double()
    built = balanced_targets(support, centers, grouping['member_indices'], real)
    counts = sorted(built['counts'])
    assert counts == [1, 1, 1, 1, 2, 2, 2, 2]
    step = balanced_mm_step(support, centers, grouping['member_indices'], real)
    assert step['after'] <= 1e-8
    assert step['before'] > step['after']
    target = torch.tensor(step['target']).double()
    which = torch.cdist(target, means.double()).argmin(1)
    masses = torch.bincount(which, minlength=8).double() / len(target)
    tv = float((masses - 0.125).abs().sum() / 2)
    assert abs(tv - 1 / 6) < 1e-6
    offsets = target - means[which].double()
    assert float(offsets.norm(dim=1).max()) < HQ_RADIUS
    doubled = [group for group, count in enumerate(torch.bincount(which, minlength=8).tolist()) if count == 2]
    assert doubled
    spread = (offsets[which == doubled[0]].square().mean() + SIGMA_OUT ** 2).sqrt()
    assert float(spread) > SIGMA_OUT + 1e-4


def test_assignment_is_a_partition():
    centers = mode_hold.ring_means().double()
    support = torch.randn(12, 2)
    groups = assign_balanced(support, centers)
    assert sorted(p for row in groups for p in row) == list(range(12))
    assert max(len(row) for row in groups) - min(len(row) for row in groups) <= 1
