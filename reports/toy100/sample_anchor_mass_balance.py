"""Capacity-balanced second-moment targets for the pre-start anchor line.

The pinned pre-start objective keeps one distinct particle per MST group and
then pulls every surplus particle onto its nearest centroid. On the 12-particle
ring that lands on counts [1,1,1,1,1,1,1,5]: mode-mass TV 0.2917, and coincident
particles emit only the output noise 0.029 against target sigma 0.07.

This module changes that one objective. Quotas differ by at most one particle,
so 12 particles and 8 groups take four doubles and four singles (TV 1/6). A
double is placed on the real group's principal axis, symmetric about the
centroid, at the residual scale sqrt(max(0, empirical variance - 0.029^2)),
capped inside the high-quality ball. Groups, axes and scales come from the
current real minibatch. No mode centers and no configured mode count.
"""
import math

import torch
from scipy.optimize import linear_sum_assignment

from benchmarks.locked_shared import mode_hold

SIGMA_OUT = 0.029
HQ_RADIUS = 3.0 * mode_hold.SIGMA
OFFSET_CAP = 2.0 * mode_hold.SIGMA


def assign_balanced(support, centers):
    """Assign every particle so group counts differ by at most one."""
    y = support.detach().double()
    c = centers.detach().double()
    n, k = len(y), len(c)
    if k <= 0 or n <= k:
        raise ValueError('balanced anchors need strictly more particles than groups')
    base, extra = divmod(n, k)
    groups = [[] for _ in range(k)]
    slots = c.repeat(base, 1)
    slot_group = [i % k for i in range(base * k)]
    cost = torch.cdist(slots, y).square().cpu().numpy()
    slot_idx, particle_idx = linear_sum_assignment(cost)
    used = set()
    for slot, particle in zip(slot_idx, particle_idx):
        groups[slot_group[int(slot)]].append(int(particle))
        used.add(int(particle))
    if extra:
        rest = [i for i in range(n) if i not in used]
        extra_cost = torch.cdist(y[rest], c).square().cpu().numpy()
        row, col = linear_sum_assignment(extra_cost)
        if len(row) != extra:
            raise RuntimeError('extra-particle assignment did not cover the surplus')
        for particle_row, group in zip(row, col):
            groups[int(group)].append(rest[int(particle_row)])
    if sorted(p for row in groups for p in row) != list(range(n)):
        raise RuntimeError('balanced assignment is not a partition of the particles')
    counts = [len(row) for row in groups]
    if max(counts) - min(counts) > 1:
        raise RuntimeError('balanced assignment counts differ by more than one')
    return groups


def _axis_offset(members):
    x = members.detach().double()
    if len(x) < 2:
        return torch.zeros(x.shape[-1], dtype=torch.float64)
    centered = x - x.mean(0)
    cov = centered.transpose(0, 1) @ centered / len(x)
    evals, evecs = torch.linalg.eigh(cov)
    axis = evecs[:, -1]
    if float(axis[0]) < 0 or (float(axis[0]) == 0. and float(axis[1]) < 0):
        axis = -axis
    scale = math.sqrt(max(0.0, float(evals[-1]) - SIGMA_OUT ** 2))
    scale = min(scale, OFFSET_CAP)
    return axis * scale


def balanced_targets(support, centers, member_indices, real):
    """One active target per particle: balanced quotas, pair spread on doubles."""
    y = support.detach().double()
    c = centers.detach().double()
    groups = assign_balanced(y, c)
    target = torch.empty_like(y)
    scales = []
    for group, particles in enumerate(groups):
        offset = _axis_offset(real[member_indices[group]].double())
        scales.append(float(offset.norm()))
        if len(particles) == 2 and float(offset.norm()) > 0:
            pair = y[particles]
            poles = torch.stack((c[group] + offset, c[group] - offset))
            cost = torch.cdist(pair, poles).square().cpu().numpy()
            row, col = linear_sum_assignment(cost)
            for particle_row, pole in zip(row, col):
                target[particles[int(particle_row)]] = poles[int(pole)]
        else:
            target[particles] = c[group]
    return dict(target=target, groups=groups,
                counts=[len(row) for row in groups],
                offset_norms=scales)


def balanced_field(support, centers, member_indices, real):
    """Squared distance to the balanced targets of this support. Zero at the target."""
    built = balanced_targets(support, centers, member_indices, real)
    y = support.detach().double()
    residual = y - built['target']
    total = residual.square().sum(-1).mean()
    return dict(total=float(total.detach()),
                counts=built['counts'],
                offset_norms=built['offset_norms'],
                occupied=sum(count > 0 for count in built['counts']),
                max_target_offset=max(built['offset_norms']) if built['offset_norms'] else 0.0)


def balanced_mm_step(support, centers, member_indices, real):
    before = balanced_field(support, centers, member_indices, real)
    built = balanced_targets(support, centers, member_indices, real)
    after = balanced_field(built['target'], centers, member_indices, real)
    if after['total'] > 1e-8:
        raise AssertionError('balanced targets are not a fixed point of the objective')
    if after['total'] > before['total'] + 1e-10:
        raise AssertionError('balanced target increased the objective')
    return dict(before=before['total'], after=after['total'],
                target=built['target'].tolist(),
                counts=built['counts'],
                offset_norms=built['offset_norms'],
                max_output_displacement=float((built['target'] - support.detach().double()).norm(dim=1).max()))
