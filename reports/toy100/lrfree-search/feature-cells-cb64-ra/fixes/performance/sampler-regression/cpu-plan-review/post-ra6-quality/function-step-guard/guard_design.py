"""Private feasibility design. This is not installed in any package.

The bound is an engineering use of the existing Q, not a statistical test.
No query/fake point chooses the real-only action topology or covariance.
"""
import math

import torch


def bounded_fifo_rows(fill, cursor, *, current=128, older=256):
    """Chronological current rows plus evenly spaced older rows; no RNG."""
    if type(fill) is not int or type(cursor) is not int or fill < 6 or not 0 <= cursor < fill:
        raise ValueError('need a filled, valid FIFO')
    ordered = (torch.arange(fill) + cursor).remainder(fill)
    recent_count = min(current, fill)
    old_count = min(older, fill - recent_count)
    old_positions = torch.div(torch.arange(old_count) * (fill - recent_count),
        max(1, old_count), rounding_mode='floor')
    return torch.cat((ordered[old_positions], ordered[-recent_count:]))


def interpolate_parameters(before, proposed, fraction):
    if fraction not in (0., .125, .25, .5, 1.) or before.keys() != proposed.keys():
        raise ValueError('invalid bounded step')
    # The endpoints must retain their original bits; no arithmetic there.
    if fraction == 0.:
        return {name: value.detach().clone() for name, value in before.items()}
    if fraction == 1.:
        return {name: value.detach().clone() for name, value in proposed.items()}
    return {name: value.detach() + (proposed[name].detach() - value.detach()) * fraction
        for name, value in before.items()}


class LocalMotionChart:
    """A current-head bounded chart with within-cell directional geometry.

    Its input snapshot is fitted from at most 384 current-head real rows.
    Cell covariances use only even-reference members of that fitted cell.
    Sparse cells shrink toward the original cell_scale, without fetching
    points from other cells. Topology is the existing real-only MST law.
    """
    def __init__(self, snapshot, real_features, baseline_features, *, q=.05, prior=4.):
        if (len(real_features) > 384 or len(baseline_features) > 128
                or len(baseline_features) == 0 or not snapshot.valid_metric
                or snapshot.rank < 1 or snapshot.duplicate_fraction > q):
            raise ValueError('invalid or insufficient bounded motion chart')
        self.snapshot, self.q = snapshot, q
        self.baseline = snapshot.transform(baseline_features)
        self.cell_ids, _ = snapshot._assign_metric(self.baseline)
        self.real = snapshot.transform(real_features[0::2])
        real_ids, _ = snapshot._assign_metric(self.real)
        rank, k = snapshot.rank, snapshot.cells
        eye = torch.eye(rank, dtype=self.real.dtype, device=self.real.device)
        covariance = []
        for cell in range(k):
            members = self.real[real_ids == cell]
            centered = members - members.mean(0) if len(members) else members
            scatter = centered.T @ centered
            # Existing scalar radius supplies the only small-cell scale.
            prior_cov = snapshot.cell_scale[cell].square() / rank * eye
            covariance.append((scatter + prior * prior_cov) / (max(len(members) - 1, 0) + prior))
        self.covariance = torch.stack(covariance)
        self.cholesky = torch.linalg.cholesky(self.covariance)
        self.anchor = snapshot.real_representatives[self.cell_ids]
        distance = self.norm(self.baseline - self.anchor)
        self.denominator = distance.clamp_min(math.sqrt(rank))
        categories = snapshot.count_categories(baseline_features)
        _, pvalues, _ = snapshot.support(baseline_features)
        self.initial_inside = (pvalues > q) & (categories.remainder(2) == 0)
        self.initial_categories = categories
        groups = snapshot._mass_topology()
        covered = torch.zeros(snapshot.mass_groups, dtype=torch.bool, device=groups.device)
        covered[groups[self.cell_ids[self.initial_inside]]] = True
        group_reference = torch.zeros(snapshot.mass_groups, dtype=torch.long, device=groups.device)
        group_reference.scatter_add_(0, groups, snapshot.reference_counts)
        self.represented_real_mass = float(group_reference[covered].sum()) / len(self.real)
        self.active = self.represented_real_mass >= 1. - q
        self.ordinal = max(0, math.ceil((1. - q) * len(self.baseline)) - 1)
        self.limit = math.sqrt(q)

    def norm(self, displacement):
        solved = torch.linalg.solve_triangular(self.cholesky[self.cell_ids],
            displacement[:, :, None], upper=False)[:, :, 0]
        return solved.square().sum(1).sqrt()

    def inspect(self, features):
        projected = self.snapshot.transform(features)
        ratios = self.norm(projected - self.baseline) / self.denominator
        finite = bool(torch.isfinite(features).all() and torch.isfinite(ratios).all())
        quantile = float(ratios.sort().values[self.ordinal]) if finite else float('inf')
        categories = self.snapshot.count_categories(features)
        _, pvalues, _ = self.snapshot.support(features)
        inside = (pvalues > self.q) & (categories.remainder(2) == 0)
        initial = self.initial_inside
        result = dict(finite=finite, normalized_motion_q95=quantile,
            normalized_motion_max=float(ratios.max()) if finite else None,
            normalized_motion_limit=self.limit, displacement_bound_pass=finite and quantile <= self.limit,
            initial_eligible_inside=int(initial.sum()), eligible_inside=int(inside.sum()),
            lost_initial_inside=int((initial & ~inside).sum()),
            changed_cells=int((self.cell_ids != torch.div(categories, 2, rounding_mode='floor')).sum()),
            changed_categories=int((self.initial_categories != categories).sum()),
            projected_displacement_rms=float((projected - self.baseline).square().sum(1).mean().sqrt()),
            original_cell_scale_q50=float(self.snapshot.cell_scale[self.cell_ids].median()),
            paired_real_distance_q50=float(self.norm(projected - self.anchor).median()))
        return result

    def choose(self, features_at_fraction):
        """At most four model evaluations; first adequate fraction wins.

        Early exploration is neutral until supported probes represent at
        least 1-Q of the real-only topology's reference mass. A failed
        active sequence keeps G fixed; joint Adam state still advanced once.
        """
        attempts = []
        for fraction in (1., .5, .25, .125):
            result = self.inspect(features_at_fraction(fraction))
            attempts.append(dict(fraction=fraction, **result))
            if not self.active or result['displacement_bound_pass']:
                return fraction, attempts
        return 0., attempts
