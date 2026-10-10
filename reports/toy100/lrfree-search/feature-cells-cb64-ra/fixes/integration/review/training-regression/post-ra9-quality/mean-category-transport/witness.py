"""One fixed clipped conditional mean witness; no production gate or fit sweep."""
import math
from dataclasses import dataclass

import torch

Q = .05
POLICY = "even_group_radial_clip_unit_residual_EB_3K_plus_3_v1"


def group_means(values, groups, count):
    counts = torch.bincount(groups, minlength=count)
    sums = torch.zeros(count, values.shape[1], dtype=torch.float64, device=values.device)
    sums.index_add_(0, groups, values.double())
    return sums / counts.clamp_min(1)[:, None], counts


@dataclass(frozen=True)
class FixedMoment:
    centers: torch.Tensor
    scales: torch.Tensor
    even_means: torch.Tensor
    ema_means: torch.Tensor
    ema_counts: torch.Tensor
    weights: torch.Tensor
    directions: torch.Tensor
    radius: float
    rank: int
    cells: int

    def psi(self, metric, groups):
        z = (metric.double() - self.centers[groups]) / self.scales[groups, None]
        fraction = (self.radius / z.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)
        return z * fraction[:, None]

    def energy(self, means=None):
        means = self.ema_means if means is None else means
        return (self.weights * (self.even_means - means).square().sum(1)).sum()


@torch.no_grad()
def freeze_moment(snapshot, even_features, ema_features):
    """This function has no odd witness input; no group can be score-selected."""
    if (not snapshot.valid_metric or snapshot.rank <= 0
            or snapshot.duplicate_fraction > Q):
        return None, "invalid_or_duplicate_chart"
    if (not bool(torch.isfinite(even_features).all())
            or not bool(torch.isfinite(ema_features).all())):
        return None, "nonfinite_fit_or_EMA"
    topology = snapshot._mass_topology()
    metric = snapshot.transform(even_features)
    even_groups = topology[snapshot._assign_metric(metric)[0]]
    count = snapshot.mass_groups
    centers, even_counts = group_means(metric, even_groups, count)
    if bool((even_counts < 2).any()):
        return None, "missing_or_insufficient_even_group"
    squared = (metric - centers[even_groups]).square().sum(1)
    ss = torch.zeros(count, dtype=torch.float64, device=metric.device)
    ss.index_add_(0, even_groups, squared)
    scales = (ss / even_counts / snapshot.rank).sqrt()
    if not bool(torch.isfinite(scales).all() & (scales > 0).all()):
        return None, "nonfinite_or_zero_even_scale"
    radius = math.sqrt(snapshot.rank / Q)
    # The temporary object uses only even fitted reference geometry.
    z = (metric - centers[even_groups]) / scales[even_groups, None]
    psi_even = z * (radius / z.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:, None]
    even_means, _ = group_means(psi_even, even_groups, count)
    ema_metric = snapshot.transform(ema_features)
    ema_groups = topology[snapshot._assign_metric(ema_metric)[0]]
    ze = (ema_metric - centers[ema_groups]) / scales[ema_groups, None]
    psi_ema = ze * (radius / ze.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:, None]
    ema_means, ema_counts = group_means(psi_ema, ema_groups, count)
    if bool((ema_counts == 0).any()):
        return None, "missing_EMA_group"
    residual = even_means - ema_means
    lengths = residual.norm(dim=1)
    directions = torch.where(lengths[:, None] > 0,
                             residual / lengths.clamp_min(1e-30)[:, None],
                             torch.zeros_like(residual))
    weights = even_counts.double() / int(even_counts.sum())
    if not bool(torch.isfinite(directions).all() & torch.isfinite(ema_means).all()):
        return None, "nonfinite_frozen_direction"
    return FixedMoment(centers, scales, even_means, ema_means, ema_counts,
                       weights, directions, radius, snapshot.rank, snapshot.cells), None


@torch.no_grad()
def odd_witness(snapshot, fixed, odd_features):
    """All odd observations, including zero directions, use the known 4R range."""
    alpha = Q / (3 * snapshot.cells + 3)
    base = dict(policy=POLICY, alpha=alpha, multiplicity=3 * snapshot.cells + 3,
                observations=len(odd_features), authoritative=False,
                limit="conditional iid bounded-score algebra; trained D/shared FIFO empirical evidence only")
    if fixed is None:
        return dict(base, valid=False, fires=False, reason="no_valid_frozen_moment")
    if len(odd_features) <= 1 or not bool(torch.isfinite(odd_features).all()):
        return dict(base, valid=False, fires=False, reason="insufficient_or_nonfinite_odd_rows")
    metric = snapshot.transform(odd_features)
    groups = snapshot._mass_topology()[snapshot._assign_metric(metric)[0]]
    if (bool((fixed.scales[groups] <= 0).any())
            or bool((fixed.ema_counts[groups] == 0).any())):
        return dict(base, valid=False, fires=False, reason="missing_odd_group_definition")
    psi = fixed.psi(metric, groups)
    values = (fixed.directions[groups] * (psi - fixed.ema_means[groups])).sum(1)
    if not bool(torch.isfinite(values).all()):
        return dict(base, valid=False, fires=False, reason="nonfinite_odd_scalar")
    known_range = 4 * fixed.radius
    # Multiplying the theorem for [0,1] by the known range gives this bound.
    variance = values.var(unbiased=True)
    mean = values.mean()
    t = math.log(2 / alpha)
    variance_penalty = (2 * variance * t / len(values)).sqrt()
    range_penalty = (7 / 3) * known_range * t / (len(values) - 1)
    lcb = mean - variance_penalty - range_penalty
    assert bool((values.abs() <= 2 * fixed.radius + 1e-10).all())
    return dict(base, valid=True, fires=bool(lcb > 0), reason=None,
                mean=float(mean), variance_ddof1=float(variance), radius=fixed.radius,
                known_range=known_range, variance_penalty=float(variance_penalty),
                range_penalty=range_penalty, lower_bound=float(lcb),
                zero_direction_observations=int((fixed.directions[groups].norm(dim=1) == 0).sum()),
                scalar_min=float(values.min()), scalar_max=float(values.max()),
                even_mass_objective=float(fixed.energy()))


@torch.no_grad()
def common_count_family(snapshot, features):
    """Fresh raw laws, all decisions at 3K+3; clean-table use is descriptive."""
    old = snapshot.cell_comparison(features)
    cutoff = Q / (3 * snapshot.cells + 3)
    result = dict(multiplicity=3 * snapshot.cells + 3, cutoff=cutoff,
                  family_sizes=[snapshot.cells, 2 * snapshot.cells, 2, 1],
                  count_observations="clean FAST table; not an iid emitted cloud",
                  inferential_authority=False)
    for key in ("mass", "support", "global_support"):
        data = old[key]
        significant = data["pvalues"] <= cutoff
        if not snapshot.valid_metric:
            significant.zero_()
        result[key] = dict(pvalues=data["pvalues"].tolist(),
            difference=data["difference"].tolist(), real_counts=data["real_counts"].tolist(),
            fake_counts=data["fake_counts"].tolist(),
            excess=(significant & (data["difference"] > 0)).tolist(),
            deficit=(significant & (data["difference"] < 0)).tolist())
    return result
