"""Feature-cell constants, finite-resolution feasibility and count tests.

Shared by the feature-cell snapshot, transport planner and controller. Every
function here is deterministic and draws no random numbers.
"""
import math
from fractions import Fraction

import torch

LLOYD_PASSES = 4
POWER_PASSES = 4
SCALE_PRIOR = 4.
PARENT_RESERVOIR = 64
REAL_ANCHORS = 4
PARENT_RANK_CAP = 16
JITTER_STD = .025
JITTER_CAP = .05
Q = .05

CELL_RESOLUTION_POLICY = "even_fit_average_rows_per_effective_rank_floor1_v1"


def population_policy(n):
    """Whether finite conformal/BH resolution permits the existing guard.

    An N-row FIFO leaves floor(N/2) independent calibration rows. A flagged
    set needs ceil(N / (Q*(M+1))) rows, while the guard permits floor(Q*N).
    Use exact rational arithmetic: this is a feasibility rule, not a cutoff
    fitted to task quality.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("population must be a positive integer")
    q = Fraction(str(Q))
    calibration_rows = n // 2
    denominator = q.numerator * (calibration_rows + 1)
    minimum_flags = (n * q.denominator + denominator - 1) // denominator
    allowed_flags = n * q.numerator // q.denominator
    feasible = minimum_flags <= allowed_flags
    return dict(schema=1, requested_backend="feature_cells", population=n,
                q=Q, calibration_rows=calibration_rows,
                minimum_bh_flags=minimum_flags, maximum_guard_flags=allowed_flags,
                finite_resolution_feasible=feasible,
                actual_backend="feature_cells" if feasible else "knn",
                matching_sampler="feature_cells" if feasible else "controller_reference",
                rule="ceil(N/[Q*(floor(N/2)+1)]) <= floor(Q*N)")


def _fit_cell_count(requested_cells, fitted_rows, effective_rank):
    """Average finite-fit resolution cap, not a per-cell occupancy guarantee.

    Rank zero uses denominator one; its metric/discoveries remain disabled.
    Only even fitted rows and their effective metric rank determine the cap.
    """
    return min(requested_cells, max(1, fitted_rows // max(1, effective_rank)))


def q_order_statistic(rows):
    """1-based ordinal of the exact ceil((1-Q)*rows) fitted order statistic."""
    q = Fraction(str(Q))
    return (rows * (q.denominator - q.numerator) + q.denominator - 1) // q.denominator


def move_budget(rows, max_moves=None):
    """Ordinary move budget floor(Q*rows), optionally capped by ``max_moves``."""
    budget = math.floor(Q * rows)
    return budget if max_moves is None else min(int(max_moves), budget)


def certified_capacity(rows, difference, certified):
    """Whole rows of certified positive count difference (zero if uncertified)."""
    return torch.floor(rows * difference.clamp_min(0.) + 1e-10).long() * certified


# Validators -------------------------------------------------------------------

def require_positive_ints(**values):
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def check_flags(flags, rows, device):
    if flags.shape != (rows,) or flags.dtype != torch.bool or flags.device != device:
        raise ValueError("invalid isolation flags")


def _features_ok(x, *, minimum_rows=1):
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or len(x) < minimum_rows
            or x.shape[1] < 1 or not x.is_floating_point()
            or not bool(torch.isfinite(x).all())):
        raise ValueError("feature cells require finite floating [rows, learned_features] tensors")


# Integer allocation -----------------------------------------------------------

def _integer_allocate(capacity, total):
    """Largest remainders, always <=integer capacity, with stable cell ties."""
    total = min(int(total), int(capacity.sum()))
    if total <= 0:
        return torch.zeros_like(capacity)
    quota = capacity.double() * (total / int(capacity.sum()))
    allocated = quota.floor().long()
    remaining = total - int(allocated.sum())
    if remaining:
        order = (quota - allocated).argsort(descending=True, stable=True)
        allocated[order[:remaining]] += 1
    return allocated


def _group_integer_allocate(capacity, groups, totals):
    """Same stable largest remainders for every disjoint group, in one batch."""
    members = groups[None] == torch.arange(len(totals), device=groups.device)[:, None]
    capacities = capacity[None] * members
    sums = capacities.sum(1)
    requested = torch.minimum(totals, sums).clamp_min(0)
    quota = capacities.double() * (requested.double() / sums.clamp_min(1).double())[:, None]
    allocated = quota.floor().long()
    remaining = requested - allocated.sum(1)
    order = (quota - allocated).argsort(dim=1, descending=True, stable=True)
    slots = torch.arange(len(capacity), device=capacity.device)[None]
    increments = (slots < remaining[:, None]).long()
    allocated += torch.zeros_like(allocated).scatter(1, order, increments)
    return allocated.sum(0)


# Exact conditional count tests -----------------------------------------------

def conditional_count_pvalues(real_counts, fake_counts, real_rows, fake_rows, work=None):
    """Two-sided exact conditional Hypergeom tests, in float64 log arithmetic.

    Validity is conditional on a fixed partition and independent categorical
    draws. Repeated adaptive FIFO/model snapshots have no cumulative guarantee.
    Enumeration is <=K*(min(real_rows,fake_rows)+1), never reference-by-reference.
    """
    if (not isinstance(real_counts, torch.Tensor) or not isinstance(fake_counts, torch.Tensor)
            or real_counts.shape != fake_counts.shape or real_counts.ndim != 1
            or real_counts.dtype != torch.long or fake_counts.dtype != torch.long
            or real_counts.device != fake_counts.device or real_rows <= 0 or fake_rows <= 0):
        raise ValueError("invalid categorical counts")
    invalid = torch.stack(((real_counts < 0).any(), (fake_counts < 0).any(),
                           real_counts.sum() != real_rows, fake_counts.sum() != fake_rows))
    if bool(invalid.any()):
        raise ValueError("invalid categorical counts")
    device = real_counts.device
    factorial = torch.lgamma(
        torch.arange(real_rows + fake_rows + 1, device=device, dtype=torch.float64) + 1)
    total = real_counts + fake_counts
    low = (total - fake_rows).clamp_min(0)
    high = total.clamp_max(real_rows)
    lengths = high - low + 1
    # One bulk metadata transfer replaces the per-cell CUDA scalar reads.
    max_length, terms = torch.stack((lengths.max(), lengths.sum())).cpu().tolist()
    offsets = torch.arange(max_length, device=device, dtype=torch.long)[None, :]
    valid = offsets < lengths[:, None]
    # Clamp padding to a valid support endpoint before factorial indexing.
    j = torch.minimum(low[:, None] + offsets, high[:, None])
    log_probability = (factorial[real_rows] - factorial[j] - factorial[real_rows - j]
                       + factorial[fake_rows] - factorial[total[:, None] - j]
                       - factorial[fake_rows - total[:, None] + j]
                       - factorial[real_rows + fake_rows] + factorial[total, None]
                       + factorial[real_rows + fake_rows - total, None])
    observed_log_probability = log_probability.gather(1, (real_counts - low)[:, None])
    # Same two-sided conditional test and mathematical equality tolerance.
    selected = valid & (log_probability <= observed_log_probability + 1e-10)
    result = torch.logsumexp(log_probability.masked_fill(~selected, -float("inf")), 1)
    result = result.exp().clamp_max(1.)
    if work is not None:
        work["count_test_terms"] += terms
    return result


def count_family(real_counts, fake_counts, real_rows, fake_rows, cutoff, valid_metric, work):
    """One count-test family: p-values, signed differences and significance."""
    pvalues = conditional_count_pvalues(real_counts, fake_counts, real_rows, fake_rows, work)
    difference = fake_counts.double() / fake_rows - real_counts.double() / real_rows
    significant = pvalues <= cutoff
    if not valid_metric:
        significant.zero_()
    return dict(real_counts=real_counts, fake_counts=fake_counts, pvalues=pvalues,
                difference=difference, excess=significant & (difference > 0),
                deficit=significant & (difference < 0))
