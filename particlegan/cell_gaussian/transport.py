"""Unregularized, detached fractional-capacity assignment and weighted moments."""
from functools import lru_cache
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@lru_cache(maxsize=32)
def constraints(rows, cells):
    index = np.arange(rows * cells)
    return coo_matrix((np.ones(2 * len(index)),
                       (np.r_[index // cells, rows + index % cells], np.r_[index, index])),
                      shape=(rows + cells, rows * cells)).tocsr()


@torch.no_grad()
def solve(cost, capacities):
    rows, cells = cost.shape
    capacity = capacities.detach().double().cpu().numpy().copy()
    capacity[-1] = rows - capacity[:-1].sum()
    if not np.isfinite(capacity).all() or not (capacity > 0).all():
        raise ValueError('Nonpositive or nonfinite fractional capacity')
    values = cost.detach().double().cpu().numpy()
    result = linprog(values.ravel(), A_eq=constraints(rows, cells),
                     b_eq=np.r_[np.ones(rows), capacity], bounds=(0, None),
                     method='highs-ds', options={'dual_feasibility_tolerance': 1e-9,
                                                'primal_feasibility_tolerance': 1e-9})
    if not result.success:
        raise ValueError('Fractional transport solve failed: ' + result.message)
    plan = result.x.reshape(rows, cells)
    row_error = float(np.abs(plan.sum(1) - 1).max())
    col_error = float(np.abs(plan.sum(0) - capacity).max())
    if not np.isfinite(plan).all() or plan.min() < -1e-9 or max(row_error, col_error) > 1e-7:
        raise ValueError('Invalid fractional transport marginals')
    return torch.as_tensor(plan, dtype=cost.dtype, device=cost.device), dict(
        transport_row_error=row_error, transport_column_error=col_error,
        transport_split_rows=int(((plan > 1e-9).sum(1) > 1).sum()),
        transport_solver_objective=float(result.fun),
        transport_capacities=capacity.tolist())


def weighted_loss(mean, covariance, reference, plan):
    dimension = mean.shape[-1]
    capacities = plan.sum(0)
    if not bool((capacities > 0).all()):
        raise ValueError('Empty fractional cell')
    center = plan.T @ mean / capacities[:, None]
    residual = mean[:, None, :] - center[None, :, :]
    mixture = torch.einsum('nc,ncij->cij', plan,
                          covariance[:, None] + residual[..., :, None] * residual[..., None, :])
    mixture = mixture / capacities[:, None, None]
    mixture = mixture + reference['jitter'] * torch.eye(dimension, device=mean.device, dtype=mean.dtype)
    sign, logdet = torch.linalg.slogdet(mixture)
    if not bool((sign.detach() > 0).all()):
        raise ValueError('Nonpositive fractional mixture covariance')
    delta = center - reference['means']
    placement = torch.einsum('ci,cij,cj->c', delta, reference['inverse'], delta) / dimension
    shape = (torch.einsum('cij,cji->c', reference['inverse'], mixture) - logdet + reference['logdet'] - dimension) / dimension
    weights = capacities / len(mean)
    parts = dict(placement=(weights * placement).sum(), shape=(weights * shape).sum())
    return parts['placement'] + parts['shape'], parts


def fractional_loss(mean, covariance, reference):
    from .legacy_cells import moment_cost
    cost, _, _ = moment_cost(mean, covariance, reference)
    expected = reference['counts'].double() / reference['counts'].double().sum() * len(mean)
    plan, diagnostics = solve(cost, expected)
    value, parts = weighted_loss(mean, covariance, reference, plan)
    diagnostics.update(transport_plan=plan.detach().double().cpu().tolist(),
                       assignment_label_semantics='transport_row_argmax_not_hard_quota')
    return value, parts, plan.argmax(1), diagnostics
