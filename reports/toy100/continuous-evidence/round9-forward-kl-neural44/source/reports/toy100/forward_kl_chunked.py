"""Bounded-memory exact finite-quadrature donor search.

Chunking changes floating-point reduction order, not the declared weighted
likelihood. It avoids retaining candidate_count * all_history_quadrature_rows
log kernels. Work still grows with history; this is not a bounded-work stream
solver or a continuous-integral certificate.
"""
import math

import torch

from reports.toy100.forward_kl_free_filter import cross_entropy, log_kernels


def donor_values(candidates, points, locations, weights, variance, *, chunk_rows=2048):
    """Return the loss for every (donor, candidate) replacement."""
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('positive integer chunk_rows required')
    candidates = candidates.double()
    points = points.double()
    if (len(points) < 2 or not len(candidates) or locations.ndim != 2
            or points.shape[1:] != locations.shape[1:] or candidates.shape[1:] != points.shape[1:]
            or weights.shape != (len(locations),) or not bool((weights > 0).all())
            or abs(float(weights.sum())-1.) > 1e-12
            or not all(bool(torch.isfinite(x).all()) for x in
                       (candidates, points, locations, weights))):
        raise ValueError('finite points and positive normalized quadrature required')
    result = torch.zeros((len(points),len(candidates)), dtype=torch.float64)
    maximum_matrix_elements = 0
    for start in range(0,len(locations),chunk_rows):
        x = locations[start:start+chunk_rows].double()
        w = weights[start:start+chunk_rows].double()
        candidate_logk = log_kernels(x,candidates,variance).T.contiguous()
        matrix = log_kernels(x,points,variance)
        maximum_matrix_elements = max(maximum_matrix_elements,candidate_logk.numel())
        for donor in range(len(points)):
            keep = torch.cat((matrix[:,:donor],matrix[:,donor+1:]),dim=1)
            # Explicit exclusion remains stable even when one atom dominates
            # and subtracting its rounded probability from one would give zero.
            logremain = torch.logsumexp(keep,dim=1)
            replacement = torch.logaddexp(candidate_logk,logremain[None,:])
            result[donor] -= replacement@w
    result += math.log(len(points))+math.log(2*math.pi*variance)
    return result, dict(chunk_rows=chunk_rows, quadrature_rows=len(locations),
                       maximum_candidate_matrix_elements=maximum_matrix_elements)


def global_donor(candidates, points, locations, weights, variance, *, limit=12,
                 chunk_rows=2048):
    """Greedy whole-objective strict descent, <=limit exact global replacements.

    Numerical quadrature auditing belongs to the caller. No quality score,
    model, native random stream, target label, or training clock is used.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError('nonnegative integer donor limit required')
    current = points.detach().double().clone()
    records = []
    for move in range(limit):
        before = float(cross_entropy(locations,weights,current,variance))
        values, accounting = donor_values(candidates,current,locations,weights,
                                          variance,chunk_rows=chunk_rows)
        flattened = int(values.argmin())
        donor, candidate = divmod(flattened,len(candidates))
        predicted = float(values[donor,candidate])
        tolerance = 64*torch.finfo(torch.float64).eps*max(1.,abs(before))
        if predicted >= before-tolerance:
            break
        proposal = current.clone()
        proposal[donor] = candidates[candidate].double()
        actual = float(cross_entropy(locations,weights,proposal,variance))
        if abs(predicted-actual)>1e-10*max(1.,abs(actual)):
            raise RuntimeError('chunked donor score differs from full objective')
        if actual >= before-tolerance:
            # Reduction-order roundoff must never authorize a loss increase.
            break
        records.append(dict(move=move+1,donor_index=donor,real_index=candidate,
                            before=before,after=actual,predicted=predicted,
                            accounting=accounting))
        current = proposal
    return current,records
