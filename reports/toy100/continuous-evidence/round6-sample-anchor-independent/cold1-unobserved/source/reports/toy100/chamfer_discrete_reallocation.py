"""Data-only discrete escape from a sampled unit-mean C+Q Chamfer basin.

This operates on free output points. It neither calls a generator nor changes
model/optimizer state. A neural pullback and its own acceptance test are still
needed before using any proposed cloud in training.
"""

import torch


def _cost(real, points):
    squared = torch.cdist(real.double(), points.double()).square()
    return float(squared.min(dim=1).values.mean()
                 + squared.min(dim=0).values.mean())


def best_real_relocation(real, points):
    """Return the globally best one-particle replacement by an observed row.

    Searches every donor particle and every real minibatch row. The algebraic
    C+Q cost is exact for each replacement: non-donor backward terms are fixed,
    the donor's backward term becomes zero, and forward terms take the smaller
    of the remaining particles' distance and the chosen real row's distance.
    Deterministic first-index ties follow ``torch.min``. No oracle centers,
    score threshold, random draw, or step number enter the decision.
    """
    if (real.ndim != 2 or points.ndim != 2 or not len(real)
            or len(points) < 2 or real.shape[1] != points.shape[1]):
        raise ValueError("expected nonempty [real,dimension] and >=2 output points")
    if not bool(torch.isfinite(real).all() and torch.isfinite(points).all()):
        raise FloatingPointError("nonfinite real data or points")
    with torch.no_grad():
        r, p = real.double(), points.double()
        distance = torch.cdist(r, p).square()
        real_distance = torch.cdist(r, r).square()
        real_distance.fill_diagonal_(0.)
        backward = distance.min(dim=0).values
        base = float(distance.min(dim=1).values.mean() + backward.mean())
        best = (float("inf"), None, None)
        for donor in range(len(p)):
            other = torch.cat((distance[:, :donor], distance[:, donor + 1 :]), dim=1)
            remaining = other.min(dim=1).values
            costs = torch.minimum(remaining[:, None], real_distance).mean(dim=0)
            costs += (backward.sum() - backward[donor]) / len(p)
            candidate, sample = costs.min(dim=0)
            if float(candidate) < best[0]:
                best = (float(candidate), donor, int(sample))
        candidate, donor, sample = best
        return dict(donor=donor, real_sample=sample,
                    objective_before=base, objective_if_relocated=candidate)


def greedy_real_reallocate(real, points):
    """Take up to N strictly improving global sampled-data relocations.

    The N cap is the number of output particles, not a tuned coefficient. The
    returned cloud is an output-space proposal, not a realizable neural step.
    All objective values are recomputed after each accepted replacement.
    """
    with torch.no_grad():
        current = points.detach().clone()
        moves = []
        for _ in range(len(current)):
            choice = best_real_relocation(real, current)
            before = choice["objective_before"]
            trial = current.clone()
            trial[choice["donor"]] = real[choice["real_sample"]]
            after = _cost(real, trial)
            # Numerical guard only; the method has no improvement coefficient.
            tolerance = 64 * torch.finfo(torch.float64).eps * max(1., abs(before))
            if after >= before - tolerance:
                break
            if abs(after - choice["objective_if_relocated"]) > 1e-10 * max(1., abs(after)):
                raise RuntimeError("algebraic relocation cost differs from direct C+Q")
            moves.append(dict(move=len(moves) + 1, donor=choice["donor"],
                              real_sample=choice["real_sample"],
                              objective_before=before, objective_after=after,
                              decrease=before - after))
            current = trial
        return current, dict(objective="unit-mean sampled C+Q",
                             input_objective=_cost(real, points),
                             final_objective=_cost(real, current),
                             moves=moves, cap=len(points),
                             stopped_before_cap=len(moves) < len(points))
