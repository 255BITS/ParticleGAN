"""Finite same-batch descent check on an already constructed joint proposal."""
import math
import torch


@torch.no_grad()
def kinetic_backtrack(parameters, before, gradient, evaluate, before_value):
    """Try actual proposal, then seven halvings; restore exactly if all fail.

    Optimizer state is never rewound. The caller supplies a side-effect-free
    objective replay on the original rows, jitter and real batch. Full accepted
    proposals keep their original parameter bits. This finite Armijo check
    controls this batch's value, not a changing GAN game or held-out density.
    """
    delta = [p.detach()-old for p,old in zip(parameters,before)]
    slope = float(sum((g*d).sum() for g,d in zip(gradient,delta)))
    initial = float(before_value)
    full_value = None
    for trial in range(8):
        scale = 2.**(-trial)
        if trial:
            for p,old,d in zip(parameters,before,delta):
                p.copy_(old+scale*d)
        value = float(evaluate())
        if full_value is None:
            full_value = value
        bound = initial + 1e-4 * scale * min(slope,0.)
        if math.isfinite(value) and value <= bound:
            return dict(scale=scale,trials=trial+1,rejected=False,initial=initial,
                        final=value,full_proposal=full_value,slope=slope,bound=bound)
    for p,old in zip(parameters,before):
        p.copy_(old)
    return dict(scale=0.,trials=8,rejected=True,initial=initial,final=initial,
                full_proposal=full_value,slope=slope,bound=initial)
