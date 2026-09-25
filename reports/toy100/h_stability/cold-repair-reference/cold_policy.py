"""Fixed positive preconditioning of adversarial particle Adam displacements.

The centroid eigenvalue is positive: particles remain free to translate. No
target, evaluation metric, initial parameter anchor, or extra objective is used.
The zero-mean component of every proposed particle displacement is unchanged.
"""
from contextlib import contextmanager, ExitStack
import math
from unittest.mock import patch

import torch
from particlegan import ParticlePrior
from selected_h_extension import extended_signal_policy


@contextmanager
def cold_policy(options):
    options = dict(options)
    mobility = options.pop("prior_centroid_mobility", 1.)
    if (type(mobility) not in (int, float) or not math.isfinite(mobility)
            or not 0 < mobility <= 1):
        raise ValueError("prior_centroid_mobility must be fixed and in (0, 1]")
    prior_ids = set()
    original_init = ParticlePrior.__init__
    original_step = torch.optim.Adam.step
    work = dict(policy="positive_prior_centroid_preconditioner_v1",
                centroid_mobility=mobility, extra_forwards=0, extra_backwards=0,
                transformed_tensors=0, updates=[])

    def prior_init(prior, *args, **kwargs):
        original_init(prior, *args, **kwargs)
        prior_ids.update(id(p) for p in prior.parameters())

    def adam_step(optimizer, closure=None):
        if closure is not None:
            raise ValueError("explicit adversarial gradients required")
        before = []
        if mobility != 1:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None and (id(p) in prior_ids or group.get("_comparison_prior", False)):
                        if p.ndim < 2:
                            raise ValueError("particle tensor needs a leading particle axis")
                        before.append((p, p.detach().clone(), group))
        result = original_step(optimizer)
        with torch.no_grad():
            for p, anchor, group in before:
                displacement = p - anchor
                centroid = displacement.mean(dim=0, keepdim=True)
                p.add_(centroid, alpha=-(1 - mobility))
                work["transformed_tensors"] += 1
                work["updates"].append(dict(
                    step=int(optimizer.state[p]["step"]),
                    shape=list(p.shape), relative_rate=float(group["lr"]),
                    centroid_rate=float(group["lr"]) * mobility,
                    proposed_centroid_rms=float(centroid.square().mean().sqrt()),
                    applied_centroid_rms=float((p-anchor).mean(0).square().mean().sqrt())))
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(ParticlePrior, "__init__", prior_init))
        stack.enter_context(patch.object(torch.optim.Adam, "step", adam_step))
        receipt = stack.enter_context(extended_signal_policy(options))
        receipt["particle_preconditioner"] = work
        receipt["host_extension"]["archived_candidate_unchanged"] = mobility == 1
        yield receipt
