"""Meaningful in-cluster acquisition around the shared vector shape scorer."""
import torch

from benchmarks.transfer_suite.vector_tasks import score_samples as vector_score_samples


@torch.no_grad()
def score_samples(points, spec, completed_steps):
    result = vector_score_samples(points, spec, completed_steps)
    if not torch.isfinite(points).all():
        return result
    means = points.new_tensor(spec["means"])
    covariance = points.new_tensor(spec["covariances"])
    labels = torch.cdist(points, means).argmin(1)
    delta = points-means[labels]
    distance2 = torch.einsum("ni,nij,nj->n", delta,
                             torch.linalg.inv(covariance)[labels], delta)
    counts = torch.bincount(labels[distance2 <= 9], minlength=len(means))
    mass_ratio = counts / len(points) / points.new_tensor(spec["masses"])
    result.update(modes=int((mass_ratio >= .25).sum()),
                  min_hq_mass_ratio=float(mass_ratio.min()),
                  hq_component_counts=counts.tolist())
    return result
