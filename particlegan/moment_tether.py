"""A two-sided tether to a latent table's fixed initial first two moments."""
import math

import torch
from torch import nn


REFERENCE_NAMES = ("moment_tether_mean", "moment_tether_covariance", "moment_tether_count")


def capture_initial_moments(prior):
    """Call after initialization; persistent buffers survive own continuations.

    Rebinding a restored prior must never replace its original reference with
    its trained locations. No particles are retained or random draws consumed.
    """
    present = [hasattr(prior, name) for name in REFERENCE_NAMES]
    if all(present):
        return
    if any(present):
        raise ValueError("incomplete moment_tether initial reference")
    with torch.no_grad():
        mean = prior.z.mean(0)
        centered = prior.z - mean
        covariance = centered.T @ centered / max(len(prior.z) - 1, 1)
        prior.register_buffer(REFERENCE_NAMES[0], mean.detach().clone())
        prior.register_buffer(REFERENCE_NAMES[1], covariance.detach().clone())
        prior.register_buffer(REFERENCE_NAMES[2],
                              torch.tensor(len(prior.z), dtype=torch.int64, device=prior.z.device))


class MomentTetherRegularizer(nn.Module):
    """Dimension-averaged squared mean and full covariance discrepancies.

    ``eps`` has latent variance units. Mean discrepancies are divided by initial
    marginal variance + eps; covariance discrepancies by the outer product of
    the corresponding standard deviations. This remains finite for rank-poor
    or zero initial tables and never inverts their covariance matrix.

    The caller supplies its usual rows. For a subset, comparison to full initial
    moments has sampling noise; we do not introduce a second sampling law.
    """
    def __init__(self, reference, *, eps=1e-4, weight=1.):
        super().__init__()
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("weight must be nonnegative and eps positive, both finite")
        self.weight, self.eps = float(weight), float(eps)
        self.references = tuple(reference) if isinstance(reference, (list, tuple)) else (reference,)
        if not self.references or any(not all(hasattr(prior, name) for name in REFERENCE_NAMES)
                                      for prior in self.references):
            raise ValueError("moment_tether requires initialized priors bound by Recipe.apply_prior_policy")

    def reference_moments(self):
        """Pool multiple tables with their original counts, including mean shifts."""
        if len(self.references) == 1:
            prior = self.references[0]
            return prior.moment_tether_mean, prior.moment_tether_covariance
        count = sum(prior.moment_tether_count for prior in self.references)
        mean = sum(prior.moment_tether_mean * prior.moment_tether_count
                   for prior in self.references) / count
        scatter = sum((prior.moment_tether_count - 1) * prior.moment_tether_covariance
                      + prior.moment_tether_count * torch.outer(prior.moment_tether_mean - mean,
                                                               prior.moment_tether_mean - mean)
                      for prior in self.references)
        return mean, scatter / (count - 1).clamp_min(1)

    def forward(self, rows):
        if self.weight == 0:
            return rows[:0].sum()
        if rows.ndim != 2 or rows.shape[1] == 0:
            raise ValueError("particles must have shape [batch, positive latent dimension]")
        if len(rows) == 0:
            return rows.sum()
        initial_mean, initial_covariance = self.reference_moments()
        if rows.shape[1] != len(initial_mean):
            raise ValueError("moment_tether rows and initial reference dimensions differ")
        mean = rows.mean(0)
        variance_scale = initial_covariance.diagonal().clamp_min(0) + self.eps
        mean_error = ((mean - initial_mean).square() / variance_scale).mean()
        centered = rows - mean
        covariance = centered.T @ centered / max(len(rows) - 1, 1)
        covariance_error = ((covariance - initial_covariance).square()
                            / torch.outer(variance_scale, variance_scale)).mean()
        return self.weight * (mean_error + covariance_error)
