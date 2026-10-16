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
    def __init__(self, reference, *, eps=1e-4, weight=1., relative_cap=0.):
        super().__init__()
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("weight must be nonnegative and eps positive, both finite")
        if (type(relative_cap) not in (int, float) or not math.isfinite(relative_cap)
                or relative_cap < 0):
            raise ValueError("relative_cap must be finite and nonnegative")
        self.weight, self.eps = float(weight), float(eps)
        # Zero keeps the fixed tether exactly: no hooks and no extra autograd.
        self.relative_cap = float(relative_cap)
        self._cap_handles = []
        self.relative_cap_stats = dict(applications=0, capped=0, last_ratio=None,
                                       last_scale=None, min_scale=None)
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
        value = self.weight * (mean_error + covariance_error)
        if self.relative_cap > 0 and torch.is_grad_enabled() and value.requires_grad:
            value = self._bind_relative_cap(value)
        return value

    def _bind_relative_cap(self, value):
        """Cap this tether's latent-table gradient at ``relative_cap`` times the rest.

        Per latent table ``z`` with total incoming gradient ``g = g_other + g_tether``
        (one backward pass), the table receives ``g_other + s * g_tether`` with
        ``s = min(1, relative_cap * ||g_other|| / ||g_tether||)``. ``g_other`` is
        every non-tether contribution to that backward (normally the adversarial
        generator objective). Only the tether's own direction is rescaled; no
        sample, RNG draw, task statistic or optimizer state is involved.
        """
        for handle in self._cap_handles:
            handle.remove()
        self._cap_handles = []
        leaves = [prior.z for prior in self.references if prior.z.requires_grad]
        if not leaves:
            return value
        # Unit tether gradient (for this scalar) with respect to each table.
        units = torch.autograd.grad(value, leaves, retain_graph=True, allow_unused=True)
        state = {"upstream": None}
        value = _UpstreamTap.apply(value, state)
        stats = self.relative_cap_stats
        cap = self.relative_cap

        def make_hook(unit):
            def hook(grad):
                upstream = state["upstream"]
                if upstream is None or unit is None:
                    return None  # A backward that does not include this tether.
                tether = unit * upstream.to(unit)
                other = grad - tether
                tether_norm = tether.norm()
                other_norm = other.norm()
                if float(tether_norm) > 0:
                    scale = torch.clamp(cap * other_norm / tether_norm, max=1.)
                else:
                    scale = tether_norm.new_ones(())
                ratio = float(tether_norm / other_norm) if float(other_norm) > 0 else math.inf
                s = float(scale)
                stats["applications"] += 1
                stats["capped"] += int(s < 1.)
                stats["last_ratio"], stats["last_scale"] = ratio, s
                stats["min_scale"] = s if stats["min_scale"] is None else min(stats["min_scale"], s)
                return other + scale.to(grad) * tether
            return hook

        pending = []
        for leaf, unit in zip(leaves, units):
            handle_box = {}
            inner = make_hook(None if unit is None else unit.detach())

            def once(grad, inner=inner, handle_box=handle_box):
                result = inner(grad)
                if state["upstream"] is not None:
                    handle_box["handle"].remove()
                return result
            handle_box["handle"] = leaf.register_hook(once)
            pending.append(handle_box["handle"])
        self._cap_handles = pending
        return value


class _UpstreamTap(torch.autograd.Function):
    """Identity on a scalar that records the gradient flowing into it."""

    @staticmethod
    def forward(ctx, value, state):
        ctx.state = state
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad):
        ctx.state["upstream"] = grad.detach()
        return grad, None
