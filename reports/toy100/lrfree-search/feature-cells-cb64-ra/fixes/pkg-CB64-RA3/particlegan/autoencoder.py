"""Particle encodings with caller-owned networks and reconstruction objectives."""
from dataclasses import dataclass
import math

import torch

from .particle_prior import MoGParticlePrior


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass
class ParticleEncoding:
    """Codes [B,S,D], indices [B,S], joint KL [B], and posterior log probs.

    AE has one deterministic draw and no posterior/ELBO. Soft categorical
    samples retain log probabilities for the leave-one-out score estimator.
    Hard VAE uses a biased straight-through routing gradient.
    """
    codes: torch.Tensor
    indices: torch.Tensor
    kl: torch.Tensor
    log_probs: torch.Tensor | None
    mode: str

    def _cost(self, prediction, target):
        if (target.ndim < 2 or target.numel() == 0
                or prediction.shape != (target.shape[0], self.codes.shape[1], *target.shape[1:])
                or target.shape[0] != self.codes.shape[0]):
            raise ValueError("prediction must be [B,S,...] and target [B,...], matching codes")
        cost = (prediction - target[:, None]).square().flatten(2).mean(2)
        if self.mode == "categorical":
            draws = cost.shape[1]
            if draws < 2:
                raise ValueError("categorical training needs at least two independent draws")
            baseline = (cost.sum(1, keepdim=True) - cost) / (draws - 1)
            selected = self.log_probs.gather(1, self.indices)
            score = ((cost - baseline).detach() * selected).mean()
            return cost.mean() + (score - score.detach())
        return cost.mean()

    def reconstruction_loss(self, prediction, target):
        """Mean squared reconstruction error; never adds a KL penalty.

        Soft categorical backward includes a leave-one-out score estimator.
        KL, GAN and particle spread are separate caller-composed objectives.
        """
        return self._cost(prediction, target)

    def negative_elbo(self, prediction, target, *, observation_sigma=0.03):
        """Gaussian negative ELBO in nats, including the constant and joint KL."""
        if self.mode == "ae":
            raise ValueError("deterministic AE has no variational ELBO")
        _positive(observation_sigma, "observation_sigma")
        mse = self._cost(prediction, target)
        dimensions = target[0].numel()
        return (dimensions * mse / (2 * observation_sigma**2) + self.kl.mean()
                + dimensions / 2 * math.log(2 * math.pi * observation_sigma**2))


def _routing(query, prior, temperature, distance_reduction, detach_means):
    if not isinstance(prior, MoGParticlePrior):
        raise TypeError("encoding requires a MoGParticlePrior")
    _positive(temperature, "temperature")
    if distance_reduction not in ("sum", "mean"):
        raise ValueError("distance_reduction must be sum or mean")
    means = prior.means()
    if (query.ndim != 2 or query.shape[0] == 0 or query.shape[1] != means.shape[1]
            or query.device != means.device or query.dtype != means.dtype):
        raise ValueError("query must be nonempty [B,z_dim], with prior device and dtype")
    fixed = means.detach() if detach_means else means
    # O(B*K) storage, avoiding a B*K*D distance tensor.
    distance = query.square().sum(1, keepdim=True) + fixed.square().sum(1)[None] - 2 * query @ fixed.T
    if distance_reduction == "mean":
        distance = distance / query.shape[1]
    return means, (-distance / temperature).log_softmax(1)


def _hard_center(means, log_probs):
    indices = log_probs.argmax(1)
    proxy = log_probs.exp() @ means.detach()
    return means[indices] + (proxy - proxy.detach()), indices


def particle_ae(query, offset, prior, *, temperature=0.25, distance_reduction="sum", offset_bound=3.0):
    """Hard selected center + sigma * bounded offset; no sampling or KL.

    Query uses a biased soft straight-through derivative; table gradients pass
    through selected centers only (and through prior read standardization).
    """
    _positive(offset_bound, "offset_bound")
    means, log_probs = _routing(query, prior, temperature, distance_reduction, True)
    if offset.shape != query.shape or offset.device != query.device or offset.dtype != query.dtype:
        raise ValueError("offset must match query shape, device and dtype")
    center, indices = _hard_center(means, log_probs)
    codes = center + prior.sigma * offset_bound * torch.tanh(offset / offset_bound)
    return ParticleEncoding(codes[:, None], indices[:, None], query.new_zeros(len(query)), None, "ae")


def particle_vae(query, prior, *, temperature=0.25, distance_reduction="sum", draws=2,
                 hard=True, generator=None):
    """Sample q(k|X) and prior-matching local Gaussian noise.

    Soft q is distance-based categorical: joint KL = log(K) - H(q).
    Hard q is one-hot: joint KL = log(K), with biased straight-through routing.
    Neither variant learns a local posterior offset or variance.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer")
    if type(hard) is not bool:
        raise ValueError("hard must be a boolean")
    means, log_probs = _routing(query, prior, temperature, distance_reduction, hard)
    if hard:
        center, chosen = _hard_center(means, log_probs)
        indices = chosen[:, None].expand(-1, draws)
        centers = center[:, None]
        # Expose the true posterior, never the soft backward weights.
        log_probs = torch.full_like(log_probs, -torch.inf).scatter(1, chosen[:, None], 0)
        kl = query.new_full((len(query),), math.log(len(means)))
    else:
        indices = torch.multinomial(log_probs.exp(), draws, replacement=True, generator=generator)
        centers = means[indices]
        kl = (log_probs.exp() * (log_probs + math.log(len(means)))).sum(1)
    eps = torch.randn((len(query), draws, query.shape[1]), device=query.device,
                      dtype=query.dtype, generator=generator)
    codes = centers + prior.sigma * eps
    return ParticleEncoding(codes, indices, kl, log_probs, "hard" if hard else "categorical")
