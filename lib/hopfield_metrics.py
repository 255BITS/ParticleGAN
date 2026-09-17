"""EMA read evaluation for the Hopfield particle-prior study.

Evaluation batches query rows, never the particle axis: every read remains dense.
Callers provide an independent RNG so evaluation does not change training draws.
"""

import math

import torch

from lib.toy_metrics import per_mode_core_ratio
from lib.toy_models import sample_100gaussians


def nearest_grid(samples):
    """Exact nearest unit-grid center without allocating an N x 100 matrix."""
    cells = (samples + 4.5).round().clamp(0, 9).long()
    nearest = cells[:, 0] * 10 + cells[:, 1]
    distances = (samples - (cells.to(samples.dtype) - 4.5)).norm(dim=1)
    return nearest, distances


def weight_metrics(samples, target_weights, std=0.03, min_count=10):
    nearest, distance = nearest_grid(samples)
    hq = distance <= 3 * std
    counts = torch.bincount(nearest[hq], minlength=100)
    target = torch.as_tensor(target_weights, device=samples.device, dtype=torch.float64)
    target = target / target.sum()
    empirical = counts.to(torch.float64) / counts.sum().clamp_min(1)
    # There is no conditional HQ distribution when no generated point is HQ.
    # Mark this as maximal TV, rather than the misleading half-distance to zero.
    tv = float(0.5 * (empirical - target).abs().sum()) if counts.sum() else 1.0
    kl = (target * (target.clamp_min(1e-300).log() - empirical.clamp_min(1e-6).log())).sum()
    return {"modes": int((counts >= min_count).sum()), "hq": float(hq.float().mean()),
            "tv": tv, "kl": float(kl), "mode_weights": empirical.cpu().tolist()}


@torch.no_grad()
def measure_sampling_floor(weights, *, n_eval=100000, repeats=20, generator=None):
    """Empirical HQ-histogram TV floor from independent true-mixture draws."""
    weights = torch.as_tensor(weights)
    if generator is None:
        generator = torch.Generator(device=weights.device).manual_seed(17001)
    values = []
    for _ in range(repeats):
        true = sample_100gaussians(n_eval, weights.device, weights=weights,
                                   generator=generator)
        values.append(weight_metrics(true, weights)["tv"])
    tv = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(tv.mean()), "sd": float(tv.std(correction=1)) if repeats > 1 else 0.0,
            "values": values, "n_eval": n_eval, "repeats": repeats}


def steps_to_tv(rows, threshold=0.03):
    """First sampled step in the final uninterrupted below-threshold suffix.

    None means convergence was not observed by the end of the run. This does
    not claim anything about the steps between evaluation checkpoints.
    """
    result = None
    for row in reversed(rows):
        if not row["tv"] < threshold:
            break
        result = int(row["step"])
    return result


@torch.no_grad()
def evaluate_read(generator, prior, target_weights, *, read=None, n_eval=100000,
                  batch_size=1024, sample_generator=None):
    """Return scalar/histogram metrics and CPU samples from the provided EMA G/read.

    Read entropy/max weight use the first 4096 evaluation queries. Dead fraction
    uses argmax assignments from all n_eval queries. Interpolation evaluates 256
    independent pairs at 32 evenly spaced points including both endpoints.
    """
    if n_eval <= 0 or batch_size <= 0:
        raise ValueError("n_eval and batch_size must be positive")
    was_training = generator.training
    generator.eval()
    try:
        samples = []
        assignments = torch.zeros(prior.num_particles, device=prior.z.device,
                                  dtype=torch.long) if read is not None else None
        max_sum = 0.0
        eff_sum = 0.0
        health_count = 0
        for start in range(0, n_eval, batch_size):
            size = min(batch_size, n_eval - start)
            if read is None:
                z, _ = prior.sample(size, generator=sample_generator)
            else:
                z, p = read(size, generator=sample_generator)
                assignments += torch.bincount(p.argmax(dim=-1), minlength=prior.num_particles)
                count = min(size, 4096 - health_count)
                if count > 0:
                    health = p[:count]
                    max_sum += float(health.max(dim=-1).values.sum())
                    entropy = -(health * health.clamp_min(torch.finfo(health.dtype).tiny).log()).sum(dim=-1)
                    eff_sum += float(entropy.exp().sum())
                    health_count += count
            samples.append(generator(z).cpu())
        fake = torch.cat(samples)
        metrics = weight_metrics(fake, target_weights)
        core = {key: value if math.isfinite(value) else None
                for key, value in per_mode_core_ratio(fake).items()}
        metrics.update(core)
        metrics["sigma_ratio"] = core["per_mode_core_ratio"]
        metrics.update(max_w=None, eff_n=None, dead_frac=None, interp_hq=None, log_beta=None)
        if read is not None:
            metrics.update(max_w=max_sum / health_count, eff_n=eff_sum / health_count,
                           dead_frac=float((assignments == 0).float().mean()),
                           log_beta=float(read.log_beta))
            pairs = torch.randn(256, 2, prior.z_dim, device=prior.z.device,
                                dtype=prior.z.dtype, generator=sample_generator)
            t = torch.linspace(0, 1, 32, device=prior.z.device, dtype=prior.z.dtype)
            q = ((1 - t[None, :, None]) * pairs[:, 0, None, :] +
                 t[None, :, None] * pairs[:, 1, None, :]).reshape(-1, prior.z_dim)
            good = 0
            for start in range(0, len(q), batch_size):
                z, _ = read.retrieve(q[start:start + batch_size])
                _, distances = nearest_grid(generator(z))
                good += int((distances <= 0.09).sum())
            metrics["interp_hq"] = good / len(q)
        return metrics, fake
    finally:
        generator.train(was_training)
