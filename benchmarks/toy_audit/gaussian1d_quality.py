"""Exact location, width and CDF checks for a single scalar Gaussian."""
import math

import numpy as np
import torch
from scipy.special import ndtr


def sample_target(spec, count, rng, completed_steps=0):
    """Independent scalar data draws; the target never initializes G or the prior."""
    mean = spec["means"][0][0]
    sigma = math.sqrt(spec["covariances"][0][0][0])
    return mean + sigma * torch.randn(count, 1, generator=rng)


def score_samples(points, spec, completed_steps=0):
    """Score the entire law, without histogram bins or target sample noise."""
    if spec["kind"] != "gaussian_mixture" or len(spec["means"]) != 1 or len(spec["means"][0]) != 1:
        raise ValueError("gaussian1d scorer requires one scalar Gaussian")
    if spec["masses"] != [1.0] or spec.get("scale_start", 1.) != 1. or spec.get("scale_end", 1.) != 1.:
        raise ValueError("gaussian1d scorer requires fixed unit mass and scale")
    mean = spec["means"][0][0]
    sigma = math.sqrt(spec["covariances"][0][0][0])
    if not math.isfinite(mean) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian target must have finite mean and positive sigma")
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("gaussian1d samples must have shape (N, 1)")
    n = len(values)
    if not n or not np.isfinite(values).all():
        return dict(sample_count=n, finite_fraction=float(np.isfinite(values).mean()) if n else 0.,
                    mean_error_sigma=None, std_ratio=None, cdf_ks=None, mean=None, std=None)
    ordered = np.sort(values[:, 0])
    cdf = ndtr((ordered - mean) / sigma)
    ranks = np.arange(n, dtype=np.float64) / n
    ks = max(float(np.max(cdf - ranks)), float(np.max(ranks + 1 / n - cdf)))
    actual_mean, actual_std = float(ordered.mean()), float(ordered.std())
    return dict(sample_count=n, finite_fraction=1., mean=actual_mean, std=actual_std,
                mean_error_sigma=abs(actual_mean - mean) / sigma,
                std_ratio=actual_std / sigma, cdf_ks=ks)
