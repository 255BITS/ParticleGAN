"""Bounded real-neighborhood Gaussian features for an output-marginal loss.

The finite feature frame is rebuilt from each existing real batch. It consumes
no RNG, labels, target geometry, evaluation observations or persistent state.
"""
import math

import torch


MAX_DIMENSIONS = 8
MAX_REFERENCES = 256
MAX_ANCHORS = 64
NEIGHBORS = 16
RIDGE_FRACTION = .1
MULTIPLIERS = (1., 2., 4.)


def _coordinates(fake, real):
    x, y = fake.flatten(1), real.detach().flatten(1)
    if x.dtype in (torch.float16, torch.bfloat16):
        x, y = x.float(), y.float()
    dimension = x.shape[1]
    if dimension > MAX_DIMENSIONS:
        # First eight orthonormal DCT-II columns, including the constant column.
        # This deterministic compression is a declared information limitation.
        rows = torch.arange(dimension, device=x.device, dtype=x.dtype)[:, None] + .5
        columns = torch.arange(MAX_DIMENSIONS, device=x.device, dtype=x.dtype)[None, :]
        frame = torch.cos(rows * columns * (math.pi / dimension)) * math.sqrt(2 / dimension)
        frame[:, 0] *= math.sqrt(.5)
        x, y = x @ frame, y @ frame
    return x, y


def _real_geometry(y, diagnostics=None):
    """At most 64 detached, positive-definite covariance ellipsoids of rank 8."""
    n, dimension = y.shape
    count = min(n, MAX_REFERENCES)
    reference_indices = torch.arange(count, device=y.device) * n // count
    reference = y[reference_indices]
    anchor_count = min(count, MAX_ANCHORS)
    anchor_reference_indices = torch.arange(anchor_count, device=y.device) * count // anchor_count
    anchors = reference[anchor_reference_indices]
    distance = torch.cdist(anchors, reference, compute_mode="donot_use_mm_for_euclid_dist").square()
    distance[torch.arange(anchor_count, device=y.device), anchor_reference_indices] = float("inf")
    # Stable ties retain reference order; duplicate real rows are valid neighbors.
    indices = distance.argsort(dim=1, stable=True)[:, :min(NEIGHBORS, count - 1)]
    neighborhood = reference[indices]
    centered = neighborhood - neighborhood.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / centered.shape[1]
    local_scale = covariance.diagonal(dim1=1, dim2=2).mean(dim=1)
    global_scale = (y - y.mean(0)).square().mean().clamp_min(torch.finfo(y.dtype).eps)
    ridge = (RIDGE_FRACTION * local_scale).clamp_min(torch.finfo(y.dtype).eps * global_scale)
    covariance = covariance + ridge[:, None, None] * torch.eye(dimension, device=y.device, dtype=y.dtype)
    precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
    if diagnostics is not None:
        observations = dict(calls=1, anchors=anchor_count, references=count,
                            compressed_calls=0, zero_covariance_anchors=int((local_scale == 0).sum()),
                            minimum_ridge=float(ridge.min()), maximum_ridge=float(ridge.max()),
                            maximum_condition_bound=float((1 + dimension * local_scale / ridge).max()))
        record_geometry_stats(diagnostics, observations)
    return anchors, precision


def _distances(values, anchors, precision):
    delta = values[:, None, :] - anchors[None, :, :]
    # A x N x d layout bounds temporary storage and avoids an N x N kernel.
    row_delta = delta.transpose(0, 1)
    return ((row_delta @ precision) * row_delta).sum(2).transpose(0, 1).clamp_min(0)


def anisotropic_transport_local_loss(fake, real, *, diagnostics=None):
    """Relative real-anchor feature moments under local Mahalanobis geometry.

    L = mean_(anchor,m) ((mean_fake phi - mean_real phi)/mean_real phi)^2,
    phi(z) = exp(- (z-a)^T H^-1 (z-a)/(2 m^2)), m in {1,2,4}.
    H is the covariance of sixteen other real-reference neighbors, plus a
    ten-percent trace ridge. All real rows contribute to the real feature mean,
    and all fake rows receive gradients. No balanced coupling/global W2 enters.
    It is a finite empirical feature discrepancy, not a calibrated density test
    or characteristic population MMD; the real-derived frame is batch-dependent.
    """
    if (fake.ndim < 2 or fake.shape != real.shape or len(fake) < 2
            or fake.flatten(1).shape[1] < 1 or not fake.is_floating_point()
            or fake.dtype != real.dtype or fake.device != real.device):
        raise ValueError("anisotropic transport needs matching floating batches of at least two samples")
    x, y = _coordinates(fake, real)
    with torch.no_grad():
        anchors, precision = _real_geometry(y, diagnostics)
        if diagnostics is not None and fake.flatten(1).shape[1] > MAX_DIMENSIONS:
            diagnostics['compressed_calls'] += 1
        distance_real = _distances(y, anchors, precision)
    distance_fake = _distances(x, anchors, precision)
    residuals = []
    for multiplier in MULTIPLIERS:
        p = torch.exp(-distance_real / (2 * multiplier**2)).mean(0)
        q = torch.exp(-distance_fake / (2 * multiplier**2)).mean(0)
        # Each real-derived anchor contributes a unit feature, hence p >= 1/n.
        residuals.append(((q - p) / p).square().mean())
    return torch.stack(residuals).mean()


def new_geometry_stats():
    """Cumulative scalar telemetry, never consumed by the training mechanism."""
    return dict(calls=0, anchors=0, references=0, compressed_calls=0,
                zero_covariance_anchors=0, minimum_ridge=0., maximum_ridge=0.,
                maximum_condition_bound=0.)


def record_geometry_stats(state, observation):
    previous_calls = state['calls']
    for key in ('calls', 'anchors', 'references', 'compressed_calls', 'zero_covariance_anchors'):
        state[key] += observation[key]
    state['minimum_ridge'] = (min(state['minimum_ridge'], observation['minimum_ridge'])
                              if previous_calls else observation['minimum_ridge'])
    for key in ('maximum_ridge', 'maximum_condition_bound'):
        state[key] = max(state[key], observation[key])


def validate_geometry_stats(state):
    expected = new_geometry_stats()
    if (not isinstance(state, dict) or state.keys() != expected.keys()
            or any(type(state[key]) is not int or state[key] < 0
                   for key in ('calls', 'anchors', 'references', 'compressed_calls', 'zero_covariance_anchors'))
            or state['compressed_calls'] > state['calls']
            or state['zero_covariance_anchors'] > state['anchors']
            or any(type(state[key]) not in (int, float) or not math.isfinite(state[key]) or state[key] < 0
                   for key in ('minimum_ridge', 'maximum_ridge', 'maximum_condition_bound'))
            or state['minimum_ridge'] > state['maximum_ridge']):
        raise ValueError('invalid anisotropic geometry counters')
