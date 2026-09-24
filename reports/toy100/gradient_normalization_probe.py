"""Scratch gradient-normalized discriminator for a bounded three-host screen.

Wu et al., arXiv:2109.02235, eqs. 11 and 13: f/(||grad_x f||+|f|).
The denominator stays in autograd for D and G updates. A dtype-machine-epsilon
term handles the otherwise undefined zero/zero case; it is not tuned. The
paper's piecewise-linear theorem does not apply to all frozen critics here.
Only critics with independent per-sample scores are supported. In particular,
BatchDistanceDiscriminator is rejected because its score depends on all rows.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import math

import torch
from torch import nn

from particlegan import BatchDistanceDiscriminator


@dataclass
class GNReceipt:
    calls: int = 0
    grad_enabled_calls: int = 0
    no_grad_calls: int = 0
    data_elements: int = 0
    data_indices: Counter = field(default_factory=Counter)
    routes: Counter = field(default_factory=Counter)
    denominator_min: float = math.inf
    denominator_max: float = 0.0

    def as_dict(self) -> dict:
        return dict(calls=self.calls, grad_enabled_calls=self.grad_enabled_calls,
                    no_grad_calls=self.no_grad_calls, data_elements=self.data_elements,
                    data_indices={str(key): value for key, value in self.data_indices.items()},
                    routes=dict(self.routes),
                    denominator_min=self.denominator_min,
                    denominator_max=self.denominator_max,
                    formula="f/(||grad_data f||_2+abs(f)+finfo(dtype).eps)",
                    denominator_detached=False,
                    per_sample_scores_required=True)


def normalized_score(raw_forward, args: tuple, *, data_index: int,
                     receipt: GNReceipt, route: str,
                     coordinate_scale: torch.Tensor | float = 1.0) -> torch.Tensor:
    """Normalize scalar per-row scores w.r.t. the raw data coordinate only."""
    if not 0 <= data_index < len(args):
        raise ValueError("data coordinate index is absent")
    original = args[data_index]
    if not isinstance(original, torch.Tensor) or original.ndim < 2:
        raise ValueError("GN data coordinate must have a batch axis")
    outside_grad = torch.is_grad_enabled()
    point = original if original.requires_grad and outside_grad else original.detach().requires_grad_(True)
    values = list(args)
    values[data_index] = point
    with torch.enable_grad():
        score = raw_forward(*values)
        if (not isinstance(score, torch.Tensor) or score.shape not in
                ((len(point),), (len(point), 1))):
            raise ValueError("GN requires exactly one scalar score per sample")
        gradient = torch.autograd.grad(
            score.sum(), point, create_graph=outside_grad, retain_graph=True,
        )[0]
        norm = torch.linalg.vector_norm(gradient.flatten(1), ord=2, dim=1)
        scale = torch.as_tensor(coordinate_scale, dtype=norm.dtype, device=norm.device)
        if torch.any(~torch.isfinite(scale)) or torch.any(scale <= 0):
            raise ValueError("GN raw-data coordinate scale must be positive and finite")
        norm = norm * scale
        score_flat = score.reshape(len(point))
        denominator = norm + score_flat.abs() + torch.finfo(score.dtype).eps
        normalized = (score_flat / denominator).reshape(score.shape)
    receipt.calls += 1
    receipt.grad_enabled_calls += int(outside_grad)
    receipt.no_grad_calls += int(not outside_grad)
    receipt.data_elements += point.numel()
    receipt.data_indices[data_index] += 1
    receipt.routes[route] += 1
    receipt.denominator_min = min(receipt.denominator_min,
                                  float(denominator.detach().min()))
    receipt.denominator_max = max(receipt.denominator_max,
                                  float(denominator.detach().max()))
    return normalized if outside_grad else normalized.detach()


class GradientNormalizedCritic(nn.Module):
    """Parameter-free logit wrapper; leaves feature matching on base features."""

    def __init__(self, model: nn.Module, *, data_index: int, receipt: GNReceipt):
        super().__init__()
        if isinstance(model, BatchDistanceDiscriminator):
            raise ValueError("batch-coupled critic needs an exact per-row Jacobian")
        self.model = model
        self.data_index = data_index
        self.receipt = receipt

    def forward(self, *args, **kwargs):
        if kwargs:
            raise ValueError("scratch GN wrapper expects positional host data arguments")
        return normalized_score(self.model, args, data_index=self.data_index,
                                receipt=self.receipt, route="wrapped_critic")

    def features(self, *args, **kwargs):
        return self.model.features(*args, **kwargs)


@contextmanager
def legacy_gn_patch(receipt: GNReceipt):
    """Adapt the existing data-coordinate insertion points, including 2 ScaleCritics."""
    from unittest.mock import patch
    from benchmarks.transfer_suite import legacy_noise_adapters as adapters
    from benchmarks.locked_shared.hosts import mid_scale_identity, unipolar

    original_wrap = adapters.wrap_input
    original_unipolar = unipolar.ScaleCritic.score
    original_mid = mid_scale_identity.ScaleCritic.score

    def wrap_input(model, policy, *, data_index=0):
        return original_wrap(
            GradientNormalizedCritic(model, data_index=data_index, receipt=receipt),
            policy, data_index=data_index,
        )

    def manual_score(original, route):
        def wrapped(self, z, scale):
            # score() uses z = raw_data / self.input_scale. Convert its
            # derivative back to raw data units before normalization.
            return normalized_score(
                lambda data, label: original(self, data, label), (z, scale),
                data_index=0, receipt=receipt, route=route,
                coordinate_scale=1.0 / self.input_scale,
            )
        return wrapped

    with patch.object(adapters, "wrap_input", wrap_input), patch.object(
        unipolar.ScaleCritic, "score", manual_score(original_unipolar, "unipolar_score"),
    ), patch.object(
        mid_scale_identity.ScaleCritic, "score", manual_score(original_mid, "mid_scale_score"),
    ):
        yield
