"""The exit clip uses only the real batch and the proposed particle step."""

import torch

from benchmarks.locked_shared.mlp import SimpleMLPGenerator
from reports.toy100.exit_aware_step_clip import (
    exit_scales, realize_output_scales, support_fence,
)


def _reals(n=32, sigma=0.07, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, generator=gen) * sigma


def test_fence_is_a_handful_of_real_nearest_neighbor_lengths():
    reals = _reals()
    distance = torch.cdist(reals, reals)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.min(dim=1).values
    fence = support_fence(reals)
    assert float(fence) > float(nearest.median())
    assert float(fence) < float(nearest.max()) * 4


def test_interior_outward_step_is_not_shrunk():
    reals = _reals()
    before = torch.zeros(1, 2)
    after = torch.tensor([[0.01, 0.0]])
    scales, _ = exit_scales(before, after, reals)
    assert float(scales) == 1.0


def test_edge_step_toward_reals_is_not_shrunk():
    reals = _reals()
    before = torch.tensor([[0.4, 0.0]])
    after = torch.tensor([[0.2, 0.0]])
    scales, fence = exit_scales(before, after, reals)
    assert float(torch.cdist(before, reals).min()) > float(fence)
    assert float(scales) == 1.0


def test_edge_step_away_from_reals_shrinks_and_stops_worsening():
    reals = _reals()
    before = torch.tensor([[0.25, 0.0]])
    after = torch.tensor([[0.45, 0.0]])
    scales, fence = exit_scales(before, after, reals)
    assert float(torch.cdist(before, reals).min()) > float(fence)
    assert float(scales) < 1.0
    landed = before + scales[:, None] * (after - before)
    d0 = torch.cdist(before, reals).min()
    d1 = torch.cdist(landed, reals).min()
    assert float(d1) <= float(d0) + 1e-5


def test_only_the_exposed_particle_is_scaled():
    reals = _reals()
    before = torch.tensor([[0.0, 0.0], [0.30, 0.0]])
    after = torch.tensor([[0.02, 0.0], [0.50, 0.0]])
    scales, _ = exit_scales(before, after, reals)
    assert float(scales[0]) == 1.0
    assert float(scales[1]) < 1.0


def test_latent_correction_moves_only_the_scaled_particle():
    torch.manual_seed(1)
    clean = SimpleMLPGenerator(z_dim=4, hidden_dim=16, n_hidden=2)
    latents = torch.randn(3, 4)
    with torch.no_grad():
        before = clean(latents).detach()
    shifted = latents.clone()
    shifted[1] = shifted[1] + torch.tensor([0.4, -0.2, 0.1, 0.0])
    with torch.no_grad():
        after = clean(shifted).detach()
    # Pretend the proposed state is `shifted`; realize a half step on particle 1.
    scales = torch.tensor([1.0, 0.5, 1.0])
    work = shifted.clone()
    residual = realize_output_scales(clean, work, before, after, scales)
    with torch.no_grad():
        landed = clean(work)
    assert residual < 1e-4
    assert torch.allclose(work[0], shifted[0])
    assert torch.allclose(work[2], shifted[2])
    target = before[1] + 0.5 * (after[1] - before[1])
    assert torch.allclose(landed[1], target, atol=1e-4)
