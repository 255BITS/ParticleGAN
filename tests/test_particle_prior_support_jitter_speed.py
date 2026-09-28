"""ParticlePrior.perturb's nearest-particle search always broadcasts (no cdist fallback).

Against the cdist(donot_use_mm) search it replaced it is bitwise on CUDA for contiguous
float32/float64 with z_dim <= 32, and within 2 ulp everywhere else (CPU, wider z, strided
inputs). It also handles half precision, z_dim 0, empty tables, and keeps its blocks no
bigger than the old distance matrix (or a fixed 2**20-element floor)."""
import pytest
import torch

import particlegan.particle_prior as particle_prior
from particlegan.particle_prior import ParticlePrior, _nearest_other

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _old_nearest(latent, table):
    nearest = torch.full((len(latent),), float("inf"), device=latent.device, dtype=latent.dtype)
    for centers in table.split(4096):
        distance = torch.cdist(latent, centers, compute_mode="donot_use_mm_for_euclid_dist")
        distance.masked_fill_(distance == 0, float("inf"))
        nearest = torch.minimum(nearest, distance.min(1).values)
    return nearest


def _old_perturb(prior, latent, generator, search=_old_nearest):
    # The pre-speedup ParticlePrior.perturb, verbatim apart from the search helper.
    if not bool(prior.support_ready):
        prior.track_support_()
    noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
    displacement = prior.support_width * noise
    with torch.no_grad():
        nearest = search(latent.detach(), prior.z.detach())
        radius = torch.where(torch.isfinite(nearest), nearest * .5, torch.zeros_like(nearest))
        norm = displacement.norm(dim=1)
        fraction = (radius / norm.clamp_min(1e-20)).clamp_max(1.)
    return latent + displacement * fraction.unsqueeze(1)


def _table(n, z_dim, device, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(n, z_dim, generator=g, dtype=dtype)
    if n > 3:
        table[1] = table[0]          # exact duplicate rows
        table[3] = table[2]
        table[n // 2:n // 2 + 3] *= 1e-3  # a tight cluster
    return table.to(device)


def _bitwise(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(-1), b.view(-1)) and \
        torch.equal(torch.isinf(a), torch.isinf(b))


def _ulp_close(a, b, ulps=2):
    # Same dtype/shape/inf pattern, and finite values within ``ulps`` units in the last place.
    if not (a.dtype == b.dtype and a.shape == b.shape and torch.equal(torch.isinf(a), torch.isinf(b))):
        return False
    finite = torch.isfinite(b)
    eps = torch.finfo(b.dtype).eps
    a, b = a[finite].double(), b[finite].double()
    return bool(((a - b).abs() <= ulps * eps * b.abs()).all())


def _expect_same(old, new, bitwise):
    assert _bitwise(old, new) if bitwise else _ulp_close(old, new)


def _bitwise_case(device, dtype, z_dim):
    return device == "cuda" and dtype in (torch.float32, torch.float64) and 0 < z_dim <= 32


def _latent(table, seed):
    n, z_dim = table.shape
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(n, (257,), generator=g).to(table.device)
    off = torch.randn(64, z_dim, generator=g, dtype=table.dtype).to(table.device)
    return torch.cat([table[rows], table[:1], off])  # particles, a duplicated one, off-table points


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("z_dim", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("n", [1, 12, 5000])
def test_search_matches_cdist(device, dtype, z_dim, n):
    table = _table(n, z_dim, device, dtype, seed=n * 10 + z_dim)
    latent = _latent(table, 7)
    old, new = _old_nearest(latent, table), _nearest_other(latent, table)
    _expect_same(old, new, _bitwise_case(device, dtype, z_dim))
    if n == 1:
        assert torch.isinf(new[:258]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("z_dim", [3, 31, 32, 33, 64, 256, 512])
def test_search_matches_cdist_wide_and_strided(device, dtype, z_dim):
    g = torch.Generator().manual_seed(z_dim)
    scale = torch.logspace(-2, 2, z_dim, dtype=dtype)
    table = (torch.randn(3000, z_dim, generator=g, dtype=dtype) * scale).to(device)
    latent = torch.cat([table[:400], (torch.randn(400, z_dim, generator=g, dtype=dtype) * scale).to(device)])
    strided = table.t().contiguous().t()
    for lat, tab in ((latent, table), (latent, strided), (latent[::2], table[::3]),
                     (latent.t().contiguous().t(), strided)):
        # Inputs are made contiguous first, so strided ones give exactly the contiguous result.
        new = _nearest_other(lat, tab)
        assert _bitwise(new, _nearest_other(lat.contiguous(), tab.contiguous()))
        # cdist on the contiguous inputs, since strided cdist itself sums in another order.
        _expect_same(_old_nearest(lat.contiguous(), tab.contiguous()), new,
                     _bitwise_case(device, dtype, z_dim))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("z_dim", [2, 32, 100])
@pytest.mark.parametrize("block", [1, 7, 300, 1 << 16])
def test_search_independent_of_block_size(device, z_dim, block, monkeypatch):
    # The min is exact, so the row/center blocking cannot change a bit, except that past one warp
    # (z_dim > 32) CUDA's reduction splits a row's sum by block shape: then it is ulp-close.
    table = _table(2000 if block > 300 else 150, z_dim, device, torch.float32, seed=z_dim)
    latent = _latent(table, 3)[::4]
    reference = _nearest_other(latent, table)
    monkeypatch.setitem(particle_prior._NEAREST_BLOCK, device, block)
    _expect_same(reference, _nearest_other(latent, table), device == "cpu" or z_dim <= 32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows,count,z_dim", [(5, 20000, 2), (300, 20000, 8), (2048, 20000, 512),
                                              (3000, 50, 16), (1, 1, 0), (1, 20000, 512)])
def test_search_blocks_never_exceed_old_distance_matrix(device, rows, count, z_dim, monkeypatch):
    seen = []
    norm = torch.linalg.vector_norm

    def recording(x, *args, **kwargs):
        seen.append(x.shape)
        return norm(x, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", recording)
    table = torch.randn(count, z_dim, device=device)
    _nearest_other(table[torch.randint(count, (rows,))], table)
    budget = min(particle_prior._NEAREST_BLOCK[device], max(rows * min(count, 4096), particle_prior._NEAREST_FLOOR))
    assert seen and all(r * c * (z_dim + 1) <= budget or r * c == 1 for r, c, _ in seen)
    assert sum(r * c for r, c, _ in seen) == rows * count  # every pair visited exactly once
    # and in blocks near the budget, not thousands of tiny launches (few rows, wide z)
    assert len(seen) <= 1.5 * -(-rows * count * (z_dim + 1) // budget) + 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("z_dim", [1, 2, 4, 8, 64])
@pytest.mark.parametrize("n", [1, 12, 5000])
def test_perturb_matches_old_implementation(device, dtype, z_dim, n):
    prior = ParticlePrior(n, z_dim, device=device, dtype=dtype, support_jitter=True,
                          generator=torch.Generator(device=device).manual_seed(3))
    with torch.no_grad():
        prior.z.copy_(_table(n, z_dim, device, dtype, seed=n + z_dim))
    prior.track_support_()
    latent = prior.z[torch.randint(n, (300,), generator=torch.Generator().manual_seed(1)).to(device)]
    new = prior.perturb(latent, torch.Generator(device=device).manual_seed(11))
    old = _old_perturb(prior, latent, torch.Generator(device=device).manual_seed(11))
    assert new.dtype == latent.dtype and new.device == latent.device
    if _bitwise_case(device, dtype, z_dim):
        assert _bitwise(old, new)
    else:
        # The same jitter up to the ulp-level radius difference.
        tol = 8 * torch.finfo(dtype).eps
        torch.testing.assert_close(new, old, rtol=tol, atol=tol * latent.abs().max().item())
        # and exactly what the old perturb gives with the new search plugged in
        assert _bitwise(new, _old_perturb(prior, latent, torch.Generator(device=device).manual_seed(11),
                                          search=_nearest_other))
    if n == 1:
        assert _bitwise(new, latent)
    # gradients still flow to latent unchanged
    leaf = latent.detach().requires_grad_()
    prior.perturb(leaf, torch.Generator(device=device).manual_seed(11)).sum().backward()
    assert torch.equal(leaf.grad, torch.ones_like(leaf))


@pytest.mark.parametrize("device", DEVICES)
def test_zero_dim_and_empty_tables_give_no_jitter(device):
    assert torch.isinf(_nearest_other(torch.zeros(5, 0, device=device), torch.zeros(7, 0, device=device))).all()
    assert torch.isinf(_nearest_other(torch.randn(5, 3, device=device), torch.zeros(0, 3, device=device))).all()
    assert _nearest_other(torch.zeros(0, 3, device=device), torch.randn(4, 3, device=device)).shape == (0,)
    # A lone particle has no other one to keep clear of, so perturb leaves it in place.
    prior = ParticlePrior(1, 3, device=device, support_jitter=True)
    latent = prior.z.repeat(4, 1)
    assert _bitwise(prior.perturb(latent, torch.Generator(device=device).manual_seed(0)), latent)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("z_dim", [2, 64])
def test_half_precision_works(device, dtype, z_dim):
    # cdist rejects these dtypes; the search runs in float32 and rounds the distance once.
    table = _table(500, z_dim, device, torch.float32, seed=z_dim).to(dtype)
    latent = _latent(table, 5)
    new = _nearest_other(latent, table)
    assert new.dtype == dtype
    expect = _old_nearest(latent.double(), table.double())
    assert torch.equal(torch.isinf(new), torch.isinf(expect))
    finite = torch.isfinite(expect)
    rel = ((new[finite].double() - expect[finite]).abs() / expect[finite]).max().item()
    assert rel <= torch.finfo(dtype).eps  # correctly rounded up to float32 sum error: <= 1 ulp

    prior = ParticlePrior(500, z_dim, device=device, dtype=dtype, support_jitter=True)
    with torch.no_grad():
        prior.z.copy_(table)
    prior.track_support_()
    leaf = latent.detach().requires_grad_()
    out = prior.perturb(leaf, torch.Generator(device=device).manual_seed(2))
    assert out.dtype == dtype and torch.isfinite(out).all()
    step = (out - latent).double().norm(dim=1)
    radius = torch.where(torch.isfinite(new), new.double() * .5, torch.zeros_like(step))
    # within the half-way radius, up to rounding the jittered point back to ``dtype``
    eps = torch.finfo(dtype).eps
    assert (step <= radius * (1 + 4 * eps) + eps * latent.double().norm(dim=1)).all()
    out.sum().backward()
    assert torch.equal(leaf.grad, torch.ones_like(leaf))
