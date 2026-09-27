"""ParticlePrior.perturb's nearest-particle search (broadcast on CUDA, cdist on
CPU) is bitwise identical to the cdist(donot_use_mm) search it replaced."""
import pytest
import torch

from particlegan.particle_prior import ParticlePrior, _nearest_other

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _old_nearest(latent, table):
    nearest = torch.full((len(latent),), float("inf"), device=latent.device, dtype=latent.dtype)
    for centers in table.split(4096):
        distance = torch.cdist(latent, centers, compute_mode="donot_use_mm_for_euclid_dist")
        distance.masked_fill_(distance == 0, float("inf"))
        nearest = torch.minimum(nearest, distance.min(1).values)
    return nearest


def _old_perturb(prior, latent, generator):
    # The pre-speedup ParticlePrior.perturb, verbatim apart from the search helper.
    if not bool(prior.support_ready):
        prior.track_support_()
    noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)
    displacement = prior.support_width * noise
    with torch.no_grad():
        nearest = _old_nearest(latent.detach(), prior.z.detach())
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("z_dim", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("n", [1, 12, 5000])
def test_broadcast_search_equals_cdist_bitwise(device, dtype, z_dim, n):
    table = _table(n, z_dim, device, dtype, seed=n * 10 + z_dim)
    g = torch.Generator().manual_seed(7)
    rows = torch.randint(n, (257,), generator=g).to(device)
    off = torch.randn(64, z_dim, generator=g, dtype=dtype).to(device)
    latent = torch.cat([table[rows], table[:1], off])  # particles, a duplicated one, off-table points
    old, new = _old_nearest(latent, table), _nearest_other(latent, table)
    assert _bitwise(old, new)
    if n == 1:
        assert torch.isinf(new[:258]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("z_dim", [1, 2, 4, 8])
@pytest.mark.parametrize("n", [1, 12, 5000])
def test_perturb_equals_old_implementation_bitwise(device, dtype, z_dim, n):
    prior = ParticlePrior(n, z_dim, device=device, dtype=dtype, support_jitter=True,
                          generator=torch.Generator(device=device).manual_seed(3))
    with torch.no_grad():
        prior.z.copy_(_table(n, z_dim, device, dtype, seed=n + z_dim))
    prior.track_support_()
    latent = prior.z[torch.randint(n, (300,), generator=torch.Generator().manual_seed(1)).to(device)]
    new = prior.perturb(latent, torch.Generator(device=device).manual_seed(11))
    old = _old_perturb(prior, latent, torch.Generator(device=device).manual_seed(11))
    assert new.dtype == latent.dtype and new.device == latent.device
    assert _bitwise(old, new)
    if n == 1:
        assert _bitwise(new, latent)
    # gradients still flow to latent unchanged
    leaf = latent.detach().requires_grad_()
    prior.perturb(leaf, torch.Generator(device=device).manual_seed(11)).sum().backward()
    assert torch.equal(leaf.grad, torch.ones_like(leaf))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("z_dim", [3, 31, 32, 33, 64, 256])
def test_search_bitwise_for_wide_and_strided_tables(device, dtype, z_dim):
    # cdist sums in a different order past one warp (z_dim > 32) or on a strided table; those
    # cases must stay on cdist, so they too are unchanged.
    g = torch.Generator().manual_seed(z_dim)
    scale = torch.logspace(-2, 2, z_dim, dtype=dtype)
    table = (torch.randn(3000, z_dim, generator=g, dtype=dtype) * scale).to(device)
    latent = torch.cat([table[:400], (torch.randn(400, z_dim, generator=g, dtype=dtype) * scale).to(device)])
    strided = table.t().contiguous().t()
    for lat, tab in ((latent, table), (latent, strided), (latent[::2], table[::3])):
        assert _bitwise(_old_nearest(lat, tab), _nearest_other(lat, tab))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_search_half_precision_behaves_as_before(device, dtype):
    table = torch.randn(50, 4, generator=torch.Generator().manual_seed(0)).to(device, dtype)
    try:
        old = _old_nearest(table[:10], table)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            _nearest_other(table[:10], table)
    else:
        assert _bitwise(old, _nearest_other(table[:10], table))
