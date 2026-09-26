"""Orthogonality, scale, and seed-independence of family E inits."""
import math

import pytest
import torch
from torch import nn

from particlegan import ParticlePrior
from particlegan.family_e_init import FAMILIES, _sobol, basis, install, names, uninstall


def _orth_error(q: torch.Tensor) -> float:
    rows, cols = q.shape
    gram = q.T @ q if rows >= cols else q @ q.T
    eye = torch.eye(gram.shape[0], dtype=torch.float64)
    return float((gram - eye).abs().max())


@pytest.fixture(autouse=True)
def _clean_hooks():
    yield
    uninstall()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8])
def test_square_bases_are_orthogonal(family, n):
    err = _orth_error(basis(family, n, n, key=1))
    assert err < 1e-8, (family, n, err)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_rectangular_bases_are_semi_orthogonal(family, shape):
    err = _orth_error(basis(family, *shape, key=2))
    assert err < 1e-8, (family, shape, err)


def test_sobol_one_d_matches_the_gray_code_sequence():
    # After the skipped origin the Gray-code points are 1/2, 3/4, 1/4, 3/8.
    unit = torch.tensor([0.5, 0.75, 0.25, 0.375], dtype=torch.float64)
    got = torch.special.ndtri(unit.clamp(1e-12, 1 - 1e-12))
    assert torch.allclose(_sobol(4, 1, key=0).squeeze(1), got, atol=1e-8)


def test_layer_keys_change_the_matrix_and_keep_it_orthogonal():
    left = basis("giv", 8, 8, key=0)
    right = basis("giv", 8, 8, key=1)
    assert not torch.allclose(left, right)
    assert _orth_error(left) < 1e-8 and _orth_error(right) < 1e-8


def test_families_are_not_the_same_matrix():
    mats = {family: basis(family, 8, 8, key=0) for family in FAMILIES}
    for left in FAMILIES:
        for right in FAMILIES:
            if left < right:
                assert not torch.allclose(mats[left], mats[right], atol=1e-5), (left, right)


def _build(name, seed):
    uninstall()
    install(name)
    torch.manual_seed(seed)
    layer = nn.Linear(6, 9)
    torch.optim.Adam(layer.parameters(), lr=1e-3)
    return layer.weight.detach().clone(), layer.bias.detach().clone()


def test_weights_ignore_the_torch_seed_and_match_kaiming_rms():
    w0, b0 = _build("giv_bz_pq", 0)
    w1, b1 = _build("giv_bz_pq", 99)
    assert torch.equal(w0, w1) and torch.equal(b0, b1)
    assert torch.equal(b0, torch.zeros_like(b0))
    fan_in = 6
    target = 1.0 / math.sqrt(3.0 * fan_in)
    rms = float(w0.double().pow(2).mean().sqrt())
    assert abs(rms - target) / target < 1e-5
    err = _orth_error(w0.double() / (target * math.sqrt(9)))
    assert err < 1e-5


def test_frob_matches_std_on_a_linear():
    std_w, std_b = _build("cay_pb_pq", 1)
    frob_w, frob_b = _build("cay_pb_pq_frob", 3)
    assert torch.equal(std_w, frob_w)
    assert torch.equal(std_b, frob_b)


def test_pattern_and_quarter_bias_use_the_declared_scale():
    _, pattern = _build("rft_pb_pq", 0)
    _, quarter = _build("rft_wq_pq", 0)
    fan_in = 6.0
    bound = 1.0 / math.sqrt(fan_in)
    declared = (2.0 * bound) / math.sqrt(12.0)
    assert abs(float(pattern.double().std(unbiased=False)) - declared) / declared < 1e-5
    assert float(quarter.abs().max()) <= bound / 4.0 + 1e-6


def test_r2_prior_is_seed_independent_and_scaled():
    def draw(seed):
        uninstall()
        install("haar_bz_pq")
        torch.manual_seed(seed)
        prior = ParticlePrior(64, 3, init_std=0.5)
        torch.optim.Adam(prior.parameters(), lr=1e-3)
        return prior.z.detach().clone()

    z0, z1 = draw(0), draw(7)
    assert torch.equal(z0, z1)
    assert abs(float(z0.mean())) < 0.15
    assert abs(float(z0.std(unbiased=False)) - 0.5) < 0.08


def test_host_zero_after_reset_is_kept():
    uninstall()
    install("exp_bz_pq")
    layer = nn.Linear(4, 4, bias=False)
    nn.init.zeros_(layer.weight)
    torch.optim.Adam(layer.parameters(), lr=1e-3)
    assert torch.equal(layer.weight, torch.zeros_like(layer.weight))


def test_names_cover_every_family_and_bias():
    assert len(names()) == len(FAMILIES) * 3 * 2
    assert "giv_bz_pq" in names() and "exp_wq_pq_frob" in names()
