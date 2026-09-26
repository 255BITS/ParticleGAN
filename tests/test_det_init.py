"""Family F inits are seed-independent and share an arm's weights."""
import hashlib
import subprocess
import sys
import textwrap

import torch
from torch import nn

from particlegan import ParticlePrior
from particlegan.det_init import VARIANTS, install, parse, uninstall


def _digest(parts) -> str:
    raw = b"".join(part.detach().cpu().contiguous().numpy().tobytes() for part in parts)
    return hashlib.sha256(raw).hexdigest()


def _stack(seed: int):
    torch.manual_seed(seed)
    layers = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 16), nn.Linear(16, 2))
    prior = ParticlePrior(12, 4, init_std=0.5, generator=torch.Generator().manual_seed(seed))
    opt_g = torch.optim.Adam(list(layers.parameters()) + [prior.z], lr=1e-3)
    opt_d = torch.optim.Adam(nn.Linear(2, 1).parameters(), lr=1e-3)
    del opt_g, opt_d
    weights = []
    for module in layers.modules():
        if isinstance(module, nn.Linear):
            weights.extend((module.weight, module.bias))
    return prior.z, weights


def test_names_parse_back():
    assert len(VARIANTS) == 64
    assert len(set(VARIANTS)) == 64
    for name in VARIANTS:
        arm, seq, mp = parse(name)
        assert arm in ("hid_q", "qr_pb_pq")
        assert seq
        assert mp in ("g", "x", "b", "s")
    assert parse("hid_q") == ("hid_q", "weyl", "g")
    assert parse("qr_pb_pq") == ("qr_pb_pq", "r2", "g")
    assert parse("hid_q_r2_x") == ("hid_q", "r2", "x")
    assert parse("qr_pb_pq_strat") == ("qr_pb_pq", "strat", "g")


def test_same_arm_shares_weights_and_changes_only_the_prior():
    try:
        install("hid_q")
        prior_a, weights_a = _stack(0)
        install("hid_q_r2")
        prior_b, weights_b = _stack(7)
        for left, right in zip(weights_a, weights_b):
            assert torch.equal(left, right)
        assert not torch.equal(prior_a, prior_b)
        install("qr_pb_pq")
        prior_c, weights_c = _stack(1)
        install("qr_pb_pq_halton_x")
        prior_d, weights_d = _stack(9)
        for left, right in zip(weights_c, weights_d):
            assert torch.equal(left, right)
        assert not torch.equal(prior_c, prior_d)
        assert not torch.equal(weights_a[0], weights_c[0])
    finally:
        uninstall()


def test_exact_map_matches_declared_std_and_mean():
    try:
        install("hid_q_sobol_x")
        prior, _ = _stack(3)
        values = prior.detach().double()
        assert torch.allclose(values.mean(0), torch.zeros(4, dtype=torch.float64), atol=1e-5)
        assert torch.allclose(values.std(0, unbiased=False), torch.full((4,), 0.5, dtype=torch.float64), atol=1e-5)
        install("qr_pb_pq_fib_x")
        prior, _ = _stack(3)
        values = prior.detach().double()
        assert torch.allclose(values.mean(0), torch.zeros(4, dtype=torch.float64), atol=1e-5)
        assert torch.allclose(values.std(0, unbiased=False), torch.full((4,), 0.5, dtype=torch.float64), atol=1e-5)
    finally:
        uninstall()


def test_square_hidden_is_orthogonal_under_every_hid_prior():
    install("hid_q_strat_s")
    try:
        hidden = nn.Linear(16, 16)
        gain = torch.tensor(2.0 / 6.0, dtype=torch.float64).sqrt()
        q = hidden.weight.detach().double() / gain
        assert torch.allclose(q.T @ q, torch.eye(16, dtype=torch.float64), atol=1e-5)
        assert hidden.bias.abs().sum() > 0
    finally:
        uninstall()


def test_default_init_still_follows_the_seed():
    uninstall()
    torch.manual_seed(0)
    first = nn.Linear(4, 8).weight.detach().clone()
    torch.manual_seed(1)
    second = nn.Linear(4, 8).weight.detach().clone()
    assert not torch.equal(first, second)


def test_rng_consumption_matches_the_default_init():
    uninstall()
    torch.manual_seed(3)
    stream = torch.Generator().manual_seed(5)
    ParticlePrior(6, 2, init_std=0.5, generator=stream)
    nn.Linear(3, 5)
    global_state = torch.get_rng_state().clone()
    prior_state = stream.get_state().clone()
    install("qr_pb_pq_lhs_b")
    try:
        torch.manual_seed(3)
        stream = torch.Generator().manual_seed(5)
        prior = ParticlePrior(6, 2, init_std=0.5, generator=stream)
        layer = nn.Linear(3, 5)
        torch.optim.Adam(list(layer.parameters()) + [prior.z])
        assert torch.equal(global_state, torch.get_rng_state())
        assert torch.equal(prior_state, stream.get_state())
    finally:
        uninstall()


def test_every_variant_matches_across_seeds():
    try:
        for name in VARIANTS:
            install(name)
            prior, weights = _stack(0)
            first = _digest([prior, *weights])
            install(name)
            prior, weights = _stack(707)
            second = _digest([prior, *weights])
            assert first == second, name
    finally:
        uninstall()


def test_hid_q_matches_the_published_parameter_blob():
    """PR #173 hid_q blob: Linear 7x11, Linear 8x8, Conv2d 3->3 k=3, ParticlePrior 5x2."""
    script = textwrap.dedent("""
        import hashlib, torch
        from torch import nn
        from particlegan import ParticlePrior
        from particlegan.det_init import install
        install("hid_q")
        torch.manual_seed(0)
        layer = nn.Linear(7, 11)
        hidden = nn.Linear(8, 8)
        conv = nn.Conv2d(3, 3, 3)
        prior = ParticlePrior(5, 2, init_std=0.25, generator=torch.Generator().manual_seed(9))
        raw = b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                       for t in (layer.weight, layer.bias, hidden.weight, hidden.bias,
                                 conv.weight, conv.bias, prior.z))
        print(hashlib.sha256(raw).hexdigest())
    """)
    done = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert done.stdout.strip().splitlines()[-1] == (
        "9368a2e2862cb2e592c281281f83f8fb6eb20c1903539efa46547c1118b90108")


def test_two_processes_hash_equal():
    script = textwrap.dedent("""
        import hashlib, sys, torch
        from torch import nn
        from particlegan import ParticlePrior
        from particlegan.det_init import install
        install(sys.argv[2])
        torch.manual_seed(int(sys.argv[1]))
        layer = nn.Linear(7, 11)
        hidden = nn.Linear(8, 8)
        conv = nn.Conv2d(3, 3, 3)
        prior = ParticlePrior(5, 2, init_std=0.25, generator=torch.Generator().manual_seed(int(sys.argv[1]) + 9))
        torch.optim.Adam(list(layer.parameters()) + list(hidden.parameters()) + list(conv.parameters()) + [prior.z])
        raw = b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                       for t in (layer.weight, layer.bias, hidden.weight, hidden.bias,
                                 conv.weight, conv.bias, prior.z))
        print(hashlib.sha256(raw).hexdigest())
    """)
    for name in ("hid_q", "qr_pb_pq", "hid_q_fib_s", "qr_pb_pq_strat_b"):
        hashes = []
        for seed in (0, 404):
            done = subprocess.run(
                [sys.executable, "-c", script, str(seed), name],
                check=True, capture_output=True, text=True,
            )
            hashes.append(done.stdout.strip().splitlines()[-1])
        assert hashes[0] == hashes[1], name
