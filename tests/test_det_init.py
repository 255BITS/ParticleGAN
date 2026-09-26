"""Deterministic init is seed-independent and leaves the default init alone."""
import hashlib
import subprocess
import sys
import textwrap

import torch
from torch import nn

from particlegan import ParticlePrior
from particlegan.det_init import VARIANTS, install, uninstall


def _blob(seed, variant):
    uninstall()
    torch.manual_seed(seed)
    if variant:
        install(variant)
    layers = nn.Sequential(
        nn.Linear(4, 16), nn.Linear(16, 16), nn.Linear(16, 2),
        nn.Conv2d(1, 4, 3), nn.ConvTranspose2d(4, 4, 4),
    )
    prior = ParticlePrior(8, 3, init_std=0.5, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        prior.z.uniform_(-5.0, 5.0)
    parts = [prior.z.detach().cpu().contiguous()]
    for module in layers.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            parts.append(module.weight.detach().cpu().contiguous())
            parts.append(module.bias.detach().cpu().contiguous())
    raw = b"".join(part.numpy().tobytes() for part in parts)
    return hashlib.sha256(raw).hexdigest()


def test_variants_match_across_seeds():
    try:
        for name in VARIANTS:
            first = _blob(0, name)
            second = _blob(707, name)
            assert first == second, name
            assert _blob(101, name) != _blob(101, None)
    finally:
        uninstall()


def test_default_init_still_follows_the_seed():
    uninstall()
    torch.manual_seed(0)
    first = nn.Linear(4, 8).weight.detach().clone()
    torch.manual_seed(1)
    second = nn.Linear(4, 8).weight.detach().clone()
    assert not torch.equal(first, second)


def test_square_hidden_is_orthogonal_and_rectangular_matches_kaiming_rms():
    install("hid")
    try:
        hidden = nn.Linear(16, 16)
        gain = torch.tensor(2.0 / 6.0, dtype=torch.float64).sqrt()
        q = hidden.weight.detach().double() / gain
        assert torch.allclose(q.T @ q, torch.eye(16, dtype=torch.float64), atol=1e-5)
        rect = nn.Linear(4, 16)
        fan = rect.weight.shape[1]
        std = gain / torch.tensor(float(fan)).sqrt()
        rms = rect.weight.detach().double().pow(2).mean().sqrt()
        assert torch.allclose(rms, std, rtol=1e-5, atol=0)
        conv = nn.Conv2d(8, 8, 3)
        assert conv.weight[:, :, 0, :].abs().sum() == 0
        assert conv.weight[:, :, :, 0].abs().sum() == 0
        assert conv.bias.abs().sum() == 0
    finally:
        uninstall()


def test_rng_consumption_matches_the_default_init():
    uninstall()
    torch.manual_seed(3)
    stream = torch.Generator().manual_seed(5)
    ParticlePrior(6, 2, init_std=0.5, generator=stream)
    nn.Linear(3, 5)
    nn.Conv2d(2, 2, 3)
    global_state = torch.get_rng_state().clone()
    prior_state = stream.get_state().clone()
    install("eye_bias")
    try:
        torch.manual_seed(3)
        stream = torch.Generator().manual_seed(5)
        ParticlePrior(6, 2, init_std=0.5, generator=stream)
        nn.Linear(3, 5)
        nn.Conv2d(2, 2, 3)
        assert torch.equal(global_state, torch.get_rng_state())
        assert torch.equal(prior_state, stream.get_state())
    finally:
        uninstall()


def test_two_processes_with_different_seeds_hash_equal():
    script = textwrap.dedent("""
        import hashlib, sys, torch
        from torch import nn
        from particlegan import ParticlePrior
        from particlegan.det_init import install
        seed = int(sys.argv[1])
        install(sys.argv[2])
        torch.manual_seed(seed)
        layer = nn.Linear(7, 11)
        conv = nn.Conv2d(3, 3, 3)
        prior = ParticlePrior(5, 2, init_std=0.25, generator=torch.Generator().manual_seed(seed + 9))
        raw = b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                       for t in (layer.weight, layer.bias, conv.weight, conv.bias, prior.z))
        print(hashlib.sha256(raw).hexdigest())
    """)
    hashes = []
    for seed in (0, 404):
        done = subprocess.run(
            [sys.executable, "-c", script, str(seed), "eye"],
            check=True, capture_output=True, text=True,
        )
        hashes.append(done.stdout.strip().splitlines()[-1])
    assert hashes[0] == hashes[1]
    default = []
    plain = textwrap.dedent("""
        import hashlib, sys, torch
        from torch import nn
        torch.manual_seed(int(sys.argv[1]))
        layer = nn.Linear(7, 11)
        raw = layer.weight.detach().cpu().contiguous().numpy().tobytes()
        print(hashlib.sha256(raw).hexdigest())
    """)
    for seed in (0, 404):
        done = subprocess.run(
            [sys.executable, "-c", plain, str(seed)],
            check=True, capture_output=True, text=True,
        )
        default.append(done.stdout.strip().splitlines()[-1])
    assert default[0] != default[1]
