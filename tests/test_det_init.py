"""Family D init is seed-independent and leaves the default init alone."""
import hashlib
import os
import subprocess
import sys
import textwrap

import torch
from torch import nn

from particlegan import ParticlePrior
from particlegan.det_init import VARIANTS, install, spec, uninstall


def _blob():
    layers = nn.Sequential(
        nn.Linear(4, 16), nn.Linear(16, 16), nn.Linear(16, 2),
        nn.Conv2d(1, 4, 3), nn.Conv2d(4, 4, 3),
    )
    prior = ParticlePrior(8, 3, init_std=0.5)
    parts = [prior.z.detach().cpu().contiguous()]
    for module in layers.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parts.append(module.weight.detach().cpu().contiguous())
            parts.append(module.bias.detach().cpu().contiguous())
    return b"".join(part.numpy().tobytes() for part in parts)


def _sha(variant, seed):
    uninstall()
    torch.manual_seed(seed)
    if variant:
        install(variant)
    try:
        return hashlib.sha256(_blob()).hexdigest()
    finally:
        uninstall()


def test_variants_match_across_seeds():
    seen = {}
    for name in VARIANTS:
        first = _sha(name, 0)
        second = _sha(name, 707)
        assert first == second, name
        assert _sha(name, 101) != _sha(None, 101)
        seen[name] = first
    assert len(set(seen.values())) == len(VARIANTS)


def test_default_init_still_follows_the_seed():
    uninstall()
    torch.manual_seed(0)
    first = nn.Linear(4, 8).weight.detach().clone()
    torch.manual_seed(1)
    second = nn.Linear(4, 8).weight.detach().clone()
    assert not torch.equal(first, second)


def test_square_hidden_is_orthogonal_and_rectangular_matches_kaiming_rms():
    install("hid_q")
    try:
        hidden = nn.Linear(16, 16)
        gain = torch.tensor(2.0 / 6.0, dtype=torch.float64).sqrt()
        q = hidden.weight.detach().double() / gain
        eye = torch.eye(16, dtype=torch.float64)
        assert torch.allclose(q.T @ q, eye, atol=1e-5)
        rect = nn.Linear(4, 16)
        fan = rect.weight.shape[1]
        std = gain / torch.tensor(float(fan)).sqrt()
        rms = rect.weight.detach().double().pow(2).mean().sqrt()
        assert torch.allclose(rms, std, rtol=1e-5, atol=0)
        assert rect.bias.abs().max() <= (0.25 / fan ** 0.5) + 1e-6
        conv = nn.Conv2d(8, 8, 3)
        assert conv.weight[:, :, 0, :].abs().sum() == 0
        assert conv.weight[:, :, :, 0].abs().sum() == 0
    finally:
        uninstall()


def test_mix_uses_householder_hidden_and_qr_readout():
    install("mix_pb_pq")
    try:
        hidden = nn.Linear(16, 16)
        gain = torch.tensor(2.0 / 6.0, dtype=torch.float64).sqrt()
        q = hidden.weight.detach().double() / gain
        assert torch.allclose(q.T @ q, torch.eye(16, dtype=torch.float64), atol=1e-5)
        readout = nn.Linear(16, 2)
        weight = readout.weight.detach().double()
        gram = weight @ weight.T
        scale = gram.diag().mean().sqrt()
        assert torch.allclose(gram / gram.diag().mean(), torch.eye(2, dtype=torch.float64), atol=1e-4)
        assert scale > 0
        assert readout.weight.abs().sum() > 0
        assert not torch.allclose(readout.weight.detach().double()[:, 2:], torch.zeros(2, 14, dtype=torch.float64))
    finally:
        uninstall()


def test_house_all_rect_is_semi_orthogonal_at_declared_rms():
    install("hh_pb_pq")
    try:
        layer = nn.Linear(4, 16)
        weight = layer.weight.detach().double()
        gram = weight.T @ weight
        rms = weight.pow(2).mean().sqrt()
        fan = 4.0
        declared = torch.tensor(2.0 / 6.0, dtype=torch.float64).sqrt() / fan ** 0.5
        assert torch.allclose(rms, declared, rtol=1e-4, atol=0)
        column_scale = gram.diag().mean()
        assert torch.allclose(gram / column_scale, torch.eye(4, dtype=torch.float64), atol=1e-4)
    finally:
        uninstall()


def test_pattern_bias_matches_declared_std_and_quarter_weyl_is_smaller():
    install("hq_pb")
    try:
        layer = nn.Linear(16, 16)
        bias = layer.bias.detach().double()
        bound = 1.0 / 16 ** 0.5
        declared = bound / 3.0 ** 0.5
        assert abs(float(bias.mean())) < 1e-6
        assert abs(float(bias.std(unbiased=False)) - declared) / declared < 1e-5
    finally:
        uninstall()
    install("hid_q")
    try:
        layer = nn.Linear(16, 16)
        assert float(layer.bias.detach().abs().max()) <= 0.25 / 16 ** 0.5 + 1e-6
    finally:
        uninstall()


def test_r2_prior_differs_from_weyl_and_ignores_the_seed():
    install("hq_pq")
    try:
        torch.manual_seed(1)
        r2 = ParticlePrior(32, 2, init_std=0.5).z.detach().clone()
        torch.manual_seed(2)
        again = ParticlePrior(32, 2, init_std=0.5).z.detach().clone()
        assert torch.equal(r2, again)
    finally:
        uninstall()
    install("hid_q")
    try:
        weyl = ParticlePrior(32, 2, init_std=0.5).z.detach().clone()
    finally:
        uninstall()
    assert not torch.equal(r2, weyl)


def test_host_eye_and_zero_bias_win():
    install("hq_pb")
    try:
        layer = nn.Linear(2, 2)
        assert layer.bias.abs().sum() > 0
        with torch.no_grad():
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        assert torch.equal(layer.weight, torch.eye(2))
        assert torch.equal(layer.bias, torch.zeros(2))
    finally:
        uninstall()


def test_xavier_refill_keeps_householder_and_changes_qr_scale():
    install("hid_q")
    try:
        layer = nn.Linear(8, 8)
        before = layer.weight.detach().clone()
        nn.init.xavier_uniform_(layer.weight)
        assert torch.equal(before, layer.weight)
    finally:
        uninstall()
    install("qr_pb_pq")
    try:
        layer = nn.Linear(8, 8)
        before = layer.weight.detach().clone()
        nn.init.xavier_uniform_(layer.weight)
        assert not torch.equal(before, layer.weight)
        rms = layer.weight.detach().double().pow(2).mean().sqrt()
        # Xavier uniform on a square map has RMS sqrt(2 / (fan_in + fan_out)).
        assert torch.allclose(rms, torch.tensor(2.0 / 16.0, dtype=torch.float64).sqrt(), rtol=1e-4, atol=0)
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
    install("hq_pb_pq")
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


def test_deepcopy_sees_the_deterministic_critic():
    from copy import deepcopy
    install("mix_pb_pq")
    try:
        critic = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 1))
        shadow = deepcopy(critic)
        for left, right in zip(critic.parameters(), shadow.parameters()):
            assert torch.equal(left, right)
    finally:
        uninstall()


def test_two_processes_hash_equal():
    script = textwrap.dedent("""
        import hashlib, sys, torch
        from torch import nn
        from particlegan import ParticlePrior
        from particlegan.det_init import install
        install(sys.argv[1])
        torch.manual_seed(int(sys.argv[2]))
        layer = nn.Linear(7, 11)
        hidden = nn.Linear(11, 11)
        conv = nn.Conv2d(3, 3, 3)
        prior = ParticlePrior(5, 2, init_std=0.25, generator=torch.Generator().manual_seed(int(sys.argv[2]) + 9))
        raw = b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                       for t in (layer.weight, layer.bias, hidden.weight, hidden.bias,
                                 conv.weight, conv.bias, prior.z))
        print(hashlib.sha256(raw).hexdigest())
    """)
    hashes = []
    for seed in (0, 404):
        done = subprocess.run(
            [sys.executable, "-c", script, "hh_pb_pq", str(seed)],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        hashes.append(done.stdout.strip().splitlines()[-1])
    assert hashes[0] == hashes[1]
    assert spec("hh_pb_pq") == dict(w="house_all", b="pattern", p="r2")
    assert spec("qr_pb_pq")["p"] == "r2"
    assert spec("hid_q")["b"] == "weyl_q"
