"""Structured init is seed-independent, orthogonal, and Kaiming-scaled."""
import hashlib
import math
import subprocess
import sys

import torch
from torch import nn

from particlegan import ParticlePrior
from particlegan.structured_init import NAMES, configure, deactivate, digest_params


def _sigma(fan_in):
    return 1.0 / math.sqrt(3.0 * fan_in)


def test_catalog_names_are_unique():
    assert len(NAMES) == len(set(NAMES))


def test_default_init_still_follows_the_seed():
    deactivate()
    def draw(seed):
        torch.manual_seed(seed)
        return nn.Linear(8, 8).weight.detach().clone()
    assert not torch.equal(draw(0), draw(1))


def test_weights_match_across_seeds_and_variants_differ():
    def build(name, seed):
        configure(name)
        try:
            torch.manual_seed(seed)
            generator = nn.Sequential(nn.Linear(4, 96), nn.Linear(96, 96), nn.Linear(96, 2))
            critic = nn.Linear(14, 96)
            conv = nn.Conv2d(1, 4, 3)
            prior = ParticlePrior(12, 4, init_std=0.5, generator=torch.Generator().manual_seed(seed + 9))
            params = (list(generator.parameters()) + list(critic.parameters())
                      + list(conv.parameters()) + list(prior.parameters()))
            return digest_params(params)
        finally:
            deactivate()
    assert build("had_tm_frob", 0) == build("had_tm_frob", 101)
    assert build("dct_spec", 0) == build("dct_spec", 707)
    assert build("house_frob", 1) == build("house_frob", 2)
    assert build("had_tm_frob", 0) != build("dct_frob", 0)
    assert build("dct_frob", 0) != build("dct_spec", 0)
    assert build("dct_frob", 0) != build("dct_frob_weyl", 0)


def test_frobenius_and_spectral_scales_are_orthonormal():
    configure("dct_frob")
    try:
        layer = nn.Linear(16, 32)
        weight = layer.weight.detach().double().cpu()
        values = torch.linalg.svdvals(weight)
        assert torch.allclose(values, values[:1].expand_as(values), rtol=1e-5, atol=1e-5)
        sigma = _sigma(16)
        assert torch.allclose(weight.norm(), torch.tensor(sigma * math.sqrt(weight.numel()), dtype=torch.float64), rtol=1e-5, atol=1e-5)
        assert torch.count_nonzero(layer.bias) == 0
    finally:
        deactivate()
    configure("dst_spec")
    try:
        layer = nn.Linear(16, 32)
        weight = layer.weight.detach().double().cpu()
        values = torch.linalg.svdvals(weight)
        target = _sigma(16) * (math.sqrt(32) + math.sqrt(16))
        assert torch.allclose(values, torch.full_like(values, target), rtol=1e-5, atol=1e-5)
    finally:
        deactivate()


def test_dct_bias_matches_kaiming_rms_and_identity_is_kept():
    configure("had_tm_frob_bias")
    try:
        layer = nn.Linear(8, 4)
        rms = layer.bias.detach().double().square().mean().sqrt()
        assert torch.allclose(rms, torch.tensor(_sigma(8), dtype=torch.float64), rtol=1e-5, atol=1e-5)
        eye = nn.Linear(2, 2)
        with torch.no_grad():
            eye.weight.copy_(torch.eye(2))
            eye.bias.zero_()
        assert torch.equal(eye.weight, torch.eye(2))
        assert torch.equal(eye.bias, torch.zeros(2))
    finally:
        deactivate()


def test_prior_whiten_and_uniform_box_ignore_the_seed():
    def cloud(name, seed):
        configure(name)
        try:
            torch.manual_seed(seed)
            prior = ParticlePrior(48, 3, init_std=0.5, generator=torch.Generator().manual_seed(seed))
            return prior.z.detach().clone()
        finally:
            deactivate()
    left, right = cloud("dct_frob", 0), cloud("dct_frob", 5)
    assert torch.equal(left, right)
    cov = left.double().T @ left.double() / left.shape[0]
    assert torch.allclose(cov, 0.25 * torch.eye(3, dtype=torch.float64), rtol=1e-5, atol=1e-5)
    weyl = cloud("dct_frob_weyl", 0)
    assert not torch.equal(left, weyl)

    def box(seed):
        configure("had_tm_spec")
        try:
            torch.manual_seed(seed)
            prior = ParticlePrior(40, 2, init_std=1.0)
            with torch.no_grad():
                prior.z.uniform_(-5.0, 5.0)
            return prior.z.detach().clone()
        finally:
            deactivate()
    filled = box(0)
    assert torch.equal(filled, box(8))
    assert float(filled.min()) >= -5.0 and float(filled.max()) <= 5.0


def test_two_processes_match_and_default_processes_do_not():
    script = r"""
import sys, torch
from torch import nn
from particlegan import ParticlePrior
from particlegan.structured_init import configure, digest_params
seed = int(sys.argv[1])
mode = sys.argv[2]
if mode != "default":
    configure(mode)
torch.manual_seed(seed)
net = nn.Sequential(nn.Linear(4, 32), nn.Linear(32, 2))
prior = ParticlePrior(16, 4, init_std=0.5, generator=torch.Generator().manual_seed(seed))
print(digest_params(list(net.parameters()) + list(prior.parameters())))
"""
    def run(seed, mode):
        proc = subprocess.run([sys.executable, "-c", script, str(seed), mode],
                              check=True, capture_output=True, text=True)
        return proc.stdout.strip().splitlines()[-1]
    assert run(0, "house_spec") == run(101, "house_spec")
    assert run(0, "default") != run(101, "default")
    assert len(run(0, "dst_frob")) == 64


def test_conv_frobenius_uses_pytorch_fan_in():
    configure("had_tm_frob")
    try:
        conv = nn.ConvTranspose2d(3, 5, 4)
        weight = conv.weight.detach().double().cpu()
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
        sigma = _sigma(fan_in)
        assert torch.allclose(weight.norm(), torch.tensor(sigma * math.sqrt(weight.numel()), dtype=torch.float64), rtol=1e-4, atol=1e-4)
        digest = hashlib.sha256(weight.float().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert len(digest) == 64
    finally:
        deactivate()
