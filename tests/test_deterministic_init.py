"""Deterministic inits: same parameters at two seeds, sample RNG still follows the seed."""
import hashlib
import json
import subprocess
import sys

import pytest

from benchmarks.gan_v3 import legacy_dict
from benchmarks.toy100.train import load_config
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


_CHILD = r"""
import hashlib, json, sys
import torch
from torch import nn
kind, seed = sys.argv[1], int(sys.argv[2])
torch.manual_seed(seed)
installed = kind != "default"
if installed:
    from particlegan.deterministic_init import install, prepare_modules
    install(kind)
from particlegan import ParticlePrior
G = nn.Sequential(nn.Linear(8, 8), nn.LeakyReLU(0.2), nn.Linear(8, 2))
D = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1))
prior = ParticlePrior(32, 2, init_std=0.5)
if kind == "ortho_lsuv":
    prepare_modules(G, D)
follow = torch.rand(8)
parts = [p.detach().cpu().reshape(-1) for module in (G, D) for p in module.parameters()]
parts.append(prior.z.detach().cpu().reshape(-1))
raw = torch.cat(parts).numpy().tobytes()
W = G[0].weight.detach().double()
gram = W @ W.T
scale = float(gram.diag().mean())
eye = torch.eye(W.shape[0], dtype=torch.float64)
off = float((gram / scale - eye).abs().max()) if scale > 0 else 1.0
bound = 8 ** -0.5
weight = G[0].weight.detach()
bias = G[0].bias.detach()
print(json.dumps({
    "hash": hashlib.sha256(raw).hexdigest(),
    "follow": [float(x) for x in follow],
    "off": off,
    "scale": scale,
    "in_bounds": bool(weight.abs().max() <= bound + 1e-6 and bias.abs().max() <= bound + 1e-6),
    "prior_finite": bool(torch.isfinite(prior.z).all()),
}))
"""


def _run(kind, seed):
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, kind, str(seed)],
        check=True, capture_output=True, text=True,
    )
    line = [row for row in proc.stdout.splitlines() if row.startswith("{")][-1]
    return json.loads(line)


@pytest.mark.parametrize("kind", ["ortho_lsuv", "sobol", "halton", "fixedgen"])
def test_parameters_ignore_the_global_seed(kind):
    first, second = _run(kind, 0), _run(kind, 101)
    base0, base1 = _run("default", 0), _run("default", 101)
    assert first["hash"] == second["hash"]
    assert base0["hash"] != base1["hash"]
    assert first["hash"] != base0["hash"]
    assert first["follow"] == pytest.approx(base0["follow"])
    assert second["follow"] == pytest.approx(base1["follow"])
    assert first["follow"] != second["follow"]
    assert first["prior_finite"] and second["prior_finite"]
    if kind == "ortho_lsuv":
        assert first["off"] < 1e-4
        assert first["scale"] > 0
    else:
        assert first["off"] > 1e-2
        assert first["in_bounds"]


def test_declared_recipe_keeps_b_cap_and_forwards_k3p():
    constraints = load_config("configs/toy100/constraints_simple_regularization.json")
    recipe, _, _ = declared_recipe(constraints)
    assert recipe.reg_arm == "b_cap"
    assert recipe.latent_damping_max_rate == 0.0
    assert "latent_damping_max_rate" not in legacy_dict(recipe)
    screen = load_config("configs/toy100/k3p_screen.json")
    k3p, _, _ = declared_recipe(screen)
    assert k3p.reg_arm == "k3p"
    assert k3p.latent_damping_max_rate == 0.5
    assert k3p.d_guard_ratio == 5.0
    assert k3p.reg_anchor_decay == 0.999
    assert k3p.network_lr_floor == 0.01
    assert k3p.network_lr_horizon_cap == 1600
    assert k3p.lr == constraints["lr"]
    assert k3p.reg_coeff == constraints["reg_coeff"]
    assert k3p.input_noise_std == 0.0
    assert k3p.output_noise_std == 0.0


def test_k3p_ring_records_the_critic_step():
    import torch
    from benchmarks.locked_shared.baseline import Candidate
    from benchmarks.locked_shared.mode_hold import ModeHoldRecipe, train_mode_hold
    torch.manual_seed(0)
    candidate = Candidate("k3p_smoke", reg_arm="k3p", particle_l2=0.0, vicreg_weight=0.0)
    result = train_mode_hold(
        ModeHoldRecipe(steps=2, particle_l2=0.0, vicreg_weight=0.0),
        gan_factory=candidate.make_loss, cap_factory=candidate.make_penalty,
    )
    assert "modes" in result
