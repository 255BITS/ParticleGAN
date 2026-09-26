"""Symplectic gradient adjustment at the published sign rule, |lambda| = 1."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from particlegan.dynamics import sga


ROOT = Path(__file__).resolve().parents[1]
MECHANISM = ROOT / "reports/toy100/gap-fill-20260925/sources/k3p"
# Flag-off hashes captured before this mechanism, same seeds and widths.
_TRAINER_OFF = "e2c4ff3b0e5ceef865dc12cb72d5744553db1eda6f663d20169c63e91504191a"
_MODE_HOLD_OFF = "3237b73ba34b4c4ecbfcefe122b46652d0037b2d6c25dee5e9a9804342c2bf28"


def _hash_modules(modules):
    digest = hashlib.sha256()
    for module in modules:
        for parameter in module.parameters():
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _trainer_hash(flag):
    from particlegan import GANTrainer, get_recipe

    if flag:
        os.environ["K3P_DYNAMICS"] = "sga"
    else:
        os.environ.pop("K3P_DYNAMICS", None)
    torch.manual_seed(0)
    recipe = get_recipe(num_particles=16, z_dim=2, batch_size=4, total_steps=5,
                        lr=0.00425, prior_lr_mult=2.0, lr_anneal_start=0.0,
                        lr_floor=1.0, network_lr_floor=1.0)
    generator = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2, inplace=True), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2, inplace=True), nn.Linear(8, 1))
    trainer = GANTrainer(recipe, generator, critic, seed=0)
    rng = torch.Generator().manual_seed(1)
    for _ in range(3):
        trainer.step(torch.randn(4, 2, generator=rng))
    return _hash_modules([trainer.G, trainer.D, trainer.prior])


def _mode_hold_hash(flag):
    from benchmarks.locked_shared.mode_hold import ModeHoldRecipe, train_mode_hold

    if flag:
        os.environ["K3P_DYNAMICS"] = "sga"
    else:
        os.environ.pop("K3P_DYNAMICS", None)
    saved = {}
    original = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        saved[id(optimizer)] = [p.detach().cpu().clone()
                                for group in optimizer.param_groups for p in group["params"]]
        return result

    torch.optim.Adam.step = step
    try:
        torch.manual_seed(0)
        train_mode_hold(ModeHoldRecipe(steps=2, n_particles=8), diagnostics=False)
    finally:
        torch.optim.Adam.step = original
        os.environ.pop("K3P_DYNAMICS", None)
    # Optimizer ids are not a stable order across processes. Sort by width.
    ordered = sorted(saved.values(), key=lambda tensors: (
        -sum(tensor.numel() for tensor in tensors), tuple(tensors[0].shape)))
    digest = hashlib.sha256()
    for tensors in ordered:
        for parameter in tensors:
            digest.update(parameter.contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_sign_rule_on_the_paper_examples():
    # Example 6 at (x, y) = (1, 0), ε = 0.1: unstable, so λ = -1.
    # adjusted = ξ + λ Aᵀξ = (-1.1, 0.9).
    epsilon = 0.1
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    f = -epsilon / 2 * x ** 2 - x * y
    g = -epsilon / 2 * y ** 2 + x * y
    grads = sga.adjust([([x], f), ([y], g)])
    assert torch.allclose(grads[0][0], torch.tensor(-1.1), atol=1e-5)
    assert torch.allclose(grads[1][0], torch.tensor(0.9), atol=1e-5)

    # Hamiltonian game L1 = xy, L2 = -xy. ⟨ξ, ∇H⟩ = 0, so λ = sign(ε) = +1
    # and Aᵀξ = (x, y). adjusted = (y + x, -x + y).
    x = torch.tensor(0.3, requires_grad=True)
    y = torch.tensor(-0.4, requires_grad=True)
    grads = sga.adjust([([x], x * y), ([y], -x * y)])
    assert torch.allclose(grads[0][0], torch.tensor(-0.1), atol=1e-5)
    assert torch.allclose(grads[1][0], torch.tensor(-0.7), atol=1e-5)

    # Potential game: A = 0, the adjustment is zero, λ's sign does not matter.
    x = torch.tensor(0.3, requires_grad=True)
    y = torch.tensor(-0.4, requires_grad=True)
    loss = 0.5 * (x ** 2 + y ** 2)
    grads = sga.adjust([([x], loss), ([y], loss)])
    assert torch.allclose(grads[0][0], x.detach(), atol=1e-6)
    assert torch.allclose(grads[1][0], y.detach(), atol=1e-6)


def test_flag_off_parameter_hash_matches_the_baseline():
    os.environ.pop("K3P_DYNAMICS", None)
    assert sga.active() is False
    assert _trainer_hash(False) == _TRAINER_OFF
    assert _mode_hold_hash(False) == _MODE_HOLD_OFF


def test_sga_is_deterministic_and_moves_the_baseline():
    assert _trainer_hash(True) == _trainer_hash(True)
    assert _trainer_hash(True) != _TRAINER_OFF
    first = _mode_hold_hash(True)
    assert first == _mode_hold_hash(True)
    assert first != _MODE_HOLD_OFF


def test_scaled_penalty_accepts_k3p_ema_critic():
    script = r"""
import torch
from torch import nn
from particlegan.grad_regularizers import GradientPenalty
original = GradientPenalty.penalty
import mechanism
critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1))
real, fake = torch.randn(4, 2), torch.randn(4, 2)
penalty = GradientPenalty(coeff=1.0, kappa=1.0)

def score(x):
    return critic(x).squeeze(-1)

got = penalty.penalty(score, real, fake, 1, False, ema_critic=score)
expected = original(penalty, score, real, fake, 1, False, ema_critic=score)
assert torch.allclose(got[0], expected[0])
print("ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=MECHANISM,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
