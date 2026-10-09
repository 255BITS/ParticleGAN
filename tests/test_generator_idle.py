"""CPU checks for the opt-in generator idle rule and constant-LR overlay."""
import hashlib

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, learning_rate_scale, learning_rate_scales
from particlegan.generator_idle import generator_is_idle, release_generator_step
from benchmarks.legacy.recipe import LegacyRecipe
from benchmarks.toy100.schedule import policy_multipliers


# Short develop run captured before this rule existed. Disabled training must match it.
_DEVELOP_SHA256 = "75da077247f3f7e69e7ea32dbc16e16a53aabedc45937a20b3aa53ecfbcd7717"


def _walk(value, prefix, chunks):
    if isinstance(value, torch.Tensor):
        chunks.append(prefix.encode() + value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _walk(value[key], prefix + "/" + str(key), chunks)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, prefix + f"[{index}]", chunks)
    else:
        chunks.append(f"{prefix}={value!r}".encode())


def _fingerprint(trainer):
    chunks = []
    _walk(trainer.state_dict(), "state", chunks)
    for name, module in ("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior):
        for pname, parameter in module.named_parameters():
            grad = parameter.grad
            payload = b"" if grad is None else grad.detach().cpu().contiguous().numpy().tobytes()
            chunks.append(f"grad/{name}/{pname}".encode() + payload)
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def _short_run():
    torch.manual_seed(0)
    recipe = get_recipe(num_particles=8, z_dim=2, batch_size=4, total_steps=4)
    generator = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    trainer = GANTrainer(recipe, generator, critic)
    real = torch.randn(4, 2)
    for _ in range(3):
        trainer.step(real)
    return trainer


def test_disabled_path_matches_develop_and_does_not_score_the_gap(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("idle statistic evaluated while the rule is off")

    monkeypatch.setattr("particlegan.generator_idle.generator_is_idle", boom)
    first = _short_run()
    second = _short_run()
    assert _fingerprint(first) == _DEVELOP_SHA256
    assert _fingerprint(second) == _fingerprint(first)
    assert "generator_idle_se" not in first.state_dict()["recipe"]
    assert "constant_lr" not in first.state_dict()["recipe"]


def test_gap_rule():
    matched = torch.zeros(4, 1)
    assert generator_is_idle(matched, matched.clone(), 1.0)
    assert generator_is_idle(torch.zeros(4), torch.zeros(4), 0.0)
    separated = torch.full((4, 1), 3.0)
    assert not generator_is_idle(separated, torch.zeros(4, 1), 1.0)
    assert not generator_is_idle(torch.zeros(1), torch.zeros(1), 1.0)
    assert not generator_is_idle(torch.tensor([float("nan"), 0.0]), torch.zeros(2), 1.0)
    noisy = torch.tensor([0.1, -0.2, 0.05, 0.0])
    assert generator_is_idle(noisy, torch.zeros(4), 1.0)
    assert not generator_is_idle(noisy + 10, torch.zeros(4), 1.0)


class _Axis(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x[:, :1] + self.bias


def _axis_trainer(real, idle):
    torch.manual_seed(0)
    generator = nn.Linear(2, 2)
    critic = _Axis()
    recipe = get_recipe(num_particles=4, z_dim=2, batch_size=4, total_steps=2,
                        generator_idle_se=idle)
    trainer = GANTrainer(recipe, generator, critic)
    with torch.no_grad():
        generator.weight.zero_()
        generator.bias.zero_()
    before = [parameter.detach().clone() for parameter in generator.parameters()]
    trainer.step(real)
    return trainer, before


def _moved(trainer, before):
    return any(not torch.equal(parameter.detach(), saved)
               for parameter, saved in zip(trainer.G.parameters(), before))


def test_idle_skips_generator_step_and_resolved_gap_still_steps():
    zeros = torch.zeros(4, 2)
    idle_trainer, idle_before = _axis_trainer(zeros, 1.0)
    live_trainer, live_before = _axis_trainer(zeros, None)
    assert not _moved(idle_trainer, idle_before)
    assert _moved(live_trainer, live_before)
    assert len(idle_trainer.opt_d.state) == 1
    assert idle_trainer.opt_g.state == {}

    separated = torch.zeros(4, 2)
    separated[:, 0] = 10
    stepping, before = _axis_trainer(separated, 1.0)
    assert _moved(stepping, before)


def test_release_is_a_noop_until_the_scope_is_set():
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.ones(2)
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.0, 0.999))
    assert release_generator_step(optimizer, torch.zeros(4), torch.zeros(4)) is False
    assert torch.equal(parameter.grad, torch.ones(2))
    from particlegan.generator_idle import generator_idle_scope
    with generator_idle_scope(1.0):
        assert release_generator_step(optimizer, torch.zeros(4), torch.zeros(4)) is True
    assert parameter.grad is None
    parameter.grad = torch.ones(2)
    with generator_idle_scope(1.0):
        assert release_generator_step(optimizer, torch.full((4,), 5.0), torch.zeros(4)) is False
    assert torch.equal(parameter.grad, torch.ones(2))


def test_constant_lr_overlay_is_default_off_and_forces_unit_multipliers():
    base = get_recipe(total_steps=100, network_lr_horizon_cap=40, network_lr_floor=0.01, lr_floor=0.05)
    network, prior = learning_rate_scales(90, base)
    assert network == learning_rate_scale(90, 40, base.lr_anneal_start, 0.01)
    assert prior < 1.0
    assert learning_rate_scales(90, base.replace(constant_lr=True)) == (1.0, 1.0)
    assert policy_multipliers(90, 100, base.lr_anneal_start, base.lr_floor, 40,
                              network_lr_floor=0.01) == (network, prior)
    assert policy_multipliers(90, 100, base.lr_anneal_start, base.lr_floor, 40,
                              network_lr_floor=0.01, constant_lr=True) == (1.0, 1.0)
    assert "constant_lr" not in get_recipe().to_dict()
    assert "generator_idle_se" not in get_recipe().to_dict()
    assert get_recipe().replace(constant_lr=True).to_dict()["constant_lr"] is True
    assert get_recipe().replace(generator_idle_se=1).to_dict()["generator_idle_se"] == 1.0
    packet = LegacyRecipe().to_dict()
    assert "constant_lr" not in packet and "generator_idle_se" not in packet
    assert LegacyRecipe(constant_lr=True, generator_idle_se=1.0).to_dict()["generator_idle_se"] == 1.0
    import json
    from pathlib import Path
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    winner = json.loads(Path("configs/toy100/constraints_simple_regularization.json").read_text())
    plain, _, _ = declared_recipe(winner)
    assert plain.constant_lr is False and plain.generator_idle_se is None
    overlay = {**winner, "constant_lr": True, "generator_idle_se": 1}
    armed, _, _ = declared_recipe(overlay)
    assert armed.constant_lr is True and armed.generator_idle_se == 1.0


def test_constant_lr_and_idle_threshold_reject_bad_values():
    with pytest.raises(ValueError):
        get_recipe(constant_lr=1)
    with pytest.raises(ValueError):
        get_recipe(generator_idle_se=-1)
    with pytest.raises(ValueError):
        get_recipe(constant_lr=True, lr_schedule="constant", network_lr_horizon_cap=None)
    with pytest.raises(ValueError):
        get_recipe(constant_lr=True, continuous_policy="dv7")
