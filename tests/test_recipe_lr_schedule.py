"""Recipe-built optimizers own the LR schedule (``LRSchedule`` inside ``step()``)."""
import copy
import hashlib

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, NetworkLRTransition, get_recipe, learning_rate_scales


def _nets(seed=0):
    torch.manual_seed(seed)
    g = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    d = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    return g, d


def _step(opt, params):
    opt.zero_grad()
    sum(p.square().sum() for p in params).backward()
    opt.step()


def test_gan_trainer_is_bit_identical_to_the_caller_scaled_schedule():
    # Hash recorded on develop@92dc0319, where GANTrainer.step wrote
    # group["lr"] = initial_lr * learning_rate_scales(...) before each update.
    torch.set_num_threads(1)
    recipe = get_recipe(num_particles=64, z_dim=2, batch_size=16, total_steps=24,
                        network_lr_horizon_cap=10, d_guard_min_steps=2)
    g, d = _nets()
    trainer = GANTrainer(recipe, g, d, seed=3)
    data = torch.Generator().manual_seed(7)
    for _ in range(recipe.total_steps):
        out = trainer.step(torch.randn(16, 2, generator=data) * 0.5 + 1.0)
    digest = hashlib.sha256()
    for module in (trainer.G, trainer.D, trainer.prior, trainer.ema_G, trainer.ema_prior, trainer.ema_D):
        for tensor in module.state_dict().values():
            digest.update(tensor.detach().contiguous().numpy().tobytes())
    assert digest.hexdigest() == "a91ae034b45e54a6e6f2e9e7014f46e68352a4ad2698f7304547c2881329a783"
    assert (float(out["loss_d"]), float(out["loss_g"])) == (0.6109020113945007, 0.7791739702224731)
    assert trainer.opt_g.completed_steps == trainer.opt_d.completed_steps == recipe.total_steps


def test_optimizers_schedule_roles_from_their_own_count_without_compounding():
    recipe = get_recipe(num_particles=32, total_steps=10, network_lr_horizon_cap=4)
    g, d = _nets()
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    assert [group["role"] for group in opt_g.param_groups] == ["network", "prior"]
    base = [group["base_lr"] for group in opt_g.param_groups]
    assert base == [recipe.lr, recipe.lr * recipe.prior_lr_mult]
    for completed in range(14):  # 4 past the budget: rates hold at their floors
        opt_g.param_groups[0]["lr"] = 123.0  # a stray write is overwritten, never compounded
        _step(opt_g, [*g.parameters(), prior.z])
        network, prior_scale = learning_rate_scales(completed, recipe)
        assert opt_g.param_groups[0]["lr"] == base[0] * network
        assert opt_g.param_groups[1]["lr"] == base[1] * prior_scale
    assert network == recipe.network_lr_floor and prior_scale == recipe.lr_floor
    assert opt_g.completed_steps == 14 and opt_d.completed_steps == 0
    extra = nn.Parameter(torch.ones(3))
    opt_g.add_param_group({"params": [extra], "lr": 0.5})
    _step(opt_g, [extra])
    assert opt_g.param_groups[2]["base_lr"] == 0.5
    assert opt_g.param_groups[2]["lr"] == 0.5 * recipe.network_lr_floor
    with pytest.raises(ValueError, match="role"):
        opt_g.add_param_group({"params": [nn.Parameter(torch.ones(1))], "role": "critic"})


def test_schedule_state_resumes_exactly_and_rejects_mismatch():
    recipe = get_recipe(total_steps=8, network_lr_horizon_cap=4)
    g, d = _nets()
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d))
    for _ in range(3):
        _step(opt, list(d.parameters()))
    state = copy.deepcopy(opt.state_dict())
    assert state["lr_schedule"] == {"completed_steps": 3, "network_transition": None}
    _, d2 = _nets(1)
    resumed = recipe.make_critic_optimizer(d2, ema_critic=copy.deepcopy(d2))
    resumed.load_state_dict(state)
    assert resumed.completed_steps == 3
    _step(resumed, list(d2.parameters()))
    assert resumed.param_groups[0]["lr"] == recipe.lr * learning_rate_scales(3, recipe)[0]
    stale = {key: value for key, value in state.items() if key != "lr_schedule"}
    with pytest.raises(ValueError, match="lr_schedule"):
        resumed.load_state_dict(stale)


def test_network_transition_is_shared_and_checkpointed_by_the_optimizers():
    recipe = get_recipe(num_particles=32, total_steps=20, network_lr_horizon_cap=4)
    g, d = _nets()
    prior = recipe.make_prior()
    transition = NetworkLRTransition(decay_steps=4)
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d),
                                          network_transition=transition)
    for completed in range(12):
        if completed == 6:
            transition.mark_plateau(opt_d.completed_steps)
        _step(opt_d, list(d.parameters()))
        network, _ = learning_rate_scales(completed, recipe, network_transition=transition)
        assert opt_d.param_groups[0]["lr"] == recipe.lr * recipe.d_lr_mult * network
    assert network == recipe.network_lr_floor
    assert opt_d.state_dict()["lr_schedule"]["network_transition"] == {"decay_steps": 4, "start_step": 6}
    fresh = NetworkLRTransition(decay_steps=4)
    _, d2 = _nets(1)
    resumed = recipe.make_critic_optimizer(d2, ema_critic=copy.deepcopy(d2), network_transition=fresh)
    resumed.load_state_dict(opt_d.state_dict())
    assert fresh.start_step == 6


def test_gan_trainer_upgrades_checkpoints_saved_before_the_optimizer_schedule():
    recipe = get_recipe(num_particles=64, z_dim=2, batch_size=8, total_steps=10, network_lr_horizon_cap=4)
    g, d = _nets()
    trainer = GANTrainer(recipe, g, d)
    reals = torch.Generator().manual_seed(1)
    for _ in range(4):
        trainer.step(torch.randn(8, 2, generator=reals))
    state = trainer.state_dict()
    old = copy.deepcopy(state)
    for values in old["optimizers"]:
        values.pop("lr_schedule")
        for group in values["param_groups"]:
            group.pop("role"), group.pop("base_lr")
    continued, restored = trainer, GANTrainer(recipe, *_nets(1))
    restored.load_state_dict(old)
    assert restored.opt_g.completed_steps == restored.opt_d.completed_steps == 4
    batch = torch.randn(8, 2, generator=reals)
    a, b = continued.step(batch), restored.step(batch)
    assert torch.equal(a["loss_d"], b["loss_d"]) and torch.equal(a["loss_g"], b["loss_g"])
