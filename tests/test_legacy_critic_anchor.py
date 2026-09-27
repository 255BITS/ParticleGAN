"""The archived K3P's EMA-critic anchor, kept for benchmark replays (benchmarks.legacy)."""
import copy

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from benchmarks.legacy.critic_optimizer import LegacyCriticAdam
from benchmarks.legacy.recipe import get_recipe
from particlegan import GANTrainer, InputNoise


def _recipe(**overrides):
    # The legacy schedule anneals over 20 steps, so the blend reaches its anchored phase.
    options = dict(num_particles=64, z_dim=2, batch_size=8, total_steps=20, d_guard_min_steps=2)
    return get_recipe(**{**options, **overrides})


def _reals(n, seed=1):
    rng = torch.Generator().manual_seed(seed)
    return [torch.randn(8, 2, generator=rng) * 2 for _ in range(n)]


def _buffers(module):
    return {k: v.detach().clone() for k, v in module.named_buffers()}


def test_legacy_trainer_keeps_its_ema_critic_without_bn_or_sn_mutation():
    torch.manual_seed(0)
    critic = nn.Sequential(spectral_norm(nn.Linear(2, 16)), nn.BatchNorm1d(16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    trainer = GANTrainer(_recipe(), G, critic)
    assert isinstance(trainer.opt_d, LegacyCriticAdam) and trainer.opt_d.ema_critic is not None
    for real in _reals(20):
        trainer.step(real)
    assert trainer.opt_d.record.anchor_started
    ema, live = trainer.opt_d.ema_critic, trainer.D
    # The EMA averages float buffers (not a copy of the live ones).
    assert not torch.equal(ema[1].running_mean, live[1].running_mean)
    assert ema[1].num_batches_tracked == live[1].num_batches_tracked
    # An anchor evaluation in train mode changes no EMA or live state.
    live.train()
    ema_modes = [m.training for m in ema.modules()]
    before_ema, before_live = _buffers(ema), _buffers(live)
    before_params = [p.detach().clone() for p in ema.parameters()]
    x = torch.randn(8, 2, requires_grad=True)
    torch.autograd.grad(trainer.opt_d.anchor(x).sum(), x)
    for key, value in _buffers(ema).items():
        assert torch.equal(value, before_ema[key]), key
    for key, value in _buffers(live).items():
        assert torch.equal(value, before_live[key]), key
    assert all(torch.equal(a, b) for a, b in zip(ema.parameters(), before_params))
    assert [m.training for m in ema.modules()] == ema_modes
    # The optimizer checkpoint carries the EMA critic and restores it.
    state = copy.deepcopy(trainer.opt_d.state_dict())
    assert set(state["regularizer"]) == {"record", "ema", "guard"}
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(1.0)
    trainer.opt_d.load_state_dict(state)
    assert all(torch.equal(a, b) for a, b in zip(ema.parameters(), before_params))


def test_legacy_input_noise_wrapper_penalty_uses_same_noise_on_ema():
    recipe = _recipe()
    torch.manual_seed(0)
    d = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d))
    penalty = recipe.make_critic_penalty(opt)
    stream = torch.Generator().manual_seed(3)
    noisy = InputNoise(d, 0.1, stream)
    opt.record.load_state_dict({**opt.record.state_dict(), "lr_max": 1.0, "lr_last": 0.0,
                                "observed_steps": 1})  # force the b phase (anchor in use)
    x = torch.randn(4, 2)
    before = stream.get_state()
    penalty(noisy, x, x + 1)
    # The anchor starts on its first b-phase call (prox == 0); the second draws EMA noise too.
    penalty(noisy, x, x + 1)
    draws = 0
    probe = torch.Generator().manual_seed(0)
    probe.set_state(before)
    while not torch.equal(probe.get_state(), stream.get_state()):
        torch.randn(4, 2, generator=probe)
        draws += 1
    assert draws == 2 + 3  # each call: live real + fake; the 2nd adds the EMA real draw
