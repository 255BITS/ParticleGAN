"""GANTrainer / recipe.make_critic_regularizer: K3P as the default, trainer-owned EMA critic and checkpoints."""
import copy

import pytest
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from particlegan import (GANTrainer, GradientPenalty, Recipe, get_recipe, learning_rate_scale,
                         learning_rate_scales, scale_learning_rates)


def _recipe(**overrides):
    # Small sparse table (64 rows, batch 8) so A2 damping is active; a short
    # network horizon so the K3P blend and floor are reached quickly.
    options = dict(num_particles=64, z_dim=2, batch_size=8, total_steps=20,
                   network_lr_horizon_cap=16, d_guard_min_steps=2)
    return get_recipe(**{**options, **overrides})


def _trainer(recipe=None, *, critic=None, seed=0):
    torch.manual_seed(seed)
    recipe = recipe or _recipe()
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    D = critic or nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    return GANTrainer(recipe, G, D, seed=seed)


def _reals(n, seed=1):
    rng = torch.Generator().manual_seed(seed)
    return [torch.randn(8, 2, generator=rng) * 2 for _ in range(n)]


def test_trainer_default_is_k3p():
    recipe = get_recipe()
    assert recipe == Recipe() and recipe.reg_arm == "k3p" and recipe.name == "k3p"
    trainer = _trainer()
    assert trainer.penalty.arm == "k3p" and trainer.penalty.lr_floor == trainer.recipe.network_lr_floor
    assert trainer.ema_D is not None and trainer.critic.guard is not None
    assert trainer.latent_damping is not None
    phases = [trainer.step(real, collect_stats=True)["penalty_stats"]["phase"] for real in _reals(20)]
    assert phases[0] == "a" and "blend" in phases and phases[-1] == "b"
    assert trainer.penalty.state_dict()["anchor_started"]
    assert trainer.latent_damping.started


def test_trainer_lr_scales_network_horizon_and_prior():
    trainer = _trainer()
    recipe, base = trainer.recipe, trainer.initial_lrs
    for step, real in enumerate(_reals(20)):
        trainer.step(real)
        network, prior = learning_rate_scales(step, recipe)
        assert network == learning_rate_scale(step, 16, recipe.lr_anneal_start, recipe.network_lr_floor)
        assert prior == learning_rate_scale(step, 20, recipe.lr_anneal_start, recipe.lr_floor)
        assert [g["lr"] for g in trainer.opt_g.param_groups] == [base[0][0] * network, base[0][1] * prior]
        assert trainer.opt_d.param_groups[0]["lr"] == base[1][0] * network
    # At the network floor the blend weight is exactly zero; the prior is still annealing.
    assert trainer.penalty.blend_weight() == 0.0
    assert learning_rate_scales(17, recipe)[0] == recipe.network_lr_floor
    # No cap means the full budget; no network floor means lr_floor.
    same = _recipe(network_lr_horizon_cap=None, network_lr_floor=None)
    assert all(a == b for a, b in (learning_rate_scales(s, same) for s in range(20)))


def test_trainer_constant_lr_s_one_no_ema_forward():
    recipe = _recipe(lr_floor=1.0, network_lr_floor=1.0)
    trainer = _trainer(recipe)
    calls = []
    trainer.ema_D.register_forward_pre_hook(lambda module, inputs: calls.append(1))
    for real in _reals(20):
        stats = trainer.step(real, collect_stats=True)["penalty_stats"]
        assert stats["s"] == 1.0 and stats["phase"] == "a"
    assert calls == [] and not trainer.penalty.state_dict()["anchor_started"]


def _run(trainer, reals):
    return [trainer.step(real, generator_real=lambda r=real: r.flip(0), collect_stats=True) for real in reals]


def test_trainer_k3p_resume_bit_exact():
    reals = _reals(20)
    full = _trainer()
    full_out = _run(full, reals)
    phases = [o["penalty_stats"]["phase"] for o in full_out]
    split = phases.index("blend") + 1
    assert phases[split] == "blend" and phases[-1] == "b"  # resume inside the blend

    first = _trainer()
    _run(first, reals[:split])
    checkpoint = first.state_dict()
    assert checkpoint["schema"] == 2 and checkpoint["k3p"]["critic"]["penalty"]["anchor_started"]
    assert checkpoint["k3p"]["latent"]["state"]["started"] and "noise_generator" in checkpoint["streams"]
    resumed = _trainer(seed=5)  # different construction randomness; the checkpoint wins
    resumed.load_state_dict(checkpoint)
    resumed_out = _run(resumed, reals[split:])
    for a, b in zip(resumed_out, full_out[split:]):
        for key in ("loss_d", "loss_g", "penalty"):
            assert torch.equal(a[key], b[key]), key
        assert a["penalty_stats"] == b["penalty_stats"]
    left, right = resumed.state_dict(), full.state_dict()
    for name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for key, value in left["models"][name].items():
            assert torch.equal(value, right["models"][name][key]), (name, key)
    for key, value in left["k3p"]["critic"]["ema"].items():
        assert torch.equal(value, right["k3p"]["critic"]["ema"][key]), key
    assert left["k3p"]["critic"]["penalty"] == right["k3p"]["critic"]["penalty"]
    assert left["k3p"]["critic"]["guard"] == right["k3p"]["critic"]["guard"]
    assert torch.equal(left["k3p"]["latent"]["history"], right["k3p"]["latent"]["history"])
    for name, value in left["streams"].items():
        assert torch.equal(value, right["streams"][name]), name


def _buffers(module):
    return {k: v.detach().clone() for k, v in module.named_buffers()}


def test_trainer_ema_critic_buffers_no_bn_or_sn_mutation():
    critic = nn.Sequential(spectral_norm(nn.Linear(2, 16)), nn.BatchNorm1d(16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    trainer = _trainer(critic=critic)
    for real in _reals(20):
        trainer.step(real)
    assert trainer.penalty.state_dict()["anchor_started"]
    ema, live = trainer.ema_D, trainer.D
    # The EMA averages float buffers (not a copy of the live ones).
    assert not torch.equal(ema[1].running_mean, live[1].running_mean)
    assert ema[1].num_batches_tracked == live[1].num_batches_tracked
    # An anchor evaluation in train mode changes no EMA or live state.
    live.train()
    ema_modes = [m.training for m in ema.modules()]
    before_ema, before_live = _buffers(ema), _buffers(live)
    before_params = [p.detach().clone() for p in ema.parameters()]
    x = torch.randn(8, 2, requires_grad=True)
    torch.autograd.grad(trainer.critic.anchor(x).sum(), x)
    for key, value in _buffers(ema).items():
        assert torch.equal(value, before_ema[key]), key
    for key, value in _buffers(live).items():
        assert torch.equal(value, before_live[key]), key
    assert all(torch.equal(a, b) for a, b in zip(ema.parameters(), before_params))
    assert [m.training for m in ema.modules()] == ema_modes


def test_schema1_checkpoint_rejected():
    trainer = _trainer()
    state = trainer.state_dict()
    state["schema"] = 1
    with pytest.raises(ValueError, match="schema-1"):
        trainer.load_state_dict(state)
    bad = trainer.state_dict()
    bad["k3p"]["critic"]["penalty"] = {}
    with pytest.raises(ValueError, match="K3P state"):
        trainer.load_state_dict(bad)


def test_multiple_critics_each_own_k3p_critic_no_global_hooks():
    from torch.optim import optimizer as optim_module
    hooks = (len(optim_module._global_optimizer_pre_hooks), len(optim_module._global_optimizer_post_hooks))
    recipe = _recipe()
    torch.manual_seed(0)
    critics = [nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)) for _ in range(2)]
    optimizers = [torch.optim.Adam(c.parameters(), lr=1e-2, betas=recipe.betas) for c in critics]
    bundles = [recipe.make_critic_regularizer(c, o) for c, o in zip(critics, optimizers)]
    for step, real in enumerate(_reals(12), start=1):
        fake = real + 1.0
        for k, (critic, opt, bundle) in enumerate(zip(critics, optimizers, bundles)):
            # Critic 0 anneals, critic 1 keeps a constant LR.
            opt.param_groups[0]["lr"] = 1e-2 * (learning_rate_scale(step - 1, 8, .5, .01) if k == 0 else 1.)
            loss = critic(real).mean() - critic(fake).mean() + bundle.penalty(critic, real, fake, step)[0]
            opt.zero_grad()
            loss.backward()
            bundle.step()
    assert bundles[0].diagnostics()["blend_weight"] == 0.0 and bundles[1].diagnostics()["blend_weight"] == 1.0
    assert bundles[0].regularizer.state_dict()["anchor_started"]
    assert not bundles[1].regularizer.state_dict()["anchor_started"]
    assert (len(optim_module._global_optimizer_pre_hooks), len(optim_module._global_optimizer_post_hooks)) == hooks


class _TwoRoles(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({r: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
                                    for r in ("joint", "marginal")})

    def critic_for(self, role):
        return self.heads[role]


def test_shared_module_multi_role_k3p_critic():
    recipe = _recipe()
    torch.manual_seed(0)
    d = _TwoRoles()
    opt = torch.optim.Adam(d.parameters(), lr=1e-2, betas=recipe.betas)
    k3p = recipe.make_critic_regularizer(d, opt)
    for step, real in enumerate(_reals(12), start=1):
        opt.param_groups[0]["lr"] = 1e-2 * learning_rate_scale(step - 1, 8, .5, .01)
        fake, loss, phases = real + 1.0, 0.0, []
        for role in ("joint", "marginal"):
            critic = d.critic_for(role)
            pen, stats = k3p.penalty(lambda x: critic(x), real, fake, step, collect_stats=True,
                                     ema_critic=k3p.ema_critic(lambda m, x: m.critic_for(role)(x)))
            phases.append(stats["phase"])
            loss = loss + critic(real).mean() - critic(fake).mean() + pen
        opt.zero_grad()
        loss.backward()
        k3p.step()
        assert phases[0] == phases[1]
    assert phases == ["b", "b"] and k3p.regularizer.state_dict()["observed_steps"] == 12
    for role in ("joint", "marginal"):
        live, ema = d.critic_for(role)[0].weight, k3p.ema.critic_for(role)[0].weight
        assert not torch.equal(live, ema)
    with pytest.raises(RuntimeError, match="without an optimizer"):
        recipe.make_critic_regularizer(d).step()


def test_gradient_penalty_constructor_defaults_to_k3p():
    assert GradientPenalty().arm == "k3p"
    assert GradientPenalty(arm="b_cap").arm == "b_cap"


def test_scale_learning_rates_drives_k3p_to_its_floor():
    # A custom loop using scale_learning_rates gives D the network schedule, so
    # the critic LR reaches the same floor f K3P blends against (s == 0).
    recipe = _recipe()
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior)
    base = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    k3p = recipe.make_critic_regularizer(d, opt_d)
    for step, real in enumerate(_reals(recipe.total_steps), start=1):
        network, prior_scale = scale_learning_rates(step - 1, recipe, (opt_g, opt_d), base, prior)
        assert (network, prior_scale) == learning_rate_scales(step - 1, recipe)
        assert opt_g.param_groups[0]["lr"] == base[0][0] * network
        assert opt_g.param_groups[1]["lr"] == base[0][1] * prior_scale
        assert opt_d.param_groups[0]["lr"] == base[1][0] * network
        loss = d(real).mean() - d(real + 1).mean() + k3p.penalty(d, real, real + 1, step)[0]
        opt_d.zero_grad()
        loss.backward()
        k3p.step()
    assert k3p.diagnostics()["blend_weight"] == 0.0


def _sparse_prior_run(recipe, use_factory, steps=6):
    from particlegan.k3p import LatentRowDamping
    torch.manual_seed(0)
    g = nn.Linear(2, 2)
    prior = recipe.make_prior()
    opt_g, _ = recipe.make_optimizers(g, nn.Linear(2, 1), prior)
    reg = None
    if use_factory:
        reg = recipe.make_generator_regularizer(opt_g, latent_table=prior.z)
    elif recipe.latent_damping_max_rate > 0:
        reg = LatentRowDamping(prior.z, torch.zeros_like(prior.z.detach()),
                               max_rate=recipe.latent_damping_max_rate)
    for step in range(steps):
        z, _ = prior.sample(8, generator=torch.Generator().manual_seed(step))
        opt_g.zero_grad()
        g(z).square().sum().backward()
        if use_factory:
            reg.step()
        elif recipe.latent_damping_max_rate > 0:
            with reg.around(opt_g):
                opt_g.step()
        else:
            opt_g.step()
    return prior.z.detach().clone(), reg


def test_generator_regularizer_matches_latent_damping_and_resumes():
    recipe = _recipe()
    via_recipe, reg = _sparse_prior_run(recipe, True)
    direct, damping = _sparse_prior_run(recipe, False)
    assert torch.equal(via_recipe, direct)
    assert damping.started and reg.state_dict()["latent"]["state"] == damping.state_dict()
    assert reg.state_dict()["direct"] is None
    state = copy.deepcopy(reg.state_dict())
    reg.load_state_dict(state)
    assert torch.equal(reg.latent_history, state["latent"]["history"])
    with pytest.raises(ValueError):
        reg.load_state_dict({"latent": None, "direct": None})
    # Disabled damping: step() is exactly optimizer.step().
    off = recipe.replace(latent_damping_max_rate=0.0)
    plain, plain_reg = _sparse_prior_run(off, True)
    assert torch.equal(plain, _sparse_prior_run(off, False)[0])
    assert plain_reg.state_dict() == {"latent": None, "direct": None}


def test_generator_regularizer_direct_particles():
    from particlegan.k3p import DirectParticleResponse
    recipe = _recipe()
    results = []
    for use_factory in (True, False):
        torch.manual_seed(0)
        particles = nn.Parameter(torch.randn(16, 2))
        opt = torch.optim.Adam([particles], lr=1e-2, betas=(0.0, 0.999))
        if use_factory:
            reg = recipe.make_generator_regularizer(opt, direct_particles=[particles])
        else:
            resp = DirectParticleResponse([particles], torch.zeros(particles.numel()),
                                          betas=recipe.direct_particle_betas)
        for step in range(5):
            opt.zero_grad()
            (particles - torch.tensor([1.0, -1.0]) * step).square().sum().backward()
            if use_factory:
                reg.before_step()
                opt.step()
                reg.after_step()
            else:
                with resp.around(opt):
                    opt.step()
        results.append(particles.detach().clone())
    assert torch.equal(*results)
    assert reg.state_dict()["direct"]["state"] == {"started": True}


def test_api_doc_example_runs():
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[1] / "docs/api.md").read_text()
    block = next(b for b in re.findall(r"```python\n(.*?)```", text, re.S)
                 if "make_critic_regularizer(D, opt_d)" in b and "for step in range" in b)
    code = block.replace("num_classes=2)", "num_classes=2, total_steps=3, batch_size=32)", 1)
    assert code != block
    exec(compile(code, "docs/api.md", "exec"), {"__name__": "__api_example__"})
