"""GANTrainer with the KA2 default: private controller, EMA critic and checkpoints."""
import copy

import pytest
import torch
from particlegan import (
    GANTrainer,
    InputNoise,
    NetworkLRTransition,
    Recipe,
    get_recipe,
    learning_rate_scale,
    learning_rate_scales,
    scale_learning_rates,
)
from particlegan.ka2 import KA2GradientPenalty
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

@pytest.fixture(autouse=True)
def short_acquisition_phase(monkeypatch):
    # Exercise both phases in small CPU loops. The exact 799/800 boundary is
    # covered by the frozen-source KA2 tests; production settings are unchanged.
    monkeypatch.setattr("particlegan.ka2.WARMUP_CALLS", 4)

def _recipe(**overrides):
    # Small sparse table (64 rows, batch 8) so A2 damping is active; a short
    # network horizon so schedules reach their floors independently of KA2.
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


def test_trainer_default_is_ka2():
    recipe = get_recipe()
    assert recipe == Recipe() and recipe.name == "ka2"
    trainer = _trainer()
    assert isinstance(trainer.penalty.regularizer, KA2GradientPenalty)
    assert trainer.ema_D is not None and trainer.opt_d.guard is not None
    assert trainer.latent_damping is not None
    phases = [trainer.step(real, collect_stats=True)["penalty_stats"]["phase"] for real in _reals(20)]
    assert phases[:3] == ["a"] * 3 and phases[3:] == ["blend"] * 17
    assert trainer.opt_d.state_dict()["regularizer"]["record"]["anchor_started"]
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
    # The network floor does not change KA2's fixed blend.
    assert trainer.penalty.diagnostics()["blend_weight"] == 0.5
    assert learning_rate_scales(17, recipe)[0] == recipe.network_lr_floor
    # No cap means the full budget; no network floor means lr_floor.
    same = _recipe(network_lr_horizon_cap=None, network_lr_floor=None)
    assert all(a == b for a, b in (learning_rate_scales(s, same) for s in range(20)))


def test_trainer_constant_lr_still_blends_and_evaluates_ema():
    recipe = _recipe(lr_floor=1.0, network_lr_floor=1.0)
    trainer = _trainer(recipe)
    calls = []
    trainer.ema_D.register_forward_pre_hook(lambda module, inputs: calls.append(1))
    for step, real in enumerate(_reals(20), start=1):
        stats = trainer.step(real, collect_stats=True)["penalty_stats"]
        assert stats["s"] == 0.5
        assert stats["phase"] == ("a" if step < 4 else "blend")
    # Call 4 initializes the anchor; each later blended call evaluates it.
    assert len(calls) == 16 and trainer.opt_d.record.anchor_started


def _run(trainer, reals):
    return [trainer.step(real, generator_real=lambda r=real: r.flip(0), collect_stats=True) for real in reals]


def test_trainer_ka2_resume_bit_exact():
    reals = _reals(40)
    recipe = _recipe(total_steps=40)
    full = _trainer(recipe)
    full_out = _run(full, reals)
    phases = [o["penalty_stats"]["phase"] for o in full_out]
    split = 30
    assert phases[split] == phases[-1] == "blend"  # resume with mature surprise history

    first = _trainer(recipe)
    _run(first, reals[:split])
    checkpoint = first.state_dict()
    opt_g_state, opt_d_state = (state["regularizer"] for state in checkpoint["optimizers"])
    assert checkpoint["schema"] == 4 and opt_d_state["record"]["anchor_started"]
    assert len(opt_d_state["record"]["sur_hist"]) >= 25
    assert opt_d_state["record"]["sur_base"] is not None
    assert opt_g_state["latent"]["state"]["started"] and "noise_generator" in checkpoint["streams"]
    resumed = _trainer(recipe, seed=5)  # different construction randomness; the checkpoint wins
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
    (left_g, left_d), (right_g, right_d) = ([o["regularizer"] for o in x["optimizers"]] for x in (left, right))
    for key, value in left_d["ema"].items():
        assert torch.equal(value, right_d["ema"][key]), key
    assert left_d["record"] == right_d["record"] and left_d["guard"] == right_d["guard"]
    assert torch.equal(left_g["latent"]["history"], right_g["latent"]["history"])
    for name, value in left["streams"].items():
        assert torch.equal(value, right["streams"][name]), name


def _buffers(module):
    return {k: v.detach().clone() for k, v in module.named_buffers()}


def test_trainer_ema_critic_buffers_no_bn_or_sn_mutation():
    critic = nn.Sequential(spectral_norm(nn.Linear(2, 16)), nn.BatchNorm1d(16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    trainer = _trainer(critic=critic)
    for real in _reals(19):
        trainer.step(real)
    assert trainer.opt_d.record.anchor_started
    # Exercise a nonzero adaptive EMA update before checking its buffer policy.
    trainer.opt_d.record.alpha = .5
    trainer.step(_reals(1, seed=7)[0])
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
    torch.autograd.grad(trainer.opt_d.anchor(x).sum(), x)
    for key, value in _buffers(ema).items():
        assert torch.equal(value, before_ema[key]), key
    for key, value in _buffers(live).items():
        assert torch.equal(value, before_live[key]), key
    assert all(torch.equal(a, b) for a, b in zip(ema.parameters(), before_params))
    assert [m.training for m in ema.modules()] == ema_modes


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_older_formulation_checkpoint_rejected(schema):
    trainer = _trainer()
    state = trainer.state_dict()
    state["schema"] = schema
    with pytest.raises(ValueError, match=f"schema-{schema}"):
        trainer.load_state_dict(state)
    bad = trainer.state_dict()
    bad["optimizers"][1]["regularizer"]["record"] = {}
    with pytest.raises(ValueError, match="optimizer state"):
        trainer.load_state_dict(bad)


def test_multiple_critics_each_own_ka2_controller_no_global_hooks():
    from torch.optim import optimizer as optim_module
    hooks = (len(optim_module._global_optimizer_pre_hooks), len(optim_module._global_optimizer_post_hooks))
    recipe = _recipe()
    torch.manual_seed(0)
    critics = [nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)) for _ in range(2)]
    optimizers = [recipe.make_critic_optimizer(c, ema_critic=copy.deepcopy(c), lr=1e-2) for c in critics]
    penalties = [recipe.make_critic_penalty(o) for o in optimizers]
    for step, real in enumerate(_reals(12), start=1):
        fake = real + 1.0
        for k, (critic, opt, penalty) in enumerate(zip(critics, optimizers, penalties)):
            # Critic 0 anneals, critic 1 keeps a constant LR.
            opt.param_groups[0]["lr"] = 1e-2 * (learning_rate_scale(step - 1, 8, .5, .01) if k == 0 else 1.)
            loss = critic(real).mean() - critic(fake).mean() + penalty(critic, real, fake)
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert all(penalty.diagnostics()["blend_weight"] == .5 for penalty in penalties)
    assert all(optimizer.record.anchor_started for optimizer in optimizers)
    assert optimizers[0].record is not optimizers[1].record
    assert optimizers[0].record.sur_hist != optimizers[1].record.sur_hist
    assert (len(optim_module._global_optimizer_pre_hooks), len(optim_module._global_optimizer_post_hooks)) == hooks
    with pytest.raises(ValueError, match="EMA"):
        recipe.make_critic_penalty(recipe.make_critic_optimizer(critics[0]))
    with pytest.raises(TypeError, match="make_critic_optimizer"):
        recipe.make_critic_penalty(torch.optim.Adam(critics[0].parameters()))
    with pytest.raises(TypeError, match="paired"):
        penalties[0](critics[1], real, fake)


class _TwoRoles(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({r: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
                                    for r in ("joint", "marginal")})

    def critic_for(self, role):
        return self.heads[role]


def test_shared_module_multi_role_ka2_critic():
    recipe = _recipe()
    torch.manual_seed(0)
    d = _TwoRoles()
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d), lr=1e-2)
    penalty = recipe.make_critic_penalty(opt, collect_stats=True)
    for step, real in enumerate(_reals(12), start=1):
        opt.param_groups[0]["lr"] = 1e-2 * learning_rate_scale(step - 1, 8, .5, .01)
        fake, loss, phases = real + 1.0, 0.0, []
        for role in ("joint", "marginal"):
            critic = d.critic_for(role)
            pen = penalty(critic, real, fake)  # the EMA uses the same-named role submodule
            phases.append(penalty.last_stats["phase"])
            loss = loss + critic(real).mean() - critic(fake).mean() + pen
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Applied penalty calls advance acquisition, including separate roles.
        assert phases == (["a", "a"] if step == 1 else ["a", "blend"] if step == 2 else ["blend", "blend"])
    assert phases == ["blend", "blend"] and opt.record.observed_steps == 12
    assert opt.record.calls == 24
    for role in ("joint", "marginal"):
        live, ema = d.critic_for(role)[0].weight, penalty.ema_critic.critic_for(role)[0].weight
        assert not torch.equal(live, ema)


class _Conditional(nn.Module):
    """A conditional critic returning (logits, features), as the DDGAN critics do."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        self.embed = nn.Embedding(3, 1)

    def forward(self, x, labels, *, t):
        h = torch.cat([x, t.expand(len(x), 1), self.embed(labels)], dim=1)
        return self.body(h), h


def test_conditional_penalty_forwards_conditioning_to_critic_and_ema():
    recipe = _recipe()
    torch.manual_seed(0)
    d = _Conditional()
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d), lr=1e-2)
    penalty = recipe.make_critic_penalty(opt, collect_stats=True)
    ref_d = copy.deepcopy(d)
    ref_opt = recipe.make_critic_optimizer(ref_d, ema_critic=copy.deepcopy(ref_d), lr=1e-2)
    ref_anchor = ref_opt.anchor
    # Explicit conditional callables are the reference for the public wrapper's
    # argument and output forwarding; formula parity has separate frozen tests.
    ref = KA2GradientPenalty(record=ref_opt.record, **recipe._penalty_options())
    labels, t = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), torch.tensor([[0.3]])
    for step, real in enumerate(_reals(12), start=1):
        fake = real + 1.0
        for group in (*opt.param_groups, *ref_opt.param_groups):
            group["lr"] = 1e-2 * learning_rate_scale(step - 1, 8, .5, .01)
        pen = penalty(d, real, fake, labels, t=t)
        ref_pen, _ = ref.penalty(lambda x: ref_d(x, labels, t=t)[0], real, fake, step,
                                 ema_critic=lambda x: ref_anchor.forward(lambda m, y: m(y, labels, t=t)[0], x))
        assert torch.equal(pen, ref_pen), step
        for critic, o, p in ((d, opt, pen), (ref_d, ref_opt, ref_pen)):
            loss = critic(real, labels, t=t)[0].mean() - critic(fake, labels, t=t)[0].mean() + p
            o.zero_grad()
            loss.backward()
        opt.step()
        ref_opt.step()
    assert penalty.last_stats["phase"] == "blend"
    assert all(torch.equal(a, b) for a, b in zip(d.parameters(), ref_d.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(opt.ema_critic.parameters(), ref_anchor.ema_critic.parameters()))
    # output= selects the logits from other layouts, at construction.
    swapped = recipe.make_critic_penalty(opt, output=lambda out: out[0])
    assert torch.equal(swapped(d, real, fake, labels, t=t), penalty(d, real, fake, labels, t=t))


def test_input_noise_wrapper_penalty_uses_same_noise_on_ema():
    recipe = _recipe()
    torch.manual_seed(0)
    d = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d))
    penalty = recipe.make_critic_penalty(opt)
    stream = torch.Generator().manual_seed(3)
    noisy = InputNoise(d, 0.1, stream)
    opt.record.load_state_dict({**opt.record.state_dict(), "calls": 3,
                                "observed_steps": 1})  # next call blends and starts the anchor
    x = torch.randn(4, 2)
    before = stream.get_state()
    penalty(noisy, x, x + 1)
    # The anchor starts on its first blended call; the second draws EMA noise too.
    penalty(noisy, x, x + 1)
    draws = 0
    probe = torch.Generator().manual_seed(0)
    probe.set_state(before)
    while not torch.equal(probe.get_state(), stream.get_state()):
        torch.randn(4, 2, generator=probe)
        draws += 1
    assert draws == 2 + 3  # each call: live real + fake; the 2nd adds the EMA real draw


def test_plain_torch_checkpoint_resumes_exactly():
    """The usual ``torch.save({'G', 'D', 'opt_g', 'opt_d'})`` pattern restores KA2 state."""
    recipe = _recipe()
    reals = _reals(20)

    def build():
        torch.manual_seed(0)
        g = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
        d = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
        prior = recipe.make_prior()
        opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
        base = [[group["lr"] for group in o.param_groups] for o in (opt_g, opt_d)]
        return g, d, prior, opt_g, opt_d, recipe.make_critic_penalty(opt_d), base

    def run(parts, steps):
        g, d, prior, opt_g, opt_d, penalty, base = parts
        out = []
        for step in steps:
            scale_learning_rates(step, recipe, (opt_g, opt_d), base, prior)
            real = reals[step]
            z, _ = prior.sample(8, generator=torch.Generator().manual_seed(100 + step))
            fake = g(z)
            d_loss = nn.functional.softplus(d(fake.detach())).mean() + nn.functional.softplus(-d(real)).mean()
            d_loss = d_loss + penalty(d, real, fake.detach())
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            g_loss = nn.functional.softplus(-d(fake)).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            out.append((d_loss.detach(), g_loss.detach()))
        return out

    full = build()
    full_out = run(full, range(20))
    first = build()
    run(first, range(15))
    g, d, prior, opt_g, opt_d, _, _ = first
    checkpoint = copy.deepcopy({"G": g.state_dict(), "D": d.state_dict(), "prior": prior.state_dict(),
                                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()})
    resumed = build()
    g, d, prior, opt_g, opt_d, _, _ = resumed
    with torch.no_grad():  # scramble everything the checkpoint must restore
        for module in (g, d, prior):
            for p in module.parameters():
                p.add_(1.0)
    g.load_state_dict(checkpoint["G"])
    d.load_state_dict(checkpoint["D"])
    prior.load_state_dict(checkpoint["prior"])
    opt_g.load_state_dict(checkpoint["opt_g"])
    opt_d.load_state_dict(checkpoint["opt_d"])
    assert opt_d.record.anchor_started and opt_g.latent_damping.started
    for a, b in zip(run(resumed, range(15, 20)), full_out[15:]):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for x, y in zip(resumed[4].ema_critic.parameters(), full[4].ema_critic.parameters()):
        assert torch.equal(x, y)
    assert torch.equal(resumed[3].latent_history, full[3].latent_history)


def test_recipe_optimizers_are_ordinary_adam():
    from torch.optim.lr_scheduler import LambdaLR
    recipe = _recipe()
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), nn.Linear(2, 1)
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    assert isinstance(opt_g, torch.optim.Adam) and isinstance(opt_d, torch.optim.Adam)
    schedulers = [LambdaLR(opt_d, lambda epoch: 0.5 ** epoch)]
    calls = []
    opt_d.register_step_post_hook(lambda *args: calls.append(1))
    x = torch.randn(8, 2)

    def closure():
        opt_d.zero_grad()
        loss = d(x).square().mean()
        loss.backward()
        return loss
    assert opt_d.step(closure) is not None and calls == [1] and opt_d.record.observed_steps == 1
    for scheduler in schedulers:
        scheduler.step()
    assert opt_d.param_groups[0]["lr"] == recipe.lr * recipe.d_lr_mult * 0.5
    with pytest.raises(ValueError, match="regularizer"):
        opt_d.load_state_dict(torch.optim.Adam(d.parameters()).state_dict())


def test_scale_learning_rates_keeps_ka2_blend_independent_of_floor():
    # A custom loop applies the network schedule without changing the .5 blend.
    recipe = _recipe()
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    base = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    penalty = recipe.make_critic_penalty(opt_d)
    for step, real in enumerate(_reals(recipe.total_steps), start=1):
        network, prior_scale = scale_learning_rates(step - 1, recipe, (opt_g, opt_d), base, prior)
        assert (network, prior_scale) == learning_rate_scales(step - 1, recipe)
        assert opt_g.param_groups[0]["lr"] == base[0][0] * network
        assert opt_g.param_groups[1]["lr"] == base[0][1] * prior_scale
        assert opt_d.param_groups[0]["lr"] == base[1][0] * network
        loss = d(real).mean() - d(real + 1).mean() + penalty(d, real, real + 1)
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
    assert penalty.diagnostics()["blend_weight"] == 0.5


def test_caller_marked_network_transition_keeps_prior_schedule_and_resumes():
    recipe = _recipe(total_steps=20, network_lr_horizon_cap=4)
    transition = NetworkLRTransition(decay_steps=4)
    assert learning_rate_scales(8, recipe, network_transition=transition) == (
        1.0, learning_rate_scales(8, recipe)[1])
    transition.mark_plateau(8)
    transition.mark_plateau(8)
    with pytest.raises(ValueError, match="already marked"):
        transition.mark_plateau(9)
    for step, expected in ((8, 1.0), (10, (1 + recipe.network_lr_floor) / 2),
                           (12, recipe.network_lr_floor), (18, recipe.network_lr_floor)):
        network, prior = learning_rate_scales(step, recipe, network_transition=transition)
        assert network == pytest.approx(expected)
        assert prior == learning_rate_scales(step, recipe)[1]
    resumed = NetworkLRTransition(decay_steps=4)
    resumed.load_state_dict(transition.state_dict())
    assert [learning_rate_scales(step, recipe, network_transition=resumed)
            for step in range(8, 20)] == [learning_rate_scales(step, recipe, network_transition=transition)
                                          for step in range(8, 20)]
    with pytest.raises(ValueError, match="decay_steps differ"):
        NetworkLRTransition(decay_steps=5).load_state_dict(transition.state_dict())


def test_caller_marked_transition_changes_rates_without_changing_ka2_blend():
    recipe = _recipe(total_steps=20, network_lr_horizon_cap=4)
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    base = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    penalty = recipe.make_critic_penalty(opt_d, collect_stats=True)
    transition = NetworkLRTransition(decay_steps=4)
    phases = []
    for step, real in enumerate(_reals(14), start=1):
        if step == 9:
            transition.mark_plateau(step - 1)
        network, prior_scale = scale_learning_rates(
            step - 1, recipe, (opt_g, opt_d), base, prior, network_transition=transition)
        assert opt_d.param_groups[0]["lr"] == base[1][0] * network
        assert opt_g.param_groups[0]["lr"] == base[0][0] * network
        assert opt_g.param_groups[1]["lr"] == base[0][1] * prior_scale
        assert prior_scale == learning_rate_scales(step - 1, recipe)[1]
        if step <= 9:
            assert network == 1.0
        loss = d(real).mean() - d(real + 1).mean() + penalty(d, real, real + 1)
        phases.append(penalty.last_stats.get("phase"))
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
    assert network == recipe.network_lr_floor
    # KA2 warms up by penalty calls, independent of the caller's LR transition.
    assert phases[:3] == ["a"] * 3 and phases[3:] == ["blend"] * 11
    assert penalty.diagnostics()["blend_weight"] == 0.5


def _sparse_prior_run(recipe, use_factory, steps=6):
    from particlegan.k3p import LatentRowDamping
    torch.manual_seed(0)
    g = nn.Linear(2, 2)
    prior = recipe.make_prior()
    plain = recipe.replace(latent_damping_max_rate=0.0)
    opt_g, _ = (recipe if use_factory else plain).make_optimizers(g, nn.Linear(2, 1), prior)
    damping = None
    if not use_factory and recipe.latent_damping_max_rate > 0:
        damping = LatentRowDamping(prior.z, torch.zeros_like(prior.z.detach()),
                                   max_rate=recipe.latent_damping_max_rate)
    for step in range(steps):
        z, _ = prior.sample(8, generator=torch.Generator().manual_seed(step))
        opt_g.zero_grad()
        g(z).square().sum().backward()
        if damping is not None:
            with damping.around(opt_g):
                opt_g.step()
        else:
            opt_g.step()
    return prior.z.detach().clone(), (opt_g if use_factory else damping)


def test_generator_optimizer_matches_latent_damping_and_resumes():
    recipe = _recipe()
    opt_g_z, opt_g = _sparse_prior_run(recipe, True)
    direct, damping = _sparse_prior_run(recipe, False)
    assert torch.equal(opt_g_z, direct)
    extra = opt_g.state_dict()["regularizer"]
    assert damping.started and extra["latent"]["state"] == damping.state_dict()
    assert extra["direct"] is None
    state = copy.deepcopy(opt_g.state_dict())
    opt_g.load_state_dict(state)
    assert torch.equal(opt_g.latent_history, state["regularizer"]["latent"]["history"])
    with pytest.raises(ValueError):
        opt_g.load_state_dict({**state, "regularizer": {"latent": None, "direct": None}})
    # Disabled damping: step() is exactly Adam.step().
    off = recipe.replace(latent_damping_max_rate=0.0)
    plain, plain_opt = _sparse_prior_run(off, True)
    assert torch.equal(plain, _sparse_prior_run(off, False)[0])
    assert plain_opt.state_dict()["regularizer"] == {"latent": None, "direct": None}


def test_generator_optimizer_direct_particles():
    from particlegan.k3p import DirectParticleResponse
    recipe = _recipe()
    results = []
    for use_factory in (True, False):
        torch.manual_seed(0)
        particles = nn.Parameter(torch.randn(16, 2))
        if use_factory:
            opt = recipe.make_generator_optimizer([particles], direct_particles=[particles],
                                                  lr=1e-2, betas=(0.0, 0.999))
        else:
            opt = torch.optim.Adam([particles], lr=1e-2, betas=(0.0, 0.999))
            resp = DirectParticleResponse([particles], torch.zeros(particles.numel()),
                                          betas=recipe.direct_particle_betas)
        for step in range(5):
            opt.zero_grad()
            (particles - torch.tensor([1.0, -1.0]) * step).square().sum().backward()
            if use_factory:
                opt.step()
            else:
                with resp.around(opt):
                    opt.step()
        results.append(particles.detach().clone())
    assert torch.equal(*results)
    assert opt_state_direct(recipe) == {"started": True}


def opt_state_direct(recipe):
    particles = nn.Parameter(torch.randn(4, 2))
    opt = recipe.make_generator_optimizer([particles], direct_particles=[particles])
    particles.sum().backward()
    opt.step()
    return opt.state_dict()["regularizer"]["direct"]["state"]


def test_api_doc_example_runs():
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[1] / "docs/api.md").read_text()
    block = next(b for b in re.findall(r"```python\n(.*?)```", text, re.S)
                 if "make_critic_penalty(opt_d)" in b and "for step in range" in b)
    code = block.replace("num_classes=2)", "num_classes=2, total_steps=3, batch_size=32)", 1)
    assert code != block
    exec(compile(code, "docs/api.md", "exec"), {"__name__": "__api_example__"})
