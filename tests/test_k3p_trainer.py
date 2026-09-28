"""GANTrainer / the recipe's optimizers, loss and critic penalty: DV12 + KA2 as the default, EMA critic and checkpoints."""
import copy

import pytest
import torch
from particlegan import (
    GANTrainer,
    InputNoise,
    NetworkLRTransition,
    Recipe,
    get_recipe,
    learning_rate_scales,
    scale_learning_rates,
)
from particlegan import grad_regularizers
from particlegan.dv12 import DV12Controller
from particlegan.grad_regularizers import GradientPenalty
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


@pytest.fixture
def short_warmup(monkeypatch):
    """Reach KA2's blended phase after 5 applied calls instead of 799."""
    monkeypatch.setattr(grad_regularizers, "WARMUP_CALLS", 6)


def _recipe(**overrides):
    # Small sparse table (64 rows, batch 8) so A2 damping is active.
    options = dict(num_particles=64, z_dim=2, batch_size=8, total_steps=40, d_guard_min_steps=2)
    return get_recipe(**{**options, **overrides})


def _trainer(recipe=None, *, critic=None, seed=0):
    torch.manual_seed(seed)
    recipe = recipe or _recipe()
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    D = critic or nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    return GANTrainer(recipe, G, D, seed=seed)


def _reals(n, seed=1, shift=0.0):
    rng = torch.Generator().manual_seed(seed)
    return [torch.randn(8, 2, generator=rng) * 2 + shift for _ in range(n)]


def test_trainer_default_is_dv12_ka2(short_warmup):
    recipe = get_recipe()
    assert recipe == Recipe() and recipe.name == "dv12" and recipe.amsgrad
    trainer = _trainer()
    assert isinstance(trainer.penalty.regularizer, GradientPenalty)
    controller = trainer.opt_d.controller
    assert isinstance(controller, DV12Controller)
    assert trainer.opt_g.controller is controller and trainer.loss.controller is controller
    assert trainer.penalty.regularizer.controller is controller
    assert trainer.ema_D is not None and trainer.opt_d.guard is not None
    assert trainer.latent_damping is not None and trainer.prior.support_jitter
    phases = [trainer.step(real, collect_stats=True)["penalty_stats"]["phase"] for real in _reals(20)]
    assert phases[:5] == ["a"] * 5 and phases[5:] == ["blend"] * 15
    assert trainer.opt_d.state_dict()["regularizer"]["record"]["anchor_started"]
    assert trainer.latent_damping.started
    assert all(group["amsgrad"] for opt in (trainer.opt_g, trainer.opt_d) for group in opt.param_groups)


def test_trainer_rates_are_controller_fractions_of_the_peaks():
    trainer = _trainer()
    base = trainer.initial_lrs
    for real in _reals(12):
        out = trainer.step(real)
        c = trainer.opt_g.controller
        # The groups keep their peaks; each step ran at the controller's fractions.
        assert [[g["lr"] for g in o.param_groups] for o in (trainer.opt_g, trainer.opt_d)] == base
        assert trainer.opt_g.applied_lrs == [base[0][0] * c.network_scale, base[0][1] * c.prior_scale]
        assert trainer.opt_d.applied_lrs[0] <= base[1][0] * c.network_scale
        assert torch.isfinite(out["loss_g"])
    diag = trainer.penalty.diagnostics()
    assert 0 < diag["network_lr_scale"] <= 1 and 0 < diag["prior_lr_scale"] <= 1
    assert diag["network_lr_scale"] == (.01 + .99 * diag["mobility"]) * diag["game_trust"]
    # The optional caller schedule is off: every multiplier is 1.
    assert all(learning_rate_scales(s, trainer.recipe) == (1.0, 1.0) for s in range(40))


def test_mobility_relaxes_on_stationary_data_and_reopens_when_the_data_moves():
    trainer = _trainer(_recipe(total_steps=400))
    for real in _reals(250):
        trainer.step(real)
    settled = trainer.opt_d.controller.mobility
    assert settled < .5 and trainer.opt_d.controller.data_drive == 0.0
    for real in _reals(40, seed=2, shift=6.0):
        trainer.step(real)
    moved = trainer.opt_d.controller
    assert moved.data_drive > 0 and moved.mobility > settled


def _run(trainer, reals):
    return [trainer.step(real, generator_real=lambda r=real: r.flip(0), collect_stats=True) for real in reals]


def test_trainer_resume_bit_exact(short_warmup):
    reals = _reals(20)
    full = _trainer()
    full_out = _run(full, reals)
    split = 10
    assert full_out[split]["penalty_stats"]["phase"] == "blend"  # resume inside the blend

    first = _trainer()
    _run(first, reals[:split])
    checkpoint = first.state_dict()
    opt_g_state, opt_d_state = (state["regularizer"] for state in checkpoint["optimizers"])
    assert checkpoint["schema"] == 3 and opt_d_state["record"]["anchor_started"]
    assert opt_d_state["controller"]["updates"] == split
    assert opt_g_state["latent"]["state"]["started"] and "noise_generator" in checkpoint["streams"]
    assert torch.isfinite(checkpoint["models"]["prior"]["support_width"]).all()
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
    (left_g, left_d), (right_g, right_d) = ([o["regularizer"] for o in x["optimizers"]] for x in (left, right))
    for key, value in left_d["ema"].items():
        assert torch.equal(value, right_d["ema"][key]), key
    assert left_d["record"] == right_d["record"] and left_d["guard"] == right_d["guard"]
    assert torch.equal(left_g["latent"]["history"], right_g["latent"]["history"])
    for name, value in left["streams"].items():
        assert torch.equal(value, right["streams"][name]), name


def _buffers(module):
    return {k: v.detach().clone() for k, v in module.named_buffers()}


def test_trainer_ema_critic_buffers_no_bn_or_sn_mutation(short_warmup):
    critic = nn.Sequential(spectral_norm(nn.Linear(2, 16)), nn.BatchNorm1d(16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    trainer = _trainer(critic=critic)
    for real in _reals(20):
        trainer.step(real)
    assert trainer.opt_d.record.anchor_started
    ema, live = trainer.ema_D, trainer.D
    # KA2 updates the EMA only while the surprise gain is nonzero (then copies integer buffers).
    assert trainer.opt_d.record.ema_updates + trainer.opt_d.record.ema_skips > 0
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


def test_old_checkpoints_rejected():
    trainer = _trainer()
    state = trainer.state_dict()
    state["schema"] = 1
    with pytest.raises(ValueError, match="schema-1"):
        trainer.load_state_dict(state)
    bad = trainer.state_dict()
    bad["optimizers"][1]["regularizer"]["record"] = {}
    with pytest.raises(ValueError, match="optimizer state"):
        trainer.load_state_dict(bad)
    k3p = trainer.state_dict()
    k3p["recipe"] = {**{k: v for k, v in k3p["recipe"].items() if k != "reg_anchor_min_decay"},
                     "reg_anchor_decay": 0.999}
    with pytest.raises(ValueError, match="K3P"):
        trainer.load_state_dict(k3p)


def _critic_step(critic, opt, penalty, real, fake):
    loss = critic(real).mean() - critic(fake).mean() + penalty(critic, real, fake)
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_multiple_critics_each_own_controller_no_global_hooks(short_warmup):
    from torch.optim import optimizer as optim_module
    hooks = (len(optim_module._global_optimizer_pre_hooks), len(optim_module._global_optimizer_post_hooks))
    recipe = _recipe()
    torch.manual_seed(0)
    critics = [nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)) for _ in range(2)]
    optimizers = [recipe.make_critic_optimizer(c, ema_critic=copy.deepcopy(c), lr=1e-2) for c in critics]
    penalties = [recipe.make_critic_penalty(o) for o in optimizers]
    for real in _reals(12):
        for k, (critic, opt, penalty) in enumerate(zip(critics, optimizers, penalties)):
            _critic_step(critic, opt, penalty, real, real + 1.0 + k)
    assert optimizers[0].controller is not optimizers[1].controller
    assert all(o.record.anchor_started and o.controller.updates == 12 for o in optimizers)
    assert optimizers[0].record.last_sur != optimizers[1].record.last_sur
    assert (len(optim_module._global_optimizer_pre_hooks), len(optim_module._global_optimizer_post_hooks)) == hooks
    with pytest.raises(ValueError, match="EMA"):
        recipe.make_critic_penalty(recipe.make_critic_optimizer(critics[0]))
    with pytest.raises(TypeError, match="make_critic_optimizer"):
        recipe.make_critic_penalty(torch.optim.Adam(critics[0].parameters()))
    with pytest.raises(TypeError, match="make_critic_optimizer"):
        recipe.make_loss(torch.optim.Adam(critics[0].parameters()))
    with pytest.raises(TypeError, match="paired"):
        penalties[0](critics[1], real, real + 1)


def test_optimizer_steps_require_the_signals_they_read():
    recipe = _recipe()
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    real = torch.randn(8, 2)
    d(real).mean().backward()
    before = [p.detach().clone() for p in d.parameters()]
    with pytest.warns(RuntimeWarning, match="critic penalty"):
        opt_d.step()  # unobserved: the controller keeps its (initial) rates
    assert opt_d.applied_lrs == [recipe.lr * recipe.d_lr_mult]
    assert not all(torch.equal(a, b) for a, b in zip(before, d.parameters()))
    # A generator-only step (no critic step since the last one) is allowed.
    z, _ = prior.sample(8)
    g(z).square().mean().backward()
    opt_g.step()
    # After a critic step, the generator step needs the bound loss's values.
    penalty = recipe.make_critic_penalty(opt_d)
    _critic_step(d, opt_d, penalty, real, real + 1)
    opt_g.zero_grad()
    g(prior.sample(8)[0]).square().mean().backward()
    with pytest.raises(RuntimeError, match="make_loss"):
        opt_g.step()


class _TwoRoles(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({r: nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
                                    for r in ("joint", "marginal")})

    def critic_for(self, role):
        return self.heads[role]


def test_shared_module_multi_role_critic(short_warmup):
    recipe = _recipe()
    torch.manual_seed(0)
    d = _TwoRoles()
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d), lr=1e-2)
    penalty = recipe.make_critic_penalty(opt, collect_stats=True)
    for real in _reals(12):
        fake, loss, phases = real + 1.0, 0.0, []
        for role in ("joint", "marginal"):
            critic = d.critic_for(role)
            pen = penalty(critic, real, fake)  # the EMA uses the same-named role submodule
            phases.append(penalty.last_stats["phase"])
            loss = loss + critic(real).mean() - critic(fake).mean() + pen
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Two applied calls per step advance KA2's call clock twice; the controller observes once.
    assert phases == ["blend", "blend"] and opt.record.observed_steps == 12 and opt.controller.updates == 12
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


def test_conditional_penalty_forwards_conditioning_to_critic_and_ema(short_warmup):
    recipe = _recipe()
    torch.manual_seed(0)
    d = _Conditional()
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d), lr=1e-2)
    penalty = recipe.make_critic_penalty(opt, collect_stats=True)
    labels, t = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), torch.tensor([[0.3]])
    for real in _reals(12):
        fake = real + 1.0
        # The kernel on an identical copy of the state, with the conditioning bound by hand.
        record = copy.deepcopy(opt.record)
        controller = copy.deepcopy(opt.controller)
        controller.begin_critic_step(real, record)
        ref = GradientPenalty(record=record, controller=controller, **recipe._penalty_options())
        live = copy.deepcopy(d)
        ref_pen, _ = ref.penalty(lambda x: live(x, labels, t=t)[0], real, fake, record.observed_steps + 1,
                                 ema_critic=lambda x: record.anchor.forward(lambda m, y: m(y, labels, t=t)[0], x))
        pen = penalty(d, real, fake, labels, t=t)
        assert torch.equal(pen, ref_pen)
        loss = d(real, labels, t=t)[0].mean() - d(fake, labels, t=t)[0].mean() + pen
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert penalty.last_stats["phase"] == "blend" and opt.record.anchor_started
    # output= selects the logits from other layouts, at construction.
    swapped = recipe.make_critic_penalty(opt, output=lambda out: out[0])
    state = copy.deepcopy(opt.record.state_dict())
    first = penalty(d, real, fake, labels, t=t)
    opt.record.load_state_dict(state)
    assert torch.equal(swapped(d, real, fake, labels, t=t), first)


def test_input_noise_wrapper_penalty_uses_same_noise_on_ema():
    recipe = _recipe()
    torch.manual_seed(0)
    d = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d))
    penalty = recipe.make_critic_penalty(opt)
    stream = torch.Generator().manual_seed(3)
    noisy = InputNoise(d, 0.1, stream)
    opt.record.calls = grad_regularizers.WARMUP_CALLS  # force the blended phase (anchor in use)
    x = torch.randn(4, 2)
    before = stream.get_state()
    penalty(noisy, x, x + 1)
    # The anchor starts on its first blended call (prox == 0); the second draws EMA noise too.
    penalty(noisy, x, x + 1)
    draws = 0
    probe = torch.Generator().manual_seed(0)
    probe.set_state(before)
    while not torch.equal(probe.get_state(), stream.get_state()):
        torch.randn(4, 2, generator=probe)
        draws += 1
    assert draws == 2 + 3  # each call: live real + fake; the 2nd adds the EMA real draw


def _plain_parts(recipe):
    torch.manual_seed(0)
    g = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    d = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    return g, d, prior, opt_g, opt_d, recipe.make_loss(opt_d), recipe.make_critic_penalty(opt_d)


def _plain_run(parts, reals, steps):
    g, d, prior, opt_g, opt_d, gan, penalty = parts
    out = []
    for step in steps:
        real = reals[step]
        z, _ = prior.sample(8, generator=torch.Generator().manual_seed(100 + step))
        fake = g(z)
        d_loss = gan.d_loss(d(real), d(fake.detach())) + penalty(d, real, fake.detach())
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        g_loss = gan.g_loss(d(fake), d(real))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        out.append((d_loss.detach(), g_loss.detach()))
    return out


def test_plain_torch_checkpoint_resumes_exactly(short_warmup):
    """The usual ``torch.save({'G', 'D', 'prior', 'opt_g', 'opt_d'})`` pattern restores all state."""
    recipe = _recipe()
    reals = _reals(20)
    full = _plain_parts(recipe)
    full_out = _plain_run(full, reals, range(20))
    first = _plain_parts(recipe)
    _plain_run(first, reals, range(15))
    g, d, prior, opt_g, opt_d, _, _ = first
    checkpoint = copy.deepcopy({"G": g.state_dict(), "D": d.state_dict(), "prior": prior.state_dict(),
                                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()})
    resumed = _plain_parts(recipe)
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
    assert opt_d.record.anchor_started and opt_g.latent_damping.started and opt_d.controller.updates == 15
    for a, b in zip(_plain_run(resumed, reals, range(15, 20)), full_out[15:]):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for x, y in zip(resumed[4].ema_critic.parameters(), full[4].ema_critic.parameters()):
        assert torch.equal(x, y)
    assert torch.equal(resumed[3].latent_history, full[3].latent_history)
    assert torch.equal(resumed[2].support_width, full[2].support_width)


def test_plain_pytorch_loop_reproduces_gan_trainer_bit_for_bit(short_warmup):
    """A caller-owned loop over the recipe's objects is GANTrainer's update, exactly."""
    recipe = _recipe()
    reals = _reals(16)
    trainer = _trainer(recipe)
    torch.manual_seed(0)
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
    gan, penalty = recipe.make_loss(opt_d), recipe.make_critic_penalty(opt_d)
    spread = recipe.make_prior_regularizer(weight=1.0)
    latent = torch.Generator().manual_seed(2)   # GANTrainer's streams for seed 0
    noise = torch.Generator().manual_seed(5)

    def generate(z):
        x = G(z)
        return x + recipe.output_noise_std * torch.randn(x.shape, generator=noise)
    for real in reals:
        trainer.step(real)
        with torch.no_grad():
            fake = generate(prior.sample(8, generator=latent, noise_generator=noise)[0])
        d_loss = gan.d_loss(D(real), D(fake)) + penalty(D, real, fake)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        D.requires_grad_(False)
        z, ids = prior.sample(8, generator=latent, noise_generator=noise)
        g_loss = gan.g_loss(D(generate(z)), D(real)) + recipe.prior_reg * spread(prior.z)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        D.requires_grad_(True)
    for mine, theirs in ((G, trainer.G), (D, trainer.D), (prior, trainer.prior), (opt_d.ema_critic, trainer.ema_D)):
        for a, b in zip(mine.state_dict().values(), theirs.state_dict().values()):
            assert torch.equal(a, b)
    assert opt_d.controller.state_dict().keys() == trainer.opt_d.controller.state_dict().keys()
    assert opt_d.controller.mobility == trainer.opt_d.controller.mobility


def test_recipe_optimizers_are_ordinary_adam():
    from torch.optim.lr_scheduler import LambdaLR
    recipe = _recipe()
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), nn.Linear(2, 1)
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    penalty = recipe.make_critic_penalty(opt_d)
    assert isinstance(opt_g, torch.optim.Adam) and isinstance(opt_d, torch.optim.Adam)
    scheduler = LambdaLR(opt_d, lambda epoch: 0.5 ** epoch)
    calls = []
    opt_d.register_step_post_hook(lambda *args: calls.append(1))
    x = torch.randn(8, 2)

    def closure():
        opt_d.zero_grad()
        loss = d(x).square().mean() + penalty(d, x, x + 1)
        loss.backward()
        return loss
    assert opt_d.step(closure) is not None and calls == [1] and opt_d.record.observed_steps == 1
    scheduler.step()
    # A scheduler scales the peak; the controller's fraction applies on top of it.
    assert opt_d.param_groups[0]["lr"] == recipe.lr * recipe.d_lr_mult * 0.5
    opt_d.step(closure)
    c = opt_d.controller
    assert opt_d.applied_lrs == [recipe.lr * recipe.d_lr_mult * 0.5 * c.network_scale * c.critic_scale()]
    with pytest.raises(ValueError, match="regularizer"):
        opt_d.load_state_dict(torch.optim.Adam(d.parameters()).state_dict())


def test_scale_learning_rates_sets_the_peaks_the_controller_scales():
    recipe = _recipe(lr_floor=0.05, network_lr_floor=0.01, network_lr_horizon_cap=8)
    torch.manual_seed(0)
    g, d = nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, ema_critic=copy.deepcopy(d))
    base = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    penalty = recipe.make_critic_penalty(opt_d)
    for step, real in enumerate(_reals(12), start=1):
        network, prior_scale = scale_learning_rates(step - 1, recipe, (opt_g, opt_d), base, prior)
        assert (network, prior_scale) == learning_rate_scales(step - 1, recipe)
        _critic_step(d, opt_d, penalty, real, real + 1)
        c = opt_d.controller
        assert opt_d.param_groups[0]["lr"] == base[1][0] * network
        assert opt_d.applied_lrs == [base[1][0] * network * c.network_scale * c.critic_scale()]
    assert network == recipe.network_lr_floor


def test_caller_marked_network_transition_keeps_prior_schedule_and_resumes():
    recipe = _recipe(total_steps=20, network_lr_horizon_cap=4, lr_floor=0.05, network_lr_floor=0.01)
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
    # Generator-only steps: the controller keeps its initial rates (fraction 1).
    recipe = _recipe()
    opt_g_z, opt_g = _sparse_prior_run(recipe, True)
    direct, damping = _sparse_prior_run(recipe, False)
    assert torch.equal(opt_g_z, direct)
    extra = opt_g.state_dict()["regularizer"]
    assert damping.started and extra["latent"]["state"] == damping.state_dict()
    state = copy.deepcopy(opt_g.state_dict())
    opt_g.load_state_dict(state)
    assert torch.equal(opt_g.latent_history, state["regularizer"]["latent"]["history"])
    with pytest.raises(ValueError):
        opt_g.load_state_dict({**state, "regularizer": {"latent": None}})
    # Saved before the direct-particle response was removed: still loads.
    opt_g.load_state_dict({**state, "regularizer": {**state["regularizer"], "direct": None}})
    # Disabled damping: step() is exactly Adam.step().
    off = recipe.replace(latent_damping_max_rate=0.0)
    plain, plain_opt = _sparse_prior_run(off, True)
    assert torch.equal(plain, _sparse_prior_run(off, False)[0])
    assert plain_opt.state_dict()["regularizer"] == {"latent": None}


def test_direct_particle_response_is_gone():
    recipe = _recipe()
    particles = nn.Parameter(torch.randn(4, 2))
    with pytest.raises(TypeError):
        recipe.make_generator_optimizer([particles], direct_particles=[particles])
    with pytest.raises(TypeError):
        get_recipe(direct_particle_gain=False)


def test_api_doc_example_runs():
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[1] / "docs/api.md").read_text()
    block = next(b for b in re.findall(r"```python\n(.*?)```", text, re.S)
                 if "make_critic_penalty(opt_d)" in b and "for step in range" in b)
    code = block.replace("num_classes=2)", "num_classes=2, batch_size=32)", 1)
    assert code != block and "steps = 2000" in code
    code = code.replace("steps = 2000", "steps = 3", 1)
    exec(compile(code, "docs/api.md", "exec"), {"__name__": "__api_example__"})
