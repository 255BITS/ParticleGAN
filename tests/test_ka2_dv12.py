"""The shipped formulation's parts: the KA2 kernel and record, the DV12 controller, prior support jitter."""
import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from particlegan import ParticlePrior, get_recipe
from particlegan import grad_regularizers as gr
from particlegan.dv12 import DV12Controller
from particlegan.grad_regularizers import CriticStepRecord, GradientPenalty
from particlegan.k3p import CriticAnchor, CriticSpikeGuard

DT = torch.float64


def _critic(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)).to(DT)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, 2, generator=g, dtype=DT), torch.randn(32, 2, generator=g, dtype=DT) * 1.5 + .5


def _grad(D, x):
    x = x.detach().clone().requires_grad_(True)
    return torch.autograd.grad(D(x).sum(), x)[0]


def test_phase_a_is_r1_plus_a_one_sided_fake_cap():
    D = _critic()
    xr, xf = _batch()
    reg = GradientPenalty(coeff=3.0, kappa=0.5)
    pen, stats = reg.penalty(D, xr, xf, 1)
    gr_, gf = _grad(D, xr), _grad(D, xf)
    expected = 1.5 * ((gr_.square().sum(1) / 2).mean()
                      + ((gf.square().sum(1) + 1e-12).sqrt() / 2 ** .5 - .5).relu().square().mean())
    torch.testing.assert_close(pen, expected, rtol=1e-12, atol=0)
    assert stats["phase"] == "a" and stats["s"] == 1.0 and reg.record.calls == 1


def test_blend_is_half_a_plus_half_caps_and_gated_anchor(monkeypatch):
    monkeypatch.setattr(gr, "WARMUP_CALLS", 2)
    D = _critic()
    ema = _critic(seed=3)
    anchor = CriticAnchor(D, ema.requires_grad_(False))
    reg = GradientPenalty(coeff=2.0, kappa=0.5, anchor=anchor)
    xr, xf = _batch()
    assert reg.penalty(D, xr, xf, 1)[1]["phase"] == "a"
    reg.record.observed_steps = 1
    first, stats = reg.penalty(D, xr, xf, 2)  # starts the anchor: prox is 0 on this call
    assert stats["phase"] == "blend" and stats["prox"] == 0.0 and reg.record.anchor_started
    assert torch.equal(ema[0].weight, D[0].weight)  # start_ copied the live critic
    with torch.no_grad():
        ema[0].weight.add_(0.1)
    pen, stats = reg.penalty(D, xr, xf, 2)
    gr_, gf, gb = _grad(D, xr), _grad(D, xf), _grad(ema, xr)
    nr, nf = (gr_.square().sum(1) + 1e-12).sqrt(), (gf.square().sum(1) + 1e-12).sqrt()
    a = (gr_.square().sum(1) / 2).mean() + (nf / 2 ** .5 - .5).relu().square().mean()
    prox = (gr_ - gb).square().sum(1).mean() / 2
    b = F.relu(nr - .5).square().mean() + F.relu(nf - .5).square().mean() + 1.0 * prox
    torch.testing.assert_close(pen, 1.0 * (.5 * a + .5 * b), rtol=1e-12, atol=0)
    assert stats["w"] == 1.0 and stats["prox"] == pytest.approx(float(prox), rel=1e-12)


def test_record_gate_drops_and_restores_the_anchor_on_moment_surprise():
    record = CriticStepRecord()
    weights, alphas = [], []
    for value in [1.0] * 30 + [4.0] * 30 + [1.0] * 40:
        record.last_sur = value
        weights.append(record.advance_blend())
        alphas.append(record.alpha)
    assert record.sur_base == 1.0
    assert weights[:30] == [1.0] * 30
    drop = weights.index(0.0)
    assert 30 < drop < 60 and all(w == 0.0 for w in weights[drop:60])
    back = drop + weights[drop:].index(1.0)
    assert back > 60 and weights[-1] == 1.0
    assert alphas[29] == 0.0 and max(alphas) > 0 and alphas[-1] < max(alphas)  # slow attack, fast release


def test_record_step_tracks_the_ema_at_the_surprise_rate_and_reseeds():
    D = _critic()
    ema = copy.deepcopy(D).requires_grad_(False)
    anchor = CriticAnchor(D, ema, decay=.9)
    record = CriticStepRecord(anchor, anchor_min_decay=.9)
    opt = torch.optim.Adam(D.parameters(), lr=.01)
    D(torch.randn(4, 2, dtype=DT)).sum().backward()
    opt.step()
    record.anchor_started = True
    record.record_step(opt)  # alpha 0: decay 1, the EMA is skipped
    assert record.ema_skips == 1 and record.ema_updates == 0 and record.last_sur is not None
    record.alpha = .5
    with torch.no_grad():
        D[0].weight.add_(1.0)
    before = ema[0].weight.clone()
    record.record_step(opt)
    assert anchor.decay == pytest.approx(1 - .5 * .1)
    torch.testing.assert_close(ema[0].weight, before * anchor.decay + D[0].weight * (1 - anchor.decay))
    record.low_streak = gr.RESEED_STREAK
    record.record_step(opt)
    assert record.ema_reseeds == 1 and record.low_streak == 0
    state = record.state_dict()
    fresh = CriticStepRecord(anchor, anchor_min_decay=.9)
    fresh.load_state_dict(state)
    assert fresh.state_dict() == state
    with pytest.raises(ValueError, match="older formulation"):
        fresh.load_state_dict({"lr_max": 1.0, "lr_last": None, "anchor_started": False,
                               "calls": 0, "observed_steps": 0})
    with pytest.raises(ValueError, match="anchor_min_decay"):
        CriticStepRecord(anchor_min_decay=.8).load_state_dict(state)


def test_controller_rates_payoff_and_data_drive():
    c = DV12Controller()
    record = CriticStepRecord()
    g = torch.Generator().manual_seed(0)
    before = torch.get_rng_state()
    for _ in range(3):
        c.begin_critic_step(torch.randn(64, 3, generator=g), record)
        record.observed_steps += 1
    assert torch.equal(torch.get_rng_state(), before)  # private feature stream only
    assert c.updates == 3 and c.data_drive == 0.0 and c.mobility < 1.0
    assert c.network_scale == (.01 + .99 * c.mobility) * c.game_trust
    assert c.prior_scale == (.05 + .95 * c.mobility) * c.game_trust
    # Payoff: the critic's adversarial term as (total - penalty), in log-2 units.
    d, p, gl = torch.tensor(.5), torch.tensor(.25), torch.tensor(1.2)
    c.record_critic_loss(d); c.record_penalty(p); c.record_generator_loss(gl)
    c._generator_pending = True
    c.take_generator_step()
    error = max(0., float(gl - ((d + p) - p)) / math.log(2.))
    assert c.payoff_error == pytest.approx(.02 * error, rel=0, abs=0)
    assert c.critic_scale() == 1. / (1. + c.payoff_error ** 2)
    # A moving target drives the rates back up.
    for step in range(60):
        c.begin_critic_step(torch.randn(64, 3, generator=g) + .1 * step, record)
        record.observed_steps += 1
    assert c.data_drive > 0 and c.mobility > .5
    state = c.state_dict()
    other = DV12Controller()
    other.load_state_dict(state)
    assert other.mobility == c.mobility and torch.equal(other.projection, c.projection)
    with pytest.raises(ValueError):
        other.load_state_dict({**state, "mobility": float("nan")})


def test_game_trust_falls_with_unexplained_surprise():
    c = DV12Controller()
    record = CriticStepRecord()
    record.sur_base, record.last_sur = 1.0, 3.0
    c.begin_critic_step(torch.randn(16, 2), record)
    assert c.game_ratio == 3.0 and c.game_trust == 1. / (1. + 4.)


def test_prior_support_jitter_stays_inside_each_cell_and_keeps_gradients():
    prior = ParticlePrior(num_particles=50, z_dim=2, support_jitter=True)
    assert not bool(prior.support_ready)
    z, idx = prior.sample(400, generator=torch.Generator().manual_seed(0),
                          noise_generator=torch.Generator().manual_seed(1))
    assert bool(prior.support_ready)
    width = prior.z.detach().std(0, unbiased=False) * 50 ** (-1 / 2)
    assert torch.equal(prior.support_width, width)
    step = (z - prior.z[idx]).detach()
    others = torch.cdist(prior.z[idx].detach(), prior.z.detach(), compute_mode="donot_use_mm_for_euclid_dist")
    others[others == 0] = float("inf")
    assert (step.norm(dim=1) <= others.min(1).values * .5 * (1 + 1e-5)).all()
    assert step.abs().sum() > 0
    z.sum().backward()
    counts = torch.bincount(idx, minlength=50).to(prior.z.dtype)
    torch.testing.assert_close(prior.z.grad, counts[:, None].expand(-1, 2))
    # Same indices whether or not the prior jitters; the jitter uses its own stream.
    plain = ParticlePrior(num_particles=50, z_dim=2)
    _, plain_idx = plain.sample(400, generator=torch.Generator().manual_seed(0))
    assert torch.equal(idx, plain_idx)
    # The width follows the table at .01 per generator step.
    with torch.no_grad():
        prior.z.mul_(2)
    prior.track_support_()
    torch.testing.assert_close(prior.support_width, width.lerp(2 * width, .01))
    # Tables saved without jitter state load (the width restarts from the table).
    fresh = ParticlePrior(num_particles=50, z_dim=2, support_jitter=True)
    fresh.load_state_dict(plain.state_dict())
    assert not bool(fresh.support_ready)


def test_recipe_prior_jitters_and_generator_optimizer_tracks_it():
    recipe = get_recipe(num_particles=32, z_dim=2)
    prior = recipe.make_prior()
    assert prior.support_jitter and not get_recipe("mog", num_particles=8).make_prior().support_jitter
    g = nn.Linear(2, 2)
    opt_g, opt_d = recipe.make_optimizers(g, nn.Linear(2, 1), prior)
    assert opt_g.prior is prior and opt_g.controller is opt_d.controller
    g(prior.sample(8)[0]).square().sum().backward()
    before = prior.support_width.clone()
    opt_g.step()
    expected = before.lerp(prior.z.detach().std(0, unbiased=False) * 32 ** (-1 / 2), .01)
    torch.testing.assert_close(prior.support_width, expected, rtol=0, atol=0)


def test_guard_reads_the_amsgrad_running_max():
    p = nn.Parameter(torch.zeros(4, dtype=DT))
    opt = torch.optim.Adam([p], lr=0.1, betas=(0.0, 0.999), amsgrad=True)
    guard = CriticSpikeGuard(ratio=5.0, min_steps=1)
    for scale in (10.0, 0.1, 0.1):  # the second moment decays after the spike; its max does not
        p.grad = torch.full_like(p, scale)
        opt.step()
    state = opt.state[p]
    assert state["max_exp_avg_sq"].mean() > state["exp_avg_sq"].mean()
    vmax = state["max_exp_avg_sq"].mean() / (1 - .999 ** 3)
    p.grad = torch.full_like(p, 1000.0)
    assert int(guard.apply_(opt)) == 1
    # Clipped to 5x the RMS of the denominator AMSGrad divides by.
    torch.testing.assert_close(p.grad.square().mean().sqrt(), 5.0 * vmax.sqrt(), rtol=1e-5, atol=0)
