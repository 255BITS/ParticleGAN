"""Shared K3P parity scenarios (CPU, float64).

The same scenario code drives the frozen research mechanism (global hooks,
run in a subprocess by tests/test_k3p.py) and the package API (in-process),
so any difference in the traces is a difference in the K3P math.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DT = torch.float64
STEPS = 16
LR0 = 0.05
SPIKE_STEP = 7
FROZEN_SOURCES = "reports/toy100/gap-fill-20260925/sources/k3p"


def lr_mult(t):
    """Flat (s == 1) for 1-4, cosine down to 0.1x for 5-12 (blend), then below the floor (s == 0)."""
    if t <= 4:
        return 1.0
    if t <= 12:
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * (t - 4) / 8))
    return 0.005


def make_critic(seed=0):
    g = torch.Generator().manual_seed(seed)
    D = nn.Sequential(nn.Linear(2, 16, dtype=DT), nn.Tanh(), nn.Linear(16, 16, dtype=DT), nn.Tanh(),
                      nn.Linear(16, 1, dtype=DT))
    with torch.no_grad():
        for p in D.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=DT) * 0.5)
    return D


def critic_optimizer(D):
    return torch.optim.Adam(D.parameters(), lr=LR0, betas=(0.0, 0.999), foreach=False)


def critic_batch(t, seed=1):
    g = torch.Generator().manual_seed(seed * 1000 + t)
    xr = torch.randn(32, 2, generator=g, dtype=DT)
    xf = torch.randn(32, 2, generator=g, dtype=DT) * 1.5 + 0.5
    return xr, xf


def run_critic(reg, D, opt, steps, *, guard=None, after=None, lr_fn=lr_mult, trace=None, seed=1):
    trace = [] if trace is None else trace
    for t in steps:
        for group in opt.param_groups:
            group["lr"] = LR0 * lr_fn(t)
        xr, xf = critic_batch(t, seed)
        pen, st = reg.penalty(D, xr, xf, t)
        loss = F.softplus(-D(xr)).mean() + F.softplus(D(xf)).mean()
        if t == SPIKE_STEP:
            loss = loss * 1000.0
        opt.zero_grad(set_to_none=True)
        (loss + pen).backward()
        if guard is not None:
            guard.apply_(opt)
        opt.step()
        if after is not None:
            after(opt)
        trace.append(dict(t=t, pen=pen.detach().clone(), s=st.get("s", 1.0), applied=st["applied"],
                          params=[p.detach().clone() for p in D.parameters()]))
    return trace


def make_latent(seed=2):
    from particlegan import ParticlePrior
    g = torch.Generator().manual_seed(seed)
    prior = ParticlePrior(64, 2, dtype=DT, generator=g)
    G = nn.Linear(2, 3, dtype=DT)
    with torch.no_grad():
        for p in G.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=DT))
    opt = torch.optim.Adam([{"params": list(G.parameters())}, {"params": list(prior.parameters())}],
                           lr=0.02, betas=(0.0, 0.999), foreach=False)
    return prior, G, opt


def latent_indices(t):
    if 5 <= t <= 9:  # full coverage: not sparse, and pushes the observed rate past 1/2
        return torch.arange(64)
    return torch.randint(0, 64, (8,), generator=torch.Generator().manual_seed(5000 + t))


def run_latent(prior, G, opt, steps, step_fn, trace=None):
    trace = [] if trace is None else trace
    for t in steps:
        idx = latent_indices(t)
        target = torch.randn(idx.numel(), 3, generator=torch.Generator().manual_seed(7000 + t), dtype=DT)
        loss = (G(prior(idx)) - target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        step_fn(opt)
        trace.append(dict(t=t, params=[prior.z.detach().clone()] + [p.detach().clone() for p in G.parameters()],
                          exp_avg=opt.state[prior.z]["exp_avg"].clone()))
    return trace


def make_direct(seed=3):
    g = torch.Generator().manual_seed(seed)
    particles = nn.Parameter(torch.randn(32, 2, generator=g, dtype=DT))
    opt = torch.optim.Adam([{"params": [particles], "_comparison_prior": True}], lr=0.03,
                           betas=(0.0, 0.999), foreach=False)
    return particles, opt


def run_direct(particles, opt, steps, step_fn, trace=None):
    trace = [] if trace is None else trace
    for t in steps:
        center = torch.tensor([math.cos(0.3 * t), math.sin(0.3 * t)], dtype=DT)
        loss = (particles - center).pow(2).sum(-1).mean() + 0.1 * particles.pow(3).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gain = step_fn(opt)
        trace.append(dict(t=t, params=particles.detach().clone(), lr=opt.param_groups[0]["lr"],
                          betas=opt.param_groups[0]["betas"], gain=gain))
    return trace


# ---------------------------------------------------------------- frozen side

def frozen_all(mechanism, latent, response):
    """Run every scenario through the frozen hook-based mechanism (subprocess only)."""
    from benchmarks.legacy.grad_regularizers import GradRegularizer
    keep = []
    prox_log = []
    original_gap = mechanism.anchored_gradient_gap

    def gap(*args):
        value = original_gap(*args)
        prox_log.append(float(value))
        return value

    mechanism.anchored_gradient_gap = gap
    out = {}
    for name, lazy_k in (("critic", 1), ("lazy", 2)):
        mechanism._state.update(ema=None, critic=None, pending=False, lr_max=0.0, lr_last=None, depth=0,
                                clips=[], critic_ref=None)
        mechanism.receipt.update(calls=0, anchor_started_call=None)
        prox_log.clear()
        D = make_critic()
        opt = critic_optimizer(D)
        keep.append(opt)
        reg = GradRegularizer(arm="a_r1r2", coeff=1.0, kappa=0.5, lazy_k=lazy_k)
        emas = []
        trace = run_critic(reg, D, opt, range(1, STEPS + 1),
                           after=lambda o: emas.append(None if mechanism._state["ema"] is None
                                                       else [e.clone() for e in mechanism._state["ema"]]))
        for row, ema in zip(trace, emas):
            row["ema"] = ema
        clips = mechanism._state["clips"]
        out[name] = dict(trace=trace, prox=list(prox_log), anchor_started_call=mechanism.receipt["anchor_started_call"],
                         clipped=int(torch.stack(clips).sum()) if clips else 0, calls=mechanism.receipt["calls"])
    mechanism._state.update(critic=-1, pending=False)  # later optimizers are not critics
    mechanism.anchored_gradient_gap = original_gap

    prior, G, opt = make_latent()

    def latent_step(o):
        for group in o.param_groups:
            for v in group["params"]:
                if v.grad is not None and not o.state[v]:
                    o.state[v]["step"] = torch.zeros((), dtype=torch.float32)
                    o.state[v]["exp_avg"] = torch.zeros_like(v, memory_format=torch.preserve_format)
                    o.state[v]["exp_avg_sq"] = torch.zeros_like(v, memory_format=torch.preserve_format)
        saved = latent.begin(o)
        o.step()
        latent.end(saved)

    out["latent"] = dict(trace=run_latent(prior, G, opt, range(1, 15), latent_step),
                         scoped_calls=latent.receipt["scoped_calls"])

    particles, opt = make_direct()

    def direct_step(o):
        before = response.receipt["calls"]
        saved = response.begin(o)
        o.step()
        response.end(saved)
        return response.receipt["rows"][-1]["gain"] if response.receipt["calls"] > before else 1.0

    out["direct"] = dict(trace=run_direct(particles, opt, range(1, 11), direct_step))
    return out


# ---------------------------------------------------------------- package side

def package_critic(lazy_k=1, steps=range(1, STEPS + 1), setup=None):
    """Build the package K3P critic stack; returns (objects, trace)."""
    from particlegan.grad_regularizers import GradientPenalty
    from particlegan.k3p import CriticAnchor, CriticSpikeGuard
    D = make_critic()
    ema = copy.deepcopy(D).requires_grad_(False)
    opt = critic_optimizer(D)
    anchor = CriticAnchor(D, ema, decay=0.999)
    reg = GradientPenalty(coeff=1.0, kappa=0.5, lazy_k=lazy_k, lr_floor=0.01, anchor=anchor)
    guard = CriticSpikeGuard(ratio=5.0, min_steps=3)
    objs = dict(D=D, ema=ema, opt=opt, anchor=anchor, reg=reg, guard=guard)
    if setup is not None:
        setup(objs)
    trace = run_package_critic(objs, steps)
    return objs, trace


def run_package_critic(objs, steps):
    reg, anchor = objs["reg"], objs["anchor"]
    started = []

    def after(o):
        reg.after_critic_step(o)
        started.append([e.clone() for e in objs["ema"].parameters()] if reg.state_dict()["anchor_started"] else None)

    prox = []
    original = reg.penalty

    def penalty(*args, **kwargs):
        pen, st = original(*args, **kwargs)
        if st.get("phase") in ("blend", "b"):
            prox.append(st["prox"])
        return pen, st

    reg.penalty = penalty
    try:
        trace = run_critic(reg, objs["D"], objs["opt"], steps, guard=objs["guard"], after=after)
    finally:
        del reg.penalty
    for row, ema in zip(trace, started):
        row["ema"] = ema
    return dict(trace=trace, prox=prox)
