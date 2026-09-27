"""Deterministic CPU traces shared by the frozen KA2 hook and package tests.

The real penalty and its parameter gradients are evaluated every step. After
backward, prescribed critic gradients provide repeatable quiet / sustained
surprise / quiet intervals, independently of whether a tiny GAN happens to
produce a useful transient. This is a controller contract, not an experiment.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


DT = torch.float64
CALLS = 1220
LR = 0.0003
FROZEN_SOURCE = "reports/ka2-default-candidate/source/mechanism.py"
FROZEN_SHA256 = "9f1d5eda4bb2f9e0e51832af77db27bd87fba214e29146b224d9963da390c478"
CONTROL_KEYS = ("last_sur", "sur_hist", "sur_base", "w", "low_streak", "alpha", "last_ratio")


def make_critic():
    critic = nn.Sequential(nn.Linear(2, 3, dtype=DT), nn.Tanh(), nn.Linear(3, 1, dtype=DT))
    with torch.no_grad():
        for i, p in enumerate(critic.parameters()):
            p.copy_(torch.sin(torch.arange(p.numel(), dtype=DT).reshape(p.shape) + i) * .4)
    return critic


def batch(t):
    real = torch.sin(torch.arange(8, dtype=DT).reshape(4, 2) + t * .017)
    fake = torch.cos(torch.arange(8, dtype=DT).reshape(4, 2) + t * .031) + .3
    return real, fake


def controlled_gradients(critic, t, lazy_k):
    call = t // lazy_k
    scale = 1.18 ** (call - 830) if 830 <= call < 960 else 1.0
    for i, p in enumerate(critic.parameters()):
        # Distinct tensor surprises exercise the frozen *upper* median.
        value = scale * (1 + i * .13) * (.8 + .2 * math.sin(t * .3 + i))
        p.grad.fill_(value)


def run(critic, optimizer, penalty, snapshot, steps, lazy_k):
    trace = []
    for t in steps:
        real, fake = batch(t)
        optimizer.zero_grad(set_to_none=True)
        pen, stats = penalty(critic, real, fake, t)
        (F.softplus(-critic(real)).mean() + F.softplus(critic(fake)).mean() + pen).backward()
        raw_grads = [p.grad.detach().clone() for p in critic.parameters()]
        controlled_gradients(critic, t, lazy_k)
        optimizer.step()
        trace.append(dict(t=t, penalty=pen.detach().clone(), applied=stats["applied"],
                          raw_grads=raw_grads, params=[p.detach().clone() for p in critic.parameters()],
                          **snapshot()))
    return trace


def frozen(mechanism, lazy_k):
    from benchmarks.legacy.grad_regularizers import GradRegularizer

    critic = make_critic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=LR, betas=(0., .999), foreach=False)
    regularizer = GradRegularizer(arm="a_r1r2", coeff=1., kappa=.1, lazy_k=lazy_k)

    def snapshot():
        state, receipt = mechanism._state, mechanism.receipt
        return dict(control={k: copy.deepcopy(state[k]) for k in CONTROL_KEYS},
                    calls=receipt["calls"],
                    ema=None if state["ema"] is None else [e.clone() for e in state["ema"]],
                    ema_updates=receipt["ema_updates"], ema_skips=receipt["ema_skips"],
                    ema_reseeds=receipt["ema_reseeds"])

    trace = run(critic, optimizer, regularizer.penalty, snapshot, range(1, CALLS * lazy_k + 1), lazy_k)
    return dict(trace=trace, clipped=int(torch.stack(mechanism._state["clips"]).sum()),
                first_blend=mechanism.receipt["first_blend_call"])


def build(lazy_k=1):
    from particlegan import get_recipe

    critic = make_critic()
    recipe = get_recipe(lr=LR, reg_kappa=.1, reg_every=lazy_k)
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=copy.deepcopy(critic), foreach=False)
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
    return critic, optimizer, penalty


def package_trace(objects, steps, lazy_k=1):
    critic, optimizer, penalty = objects

    def call(critic, real, fake, t):
        assert t == optimizer.record.observed_steps + 1
        value = penalty(critic, real, fake)
        return value, penalty.last_stats

    def snapshot():
        record = optimizer.record
        return dict(control={k: copy.deepcopy(getattr(record, k)) for k in CONTROL_KEYS}, calls=record.calls,
                    ema=None if not record.anchor_started else
                    [p.detach().clone() for p in optimizer.ema_critic.parameters()],
                    ema_updates=record.ema_updates, ema_skips=record.ema_skips, ema_reseeds=record.ema_reseeds)

    return run(critic, optimizer, call, snapshot, steps, lazy_k)
