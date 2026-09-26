"""Symplectic gradient adjustment on the simultaneous D/G/particle game.

Balduzzi et al. 2018, Algorithm 1 (The Mechanics of n-Player Differentiable
Games). The game vector ξ stacks each player's gradient of its own loss.
A is the antisymmetric part of the game Jacobian. The adjusted vector is

    ξ + λ Aᵀ ξ

with |λ| = 1. The sign is the paper's alignment rule,

    λ = sign( (1/d) ⟨ξ, ∇H⟩ ⟨Aᵀ ξ, ∇H⟩ + ε ),  ε = 1e-10,

where ∇H = Jᵀ ξ = ∇(½ ‖ξ‖²) and d = dim(ξ). Aᵀ ξ = (Jᵀ ξ − J ξ) / 2, both
products from Hessian-vector products (appendix C). The result is written
onto ``.grad`` and the existing Adam step consumes it at the constant LRs.

Losses are evaluated at one parameter point, before either player steps, so
the Jacobian is the simultaneous game. Samples are still drawn in the host's
order. Flag unset does not call into this module's update.
"""
from __future__ import annotations

import atexit
import json
import os
from contextlib import nullcontext

import torch
from torch import nn

EPS = 1e-10
receipt = {"mechanism": "sga", "steps": 0, "lambda_pos": 0, "lambda_neg": 0, "lambda_zero": 0}
_LOGGED = False
_INSTALLED = False


def active() -> bool:
    return os.environ.get("K3P_DYNAMICS") == "sga"


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_emit)
    print(json.dumps({
        "event": "dynamics",
        "name": "sga",
        "setting": "Balduzzi 2018 Algorithm 1, |lambda|=1, sign((1/d)<xi,gradH><A^T xi,gradH>+1e-10), Adam on the adjusted D/G/particle vector",
    }), flush=True)


def _emit() -> None:
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def relax_inplace(*modules) -> None:
    """Out-of-place LeakyReLU so the Hessian-vector products can backward.

    Inplace and out-of-place LeakyReLU match on values. The ring critic and
    generator are inplace, which frees the activations a second backward needs.
    """
    for module in modules:
        if module is None:
            continue
        for sub in module.modules():
            if isinstance(sub, nn.LeakyReLU) and sub.inplace:
                sub.inplace = False


def assign_grads(params, grads) -> None:
    live = [p for p in params if p.requires_grad]
    if len(live) != len(grads):
        raise RuntimeError("sga gradient list does not match the player parameters")
    for parameter, grad in zip(live, grads):
        if grad is None:
            continue
        parameter.grad = grad.detach()


def _dot(grads, vecs):
    total = None
    for grad, vec in zip(grads, vecs):
        if grad is None or vec is None:
            continue
        term = (grad * vec).sum()
        total = term if total is None else total + term
    return total


def _value_dot(left, right):
    total = None
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        term = (a.detach() * b.detach()).sum()
        total = term if total is None else total + term
    return total


def _half_diff(left, right):
    """(left - right) / 2, treating a missing product as zero."""
    if left is None and right is None:
        return None
    if left is None:
        return right.mul(-0.5)
    if right is None:
        return left.mul(0.5)
    return (left - right).mul(0.5)


def adjust(players):
    """Return detached adjusted gradients, one list per ``(params, loss)`` player.

    ``players`` is D, then G, then particles. G and particles may share a loss.
    Empty players (no trainable parameters) get an empty list.
    """
    global _LOGGED
    groups = [[p for p in params if p.requires_grad] for params, _loss in players]
    flat = [p for group in groups for p in group]
    if not flat:
        return [[] for _ in players]

    spans = []
    offset = 0
    for group in groups:
        spans.append((offset, offset + len(group)))
        offset += len(group)

    cache = {}
    full = []
    for _params, loss in players:
        key = id(loss)
        if key not in cache:
            cache[key] = torch.autograd.grad(
                loss, flat, create_graph=True, retain_graph=True, allow_unused=True,
            )
        full.append(cache[key])

    xis = [grads[start:stop] for grads, (start, stop) in zip(full, spans)]
    flat_xi = [grad for grads in xis for grad in grads]
    velocity = [None if grad is None else grad.detach() for grad in flat_xi]

    dot_xi = _dot(flat_xi, velocity)
    if dot_xi is None:
        jtv = [None] * len(flat)
    else:
        jtv = list(torch.autograd.grad(dot_xi, flat, retain_graph=True, allow_unused=True))

    jv = [None] * len(flat)
    seen = set()
    for loss_index, (_params, loss) in enumerate(players):
        key = id(loss)
        if key in seen:
            continue
        seen.add(key)
        owned = []
        for other, (_other_params, other_loss) in enumerate(players):
            if other_loss is loss:
                start, stop = spans[other]
                owned.extend(range(start, stop))
        dot_i = _dot(full[loss_index], velocity)
        if dot_i is None or not owned:
            continue
        part = torch.autograd.grad(
            dot_i, [flat[i] for i in owned], retain_graph=True, allow_unused=True,
        )
        for index, grad in zip(owned, part):
            jv[index] = grad

    at = [_half_diff(left, right) for left, right in zip(jtv, jv)]
    xi_h = _value_dot(flat_xi, jtv)
    at_h = _value_dot(at, jtv)
    zero = flat[0].new_zeros(())
    xi_h = zero if xi_h is None else xi_h
    at_h = zero if at_h is None else at_h
    dimension = sum(parameter.numel() for parameter in flat)
    # Algorithm 1: sign( (1/d) <ξ, ∇H> <Aᵀξ, ∇H> + ε ), ε = 1e-10.
    lam = torch.sign(xi_h * at_h / dimension + EPS)
    sign = int(lam.detach().item())
    receipt["steps"] += 1
    if sign > 0:
        receipt["lambda_pos"] += 1
    elif sign < 0:
        receipt["lambda_neg"] += 1
    else:
        receipt["lambda_zero"] += 1
    if not _LOGGED or receipt["steps"] % 200 == 0:
        _LOGGED = True
        print(json.dumps({
            "event": "dynamics_step",
            "name": "sga",
            "steps": receipt["steps"],
            "lambda": sign,
            "align": float((xi_h * at_h).detach()),
        }), flush=True)

    adjusted = []
    cursor = 0
    scale = lam.detach()
    for grads in xis:
        row = []
        for grad in grads:
            delta = at[cursor]
            cursor += 1
            if grad is None and delta is None:
                row.append(None)
                continue
            base = torch.zeros_like(delta) if grad is None else grad.detach()
            extra = torch.zeros_like(base) if delta is None else delta.detach()
            row.append(base + scale * extra)
        adjusted.append(row)
    return adjusted


def mode_hold_players(generator, critic, prior, gan, regularizer, vicreg, recipe,
                      stream, noise_policy, step, batch, means, sigma):
    """One simultaneous ring update's players, samples in the host's order.

    The discriminator loss sees a detached fake, as in the alternating host.
    The generator loss is built at those same parameters, before the critic step.
    """
    from benchmarks.locked_shared.mode_hold import sample_ring
    from particlegan.dynamics.shared_batch import shared_batch_update

    relax_inplace(generator, critic, prior)
    real = sample_ring(means, batch, sigma, stream)
    latent, _ = prior.sample(batch, generator=stream)
    context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
    with context:
        fake = generator(latent).detach()
    d_loss = gan.d_loss(critic(real), critic(fake))
    d_loss = d_loss + regularizer(critic, real, fake, step=step + 1)
    share = shared_batch_update()
    if not share:
        latent, _ = prior.sample(batch, generator=stream)
    fake = generator(latent)
    if gan.mode in ("rp", "ra"):
        real_g = real if share else sample_ring(means, batch, sigma, stream)
        g_loss = gan.g_loss(critic(fake), critic(real_g))
    else:
        g_loss = gan.g_loss(critic(fake))
    if recipe.fm_weight > 0.0:
        real_mean = sample_ring(means, batch, sigma, stream).detach().mean(0)
        g_loss = g_loss + recipe.fm_weight * (fake.mean(0) - real_mean).pow(2).sum()
    g_loss = g_loss + recipe.particle_l2 * prior.z.pow(2).mean()
    g_loss = g_loss + vicreg(prior.z)
    prior_ids = {id(p) for p in prior.parameters()}
    d_params = [p for p in critic.parameters() if p.requires_grad]
    g_params = [p for p in generator.parameters() if p.requires_grad and id(p) not in prior_ids]
    p_params = [p for p in prior.parameters() if p.requires_grad]
    return [(d_params, d_loss), (g_params, g_loss), (p_params, g_loss)]
