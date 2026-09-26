"""Generator step against a discriminator unrolled k Adam steps (Metz et al. 2017).

The real critic update is the one the host already takes. The generator loss,
and the particle step that flows through it, is scored on a functional copy of
D after ``UNROLL_K`` more steps of D's own Adam update. Those steps use the
generator batch (real paired with the live fake), so a hop onto a neighbor
mode is differentiated through the critic's reaction to that hop. The copy
carries D's moments, learning rate, betas, and epsilon. It does not write
D's parameters or Adam state.

``UNROLL_K`` is 5, the toy-mixture depth in Metz et al., Unrolled Generative
Adversarial Networks (2017). The inner steps reuse this generator batch rather
than drawing new minibatches, so the reaction is to the samples G is stepping
on and the training RNG is unchanged.

With ``K3P_DYNAMICS`` unset, ``critic_for_generator`` returns the critic and
does not call its context.
"""
from __future__ import annotations

import atexit
import copy
import json
import os
import sys
from contextlib import contextmanager

import torch
from torch import nn
from torch.func import functional_call

# Toy-mixture unrolling depth from Metz et al. 2017. Not a searched coefficient.
UNROLL_K = 5

receipt = {"mechanism": "unrolled", "k": UNROLL_K, "generator_steps": 0}
_LOGGED = False
_HOOKED = False


def install() -> None:
    """Log the setting. The training loops read ``K3P_DYNAMICS`` themselves."""
    global _HOOKED
    if _HOOKED:
        return
    _HOOKED = True
    atexit.register(_emit)
    print(json.dumps({
        "event": "dynamics",
        "name": "unrolled",
        "k": UNROLL_K,
        "setting": "G and particles differentiated through k functional Adam steps of D; real D step unchanged",
    }), flush=True)


def _emit() -> None:
    print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def critic_for_generator(critic, fake, context):
    """Critic module the generator step should call.

    ``context`` is a zero-argument callable used only when this mechanism is
    on. It returns ``(optimizer, real, d_loss_fn, penalty)`` where
    ``d_loss_fn(module, real, fake)`` is the scalar the real critic step
    minimizes, and ``penalty`` is that step's penalty object (bookkeeping
    restored after the copy).
    """
    if os.environ.get("K3P_DYNAMICS") != "unrolled":
        return critic
    optimizer, real, d_loss_fn, penalty = context()
    return _unroll(critic, fake, optimizer, real, d_loss_fn, penalty)


class _FunctionalCritic:
    """Forward ``critic`` under a replaced parameter dict. Not registered."""

    def __init__(self, critic, params):
        self.critic = critic
        self.params = params

    def __call__(self, *args, **kwargs):
        changed = _disable_inplace(self.critic)
        try:
            return functional_call(self.critic, self.params, args, kwargs)
        finally:
            _restore_inplace(changed)


def _unroll(critic, fake, optimizer, real, d_loss_fn, penalty):
    global _LOGGED
    if not _HOOKED:
        install()
    names, params, moments = _snapshot(critic, optimizer)
    record = _record_of(penalty)
    saved_record = None if record is None else record.state_dict()
    saved_mechanism = _snap_mechanism()
    try:
        with _preserve_rng(critic):
            for _ in range(UNROLL_K):
                view = _FunctionalCritic(critic, params)
                loss = d_loss_fn(view, real.detach(), fake)
                grads = torch.autograd.grad(
                    loss, [params[name] for name in names], create_graph=True, allow_unused=True,
                )
                params, moments = _adam_update(names, params, grads, moments)
        if not _LOGGED:
            _LOGGED = True
            print(json.dumps({
                "event": "dynamics_step", "name": "unrolled", "k": UNROLL_K, "generator_steps": 1,
            }), flush=True)
        receipt["generator_steps"] += 1
        return _FunctionalCritic(critic, params)
    finally:
        if record is not None:
            record.load_state_dict(saved_record)
        _restore_mechanism(saved_mechanism)


def _snapshot(critic, optimizer):
    groups = {}
    states = {}
    for group in optimizer.param_groups:
        if group.get("amsgrad", False):
            raise RuntimeError("unrolled D expects Adam without amsgrad")
        if group.get("decoupled_weight_decay", False) and float(group.get("weight_decay", 0.0) or 0.0) != 0.0:
            raise RuntimeError("unrolled D expects coupled Adam weight decay")
        for parameter in group["params"]:
            groups[id(parameter)] = group
            states[id(parameter)] = optimizer.state.get(parameter, {})
    names = []
    params = {}
    moments = {}
    guard = getattr(optimizer, "guard", None)
    for name, parameter in critic.named_parameters():
        if id(parameter) not in groups:
            continue
        names.append(name)
        params[name] = parameter.detach().clone().requires_grad_(True)
        state = states[id(parameter)]
        step = state.get("step", None)
        if step is None:
            completed = 0
            exp_avg = torch.zeros_like(parameter)
            exp_avg_sq = torch.zeros_like(parameter)
        else:
            completed = int(step.item() if isinstance(step, torch.Tensor) else step)
            exp_avg = state["exp_avg"].detach().clone()
            exp_avg_sq = state["exp_avg_sq"].detach().clone()
        group = groups[id(parameter)]
        moments[name] = {
            "param_id": id(parameter),
            "group": group,
            "step": completed,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "guard": guard,
        }
    if not names:
        raise RuntimeError("unrolled D found no critic parameters on the optimizer")
    return names, params, moments


def _adam_update(names, params, grads, moments):
    """One differentiable single-tensor Adam step. Does not write the real optimizer."""
    new_params = dict(params)
    new_moments = dict(moments)
    for name, grad in zip(names, grads):
        moment = moments[name]
        if grad is None:
            continue
        group = moment["group"]
        grad = _guard_grad(grad, moment)
        updated, exp_avg, exp_avg_sq, step = _single_adam(
            params[name], grad, moment["exp_avg"], moment["exp_avg_sq"], moment["step"], group,
        )
        new_params[name] = updated
        new_moments[name] = {**moment, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step}
    return new_params, new_moments


def _guard_grad(grad, moment):
    """K3P spike guard as a detached scale. Identity (exact) when it does not clip."""
    guard = moment["guard"]
    if guard is None or moment["step"] < guard.min_steps:
        return grad
    beta2 = moment["group"]["betas"][1]
    vhat = moment["exp_avg_sq"].detach().mean() / (1.0 - beta2 ** moment["step"])
    ratio = grad.detach().square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()
    if float(ratio) > guard.ratio:
        return grad * (guard.ratio / float(ratio))
    return grad


def _single_adam(param, grad, exp_avg, exp_avg_sq, completed, group):
    """Match ``torch.optim.Adam``'s single-tensor update (foreach/fused off)."""
    beta1, beta2 = group["betas"]
    lr = float(group["lr"])
    eps = float(group["eps"])
    decay = float(group.get("weight_decay", 0.0) or 0.0)
    if group.get("maximize", False):
        grad = -grad
    if decay != 0.0:
        grad = torch.add(grad, param, alpha=decay)
    step = completed + 1
    exp_avg = torch.lerp(exp_avg, grad, 1 - beta1)
    exp_avg_sq = torch.addcmul(exp_avg_sq * beta2, grad, grad, value=1 - beta2)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr / bias_correction1
    # Forward values match single-tensor Adam. The second-moment scale is a
    # constant for dG: sqrt(v) has no derivative at coordinates v has not seen,
    # and beta2 = 0.999 barely moves v over a handful of steps. The reaction
    # that G differentiates is the first-moment step of D.
    denom = (exp_avg_sq.detach().sqrt() / (bias_correction2 ** 0.5)).add(eps)
    param = torch.addcdiv(param, exp_avg, denom, value=-step_size)
    return param, exp_avg, exp_avg_sq.detach(), step


def _disable_inplace(module):
    changed = []
    for child in module.modules():
        flag = getattr(child, "inplace", None)
        if isinstance(flag, bool) and flag and isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
            changed.append(child)
            child.inplace = False
    return changed


def _restore_inplace(changed):
    for child in changed:
        child.inplace = True


def _record_of(penalty):
    if penalty is None:
        return None
    record = getattr(penalty, "record", None)
    if record is None:
        record = getattr(getattr(penalty, "regularizer", None), "record", None)
    if record is None or not hasattr(record, "state_dict"):
        return None
    return record


def _snap_mechanism():
    mechanism = sys.modules.get("mechanism")
    if mechanism is None or not hasattr(mechanism, "receipt"):
        return None
    return copy.deepcopy(mechanism.receipt)


def _restore_mechanism(saved):
    if saved is None:
        return
    receipt_ref = sys.modules["mechanism"].receipt
    receipt_ref.clear()
    receipt_ref.update(saved)


@contextmanager
def _preserve_rng(module):
    """Inner D forwards may draw noise. The training streams stay put."""
    generators = []
    seen = set()

    def add(generator):
        if isinstance(generator, torch.Generator) and id(generator) not in seen:
            seen.add(id(generator))
            generators.append(generator)

    for child in module.modules():
        add(getattr(child, "generator", None))
        policy = getattr(child, "policy", None)
        if policy is not None:
            add(getattr(policy, "input_stream", None))
            add(getattr(policy, "output_stream", None))
    saved = [(generator, generator.get_state().clone()) for generator in generators]
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        for generator, state in saved:
            generator.set_state(state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
