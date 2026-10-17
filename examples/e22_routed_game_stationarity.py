"""Frozen-critic local game-stationarity witness, with zero optimizer updates.

An exact local solution need not be stationary against an arbitrary frozen GAN
critic. Even critic features/scores plus paired noise signs enforce that local
property. This is a mathematical ingredient check, not joint GAN equilibrium
or evidence that the late full-Supra convergence gap has been explained.
Reflection evenization removes first-order signed mean discrimination and can
slow acquisition; point-stationarity does not establish optimizer improvement.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch

from particlegan import GANLoss, RoutedRows

if __package__:
    from . import e22_routed_convergence as held
else:
    import e22_routed_convergence as held


TASK = "routed_frozen_critic_game_stationarity_v1"
CASES = (("current_single", False, False), ("current_antithetic", False, True),
         ("even_single", True, False), ("even_antithetic", True, True))
GRADIENT_ATOL = 1e-6


class ReflectionEvenCritic(held.ConditionalCritic):
    """Same learned tensors/names; reflection-even features and scalar score.

    The held forward token-pools these features and applies its linear score
    head. It therefore also evenizes the score without introducing parameters.
    """
    def features(self, error, condition):
        return .5 * (super().features(error, condition) + super().features(-error, condition))


def paired_games(critic, residual, condition, epsilon, *, antithetic):
    """Native context-pooled RpGAN, with one or two signs of one fixed draw."""
    bases = torch.stack((epsilon, -epsilon)) if antithetic else epsilon.unsqueeze(0)
    signs, contexts = bases.shape[:2]
    repeated = condition.repeat(signs, 1)
    with torch.no_grad():
        real = critic(bases.flatten(0, 1), repeated).reshape(signs, contexts, -1)
    fake = critic((bases + residual.unsqueeze(0)).flatten(0, 1), repeated).reshape(signs, contexts, -1)
    loss = GANLoss()
    generator = torch.stack([loss.g_loss(fake[:, row], real[:, row]) for row in range(contexts)])
    discriminator = torch.stack([loss.d_loss(real[:, row], fake[:, row]) for row in range(contexts)])
    return generator, discriminator


def fit_contexts(data):
    """Private CPU7: two sorted indices per unchanged subject, in ID order."""
    stream = torch.Generator(device="cpu").manual_seed(7)
    indices = []
    for subject in range(6):
        rows = (data["fit"]["subjects"] == subject).nonzero().flatten()
        indices.extend(rows[torch.randperm(len(rows), generator=stream)[:2]].sort().values.tolist())
    return indices, stream.get_state()


def private_panels():
    """One CPU43 panel per D/G role; signs reuse it without additional draws."""
    raw = torch.randn((2, 12, held.TOKENS, held.WIDTH),
                      generator=torch.Generator(device="cpu").manual_seed(43))
    return raw


def owner_identity(loop, critics):
    modules = {**held.modules(loop), **{"judge/" + name: critic for name, critic in critics.items()}}
    return dict(native_checkpoint=held.digest(held.checkpoint(loop)),
        judge_tensors={name: held.digest(critic.state_dict()) for name, critic in critics.items()},
        runtime={role + "/" + name: dict(training=module.training,
            forward_hooks=list(module._forward_hooks), pre_hooks=list(module._forward_pre_hooks),
            backward_hooks=list(module._backward_hooks))
            for role, model in modules.items() for name, module in model.named_modules()},
        flags={role + "/" + name: parameter.requires_grad
               for role, model in modules.items() for name, parameter in model.named_parameters()},
        judge_grads={name: [None if parameter.grad is None else held.digest(parameter.grad)
                           for parameter in critic.parameters()] for name, critic in critics.items()},
        data=held.digest(loop.data), global_cpu_rng=held.digest(torch.get_rng_state()))


@contextmanager
def observed_forward(loop):
    """An owner-free public spec delegates every actual routed mixer once."""
    codes, calls = {}, []
    callback = loop.policy.routed_control.spec.model_forward
    class Observer:
        def __init__(self, routing):
            self.routing = routing
        def mix(self, site, logits):
            if site in codes:
                raise AssertionError("duplicate actual routing site")
            codes[site] = self.routing.mix(site, logits)
            return codes[site]
    def forward(models, context, candidate, routing):
        calls.append(1)
        return callback(models, context, candidate, Observer(routing))
    original = loop.policy.routed_control.spec
    spec = RoutedRows(model_forward=forward, features=original.features, sites=original.sites,
                      probe_interval=original.probe_interval, max_context_harm=0., output_error_guard=False)
    yield spec, codes, calls


def _norm(values):
    return math.sqrt(sum(float(value.detach().double().square().sum()) for value in values))


def gradients(objective, loop, raw_residual, normalized_residual, codes):
    generator = {name: value for name, value in loop.G.named_parameters() if value.requires_grad}
    router = dict(loop.policy.router.named_parameters())
    owners = {"raw_residual": raw_residual, "normalized_residual": normalized_residual,
              **{"G/" + name: value for name, value in generator.items()},
              **{"router/" + name: value for name, value in router.items()},
              "table": loop.policy.table, **{"code/" + name: value for name, value in codes.items()}}
    values = torch.autograd.grad(objective, tuple(owners.values()), retain_graph=True, allow_unused=True)
    result = {name: torch.zeros_like(owner) if value is None else value
              for (name, owner), value in zip(owners.items(), values)}
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("nonfinite known-solution game gradient")
    groups = {
        "raw_residual": [result["raw_residual"]], "normalized_residual": [result["normalized_residual"]],
        "generator_Up": [value for name, value in result.items() if name.startswith("G/") and name.endswith("up.weight")],
        "generator_down": [value for name, value in result.items() if name.startswith("G/") and name.endswith("down.weight")],
        "generator_H_b": [value[:, :held.RANK] if name.endswith("bridge.weight") else value
                           for name, value in result.items() if name.startswith("G/") and ".bridge." in name],
        "generator_C": [value[:, held.RANK:] for name, value in result.items()
                         if name.startswith("G/") and name.endswith("bridge.weight")],
        "bank": [result["table"]],
        "router": [value for name, value in result.items() if name.startswith("router/")],
        **{"code/" + site: [result["code/" + site]] for site in codes},
    }
    norms = {name: _norm(values) for name, values in groups.items()}
    return dict(norms=norms, all_owners_gradient_norm=_norm(list(result.values())),
        negative_residual_gradient_directional_derivative=-norms["raw_residual"]**2,
        negative_Up_gradient_native_autograd_linearization_slope=-norms["generator_Up"]**2,
        directional_scope="Exact continuous residual-coordinate derivative; the Up quantity is native autograd linearization through BF16 casts, not a literal infinitesimal derivative of the quantized parameter function. No step is taken.",
        local_stationarity_verified=norms["normalized_residual"] <= GRADIENT_ATOL
            and norms["raw_residual"] <= GRADIENT_ATOL and norms["generator_Up"] <= GRADIENT_ATOL,
        stationarity_absolute_norm_tolerance=GRADIENT_ATOL)


def observe_batch(loop, judges, context, raw_panels):
    """One actual clean FAST G graph shared by all four ingredients/judges."""
    models = loop.policy._training_modules()
    with observed_forward(loop) as (spec, codes, calls):
        candidate = spec.candidate_for(models, loop.policy.table)
        output, usage = spec.forward_with_usage(models, context, candidate)
    if calls != [1] or tuple(codes) != ("first", "second") or usage.shape != (held.BATCH_SIZE, held.PARTICLES):
        raise AssertionError("not exactly one complete native B4 routed forward")
    target = output.detach().clone()
    raw_residual = output - target
    if raw_residual.count_nonzero():
        raise AssertionError("constructed exact local solution is not zero")
    condition = loop.policy.encoder.condition(context)
    results = []
    for judge_name, (current, even) in judges.items():
        normalized = raw_residual / current.scale
        for case, reflection, antithetic in CASES:
            critic = even if reflection else current
            g, _ = paired_games(critic, normalized, condition, .125 * raw_panels[1], antithetic=antithetic)
            with torch.no_grad():
                _, d = paired_games(critic, normalized.detach(), condition, .125 * raw_panels[0], antithetic=antithetic)
            rows = []
            for row in range(len(context)):
                rows.append(dict(batch_row=row, generator_game=float(g[row].detach()),
                    critic_role_game=float(d[row]), **gradients(g[row], loop, raw_residual, normalized, codes)))
            results.append(dict(judge=judge_name, case=case, reflection_even=reflection,
                antithetic=antithetic, noise_sign_count=2 if antithetic else 1,
                native_context_count=len(context), tokens_per_context=held.TOKENS,
                known_solution_residual_nonzero_coordinates=0,
                known_solution_generator_anchor=math.log(2), per_context=rows,
                batch_mean=dict(generator_game=float(g.mean().detach()),
                    **gradients(g.mean(), loop, raw_residual, normalized, codes)),
                batch_mean_scope="Includes summation across the four contexts; individual contexts are also retained."))
    return results


def fresh_witness(loop):
    if loop.completed_steps or loop.policy.completed_steps:
        raise AssertionError("stationarity witness must be fresh and zero-update")
    for site in ("first", "second"):
        branch = getattr(loop.G, site)
        if branch.up.weight.count_nonzero() or not branch.bridge.weight[:, held.RANK:].count_nonzero():
            raise AssertionError("expected zero-Up and preserved sampled nonzero particle C")
        if not all(value.requires_grad for value in branch.parameters() if value.dtype != torch.bfloat16):
            raise AssertionError("all FP32 particle branch parameters must remain trainable")
    if not loop.policy.table.requires_grad or not loop.policy.table.count_nonzero():
        raise AssertionError("nonzero trainable shared bank required")
    if not all(parameter.requires_grad for parameter in loop.policy.router.parameters()):
        raise AssertionError("live trainable router required")
    if not any(parameter.count_nonzero() for parameter in loop.policy.router.parameters()):
        raise AssertionError("nonzero sampled router required")
    return dict(rank=held.RANK, sites=2, particles=held.PARTICLES, z_dim=held.Z_DIM,
        public_initializer="particlegan.init.initialize_(sample_distributions_v1), unchanged held per-parameter streams",
        init_map=deepcopy(loop.law["init_map"]), zero_Up=True, sampled_C_nonzero=True,
        bridge_bank_router_trainable=True, native_recipe=loop.policy.recipe.to_dict(),
        all_non_Up_game_gradients_at_zero_Up_expected_zero="Code, H/b/C, down, bank and router paths are multiplied by zero Up; their zero gradients are not evidence of particle inactivity after acquisition.")
