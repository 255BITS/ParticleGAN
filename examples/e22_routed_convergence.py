"""Policy-aware conditional convergence fixture, outside Forge qualification.

Two frozen BF16 sites and an exactly reachable ordinary-LoRA teacher compare
ordinary native-game training with genuine shared-bank gated-particle training.
An explicitly separate ordinary MSE/AdamW reference retains the historical
objective/optimizer. See docs/e22_routed_convergence_v1.json for the fixed law.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn
import torch.nn.functional as F

from particlegan import E22Policy, RoutedBatch, RoutedRows, UpdatePolicy, get_recipe, init


ARMS = ("ordinary_native_game", "particle_native_game", "ordinary_mse_reference")
WIDTH, TOKENS, RANK, PARTICLES, Z_DIM, BATCH_SIZE = 16, 16, 2, 128, 4, 4
BRANCH_LR, TEACHER_UP_ROW_NORM, PANEL_SIGMA, PANEL_DRAWS = 5e-5, .5, .125, 4
SPLITS = {"fit": ((.1, .35, .6, .85), range(4)),
          "guard": ((.22, .72), range(4, 6)),
          "test": ((.18, .43, .68, .93), range(6, 10))}


def digest(value):
    result = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                result.update(str(key).encode())
                add(item[key])
        elif isinstance(item, (tuple, list)):
            for element in item:
                add(element)
        else:
            result.update(json.dumps(item, sort_keys=True).encode())
    add(value)
    return result.hexdigest()


def initialize(module, role):
    seeds = {name: int.from_bytes(hashlib.sha256(("convergence-v1:" + role + ":" + name).encode()).digest()[:8],
                                  "little") % (2**63 - 1)
             for name, parameter in module.named_parameters() if parameter.requires_grad and parameter.numel()}
    generators = {name: torch.Generator(device="cpu").manual_seed(seed) for name, seed in seeds.items()}
    init.initialize_(module, method="sample_distributions_v1", parameter_generators=generators)
    return seeds


class Projection(nn.Module):
    def __init__(self, particle):
        super().__init__()
        self.base = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.base.weight.copy_(torch.eye(WIDTH))
        self.base.requires_grad_(False)
        self.down = nn.Linear(WIDTH, RANK, bias=False)
        if particle:
            self.bridge = nn.Linear(RANK + Z_DIM, RANK)
        self.up = nn.Linear(RANK, WIDTH, bias=False)

    def forward(self, value, codes=None):
        base = self.base(value.to(torch.bfloat16)).float()
        hidden = self.down(value.float())
        if hasattr(self, "bridge"):
            hidden_modulation = F.linear(hidden, self.bridge.weight[:, :RANK], self.bridge.bias).tanh()
            gate = F.linear(codes, self.bridge.weight[:, RANK:]).tanh()
            hidden = hidden + hidden_modulation + hidden * gate
        return (base + self.up(hidden)).to(torch.bfloat16).float()


class Host(nn.Module):
    def __init__(self, particle=False):
        super().__init__()
        self.first, self.second = Projection(particle), Projection(particle)

    def zero_up(self):
        with torch.no_grad():
            for branch in (self.first, self.second):
                branch.up.weight.zero_()

    def forward(self, context):
        hidden = self.first(context[..., :WIDTH]).to(torch.bfloat16).tanh().float()
        return self.second(hidden)

    def forward_routed(self, context, router, candidate, routing):
        value = context[..., :WIDTH]
        first = routing.mix("first", router.first_query(value) @ candidate.table.T / math.sqrt(Z_DIM))
        hidden = self.first(value, first).to(torch.bfloat16).tanh().float()
        second = routing.mix("second", router.second_query(hidden) @ candidate.table.T / math.sqrt(Z_DIM))
        return self.second(hidden, second)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_query, self.second_query = nn.Linear(WIDTH, Z_DIM), nn.Linear(WIDTH, Z_DIM)
        self.register_buffer("log_mass", torch.zeros(PARTICLES))


class FrozenContexts(nn.Module):
    def __init__(self, sources):
        super().__init__()
        self.register_buffer("sources", sources.clone())

    def condition(self, context):
        return context[:, 0, WIDTH:]


class ConditionalCritic(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.error_input = nn.Linear(WIDTH, 48)
        self.condition_input = nn.Linear(769, 48, bias=False)
        self.feature_output = nn.Linear(48, 16)
        self.score = nn.Linear(16, 1)
        self.register_buffer("scale", scale.clone())

    def features(self, error, condition):
        hidden = (self.error_input(error) + self.condition_input(condition)[:, None]).tanh()
        return self.feature_output(hidden).tanh()

    def forward(self, error, condition):
        return self.score(self.features(error, condition).mean(1))


class TokenPenaltyView(nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, error, condition):
        return self.critic(error.reshape(-1, TOKENS, WIDTH), condition).repeat_interleave(TOKENS, 0)


def model_forward(models, context, candidate, routing):
    return models["generator"].forward_routed(context, models["router"], candidate, routing)


def row_features(models, context, samples, targets):
    critic = models["critic"]
    return critic.features((samples - targets) / critic.scale,
                           models["encoder"].condition(context)).flatten(1)


@torch.no_grad()
def batched_host(model, context):
    return torch.cat([model(context[start:start + BATCH_SIZE]) for start in range(0, len(context), BATCH_SIZE)])


def make_data():
    with torch.random.fork_rng(devices=[]):
        ordinary, particle, teacher = Host(), Host(True), Host()
    maps = {"generator": initialize(ordinary, "generator"),
            "particle_generator": initialize(particle, "generator"),
            "teacher": initialize(teacher, "teacher")}
    ordinary.zero_up()
    particle.zero_up()
    with torch.no_grad():
        for name in ("first", "second"):
            source, destination = getattr(ordinary, name), getattr(teacher, name)
            destination.down.weight.copy_(source.down.weight)
            destination.up.weight.mul_(TEACHER_UP_ROW_NORM / destination.up.weight.norm(dim=1, keepdim=True))
    teacher.eval().requires_grad_(False)
    channel = torch.arange(768, dtype=torch.float32)[None]
    source_id = torch.arange(1, 7, dtype=torch.float32)[:, None]
    sources = ((source_id * (channel + 1) * .017).sin() + (source_id * (channel + 1) * .031).cos())
    sources *= .052 / sources.square().mean(1, keepdim=True).sqrt()
    projection = torch.cos((torch.arange(WIDTH)[:, None] + 1) * (channel + 1) * .013)
    projection /= projection.norm(dim=1, keepdim=True)
    projected_source = sources @ projection.T
    data = {"sources": sources, "source_projection": projection, "initial_ordinary": ordinary.state_dict(),
            "initial_particle": particle.state_dict(), "teacher": teacher.state_dict(), "initialization": maps}
    axis = torch.arange(1, WIDTH + 1, dtype=torch.float32)[None, None]
    token = torch.arange(1, TOKENS + 1, dtype=torch.float32)[None, :, None]
    weights = torch.ones((1, TOKENS, 1))
    weights[:, :2] = math.sqrt(7.)
    for pool, (times, draws) in SPLITS.items():
        contexts, subjects, flow_times, latent_ids = [], [], [], []
        for subject in range(6):
            for flow_time in times:
                for draw in draws:
                    latent = ((draw + 1) * token * axis * .071).sin() + ((draw + 2) * token * axis * .039).cos()
                    latent = latent * weights
                    latent *= .77 / latent.square().mean().sqrt()
                    value = latent + projected_source[subject][None, None] + .15 * (2 * flow_time - 1) * (axis * .19).sin()
                    condition = torch.cat((sources[subject], torch.tensor([flow_time])))
                    contexts.append(torch.cat((value[0], condition[None].expand(TOKENS, -1)), dim=-1))
                    subjects.append(subject)
                    flow_times.append(flow_time)
                    latent_ids.append(draw)
        context = torch.stack(contexts)
        data[pool] = {"context": context, "targets": batched_host(teacher, context),
                      "base": batched_host(ordinary, context), "subjects": torch.tensor(subjects),
                      "times": torch.tensor(flow_times), "latent_ids": torch.tensor(latent_ids)}
    raw_scale = (data["fit"]["targets"] - data["fit"]["base"]).std(dim=(0, 1))
    data["raw_scale"], data["scale"] = raw_scale, raw_scale.clamp_min(.04)
    data["digest"] = digest(data)
    return data


@dataclass
class Loop:
    arm: str
    data: dict
    G: Host
    policy: object
    optimizer: torch.optim.Optimizer
    data_rng: torch.Generator
    paired_rng: torch.Generator
    law: dict
    completed_steps: int = 0


def make_loop(arm, data, *, bindings=None):
    if arm not in ARMS:
        raise ValueError("unknown convergence diagnostic arm")
    if data["digest"] != digest({key: value for key, value in data.items() if key != "digest"}):
        raise ValueError("fixture data changed after its digest was bound")
    particle = arm == "particle_native_game"
    with torch.random.fork_rng(devices=[]):
        generator = Host(particle)
    init_map = initialize(generator, "generator")
    generator.zero_up()
    generator.load_state_dict(data["initial_particle" if particle else "initial_ordinary"], strict=True)
    policy, recipe = None, None
    if arm == "ordinary_mse_reference":
        optimizer = torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad],
                                      lr=BRANCH_LR, weight_decay=.01)
        extras = {"reference_objective": "paired_output_mse", "clip_grad_norm": 1.}
    else:
        options = dict(num_particles=PARTICLES, z_dim=Z_DIM, batch_size=BATCH_SIZE,
                       output_noise_std=.125, reopen_guard="settled")
        options.update(birth_death_backend="auto" if particle else "knn")
        if not particle:
            options.update(particle_birth_death=False, row_evidence_gate=False,
                           birth_death_feature_scale="none", birth_death_isolation=False)
        recipe = get_recipe("e22_routed" if particle else "e22", **options)
        with torch.random.fork_rng(devices=[]):
            critic, contexts = ConditionalCritic(data["scale"]), FrozenContexts(data["sources"])
            prior = recipe.make_prior()
            router = Router() if particle else None
        init_map = {"generator": init_map, "critic": initialize(critic, "critic")}
        init.deterministic_orthogonal_(prior)
        table = prior.z.requires_grad_(particle)
        groups = [{"params": [p for p in generator.parameters() if p.requires_grad], "lr": BRANCH_LR}]
        roles = ["generator"]
        if particle:
            init_map["router"] = initialize(router, "router")
            groups.extend([{"params": list(router.parameters()), "lr": BRANCH_LR},
                           {"params": [table], "lr": recipe.lr * recipe.prior_lr_mult}])
            roles.extend(["router", "table"])
        optimizer = recipe.make_generator_optimizer(groups, latent_table=table if particle else None, foreach=False)
        critic_optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
        common = dict(table=table, encoder=contexts, generator_optimizer=optimizer,
                      critic_optimizer=critic_optimizer, roles=[roles, ["critic"]], seed=21)
        if particle:
            rows = RoutedRows(model_forward=model_forward, features=row_features, sites=("first", "second"),
                              probe_interval=100, max_context_harm=0., output_error_guard=False)
            policy = E22Policy(recipe, generator, critic, router=router, routed_rows=rows, **common)
        else:
            policy = UpdatePolicy(recipe, generator, critic, row_semantics="conditional", **common)
        policy.attach_penalty(recipe.make_critic_penalty(critic_optimizer, collect_stats=True))
        extras = {"recipe": recipe.to_dict(), "init_map": init_map,
                  "owner_roles": policy.roles, "initial_lrs": policy.initial_lrs}
    law = {"task": "supra_conditional_two_site_convergence_v1", "arm": arm,
           "data_digest": data["digest"], "bindings": dict(bindings or {}), **extras}
    return Loop(arm, data, generator, policy, optimizer, torch.Generator().manual_seed(7),
                torch.Generator().manual_seed(43), law)


def forward(loop, context, *, code_ablation=False):
    if loop.arm != "particle_native_game":
        return loop.G(context)
    p = loop.policy
    if code_ablation:
        return p.routed_control.generate(context, candidate=p.routed_control.candidate(),
                                         perturb_fn=lambda codes: torch.zeros_like(codes))
    return p.routed_generate(context, sigma=0, perturb=False)


def controls(policy):
    if policy is None:
        return {}
    groups = {}
    for index, (optimizer, rates, roles, testers) in enumerate(zip(
            policy.optimizers, policy.initial_lrs, policy.roles, policy.lr_settle.testers)):
        for j, (group, rate, role, tester) in enumerate(zip(optimizer.param_groups, rates, roles, testers)):
            groups[f"{index}.{j}"] = {"role": role, "base_lr": rate, "applied_lr": group["lr"],
                                     "effective_scale": group["lr"] / rate,
                                     "tester": None if tester is None else tester.diagnostics()}
    eligible = all(tester.s <= 1 / 64 for row, roles in zip(policy.lr_settle.testers, policy.roles)
                   for tester, role in zip(row, roles) if tester is not None and role not in ("critic", "noise"))
    return {"groups": groups, "noise_floor_eligible": eligible,
            "noise_unconstrained": float(policy.log_output_sigma.detach().exp()),
            "actual_sigma": policy.output_sigma(), "controller": policy.controller.diagnostics(),
            "surprise": policy.surprise.diagnostics(),
            "ka2": policy.opt_d.record.state_dict(),
            "routing": None if policy.routed_control is None else policy.routed_control.diagnostics(),
            "backend": None if policy._feature_selection is None else deepcopy(policy._feature_selection.state)}


def update(loop):
    indices = torch.randint(len(loop.data["fit"]["context"]), (BATCH_SIZE,), generator=loop.data_rng)
    context, targets = (loop.data["fit"][name][indices] for name in ("context", "targets"))
    loop.G.train()
    if loop.policy is None:
        loss = F.mse_loss(loop.G(context), targets)
        loop.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_([p for p in loop.G.parameters() if p.requires_grad], 1.,
                                                       error_if_nonfinite=True))
        loop.optimizer.step()
        loop.completed_steps += 1
        return {"step": loop.completed_steps, "batch_indices": indices.tolist(), "loss_reference_mse": float(loss.detach()),
                "gradient_norm": gradient, "data_rng": digest(loop.data_rng.get_state())}
    p = loop.policy
    routed = (RoutedBatch(context, targets, loop.data["guard"]["context"], loop.data["guard"]["targets"])
              if p.routed_control is not None else None)
    noise = p.begin_step(targets, routed=routed)
    condition = p.encoder.condition(context)
    critic_base = torch.randn(targets.shape, generator=loop.paired_rng)
    generator_base = torch.randn(targets.shape, generator=loop.paired_rng)
    p.D.train()
    with torch.no_grad():
        prediction = (p.routed_generate(context, sigma=0, perturb=True) if p.routed_control is not None else p.G(context))
        real = noise.output_sigma * critic_base
        fake = real + (prediction - targets) / p.D.scale
    p.observe_critic_pair(real, fake)
    penalty = p.penalty(TokenPenaltyView(p.D), real.flatten(0, 1), fake.flatten(0, 1), condition)
    loss_d_game = p.recipe.make_loss().d_loss(p.D(real, condition), p.D(fake, condition))
    loss_d = loss_d_game + penalty
    if not torch.isfinite(loss_d):
        raise FloatingPointError("native critic game or KA2 penalty became nonfinite")
    p.opt_d.zero_grad(set_to_none=True)
    p.before_critic_backward()
    loss_d.backward()
    p.opt_d.step()
    p.after_critic_step()
    p.D.eval()
    flags = [parameter.requires_grad for parameter in p.D.parameters()]
    try:
        p.D.requires_grad_(False)
        prediction = (p.routed_generate(context, sigma=0, perturb=True) if p.routed_control is not None else p.G(context))
        real = noise.output_sigma * generator_base
        with torch.no_grad():
            real_logits = p.D(real.detach(), condition)
        loss_g = p.recipe.make_loss().g_loss(p.D(real + (prediction - targets) / p.D.scale, condition), real_logits)
        if not torch.isfinite(loss_g):
            raise FloatingPointError("native generator game became nonfinite")
        p.opt_g.zero_grad(set_to_none=True)
        p.before_generator_backward()
        loss_g.backward()
        bank_rows = 0 if p.table.grad is None else int(p.table.grad.norm(dim=-1).gt(0).sum())
        bank_gradient = 0. if p.table.grad is None else float(p.table.grad.norm())
        query_gradient = (0. if p.router is None else math.sqrt(sum(float(parameter.grad.square().sum())
                           for parameter in p.router.parameters() if parameter.grad is not None)))
        p.after_generator_backward(loss_gan=loss_g.detach(), loss_critic=loss_d_game.detach())
        p.opt_g.step()
        p.after_generator_step()
    finally:
        for parameter, flag in zip(p.D.parameters(), flags):
            parameter.requires_grad_(flag)
    event = p.finish_step()
    loop.completed_steps = p.completed_steps
    return {"step": p.completed_steps, "batch_indices": indices.tolist(), "loss_g": float(loss_g.detach()),
            "loss_d_game": float(loss_d_game.detach()), "penalty": float(penalty.detach()),
            "bank_gradient_rows": bank_rows, "bank_gradient_norm": bank_gradient,
            "query_gradient_norm": query_gradient, "move": event,
            "paired_base_digest": digest((critic_base, generator_base)), "data_rng": digest(loop.data_rng.get_state()),
            "paired_rng": digest(loop.paired_rng.get_state()), "dv12_rng": digest(p.noise_generator.get_state()),
            "controls": controls(p)}


def modules(loop):
    if loop.policy is None:
        return {"generator": loop.G}
    p = loop.policy
    values = p._training_modules()
    values.update({"average_" + role: module for role, module in p._average_modules().items()})
    values["critic_average"] = p.opt_d.ema_critic
    return values


def parameters(loop):
    values = {role + "." + name: value for role, module in modules(loop).items() for name, value in module.named_parameters()}
    if loop.policy is not None:
        values.update(table=loop.policy.table, noise=loop.policy.log_output_sigma)
    return values


def checkpoint(loop):
    return deepcopy({"law": loop.law, "step": loop.completed_steps,
                     "training": (loop.policy.state_dict() if loop.policy is not None else
                                  {"generator": loop.G.state_dict(), "optimizer": loop.optimizer.state_dict()}),
                     "data_rng": loop.data_rng.get_state(), "paired_rng": loop.paired_rng.get_state(),
                     "modes": {role + "." + name: module.training for role, model in modules(loop).items()
                               for name, module in model.named_modules()},
                     "gradients": {name: None if value.grad is None else value.grad.detach().clone()
                                   for name, value in parameters(loop).items()}})


def restore(loop, state):
    required = {"law", "step", "training", "data_rng", "paired_rng", "modes", "gradients"}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("checkpoint schema does not match the convergence fixture")
    if state["law"] != loop.law:
        raise ValueError("arm, data, native recipe, initialization and source/card bindings must match")
    if not isinstance(state["step"], int) or state["step"] < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if loop.policy is not None and state["step"] != state["training"].get("completed_steps"):
        raise ValueError("fixture and native completed-step counters must match")
    expected_modes = {role + "." + name for role, model in modules(loop).items()
                      for name, module in model.named_modules()}
    if (not isinstance(state["modes"], dict) or set(state["modes"]) != expected_modes
            or not all(isinstance(flag, bool) for flag in state["modes"].values())):
        raise ValueError("checkpoint module-mode topology does not match")
    expected_parameters = parameters(loop)
    if not isinstance(state["gradients"], dict) or set(state["gradients"]) != set(expected_parameters):
        raise ValueError("checkpoint gradient topology does not match")
    for name, gradient in state["gradients"].items():
        if gradient is not None and (not isinstance(gradient, torch.Tensor)
                                     or gradient.shape != expected_parameters[name].shape
                                     or gradient.dtype != expected_parameters[name].dtype):
            raise ValueError("checkpoint gradient shape or dtype does not match")
    for key in ("data_rng", "paired_rng"):
        stream = loop.data_rng if key == "data_rng" else loop.paired_rng
        if (not isinstance(state[key], torch.Tensor) or state[key].dtype != torch.uint8
                or state[key].shape != stream.get_state().shape):
            raise ValueError("checkpoint named stream state does not match")
    if loop.policy is not None:
        loop.policy.load_state_dict(state["training"])
    else:
        loop.G.load_state_dict(state["training"]["generator"], strict=True)
        loop.optimizer.load_state_dict(state["training"]["optimizer"])
    loop.completed_steps = state["step"]
    loop.data_rng.set_state(state["data_rng"].cpu())
    loop.paired_rng.set_state(state["paired_rng"].cpu())
    for role, model in modules(loop).items():
        for name, module in model.named_modules():
            module.training = state["modes"][role + "." + name]
    for name, value in parameters(loop).items():
        gradient = state["gradients"][name]
        value.grad = None if gradient is None else gradient.to(value).clone()


def evaluation_panels(data):
    generator = torch.Generator().manual_seed(72)
    return {pool: PANEL_SIGMA * torch.randn((PANEL_DRAWS, *data[pool]["targets"].shape), generator=generator)
            for pool in SPLITS}


@torch.no_grad()
def evaluate(loop, judge, pool, panels, *, code_ablation=False):
    values = loop.data[pool]
    predictions = torch.cat([forward(loop, values["context"][start:start + BATCH_SIZE], code_ablation=code_ablation)
                             for start in range(0, len(values["context"]), BATCH_SIZE)])
    residual = (predictions - values["targets"]) / loop.data["scale"]
    return score_residual(judge, values["context"], residual, panels[pool]) | {
        "output_rmse_diagnostic": float((predictions - values["targets"]).square().mean().sqrt())}


@torch.no_grad()
def score_residual(judge, context, residual, bases):
    condition = context[:, 0, WIDTH:]
    loss = get_recipe("e22").make_loss()
    logits = [(judge(base + residual, condition), judge(base, condition)) for base in bases]
    paired = torch.stack([F.softplus(-(fake - real)).flatten() for fake, real in logits]).mean(0)
    paired_mean = torch.stack([loss.g_loss(fake, real) for fake, real in logits]).mean()
    clean_fake, clean_real = judge(residual, condition), judge(torch.zeros_like(residual), condition)
    features = (judge.features(residual, condition) - judge.features(torch.zeros_like(residual), condition)).square().mean((1, 2))
    anchor = float(loss.g_loss(judge(bases[0], condition), judge(bases[0], condition)))
    return {"paired_game": float(paired_mean), "paired_game_by_context": paired.tolist(),
            "clean_zero_noise_game": float(loss.g_loss(clean_fake, clean_real)), "feature_proxy": float(features.mean()),
            "teacher_zero_residual_anchor": anchor}


@torch.no_grad()
def reachability_witness(data):
    ordinary = make_loop("ordinary_native_game", data)
    particle = make_loop("particle_native_game", data)
    for name in ("first", "second"):
        branch = getattr(particle.G, name)
        branch.down.weight.copy_(data["teacher"][name + ".down.weight"])
        branch.up.weight.copy_(data["teacher"][name + ".up.weight"])
        branch.bridge.weight.zero_()
        branch.bridge.bias.zero_()
        ordinary_branch = getattr(ordinary.G, name)
        ordinary_branch.down.weight.copy_(data["teacher"][name + ".down.weight"])
        ordinary_branch.up.weight.copy_(data["teacher"][name + ".up.weight"])
    exact = {}
    for arm, loop in (("ordinary", ordinary), ("particle", particle)):
        exact[arm] = {}
        for pool in SPLITS:
            prediction = torch.cat([forward(loop, data[pool]["context"][start:start + BATCH_SIZE])
                                    for start in range(0, len(data[pool]["context"]), BATCH_SIZE)])
            exact[arm][pool] = torch.equal(prediction, data[pool]["targets"])
    if not all(value for pools in exact.values() for value in pools.values()):
        raise AssertionError("teacher is not exactly reachable by both declared adapter families")
    return exact
