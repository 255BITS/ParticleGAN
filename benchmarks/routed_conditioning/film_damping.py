"""GAN-only conditional FiLM diagnostic using the complete public E22 policy.

Run from the repository root, with bulk artifacts outside the checkout::

    python -u -m benchmarks.routed_conditioning.film_damping --steps 1200 \
        --output /tmp/routed-film-damping

This is an explicitly coupled particle-cloud diagnostic (sigma=0), rather than
a Forge qualification host. All arms keep native E22 observers and controls.
The third arm restores only the applied generator-group LR after begin_step;
the public after_generator_step hook therefore observes the actual LR ratio.
MSE is a clean held-out measurement, never a training or row-acceptance loss.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from particlegan import E22Policy, ParticlePrior, RoutedBatch, RoutedRows, get_recipe, init


PROFILES = ("original_native", "shift_zero_native", "shift_zero_g_bypass")
EXTRA_PROFILES = ("shift_zero_antithetic", "shift_zero_antithetic_g_bypass")
ANTITHETIC_PROFILES = frozenset(EXTRA_PROFILES)
BYPASS_PROFILES = frozenset(("shift_zero_g_bypass", "shift_zero_antithetic_g_bypass"))
NAMED_SEEDS = {"constructor": 123, "generator_init": 0, "critic_init": 1,
               "encoder_init": 2, "router_init": 3, "policy": 21,
               "batch": 42, "paired_noise": 43}


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def time_features(context):
    u = context[:, 2]
    return torch.stack((u, (math.pi * u).sin(), (math.pi * u).cos()), dim=1)


class FrozenHost(nn.Module):
    """Pinned BF16 prefix/head with an FP32 differentiable head boundary."""

    def __init__(self):
        super().__init__()
        self.prefix = nn.Linear(2, 8, bias=False).bfloat16()
        self.suffix = nn.Linear(8, 4, bias=False).bfloat16()
        with torch.no_grad():
            self.prefix.weight.copy_(torch.tensor([
                [1., 0.], [0., 1.], [.5, .5], [.5, -.5],
                [-.6, .2], [.2, -.6], [.7, .3], [-.3, .7]]))
            self.suffix.weight.copy_(torch.tensor([
                [.8, .1, .1, .2, 0., 0., .1, 0.],
                [.1, .8, .1, -.2, 0., 0., 0., .1],
                [.2, -.2, .4, .1, .2, 0., 0., 0.],
                [-.2, .2, .2, -.1, 0., .2, 0., 0.]]))
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, source):
        return self.prefix(source.bfloat16()).float()

    def decode(self, features):
        # Frozen weights still transmit derivatives to trainable features.
        return F.linear(features, self.suffix.weight.float())

    def forward(self, context):
        return self.decode(self.encode(context[:, :2]))


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, value):
        return value + .1 * self.second(F.silu(self.first(F.silu(value))))


class FiLMGenerator(nn.Module):
    """Source projection, bounded particle/time FiLM, residual body, small heads."""

    def __init__(self, width=16, blocks=2):
        super().__init__()
        self.host = FrozenHost()
        self.project = nn.Linear(8, width)
        self.condition = nn.Linear(4 + 3, 2 * width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.output = nn.Linear(width, 8)
        self.skip = nn.Linear(8, 8)
        with torch.no_grad():
            for layer in (self.condition, self.output, self.skip):
                layer.bias.zero_()
                layer.weight.mul_(.1)

    def forward(self, context, code):
        source = self.host.encode(context[:, :2])
        condition = torch.cat((F.layer_norm(code, (4,), eps=1e-3), time_features(context)), 1)
        gain, shift = (.25 * self.condition(condition).tanh()).chunk(2, 1)
        hidden = self.project(source) * (1 + gain) + shift
        residual = self.skip(source) + self.output(F.silu(self.blocks(hidden)))
        # Native image handoffs predict the recipient features through these
        # small heads; there is no outer identity from source to recipient.
        return self.host.decode(residual)


def constructor_gains(module):
    for layer in (module.condition, module.output, module.skip):
        layer.weight.mul_(.1)


init.register(FiLMGenerator, {}, finalize=constructor_gains)


class SourceTimeEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(5, width), nn.SiLU(),
                                      nn.Linear(width, width), nn.SiLU())
        self.query = nn.Linear(width, 4)

    def forward(self, context):
        value = torch.cat((context[:, :2], time_features(context)), 1)
        return self.query(self.features(value))


class DenseCosineRouter(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("log_mass", torch.zeros(rows))

    def logits(self, query, table):
        return 2. * (F.normalize(query, dim=-1, eps=1e-4)
                     @ F.normalize(table, dim=-1, eps=1e-4).T)


class OddQuadraticCritic(nn.Module):
    """Learned odd branch and a free-sign, initially zero raw-error energy head."""

    def __init__(self, width):
        super().__init__()
        self.odd = nn.Sequential(nn.Linear(4, width, bias=False), nn.Tanh(),
                                 nn.Linear(width, width, bias=False), nn.Tanh())
        self.score = nn.Linear(width, 1)
        self.quadratic = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.score.bias.zero_()
            self.quadratic.weight.zero_()

    def features(self, error):
        odd = self.odd(error)
        return torch.cat((odd, error.square()), 1)

    def forward(self, error):
        return (self.score(self.odd(error)) + self.quadratic(error.square())) / math.sqrt(2.)


def routed_forward(models, context, candidate, routing):
    query = models["encoder"](context)
    code = routing.mix("global", models["router"].logits(query, candidate.table))
    return models["generator"](context, code)


def paired_features(models, context, samples, targets):
    return models["critic"].features(samples - targets)


def context_grid(source, times):
    x, y, u = torch.meshgrid(source, source, times, indexing="ij")
    return torch.stack((x.flatten(), y.flatten(), u.flatten()), 1)


@torch.no_grad()
def target_function(context, host, target_gain):
    """Calibrated recipient handoff plus a nonlinear source/time component."""
    x, y, u = context.unbind(1)
    edit = torch.stack((.03 * (x + .6 * u).sin() + .01 * y,
                        .025 * (y - .4 * u).sin() - .012 * x,
                        .025 * x * y + .018 * u,
                        .02 * (math.pi * u).sin() + .015 * x.square()), 1)
    return target_gain * host(context) + edit


@dataclass
class Loop:
    policy: E22Policy
    profile: str
    fit_context: torch.Tensor
    fit_targets: torch.Tensor
    guard_context: torch.Tensor
    guard_targets: torch.Tensor
    test_context: torch.Tensor
    test_targets: torch.Tensor
    batch_rng: torch.Generator
    paired_noise_rng: torch.Generator
    initial_mse: float
    metadata: dict


def make_loop(profile="original_native", *, width=16, blocks=2,
              source_extent=.4, target_gain=.15):
    if profile not in PROFILES + EXTRA_PROFILES:
        raise ValueError(f"profile must be one of {PROFILES + EXTRA_PROFILES}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(NAMED_SEEDS["constructor"])
        G, E = FiLMGenerator(width, blocks), SourceTimeEncoder(width)
        D, R = OddQuadraticCritic(width), DenseCosineRouter(128)
        prior = ParticlePrior(128, 4)
    init.deterministic_orthogonal_(G, seed=NAMED_SEEDS["generator_init"])
    init.deterministic_orthogonal_(D, seed=NAMED_SEEDS["critic_init"])
    init.deterministic_orthogonal_(E, seed=NAMED_SEEDS["encoder_init"])
    init.deterministic_orthogonal_(R, seed=NAMED_SEEDS["router_init"])
    init.deterministic_orthogonal_(prior)
    if profile != "original_native":
        with torch.no_grad():
            G.condition.weight[width:].zero_()
            G.condition.bias[width:].zero_()
    fit = context_grid(torch.linspace(-source_extent, source_extent, 7), torch.tensor([.05, .25, .5, .75, .95]))
    guard = context_grid(source_extent * torch.tensor([-.87, -.29, .29, .87]), torch.tensor([.15, .4, .65, .85]))
    test = context_grid(torch.linspace(-.92 * source_extent, .92 * source_extent, 8), torch.tensor([.12, .32, .57, .82]))
    fit_target, guard_target, test_target = [target_function(c, G.host, target_gain) for c in (fit, guard, test)]
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=16,
                        lr=.000204, d_lr_mult=1.5, prior_lr_mult=10.,
                        output_noise_std=1.3, betas=(0., .999),
                        birth_death_backend="auto", reopen_guard="settled")
    opt_g = recipe.make_generator_optimizer([
        {"params": [p for p in G.parameters() if p.requires_grad]},
        {"params": list(E.parameters())},
        {"params": [prior.z], "lr": recipe.lr * recipe.prior_lr_mult},
    ], latent_table=prior.z, foreach=False)
    opt_d = recipe.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    rows = RoutedRows(model_forward=routed_forward, features=paired_features,
                      sites=("global",), probe_interval=16, probe_budget=8,
                      reservoir_size=64, min_observations=8)
    policy = E22Policy(recipe, G, D, prior=prior, encoder=E, router=R,
                       generator_optimizer=opt_g, critic_optimizer=opt_d,
                       roles=[["generator", "encoder", "table"], ["critic"]],
                       routed_rows=rows, seed=NAMED_SEEDS["policy"])
    policy.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
    initial = float((policy.served_model().routed_forward(test) - test_target).square().mean())
    hashes = {role: tensor_hash(model.state_dict())
              for role, model in policy._training_modules().items()}
    hashes["table"] = tensor_hash({"z": policy.table})
    with torch.enable_grad():
        context = test[:8]
        query = E(context)
        code = ((R.logits(query, policy.table) + R.log_mass).softmax(-1) @ policy.table).detach().requires_grad_(True)
        prediction = G(context, code)
        jacobian_square = sum(float(torch.autograd.grad(prediction[:, coordinate].sum(), code,
                                                        retain_graph=True)[0].square().sum())
                              for coordinate in range(prediction.shape[1])) / len(context)
    with torch.no_grad():
        precondition = G.condition(torch.cat((F.layer_norm(code.detach(), (4,), eps=1e-3), time_features(context)), 1))
        condition_stats = {name: {"pre_tanh_max_abs": float(value.abs().max()),
                                  "mean_tanh_derivative": float((1 - value.tanh().square()).mean()),
                                  "saturated_fraction_derivative_lt_05": float((1 - value.tanh().square()).lt(.05).float().mean())}
                           for name, value in zip(("gain", "shift"), precondition.chunk(2, 1))}
    metadata = {"profile": profile, "width": width, "blocks": blocks,
                "source_extent": source_extent, "target_gain": target_gain,
                "named_seeds": NAMED_SEEDS, "recipe": recipe.to_dict(), "routing": rows.to_dict(),
                "initial_hashes": hashes, "data_hashes": tensor_hash({
                    "fit": fit, "fit_targets": fit_target, "guard": guard,
                    "guard_targets": guard_target, "test": test, "test_targets": test_target}),
                "initial_code_jacobian_frobenius": math.sqrt(jacobian_square),
                "initial_condition_stats": condition_stats,
                "prior_kind": "particle_cloud", "prior_sigma": 0.,
                "prior_exception": "Dense coupled key/value table is the mechanism under diagnosis.",
                "evaluation_sampling": "clean live and public served routed functions; no additive noise",
                "training_objective": "paired-error RpGAN plus native KA2 critic penalty only",
                "generator_noise_coupling": "antithetic +/- same named Gaussian" if profile in ANTITHETIC_PROFILES else "single named Gaussian",
                "parameter_counts": {role: sum(p.numel() for p in model.parameters() if p.requires_grad)
                                     for role, model in policy._training_modules().items()},
                "scope": "standalone causal/software diagnostic; no Forge qualification or default adoption"}
    return Loop(policy, profile, fit, fit_target, guard, guard_target, test, test_target,
                torch.Generator().manual_seed(NAMED_SEEDS["batch"]),
                torch.Generator().manual_seed(NAMED_SEEDS["paired_noise"]), initial, metadata)


def apply_generator_bypass(loop):
    """Restore only applied G LR; retain the native tester and its observations."""
    if loop.profile in BYPASS_PROFILES:
        loop.policy.opt_g.param_groups[0]["lr"] = loop.policy.initial_lrs[0][0]


def group_rates(policy):
    return {role: {"lr": float(group["lr"]), "applied_ratio": float(group["lr"] / base),
                   "native_scale": None if tester is None else tester.s}
            for opt, bases, testers, roles in zip(policy.optimizers, policy.initial_lrs,
                                                  policy.lr_settle.testers, policy.roles)
            for group, base, tester, role in zip(opt.param_groups, bases, testers, roles)}


def parameter_snapshots(optimizer):
    return [[parameter.detach().clone() for parameter in group["params"]]
            for group in optimizer.param_groups]


@torch.no_grad()
def displacement_energy(optimizer, roles, before):
    return {role: sum(float((parameter - old).square().sum())
                      for parameter, old in zip(group["params"], snapshots))
            for role, group, snapshots in zip(roles, optimizer.param_groups, before)}


@torch.no_grad()
def generator_block_evidence(policy, applied_ratio):
    """Passive attribution before the native hook clears a completed window."""
    tester = policy.lr_settle.testers[0][0]
    if tester.tau + applied_ratio < tester.b:
        return None
    params = [p for p in policy.G.parameters() if p.requires_grad]
    delta = tester.flat(params) - tester.anchor
    slices, offset = {}, 0
    for name, p in policy.G.named_parameters():
        if not p.requires_grad:
            continue
        component = name.split(".")[0]
        if component == "condition":
            half = p.numel() // 2
            slices.setdefault("condition_gain", []).append(slice(offset, offset + half))
            slices.setdefault("condition_shift", []).append(slice(offset + half, offset + p.numel()))
        else:
            slices.setdefault(component, []).append(slice(offset, offset + p.numel()))
        offset += p.numel()
    previous = tester.blocks[-1] if len(tester.blocks) % 2 else None
    components = {}
    for name, indices in slices.items():
        motion = torch.cat([delta[s] for s in indices])
        item = {"energy": float(motion.square().sum())}
        if previous is not None:
            before = torch.cat([previous[s] for s in indices])
            denominator = float(before.norm() * motion.norm())
            item["pair_cosine"] = float(before @ motion) / denominator if denominator else None
            item["pair_dot"] = float(before @ motion)
        components[name] = item
    return {"block": tester.blocks_in_window + 1, "tested_b": tester.b,
            "pooled_energy": float(delta.square().sum()), "components": components}


def require_finite(label, tensors):
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError(f"Nonfinite {label}; update is not qualified")


def update(loop):
    """Caller-owned paired game in the public E22 hook order."""
    p = loop.policy
    indices = torch.randint(len(loop.fit_context), (p.recipe.batch_size,), generator=loop.batch_rng)
    context, target = loop.fit_context[indices], loop.fit_targets[indices]
    # These independent named draws align the arms even if policy noise changes.
    gaussian_d = torch.randn(target.shape, generator=loop.paired_noise_rng)
    gaussian_g = torch.randn(target.shape, generator=loop.paired_noise_rng)
    noise = p.begin_step(target, routed=RoutedBatch(context, target, loop.guard_context, loop.guard_targets))
    proposed = group_rates(p)
    apply_generator_bypass(loop)
    applied = group_rates(p)
    loss = p.recipe.make_loss()
    p.G.eval(); p.encoder.eval(); p.router.eval(); p.D.train()
    with torch.no_grad():
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * gaussian_d
        fake = real + prediction - target
    p.observe_critic_pair(real, fake)
    pure_d = loss.d_loss(p.D(real), p.D(fake))
    penalty = p.penalty(p.D, real, fake)
    loss_d = pure_d + penalty
    require_finite("critic loss", (loss_d, pure_d, penalty))
    p.opt_d.zero_grad(set_to_none=True)
    p.before_critic_backward(); loss_d.backward()
    require_finite("critic gradient", (v.grad for v in p.D.parameters() if v.grad is not None))
    critic_gradient_energy = sum(float(v.grad.square().sum()) for v in p.D.parameters()
                                 if v.requires_grad and v.grad is not None)
    before_d = parameter_snapshots(p.opt_d)
    p.opt_d.step(); p.after_critic_step()
    actual_motion = displacement_energy(p.opt_d, p.roles[1], before_d)
    p.D.eval(); p.G.train(); p.encoder.train(); p.router.train()
    flags = [parameter.requires_grad for parameter in p.D.parameters()]
    try:
        p.D.requires_grad_(False)
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * gaussian_g
        with torch.no_grad():
            real_logits = p.D(real.detach())
        loss_g = loss.g_loss(p.D(real + prediction - target), real_logits)
        if loop.profile in ANTITHETIC_PROFILES:
            # Same Gaussian marginal and same pure RpGAN loss; only the G
            # estimator's coupling changes. The D update and all RNG draws stay.
            with torch.no_grad():
                negative_real_logits = p.D(-real.detach())
            negative_loss = loss.g_loss(p.D(-real + prediction - target), negative_real_logits)
            loss_g = .5 * (loss_g + negative_loss)
        require_finite("generator loss", (loss_g,))
        p.opt_g.zero_grad(set_to_none=True)
        p.before_generator_backward(); loss_g.backward()
        require_finite("generator gradient", (v.grad for group in p.opt_g.param_groups
                                                for v in group["params"] if v.grad is not None))
        dense = int(p.table.grad.norm(dim=1).gt(0).sum())
        grad_energy = {role: sum(float(parameter.grad.square().sum()) for parameter in model.parameters()
                                if parameter.requires_grad and parameter.grad is not None)
                       for role, model in p._training_modules().items()}
        grad_energy["table"] = float(p.table.grad.square().sum())
        grad_energy["noise"] = float(p.log_output_sigma.grad.square().sum())
        grad_energy["critic"] = critic_gradient_energy
        p.after_generator_backward(loss_gan=loss_g.detach(), loss_critic=pure_d.detach())
        before_g = parameter_snapshots(p.opt_g)
        p.opt_g.step()
        block = generator_block_evidence(p, applied["generator"]["applied_ratio"])
        p.after_generator_step()
        actual_motion.update(displacement_energy(p.opt_g, p.roles[0], before_g))
    finally:
        for parameter, flag in zip(p.D.parameters(), flags):
            parameter.requires_grad_(flag)
    event = p.finish_step()
    return {"step": p.completed_steps, "loss_d": float(loss_d.detach()),
            "loss_g": float(loss_g.detach()), "penalty": float(penalty.detach()),
            "dense_gradient_rows": dense, "output_sigma": p.output_sigma(),
            "gradient_energy": grad_energy, "quadratic_head": p.D.quadratic.weight.detach().flatten().tolist(),
            "displacement_energy": actual_motion,
            "proposed_rates": proposed, "applied_rates": applied,
            "generator_block": block, "row_move": event,
            "settle": p.lr_settle.diagnostics()}


@torch.no_grad()
def evaluate(loop):
    """Untouched third-context clean MSE; no training or policy RNG draw."""
    p = loop.policy
    models = p._training_modules()
    query = models["encoder"](loop.test_context)
    weights = (models["router"].logits(query, p.table) + p.router.log_mass).softmax(-1)
    live = models["generator"](loop.test_context, weights @ p.table)
    served = p.served_model()
    prediction = served.routed_forward(loop.test_context)
    return {"initial_mse": loop.initial_mse,
            "live_mse": float((live - loop.test_targets).square().mean()),
            "served_mse": float((prediction - loop.test_targets).square().mean()),
            "served_source": served.source}


def checkpoint(loop):
    return {"policy": loop.policy.state_dict(), "batch_rng": loop.batch_rng.get_state(),
            "paired_noise_rng": loop.paired_noise_rng.get_state(),
            "profile": loop.profile, "initial_mse": loop.initial_mse, "metadata": loop.metadata}


def restore(loop, state):
    if state["profile"] != loop.profile or state["metadata"] != loop.metadata:
        raise ValueError("checkpoint profile/fixture contract differs")
    loop.policy.load_state_dict(state["policy"])
    loop.batch_rng.set_state(state["batch_rng"].cpu())
    loop.paired_noise_rng.set_state(state["paired_noise_rng"].cpu())
    loop.initial_mse = state["initial_mse"]


def json_safe(value):
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--max-seconds", type=float, default=900.)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--source-extent", type=float, default=.4)
    parser.add_argument("--target-gain", type=float, default=.15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="append", choices=PROFILES + EXTRA_PROFILES)
    parser.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()
    if (args.steps < 1 or args.max_seconds <= 0 or args.width < 4 or args.blocks < 1 or args.report_every < 1
            or not math.isfinite(args.source_extent) or args.source_extent <= 0
            or not math.isfinite(args.target_gain) or args.target_gain <= 0):
        parser.error("steps, seconds, width, blocks and report cadence must be positive (width>=4)")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is nonempty; choose a new artifact directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    (args.output / "source.py").write_bytes(source)
    package_root = Path(__file__).resolve().parents[2] / "particlegan"
    package_hashes = {str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(package_root.rglob("*.py"))}
    package_digest = hashlib.sha256(json.dumps(package_hashes, sort_keys=True).encode()).hexdigest()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_root.parent, text=True).strip()
    protocol = {"schema_version": 1, "source_sha256": hashlib.sha256(source).hexdigest(),
                "package_git_sha": git_sha, "package_python_sha256": package_digest,
                "package_files": package_hashes, "profiles": args.profile or list(PROFILES),
                "width": args.width, "blocks": args.blocks, "named_seeds": NAMED_SEEDS,
                "source_extent": args.source_extent, "target_gain": args.target_gain,
                "external_update_cap": args.steps, "external_seconds_cap": args.max_seconds,
                "require_external_timeout": "Launch under timeout 900s to bound a stalled update.",
                "scope": "standalone diagnostic; no Forge qualification or default adoption"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    profiles = args.profile or PROFILES
    started = time.monotonic()
    summary = {"scope": "standalone diagnostic, no Forge/default qualification", "profiles": {},
               "external_update_cap": args.steps, "external_seconds_cap": args.max_seconds,
               "python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}
    summary["source_sha256"] = protocol["source_sha256"]
    summary["package_git_sha"] = git_sha
    summary["package_python_sha256"] = package_digest
    for profile in profiles:
        loop = make_loop(profile, width=args.width, blocks=args.blocks,
                         source_extent=args.source_extent, target_gain=args.target_gain)
        (args.output / f"{profile}-metadata.json").write_text(json.dumps(json_safe(loop.metadata), indent=2) + "\n")
        profile_start = time.monotonic()
        trace_path = args.output / f"{profile}.jsonl"
        evaluations = [{"step": 0, **evaluate(loop)}]
        min_ratio, cuts, low_steps = 1., 0, 0
        previous_scale = 1.
        with trace_path.open("w", buffering=1) as trace:
            trace.write(json.dumps({"evaluation": evaluations[0]}) + "\n")
            for _ in range(args.steps):
                if time.monotonic() - started >= args.max_seconds:
                    break
                row = update(loop)
                ratio = row["applied_rates"]["generator"]["applied_ratio"]
                min_ratio = min(min_ratio, ratio)
                low_steps += ratio < 1.
                scale = loop.policy.lr_settle.testers[0][0].s
                cuts += scale < previous_scale
                previous_scale = scale
                if row["step"] % args.report_every == 0 or row["step"] == args.steps:
                    evaluation = {"step": row["step"], **evaluate(loop)}
                    evaluations.append(evaluation)
                    row["evaluation"] = evaluation
                    print(json.dumps(json_safe({"profile": profile, **evaluation,
                        "g_applied_ratio": ratio, "g_native_scale": scale,
                        "g_cuts": cuts, "sigma": row["output_sigma"],
                        "elapsed_seconds": time.monotonic() - started})), flush=True)
                trace.write(json.dumps(json_safe(row), separators=(",", ":")) + "\n")
        final = evaluate(loop)
        if evaluations[-1]["step"] != loop.policy.completed_steps:
            evaluations.append({"step": loop.policy.completed_steps, **final})
        torch.save(checkpoint(loop), args.output / f"{profile}-final.pt")
        summary["profiles"][profile] = {**final, "completed_steps": loop.policy.completed_steps,
            "elapsed_seconds": time.monotonic() - profile_start, "generator_cut_events": cuts,
            "generator_min_applied_ratio": min_ratio, "generator_damped_updates": low_steps,
            "settle": loop.policy.lr_settle.diagnostics(), "evaluations": evaluations,
            "rows": loop.policy.routed_control.diagnostics()}
        (args.output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")
        print(json.dumps(json_safe({"profile_complete": profile, **final,
              "completed_steps": loop.policy.completed_steps, "generator_cut_events": cuts,
              "generator_min_applied_ratio": min_ratio, "generator_damped_updates": low_steps,
              "elapsed_seconds": time.monotonic() - profile_start})), flush=True)
        if time.monotonic() - started >= args.max_seconds:
            break
    summary["total_elapsed_seconds"] = time.monotonic() - started
    (args.output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
