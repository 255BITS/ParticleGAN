"""Small caller-owned paired-error E22 diagnosis, not a Forge qualification.

Run from the repository root, with bulk output outside tracked source:
  PYTHONPATH=. python examples/e22_paired_residual_toy.py --steps 800 \
      --output /tmp/e22-paired-residual-toy

The exactly representable affine edit has balanced source coordinates. A frozen
BF16 base, FP32 affine/FiLM adapter, encoder and cosine-routed 128x4 bank produce
[B,2,1,32] outputs. This is 64 error coordinates, rather than Nova/Qwen's 65,536.
The critic reduces the native branches to a learned odd channel mean plus a
zero-initialized free-sign channel energy head. It is not the image critic.
All optimization uses RpGAN/KA2; clean error is reporting only. No seed sweep,
image-quality claim, independent-particle winner claim or Forge gate is implied.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from particlegan import E22Policy, ParticlePrior, RoutedBatch, RoutedRows, get_recipe, init


PROFILES = ("current", "even_critic", "d_antithetic")
OUTPUT_SHAPE = (2, 1, 32)
G_FACTOR = .5
ODD_GAIN = .0625
FORMAT = "e22_paired_residual_toy_v1"


def digest(value):
    """Stable tensor/content hash, also usable for exact checkpoint comparison."""
    result = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            result.update(str((item.dtype, item.shape)).encode())
            result.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(str(key).encode()); add(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item: add(child)
        elif isinstance(item, bytes):
            result.update(item)
        else:
            result.update(repr(item).encode())
    add(value)
    return result.hexdigest()


def grid(points, times):
    x, y, t = torch.meshgrid(points, points, times, indexing="ij")
    return torch.stack((x.flatten(), y.flatten(), t.flatten()), -1)


def centered(context):
    return context - context.new_tensor([0., 0., .5])


def base_output(context):
    weight = context.new_tensor([[1., 0., .05], [0., 1., -.07]]).bfloat16()
    values = (context.bfloat16() @ weight.T).float()
    return values[:, :, None, None].expand(-1, *OUTPUT_SHAPE)


def raw_targets(context):
    # A zero-mean affine edit; the adapter's affine path can represent it exactly.
    edit = centered(context) @ context.new_tensor([[.28, -.17], [.13, .25], [.11, -.09]])
    return base_output(context) + edit[:, :, None, None]


class ResidualAdapter(nn.Module):
    def __init__(self, target_mean, target_std):
        super().__init__()
        self.host = nn.Linear(3, 2, bias=False).bfloat16()
        with torch.no_grad():
            self.host.weight.copy_(torch.tensor([[1., 0., .05], [0., 1., -.07]]))
        self.host.requires_grad_(False)
        self.affine = nn.Linear(3, 2)
        self.condition = nn.Linear(7, 4)
        self.register_buffer("target_mean", target_mean.clone())
        self.register_buffer("target_std", target_std.clone())

    def normalize(self, target):
        return (target - self.target_mean) / self.target_std

    def forward(self, context, code):
        t = context[:, 2:3]
        time_features = torch.cat((t, torch.sin(math.pi*t), torch.cos(math.pi*t)), -1)
        condition = torch.cat((F.layer_norm(code, (4,), eps=1e-3), time_features), -1)
        gain, bias = (.25*self.condition(condition).tanh()).chunk(2, -1)
        residual = self.affine(centered(context))*(1+gain)+bias
        base = self.host(context.bfloat16()).float()
        return self.normalize((base+residual)[:, :, None, None].expand(-1, *OUTPUT_SHAPE))


def adapter_gains(module):
    for layer in (module.affine, module.condition):
        layer.weight.mul_(.1)
        layer.bias.zero_()


init.register(ResidualAdapter, finalize=adapter_gains)


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(3, 4)

    def forward(self, context):
        return self.query(centered(context))


class MassRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("log_mass", torch.zeros(128))

    def logits(self, query, table):
        return 2*(F.normalize(query, dim=-1, eps=1e-4)
                  @ F.normalize(table, dim=-1, eps=1e-4).T)


class FreeEnergyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 2))

    def forward(self, values):
        return F.linear(values, self.weight).squeeze(-1)


init.register(FreeEnergyHead, {"weight": init.KEEP})


class ResidualCritic(nn.Module):
    """Learned odd/even statistics of the noisy error; no source or target access."""
    def __init__(self, *, even_only=False):
        super().__init__()
        self.odd = nn.Linear(2, 1, bias=False)
        self.energy = FreeEnergyHead()
        self.even_only = bool(even_only)  # serialized by the caller's profile

    def features(self, error):
        means = error.mean((2, 3))
        energies = error.square().mean((2, 3))*math.sqrt(32/2)
        odd = means*self.odd.weight*(0. if self.even_only else ODD_GAIN)
        even = energies*self.energy.weight
        return torch.cat((odd, even), -1)

    def forward(self, error):
        return self.features(error).sum(-1)/math.sqrt(3)


def model_forward(models, context, candidate, routing):
    query = models["encoder"](context)
    code = routing.mix("global", models["router"].logits(query, candidate.table))
    return models["generator"](context, code)


def paired_features(models, context, samples, targets):
    return models["critic"].features(samples-targets)


@dataclass
class ToyLoop:
    policy: E22Policy
    profile: str
    fit_context: torch.Tensor
    fit_targets: torch.Tensor
    guard_context: torch.Tensor
    guard_targets: torch.Tensor
    report_context: torch.Tensor
    report_targets: torch.Tensor
    data_rng: torch.Generator
    paired_rng: torch.Generator
    draws: bytes
    initial_weights_sha256: str
    frozen_host_sha256: str
    initial_report_rmse: float


def make_loop(profile="current"):
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    fit = grid(torch.linspace(-.8, .8, 5), torch.tensor([.125, .375, .625, .875]))
    guard = grid(torch.tensor([-.7, -.23, .23, .7]), torch.tensor([.15, .35, .65, .85]))
    report = grid(torch.linspace(-.75, .75, 6), torch.tensor([.1, .3, .7, .9]))
    raw_fit = raw_targets(fit)
    mean = raw_fit.mean((0, 2, 3), keepdim=True)
    std = raw_fit.std((0, 2, 3), keepdim=True).clamp_min(.04)
    # Constructors are isolated; public orthogonal/R2 init reads no RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        G, E, R = ResidualAdapter(mean, std), ContextEncoder(), MassRouter()
        D, prior = ResidualCritic(even_only=profile == "even_critic"), ParticlePrior(128, 4)
    for module, seed in ((G, 0), (D, 1), (E, 2), (R, 3)):
        init.deterministic_orthogonal_(module, seed=seed)
    init.deterministic_orthogonal_(prior)
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=16,
        lr=.000204, d_lr_mult=1.5, prior_lr_mult=10., output_noise_std=1.3,
        betas=(0., .999), birth_death_backend="auto", reopen_guard="settled")
    groups = [{"params": [p for p in G.parameters() if p.requires_grad]},
              {"params": list(E.parameters())},
              {"params": [prior.z], "lr": recipe.lr*recipe.prior_lr_mult}]
    opt_g = recipe.make_generator_optimizer(groups, latent_table=prior.z, foreach=False)
    opt_d = recipe.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    rows = RoutedRows(model_forward=model_forward, features=paired_features,
        sites=("global",), probe_interval=100, reservoir_size=64)
    policy = E22Policy(recipe, G, D, prior=prior, encoder=E, router=R,
        generator_optimizer=opt_g, critic_optimizer=opt_d,
        roles=[["generator", "encoder", "table"], ["critic"]], routed_rows=rows, seed=21)
    policy.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
    if (policy.penalty.regularizer.record is not opt_d.record
            or policy.controller is not opt_d.continuous_controller
            or policy.penalty.regularizer.continuous_controller is not policy.controller):
        raise AssertionError("policy/KA2 controller ownership differs")
    weights = dict(generator=G.state_dict(), encoder=E.state_dict(), router=R.state_dict(),
                   critic=D.state_dict(), table=prior.z)
    initial_report = policy.served_model().routed_forward(report)
    initial_rmse = float((initial_report-G.normalize(raw_targets(report))).square().mean().sqrt())
    return ToyLoop(policy, profile, fit, G.normalize(raw_fit).detach(),
        guard, G.normalize(raw_targets(guard)).detach(),
        report, G.normalize(raw_targets(report)).detach(),
        torch.Generator().manual_seed(42), torch.Generator().manual_seed(43),
        hashlib.sha256(b"").digest(), digest(weights), digest(G.host.state_dict()), initial_rmse)


def critic_adversarial(loop, prediction, target, sigma, epsilon):
    p, loss = loop.policy, loop.policy.recipe.make_loss()
    real = sigma*epsilon
    fake = real+prediction-target
    positive = loss.d_loss(p.D(real), p.D(fake))
    if loop.profile != "d_antithetic":
        return positive, real, fake
    negative = loss.d_loss(p.D(-real), p.D(-real+prediction-target))
    return (positive+negative)*.5, real, fake


def generator_adversarial(policy, prediction, target, sigma, epsilon):
    loss = policy.recipe.make_loss()
    real = sigma*epsilon
    with torch.no_grad():
        reference_positive = policy.D(real.detach())
        reference_negative = policy.D(-real.detach())
    positive = loss.g_loss(policy.D(real+prediction-target), reference_positive)
    negative = loss.g_loss(policy.D(-real+prediction-target), reference_negative)
    return (positive+negative)*.5, positive, negative


def update(loop):
    p = loop.policy
    indices = torch.randint(len(loop.fit_context), (p.recipe.batch_size,), generator=loop.data_rng)
    context, target = loop.fit_context[indices], loop.fit_targets[indices]
    noise = p.begin_step(target, routed=RoutedBatch(context, target, loop.guard_context, loop.guard_targets))
    # Public caller-applied overlay; intrinsic time sees actual/base, not tester.s.
    p.opt_g.param_groups[0]["lr"] *= G_FACTOR
    applied = [[group["lr"] for group in optimizer.param_groups] for optimizer in p.optimizers]
    role_params = dict(generator=[v for v in p.G.parameters() if v.requires_grad],
                       encoder=list(p.encoder.parameters()), table=[p.table])
    before = {role:[v.detach().clone() for v in values] for role, values in role_params.items()}
    p.G.eval(); p.encoder.eval(); p.router.eval(); p.D.train()
    with torch.no_grad():
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        epsilon_d = torch.randn(target.shape, generator=loop.paired_rng)
        real = noise.output_sigma*epsilon_d
        fake = real+prediction-target
    p.observe_critic_pair(real, fake)
    adversarial_d, primary_real, primary_fake = critic_adversarial(
        loop, prediction, target, noise.output_sigma.detach(), epsilon_d)
    # One unchanged KA2 call on the primary pair, even for D-antithetic.
    penalty = p.penalty(p.D, primary_real, primary_fake)
    p.opt_d.zero_grad(); p.before_critic_backward()
    (adversarial_d+penalty).backward(); p.opt_d.step(); p.after_critic_step()
    p.D.eval(); p.G.train(); p.encoder.train(); p.router.train()
    flags = [value.requires_grad for value in p.D.parameters()]
    try:
        p.D.requires_grad_(False)
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        epsilon_g = torch.randn(target.shape, generator=loop.paired_rng)
        loss_g, positive, negative = generator_adversarial(p, prediction, target, noise.output_sigma, epsilon_g)
        p.opt_g.zero_grad(); p.before_generator_backward(); loss_g.backward()
        energies = {role: sum(float(v.grad.detach().square().sum()) for v in module.parameters()
                            if v.requires_grad and v.grad is not None)
                    for role, module in (("generator", p.G), ("encoder", p.encoder))}
        dense = int(p.table.grad.norm(dim=-1).gt(0).sum())
        sigma_gradient = float(p.log_output_sigma.grad.detach())
        p.after_generator_backward(loss_gan=loss_g.detach(), loss_critic=adversarial_d.detach())
        p.opt_g.step(); p.after_generator_step()
        deltas = {role: math.sqrt(sum(float((value.detach()-old).square().sum())
            for value, old in zip(values, before[role]))) for role, values in role_params.items()}
    finally:
        for value, flag in zip(p.D.parameters(), flags): value.requires_grad_(flag)
    event = p.finish_step()
    loop.draws = hashlib.sha256(loop.draws+indices.numpy().tobytes()
        +epsilon_d.numpy().tobytes()+epsilon_g.numpy().tobytes()).digest()
    row = dict(step=p.completed_steps, g_gan=float(loss_g.detach()), d_gan=float(adversarial_d.detach()),
        penalty=float(penalty.detach()), feature_error=float((prediction.detach()-target).square().mean()),
        feature_error_convention="pre-G-update DV12-perturbed normalized prediction",
        g_positive=float(positive.detach()), g_negative=float(negative.detach()),
        sigma_gradient=sigma_gradient, output_sigma=p.output_sigma(),
        dense_gradient_rows=dense, gradient_energy=energies, applied_lrs=applied,
        g_applied_base_ratio=applied[0][0]/p.initial_lrs[0][0],
        ka2_phase=p.penalty.last_stats["phase"], ka2_calls=p.opt_d.record.calls,
        generator_step_delta_l2=deltas,
        adversarial_d_score_forwards=4 if loop.profile == "d_antithetic" else 2,
        move=event, caller_draw_chain_sha256=loop.draws.hex(),
        data_rng_sha256=digest(loop.data_rng.get_state()), paired_rng_sha256=digest(loop.paired_rng.get_state()))
    for key in ("g_gan", "d_gan", "penalty", "feature_error", "g_positive", "g_negative", "sigma_gradient", "output_sigma"):
        if not math.isfinite(row[key]): raise FloatingPointError(f"nonfinite {key}")
    for optimizer in p.optimizers:
        for group in optimizer.param_groups:
            for value in group["params"]:
                if (not bool(torch.isfinite(value).all()) or value.grad is None
                        or not bool(torch.isfinite(value.grad).all())):
                    raise FloatingPointError("nonfinite parameter or missing/nonfinite trainable gradient")
    if digest(p.G.host.state_dict()) != loop.frozen_host_sha256:
        raise AssertionError("frozen BF16 host changed")
    return row


def checkpoint(loop):
    return dict(format=FORMAT, profile=loop.profile, policy=loop.policy.state_dict(),
        data_rng=loop.data_rng.get_state(), paired_rng=loop.paired_rng.get_state(), draws=loop.draws,
        initial_weights_sha256=loop.initial_weights_sha256, frozen_host_sha256=loop.frozen_host_sha256,
        initial_report_rmse=loop.initial_report_rmse)


def restore(loop, state):
    if (state.get("format") != FORMAT or state.get("profile") != loop.profile
            or state.get("initial_weights_sha256") != loop.initial_weights_sha256
            or state.get("frozen_host_sha256") != loop.frozen_host_sha256
            or state.get("initial_report_rmse") != loop.initial_report_rmse):
        raise ValueError("toy profile, model initialization or frozen host differs")
    loop.policy.load_state_dict(state["policy"])
    loop.data_rng.set_state(state["data_rng"])
    loop.paired_rng.set_state(state["paired_rng"])
    loop.draws = state["draws"]


def error_metrics(prediction, target):
    error = prediction-target
    per_context = error.square().flatten(1).mean(1)
    return dict(normalized_rmse=float(per_context.mean().sqrt()), normalized_mse=float(per_context.mean()),
        bias=[float(v) for v in error.mean((0, 2, 3))],
        max_context_rmse=float(per_context.max().sqrt()))


@torch.no_grad()
def evaluate(loop):
    before = digest(checkpoint(loop))
    p = loop.policy
    served = p.served_model()
    models = dict(generator=p.G, encoder=p.encoder, router=p.router, critic=p.D, prior=p.prior)
    result = dict(step=p.completed_steps, served_source=served.source, pools={})
    for name in ("fit", "guard", "report"):
        context, target = getattr(loop, name+"_context"), getattr(loop, name+"_targets")
        candidate = p.routed_control.spec.candidate_for(models, p.table)
        live = p.routed_control.spec.forward(models, context, candidate)
        frozen = served.routed_forward(context)
        weights = (p.router.logits(p.encoder(context), p.table)+p.router.log_mass).softmax(-1)
        codes = weights@p.table
        entropy = -(weights*weights.clamp_min(1e-30).log()).sum(-1)/math.log(len(p.table))
        result["pools"][name] = dict(count=len(context), live=error_metrics(live, target),
            served=error_metrics(frozen, target), routing=dict(normalized_entropy=float(entropy.mean()),
                mean_max_weight=float(weights.max(-1).values.mean()),
                concentration=float(weights.square().sum(-1).mean()),
                mixed_code_rms=float(codes.square().mean().sqrt()),
                code_layernorm_variance=float(codes.var(-1, unbiased=False).mean())))
    raw_sigma = float(p.log_output_sigma.detach().exp())
    output_sigma = p.output_sigma()
    tester = p.lr_settle.testers[0][0]
    intrinsic = tester.diagnostics()["intrinsic_steps_to_next_decision"]
    actual_ratio = p.opt_g.param_groups[0]["lr"]/p.initial_lrs[0][0]
    result["diagnostics"] = dict(output_sigma=output_sigma, raw_sigma=raw_sigma,
        physical_floor_binds=raw_sigma <= output_sigma,
        energy_coefficients=p.D.energy.weight.flatten().tolist(),
        odd_score_coefficient=(p.D.odd.weight.flatten()*(0. if p.D.even_only else ODD_GAIN)/math.sqrt(3)).tolist(),
        zero_residual_G_odd_force_norm=float(p.D.odd.weight.norm())*
            (0. if p.D.even_only else ODD_GAIN)/(2*math.sqrt(3*32)),
        zero_residual_force_convention="per-context output L2; odd-only antithetic G derivative, no batch mean",
        generator_tester=tester.diagnostics(), all_testers=p.lr_settle.diagnostics(),
        generator_applied_base_ratio=actual_ratio,
        next_G_decision_wall_updates_if_unchanged=math.ceil(intrinsic/actual_ratio) if actual_ratio else None,
        routed=p.birth_death.diagnostics(), ka2_calls=p.opt_d.record.calls,
        ka2_phase=(p.penalty.last_stats.get("phase") or
                   ("unstarted" if not p.opt_d.record.calls else "a" if p.opt_d.record.calls < 800 else "blend")),
        actual_backend=p.served_snapshot().get("backend_selection", {}).get("actual_backend", "routed"))
    result["report_rmse_ratio"] = {name:result["pools"]["report"][name]["normalized_rmse"]/
        loop.initial_report_rmse for name in ("live", "served")}
    if digest(checkpoint(loop)) != before:
        raise AssertionError("ordinary clean reporting changed models, optimizers or RNG")
    result["all_owners_and_rng_unchanged"] = True
    return result


@contextmanager
def watchdog(seconds):
    def expired(signum, frame): raise TimeoutError(f"CPU update exceeded {seconds}s")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def package_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except (OSError, subprocess.CalledProcessError): return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def json_safe(value):
    """Preserve legitimate infinite t-statistics explicitly, never as JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, torch.Tensor): return json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict): return {key:json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)): return [json_safe(child) for child in value]
    return value


def budgeted(deadline, function, *args, update=False):
    remaining = deadline-time.perf_counter()
    if remaining <= 0: raise TimeoutError("declared300s CPU campaign budget exhausted")
    with watchdog(min(5., remaining) if update else remaining):
        return function(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800, help="external per-profile budget, at most1200")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1200 or len(set(args.profiles)) != len(args.profiles):
        parser.error("steps must be1..1200 and profiles must be distinct")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()): parser.error("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    deadline = started+300
    common_weights = common_callers = None
    protocol = dict(format=FORMAT, scope="standalone paired-policy diagnostic; no Forge qualification",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), package_sha=package_commit(),
        profiles=args.profiles, steps=args.steps, device="cpu", threads=1,
        wall_budget_seconds=300, update_watchdog_seconds=5, seed_sweep=False,
        output_shape=list(OUTPUT_SHAPE), penalty_dimension=64, native_reference_dimension=65536,
        loss="RpGAN only; no reconstruction/VIC loss", output_noise_std=1.3,
        critic="reduced odd channel-mean(.0625) and learned free-sign energy(1), divided by sqrt(3); no global branch",
        profile_deltas=dict(current="baseline antitheticG/odd+evenD",
            even_critic="only score/features parity projects out D odd branch; identical parameter groups/init",
            d_antithetic="only adversarialD mean +/-epsilon; primary-pair KA2/observation once, no extra draws"),
        structural_counterfactual_law="package clean learned-critic-feature fit/guard law; no sampler replay callback",
        reporting="clean ordinary live and state-selected served forward; disjoint report grid; error never optimized",
        prior="explicit tied particle cloud128x4; conditional routed exception, not a MoG Forge task",
        initialization="public orthogonal G0 D1 E2 R3 + R2 prior; declared G .1 finalizer; zero energy head KEEP",
        generator_applied_factor=G_FACTOR, caller_seeds=dict(data=42, paired=43), policy_seed=21,
        raw_caller_draws_matched=True, private_policy_stream_identity_claim=False,
        nonfinite_diagnostic_encoding="explicit NaN/+Infinity/-Infinity strings; trainable state/loss must stay finite",
        resolved={}, qualification="mechanism toy only; reduced dimension/critic and affine skip can bypass routing")
    budgeted(deadline, write_json, output/"protocol.json", protocol)
    results = {}
    profile = None
    loop = None
    arm = None
    with torch.autograd.set_multithreading_enabled(False):
        try:
            for profile in args.profiles:
                loop = None
                arm = None
                arm = output/profile; arm.mkdir()
                loop = budgeted(deadline, make_loop, profile)
                caller_hash = digest((loop.data_rng.get_state(), loop.paired_rng.get_state()))
                if common_weights is None:
                    common_weights, common_callers = loop.initial_weights_sha256, caller_hash
                if (loop.initial_weights_sha256 != common_weights or caller_hash != common_callers):
                    raise AssertionError("profiles have different common initialization or caller streams")
                protocol["resolved"][profile] = dict(recipe=loop.policy.recipe.to_dict(),
                    routing=loop.policy.routed_control.spec.to_dict(),
                    initial_weights_sha256=common_weights, initial_caller_rng_sha256=common_callers,
                    context_hashes={name:digest(getattr(loop, name+"_context")) for name in ("fit", "guard", "report")})
                budgeted(deadline, write_json, output/"protocol.json", protocol)
                budgeted(deadline, lambda:torch.save(checkpoint(loop), arm/"initial.pt"))
                evaluations = []
                last_row = None
                active = 0.
                with (arm/"updates.jsonl").open("w") as updates, (arm/"metrics.jsonl").open("w") as metrics:
                    for step in range(args.steps+1):
                        if time.perf_counter()-started >= 300:
                            raise TimeoutError("declared300s CPU campaign budget exhausted")
                        if step:
                            began = time.perf_counter()
                            last_row = budgeted(deadline, update, loop, update=True)
                            active += time.perf_counter()-began
                            budgeted(deadline, lambda:(updates.write(json.dumps(json_safe(last_row), allow_nan=False)+"\n"), updates.flush()))
                        if step in (0, 10, 25, 50) or step % 100 == 0 or step == args.steps:
                            report = budgeted(deadline, evaluate, loop); report["active_update_seconds"] = active
                            evaluations.append(report)
                            budgeted(deadline, lambda:(metrics.write(json.dumps(json_safe(report), allow_nan=False)+"\n"), metrics.flush()))
                            budgeted(deadline, lambda:torch.save(checkpoint(loop), arm/"recovery.pt"))
                            budgeted(deadline, lambda:print(json.dumps(json_safe(dict(profile=profile, **report)), allow_nan=False), flush=True))
                        budgeted(deadline, write_json, arm/"progress.json", dict(status="running", step=step, budget_steps=args.steps,
                            last_report_step=evaluations[-1]["step"], active_update_seconds=active))
                results[profile] = dict(step=loop.policy.completed_steps, initial=evaluations[0], final=evaluations[-1],
                    active_update_seconds=active, caller_draw_chain_sha256=loop.draws.hex(),
                    last_update=last_row, accepted_moves=loop.policy.birth_death.diagnostics()["counters"]["moves"],
                    first_recorded_half_initial_rmse_step=next((row["step"] for row in evaluations
                        if row["report_rmse_ratio"]["served"] <= .5), None))
                budgeted(deadline, write_json, arm/"progress.json", dict(status="completed", step=args.steps, budget_steps=args.steps,
                    last_report_step=args.steps, active_update_seconds=active))
            chains = {value["caller_draw_chain_sha256"] for value in results.values()}
            if len(chains) != 1: raise AssertionError("equal-budget raw caller draws did not match")
            budgeted(deadline, write_json, output/"result.json", dict(format=FORMAT, status="completed", results=results,
                common_initial_weights=True, raw_caller_draws_matched=True,
                private_policy_stream_identity_claim=False, wall_seconds=time.perf_counter()-started,
                ordinary_Forge_qualification=False, image_quality_claim=False))
        except Exception as error:
            failure = dict(status="failed", profile=profile,
                last_completed_step=loop.policy.completed_steps if loop is not None else 0,
                completed_profiles=list(results), exception=type(error).__name__, reason=str(error),
                interrupted_update_qualified=False, wall_seconds=time.perf_counter()-started)
            write_json(output/"failure.json", failure)
            if arm is not None and arm.is_dir(): write_json(arm/"progress.json", failure)
            raise


if __name__ == "__main__": main()
