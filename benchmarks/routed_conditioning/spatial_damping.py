"""Tiny spatial recipient handoff with public E22 and pure paired-error RpGAN.

    timeout 900s python -u -m benchmarks.routed_conditioning.spatial_damping \
        --steps 1200 --output /tmp/spatial-routed-damping

The public fixture retains the native architecture's frozen BF16 prefix/head,
FP32 projected nonlinear residual FiLM map, 128x4 dense bank, source/time query,
small constructor gains, and a learned local plus raw-energy critic. No outer
source identity reaches the recipient. All policy controls remain active.
Clean held-out MSE is a report only; every training loss is the native GAN game.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from benchmarks.routed_conditioning import film_damping as common


CODE_PRESERVING_PROFILE = "shift_time_zero_native"
PROFILES = common.PROFILES + common.EXTRA_PROFILES + (CODE_PRESERVING_PROFILE,)


def time_features(context):
    u = context[:, 2, 0, 0]
    return torch.stack((u, (math.pi * u).sin(), (math.pi * u).cos()), 1)


class FrozenSpatialHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Conv2d(2, 3, 1, bias=False).bfloat16()
        self.suffix = nn.Conv2d(3, 2, 1, bias=False).bfloat16()
        with torch.no_grad():
            self.prefix.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [.5, .5]])[:, :, None, None])
            self.suffix.weight.copy_(torch.tensor([[.8, .1, .2], [-.1, .8, .2]])[:, :, None, None])
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, source):
        return self.prefix(source.bfloat16()).float()

    def decode(self, features):
        # Preserve the actual BF16 frozen head and its input-gradient boundary.
        return self.suffix(features.bfloat16()).float()


class SpatialResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.first = nn.Conv2d(width, width, 3, padding=1)
        self.second = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, value):
        return value + .1 * self.second(F.silu(self.first(F.silu(value))))


class SpatialFiLMGenerator(nn.Module):
    def __init__(self, width=4, blocks=1):
        super().__init__()
        self.host = FrozenSpatialHost()
        self.project = nn.Conv2d(3, width, 1)
        self.input = nn.Conv2d(width, width, 3, padding=1)
        self.condition = nn.Linear(4 + 3, 2 * width)
        self.blocks = nn.Sequential(*(SpatialResidual(width) for _ in range(blocks)))
        self.output = nn.Conv2d(width, 3, 3, padding=1)
        self.skip = nn.Conv2d(width, 3, 1)
        with torch.no_grad():
            for layer in (self.condition, self.output, self.skip):
                layer.bias.zero_()
                layer.weight.mul_(.1)

    def forward(self, context, code):
        source = self.project(self.host.encode(context[:, :2]))
        condition = torch.cat((F.layer_norm(code, (4,), eps=1e-3), time_features(context)), 1)
        gain, shift = (.25 * self.condition(condition).tanh()).chunk(2, 1)
        hidden = self.input(source) * (1 + gain[:, :, None, None]) + shift[:, :, None, None]
        recipient = self.skip(source) + self.output(F.silu(self.blocks(hidden)))
        return self.host.decode(recipient)


init.register(SpatialFiLMGenerator, {}, finalize=common.constructor_gains)


class SpatialEncoder(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(2, width, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(width, width, 3, stride=2, padding=1), nn.SiLU())
        self.query = nn.Linear(width * 4 * 4 + 3, 4)

    def forward(self, context):
        features = self.features(context[:, :2]).flatten(1)
        return F.layer_norm(self.query(torch.cat((features, time_features(context)), 1)), (4,), eps=1e-3)


class SpatialErrorCritic(nn.Module):
    """Learned local/global maps plus neutral, free-sign channel-energy curvature."""

    def __init__(self, width=4):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(2, width, 3, padding=1), nn.LeakyReLU(.2),
                                   nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(.2))
        self.local_score = nn.Conv2d(width, 1, 1)
        self.global_features = nn.Sequential(nn.Linear(128, 4 * width), nn.LeakyReLU(.2),
                                            nn.Linear(4 * width, width), nn.LeakyReLU(.2))
        self.global_score = nn.Linear(width, 1)
        self.quadratic = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.quadratic.weight.zero_()

    @staticmethod
    def energy(error):
        return error.square().mean((2, 3)) * math.sqrt(8 * 8 / 2)

    def features(self, error):
        return torch.cat((self.global_features(error.flatten(1)), self.local(error).mean((2, 3)),
                          self.energy(error)), 1)

    def forward(self, error):
        global_score = self.global_score(self.global_features(error.flatten(1)))
        local_score = self.local_score(self.local(error)).mean((2, 3))
        return (.01 * global_score + local_score + self.quadratic(self.energy(error))) / math.sqrt(3.)


@torch.no_grad()
def contexts_and_targets():
    # Fixed named data/time streams; this task does not search seeds.
    source_rng = torch.Generator().manual_seed(44)
    time_rng = torch.Generator().manual_seed(45)
    pools = []
    for count in (245, 64, 256):
        source = .2 * torch.randn(count, 2, 8, 8, generator=source_rng)
        u = .02 + .93 * torch.rand(count, generator=time_rng)
        context = torch.cat((source, u[:, None, None, None].expand(-1, 1, 8, 8)), 1)
        edit = torch.stack((.008 * (2 * u - 1), .006 * (math.pi * u).sin()), 1)[:, :, None, None]
        target = .15 * source.roll(1, dims=1) + edit
        pools.extend((context, target))
    return pools


def make_loop(profile="original_native", *, width=4, blocks=1):
    if profile not in PROFILES:
        raise ValueError("unknown diagnostic profile")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(common.NAMED_SEEDS["constructor"])
        G, E, D = SpatialFiLMGenerator(width, blocks), SpatialEncoder(width), SpatialErrorCritic(width)
        R, prior = common.DenseCosineRouter(128), ParticlePrior(128, 4)
    for module, role in ((G, "generator"), (E, "encoder"), (D, "critic"), (R, "router")):
        init.deterministic_orthogonal_(module, seed=common.NAMED_SEEDS[role + "_init"])
    init.deterministic_orthogonal_(prior)
    if profile != "original_native":
        with torch.no_grad():
            # Additive code features are the direct route to a conditioned
            # hidden state even when the source activation is zero. The new
            # diagnostic neutralizes time features while retaining that path.
            columns = slice(4, None) if profile == CODE_PRESERVING_PROFILE else slice(None)
            G.condition.weight[width:, columns].zero_()
            G.condition.bias[width:].zero_()
    fit, fit_target, guard, guard_target, test, test_target = contexts_and_targets()
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=16,
                        lr=.000204, d_lr_mult=1.5, prior_lr_mult=10., output_noise_std=1.3,
                        betas=(0., .999), birth_death_backend="auto", reopen_guard="settled")
    opt_g = recipe.make_generator_optimizer([
        {"params": [p for p in G.parameters() if p.requires_grad]}, {"params": list(E.parameters())},
        {"params": [prior.z], "lr": recipe.lr * recipe.prior_lr_mult},
    ], latent_table=prior.z, foreach=False)
    opt_d = recipe.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    rows = RoutedRows(model_forward=common.routed_forward, features=common.paired_features,
                      sites=("global",), probe_interval=16, probe_budget=8,
                      reservoir_size=64, min_observations=8)
    p = E22Policy(recipe, G, D, prior=prior, encoder=E, router=R,
                  generator_optimizer=opt_g, critic_optimizer=opt_d,
                  roles=[["generator", "encoder", "table"], ["critic"]], routed_rows=rows,
                  seed=common.NAMED_SEEDS["policy"])
    p.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
    initial = float((p.served_model().routed_forward(test) - test_target).square().mean())
    with torch.enable_grad():
        context = test[:2]
        code = ((R.logits(E(context), p.table) + R.log_mass).softmax(-1) @ p.table).detach().requires_grad_(True)
        prediction = G(context, code).flatten(1)
        jacobian_square = sum(float(torch.autograd.grad(prediction[:, coordinate].sum(), code,
                                                        retain_graph=True)[0].square().sum())
                              for coordinate in range(prediction.shape[1])) / len(context)
    with torch.no_grad():
        precondition = G.condition(torch.cat((F.layer_norm(code.detach(), (4,), eps=1e-3), time_features(context)), 1))
        stats = {name: {"pre_tanh_max_abs": float(value.abs().max()),
                        "mean_tanh_derivative": float((1 - value.tanh().square()).mean()),
                        "saturated_fraction_derivative_lt_05": float((1 - value.tanh().square()).lt(.05).float().mean())}
                 for name, value in zip(("gain", "shift"), precondition.chunk(2, 1))}
    metadata = {"profile": profile, "architecture": "spatial_2x8x8_frozen3_residual_film",
                "width": width, "blocks": blocks, "source_std": .2, "recipient_scale": .15,
                "frozen_boundary_dtype": "torch.bfloat16", "penalty_dimension": 128,
                "penalty_units": "native_whole_context", "recipe": recipe.to_dict(), "routing": rows.to_dict(),
                "initial_hashes": {role: common.tensor_hash(model.state_dict())
                                   for role, model in p._training_modules().items()},
                "data_hashes": common.tensor_hash(dict(fit=fit, fit_targets=fit_target, guard=guard,
                    guard_targets=guard_target, test=test, test_targets=test_target)),
                "initial_code_jacobian_frobenius": math.sqrt(jacobian_square), "initial_condition_stats": stats,
                "named_seeds": {**common.NAMED_SEEDS, "source_data": 44, "source_time": 45},
                "prior_kind": "particle_cloud", "prior_sigma": 0.,
                "prior_exception": "Dense coupled key/value bank is the mechanism under diagnosis.",
                "training_objective": "pure paired-error RpGAN plus native KA2 critic penalty",
                "evaluation_sampling": "clean held-out live and public served functions",
                "generator_noise_coupling": "antithetic" if "antithetic" in profile else "single Gaussian",
                "scope": "standalone convergence/variance diagnostic; no Forge/default qualification"}
    metadata["additive_initialization"] = (
        "zero time columns and bias; retain code columns" if profile == CODE_PRESERVING_PROFILE
        else "original public initialization" if profile == "original_native"
        else "zero all additive columns and bias")
    metadata["initial_hashes"]["table"] = common.tensor_hash({"z": p.table})
    return common.Loop(p, profile, fit, fit_target, guard, guard_target, test, test_target,
                       torch.Generator().manual_seed(common.NAMED_SEEDS["batch"]),
                       torch.Generator().manual_seed(common.NAMED_SEEDS["paired_noise"]), initial, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--max-seconds", type=float, default=900.)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.width, args.blocks, args.report_every, args.max_seconds) <= 0:
        parser.error("budgets, widths, blocks and reporting cadence must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is nonempty; choose a new artifact directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = (Path(__file__), Path(common.__file__))
    sources = {}
    for path in source_paths:
        data = path.read_bytes()
        (args.output / path.name).write_bytes(data)
        sources[path.name] = hashlib.sha256(data).hexdigest()
    root = Path(__file__).resolve().parents[2]
    package_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted((root / "particlegan").rglob("*.py"))}
    package_digest = hashlib.sha256(json.dumps(package_hashes, sort_keys=True).encode()).hexdigest()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    profiles = args.profile or [*common.PROFILES, "shift_zero_antithetic"]
    protocol = {"source_sha256": sources, "package_git_sha": git_sha,
                "package_python_sha256": package_digest, "package_files": package_hashes,
                "profiles": profiles, "external_update_cap": args.steps,
                "external_seconds_cap": args.max_seconds, "width": args.width, "blocks": args.blocks,
                "require_external_timeout": "Launch under timeout 900s to bound a stalled update."}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = {"protocol": protocol, "python": platform.python_version(), "torch": torch.__version__,
               "device": "cpu", "profiles": {}, "scope": "standalone diagnostic, no Forge/default claim"}
    start = time.monotonic()
    for profile in profiles:
        loop = make_loop(profile, width=args.width, blocks=args.blocks)
        (args.output / f"{profile}-metadata.json").write_text(json.dumps(common.json_safe(loop.metadata), indent=2) + "\n")
        profile_start = time.monotonic()
        evaluations = [{"step": 0, **common.evaluate(loop)}]
        common.require_finite("initial clean error", (torch.tensor(evaluations[0]["live_mse"]),))
        cuts, low_steps, minimum, previous_scale = 0, 0, 1., 1.
        with (args.output / f"{profile}.jsonl").open("w", buffering=1) as trace:
            trace.write(json.dumps({"evaluation": evaluations[0]}) + "\n")
            for _ in range(args.steps):
                if time.monotonic() - start >= args.max_seconds:
                    break
                row = common.update(loop)
                ratio = row["applied_rates"]["generator"]["applied_ratio"]
                minimum, low_steps = min(minimum, ratio), low_steps + int(ratio < 1.)
                scale = loop.policy.lr_settle.testers[0][0].s
                cuts += scale < previous_scale
                previous_scale = scale
                if row["step"] % args.report_every == 0 or row["step"] == args.steps:
                    result = {"step": row["step"], **common.evaluate(loop)}
                    common.require_finite("clean error", (torch.tensor(result["live_mse"]),
                                                           torch.tensor(result["served_mse"])))
                    if max(result["live_mse"], result["served_mse"]) > 4 * loop.initial_mse:
                        raise FloatingPointError("Clean error exceeds 4x starting error; run is unqualified")
                    evaluations.append(result)
                    row["evaluation"] = result
                    print(json.dumps(common.json_safe({"profile": profile, **result,
                        "g_applied_ratio": ratio, "g_native_scale": scale, "g_cuts": cuts,
                        "sigma": row["output_sigma"], "elapsed_seconds": time.monotonic() - start})), flush=True)
                trace.write(json.dumps(common.json_safe(row), separators=(",", ":")) + "\n")
        final = common.evaluate(loop)
        if evaluations[-1]["step"] != loop.policy.completed_steps:
            evaluations.append({"step": loop.policy.completed_steps, **final})
        torch.save(common.checkpoint(loop), args.output / f"{profile}-final.pt")
        summary["profiles"][profile] = {**final, "completed_steps": loop.policy.completed_steps,
            "generator_cut_events": cuts, "generator_min_applied_ratio": minimum,
            "generator_damped_updates": low_steps, "evaluations": evaluations,
            "elapsed_seconds": time.monotonic() - profile_start,
            "settle": loop.policy.lr_settle.diagnostics(), "rows": loop.policy.routed_control.diagnostics()}
        summary["total_elapsed_seconds"] = time.monotonic() - start
        (args.output / "summary.json").write_text(json.dumps(common.json_safe(summary), indent=2) + "\n")
        print(json.dumps(common.json_safe({"profile_complete": profile, **final,
            "completed_steps": loop.policy.completed_steps, "generator_cut_events": cuts,
            "generator_min_applied_ratio": minimum, "generator_damped_updates": low_steps,
            "elapsed_seconds": time.monotonic() - profile_start})), flush=True)
        if time.monotonic() - start >= args.max_seconds:
            break


if __name__ == "__main__":
    main()
