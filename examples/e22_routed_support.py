"""A bank-dependent, source/time-conditioned routed E22 benchmark.

The two residual branches start neutral: their output factors are zero by
construction, while all other trainable layers and the bank use the public
deterministic orthogonal initializer. The targets contain two moving spatial
transitions. Compare frozen, movable and structurally controlled banks using
one initialization and matching batches and Gaussian base draws.

python -u examples/e22_routed_support.py --compare --steps 1200 --device cuda
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from particlegan import E22Policy, RoutedRows, get_recipe, init
from e22_routed_game import capture_game
import e22_routed_sites as pooled
from e22_routed_sites import (ContextEncoder, MODES, SiteLoop, SiteQueries,
                              TokenErrorCritic, checkpoint, context_grid,
                              diagnostics, evaluate, neutral, paired_features,
                              restore, synchronize, update, warmup)


class NeutralResidualHost(nn.Module):
    """Frozen BF16 host plus neutral, low-rank nonlinear residual branches."""

    def __init__(self, z_dim):
        super().__init__()
        self.first_host = nn.Linear(4, 4, bias=False).bfloat16()
        self.second_host = nn.Linear(4, 2, bias=False).bfloat16()
        with torch.no_grad():
            self.first_host.weight.copy_(torch.eye(4))
            self.second_host.weight.copy_(torch.tensor([[1., 0., .05, .02], [0., 1., -.07, -.01]]))
        self.first_host.requires_grad_(False)
        self.second_host.requires_grad_(False)
        self.first_input = nn.Linear(z_dim, 8)
        self.first_output = nn.Linear(8, 4, bias=False)
        self.second_input = nn.Linear(z_dim, 8)
        self.second_output = nn.Linear(8, 2, bias=False)
        nn.init.zeros_(self.first_output.weight)
        nn.init.zeros_(self.second_output.weight)

    def first(self, context, codes):
        return self.first_host(context.bfloat16()).float() + self.first_output(self.first_input(codes).tanh())

    def second(self, hidden, codes):
        return self.second_host(hidden.bfloat16()).float() + self.second_output(self.second_input(codes).tanh())


def model_forward(models, context, candidate, routing):
    generator, encoder, router = (models[key] for key in ("generator", "encoder", "router"))
    first_query = router.first_query(encoder(context))
    codes = routing.mix("first", first_query @ candidate.table.T / math.sqrt(candidate.table.shape[1]))
    hidden = generator.first(context, codes)
    second_query = router.second_query(hidden)
    codes = routing.mix("second", second_query @ candidate.table.T / math.sqrt(candidate.table.shape[1]))
    return generator.second(hidden, codes)


def targets(context):
    x, y, time, position = context.unbind(-1)
    edit = torch.stack((.18 + .12 * (5 * (position + .4 * x - .6 * time)).tanh(),
                        .16 + .10 * (5 * (position - .35 * y + .5 * time - .2)).tanh()), -1)
    return neutral(context) + edit


def make_loop(*, mode="full", device="cpu", tokens=128, particles=128,
              z_dim=4, batch_size=8, probe_interval=1, max_context_harm=0.):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    device = torch.device(device)
    fit = context_grid(torch.linspace(-.8, .8, 5), torch.tensor([.1, .35, .6, .85]), tokens).to(device)
    guard = context_grid(torch.tensor([-.72, -.24, .24, .72]), torch.tensor([.18, .48, .78]), tokens).to(device)
    test = context_grid(torch.linspace(-.75, .75, 6), torch.tensor([.13, .29, .45, .61, .77]), tokens).to(device)
    fit_target, guard_target, test_target = (targets(value) for value in (fit, guard, test))
    scale = (fit_target - neutral(fit)).std(dim=(0, 1)).clamp_min(.04)
    recipe = get_recipe("e22_routed", num_particles=particles, z_dim=z_dim, batch_size=batch_size,
                        output_noise_std=.125, row_evidence_gate=mode == "full",
                        particle_birth_death=mode == "full")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        generator, encoder = NeutralResidualHost(z_dim).to(device), ContextEncoder().to(device)
        router = SiteQueries(particles, z_dim, conformance=False).to(device)
        critic = TokenErrorCritic(scale).to(device)
        for model, role in ((generator, 0), (critic, 1), (encoder, 2), (router, 3)):
            init.deterministic_orthogonal_(model, seed=role)
        prior = init.deterministic_orthogonal_(recipe.make_prior()).to(device)
        table = prior.z.requires_grad_(mode != "frozen")
    groups = [{"params": [parameter for parameter in generator.parameters() if parameter.requires_grad]},
              {"params": list(encoder.parameters())}, {"params": list(router.parameters())}]
    roles = ["generator", "encoder", "router"]
    if table.requires_grad:
        groups.append({"params": [table], "lr": recipe.lr * recipe.prior_lr_mult})
        roles.append("table")
    opt_g = recipe.make_generator_optimizer(groups, latent_table=table if table.requires_grad else None,
                                             foreach=False)
    opt_d = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    rows = RoutedRows(model_forward=model_forward, sites=("first", "second"), features=paired_features,
                      probe_interval=probe_interval, max_context_harm=max_context_harm)
    policy = E22Policy(recipe, generator, critic, table=table, encoder=encoder, router=router,
                       generator_optimizer=opt_g, critic_optimizer=opt_d, roles=[roles, ["critic"]],
                       routed_rows=rows, seed=21)
    policy.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
    initial = float((policy.served_model().routed_forward(test) - test_target).square().mean().sqrt())
    config = dict(mode=mode, tokens=tokens, particles=particles, z_dim=z_dim, batch_size=batch_size,
                  initialization="api", penalty_units="token", probe_interval=probe_interval,
                  max_context_harm=max_context_harm, task="moving_transitions_v1")
    return SiteLoop(policy, fit, fit_target, guard, guard_target, test, test_target,
                    torch.Generator(device=device).manual_seed(42),
                    torch.Generator(device=device).manual_seed(43), initial, config)


def initial_weights(loop):
    state = {f"{role}.{name}": value.detach().cpu().clone()
             for role, model in loop.policy._training_modules().items() for name, value in model.state_dict().items()}
    state["table"] = loop.policy.table.detach().cpu().clone()
    state["log_output_sigma"] = loop.policy.log_output_sigma.detach().cpu().clone()
    return state


def game_capture(loop, context, targets, draws=4):
    """Fixed private Monte Carlo draws; these states never belong to training."""
    policy = loop.policy
    latent = torch.Generator(device=policy.device).manual_seed(71)
    paired = torch.Generator(device=policy.device).manual_seed(72)
    latent_states, paired_states = [], []
    for _ in range(draws):
        latent_states.append(latent.get_state())
        paired_states.append(paired.get_state())
        for _ in range(2 * len(policy.routed_control.spec.sites)):
            torch.randn((len(context) * context.shape[1], policy.table.shape[1]),
                        device=policy.device, generator=latent)
        for _ in range(2):
            torch.randn(targets.shape, device=policy.device, generator=paired)
    return capture_game(models=policy._training_modules(), table=policy.table, controller=policy.controller,
                        critic_optimizer=policy.opt_d, penalty=policy.penalty, rows=policy.routed_control.spec,
                        output_sigma=policy.output_sigma(), latent_rng_states=latent_states,
                        paired_rng_states=paired_states, penalty_units=loop.config["penalty_units"])


def compact_game(result):
    """Keep diagnostics small while preserving each common-noise paired result."""
    result = deepcopy(result)
    for draw in result["draws"]:
        for arm in ("before", "after"):
            record = draw[arm]
            for name in ("row_usage", "row_context_ess", "row_gradient_norm"):
                values = torch.tensor(record.pop(name), dtype=torch.float64)
                record[name + "_summary"] = dict(min=float(values.min()), mean=float(values.mean()),
                                                 max=float(values.max()))
            record.pop("table_gradient")
            record.pop("role_gradient", None)
    return result


def gradient_summary(game):
    gradients = torch.tensor([draw["before"]["table_gradient"] for draw in game["draws"]], dtype=torch.float64)
    mean = gradients.mean(0)
    noise = (gradients - mean).square().sum((1, 2)).mean()
    denominator = gradients.norm(dim=2).mean(0)
    consistency = mean.norm(dim=1) / denominator.clamp_min(1e-300)
    usage = torch.tensor([draw["before"]["row_usage"] for draw in game["draws"]], dtype=torch.float64).mean(0)
    roles = {role: torch.tensor([draw["before"]["role_gradient"][role] for draw in game["draws"]],
                                dtype=torch.float64).mean(0)
             for role in game["draws"][0]["before"]["role_gradient"]}
    return dict(mean=mean, usage=usage, roles=roles,
                summary={"mean_gradient_norm": float(mean.norm()),
                         "draw_gradient_noise_norm": float(noise.sqrt()),
                         "row_gradient_consistency_mean": float(consistency.mean()),
                         "row_gradient_consistency_min": float(consistency.min()),
                         "usage_effective_rows": float(usage.square().sum().reciprocal()),
                         "least_used_mass": float(usage.min()), "most_used_mass": float(usage.max())})


def train(loop, steps, *, log_every=100, game_diagnostics=False):
    seconds, trace, checks, moves, pending = 0., hashlib.sha256(), [], [], []
    previous_gradient, previous_roles, previous_splits = None, None, None
    control, policy = loop.policy.routed_control, loop.policy
    commit = control._commit
    capture_seconds = 0.

    def observed_commit(child, parent, candidate, average):
        nonlocal capture_seconds
        if game_diagnostics:
            synchronize(policy.device)
            start = time.perf_counter()
            context, target = control.guard_context[:control.guard_fill], control.guard_targets[:control.guard_fill]
            pending.append((game_capture(loop, context, target), context.clone(), target.clone(),
                            control.candidate(copy=True), deepcopy(candidate), child, parent,
                            dict(step=policy.completed_steps + 1,
                                 gradient_persistence_parent=float(control.evidence.gradient_persistence[parent]),
                                 gradient_persistence_child=float(control.evidence.gradient_persistence[child]),
                                 parent_flagged=bool(control.evidence.flag[parent]),
                                 child_flagged=bool(control.evidence.flag[child]),
                                 clean_decision=deepcopy(control.last))))
            synchronize(policy.device)
            capture_seconds += time.perf_counter() - start
        return commit(child, parent, candidate, average)

    if game_diagnostics:
        control._commit = observed_commit
    for index in range(steps):
        synchronize(loop.policy.device)
        start = time.perf_counter()
        with torch.autograd.set_multithreading_enabled(False):
            row = update(loop)
        synchronize(loop.policy.device)
        seconds += time.perf_counter() - start
        while pending:
            replay, context, target, before, after, child, parent, event = pending.pop(0)
            game = replay.compare(context, target, before, after, residual_scale=policy.D.scale,
                                  log_output_error=True, split_rows=(parent, child))
            future = replay.compare(context, target, before, after, residual_scale=policy.D.scale,
                                    prospective_bandwidth=True, split_rows=(parent, child))
            event.update(game=compact_game(game), prospective=compact_game(future))
            moves.append(event)
            print(json.dumps({"event": "move_game", "mode": loop.config["mode"], "step": event["step"],
                              "gradient_persistence_parent": event["gradient_persistence_parent"],
                              "parent_flagged": event["parent_flagged"], "mean_delta": game["mean_delta"]}), flush=True)
        trace.update(json.dumps([row[key] for key in ("batch_indices", "base_noise_sums", "paired_rng_digest",
                                                   "dv12_rng_digest")]).encode())
        if (index + 1) % log_every == 0 or index == steps - 1:
            check = {"step": row["step"], **evaluate(loop), "game_loss_d": row["loss_d"],
                     "game_loss_g": row["loss_g"], "output_sigma": row["output_sigma"],
                     "ka2_calls": loop.policy.opt_d.record.calls,
                     "penalty_phase": loop.policy.penalty.last_stats.get("phase"),
                     "row_diagnostics": diagnostics(loop.policy)}
            if game_diagnostics:
                context, target = loop.guard_context, loop.guard_targets
                replay = game_capture(loop, context, target)
                candidate = replay.candidate()
                game = replay.compare(context, target, candidate, candidate, residual_scale=policy.D.scale)
                gradient = gradient_summary(game)
                temporal = None
                splits = control.counters["splits"]
                if previous_gradient is not None and previous_splits == splits:
                    denominator = previous_gradient.norm() * gradient["mean"].norm()
                    temporal = None if not denominator > 0 else float((previous_gradient * gradient["mean"]).sum() / denominator)
                role_alignment = {}
                if previous_roles is not None:
                    for role, value in gradient["roles"].items():
                        old = previous_roles[role]
                        denominator = value.norm() * old.norm()
                        role_alignment[role] = None if not denominator > 0 else float(value @ old / denominator)
                previous_gradient = gradient["mean"]
                previous_roles, previous_splits = gradient["roles"], splits
                check["noisy_game"] = compact_game(game)
                check["gradient_observation"] = {**gradient["summary"], "previous_check_gradient_cosine": temporal,
                                                  "previous_global_role_cosine": role_alignment,
                                                  "cadence_updates": log_every}
            checks.append(check)
            logged = {name: value for name, value in check.items() if name != "noisy_game"}
            print(json.dumps({"event": "check", "mode": loop.config["mode"],
                              "max_context_harm": loop.config["max_context_harm"], **logged}), flush=True)
    control._commit = commit
    return {**loop.config, "steps": steps, **evaluate(loop), "training_seconds": seconds - capture_seconds,
            "diagnostic_capture_seconds": capture_seconds, "trace_sha256": trace.hexdigest(),
            "row_diagnostics": diagnostics(loop.policy), "checks": checks, "move_game_observations": moves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--task", choices=("transitions", "pooled"), default="transitions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--probe-interval", type=int, default=1)
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--max-context-harm", type=float, default=0.,
                        help="Explicit benchmark feature-harm allowance; RoutedRows defaults to zero")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--game-diagnostics", action="store_true")
    parser.add_argument("--include-default-full", action="store_true",
                        help="Also run full controls with the zero default when comparing an explicit allowance")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--receipt")
    args = parser.parse_args()
    if min(args.steps, args.tokens, args.particles, args.probe_interval, args.log_every) < 1:
        parser.error("counts must be positive")
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "particlegan").glob("*.py")),
             *(root / "examples" / name for name in ("e22_routed_sites.py", "e22_routed_replay.py",
                                                     "e22_routed_game.py", "e22_routed_support.py"))]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in paths if path.is_file()}
    torch.set_num_threads(1)
    reports, reference, trace = [], None, None
    arms = [(mode, args.max_context_harm) for mode in (MODES if args.compare else (args.mode,))]
    if args.include_default_full:
        if not args.compare or not args.max_context_harm > 0:
            parser.error("--include-default-full needs --compare and an explicit positive allowance")
        arms.append(("full", 0.))
    for mode, harm in arms:
        options = dict(mode=mode, device=args.device, tokens=args.tokens,
                       particles=args.particles, z_dim=4, probe_interval=args.probe_interval,
                       max_context_harm=harm)
        if args.task == "pooled":
            loop = pooled.make_loop(initialization="api", **options)
            loop.policy.attach_penalty(loop.policy.recipe.make_critic_penalty(loop.policy.opt_d, collect_stats=True))
            loop.config["task"] = "pooled_edit_v1"
        else:
            loop = make_loop(**options)
        weights = initial_weights(loop)
        if reference is None:
            reference = weights
        elif weights.keys() != reference.keys() or any(not torch.equal(value, reference[key])
                                                     for key, value in weights.items()):
            raise RuntimeError("comparison arms must share initial weights")
        report = train(loop, args.steps, log_every=args.log_every, game_diagnostics=args.game_diagnostics)
        reports.append(report)
        if trace is None:
            trace = report["trace_sha256"]
        elif trace != report["trace_sha256"]:
            raise RuntimeError("comparison arms must share batches and noise draw schedule")
    receipt = {"matched_initial_weights": True, "matched_batches_and_draws": True,
               "row_spec_defaults": {"max_context_harm": 0., "output_error_guard": False},
               "diagnostic_draws": 4 if args.game_diagnostics else 0,
               "diagnostic_latent_seed": 71, "diagnostic_paired_seed": 72,
               "game_diagnostics_do_not_select_moves": True,
               "timing_contract": "Synchronized optimizer/controller updates; private game replay and capture excluded. Shared-device wall time, no isolated throughput claim.",
               "device": str(args.device), "torch_version": torch.__version__, "reports": reports}
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
           for name, digest in source_hashes.items()):
        raise RuntimeError("benchmark source changed during the run; preserve and replay fixed source")
    receipt["source_sha256"] = source_hashes
    if torch.device(args.device).type == "cuda":
        receipt["hardware"] = torch.cuda.get_device_name(torch.device(args.device))
    logged = {**receipt, "reports": [{name: value for name, value in report.items()
                                      if name not in ("checks", "move_game_observations")}
                                     for report in reports]}
    print(json.dumps({"event": "comparison", **logged}), flush=True)
    if args.receipt:
        with open(args.receipt, "w") as file:
            json.dump(receipt, file, indent=2)
            file.write("\n")


if __name__ == "__main__":
    main()
