"""Two sequential token-routing sites sharing one E22 bank and controller.

python -u examples/e22_routed_sites.py --steps 60 --output /tmp/e22-sites.pt
python -u examples/e22_routed_sites.py --steps 2 --resume /tmp/e22-sites.pt
python -u examples/e22_routed_sites.py --compare --steps 40 --tokens 128 --z-dim 4 --particles 128

Every structural counterfactual reruns the entire conditioned model, including
the second site's queries that depend on the first site's output. Training is
paired-error RpGAN/KA2. Clean output RMSE is evaluated on a third context grid.
Optional ``--output-error-guard`` also checks paired output MSE on separate
protected contexts; it never supplies a training loss or selects proposals.

The CLI defaults to the public deterministic initializer for every trainable
network and the R2 particle prior. Python ``make_loop`` defaults to the older
``conformance`` fixture so the focused lifecycle/recovery tests retain their
deliberately constructed row-move trajectory. That fixture has a hand-built
bank, a downweighted outlier, and constant query weights; it is not an API-
initialized quality benchmark. Select either path explicitly with
``--initialization api|conformance``.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time

import torch
from torch import nn

from particlegan import E22Policy, RoutedBatch, RoutedRows, get_recipe, init


MODES = ("frozen", "no_rows", "full")
INITIALIZATIONS = ("api", "conformance")


class TokenHost(nn.Module):
    """Two frozen BF16 layers with FP32 residual projections between them."""

    def __init__(self, z_dim):
        super().__init__()
        self.first_host = nn.Linear(4, 4, bias=False).bfloat16()
        self.second_host = nn.Linear(4, 2, bias=False).bfloat16()
        with torch.no_grad():
            self.first_host.weight.copy_(torch.eye(4))
            self.second_host.weight.copy_(torch.tensor([[1., 0., .05, .02], [0., 1., -.07, -.01]]))
        self.first_host.requires_grad_(False)
        self.second_host.requires_grad_(False)
        self.first_adapter = nn.Linear(4 + z_dim, 4)
        self.second_adapter = nn.Linear(4 + z_dim, 2)
        with torch.no_grad():
            for layer in (self.first_adapter, self.second_adapter):
                layer.weight.zero_()
                layer.weight[:, 4:] = .01
                layer.bias.zero_()

    def first(self, context, encoded, codes):
        frozen = self.first_host(context.bfloat16()).float()
        return frozen + self.first_adapter(torch.cat((encoded, codes), dim=-1))

    def second(self, hidden, codes):
        frozen = self.second_host(hidden.bfloat16()).float()
        return frozen + self.second_adapter(torch.cat((hidden, codes), dim=-1))


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))
            self.projection.bias.zero_()

    def forward(self, context):
        return self.projection(context).tanh()


class SiteQueries(nn.Module):
    def __init__(self, rows, z_dim, *, conformance=True):
        super().__init__()
        self.first_query = nn.Linear(4, z_dim)
        self.second_query = nn.Linear(4, z_dim)
        if conformance:
            with torch.no_grad():
                for query in (self.first_query, self.second_query):
                    query.weight.fill_(.025)
                    query.bias.fill_(.1)
        self.register_buffer("log_mass", torch.zeros(rows))
        if conformance:
            self.log_mass[-1] = -1.5


class TokenErrorCritic(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 8), nn.Tanh())
        self.score = nn.Linear(8, 1)
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, error):
        return self.score(self.features(error).mean(dim=1))


def model_forward(models, context, candidate, routing):
    G, E, R = models["generator"], models["encoder"], models["router"]
    encoded = E(context)
    first_logits = R.first_query(encoded) @ candidate.table.T / math.sqrt(candidate.table.shape[1])
    first_codes = routing.mix("first", first_logits)
    hidden = G.first(context, encoded, first_codes)
    # These queries must be recomputed after a counterfactual first-site change.
    second_logits = R.second_query(hidden) @ candidate.table.T / math.sqrt(candidate.table.shape[1])
    second_codes = routing.mix("second", second_logits)
    return G.second(hidden, second_codes)


def paired_features(models, context, samples, targets):
    critic = models["critic"]
    # Preserve every token in the final-model residual metric. Routing usage
    # separately averages tokens/sites without counting them as extra contexts.
    return critic.features((samples - targets) / critic.scale).flatten(1)


def context_grid(source_points, times, tokens):
    x, y, t = torch.meshgrid(source_points, source_points, times, indexing="ij")
    conditions = torch.stack((x.flatten(), y.flatten(), t.flatten()), dim=-1)
    positions = torch.linspace(-1., 1., tokens)
    return torch.cat((conditions[:, None].expand(-1, tokens, -1),
                      positions[None, :, None].expand(len(conditions), -1, -1)), dim=-1)


def neutral(context):
    weight = context.new_tensor([[1., 0., .05, .02], [0., 1., -.07, -.01]]).bfloat16()
    return (context.bfloat16() @ weight.T).float()


def targets(context):
    x, y, t, position = context.unbind(dim=-1)
    edit = torch.stack((.22 + .035 * x + .07 * t + .015 * (math.pi * position).sin(),
                        .19 - .04 * y + .025 * t.square() + .01 * (math.pi * position).cos()), dim=-1)
    return neutral(context) + edit


@dataclass
class SiteLoop:
    policy: E22Policy
    fit_context: torch.Tensor
    fit_targets: torch.Tensor
    guard_context: torch.Tensor
    guard_targets: torch.Tensor
    test_context: torch.Tensor
    test_targets: torch.Tensor
    data_rng: torch.Generator
    paired_noise_rng: torch.Generator
    initial_rmse: float
    config: dict


def make_loop(*, mode="full", device="cpu", tokens=8, z_dim=2, particles=16,
              batch_size=8, initialization="conformance", output_error_guard=False,
              probe_interval=1):
    """Build a loop; conformance preserves the explicit structural-test fixture.

    Use initialization="api" for quality/timing comparisons. Its whole-network
    initializer calls use fixed role keys G=0, D=1, E=2, R=3; these keys define
    one initialization, rather than a search over seeds.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"initialization must be one of {INITIALIZATIONS}")
    device = torch.device(device)
    fit = context_grid(torch.linspace(-.8, .8, 5), torch.tensor([.1, .35, .6, .85]), tokens).to(device)
    guard = context_grid(torch.tensor([-.72, -.24, .24, .72]), torch.tensor([.18, .48, .78]), tokens).to(device)
    test = context_grid(torch.linspace(-.75, .75, 6), torch.tensor([.13, .29, .45, .61, .77]), tokens).to(device)
    fit_target, guard_target, test_target = targets(fit), targets(guard), targets(test)
    scale = (fit_target - neutral(fit)).std(dim=(0, 1)).clamp_min(.04)
    controls = mode == "full"
    recipe = get_recipe("e22_routed", num_particles=particles, z_dim=z_dim,
                        batch_size=batch_size, output_noise_std=.125,
                        row_evidence_gate=controls, particle_birth_death=controls)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        G, E = TokenHost(z_dim).to(device), ContextEncoder().to(device)
        # Fresh constructor draws let the API initialize queries: its deliberate
        # constant-preservation rule would retain the conformance fixture fills.
        R = SiteQueries(particles, z_dim, conformance=initialization == "conformance").to(device)
        D = TokenErrorCritic(scale).to(device)
        if initialization == "api":
            init.deterministic_orthogonal_(G, seed=0)
            init.deterministic_orthogonal_(D, seed=1)
            init.deterministic_orthogonal_(E, seed=2)
            init.deterministic_orthogonal_(R, seed=3)
            # Initialize while learnable, then freeze the same R2 values in the
            # frozen arm. The public initializer intentionally skips frozen state.
            prior = init.deterministic_orthogonal_(recipe.make_prior()).to(device)
            table = prior.z.requires_grad_(mode != "frozen")
        else:
            init.deterministic_orthogonal_(D.features, seed=5)
            init.deterministic_orthogonal_(D.score, seed=6)
            angle = torch.linspace(0., 2 * math.pi, particles, device=device)[:, None]
            phase = torch.arange(z_dim, device=device)[None, :]
            table = nn.Parameter(1 + .1 * (angle + phase).sin(), requires_grad=mode != "frozen")
            with torch.no_grad():
                table[-1].fill_(-2.)
    groups = [{"params": [p for p in G.parameters() if p.requires_grad]},
              {"params": list(E.parameters())}, {"params": list(R.parameters())}]
    roles = ["generator", "encoder", "router"]
    if table.requires_grad:
        groups.append({"params": [table], "lr": recipe.lr * recipe.prior_lr_mult})
        roles.append("table")
    opt_g = recipe.make_generator_optimizer(groups, latent_table=table if table.requires_grad else None,
                                             foreach=False)
    opt_d = recipe.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    rows = RoutedRows(model_forward=model_forward, features=paired_features,
                      sites=("first", "second"), probe_budget=8, reservoir_size=64,
                      min_observations=8, min_effect=1e-8, improvement_margin=1e-10,
                      max_context_harm=1e-4, persistence_threshold=.75, split_scale=.1,
                      output_error_guard=output_error_guard, probe_interval=probe_interval)
    policy = E22Policy(recipe, G, D, table=table, encoder=E, router=R,
                       generator_optimizer=opt_g, critic_optimizer=opt_d,
                       roles=[roles, ["critic"]], routed_rows=rows, seed=21)
    initial = float((policy.served_model().routed_forward(test) - test_target).square().mean().sqrt())
    config = dict(mode=mode, tokens=tokens, z_dim=z_dim, particles=particles,
                  batch_size=batch_size, initialization=initialization)
    if output_error_guard:
        config["output_error_guard"] = True
    if probe_interval != 1:
        config["probe_interval"] = probe_interval
    return SiteLoop(policy, fit, fit_target, guard, guard_target, test, test_target,
                    torch.Generator(device=device).manual_seed(42),
                    torch.Generator(device=device).manual_seed(43), initial, config)


def update(loop, *, generator_forward=None):
    p = loop.policy
    indices = torch.randint(len(loop.fit_context), (p.recipe.batch_size,),
                            generator=loop.data_rng, device=p.device)
    context, target = loop.fit_context[indices], loop.fit_targets[indices]
    # This base stream is separate from policy DV12 draws, and starts identically
    # in all comparison arms. Store it alongside the data cursor for recovery.
    critic_base = torch.randn(target.shape, device=p.device, generator=loop.paired_noise_rng)
    generator_base = torch.randn(target.shape, device=p.device, generator=loop.paired_noise_rng)
    batch = RoutedBatch(context, target, loop.guard_context, loop.guard_targets)
    noise = p.begin_step(target, routed=batch)
    loss = p.recipe.make_loss()
    p.G.eval()
    p.encoder.eval()
    p.router.eval()
    p.D.train()
    with torch.no_grad():
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * critic_base
        fake = real + (prediction - target) / p.D.scale
    p.observe_critic_pair(real, fake)
    penalty = p.penalty(p.D, real, fake)
    loss_d = loss.d_loss(p.D(real), p.D(fake)) + penalty
    p.opt_d.zero_grad()
    p.before_critic_backward()
    loss_d.backward()
    p.opt_d.step()
    p.after_critic_step()
    p.D.eval()
    p.G.train()
    p.encoder.train()
    p.router.train()
    flags = [parameter.requires_grad for parameter in p.D.parameters()]
    try:
        p.D.requires_grad_(False)
        prediction = (p.routed_generate(context, sigma=0, perturb=True)
                      if generator_forward is None else generator_forward(p, context))
        real = noise.output_sigma * generator_base
        with torch.no_grad():
            real_logits = p.D(real.detach())
        loss_g = loss.g_loss(p.D(real + (prediction - target) / p.D.scale), real_logits)
        p.opt_g.zero_grad()
        p.before_generator_backward()
        loss_g.backward()
        dense_rows = 0 if p.table.grad is None else int(p.table.grad.norm(dim=-1).gt(0).sum())
        p.after_generator_backward(loss_gan=loss_g.detach(), loss_critic=(loss_d - penalty).detach())
        p.opt_g.step()
        p.after_generator_step()
    finally:
        for parameter, flag in zip(p.D.parameters(), flags):
            parameter.requires_grad_(flag)
    event = p.finish_step()
    return {"step": p.completed_steps, "loss_d": float(loss_d.detach()), "loss_g": float(loss_g.detach()),
            "dense_gradient_rows": dense_rows, "output_sigma": p.output_sigma(), "move": event,
            "batch_indices": indices.tolist(),
            "base_noise_sums": [float(critic_base.sum()), float(generator_base.sum())],
            "paired_rng_digest": hashlib.sha256(bytes(loop.paired_noise_rng.get_state().cpu().tolist())).hexdigest(),
            "dv12_rng_digest": hashlib.sha256(bytes(p.noise_generator.get_state().cpu().tolist())).hexdigest()}


@torch.no_grad()
def evaluate(loop):
    served = loop.policy.served_model()
    error = served.routed_forward(loop.test_context) - loop.test_targets
    return {"initial_rmse": loop.initial_rmse, "heldout_rmse": float(error.square().mean().sqrt()),
            "heldout_max_error": float(error.norm(dim=-1).max()), "served_source": served.source}


def diagnostics(policy):
    rows = policy.routed_control.diagnostics()
    return {"counters": rows["counters"], "last": rows["last"],
            "evidence_counters": rows["rows"]["counters"]}


def checkpoint(loop):
    return {"policy": loop.policy.state_dict(), "data_rng": loop.data_rng.get_state(),
            "paired_noise_rng": loop.paired_noise_rng.get_state(), "initial_rmse": loop.initial_rmse,
            "config": loop.config}


def restore(loop, state):
    config = {"initialization": "conformance", **state["config"]}
    if loop.config != config:
        raise ValueError("checkpoint task shape/mode/initialization must match the caller-owned model")
    loop.policy.load_state_dict(state["policy"])
    loop.data_rng.set_state(state["data_rng"].cpu())
    loop.paired_noise_rng.set_state(state["paired_noise_rng"].cpu())
    loop.initial_rmse = state["initial_rmse"]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup(loop, steps):
    """Exercise first-use kernels, then restore all training state and modes."""
    if type(steps) is not int or steps < 0:
        raise ValueError("warmup steps must be a nonnegative integer")
    if not steps:
        return
    p = loop.policy
    initial = checkpoint(loop)
    roots = [*p._training_modules().values(), *p._average_modules().values(), p.opt_d.ema_critic]
    modes = {module: module.training for root in roots for module in root.modules()}
    parameters = dict.fromkeys([parameter for root in roots for parameter in root.parameters()]
                               + [p.table, p.log_output_sigma])
    gradients = {parameter: None if parameter.grad is None else parameter.grad.detach().clone()
                 for parameter in parameters}
    try:
        with torch.autograd.set_multithreading_enabled(False):
            for _ in range(steps):
                update(loop)
        synchronize(p.device)
    finally:
        restore(loop, initial)
        for module, flag in modes.items():
            module.training = flag
        for parameter, gradient in gradients.items():
            parameter.grad = gradient


def train(loop, steps, *, log_every=20, warmup_steps=0):
    warmup(loop, warmup_steps)
    seconds = 0.
    matched_inputs = []
    for index in range(steps):
        synchronize(loop.policy.device)
        start = time.perf_counter()
        with torch.autograd.set_multithreading_enabled(False):
            row = update(loop)
        synchronize(loop.policy.device)
        seconds += time.perf_counter() - start
        matched_inputs.append((row["batch_indices"], row["base_noise_sums"],
                               row["paired_rng_digest"], row["dv12_rng_digest"]))
        if (index + 1) % log_every == 0 or index == steps - 1:
            print(json.dumps({"event": "train", "mode": loop.config["mode"],
                              "step": row["step"], "loss_g": row["loss_g"],
                              "dense_gradient_rows": row["dense_gradient_rows"],
                              "row_diagnostics": diagnostics(loop.policy)}), flush=True)
    report = {"event": "complete", **loop.config, "steps": steps, "warmup_steps": warmup_steps,
              **evaluate(loop),
              "training_seconds": seconds, "seconds_per_update": seconds / steps,
              "row_diagnostics": diagnostics(loop.policy)}
    print(json.dumps(report), flush=True)
    return report, matched_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--z-dim", type=int)
    parser.add_argument("--particles", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--initialization", choices=INITIALIZATIONS, default="api",
                        help="Public API initialization (default), or the deliberately constructed conformance fixture")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output-error-guard", action="store_true",
                        help="Require protected final-output MSE nonincrease for fast and averaged structural proposals")
    parser.add_argument("--probe-interval", type=int, default=1,
                        help="Observed training updates between expensive row probes/proposals; gradients are observed each update")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--warmup", type=int, help="Untimed restored updates; default 1 for --compare, 0 otherwise")
    parser.add_argument("--checkpoint", "--output", dest="checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1 or (args.warmup is not None and args.warmup < 0):
        parser.error("--steps and --log-every must be positive; --warmup must be nonnegative")
    if args.probe_interval < 1:
        parser.error("--probe-interval must be positive")
    if args.compare and (args.resume or args.checkpoint):
        parser.error("--compare reports three fresh matched arms; checkpoints use one --mode")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    config = dict(tokens=args.tokens or (128 if args.compare else 8),
                  z_dim=args.z_dim or (4 if args.compare else 2),
                  particles=args.particles or (128 if args.compare else 16), batch_size=args.batch_size,
                  initialization=args.initialization, output_error_guard=args.output_error_guard,
                  probe_interval=args.probe_interval)
    warmup_steps = int(args.compare) if args.warmup is None else args.warmup
    if args.compare:
        reports, reference_inputs, reference_weights = [], None, None
        for mode in MODES:
            loop = make_loop(mode=mode, device=device, **config)
            weights = {f"{name}.{key}": value.detach().clone()
                       for name, model in loop.policy._training_modules().items()
                       for key, value in model.state_dict().items()}
            weights["table"] = loop.policy.table.detach().clone()
            weights["output_sigma"] = loop.policy.log_output_sigma.detach().clone()
            if reference_weights is None:
                reference_weights = weights
            elif weights.keys() != reference_weights.keys() or any(
                    not torch.equal(value, reference_weights[key]) for key, value in weights.items()):
                raise RuntimeError("comparison arms received different initial parameters")
            report, inputs = train(loop, args.steps, log_every=args.log_every, warmup_steps=warmup_steps)
            if reference_inputs is None:
                reference_inputs = inputs
            elif inputs != reference_inputs:
                raise RuntimeError("comparison arms received different batches or base-noise draws")
            reports.append(report)
        print(json.dumps({"event": "comparison", "matched_initial_parameters": True,
                          "matched_batches_and_base_noise": True,
                          "matched_dv12_draw_schedule": True, "dv12_draws_per_update": 4,
                          "dv12_draw_shape": [config["batch_size"] * config["tokens"], config["z_dim"]],
                          "device": str(device), "reports": reports}), flush=True)
    else:
        saved = None if not args.resume else torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved is not None:
            config = {"initialization": "conformance",
                      **{key: value for key, value in saved["config"].items() if key != "mode"}}
        loop = make_loop(mode=args.mode if saved is None else saved["config"]["mode"], device=device, **config)
        if saved is not None:
            restore(loop, saved)
        train(loop, args.steps, log_every=args.log_every, warmup_steps=warmup_steps)
        if args.checkpoint:
            torch.save(checkpoint(loop), args.checkpoint)


if __name__ == "__main__":
    main()
