"""Conditional, densely routed E22 training with a caller-owned paired-error game.

python -u examples/e22_routed_paired.py --steps 160 --output /tmp/routed-e22.pt
python -u examples/e22_routed_paired.py --steps 20 --resume /tmp/routed-e22.pt

The synthetic task supplies a source vector and time to a frozen BF16 host.
FP32 encoder, router, adapter and tied key/value table learn the paired edit
through RpGAN and KA2. RMSE is used only for reporting on a third, disjoint
context grid; it is never a training loss or a structural acceptance criterion.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math

import torch
from torch import nn

from particlegan import E22Policy, RoutedBatch, RoutedRows, get_recipe, init


class FrozenHostAdapter(nn.Module):
    """A BF16 host with an FP32 trainable residual projection."""

    def __init__(self):
        super().__init__()
        self.host = nn.Linear(3, 2, bias=False).bfloat16()
        with torch.no_grad():
            self.host.weight.copy_(torch.tensor([[1., 0., .05], [0., 1., -.07]]))
        self.host.requires_grad_(False)
        self.adapter = nn.Linear(5, 2)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[:, 3:] = .025
            self.adapter.bias.zero_()

    def forward(self, context, encoded, codes):
        # Real integrations can cast the residual back at the host boundary.
        # Keep this small task's output FP32 so the learned residual is observable.
        base = self.host(context.to(torch.bfloat16)).float()
        return base + self.adapter(torch.cat((encoded, codes), dim=-1))


class SourceTimeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(3))
            self.projection.bias.zero_()

    def forward(self, context):
        return self.projection(context).tanh()


class DenseRouter(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.query = nn.Linear(3, 2)
        with torch.no_grad():
            self.query.weight.fill_(.025)
            self.query.bias.fill_(.1)
        # Explicit mass is necessary: duplicating a softmax key must divide its
        # represented mass, rather than silently double that component's weight.
        self.register_buffer("log_mass", torch.zeros(rows))
        self.log_mass[-1] = -1.5


class PairedErrorCritic(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 8), nn.Tanh())
        self.score = nn.Linear(8, 1)
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, normalized_error):
        return self.score(self.features(normalized_error))


def route(models, context, candidate):
    encoded = models["encoder"](context)
    query = models["router"].query(encoded)
    logits = query @ candidate.table.T / math.sqrt(candidate.table.shape[1])
    return (logits + candidate.log_mass).softmax(dim=-1)


def generate(models, context, candidate, weights):
    # RoutedRows supplies weights @ table, after optional mixed-code DV12.
    # Recomputing the code here would discard the policy's perturbation.
    return models["generator"](context, models["encoder"](context), candidate.codes)


def paired_features(models, context, samples, targets):
    critic = models["critic"]
    return critic.features((samples - targets) / critic.scale)


def context_grid(source_points, times):
    source_x, source_y, time = torch.meshgrid(source_points, source_points, times, indexing="ij")
    return torch.stack((source_x.flatten(), source_y.flatten(), time.flatten()), dim=-1)


def targets(context):
    weight = context.new_tensor([[1., 0., .05], [0., 1., -.07]]).bfloat16()
    base = (context.bfloat16() @ weight.T).float()
    source_x, source_y, time = context.unbind(dim=-1)
    edit = torch.stack((.22 + .035 * source_x + .07 * time,
                        .19 - .04 * source_y + .025 * time.square()), dim=-1)
    return base + edit


@dataclass
class PairedLoop:
    policy: E22Policy
    fit_context: torch.Tensor
    fit_targets: torch.Tensor
    guard_context: torch.Tensor
    guard_targets: torch.Tensor
    test_context: torch.Tensor
    test_targets: torch.Tensor
    data_rng: torch.Generator
    initial_rmse: float


def make_loop(*, device="cpu", batch_size=32, particles=16):
    """Bind four FP32 trainable roles and a nested frozen BF16 host."""
    device = torch.device(device)
    fit = context_grid(torch.linspace(-.8, .8, 5), torch.tensor([.1, .35, .6, .85])).to(device)
    guard = context_grid(torch.tensor([-.72, -.24, .24, .72]), torch.tensor([.18, .48, .78])).to(device)
    test = context_grid(torch.linspace(-.75, .75, 6), torch.tensor([.13, .29, .45, .61, .77])).to(device)
    fit_target, guard_target, test_target = targets(fit), targets(guard), targets(test)
    # Whiten the paired edit using fitting data only. No test data enters a policy.
    base_weight = fit.new_tensor([[1., 0., .05], [0., 1., -.07]]).bfloat16()
    paired_edit = fit_target - (fit.bfloat16() @ base_weight.T).float()
    scale = paired_edit.std(dim=0).clamp_min(.04)
    recipe = get_recipe("e22_routed", num_particles=particles, z_dim=2,
                        batch_size=batch_size, output_noise_std=.125)
    # Initialize one deterministic task; this example does not sweep seeds.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        G = FrozenHostAdapter().to(device)
        E = SourceTimeEncoder().to(device)
        R = DenseRouter(particles).to(device)
        D = PairedErrorCritic(scale).to(device)
    init.deterministic_orthogonal_(D.features, seed=5)
    init.deterministic_orthogonal_(D.score, seed=6)
    angles = torch.linspace(0., 2 * math.pi, particles, device=device)
    table = nn.Parameter(torch.stack((1 + .1 * angles.sin(), 1 + .1 * angles.cos()), dim=-1))
    with torch.no_grad():
        table[-1].fill_(-2.)
    opt_g = recipe.make_generator_optimizer([
        {"params": [p for p in G.parameters() if p.requires_grad]},
        {"params": list(E.parameters())}, {"params": list(R.parameters())},
        {"params": [table], "lr": recipe.lr * recipe.prior_lr_mult},
    ], latent_table=table, foreach=False)
    opt_d = recipe.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    # Conditional deletion evidence and coupled fit/guard proposals have their
    # own named diagnostics. Independent-particle p-values are not reused here.
    rows = RoutedRows(route=route, generate=generate, features=paired_features,
                      probe_budget=8, reservoir_size=64, min_observations=8,
                      min_effect=1e-8, improvement_margin=1e-10,
                      max_context_harm=1e-4, persistence_threshold=.75,
                      split_scale=.1)
    policy = E22Policy(recipe, G, D, table=table, encoder=E, router=R,
                       generator_optimizer=opt_g, critic_optimizer=opt_d,
                       roles=[["generator", "encoder", "router", "table"], ["critic"]],
                       routed_rows=rows, seed=21)
    initial = float((policy.served_model().routed_forward(test) - test_target).square().mean().sqrt())
    return PairedLoop(policy, fit, fit_target, guard, guard_target, test, test_target,
                      torch.Generator(device=device).manual_seed(42), initial)


def update(loop):
    """One caller-owned paired-error RpGAN/KA2 update in documented hook order."""
    p = loop.policy
    indices = torch.randint(len(loop.fit_context), (p.recipe.batch_size,),
                            generator=loop.data_rng, device=p.device)
    context, target = loop.fit_context[indices], loop.fit_targets[indices]
    routed = RoutedBatch(context, target, loop.guard_context, loop.guard_targets)
    noise = p.begin_step(target, routed=routed)
    loss = p.recipe.make_loss()
    p.G.eval()
    p.encoder.eval()
    p.router.eval()
    p.D.train()
    with torch.no_grad():
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        # One shared noise vector for the real/fake pair. routed_generate's
        # sigma=0 prevents a second fake-only output-noise draw.
        real = noise.output_sigma * torch.randn(target.shape, device=p.device,
                                                generator=p.noise_generator)
        fake = real + (prediction - target) / p.D.scale
    p.observe_critic_pair(real, fake)
    adversarial_d = loss.d_loss(p.D(real), p.D(fake))
    penalty = p.penalty(p.D, real, fake)
    loss_d = adversarial_d + penalty
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
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        # The real path is fixed; the fake path retains learned log_sigma's graph.
        real = noise.output_sigma * torch.randn(target.shape, device=p.device,
                                                generator=p.noise_generator)
        with torch.no_grad():
            real_logits = p.D(real.detach())
        loss_g = loss.g_loss(p.D(real + (prediction - target) / p.D.scale), real_logits)
        p.opt_g.zero_grad()
        p.before_generator_backward()
        loss_g.backward()
        dense_rows = int(p.table.grad.norm(dim=-1).gt(0).sum())
        p.after_generator_backward(loss_gan=loss_g.detach(), loss_critic=(loss_d - penalty).detach())
        p.opt_g.step()
        p.after_generator_step()
    finally:
        for parameter, flag in zip(p.D.parameters(), flags):
            parameter.requires_grad_(flag)
    event = p.finish_step()
    return {"step": p.completed_steps, "loss_d": float(loss_d.detach()),
            "loss_g": float(loss_g.detach()), "dense_gradient_rows": dense_rows,
            "output_sigma": p.output_sigma(), "move": event}


@torch.no_grad()
def evaluate(loop):
    """Clean frozen serving on final contexts; no policy observation or RNG draw."""
    served = loop.policy.served_model()
    prediction = served.routed_forward(loop.test_context)
    error = prediction - loop.test_targets
    return {"initial_rmse": loop.initial_rmse,
            "heldout_rmse": float(error.square().mean().sqrt()),
            "heldout_max_error": float(error.norm(dim=-1).max()),
            "served_source": served.source, "output_dtype": str(prediction.dtype)}


def checkpoint(loop):
    return {"policy": loop.policy.state_dict(), "data_rng": loop.data_rng.get_state(),
            "initial_rmse": loop.initial_rmse}


def restore(loop, state):
    loop.policy.load_state_dict(state["policy"])
    loop.data_rng.set_state(state["data_rng"].cpu())
    loop.initial_rmse = state["initial_rmse"]


def row_diagnostics(policy):
    diagnostics = policy.birth_death.diagnostics()
    evidence = diagnostics["rows"]
    return {"counters": diagnostics["counters"], "last": diagnostics["last"],
            "evidence_law": evidence["law"], "evidence_fraction": evidence["fraction"],
            "evidence_counters": evidence["counters"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=160, help="Additional updates; no finite horizon")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", "--output", dest="checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1:
        parser.error("--steps and --log-every must be positive")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(1)
    loop = make_loop(device=args.device, batch_size=args.batch_size, particles=args.particles)
    if args.resume:
        restore(loop, torch.load(args.resume, map_location="cpu", weights_only=True))
    print(json.dumps({"event": "start", "step": loop.policy.completed_steps,
                      "initial_rmse": loop.initial_rmse, "row_policy": "routed_paired"}), flush=True)
    for index in range(args.steps):
        with torch.autograd.set_multithreading_enabled(False):
            stats = update(loop)
        if (stats["move"] and stats["move"].get("moves", 0)) or (index + 1) % args.log_every == 0 or index == args.steps - 1:
            stats["event"] = "train"
            stats["row_diagnostics"] = row_diagnostics(loop.policy)
            print(json.dumps(stats), flush=True)
    if args.checkpoint:
        torch.save(checkpoint(loop), args.checkpoint)
    print(json.dumps({"event": "complete", "step": loop.policy.completed_steps, **evaluate(loop),
                      "row_diagnostics": row_diagnostics(loop.policy)}), flush=True)


if __name__ == "__main__":
    main()
