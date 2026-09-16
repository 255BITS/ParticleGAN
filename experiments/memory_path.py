"""Sequential circle prediction with a D-owned, real-only fast memory.

Run: .venv/bin/python -u experiments/memory_path.py --out runs/memory_path/scout
Tail: tail -f runs/memory_path/scout/experiment.log
Uses the public ParticleGAN API; all data, models and state live in this toy.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlegan import get_recipe, learning_rate_scale


VARIANTS = ("shared", "d_only", "no_memory", "buffer")
MEMORY_DIM = 32


def mlp(inputs, outputs, width=64):
    return nn.Sequential(nn.Linear(inputs, width), nn.LeakyReLU(.2),
                         nn.Linear(width, width), nn.LeakyReLU(.2),
                         nn.Linear(width, outputs))


class FastMemory(nn.Module):
    """Rank-one writes of learned features to four fixed temporal keys.

    M <- M * decay + v(x) outer (1-decay). There is no learned G write,
    explicit velocity, target, task label, or raw-point bypass to the reader.
    This is a multi-timescale fast-weight matrix, not content-addressed memory.
    """
    def __init__(self):
        super().__init__()
        self.values = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 8), nn.Tanh())
        self.register_buffer("decay", torch.tensor([0., .5, .8, .95]))

    def write(self, memory, point):
        value = self.values(point)
        return memory * self.decay + value[:, :, None] * (1 - self.decay)

    def forward(self, context):
        memory = context.new_zeros(len(context), 8, 4)
        # Encode in one batch; the recurrence itself is deliberately explicit.
        values = self.values(context)
        for value in values.unbind(1):
            memory = memory * self.decay + value[:, :, None] * (1 - self.decay)
        return memory.flatten(1)


class Critic(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.writer = FastMemory()
        self.reader = mlp(2 + MEMORY_DIM, 1)

    def memory(self, context):
        if self.variant == "no_memory":
            return context.new_zeros(len(context), MEMORY_DIM)
        if self.variant == "buffer":
            # Chronological, right aligned raw observations; identical reader size.
            recent = context[:, -4:].flatten(1)
            return F.pad(recent, (MEMORY_DIM - recent.shape[1], 0))
        return self.writer(context)

    def forward(self, point, memory):
        return self.reader(torch.cat((point, memory), -1)).squeeze(-1)


class Generator(nn.Module):
    def __init__(self, variant, z_dim=4):
        super().__init__()
        self.variant = variant
        self.net = mlp(z_dim + MEMORY_DIM, 2)

    def forward(self, z, memory):
        if self.variant in ("no_memory", "d_only"):
            memory = torch.zeros_like(memory)
        return self.net(torch.cat((z, memory), -1))


def circles(batch, length, rng, device, reversal=False, noise=.03):
    """Independent episodes; reversal changes the transition out of index 8."""
    def rand(*shape):
        return torch.rand(*shape, generator=rng, device=device)
    center = (rand(batch, 1, 2) - .5) * 1.5
    radius = .6 + .8 * rand(batch, 1)
    phase = 2 * math.pi * rand(batch, 1)
    omega = (.12 + .28 * rand(batch, 1)) * (2 * (rand(batch, 1) > .5) - 1)
    times = torch.arange(length, device=device).float()
    if reversal:
        times = torch.where(times <= 8, times, 16 - times)
    theta = phase + omega * times
    clean = center + radius[:, :, None] * torch.stack((theta.cos(), theta.sin()), -1)
    observed = clean + noise * torch.randn(clean.shape, generator=rng, device=device)
    return observed, clean


@torch.no_grad()
def predictions(generator, critic, prior, observed, particles=64, intervention=None):
    z = prior(torch.arange(particles, device=observed.device))
    clouds, memories = [], []
    for length in range(1, observed.shape[1]):
        memory = critic.memory(observed[:, :length])
        memories.append(memory)
        if intervention == "shuffle":
            memory = memory.roll(1, 0)
        elif intervention == "zero":
            memory = torch.zeros_like(memory)
        clouds.append(generator(z.expand(len(memory), -1, -1).reshape(-1, z.shape[-1]),
                                memory[:, None].expand(-1, particles, -1).reshape(-1, MEMORY_DIM))
                      .reshape(len(memory), particles, 2))
    return torch.stack(clouds, 1), torch.stack(memories, 1)


def scores(clouds, observed, clean):
    """Conditional energy score and Euclidean centroid RMSE, by prefix length."""
    mean = clouds.mean(2)
    rmse = (mean - clean[:, 1:]).square().sum(-1).mean(0).sqrt()
    # Unbiased U-statistic estimate of E|X-y| - .5 E|X-X'|.
    n = clouds.shape[2]
    distances = torch.cdist(clouds, clouds).sum((-1, -2)) / (n * (n - 1))
    energy = ((clouds - observed[:, 1:, None]).norm(dim=-1).mean(-1) - .5 * distances).mean(0)
    spread = clouds.std(2).square().sum(-1).sqrt().mean(0)
    return {"rmse": rmse.tolist(), "energy": energy.tolist(), "spread": spread.tolist()}


def aggregate(curves):
    # Prefixes 4..12: enough history, within the training prefix range.
    return {key: float(np.mean(value[3:12])) for key, value in curves.items()}


def ridge_probe(train_m, train_y, test_m, test_y):
    # A held-out linear probe is diagnostic, not an upper bound on information.
    train_m, test_m = train_m.double(), test_m.double()
    mean, std = train_m.mean(0), train_m.std(0).clamp_min(1e-5)
    a = F.pad((train_m - mean) / std, (0, 1), value=1)
    b = F.pad((test_m - mean) / std, (0, 1), value=1)
    reg = torch.eye(a.shape[1], device=a.device, dtype=a.dtype) * .01
    weights = torch.linalg.solve(a.T @ a + reg, a.T @ train_y.double())
    return (b @ weights - test_y).square().sum(-1).mean().sqrt().item()


@torch.no_grad()
def evaluate(generator, critic, prior, device, batch=256):
    results, artifacts = {}, {}
    for name, reversal in (("circle", False), ("reversal", True)):
        rng = torch.Generator(device=device).manual_seed(20260916)
        observed, clean = circles(batch, 17, rng, device, reversal)
        clouds, memories = predictions(generator, critic, prior, observed)
        curves = scores(clouds, observed, clean)
        results[name] = {"curves": curves, **aggregate(curves)}
        artifacts[name] = dict(observed=observed[:4].cpu().numpy(), clean=clean[:4].cpu().numpy(),
                               clouds=clouds[:4].cpu().numpy())
        if name == "circle":
            shuffled, _ = predictions(generator, critic, prior, observed, intervention="shuffle")
            results["shuffle"] = aggregate(scores(shuffled, observed, clean))
            zeros, _ = predictions(generator, critic, prior, observed, intervention="zero")
            results["zero"] = aggregate(scores(zeros, observed, clean))
            cut = batch // 2
            m = memories[:, 3:12].reshape(batch, -1, MEMORY_DIM)
            target = clean[:, 4:13]
            results["probe_next_rmse"] = ridge_probe(m[:cut].flatten(0, 1), target[:cut].flatten(0, 1),
                                                       m[cut:].flatten(0, 1), target[cut:].flatten(0, 1))
            results["memory_std"] = memories[:, 3:12].flatten(0, 1).std(0).mean().item()
    return results, artifacts


def train(variant, args, log):
    device = torch.device(args.device)
    torch.manual_seed(42)  # One initialization seed, no seed sweep.
    recipe = get_recipe(total_steps=args.steps, batch_size=args.batch_size,
                        num_particles=512, z_dim=4)
    generator, critic = Generator(variant).to(device), Critic(variant).to(device)
    prior = recipe.make_prior().to(device)
    gan, penalty = recipe.make_loss(), recipe.make_gradient_penalty()
    spread = recipe.make_prior_regularizer()
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in (opt_g, opt_d)]
    data_rng = torch.Generator(device=device).manual_seed(31415)
    latent_rng = torch.Generator(device=device).manual_seed(27182)
    directory = args.out / variant
    directory.mkdir(parents=True, exist_ok=True)
    config = {"variant": variant, "recipe": recipe.to_dict(), "device": args.device,
              "seed": 42, "train_data_seed": 31415, "test_data_seed": 20260916,
              "prefixes": [1, 12], "noise": .03, "ema": False, "eval_batch": args.eval_batch,
              "memory": "8 learned features x decays [0, .5, .8, .95]",
              "writer_gradient": "D objective only; fake detached; G reads detached memory"}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    log(event="start", variant=variant, config=config)
    with (directory / "metrics.jsonl").open("w", buffering=1) as metrics:
        for step in range(1, args.steps + 1):
            scale = learning_rate_scale(step - 1, args.steps, recipe.lr_anneal_start, recipe.lr_floor)
            for opt, rates in zip((opt_g, opt_d), base_lrs):
                for group, rate in zip(opt.param_groups, rates):
                    group["lr"] = rate * scale
            length = int(torch.randint(1, 13, (), generator=data_rng, device=device))
            observed, _ = circles(args.batch_size, length + 1, data_rng, device)
            context, real = observed[:, :-1], observed[:, -1]
            memory = critic.memory(context)
            z, indices = prior.sample(args.batch_size, generator=latent_rng)
            with torch.no_grad():
                fake_d = generator(z, memory.detach())
            opt_d.zero_grad(set_to_none=True)
            d_adv = gan.d_loss(critic(real, memory), critic(fake_d, memory))
            d_reg = penalty(lambda x: critic(x, memory), real, fake_d, step=step)
            (d_adv + d_reg).backward()
            opt_d.step()
            # Rebuild context with the updated writer. No G gradient enters M/W.
            critic.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            with torch.no_grad():
                memory = critic.memory(context)
            fake = generator(z, memory)
            g_adv = gan.g_loss(critic(fake, memory), critic(real, memory).detach())
            g_reg = spread(prior(indices.unique()))
            (g_adv + g_reg).backward()
            opt_g.step()
            critic.requires_grad_(True)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = dict(event="train", variant=variant, step=step, d=d_adv.item(),
                           g=g_adv.item(), penalty=d_reg.item(), seconds=round(time.monotonic() - started, 2))
                if not all(math.isfinite(row[key]) for key in ("d", "g", "penalty")):
                    raise RuntimeError(f"non-finite training: {row}")
                metrics.write(json.dumps(row) + "\n")
                log(**row)
    generator.eval(), critic.eval(), prior.eval()
    result, artifacts = evaluate(generator, critic, prior, device, args.eval_batch)
    result.update(variant=variant, seconds=round(time.monotonic() - started, 2), config=config)
    torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                "prior": prior.state_dict(), "config": config}, directory / "model.pt")
    np.savez_compressed(directory / "particles.npz",
                        **{f"{name}_{key}": value for name, data in artifacts.items() for key, value in data.items()})
    (directory / "summary.json").write_text(json.dumps(result, indent=2))
    log(event="complete", variant=variant, circle_rmse=result["circle"]["rmse"],
        circle_energy=result["circle"]["energy"], shuffled_rmse=result["shuffle"]["rmse"],
        probe_rmse=result["probe_next_rmse"], seconds=result["seconds"])
    return result


def make_report(out, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Deterministic point predictors, scored on the same evaluation law.
    baselines = {}
    eval_device = torch.device(results[0]["config"]["device"])
    observed, clean = circles(results[0]["config"]["eval_batch"], 17,
                              torch.Generator(device=eval_device).manual_seed(20260916), eval_device)
    for name, pred in (("last observation", observed[:, :-1]),
                       ("linear extrapolation", torch.cat((observed[:, :1], 2 * observed[:, 1:-1] - observed[:, :-2]), 1))):
        curve = (pred - clean[:, 1:]).square().sum(-1).mean(0).sqrt()
        baselines[name] = {"rmse": float(curve[3:12].mean()), "curves": curve.tolist()}
    (out / "baselines.json").write_text(json.dumps(baselines, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for result in results:
        for ax, scenario, metric in ((axes[0], "circle", "rmse"), (axes[1], "circle", "energy"),
                                      (axes[2], "reversal", "rmse")):
            ax.plot(range(1, 17), result[scenario]["curves"][metric], label=result["variant"])
    for name, baseline in baselines.items():
        axes[0].plot(range(1, 17), baseline["curves"], "--", label=name)
    for ax, title in zip(axes, ("Next-point centroid RMSE", "Conditional energy score (lower better)", "Unseen direction reversal")):
        ax.set(title=title, xlabel="Number of real observations")
        ax.grid(alpha=.2)
    axes[2].axvline(9, color="black", linestyle=":", label="first reversed target")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_from_context.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), squeeze=False)
    artists = []
    for ax, result in zip(axes[0], results):
        data = np.load(out / result["variant"] / "particles.npz")
        path, obs, clouds = data["reversal_clean"][0], data["reversal_observed"][0], data["reversal_clouds"][0]
        ax.plot(path[:, 0], path[:, 1], "--", color=".7", label="true path")
        seen, = ax.plot([], [], "k.-", label="observations")
        particles = ax.scatter([], [], s=14, alpha=.5, label="G particles")
        target = ax.scatter([], [], marker="*", s=120, color="red", label="next target")
        ax.set(xlim=(path[:, 0].min() - .6, path[:, 0].max() + .6),
               ylim=(path[:, 1].min() - .6, path[:, 1].max() + .6), aspect="equal", title=result["variant"])
        artists.append((seen, particles, target, obs, path, clouds))
    axes[0, 0].legend(fontsize=7)
    title = fig.suptitle("")
    def frame(index):
        for seen, particles, target, obs, path, clouds in artists:
            seen.set_data(obs[:index+1, 0], obs[:index+1, 1])
            particles.set_offsets(clouds[index])
            target.set_offsets(path[index+1:index+2])
        title.set_text(f"Frozen networks; {index+1} real observations; " + ("direction reversed" if index >= 8 else "forward"))
    animation = FuncAnimation(fig, frame, frames=16, interval=500)
    animation.save(out / "particles.gif", writer=PillowWriter(fps=2))
    frame(7)
    fig.savefig(out / "particles.png", dpi=160)
    plt.close(fig)
    ranking = sorted(results, key=lambda r: r["circle"]["energy"])
    lines = ["# Circle memory scout", "", "Frozen-network evaluation on fresh episodes. Lower is better. "
             "Scores average prefixes 4–12. One training seed; variants change mechanisms, not seeds.", "",
             "| Variant | Energy score | Centroid RMSE | Shuffled-memory RMSE | Linear probe RMSE |", "|---|---:|---:|---:|---:|"]
    for r in ranking:
        lines.append(f"| {r['variant']} | {r['circle']['energy']:.4f} | {r['circle']['rmse']:.4f} | {r['shuffle']['rmse']:.4f} | {r['probe_next_rmse']:.4f} |")
    lines += ["", "Deterministic reference RMSE: " + "; ".join(f"{k}: {v['rmse']:.4f}" for k, v in baselines.items()),
              "", "![Context curves](learning_from_context.png)", "", "![Fixed latent particles](particles.gif)", "",
              "See config.json, metrics.jsonl, summary.json, model.pt and particles.npz in each variant directory."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/memory_path/scout"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.log_every) < 1 or args.eval_batch < 4:
        parser.error("positive steps/batch/log interval and eval-batch >= 4 required")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "experiment.log").exists():
        parser.error("output already contains experiment.log; choose a fresh --out")
    torch.set_num_threads(1)
    (args.out / "source.py").write_text(Path(__file__).read_text())
    provenance = {"torch": torch.__version__, "python": sys.version, "argv": sys.argv,
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with (args.out / "experiment.log").open("w", buffering=1) as stream:
        def log(**row):
            line = json.dumps(row)
            stream.write(line + "\n")
            print(line, flush=True)
        results = [train(variant, args, log) for variant in args.variants]
        make_report(args.out, results)
        log(event="suite_complete", report=str(args.out / "README.md"))


if __name__ == "__main__":
    main()
