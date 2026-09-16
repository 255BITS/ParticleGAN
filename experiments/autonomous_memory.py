"""Expert-free circle rollouts: one state and one fixed particle per batch row.

GPU scout: .venv/bin/python -u experiments/autonomous_memory.py --out runs/memory_path/autonomous_2k
Tail: tail -f runs/memory_path/autonomous_2k/experiment.log
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.memory_path import FastMemory, Generator, circles, mlp
from particlegan import get_recipe, learning_rate_scale


VARIANTS = ("shared", "frozen_writer", "no_memory")


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    module.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


def rollout(generator, writer, z, steps, *, use_memory=True):
    """Cold start; no real data or critic scores. Writer weights never get G grads.

    State remains differentiable through time to earlier generated points.
    Every row has its own memory and reuses its supplied z at every step.
    Returns [batch, time, 2] positions and state after the last generated write.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    memory = z.new_zeros(len(z), 8, 4)
    path = []
    with frozen(writer):
        for _ in range(steps):
            point = generator(z, memory.flatten(1))
            path.append(point)
            if use_memory:
                memory = writer.write(memory, point)
    return torch.stack(path, 1), memory


class SequenceCritic(nn.Module):
    """Identical causal feature construction for real and generated sequences.

    Each candidate is paired with the memory BEFORE its own write. A temporal
    head sees the ordered sequence so stationary points cannot pass by matching
    only the marginal coordinate distribution. Writer belongs to D's optimizer.
    """
    def __init__(self, length, variant):
        super().__init__()
        self.length, self.variant = length, variant
        self.writer = FastMemory()
        if variant in ("frozen_writer", "no_memory"):
            self.writer.requires_grad_(False)
        self.head = mlp(length * 34, 1, width=128)

    def forward(self, path):
        if path.shape[1:] != (self.length, 2):
            raise ValueError(f"expected [batch, {self.length}, 2]")
        memory = path.new_zeros(len(path), 8, 4)
        features = []
        for point in path.unbind(1):
            features.append(torch.cat((point, memory.flatten(1)), -1))
            if self.variant != "no_memory":
                memory = self.writer.write(memory, point)
        return self.head(torch.stack(features, 1).flatten(1)).squeeze(-1)


def trajectory_metrics(paths, fit_steps=32):
    """Fit the early circle once, then measure long-run drift against that fit.

    Unpaired generated trajectories cannot use next-point RMSE against arbitrary
    real trajectories. These are dynamics diagnostics, not a calibrated distance
    between trajectory distributions. Degenerate/stationary paths cannot pass.
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[-1] != 2 or paths.shape[1] < 8:
        raise ValueError("expected [batch, time >= 8, 2]")
    if not np.isfinite(paths).all():
        raise ValueError("non-finite generated trajectory")
    fit_steps = min(fit_steps, paths.shape[1] // 2)
    values, good = [], []
    for path in paths:
        initial = path[:fit_steps]
        origin = initial.mean(0)
        centered = initial - origin
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular[-1] < 1e-5 or singular[0] < .05:
            good.append(False)
            continue
        a = np.column_stack((2 * centered, np.ones(len(initial))))
        fitted = np.linalg.lstsq(a, np.square(centered).sum(1), rcond=None)[0]
        center = origin + fitted[:2]
        radius = np.sqrt(max(0., fitted[2] + np.square(fitted[:2]).sum()))
        if radius < 1e-5:
            good.append(False)
            continue
        offsets = path - center
        radii = np.linalg.norm(offsets, axis=1)
        radial_error = np.sqrt(np.mean(np.square(radii - radius))) / radius
        late_drift = abs(radii[-len(path)//4:].mean() / radius - 1)
        cross = offsets[:-1, 0] * offsets[1:, 1] - offsets[:-1, 1] * offsets[1:, 0]
        dot = (offsets[:-1] * offsets[1:]).sum(1)
        angular = np.arctan2(cross, dot)
        speed = np.abs(angular).mean()
        consistency = abs(np.sign(angular).mean())
        values.append([radius, radial_error, late_drift, speed, consistency, angular.mean()])
        good.append(bool(.5 <= radius <= 1.6 and radial_error < .1 and late_drift < .2
                         and .08 <= speed <= .45 and consistency > .95))
    names = ("radius", "relative_radial_rmse", "late_radius_drift", "angular_speed",
             "direction_consistency", "signed_angular_speed")
    result = {name: float(np.mean(np.asarray(values)[:, i])) if values else None
              for i, name in enumerate(names)}
    result.update(circle_like_fraction=float(np.mean(good)), valid_fit_fraction=len(values) / len(paths),
                  stationary_fraction=float(np.mean(np.linalg.norm(np.diff(paths, axis=1), axis=-1).mean(1) < .01)),
                  mean_step_length=float(np.linalg.norm(np.diff(paths, axis=1), axis=-1).mean()),
                  initial_position_spread=float(np.sqrt(paths[:, 0].var(0).sum())),
                  radius_std=float(np.std(np.asarray(values)[:, 0])) if values else None)
    return result


@torch.no_grad()
def evaluate(generator, critic, prior, args):
    z = prior(torch.arange(args.eval_batch, device=args.device))
    generated, _ = rollout(generator, critic.writer, z, args.eval_steps,
                           use_memory=critic.variant != "no_memory")
    observed, clean = circles(args.eval_batch, args.eval_steps,
                              torch.Generator(device=args.device).manual_seed(20260916), args.device)
    paths = {"generated": generated.cpu().numpy(), "real_noisy": observed.cpu().numpy(),
             "real_clean": clean.cpu().numpy()}
    # Real references are used only AFTER generation, to assess the diagnostics.
    return {name: trajectory_metrics(path) for name, path in paths.items()}, paths


def train(variant, args, log):
    torch.manual_seed(42)
    recipe = get_recipe(total_steps=args.steps, batch_size=args.batch_size, num_particles=512)
    generator = Generator("no_memory" if variant == "no_memory" else "shared").to(args.device)
    critic = SequenceCritic(args.train_length, variant).to(args.device)
    prior = recipe.make_prior().to(args.device)
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior)
    gan, penalty, spread = recipe.make_loss(), recipe.make_gradient_penalty(), recipe.make_prior_regularizer()
    rates = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    data_rng = torch.Generator(device=args.device).manual_seed(31415)
    latent_rng = torch.Generator(device=args.device).manual_seed(27182)
    directory = args.out / variant
    directory.mkdir()
    config = {**vars(args), "out": str(args.out), "variant": variant, "recipe": recipe.to_dict(),
              "seed": 42, "particle_policy": "fixed_per_trajectory", "initial_state": "zeros",
              "real_prefix": False, "writer_training": "D only; both real and detached fake sequence scoring",
              "g_state_gradient": "full rollout; writer weights frozen", "ema": False,
              "gradient_clipping": None}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    log(event="start", variant=variant, config=config)
    started = time.monotonic()
    with (directory / "metrics.jsonl").open("w", buffering=1) as stream:
        for step in range(1, args.steps + 1):
            scale = learning_rate_scale(step - 1, args.steps, recipe.lr_anneal_start, recipe.lr_floor)
            for opt, base in zip((opt_g, opt_d), rates):
                for group, rate in zip(opt.param_groups, base):
                    group["lr"] = rate * scale
            real, _ = circles(args.batch_size, args.train_length, data_rng, args.device)
            z, indices = prior.sample(args.batch_size, generator=latent_rng)
            with torch.no_grad():
                fake, _ = rollout(generator, critic.writer, z, args.train_length, use_memory=variant != "no_memory")
            opt_d.zero_grad(set_to_none=True)
            d_adv = gan.d_loss(critic(real), critic(fake))
            d_reg = penalty(critic, real, fake, step=step)
            (d_adv + d_reg).backward()
            opt_d.step()
            opt_d.zero_grad(set_to_none=True)
            with frozen(critic):
                opt_g.zero_grad(set_to_none=True)
                fake, _ = rollout(generator, critic.writer, z, args.train_length, use_memory=variant != "no_memory")
                with torch.no_grad():
                    real_scores = critic(real)
                g_adv = gan.g_loss(critic(fake), real_scores)
                (g_adv + spread(prior(indices.unique()))).backward()
                opt_g.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = dict(event="train", variant=variant, step=step, d=d_adv.item(), g=g_adv.item(),
                           penalty=d_reg.item(),
                           seconds=round(time.monotonic() - started, 2))
                if not all(math.isfinite(row[key]) for key in ("d", "g", "penalty")):
                    raise RuntimeError(f"non-finite training: {row}")
                stream.write(json.dumps(row) + "\n")
                log(**row)
    generator.eval(), critic.eval(), prior.eval()
    metrics, paths = evaluate(generator, critic, prior, args)
    result = dict(variant=variant, config=config, metrics=metrics, seconds=round(time.monotonic()-started, 2))
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez_compressed(directory / "trajectories.npz", **paths)
    torch.save({"generator": generator.state_dict(), "writer": critic.writer.state_dict(),
                "critic": critic.state_dict(), "prior": prior.state_dict(), "config": config}, directory / "model.pt")
    log(event="complete", variant=variant, metrics=metrics["generated"], seconds=result["seconds"])
    return result


def report(out, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 4, figsize=(12, 3 * len(results)), squeeze=False)
    for row, result in zip(axes, results):
        with np.load(out / result["variant"] / "trajectories.npz") as saved:
            paths = saved["generated"][:4]
        for ax, path in zip(row, paths):
            ax.plot(path[:, 0], path[:, 1], linewidth=1)
            ax.scatter(*path[0], color="green", s=25, label="start")
            ax.scatter(*path[-1], color="red", s=25, label="end")
            ax.set(title=result["variant"], aspect="equal")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Autonomous cold starts: first four trajectories, fixed z, no real prefix")
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=140)
    plt.close(fig)
    lines = ["# Autonomous memory rollout", "", "One trajectory and memory per batch row; fixed particle; empty initial memory.",
             "", "Geometry fractions are heuristic diagnostics. Inspect real-reference scores and diversity too.", "",
             "| Variant | Circle-like fraction ↑ | Stationary fraction ↓ | Relative radial RMSE ↓ |", "|---|---:|---:|---:|"]
    for result in sorted(results, key=lambda r: -r["metrics"]["generated"]["circle_like_fraction"]):
        m = result["metrics"]["generated"]
        radial = "n/a" if m["relative_radial_rmse"] is None else f"{m['relative_radial_rmse']:.4f}"
        lines.append(f"| {result['variant']} | {m['circle_like_fraction']:.3f} | {m['stationary_fraction']:.3f} | {radial} |")
    if results[0]["config"]["steps"] < 1000:
        lines += ["", "Short smoke run: verifies execution only; do not interpret this as a trained-model comparison."]
    lines += ["", "![Autonomous paths](trajectories.png)", "", "Next: compare learned and frozen writers; inspect long-horizon stability before increasing task complexity."]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/memory_path/autonomous_2k"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-length", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.log_every) < 1 or args.train_length < 2 or args.eval_steps < 8 or not 4 <= args.eval_batch <= 512:
        parser.error("positive steps/batch/log interval, train-length >= 2, eval-steps >= 8, eval-batch in [4,512] required")
    if len(set(args.variants)) != len(args.variants):
        parser.error("duplicate variants")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; explicitly select --device cpu to run on CPU")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for filename in ("autonomous_memory.py", "memory_path.py"):
        (args.out / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    provenance = {"torch": torch.__version__, "python": sys.version, "argv": sys.argv,
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "device": args.device,
                  "gpu": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with (args.out / "experiment.log").open("w", buffering=1) as stream:
        def log(**row):
            line = json.dumps(row, allow_nan=False)
            stream.write(line + "\n")
            print(line, flush=True)
        results = [train(variant, args, log) for variant in args.variants]
        report(args.out, results)
        log(event="suite_complete", report=str(args.out / "README.md"))


if __name__ == "__main__":
    main()
