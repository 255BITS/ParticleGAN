"""Render the README 100-Gaussian GIF from one run of the public GAN defaults.

Training is exactly ``GANTrainer(get_recipe("gan"), G, D, seed=1234)`` with no
recipe, optimizer or penalty overrides; ``--steps`` (default: the recipe's
budget) is the only training option. Everything else here is visualization.

Frames are placed on a smooth power-law time warp (``--frame-power``): step
``T * (i / N) ** p`` for frame ``i`` of ``N``, so early training (where the
motion is) gets dense frames and the cadence changes continuously, never in
jumps. ``--frame-power 1`` gives evenly spaced frames.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from particlegan import GANTrainer, get_recipe
from lib.denoising_toy import GaussianGrid, grid_metrics
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians

SEED = 1234
SIZE = (480, 520)  # pixels; readable at README width


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=get_recipe("gan").total_steps)
    parser.add_argument("--views", nargs="+", choices=("live", "ema"), default=["live", "ema"])
    parser.add_argument("--frames", type=int, default=250, help="Number of training frames (before the hold).")
    parser.add_argument("--frame-power", type=float, default=2.0,
                        help="Time-warp exponent; >1 samples early training more densely, 1 is uniform.")
    parser.add_argument("--fps", type=int, default=25, help="Must divide 100: GIF durations use 10 ms units.")
    parser.add_argument("--hold-seconds", type=float, default=2.5, help="Hold on the final frame before looping.")
    parser.add_argument("--colors", type=int, default=64, help="Shared GIF palette size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/100gaussians")
    args = parser.parse_args()
    if args.steps <= 0 or args.frames <= 0 or args.frame_power <= 0 or args.hold_seconds < 0:
        parser.error("--steps, --frames, --frame-power must be positive and --hold-seconds nonnegative")
    if args.fps <= 0 or 100 % args.fps:
        parser.error("--fps must be one of 1, 2, 4, 5, 10, 20, 25, 50, 100 (exact GIF timing)")
    if len(set(args.views)) != len(args.views):
        parser.error("--views must not contain duplicates")
    args.output_dir = args.output_dir.resolve()
    outputs = ["summary.json", "metrics.jsonl", "final-models.pt"] + [f"100gaussians-{v}.gif" for v in args.views]
    if any((args.output_dir / name).exists() for name in outputs):
        parser.error("output files already exist; choose a fresh --output-dir")
    return args


def frame_steps(total, frames, power):
    """Strictly increasing steps on a smooth power-law warp ending at ``total``."""
    steps, previous = [], 0
    for i in range(1, frames + 1):
        step = max(previous + 1, round(total * (i / frames) ** power))
        steps.append(min(step, total))
        previous = steps[-1]
    return sorted(set(steps))


def main():
    args = parse_args()
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    recipe = get_recipe("gan", total_steps=args.steps) if args.steps != get_recipe("gan").total_steps else get_recipe("gan")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.jsonl").write_text("")
    options = {**vars(args), "output_dir": str(out)}
    sources = ["particlegan/recipes.py", "particlegan/training.py", "particlegan/k3p.py",
               "lib/toy_models.py", "lib/denoising_toy.py", "reports/readme-100gaussians/generate.py"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in sources if (ROOT / name).exists()}
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    schedule = frame_steps(recipe.total_steps, args.frames, args.frame_power)
    print(json.dumps({"seed": SEED, "recipe": recipe.to_dict(), "device": str(device), "options": options,
                      "frames": len(schedule), "first_frame_steps": schedule[:8]}, default=str), flush=True)

    # Models: the same small MLPs and init as examples/100gaussians.py.
    torch.manual_seed(SEED)
    G = SimpleMLPGenerator(z_dim=recipe.z_dim).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, fourier=2).to(device)
    for m in list(G.modules()) + list(D.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    trainer = GANTrainer(recipe, G, D, seed=SEED)  # package defaults, no overrides
    data_gen = torch.Generator(device=device).manual_seed(SEED)

    coords = torch.arange(10, device=device) - 4.5
    centers = torch.cartesian_prod(coords, coords)
    centers_np = centers.cpu().numpy()

    fig, ax = plt.subplots(figsize=(SIZE[0] / 100, SIZE[1] / 100), facecolor="white")
    extent = 5.6
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.scatter(centers_np[:, 0], centers_np[:, 1], s=40, facecolors="none", edgecolors="#94a3b8", linewidths=.6)
    points = ax.scatter(np.zeros(4096), np.zeros(4096), s=1.4, alpha=.5, color="#087f8c", rasterized=True)
    fig.suptitle("ParticleGAN · 100 Gaussians", fontsize=13, color="#0f172a", y=.975)
    subtitle = ax.set_title("", fontsize=9.5, color="#334155")
    footer = fig.text(.5, .015, "", ha="center", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, .03, 1, .96))

    frames = {view: [] for view in args.views}
    histories = {view: [] for view in args.views}
    final_eval = {}

    @torch.no_grad()
    def snapshot(step, view, generator, prior):
        generator.eval()
        prior.eval()
        z, _ = prior.sample(20000, generator=torch.Generator(device=device).manual_seed(SEED + 999))
        samples = generator(z)
        distances, nearest = torch.cdist(samples, centers).min(1)
        hq = distances <= .09
        counts = torch.bincount(nearest[hq], minlength=100)
        row = dict(view=view, step=step, modes=int((counts >= 10).sum()), hq=float(hq.float().mean()))
        histories[view].append(row)
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        fixed_z, _ = prior.sample(4096, fixed_first_n=True)
        points.set_offsets(generator(fixed_z).cpu().numpy())
        subtitle.set_text(f"step {step:,}/{recipe.total_steps:,} · {row['modes']}/100 modes · "
                          f"{row['hq']:.1%} within 3σ")
        footer.set_text(f"default get_recipe(\"gan\") · {'live' if view == 'live' else 'EMA'} weights")
        fig.canvas.draw()
        frames[view].append(Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                             fig.canvas.buffer_rgba()).convert("RGB"))
        if step == recipe.total_steps:
            final_eval[view] = samples

    wanted = set(schedule)
    devices = [device.index] if device.type == "cuda" else []
    train_seconds, last_log = 0.0, time.perf_counter()
    for step in range(1, recipe.total_steps + 1):
        tick = time.perf_counter()
        stats = trainer.step(sample_100gaussians(recipe.batch_size, device, generator=data_gen),
                             generator_real=lambda: sample_100gaussians(recipe.batch_size, device, generator=data_gen))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - tick
        if step in wanted:
            modules = {"live": (trainer.G, trainer.prior), "ema": (trainer.ema_G, trainer.ema_prior)}
            with torch.random.fork_rng(devices=devices):
                for view in args.views:
                    flags = [(m, m.training) for m in modules[view]]
                    snapshot(step, view, *modules[view])
                    for m, flag in flags:
                        m.train(flag)
        if step % 250 == 0 or step == recipe.total_steps:
            rows = {view: histories[view][-1] for view in args.views}
            print(json.dumps({"step": step, "loss_d": round(float(stats["loss_d"]), 4),
                              "loss_gan": round(float(stats["loss_gan"]), 4), "train_s": round(train_seconds, 1),
                              "frames": len(frames[args.views[0]]),
                              **{f"{v}_modes": r["modes"] for v, r in rows.items()},
                              **{f"{v}_hq": round(r["hq"], 4) for v, r in rows.items()}}), flush=True)
    plt.close(fig)

    toy = GaussianGrid(device, .03, 1)
    real = sample_100gaussians(20000, device, generator=torch.Generator(device=device).manual_seed(SEED + 1000))
    labels = torch.zeros(20000, device=device, dtype=torch.long)
    for view, samples in final_eval.items():
        histories[view][-1]["distribution_metrics"] = grid_metrics(samples, labels, toy, real)
        histories[view][-1]["unique_outputs"] = len(torch.unique(samples, dim=0))

    # One palette across every view and frame avoids color flicker.
    # Build it from full-resolution frames so thin dark text keeps its colors.
    all_frames = [frame for view_frames in frames.values() for frame in view_frames]
    picks = all_frames[::max(1, len(all_frames) // 24)]
    strip = Image.new("RGB", (SIZE[0], SIZE[1] * len(picks)))
    for index, frame in enumerate(picks):
        strip.paste(frame, (0, SIZE[1] * index))
    palette = strip.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
    frame_ms, hold_ms = 1000 // args.fps, max(round(args.hold_seconds * 100) * 10, 1000 // args.fps)
    for view, view_frames in frames.items():
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in view_frames]
        indexed[0].save(out / f"100gaussians-{view}.gif", save_all=True, append_images=indexed[1:],
                        duration=[frame_ms] * (len(indexed) - 1) + [hold_ms], loop=0, optimize=True)
    torch.save({"G": trainer.G.state_dict(), "D": trainer.D.state_dict(), "prior": trainer.prior.state_dict(),
                "ema_G": trainer.ema_G.state_dict(), "ema_prior": trainer.ema_prior.state_dict()},
               out / "final-models.pt")
    summary = dict(seed=SEED, recipe=recipe.to_dict(), options=options, trainer="GANTrainer(recipe, G, D, seed=1234)",
                   generator="SimpleMLPGenerator", discriminator="SimpleMLPDiscriminator(fourier=2)",
                   views={view: dict(final=histories[view][-1], gif=f"100gaussians-{view}.gif",
                                     frames=len(frames[view]),
                                     bytes=(out / f"100gaussians-{view}.gif").stat().st_size) for view in args.views},
                   animation=dict(frame_steps=schedule, frame_ms=frame_ms, final_hold_ms=hold_ms, size=list(SIZE),
                                  shared_palette_colors=args.colors),
                   train_seconds=train_seconds, total_seconds=time.perf_counter() - started,
                   git_commit=git_commit, source_sha256=source_hashes,
                   evaluation_samples=20000, visualization_samples=4096, fixed_particle_indices=[0, 4095],
                   torch=torch.__version__, device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   limitations="Single fixed-seed illustration; metrics are noise-free generator outputs at each frame.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"complete": True, "output_dir": str(out), "train_seconds": round(train_seconds, 1),
                      "final": {view: histories[view][-1] for view in args.views}}, default=str), flush=True)


if __name__ == "__main__":
    main()
