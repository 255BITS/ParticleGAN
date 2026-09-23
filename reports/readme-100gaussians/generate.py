"""Render live/EMA 100-Gaussian GIFs from one run of the public GAN defaults."""
import argparse
import math
import time
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from particlegan import get_recipe
from lib.denoising_toy import GaussianGrid, grid_metrics
from lib.toy_models import sample_100gaussians


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=get_recipe("gan").total_steps)
    parser.add_argument("--views", nargs="+", choices=("live", "ema"), default=["live", "ema"])
    parser.add_argument("--frame-every", type=int, default=25, help="Updates between frames; always include the final step.")
    parser.add_argument("--fps", type=int, default=25, help="Must divide 100: GIF durations use 10 ms units.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/100gaussians")
    args = parser.parse_args()
    if args.steps <= 0 or args.frame_every <= 0:
        parser.error("--steps and --frame-every must be positive")
    if args.fps <= 0 or 100 % args.fps:
        parser.error("--fps must be one of 1, 2, 4, 5, 10, 20, 25, 50, 100 (exact GIF timing)")
    if len(set(args.views)) != len(args.views):
        parser.error("--views must not contain duplicates")
    args.output_dir = args.output_dir.resolve()
    outputs = ["summary.json", "metrics.jsonl", "final-models.pt"] + [f"100gaussians-{v}.gif" for v in args.views]
    if any((args.output_dir / name).exists() for name in outputs):
        parser.error("output files already exist; choose a fresh --output-dir")
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1234
    recipe = get_recipe("gan", total_steps=args.steps)
    out = scratch = args.output_dir
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "metrics.jsonl").write_text("")
    for view in args.views:
        (scratch / view).mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("grid_example", ROOT / "examples/100gaussians.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    fixed_real = sample_100gaussians(4096, device,
        generator=torch.Generator(device=device).manual_seed(seed + 1)).cpu().numpy()
    coords = torch.arange(10, device=device) - 4.5
    centers = torch.cartesian_prod(coords, coords)
    frames = {view: [] for view in args.views}
    histories = {view: [] for view in args.views}
    callback_every = math.gcd(args.frame_every, 500)
    options = {**vars(args), "output_dir": str(out)}
    sources = ["particlegan/recipes.py", "particlegan/training.py", "examples/100gaussians.py",
               "lib/toy_models.py", "lib/denoising_toy.py", "lib/toy_metrics.py",
               "reports/readme-100gaussians/generate.py"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(json.dumps({"seed": seed, "recipe": recipe.to_dict(), "device": str(device), "options": options}), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.1), facecolor="#f8fafc")
    fake_scatter = None
    for ax, points, title, color in zip(axes, (fixed_real, np.empty((0, 2))),
            ("Target distribution", "Generated samples · EMA"), ("#64748b", "#087f8c")):
        ax.set_facecolor("white")
        artist = ax.scatter(points[:, 0], points[:, 1], s=2, alpha=.5, color=color, rasterized=True)
        if ax is axes[1]:
            fake_scatter = artist
        ax.set(xlim=(-5.1, 5.1), ylim=(-5.1, 5.1), title=title, aspect="equal")
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
    fig.suptitle("100 Gaussians · learned particle prior", fontsize=16, color="#0f172a", y=.99)
    step_label = fig.text(.5, .081, "", ha="center", fontsize=11, color="#0f172a")
    metric_label = fig.text(.5, .046, "", ha="center", fontsize=10, color="#0f172a")
    fig.text(.5, .012, f"GAN defaults · {recipe.total_steps:,} updates · seed 1234 · 20,000 evaluated · 4,096 displayed",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .12, 1, .94))

    @torch.no_grad()
    def snapshot_view(step, generator, particle_prior, train_seconds, view):
        history = histories[view]
        generator.eval()
        if not history or step % 500 == 0 or step == recipe.total_steps:
            z, _ = particle_prior.sample(20000,
                generator=torch.Generator(device=device).manual_seed(seed + 999))
            samples = generator(z)
            distances, nearest = torch.cdist(samples, centers).min(1)
            hq = distances <= .09
            counts = torch.bincount(nearest[hq], minlength=100)
            row = dict(view=view, step=step, modes=int((counts >= 10).sum()),
                       hq=float(hq.float().mean()), train_seconds=train_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
            with (scratch / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        row = history[-1]
        if step % args.frame_every == 0 or step == recipe.total_steps:
            fixed_z, _ = particle_prior.sample(4096, fixed_first_n=True)
            fake = generator(fixed_z).cpu().numpy()
            axes[1].set_title("Generated samples · " + ("EMA" if view == "ema" else "live"))
            fake_scatter.set_offsets(fake)
            step_label.set_text(f"{step:,} / {recipe.total_steps:,} updates")
            metric_label.set_text(f"{row['modes']}/100 modes · {row['hq']:.1%} within 3σ "
                                  f"(evaluated at {row['step']:,})")
            path = scratch / view / f"frame_{step:06d}.png"
            fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
            with Image.open(path) as frame:
                frames[view].append(frame.convert("RGB"))
        if step == recipe.total_steps:
            toy = GaussianGrid(device, .03, 1)
            labels = torch.zeros(len(samples), device=device, dtype=torch.long)
            real = sample_100gaussians(len(samples), device,
                generator=torch.Generator(device=device).manual_seed(seed + 1000))
            row["distribution_metrics"] = grid_metrics(samples, labels, toy, real)
            row["unique_outputs"] = len(torch.unique(samples, dim=0))
            np.savez_compressed(scratch / view / "final_samples.npz", fake=samples.cpu().numpy(), real=real.cpu().numpy())

    def snapshot(step, g, prior, ema_g, ema_prior, train_seconds):
        # The example isolates RNG and restores module modes around this observer.
        modules = {"live": (g, prior), "ema": (ema_g, ema_prior)}
        for view in args.views:
            snapshot_view(step, *modules[view], train_seconds, view)

    result = example.train(
        epochs=1, steps_per_epoch=recipe.total_steps,
        use_training_api=True, seed=seed, device_str=str(device),
        out_dir=str(scratch), log_interval=500, save_plots=False,
        metric_callback=snapshot, metric_interval=callback_every, return_details=True,
    )
    plt.close(fig)
    # One palette across every view and frame avoids color flicker.
    all_frames = [frame for view_frames in frames.values() for frame in view_frames]
    palette_strip = Image.new("RGB", (90, 51 * len(all_frames)))
    for index, frame in enumerate(all_frames):
        palette_strip.paste(frame.resize((90, 51)), (0, 51 * index))
    palette = palette_strip.quantize(colors=128)
    for view, view_frames in frames.items():
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in view_frames]
        indexed[0].save(out / f"100gaussians-{view}.gif", save_all=True, append_images=indexed[1:],
                        duration=[1000 // args.fps] * (len(indexed) - 1) + [1000], loop=0, optimize=True)
    torch.save({key: result[key].state_dict() for key in ("G", "D", "prior", "ema_G", "ema_prior")},
               scratch / "final-models.pt")
    summary = dict(seed=seed, recipe=recipe.to_dict(), options=options,
                   generator="SimpleMLPGenerator", discriminator="SimpleMLPDiscriminator(fourier=2)",
                   views={view: dict(history=histories[view], final=histories[view][-1],
                                     gif=f"100gaussians-{view}.gif", frames=len(frames[view])) for view in args.views},
                   train_seconds=result["train_seconds"], total_seconds=time.perf_counter() - started,
                   git_commit=git_commit, source_sha256=source_hashes,
                   animation=dict(first_step=min(args.frame_every, recipe.total_steps), every_steps=args.frame_every,
                                  frame_ms=1000 // args.fps, final_hold_ms=1000, size=[900, 510], shared_palette_colors=128),
                   evaluation_samples=20000, visualization_samples=4096, fixed_particle_indices=[0, 4095],
                   evaluation_every_steps=500, first_evaluation_step=min(callback_every, recipe.total_steps),
                   anneal_start_step=recipe.total_steps * recipe.lr_anneal_start,
                   torch=torch.__version__, device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   limitations="Single fixed-seed illustration with finite particle support; coverage and HQ do not establish Gaussian calibration.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": True, "output_dir": str(out),
                      "final": {view: history[-1] for view, history in histories.items()}}), flush=True)



if __name__ == "__main__":
    main()
