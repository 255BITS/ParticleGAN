"""Regenerate the README animation with a longer budget and public GAN defaults.

Run from the repository root:
    python -u reports/readme-100gaussians/generate.py > /tmp/readme-image-20k.log 2>&1
"""
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


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1234
    recipe = get_recipe("gan", total_steps=20000)
    out = Path(__file__).resolve().parent
    scratch = Path("/tmp/particlegan-develop-qc/readme-image-20k")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "metrics.jsonl").write_text("")
    spec = importlib.util.spec_from_file_location("grid_example", ROOT / "examples/100gaussians.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    fixed_real = sample_100gaussians(4096, device,
        generator=torch.Generator(device=device).manual_seed(seed + 1)).cpu().numpy()
    coords = torch.arange(10, device=device) - 4.5
    centers = torch.cartesian_prod(coords, coords)
    frames, history = [], []
    sources = ["particlegan/recipes.py", "particlegan/training.py", "examples/100gaussians.py",
               "lib/toy_models.py", "lib/denoising_toy.py", "lib/toy_metrics.py",
               "reports/readme-100gaussians/generate.py"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(json.dumps({"seed": seed, "recipe": recipe.to_dict(), "device": str(device)}), flush=True)

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
    fig.text(.5, .012, "GAN defaults · 20k budget · seed 1234 · 20,000 evaluated · 4,096 displayed",
             ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .12, 1, .94))

    @torch.no_grad()
    def snapshot(step, g, prior, ema_g, ema_prior, train_seconds):
        ema_g.eval()
        if not history or step % 500 == 0 or step == recipe.total_steps:
            z, _ = ema_prior.sample(20000,
                generator=torch.Generator(device=device).manual_seed(seed + 999))
            samples = ema_g(z)
            distances, nearest = torch.cdist(samples, centers).min(1)
            hq = distances <= .09
            counts = torch.bincount(nearest[hq], minlength=100)
            row = dict(step=step, modes=int((counts >= 10).sum()),
                       hq=float(hq.float().mean()), train_seconds=train_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
            with (scratch / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        row = history[-1]
        fixed_z, _ = ema_prior.sample(4096, fixed_first_n=True)
        fake = ema_g(fixed_z).cpu().numpy()
        fake_scatter.set_offsets(fake)
        step_label.set_text(f"{step:,} / {recipe.total_steps:,} updates")
        metric_label.set_text(f"{row['modes']}/100 modes · {row['hq']:.1%} within 3σ "
                              f"(evaluated at {row['step']:,})")
        path = scratch / f"frame_{step:06d}.png"
        fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
        with Image.open(path) as frame:
            frames.append(frame.convert("RGB"))
        if step == recipe.total_steps:
            toy = GaussianGrid(device, .03, 1)
            labels = torch.zeros(len(samples), device=device, dtype=torch.long)
            real = sample_100gaussians(len(samples), device,
                generator=torch.Generator(device=device).manual_seed(seed + 1000))
            row["distribution_metrics"] = grid_metrics(samples, labels, toy, real)
            row["unique_outputs"] = len(torch.unique(samples, dim=0))
            np.savez_compressed(scratch / "final_samples.npz", fake=samples.cpu().numpy(), real=real.cpu().numpy())

    result = example.train(
        epochs=recipe.total_steps // 1000, steps_per_epoch=1000,
        use_training_api=True, seed=seed, device_str=str(device),
        out_dir=str(scratch), log_interval=500, save_plots=False,
        metric_callback=snapshot, metric_interval=100, return_details=True,
    )
    plt.close(fig)
    # A shared palette avoids color flicker and keeps this 200-frame asset compact.
    palette_strip = Image.new("RGB", (90, 51 * len(frames)))
    for index, frame in enumerate(frames):
        palette_strip.paste(frame.resize((90, 51)), (0, 51 * index))
    palette = palette_strip.quantize(colors=128)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(ROOT / "100gaussians.gif", save_all=True, append_images=indexed[1:],
                    duration=[40] * (len(frames) - 1) + [1000], loop=0, optimize=True)
    torch.save({key: result[key].state_dict() for key in ("G", "D", "prior", "ema_G", "ema_prior")},
               scratch / "final-models.pt")
    summary = dict(seed=seed, recipe=recipe.to_dict(), generator="SimpleMLPGenerator",
                   discriminator="SimpleMLPDiscriminator(fourier=2)", evaluation="EMA",
                   history=history, train_seconds=result["train_seconds"],
                   git_commit=git_commit, source_sha256=source_hashes,
                   animation=dict(frames=len(frames), first_step=100, every_steps=100,
                                  frame_ms=40, final_hold_ms=1000, size=[900, 510], colors=128),
                   evaluation_samples=20000, visualization_samples=4096,
                   evaluation_every_steps=500, first_evaluation_step=100,
                   budget_override=20000, anneal_start_step=12000,
                   previous_run="summary-7000.json",
                   torch=torch.__version__, device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   limitations="One illustrative run, not a seed study. Finite particle support; coverage and HQ do not establish Gaussian calibration.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": True, "final": history[-1], "image": "100gaussians.gif"}), flush=True)


if __name__ == "__main__":
    main()
