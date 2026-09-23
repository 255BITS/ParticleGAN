"""Regenerate the README animation from one run of the public GAN defaults.

Run from the repository root:
    python -u reports/readme-100gaussians/generate.py > /tmp/readme-image.log 2>&1
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
    recipe = get_recipe("gan")
    out = Path(__file__).resolve().parent
    scratch = Path("/tmp/particlegan-develop-qc/readme-image")
    scratch.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("grid_example", ROOT / "examples/100gaussians.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    fixed_real = sample_100gaussians(4096, device,
        generator=torch.Generator(device=device).manual_seed(seed + 1)).cpu().numpy()
    coords = torch.arange(10, device=device) - 4.5
    centers = torch.cartesian_prod(coords, coords)
    frames, history = [], []
    print(json.dumps({"seed": seed, "recipe": recipe.to_dict(), "device": str(device)}), flush=True)

    @torch.no_grad()
    def snapshot(step, g, prior, ema_g, ema_prior, train_seconds):
        ema_g.eval()
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
        fixed_z, _ = ema_prior.sample(4096, fixed_first_n=True)
        fake = ema_g(fixed_z).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), facecolor="#f8fafc")
        for ax, points, title, color in zip(axes, (fixed_real, fake),
                ("Target distribution", "Generated samples · EMA"), ("#64748b", "#087f8c")):
            ax.set_facecolor("white")
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=.5, color=color, rasterized=True)
            ax.set(xlim=(-5.1, 5.1), ylim=(-5.1, 5.1), title=title, aspect="equal")
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([-4, -2, 0, 2, 4])
            ax.tick_params(colors="#64748b", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")
        fig.suptitle("100 Gaussians · learned particle prior", fontsize=16, color="#0f172a", y=.99)
        fig.text(.5, .065,
                 f"{step:,} / {recipe.total_steps:,} updates     |     "
                 f"{row['modes']}/100 modes     |     {row['hq']:.1%} within 3σ",
                 ha="center", fontsize=11, color="#0f172a")
        fig.text(.5, .015, "get_recipe('gan') · seed 1234 · 20,000 evaluation samples · 4,096 displayed",
                 ha="center", fontsize=8, color="#475569")
        fig.tight_layout(rect=(0, .09, 1, .94))
        path = scratch / f"frame_{step:06d}.png"
        fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(Image.open(path).convert("RGB"))
        if step == recipe.total_steps:
            toy = GaussianGrid(device, .03, 1)
            labels = torch.zeros(len(samples), device=device, dtype=torch.long)
            real = sample_100gaussians(len(samples), device,
                generator=torch.Generator(device=device).manual_seed(seed + 1000))
            row["distribution_metrics"] = grid_metrics(samples, labels, toy, real)
            row["unique_outputs"] = len(torch.unique(samples, dim=0))
            np.savez_compressed(scratch / "final_samples.npz", fake=samples.cpu().numpy(), real=real.cpu().numpy())

    result = example.train(
        use_training_api=True, seed=seed, device_str=str(device),
        out_dir=str(scratch), log_interval=500, save_plots=False,
        metric_callback=snapshot, metric_interval=500, return_details=True,
    )
    frames[0].save(ROOT / "100gaussians.gif", save_all=True, append_images=frames[1:],
                   duration=[450] * (len(frames) - 1) + [3000], loop=0, optimize=True)
    sources = ["particlegan/recipes.py", "particlegan/training.py", "examples/100gaussians.py",
               "lib/toy_models.py", "reports/readme-100gaussians/generate.py"]
    summary = dict(seed=seed, recipe=recipe.to_dict(), generator="SimpleMLPGenerator",
                   discriminator="SimpleMLPDiscriminator(fourier=2)", evaluation="EMA",
                   history=history, train_seconds=result["train_seconds"],
                   git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
                   torch=torch.__version__, device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   limitations="One illustrative run, not a seed study. Finite particle support; coverage and HQ do not establish Gaussian calibration.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": True, "final": history[-1], "image": "100gaussians.gif"}), flush=True)


if __name__ == "__main__":
    main()
