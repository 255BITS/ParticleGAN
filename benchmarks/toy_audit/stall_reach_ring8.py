"""Public-API toy: the 8-mode ring through GANTrainer with the stall-reach critic read.

Question: does the generator still acquire and hold the 8-mode ring when it
reads the critic through ``generator_critic_read="stall_reach"`` (closed PR
#107) on the current BCAP direction-blend leader? The gate is declared before
training: at the final update, ``modes >= 7`` and ``hq >= 0.90`` on 4096 clean
live samples (HQ = within 3 sigma of a ring center). Width and stall counts are
diagnostics, not gates.

Run: ``python -m benchmarks.toy_audit.stall_reach_ring8 --output runs/stall-reach-ring8``
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from benchmarks.legacy.locked_shared import LOCKED_SHARED
from benchmarks.locked_shared.mode_hold import SIGMA, diversity, ring_means, sample_ring
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_

ROOT = Path(__file__).resolve().parents[2]
LEADER = ROOT / "configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json"
STEPS, BATCH, Z_DIM, EVAL_N, EVERY = 1200, 128, 4, 4096, 50
GATE = dict(modes=(">=", 7), hq=(">=", 0.90))


def leader_recipe(read="stall_reach", steps=STEPS):
    card = json.loads(LEADER.read_text())
    overrides = dict(card["recipe_overrides"])
    overrides.pop("name")
    return get_recipe(card["recipe_preset"], **overrides, generator_critic_read=read, total_steps=steps,
                      batch_size=BATCH, z_dim=Z_DIM, num_particles=LOCKED_SHARED.n_particles)


def passed(row):
    return row["modes"] >= GATE["modes"][1] and row["hq"] >= GATE["hq"][1]


def train(*, read="stall_reach", steps=STEPS, seed=0, frames=None):
    recipe = leader_recipe(read, steps)
    generator = deterministic_orthogonal_(SimpleMLPGenerator(Z_DIM, 96, 3, 2), seed=seed)
    critic = deterministic_orthogonal_(SimpleMLPDiscriminator(2, 96, 3, 3), seed=seed + 1)
    trainer = GANTrainer(recipe, generator, critic, seed=seed)
    means = ring_means()
    data = torch.Generator().manual_seed(seed + 100)
    curve = []
    for step in range(1, steps + 1):
        update = trainer.step(sample_ring(means, BATCH, SIGMA, data), collect_stats=True)
        if step % EVERY == 0:
            samples = trainer.sample(EVAL_N, generator=torch.Generator().manual_seed(seed + 9))
            row = dict(step=step, **diversity(samples, means), **update.get("critic_read_stats", {}))
            row["passed"] = passed(row)
            curve.append(row)
            if frames is not None:
                frames.append((step, samples[:1024].clone(), row))
    read_summary = trainer.critic_read.summary() if read != "none" else None
    return dict(recipe=asdict(recipe), curve=curve, final=curve[-1], passed=curve[-1]["passed"],
                gate={k: list(v) for k, v in GATE.items()}, critic_read=read_summary)


def render_gif(frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    means = ring_means()
    images = []
    for step, samples, row in frames:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), dpi=80)
        target = sample_ring(means, 1024, SIGMA, torch.Generator().manual_seed(7))
        for ax, points, title in ((axes[0], target, "target: 8-mode ring"),
                                  (axes[1], samples, f"stall_reach G, update {step}")):
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=.4)
            ax.scatter(means[:, 0], means[:, 1], s=18, c="red", marker="x")
            ax.set_xlim(-3, 3), ax.set_ylim(-3, 3), ax.set_aspect("equal"), ax.set_title(title, fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
        status = "PASS" if row["passed"] else "FAIL"
        fig.suptitle(f"modes {row['modes']}/8 (>=7), HQ {row['hq']:.3f} (>=.90): {status}; "
                     f"width {row.get('width', 0):.3f}{' stalled' if row.get('stalled') else ''}", fontsize=8)
        fig.canvas.draw()
        images.append(Image.frombytes("RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())).convert("P"))
        plt.close(fig)
    images[0].save(path, save_all=True, append_images=images[1:], duration=200, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    with torch.autograd.set_multithreading_enabled(False):
        result = train(frames=frames)
    (args.output / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k != "recipe"}, indent=1))
    render_gif(frames, args.gif or args.output / "stall-reach-ring8.gif")
    print(json.dumps(dict(event="stall_reach_ring8", passed=result["passed"], final=result["final"],
                          critic_read=result["critic_read"])), flush=True)


if __name__ == "__main__":
    main()
