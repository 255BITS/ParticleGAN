"""Bounded public-API SIGReg integration demo; no Forge qualification.

Declared goal: shrink the fixed-axis Gaussian characteristic-function error of
an explicitly broader initial R2Normal location table while G/D keep training.
The generated output panel is observed evidence, not a density qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn

from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest
from particlegan.sigreg import sigreg_statistic

ROOT = Path(__file__).resolve().parents[3]
STEPS = 128
BOUNDS = {"fixed_axis_sigreg_max": 4., "relative_error_max": .1,
          "resume_state_mismatch_max": 0, "nonfinite_count_max": 0}


def build():
    leader = json.loads((ROOT / "configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json").read_text())
    recipe = {**leader["recipe_overrides"], "prior_regularizer": "sigreg", "prior_reg": .01,
              "num_particles": 128, "z_dim": 2, "batch_size": 64, "total_steps": STEPS}
    ctx = FormulationContext(recipe_preset="bcap", recipe_overrides=recipe, seed=0,
        prior={"kind": "mog", "sigma": .025, "standardize": False, "init_std": 2.},
        prior_contract="recipe_owned_v1")
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2)), component="generator")
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1)), component="discriminator")
    return ctx, ctx.build_trainer(g, d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/prior-sigreg/demo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ctx, trainer = build()
    data = torch.Generator().manual_seed(100)
    target = torch.randn(1024, 2, generator=torch.Generator().manual_seed(101))
    frames, observations = [], []
    resume_mismatch = None

    def observe(step):
        state = trainer.prior.z.detach().cpu()
        score = float(sigreg_statistic(state, torch.eye(2)))
        samples = trainer.sample(512, generator=torch.Generator().manual_seed(102)).cpu()
        observations.append({"step": step, "fixed_axis_sigreg": score})
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), dpi=90)
        for ax, points, title in zip(axes, (state, samples), ("Actual prior locations", "Actual API generated samples")):
            ax.scatter(target[:, 0], target[:, 1], s=3, alpha=.15, color="gray", label="Desired N(0,I)")
            ax.scatter(points[:, 0], points[:, 1], s=5, alpha=.5, color="tab:blue", label="Observed")
            ax.set(xlim=(-6, 6), ylim=(-6, 6), title=title)
            ax.set_aspect("equal")
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"Update {step}/{STEPS}; prior fixed-axis SIGReg {score:.3f} / ≤4.0", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
        plt.close(fig)
        print(json.dumps(observations[-1]), flush=True)

    oracle = float(sigreg_statistic(target, torch.eye(2)))
    destructive = float(sigreg_statistic(torch.zeros_like(target), torch.eye(2)))
    assert oracle <= BOUNDS["fixed_axis_sigreg_max"] < destructive
    observe(0)
    for step in range(1, STEPS + 1):
        real = torch.randn(64, 2, generator=data)
        if step == 65:
            saved = ctx.state_dict()
            _, restored = build()
            restored.load_state_dict(saved["trainer"])
            trainer.step(real)
            expected = trainer.state_dict()
            restored.step(real)
            resume_mismatch = int(state_digest(expected) != state_digest(restored.state_dict()))
        else:
            trainer.step(real)
        if step % 8 == 0:
            observe(step)
    initial, final = observations[0]["fixed_axis_sigreg"], observations[-1]["fixed_axis_sigreg"]
    metrics = {"fixed_axis_sigreg": final, "relative_error": final / initial,
               "resume_state_mismatch": resume_mismatch,
               "nonfinite_count": sum(int((~torch.isfinite(p)).sum()) for module in (trainer.G, trainer.D, trainer.prior) for p in module.parameters())}
    failed = [name for name, limit in BOUNDS.items() if metrics[name.removesuffix("_max")] > limit]
    media = args.output / "training.gif"
    frames[0].save(media, save_all=True, append_images=frames[1:], duration=180, loop=0)
    receipt = {"scope": "128-update public API feature integration; no Tier 1/Tier 2 credit",
               "protocol_seed": 0, "steps": STEPS, "target": "N(0,I)", "recipe": ctx.recipe.to_dict(),
               "initial_prior": ctx.prior_config, "bounds": BOUNDS, "metrics": metrics,
               "passed": not failed, "failed_bounds": failed, "scorer_controls": {"oracle": oracle, "destructive_zero": destructive},
               "projection_rng": {"private_seed": 0, "checkpoint_path": "trainer.prior_regularizer.projection_rng_state"},
               "gif_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
               "initialization": ctx.initialization}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    torch.save({"context": ctx.state_dict(), "data_rng": data.get_state()}, args.output / "checkpoint.pt")
    print(json.dumps({"metrics": metrics, "passed": not failed, "failed_bounds": failed}), flush=True)


if __name__ == "__main__":
    main()
