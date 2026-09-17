#!/usr/bin/env python
"""Locate reverse-chain failures by substituting exact posterior transitions.

Oracle substitutions are evaluation diagnostics, never deployable samplers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from particlegan import DDGAN
from particlegan.diffusion import DrawSource
from experiments.train_denoising import make_prior
from lib.denoising_toy import GaussianGrid, ToyGenerator, grid_metrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=20000)
    args = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    assert cfg["model"] == "ddgan"
    assert args.samples % cfg["classes"] == 0
    toy = GaussianGrid(device, cfg["std"], cfg["classes"])
    schedule = DDGAN(cfg["alpha_bar"], validate_args=False).to(device)
    g = ToyGenerator(cfg).to(device)
    prior = make_prior(cfg, device)
    noise = DrawSource(cfg["noise"], cfg["noise_particles"], 2, cfg["seed"] + 102, device)
    for module, key in ((g, "G"), (prior, "prior"), (noise, "noise")):
        module.load_state_dict(ckpt[key])
        module.eval()
    c = torch.arange(args.samples, device=device) % cfg["classes"]
    real = toy.sample(c, torch.Generator(device=device).manual_seed(99003))
    all_steps = set(range(1, schedule.steps + 1))
    variants = {"all_model": all_steps, "all_oracle": set(),
                "oracle_last": all_steps - {1}}
    variants.update({f"model_only_t{t}": {t} for t in sorted(all_steps)})
    results, samples = {}, {}
    for name, model_steps in variants.items():
        rz, rn, rs, ro = [torch.Generator(device=device).manual_seed(99000 + k)
                          for k in (0, 1, 2, 4)]
        xt = torch.randn((len(c), 2), device=device, generator=rs)
        for step in range(schedule.steps, 0, -1):
            t = torch.full_like(c, step)
            if step in model_steps:
                x0 = g(prior.sample(len(c), rz)[0], c, xt, t)
                eta = noise.sample(len(c), rn)[0]
            else:
                x0 = toy.oracle_clean(xt, c, schedule.ab[step], ro)
                eta = torch.randn(xt.shape, device=device, generator=ro)
            xt = schedule.reverse(x0, xt, t, eta)
        results[name] = grid_metrics(xt, c, toy, real)
        samples[name] = xt.cpu().numpy()
        m = results[name]
        print(f"{name}: HQ={m['joint_hq']:.4f}, modes={m['modes']}, "
              f"TV={m['conditional_mode_tv']:.4f}, SW1={m['conditional_sw1']:.4f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": args.checkpoint,
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": cfg, "samples": args.samples, "metrics": results}
    (out / "chain_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "chain_samples.npz", c=c.cpu().numpy(), **samples)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), layout="constrained")
    for ax, (name, x) in zip(axes.flat, samples.items()):
        m = results[name]
        ax.scatter(*x[:4000].T, c=c[:4000].cpu(), cmap="tab10", vmin=0, vmax=9,
                   s=2, alpha=.4, linewidths=0)
        ax.set(title=name.replace("_", " "), xlim=(-5.2, 5.2), ylim=(-5.2, 5.2),
               aspect="equal", xlabel=f"HQ {m['joint_hq']:.1%}; class TV {m['conditional_mode_tv']:.3f}")
    for ax in list(axes.flat)[len(samples):]:
        ax.set_visible(False)
    fig.suptitle("Exact posterior substitutions: diagnostics, not model performance")
    fig.savefig(out / "chain_probe.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
