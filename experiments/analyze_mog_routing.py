#!/usr/bin/env python
"""Audit hard and soft particle usage from saved scout checkpoints; no training."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_mog_autoencoder import RoutingEncoder, draw_rng, json_write
from lib.toy_models import sample_100gaussians
from particlegan import MoGParticlePrior


@torch.no_grad()
def audit(run, device):
    cfg = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    checkpoint = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    prior = MoGParticlePrior(num_particles=400, z_dim=2, device=device).to(device)
    encoder = RoutingEncoder(cfg["width"]).to(device)
    prior.load_state_dict(checkpoint["prior"])
    encoder.load_state_dict(checkpoint["encoder"])
    assert float(prior.sigma) == cfg["sigma"]
    real = sample_100gaussians(cfg["eval_samples"], device,
                              generator=draw_rng(device, cfg["seed"] + 10000))
    ids, soft_sums = [], torch.zeros(400, device=device)
    for x in real.split(4096):
        _, chosen, _, soft = encoder(x, prior.means(), prior.sigma, cfg["arm"],
                                     torch.zeros_like(x), cfg["temperature"], return_routing=True)
        ids.append(chosen)
        soft_sums += soft.sum(0)
    ids = torch.cat(ids)
    # Reproduce the original hard selections on its 8192-example reconstruction set.
    original_counts = torch.bincount(ids[:8192], minlength=400).float()
    assert int((original_counts > 0).sum()) == metrics["used_particles"]
    original_p = original_counts / original_counts.sum()
    effective = float((-(original_p * original_p.clamp_min(1e-30).log()).sum()).exp())
    assert abs(effective - metrics["effective_particles"]) < .01
    counts = torch.bincount(ids, minlength=400).float()
    p, soft_p = counts / ids.numel(), soft_sums / ids.numel()
    return dict(arm=cfg["arm"], samples=ids.numel(), used_particles=int((counts > 0).sum()),
                effective_particles=float((-(p * p.clamp_min(1e-30).log()).sum()).exp()),
                hard_usage_tv=float((p - 1/400).abs().sum()/2),
                soft_usage_tv=float((soft_p - 1/400).abs().sum()/2),
                hard_soft_tv=float((p - soft_p).abs().sum()/2),
                hard_usage_chi2=float(400 * (p - 1/400).square().sum()),
                hard_counts=counts.long().cpu().tolist(), soft_frequencies=soft_p.cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/mog-autoencoder")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = [audit(p.parent, args.device) for p in sorted(args.run_dir.glob("*/checkpoint.pt"))
            if p.parent.name != "gan"]
    args.out.mkdir(parents=True, exist_ok=True)
    json_write(args.out / "routing_audit.json", rows)
    lines = ["# Held-out particle usage", "", "100,000 real examples per arm; hard selections audited against original reconstruction metrics. Lower TV is better; higher effective usage is more uniform.", "",
             "| Arm | Used /400 | Effective /400 | Hard usage TV | Soft usage TV | Hard/soft TV gap | Hard chi-square |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        lines.append(f"| {m['arm']} | {m['used_particles']} | {m['effective_particles']:.1f} | {m['hard_usage_tv']:.4f} | {m['soft_usage_tv']:.4f} | {m['hard_soft_tv']:.4f} | {m['hard_usage_chi2']:.4f} |")
    lines += ["", "Hard/soft TV gap is TV between the two aggregate routing distributions, not the difference of their distances to uniform. It does not measure gradient accuracy."]
    (args.out / "ROUTING.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
