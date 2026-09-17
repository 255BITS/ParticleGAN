#!/usr/bin/env python
"""Exhaustive frozen-decoder particle-selection audit; no training or offsets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_mog_autoencoder import RoutingEncoder, draw_rng, json_write
from lib.toy_models import SimpleMLPGenerator, sample_100gaussians
from particlegan import MoGParticlePrior


def center_errors(x, decoded, chosen=None):
    """Coordinate MSE to every decoded center; double precision avoids cancellation."""
    errors = (x.double()[:, None] - decoded.double()[None]).square().mean(-1)
    best_error, best = errors.min(1)
    selected = None if chosen is None else errors.gather(1, chosen[:, None]).squeeze(1)
    return best, best_error, selected


def usage(ids, k):
    counts = torch.bincount(ids, minlength=k)
    p = counts.double() / ids.numel()
    return dict(used=int((counts > 0).sum()),
                effective=float((-(p * p.clamp_min(1e-30).log()).sum()).exp()),
                tv=float((p - 1/k).abs().sum()/2), counts=counts.cpu().tolist())


@torch.no_grad()
def audit(run, device):
    cfg = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    checkpoint_path = run / "checkpoint.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert hashlib.sha256((run / "source.py").read_bytes()).hexdigest() == cfg["source_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    assert checkpoint["step"] == metrics["step"] == cfg["steps"]
    prior = MoGParticlePrior(num_particles=cfg["num_particles"], z_dim=cfg["z_dim"], device=device).to(device)
    encoder = RoutingEncoder(cfg["width"]).to(device)
    decoder = SimpleMLPGenerator(cfg["z_dim"], cfg["width"]).to(device)
    prior.load_state_dict(checkpoint["prior"])
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["g"])
    for model in (prior, encoder, decoder):
        model.eval().requires_grad_(False)
    assert float(prior.sigma) == cfg["sigma"]
    means = prior.means()
    decoded = decoder(means)
    real = sample_100gaussians(cfg["eval_samples"], device,
                              generator=draw_rng(device, cfg["seed"] + 10000))
    routed = cfg["arm"] != "gan"
    best_ids, best_errors, chosen_ids, chosen_errors = [], [], [], []
    for x in real.split(4096):
        chosen = None
        if routed:
            _, chosen, _ = encoder(x, means, prior.sigma, cfg["arm"],
                                    torch.zeros_like(x), cfg["temperature"])
        best, best_error, selected = center_errors(x, decoded, chosen)
        best_ids.append(best)
        best_errors.append(best_error)
        if routed:
            assert (best_error <= selected).all(), "oracle cannot lose to an available center"
            chosen_ids.append(chosen)
            chosen_errors.append(selected)
    best, best_error = torch.cat(best_ids), torch.cat(best_errors)
    reference_centers = (real + 4.5).round().clamp(0, 9) - 4.5
    result = dict(arm=cfg["arm"], samples=len(real), step=checkpoint["step"],
                  checkpoint_sha256=checkpoint_hash, training_source_sha256=cfg["source_sha256"],
                  seed=cfg["seed"], evaluation_seed=cfg["seed"] + 10000, sigma=float(prior.sigma),
                  initialization_sha256=cfg["initialization_sha256"],
                  oracle_mse=float(best_error.mean()), oracle_usage=usage(best, len(means)),
                  grid_center_mse=float((real.double() - reference_centers.double()).square().mean()),
                  decoded_centers=decoded.cpu().tolist(), encoder_mse=None, selection_gap=None,
                  removable_fraction=None, id_agreement=None)
    if routed:
        chosen, selected = torch.cat(chosen_ids), torch.cat(chosen_errors)
        original_usage = usage(chosen[:8192], len(means))
        assert original_usage["used"] == metrics["used_particles"]
        assert abs(original_usage["effective"] - metrics["effective_particles"]) < .01
        prefix_mse = float(selected[:8192].mean())
        assert abs(prefix_mse - metrics["zero_offset_mse"]) < 1e-8 + 1e-4 * metrics["zero_offset_mse"]
        regret = selected - best_error
        result.update(encoder_mse=float(selected.mean()), selection_gap=float(regret.mean()),
                      removable_fraction=float(regret.mean()/selected.mean()),
                      id_agreement=float((best == chosen).double().mean()),
                      strictly_better_fraction=float((regret > 1e-10).double().mean()),
                      regret_quantiles=dict(zip(("p50", "p90", "p99"),
                          torch.quantile(regret, regret.new_tensor([.5, .9, .99])).cpu().tolist())),
                      encoder_usage=usage(chosen, len(means)),
                      original_8192_encoder_mse=prefix_mse,
                      original_8192_oracle_mse=float(best_error[:8192].mean()))
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    return result


def export(rows, out):
    rows.sort(key=lambda r: r["oracle_mse"])
    json_write(out / "oracle_audit.json", dict(
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        rows=rows))
    lines = ["# Frozen-center reconstruction oracle", "",
             "Same 100,000 held-out real examples for every final 6,000-update checkpoint. All offsets are zero; no model is trained or changed. Rank by oracle MSE, measuring the available decoded centers rather than generation quality.", "",
             "```text", "Encoder: E(X) -> k -> G(p[k])",
             "Oracle:  try all k -> choose G(p[k]) closest to X", "```", "",
             "| Rank | Arm | Encoder MSE | Oracle MSE ↓ | Selection gap | Removable error % | ID agreement % |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows, 1):
        values = [rank, row["arm"]]
        values += ["—" if row[k] is None else f"{row[k]:.6f}"
                   for k in ("encoder_mse", "oracle_mse", "selection_gap")]
        values += ["—" if row[k] is None else f"{100*row[k]:.2f}"
                   for k in ("removable_fraction", "id_agreement")]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "Selection gap = encoder center MSE − oracle center MSE. Removable error is this gap divided by encoder center MSE. Oracle MSE is an exhaustive minimum over the frozen decoder's 400 center outputs; it is not a lower bound for models with offsets or a different decoder.", "",
              "The first 8,192 examples reproduce each saved zero-offset MSE and hard usage. Full raw metrics include per-example regret quantiles, selection counts, decoded centers, and checkpoint/source hashes. Checkpoint hashes are unchanged before/after the audit.", "",
              "ID disagreement can involve nearly identical outputs; assess the error gap rather than disagreement alone. Oracle assignments do not impose uniform usage and depend on the observed X, so oracle reconstructions are not unconditional generated samples.", "",
              f"Nearest true grid-center reference MSE: {rows[0]['grid_center_mse']:.6f}. This is a 100-center reference, not a lower bound for a 400-center quantizer.", "",
              "![Reconstruction error decomposition](oracle_errors.png)", ""]
    (out / "ORACLE.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    routed = sorted((r for r in rows if r["encoder_mse"] is not None), key=lambda r: r["encoder_mse"])
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["arm"] for r in routed]
    floors = [r["oracle_mse"] for r in routed]
    ax.barh(names, floors, label="Best available decoded center")
    ax.barh(names, [r["selection_gap"] for r in routed], left=floors,
            label="Additional error from encoder selection")
    ax.invert_yaxis()
    ax.set(xlabel="Coordinate MSE, zero offsets", title="Frozen checkpoints: error remaining after exhaustive selection")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "oracle_errors.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/mog-autoencoder")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for path in sorted(args.run_dir.glob("*/checkpoint.pt")):
        print(f"START {path.parent.name}", flush=True)
        row = audit(path.parent, args.device)
        rows.append(row)
        print(f"DONE {row['arm']} oracle_mse={row['oracle_mse']:.8f} encoder_mse={row['encoder_mse']}", flush=True)
    assert rows, "no checkpoints found"
    for key in ("samples", "seed", "evaluation_seed", "initialization_sha256", "step", "sigma"):
        assert len({r[key] for r in rows}) == 1, f"incomparable {key}"
    args.out.mkdir(parents=True, exist_ok=True)
    export(rows, args.out)


if __name__ == "__main__":
    main()
