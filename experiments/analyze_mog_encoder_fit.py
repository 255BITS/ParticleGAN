#!/usr/bin/env python
"""Replay final encoder choices and measure the contribution of wrong-grid errors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_mog_encoder_fit import ARMS, load_models, query
from experiments.train_mog_autoencoder import draw_rng, json_write
from lib.toy_models import sample_100gaussians


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/mog-autoencoder/encoder-fit/tail_audit.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = json.loads((args.run_dir / ARMS[0] / "config.json").read_text())
    base_run = Path(cfg["base_run"])
    assert hashlib.sha256((base_run / "checkpoint.pt").read_bytes()).hexdigest() == cfg["base_checkpoint_sha256"]
    base, saved, prior, decoder, encoder = load_models(base_run, args.device)
    means = prior.means()
    decoded = decoder(means)
    real = sample_100gaussians(cfg["eval_samples"], args.device,
                              generator=draw_rng(args.device, cfg["evaluation_seed"]))
    rows = []
    for arm in ("initial", *ARMS):
        if arm == "initial":
            encoder.load_state_dict(saved["encoder"])
            expected = cfg["baseline_mse"]
            checkpoint_hash = cfg["base_checkpoint_sha256"]
        else:
            path = args.run_dir / arm / "checkpoint.pt"
            checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            fitted = torch.load(path, map_location=args.device, weights_only=True)
            assert fitted["base_checkpoint_sha256"] == cfg["base_checkpoint_sha256"]
            encoder.load_state_dict(fitted["encoder"])
            expected = json.loads((path.parent / "metrics.json").read_text())["encoder_mse"]
        ids = torch.cat([((query(encoder, x)[:, None]-means).square().sum(-1)).argmin(1)
                         for x in real.split(4096)])
        reconstruction = decoded[ids]
        error = (reconstruction.double()-real.double()).square().mean(1)
        wrong = ((reconstruction+4.5).round().clamp(0, 9) != (real+4.5).round().clamp(0, 9)).any(1)
        assert abs(float(error.mean())-expected) < 1e-12
        contribution = float(error[wrong].sum()/len(real))
        row = dict(arm=arm, checkpoint_sha256=checkpoint_hash, samples=len(real),
                   mse=float(error.mean()), wrong_grid_count=int(wrong.sum()),
                   wrong_grid_fraction=float(wrong.double().mean()),
                   wrong_grid_mse_contribution=contribution,
                   wrong_grid_error_share=contribution/float(error.mean()),
                   same_grid_conditional_mse=float(error[~wrong].mean()),
                   error_quantiles=dict(zip(("p50", "p90", "p99", "max"),
                       torch.quantile(error, error.new_tensor([.5, .9, .99, 1.])).cpu().tolist())))
        rows.append(row)
        print(json.dumps(row), flush=True)
    json_write(args.out, dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows))


if __name__ == "__main__":
    main()
