"""Restore all nine CUDA contexts and audit bounded saved prior fields."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.toy_audit import gaussian_prior_magnitude as study
from experiments.forge.contracts import atomic_json, file_hash

spec = importlib.util.spec_from_file_location("_capped_prior_verification_host",
    ROOT / "reports/forge/bcap-past-extrapolation/verify.py")
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)
verification.declaration, verification.build = study.declaration, study.build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if torch.device(args.device).type != "cuda":
        raise ValueError("verification requires CUDA")
    rows = verification.verify(args.raw, device=args.device)
    bounded = []
    for phase in ("stationary", "shift"):
        cell = args.raw / ("extrapolation_from_past-gaussian1d_acquisition-"+phase)
        state = torch.load(cell/"state.pt", weights_only=True, map_location="cpu")
        field = state["trainer"]["extrapolation"]["previous"]["prior.z"]
        norms = field.norm(dim=1)
        nonzero = norms[norms > 0]
        if bool((norms > 1.000001).any()):
            raise ValueError("cached prior field violates cap")
        bounded.append(dict(phase=phase, nonzero_rows=len(nonzero), min_norm=float(nonzero.min()),
            mean_norm=float(nonzero.mean()), max_norm=float(nonzero.max()),
            implied_prior_mean_step=float(nonzero.mean())*.03, fraction_at_cap=float((nonzero >= .99999).float().mean()),
            checkpoint_sha256=file_hash(cell/"state.pt")))
    atomic_json(args.raw/"bounded-field-audit.json", dict(scope="secondary_saved_state_diagnostic",
        training_updates=0, model_sampling_draws=0, scale=.001, results=bounded))
    print(json.dumps(dict(exact_restores=len(rows), training_updates=0, model_sampling_draws=0)))
