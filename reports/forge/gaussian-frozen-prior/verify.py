"""Verify exact CUDA restores, frozen prior and unchanged no-update controls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks.toy_audit.gaussian_frozen_prior import build, declaration
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import require_optimizer_steps, state_digest


@reproducible_execution
def verify(raw, *, device):
    if torch.device(device).type != "cuda":
        raise ValueError("verification requires CUDA")
    declaration()
    rows = []
    for receipt in json.loads((raw / "results.json").read_text())["results"]:
        arm, task, phase = receipt["arm"], receipt["task"], receipt["phase"]
        name = f"{arm}-{task}-{phase}"
        directory = raw / name
        saved = torch.load(directory / "state.pt", weights_only=True, map_location="cpu")
        initial = torch.load(directory / "initial-state.pt", weights_only=True, map_location="cpu")
        context, trainer, _ = build(arm, task, device)
        if receipt["completed_updates"] > trainer.max_steps:
            trainer.extend_execution(receipt["completed_updates"])
        context.load_state_dict(saved)
        if state_digest(saved) != state_digest(context.state_dict()):
            raise ValueError("final restore differs: " + name)
        require_optimizer_steps(saved, receipt["completed_updates"])
        rates = [group["lr"] for optimizer in (trainer.opt_g, trainer.opt_d) for group in optimizer.param_groups]
        if rates != [.012, .012*1.5]:
            raise ValueError("effective rates changed")
        if (list(trainer.prior.parameters()) or trainer.prior.z.requires_grad or
                state_digest(initial["trainer"]["models"]["prior"]) != state_digest(saved["trainer"]["models"]["prior"])):
            raise ValueError("prior is not frozen")
        row = dict(cell=name, context_restored_exactly=True, prior_unchanged=True,
                   effective_rates=rates, optimizer_steps=receipt["completed_updates"],
                   checkpoint_sha256=file_hash(directory / "state.pt"))
        if phase == "shift":
            frozen = torch.load(directory / "frozen-state.pt", weights_only=True, map_location="cpu")
            stationary = torch.load(raw / f"{arm}-{task}-stationary" / "state.pt", weights_only=True, map_location="cpu")
            for key in ("models", "optimizers", "completed_steps", "initial_lrs", "extrapolation"):
                if state_digest(frozen["trainer"].get(key)) != state_digest(stationary["trainer"].get(key)):
                    raise ValueError("no-update control changed: " + key)
            row["frozen_training_state_unchanged"] = True
        rows.append(row)
    proof = dict(training_updates=0, model_sampling_draws=0, cells=rows)
    atomic_json(raw / "restore-proof.json", proof)
    atomic_json(Path(__file__).resolve().parent / "restore-proof.json", proof)
    print(json.dumps(dict(exact_restores=len(rows), training_updates=0, model_sampling_draws=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    verify(args.raw, device=args.device)
