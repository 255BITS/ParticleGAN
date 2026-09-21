#!/usr/bin/env python
"""Secondary frozen world-model checks after control-based checkpoint selection."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_gym_transition import load_checkpoint, predict, sha256
from experiments.evaluate_gym_transition import generation_metrics
from lib.gym_evaluation import score_predictions


@torch.no_grad()
def diagnose(checkpoints, output, data_dir, device="cpu"):
    from lib.gym_control import load_control_checkpoint
    torch.set_num_threads(1)
    output, data_dir = Path(output), Path(data_dir)
    if output.exists():
        raise FileExistsError(f"Use a fresh report path: {output}")
    started = time.perf_counter()
    data = {}
    for split in ("train", "test"):
        with np.load(data_dir / f"{split}.npz", allow_pickle=False) as archive:
            data[split] = {key: archive[key] for key in archive.files}
    rows = []
    for name, path in checkpoints:
        print(f"Secondary world-model checks: {name}", flush=True)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        loader = load_control_checkpoint if "E_control" in saved else load_checkpoint
        bundle = loader(path, device)
        test = data["test"]
        prediction = predict(bundle, test["states"], test["actions"], test["terrain"]).cpu().numpy()
        conditional = score_predictions(prediction, test, bundle["scaler"].state_scale.cpu().numpy())
        generation, _ = generation_metrics(bundle, data["train"], test, device)
        rows.append(dict(name=name, checkpoint=str(path), checkpoint_sha256=sha256(path),
                         step=bundle["step"], conditional=conditional, generation=generation))
    sources = ["experiments/diagnose_gym_control.py", "experiments/evaluate_gym_transition.py",
               "experiments/train_gym_transition.py", "lib/gym_control.py", "lib/gym_transition.py",
               "lib/gym_evaluation.py"]
    report = dict(protocol={
        "purpose": "Secondary diagnostics only; selected checkpoints come from fresh control validation episodes.",
        "reference": "Original frozen world-model test set, including counterfactual commands and mixed behavior.",
        "interpretation": "Control training uses expert behavior only: distribution changes need not improve fit to the old mixture.",
        "prediction": "E_pair(state, current action, terrain) -> z -> G3; six continuous standardized errors and separate contacts.",
        "generation": "Original generation_metrics protocol: 16 terrain contexts, at most 512 records each, fixed draws.",
        "control": "These metrics do not measure landing performance and do not select checkpoints."},
        sources={name: sha256(ROOT / name) for name in sources},
        dataset={f"{split}.npz": sha256(data_dir / f"{split}.npz") for split in data},
        rows=rows, elapsed_seconds=time.perf_counter()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Saved {output} in {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path; repeat")
    parser.add_argument("--output", default="reports/gym/lunar_lander_control/world_model_diagnostics.json")
    parser.add_argument("--data-dir", default="results/gym/lunar_lander/data")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    diagnose([item.split("=", 1) for item in args.checkpoint], args.output, args.data_dir, args.device)


if __name__ == "__main__":
    main()
