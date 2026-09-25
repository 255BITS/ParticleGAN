#!/usr/bin/env python
"""Diagnose the frozen baseline using training/validation records only, on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_gym_transition import load_checkpoint, predict
from lib.gym_transition import contact_record, composed_transition, encoded_transition


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def diagnose_checkpoint(path, data, synthetic_count=4096):
    bundle = load_checkpoint(path, device="cpu")
    scaler = bundle["scaler"]
    row = {"checkpoint": str(path), "checkpoint_sha256": sha256(path),
           "step": bundle["step"], "arm": bundle["config"]["arm"], "splits": {}}
    for split, values in data.items():
        pred = predict(bundle, values["states"], values["actions"], values["terrain"]).numpy()
        squared = ((pred[:, :6]-values["next_states"][:, :6])/scaler.state_scale.numpy())**2
        stats = {"count": len(pred), "next_mse": float(squared.mean()),
                 "sample_error_p50": float(np.quantile(squared.mean(1), .5)),
                 "sample_error_p95": float(np.quantile(squared.mean(1), .95))}
        row["splits"][split] = stats
        if bundle["G"] is None:
            continue
        inputs = torch.cat([scaler.state(torch.tensor(values["states"])),
                            scaler.action(torch.tensor(values["actions"]))], 1)
        terrain = torch.tensor(values["terrain"])
        decoded, codes, indices = [], [], []
        for start in range(0, len(inputs), 1024):
            out, encoding = encoded_transition(bundle["E"], bundle["G"], bundle["prior"],
                                               inputs[start:start+1024], terrain[start:start+1024])
            decoded.append(out)
            codes.append(encoding.codes[:, 0])
            indices.append(encoding.indices[:, 0])
        out, codes, indices = torch.cat(decoded), torch.cat(codes), torch.cat(indices)
        offsets = (codes-bundle["prior"].means()[indices])/bundle["prior"].sigma
        stats.update(state_mse=float((out[:, :6]-inputs[:, :6]).square().mean()),
                     action_mse=float((out[:, 8:10]-inputs[:, 8:10]).square().mean()),
                     used_components=len(indices.unique()),
                     offset_nearbound_fraction=float((offsets.abs() >= 2.85).float().mean()),
                     offset_abs_p95=float(offsets.abs().flatten().quantile(.95)))
    if bundle["G"] is not None:
        rng = torch.Generator().manual_seed(711)
        terrain = torch.tensor(data["train"]["terrain"][:synthetic_count])
        prior, g, e = bundle["prior"], bundle["G"], bundle["E"]
        fake = contact_record(g(prior.sample(len(terrain), rng)[0], terrain), rng=rng)
        _, decoded, _ = composed_transition(e, g, prior, fake, terrain, rng=rng)
        row["synthetic"] = {
            "count": len(terrain), "state_rms": float(fake[:, :6].square().mean().sqrt()),
            "state_abs_p95": float(fake[:, :6].abs().flatten().quantile(.95)),
            "state_reconstruction_mse": float((decoded[:, :6]-fake[:, :6]).square().mean()),
            "prior_raw_rms": float(prior.z.square().mean().sqrt()),
            "prior_std_min": float(prior.z.std(0).min()),
            "g1_parameter_norm": float(torch.cat([p.flatten() for p in g.branches[0].parameters()]).norm())}
    return row, bundle["provenance"]


def diagnose(run_root, output):
    torch.set_num_threads(1)
    run_root, output = Path(run_root), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh diagnostic report path")
    started = time.perf_counter()
    data, references = {}, {}
    for split in ("train", "validation"):
        path = run_root/"data"/f"{split}.npz"
        references[path.name] = sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            data[split] = {k: archive[k] for k in ("states", "actions", "next_states", "terrain")}
    protocol = {
        "version": 1, "device": "cpu", "torch_threads": 1,
        "checkpoint_selection": {"direct": [1000, 10000], "reconstruction": [1000, 5000, 10000],
                                 "adversarial": [1000, 5000, 10000]},
        "data_selection": "all stored training and validation records; no test data loaded",
        "conditional_metrics": "EMA checkpoints, six continuous coordinates, checkpoint training scaler",
        "synthetic_selection": "4096 shared MoG draws using first 4096 training terrain contexts",
        "synthetic_rng_seed": 711, "contact_sampling": "hard Bernoulli, no gradient",
        "offset": "(encoded code - selected standardized prior center) / fixed prior sigma",
        "near_bound": "fraction of offset coordinates with absolute value >= 2.85 (bound 3)",
        "source_verification": "trained source pins checked against current inference source files",
        "interpretation": "diagnostic snapshots, not additional training runs or a revised leaderboard"}
    rows, traces, source_hashes = [], {}, {}
    for arm, steps in protocol["checkpoint_selection"].items():
        metrics_path = run_root/arm/"metrics.jsonl"
        metrics_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        traces[arm] = {"metrics_sha256": sha256(metrics_path),
                       "rows": [row for row in metrics_rows if row["step"] in steps]}
        for step in steps:
            path = run_root/arm/f"checkpoint_{step}.pt"
            print(f"Diagnosing {arm} step {step} on CPU", flush=True)
            row, provenance = diagnose_checkpoint(path, data)
            for name, digest in references.items():
                if provenance["dataset"].get(name) != digest:
                    raise ValueError(f"Checkpoint reference mismatch: {name}")
            for name, digest in provenance["sources"].items():
                if sha256(ROOT/name) != digest:
                    raise ValueError(f"Checkpoint inference source mismatch: {name}")
                source_hashes[name] = digest
            rows.append(row)
    report = {"protocol": protocol, "references": references,
              "diagnostic_source_sha256": sha256(__file__), "inference_sources": source_hashes,
              "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                              "python": sys.version},
              "rows": rows, "training_trace": traces,
              "elapsed_seconds": time.perf_counter()-started,
              "hypotheses": [
                  "Detached synthetic targets with live synthetic inputs can create positive feedback: "
                  "for F(x)=c*x with 0<c<1, differentiating (F(x)-stopgrad(x))^2 gives "
                  "2*c*(c-1)*x, so gradient descent increases |x| despite growing reconstruction error. "
                  "This is a possible mechanism, not proof of the observed network's cause.",
                  "A bounded next ablation is to detach synthetic encoder inputs for synthetic "
                  "reconstruction only, retaining the three generators, real reconstruction, and "
                  "live original/composed adversarial gradients.",
                  "Raw prior table scale is logged, but read standardization removes global scale; "
                  "raw RMS alone does not establish expanding latent codes."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Saved {output}; runtime {report['elapsed_seconds']:.1f}s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="results/gym/lunar_lander")
    parser.add_argument("--output", default="reports/gym/lunar_lander/diagnostics.json")
    args = parser.parse_args()
    diagnose(args.run_root, args.output)


if __name__ == "__main__":
    main()
