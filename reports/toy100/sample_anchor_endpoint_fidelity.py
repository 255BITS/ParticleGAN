"""Read-only mass and shape audit of the archived prestart-anchor endpoints.

The state files are the exact cold update-1200 and own-acquired update-2400
host snapshots.  This script runs no training or random evaluation draws.
Clean particles are grouped by the evaluator's nearest ring center only for
post-hoc diagnostics; the controller never receives those centers.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports/toy100/continuous-evidence"
FILES = {
    "cold1200": {
        "state": EVIDENCE / "round6-sample-anchor-qualified/sample-anchor-prestart-cold-v2/mode_hold-final-state.pt.gz",
        "result": EVIDENCE / "round6-sample-anchor-qualified/sample-anchor-prestart-cold-v2/mode_hold.json.gz",
        "state_raw_sha256": "8a761531fe172147f923fdc2f68609b0e3c45fb8204c549b650f513b9baac163",
    },
    "ownhold2400": {
        "state": EVIDENCE / "round6-anchor-own-and-missing/sample-anchor-prestart-own-hold/hold-final-state.pt.gz",
        "result": EVIDENCE / "round6-anchor-own-and-missing/sample-anchor-prestart-own-hold/hold.json.gz",
        "state_raw_sha256": "aae43815fd731c9d7d0b1e4019c5457de9f319e53e201ceeedebfef15274d2ea",
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_archived(path: Path) -> tuple[bytes, dict[str, str]]:
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    return raw, {
        "archive": str(path.relative_to(ROOT)),
        "gzip_sha256": sha(compressed),
        "raw_sha256": sha(raw),
        "gzip_bytes": len(compressed),
        "raw_bytes": len(raw),
    }


def clean_support(state: dict) -> torch.Tensor:
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = SimpleMLPGenerator(
            mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2
        )
        model.load_state_dict({
            name.removeprefix("model."): value
            for name, value in state["generator"].items()
        })
        model.eval()
        support = model(state["prior"]["z"]).detach().double()
    if not torch.equal(torch.get_rng_state(), before):
        raise AssertionError("read-only model construction changed RNG")
    return support


def endpoint(state: dict, result: dict) -> dict:
    points = clean_support(state)
    means = mode_hold.ring_means().double()
    if points.shape != (12, 2) or means.shape != (8, 2):
        raise AssertionError("unexpected toy shape")
    emitted_sigma = float(state["noise"]["output_sigma"])
    target_sigma = float(mode_hold.SIGMA)
    if emitted_sigma != 0.029 or target_sigma != 0.07:
        raise AssertionError("noise law does not match archived late toy")

    assignment = torch.cdist(points, means).argmin(1)
    counts = torch.bincount(assignment, minlength=len(means))
    weights = counts.double() / len(points)
    mass_tv = float((weights - 1 / len(means)).abs().sum() / 2)
    groups = []
    for group, count in enumerate(counts.tolist()):
        selected = points[assignment == group]
        if not count:
            groups.append(None)
            continue
        center = selected.mean(0)
        clean_cov = (selected - center).T @ (selected - center) / count
        emitted_cov = clean_cov + emitted_sigma**2 * torch.eye(2, dtype=torch.float64)
        max_pairwise = (float(torch.cdist(selected, selected).max())
                        if count > 1 else 0.0)
        groups.append({
            "particle_count": count,
            "particle_weight": float(weights[group]),
            "center_error": float(torch.linalg.vector_norm(center - means[group])),
            "max_pairwise_clean_distance": max_pairwise,
            "clean_covariance": clean_cov.tolist(),
            "emitted_covariance": emitted_cov.tolist(),
            "emitted_coordinate_std": emitted_cov.diag().sqrt().tolist(),
            "emitted_to_target_coordinate_variance_ratio":
                (emitted_cov.diag() / target_sigma**2).tolist(),
        })

    eval_counts = result["nearest_counts"]
    if sum(eval_counts) != 4096 or result["modes"] != 8 or result["hq"] != 1.0:
        raise AssertionError("archived terminal evaluation changed")
    if result["support"]["nearest_counts"] != counts.tolist():
        raise AssertionError("archived noisy support differs in mode allocation")
    return {
        "clean_nearest_mode_counts": counts.tolist(),
        "clean_nearest_mode_mass_tv": mass_tv,
        "clean_nearest_mode_min_max_count": [int(counts.min()), int(counts.max())],
        "grouped_emitted_law": groups,
        "emitted_output_sigma": emitted_sigma,
        "target_mode_sigma": target_sigma,
        "fixed_4096_draw_hq": result["hq"],
        "fixed_4096_draw_modes": result["modes"],
        "fixed_4096_draw_nearest_mode_counts": eval_counts,
        "fixed_4096_draw_mass_tv":
            sum(abs(value / 4096 - 1 / len(means)) for value in eval_counts) / 2,
        "one_noisy_draw_per_particle_counts": result["support"]["nearest_counts"],
    }


def audit() -> dict:
    source = {}
    for name in ("benchmarks/locked_shared/mode_hold.py",
                 "benchmarks/locked_shared/mlp.py"):
        source[name] = sha((ROOT / name).read_bytes())
    output = {
        "scope": "archived stationary 8-mode / 12-equal-particle toy; post-hoc evaluator centers only",
        "source_sha256": source,
        "endpoints": {},
    }
    all_results = {}
    for label, spec in FILES.items():
        state_raw, state_record = read_archived(spec["state"])
        result_raw, result_record = read_archived(spec["result"])
        if state_record["raw_sha256"] != spec["state_raw_sha256"]:
            raise AssertionError(f"wrong archived {label} host state")
        result_doc = json.loads(result_raw)
        if result_doc["source"]["benchmarks/locked_shared/mode_hold.py"] != source["benchmarks/locked_shared/mode_hold.py"]:
            raise AssertionError("mode_hold source mismatch")
        if result_doc["source"]["benchmarks/locked_shared/mlp.py"] != source["benchmarks/locked_shared/mlp.py"]:
            raise AssertionError("MLP source mismatch")
        if result_doc["final_state_file_sha256"] != spec["state_raw_sha256"]:
            raise AssertionError("result does not bind host state")
        state = torch.load(io.BytesIO(state_raw), map_location="cpu", weights_only=True)
        result = result_doc["result"]["raw"] if label == "cold1200" else result_doc["result"]
        output["endpoints"][label] = {
            "state_file": state_record,
            "result_file": result_record,
            "result": endpoint(state, result),
        }
        all_results[label] = result_doc

    checkpoints = all_results["ownhold2400"]["receipt"]["checkpoints"]
    if len(checkpoints) != 1200 or checkpoints[0]["step"] != 1201 or checkpoints[-1]["step"] != 2400:
        raise AssertionError("hold checkpoints incomplete")
    patterns = {tuple(row["support"]["nearest_counts"]) for row in checkpoints}
    output["dense_hold"] = {
        "updates": len(checkpoints),
        "first_last": [checkpoints[0]["step"], checkpoints[-1]["step"]],
        "min_fixed_draw_hq": min(row["hq"] for row in checkpoints),
        "min_fixed_draw_modes": min(row["modes"] for row in checkpoints),
        "one_noisy_draw_particle_count_patterns": [list(row) for row in sorted(patterns)],
        "caveat": "intermediate particle counts use one noisy draw, while endpoint counts use clean saved states",
    }
    output["integer_mass_floor"] = {
        "equal_particles": 12,
        "equal_target_modes": 8,
        "minimum_tv": 1 / 6,
        "achieved_by_count_multiset": [1] * 4 + [2] * 4,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EVIDENCE / "anchor-endpoint-fidelity/receipt.json")
    args = parser.parse_args()
    row = audit()
    row["audit_source_sha256"] = sha(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "receipt": str(args.output),
        "cold_tv": row["endpoints"]["cold1200"]["result"]["clean_nearest_mode_mass_tv"],
        "hold_tv": row["endpoints"]["ownhold2400"]["result"]["clean_nearest_mode_mass_tv"],
        "dense_checkpoints": row["dense_hold"]["updates"],
    }))


if __name__ == "__main__":
    main()
