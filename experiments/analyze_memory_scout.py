"""Numerical-only comparison of completed scouts, including coverage diagnostics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.memory_scout import diagnostics
from experiments.autonomous_memory import trajectory_metrics


def orbit_parameters(paths):
    rows = []
    for path in paths:
        initial = path[:32].astype(np.float64)
        origin = initial.mean(0)
        xy = initial-origin
        if np.linalg.svd(xy, compute_uv=False)[-1] < 1e-5:
            continue
        fit = np.linalg.lstsq(np.column_stack((2*xy, np.ones(len(xy)))), (xy*xy).sum(1), rcond=None)[0]
        center = origin+fit[:2]
        radius = np.sqrt(max(0., fit[2]+(fit[:2]*fit[:2]).sum()))
        offsets = path-center
        cross = offsets[:-1, 0]*offsets[1:, 1]-offsets[:-1, 1]*offsets[1:, 0]
        dot = (offsets[:-1]*offsets[1:]).sum(1)
        angles = np.arctan2(cross, dot)
        rows.append([*center, radius, angles.mean(), np.abs(angles).mean(), angles.std()])
    return np.asarray(rows).reshape(-1, 6)


def coverage(paths, reference):
    generated, real = orbit_parameters(paths), orbit_parameters(reference)
    if not len(generated):
        return {"valid_parameters": 0}
    quantiles = np.linspace(0, 1, 101)
    distances = np.abs(np.quantile(generated, quantiles, axis=0)-np.quantile(real, quantiles, axis=0)).mean(0)
    return {"valid_parameters": len(generated),
            "cw_fraction_all_fits": float((generated[:, 3]<0).mean()),
            "center_spread": float(np.sqrt(generated[:, :2].var(0).sum())),
            "center_mean": generated[:, :2].mean(0).tolist(),
            "radius_std": float(generated[:, 2].std()),
            "speed_std": float(generated[:, 4].std()),
            "angular_jitter": float(generated[:, 5].mean()),
            "marginal_w1": dict(zip(["center_x", "center_y", "radius", "signed_speed", "absolute_speed", "angular_jitter"], distances.tolist()))}


def failures(paths):
    checks = {"radius": lambda m: .5 <= m["radius"] <= 1.6,
              "radial": lambda m: m["relative_radial_rmse"] < .1,
              "drift": lambda m: m["late_radius_drift"] < .2,
              "speed": lambda m: .08 <= m["angular_speed"] <= .45,
              "direction": lambda m: m["direction_consistency"] > .95}
    counts = {key: 0 for key in ("invalid", *checks)}
    for path in paths:
        m = trajectory_metrics(path[None])
        if not m["valid_fit_fraction"]:
            counts["invalid"] += 1
        else:
            for key, check in checks.items():
                counts[key] += int(not check(m))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.sources:
        summaries = [source/"summary.json"] if (source/"summary.json").exists() else sorted(source.glob("*/summary.json"))
        for file in summaries:
            result = json.loads(file.read_text())
            with np.load(file.parent/"trajectories.npz") as arrays:
                paths, reference = arrays["generated"], arrays["real_clean"]
                full = diagnostics(paths[:, :256])
                long = diagnostics(paths) if paths.shape[1]>256 else None
                diversity = coverage(paths[:, :256], reference)
                reference_metrics = diagnostics(arrays["real_noisy"])
                failed = failures(paths[:, :256])
                prefixes = {str(n): trajectory_metrics(paths[:, :n])["circle_like_fraction"] for n in (32, 64, 128)}
            row = {"name": result.get("name", result.get("variant")),
                   "steps": result.get("steps", result["config"]["steps"]),
                   "source": str(file.parent), "metrics_256": full, "metrics_long": long,
                   "coverage": diversity, "reference": reference_metrics,
                   "failure_counts_256_overlapping": failed, "prefix_passes": prefixes,
                   "config": result["config"],
                   "seconds": result["seconds"]}
            provenance = file.parent/"provenance.json"
            if provenance.exists():
                row["provenance"] = json.loads(provenance.read_text())
            if "zero" in result["metrics"]:
                row["interventions"] = {key: result["metrics"][key] for key in ("zero", "shuffle")}
            rows.append(row)
    rows.sort(key=lambda r: (-r["metrics_256"]["circle_like_fraction"], r["metrics_256"]["relative_radial_rmse"] or 1e9))
    (args.out/"results.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    lines = ["# Completed autonomous memory experiments", "",
             "Selections use metrics only. Full cold-start circle passes use the existing early-fit diagnostic;",
             "coverage summaries include imperfect trajectories and are not calibrated distribution tests.", "",
             "| Run | Updates | Full 256 | Full 1024 | Late stopped | Radial RMSE | Passing CW / CCW | Initial spread |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m, long = row["metrics_256"], row["metrics_long"]
        radial = "n/a" if m["relative_radial_rmse"] is None else f"{m['relative_radial_rmse']:.3f}"
        extended = "n/a" if long is None else f"{100*long['circle_like_fraction']:.1f}%"
        lines.append(f"| {row['name']} | {row['steps']} | {100*m['circle_like_fraction']:.1f}% | {extended} | "
                     f"{100*m['late_stopped_fraction']:.1f}% | {radial} | {m['passing_cw']} / {m['passing_ccw']} | {m['initial_position_spread']:.3f} |")
    lines += ["", "Per-run numerical coverage, interventions, sources and real-reference results: [results.json](results.json).", ""]
    (args.out/"leaderboard.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
