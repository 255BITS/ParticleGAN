"""Summarize the frozen BCAP winner and saved draws without training or sampling.

Original gates remain inputs. Derived moments explain failures and confer no
qualification. Raw observations, tensors and execution logs stay in the archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

# This directory also contains the historical selection script select.py.
# Avoid shadowing the standard-library select extension during imports.
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Changed original artifact: {path}")
    return actual


def passes(row, bounds):
    return all(key in row and np.isfinite(row[key]) and
               {">=": row[key] >= value, "<=": row[key] <= value,
                "==": row[key] == value}[op] for key, op, value in bounds)


def curve_summary(rows, bounds):
    if not rows:
        return None
    ok = [passes(row, bounds) for row in rows]
    suffix = 0
    for value in reversed(ok):
        if not value:
            break
        suffix += 1
    longest = current = 0
    for value in ok:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return dict(observations=len(rows), passing_observations=sum(ok),
                passing_suffix=suffix, longest_passing_run=longest,
                first_passing_step=next((r["step"] for r, p in zip(rows, ok) if p), None),
                last_failed_step=next((r["step"] for r, p in zip(reversed(rows), reversed(ok)) if not p), None),
                first={key: rows[0].get(key) for key in ["step", *dict.fromkeys(b[0] for b in bounds)]},
                endpoint={key: rows[-1].get(key) for key in ["step", *dict.fromkeys(b[0] for b in bounds)]})


def gaussian_phase(rows, bounds):
    result = curve_summary(rows, bounds)
    result["metric_ranges"] = {
        key: dict(min=min(r[key] for r in rows), max=max(r[key] for r in rows),
                  median=float(np.median([r[key] for r in rows])))
        for key in ["mean_error_sigma", "std_ratio", "cdf_ks"]}
    result["failure_counts"] = {
        f"{key} {op} {value}": sum(not passes(r, [(key, op, value)]) for r in rows)
        for key, op, value in bounds}
    result["moment_passes_with_shape_failure"] = sum(
        passes(row, [b for b in bounds if b[0] != "cdf_ks"]) and row["cdf_ks"] > .05
        for row in rows)
    return result


def normal_ks(points, mean, std):
    cdf = ndtr((np.sort(points) - mean) / std)
    n = len(points)
    return float(max(np.max(np.arange(1, n + 1) / n - cdf),
                     np.max(cdf - np.arange(n) / n)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-workspace", type=Path,
                        default=Path("/home/martyn/dev/ParticleGAN-bcap-tier2-search"))
    parser.add_argument("--output", type=Path, default=REPORT / "failure-analysis.json")
    parser.add_argument("--figure", type=Path, default=REPORT / "failure-curves.png")
    args = parser.parse_args()
    summary = read(REPORT / "summary.json")
    readout = read(REPORT / "readout.json")
    receipts = read(REPORT / "receipts.json")
    selected = summary["selected_candidate_id"]
    survivors = [t for t in readout["trials"] if t["required_passes_by_tier"][0] == 6]
    winner = next(t for t in survivors if t["candidate_id"] == selected)
    task_map = {t["task"]: t for t in winner["tasks"]}
    selected_recipe = read(ROOT / "configs/forge/configurations" / f"{selected}.json")["resolved_configuration_recipe"]
    original_tasks, curves, proofs = {}, {}, []
    selected_receipts = [r for r in receipts if r["candidate_id"] == selected]
    if len(selected_receipts) != 28:
        raise ValueError("Expected all 28 winner receipts")
    for receipt in selected_receipts:
        directory = args.original_workspace / "reports/forge/attempts" / receipt["attempt_id"]
        for key, binding in receipt["provenance"]["original_files"].items():
            verified(directory / f"{key}.json", binding["sha256"])
        original = read(directory / "result.json")["task_results"][0]
        name = original["task_id"]
        published = next(t for t in receipt["task_results"] if t["task_id"] == name)
        if original["gate_status"] != published["gate_status"]:
            raise ValueError("Original gate differs from publication")
        if receipt["provenance"]["source_digest"] != readout["source_digest"]:
            raise ValueError("Foreign scientific source")
        applied = original.get("applied", original)
        recipe = applied["recipe"]
        omitted_recipe_fields = []
        for key in ("loss", "optimizer_family", "optimizer_smoothing", "optimizer_momentum",
                    "optimizer_convolution", "lr", "d_lr_mult", "prior_lr_mult", "reg_coeff",
                    "reg_kappa", "reg_every", "lr_floor", "network_lr_floor"):
            expected = selected_recipe[key]
            if key not in recipe:
                omitted_recipe_fields.append(key)
                continue
            if recipe[key] != expected:
                raise ValueError(f"Wrong winner recipe for {name}: {key}")
        evidence = original["evidence"]
        source_receipt = read(directory / "evidence.json")
        artifact_root = Path(source_receipt["local_artifact_root"])
        original_tasks[name] = (original, artifact_root)
        curves[name] = evidence.get("observations", [])
        row = task_map[name]
        if row["qualification_tier"] != 2:
            continue
        evaluator = published["evaluator_summary"]
        checks = evaluator.get("metric_checks", {})
        proofs.append(dict(task=name, attempt_id=receipt["attempt_id"],
                           compatibility_key=original["compatibility_key"],
                           result_sha256=receipt["provenance"]["original_files"]["result"]["sha256"],
                           gate_status=original["gate_status"], prior=applied["prior"],
                           recipe_fields_omitted_by_original_serializer=omitted_recipe_fields,
                           sampling_law=evidence["sampling_law"],
                           numerical_failures=[v for v in checks.values() if v["status"] != "PASS"],
                           endpoint_metrics=row["metrics"],
                           convergence=evaluator.get("convergence"),
                           retained_summary={k: v for k, v in evaluator.items() if k not in
                                             ("convergence", "metric_checks", "final_metrics", "holdout_metrics",
                                              "holdout_ema_metrics", "oracle_metrics")},
                           derived_curve=curve_summary(curves[name], [(v["metric"], v["op"], v["threshold"])
                                                                     for v in checks.values()]) if checks else None))
    counts = Counter(p["gate_status"] for p in proofs)
    if counts != {"PASS": 7, "FAIL": 14}:
        raise ValueError(f"Unexpected Tier 2 results: {counts}")
    result = dict(schema_version=1, scope="zero_training_saved_evidence_analysis",
                  qualification_input=False, optimizer_updates_added=0, sampling_draws_added=0,
                  source_digest=readout["source_digest"], candidate_id=selected,
                  candidate_revision=winner["candidate_revision"], seed=0,
                  inputs={p.name: sha(p) for p in [REPORT / "readout.json", REPORT / "receipts.json",
                          REPORT / "summary.json", REPORT / "archive.json", Path(__file__)]},
                  verified_original_receipt_files=3 * len(selected_receipts),
                  tier2_status_counts=dict(counts), tasks=proofs, artifact_proofs=[])
    result["mechanism_source_proofs"] = {}
    for filename in ("particlegan/optim/dualnorm.py", "particlegan/grad_regularizers.py",
                     "particlegan/gan_loss.py", "benchmarks/toy100/problems.py",
                     "benchmarks/locked_shared/trajectory.py", "benchmarks/locked_shared/hosts/residual_student.py"):
        digest = verified(args.original_workspace / filename, source_receipt["source"]["files"][filename])
        verified(ROOT / filename, digest)
        result["mechanism_source_proofs"][filename] = digest
    # Attribute Gaussian failure across stationary and shifted cohorts separately.
    gaussian, _ = original_tasks["gaussian1d_stability"]
    bounds = read(ROOT / "configs/forge/tasks/gaussian1d_stability.json")["evaluation"]["thresholds"]
    result["gaussian_phases"] = {
        name: gaussian_phase([r for r in curves["gaussian1d_stability"] if lo < r["step"] <= hi], bounds)
        for name, lo, hi in [("stationary", 1000, 4000), ("reacquisition", 4000, 5000), ("shift_hold", 5000, 6000)]}
    import torch
    torch.set_num_threads(1)
    path = Path(gaussian["evidence"]["artifact_root"]) / "observed-samples.pt"
    digest = verified(path, gaussian["evidence"]["saved_observer_outputs"]["sha256"])
    records = torch.load(path, map_location="cpu", weights_only=True)
    points = records[-1]["samples"].numpy().reshape(-1).astype(np.float64)
    mean, std = points.mean(), points.std()
    result["gaussian_terminal_shape"] = dict(sample_mean=float(mean), sample_std=float(std),
        ks_against_target=normal_ks(points, 3., .5),
        ks_against_moment_matched_normal=normal_ks(points, mean, std),
        quantiles=dict(zip(["q01", "q10", "q50", "q90", "q99"],
                           np.quantile(points, [.01, .1, .5, .9, .99]).tolist())))
    result["artifact_proofs"].append(dict(task="gaussian1d_stability", path=str(path), sha256=digest))
    # Decompose saved vector shape into core, spill and between-group variance.
    from benchmarks.toy_audit.failure_diagnosis import vector_components
    result["inputs"]["vector_decomposition_source"] = sha(ROOT / "benchmarks/toy_audit/failure_diagnosis.py")
    result["vector_components"] = {}
    for name in ("vector_unequal_mass", "vector_unequal_width", "vector_anisotropic"):
        original, artifact_root = original_tasks[name]
        path = artifact_root / "observed-samples.pt"
        digest = verified(path, original["evidence"]["saved_observer_outputs"]["sha256"])
        saved = torch.load(path, map_location="cpu", weights_only=True)[-1]
        spec = original["evidence"]["host"]["definition"]
        result["vector_components"][name] = vector_components(saved["samples"].numpy(), spec, saved["step"])
        result["artifact_proofs"].append(dict(task=name, path=str(path), sha256=digest))
    # Native occupancy versus genuine hits uses the declared evaluator geometry.
    from benchmarks.toy100.problems import evaluation_geometry
    result["inputs"]["native_geometry_source"] = sha(ROOT / "benchmarks/toy100/problems.py")
    result["native_coverage"] = {}
    native_curves = {}
    for name in ("grid100", "rotated100", "staggered100"):
        original, artifact_root = original_tasks[name]
        evidence = original["evidence"]
        base = Path(evidence["artifact_root"])
        for filename in ("events.jsonl", "holdout_samples.npz"):
            path = base / name / filename
            digest = verified(path, evidence["artifact_manifest"]["files"][f"{name}/{filename}"]["sha256"])
            result["artifact_proofs"].append(dict(task=name, path=str(path), sha256=digest))
        rows = [json.loads(line) for line in (base / name / "events.jsonl").read_text().splitlines()]
        live = [r for r in rows if r.get("model") == "live"]
        native_curves[name] = live
        with np.load(base / name / "holdout_samples.npz") as arrays:
            points = arrays["live"].astype(np.float64)
        centers, sigma = evaluation_geometry(name)
        centers = centers.numpy().astype(np.float64)
        distance = np.square(points[:, None] - centers).sum(-1)
        nearest = distance.argmin(1)
        radius = np.sqrt(distance.min(1)) / sigma
        cell_counts = np.bincount(nearest, minlength=100)
        hits = np.bincount(nearest[radius <= 3.], minlength=100)
        result["native_coverage"][name] = dict(
            nearest_occupied_cells=int(np.sum(cell_counts > 0)), genuine_modes_005_mass=int(np.sum(hits >= .005 * len(points))),
            cells_without_any_3sigma_hit=int(np.sum(hits == 0)), precision_3sigma=float(np.mean(radius <= 3.)),
            median_nearest_center_distance_sigma=float(np.median(radius)),
            observed_mode_range=[min(r["metrics"]["modes"] for r in live), max(r["metrics"]["modes"] for r in live)],
            best_observed_precision=max(r["metrics"]["precision"] for r in live),
            endpoint_modes=live[-1]["metrics"]["modes"],
            native_updates=live[-1]["step"], terminal_accuracy_passing_checks=
                sum(check["passed"] for check in original["evaluator_result"]["terminal_checks"]))
    # Existing matched survivors constrain explanations without selecting anew.
    winner_gates = {t["task"]: t["gate_status"] for t in winner["tasks"] if t["qualification_tier"] == 2}
    result["matched_survivors"] = []
    result["shared_failed_tasks_across_survivors"] = sorted(
        name for name in winner_gates if all(next(t["gate_status"] for t in trial["tasks"]
                                                 if t["task"] == name) == "FAIL" for trial in survivors))
    for trial in survivors:
        gates = {t["task"]: t["gate_status"] for t in trial["tasks"] if t["qualification_tier"] == 2}
        gaussian_metrics = next(t["metrics"] for t in trial["tasks"] if t["task"] == "gaussian1d_stability")
        result["matched_survivors"].append(dict(configuration_id=trial["configuration_id"], loss=trial["loss"],
            smoothing=trial["settings"]["optimizer_smoothing"], cap=trial["settings"]["reg_kappa"],
            tier2_passes=trial["required_passes_by_tier"][1],
            changed_gates_against_winner={k: v for k, v in gates.items() if v != winner_gates[k]},
            gaussian_endpoint={k: gaussian_metrics[k] for k in ("cdf_ks", "std_ratio")}))
    if abs(result["gaussian_terminal_shape"]["ks_against_target"] - gaussian["metrics"]["cdf_ks"]) > 1e-12:
        raise ValueError("Saved Gaussian draws do not reproduce the recorded KS")
    for name in native_curves:
        if abs(result["native_coverage"][name]["precision_3sigma"] - task_map[name]["metrics"]["precision"]) > 1e-12:
            raise ValueError("Saved native holdout does not reproduce recorded precision")
    result["analysis_runtime"] = dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    rows = curves["gaussian1d_stability"]
    steps = [r["step"] for r in rows]
    axes[0, 0].plot(steps, [r["std_ratio"] for r in rows], color="#28658c")
    axes[0, 0].axhspan(.8, 1.2, color="#73a56b", alpha=.2)
    axes[0, 0].set(title="Gaussian width during continued training", ylabel="Standard deviation / target")
    axes[0, 1].plot(steps, [r["cdf_ks"] for r in rows], color="#28658c")
    axes[0, 1].axhline(.05, color="#b64a38", linestyle="--", label="KS limit 0.05")
    axes[0, 1].set(title="Gaussian distribution mismatch", ylabel="CDF KS")
    for ax in axes[0]:
        ax.axvline(4000, color="#666666", linestyle=":", label="Target shift")
        ax.legend(fontsize=8)
    for name, rows in native_curves.items():
        axes[1, 0].plot([r["step"] for r in rows], [r["metrics"]["precision"] for r in rows], label=name)
    axes[1, 0].axhline(.97, color="#b64a38", linestyle="--", label="Coverage limit 0.97")
    axes[1, 0].set(title="Native 100 mode precision", ylabel="Fraction within 3 target sigmas", ylim=(0, 1.03))
    axes[1, 0].legend(fontsize=8)
    rows = curves["mode_hold"]
    axes[1, 1].plot([r["step"] for r in rows], [r["hq"] for r in rows], marker=".")
    axes[1, 1].axhline(.9, color="#b64a38", linestyle="--", label="Quality limit 0.90")
    axes[1, 1].set(title="Mode hold quality breaks retention at update 1050", ylabel="High quality fraction", ylim=(0, 1.05))
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Completed updates")
        ax.grid(alpha=.18)
    fig.suptitle("BCAP winner Tier 2 failures from saved measurements")
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    print(f"Verified 84 original receipt files; Tier 2: {dict(counts)}; no new updates or draws.", flush=True)
    print(f"Wrote {args.output} and {args.figure}", flush=True)


if __name__ == "__main__":
    main()
