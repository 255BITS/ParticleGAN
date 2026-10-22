"""One preregistered floor-corrected signal on already saved output panels.

The proposed signal is the normalized energy U-statistic. There is no selected
or swept signal threshold. Real training batches were not saved: the first
numerically passing generated sample set supplies an explicitly approximate
reference, separately for each Gaussian phase and overlap. No training, target
sampling, model construction, random split, or new random draw occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from analyze_settling import REQUESTS, SOURCE_COMMIT, SOURCE_DIGEST, digest, gaussian_pass, overlap_pass, scalar_summary


def signal(real, fake):
    real, fake = real.double().flatten(1), fake.double().flatten(1)
    n, m = len(real), len(fake)
    rr = torch.cdist(real, real, compute_mode="donot_use_mm_for_euclid_dist")
    ff = torch.cdist(fake, fake, compute_mode="donot_use_mm_for_euclid_dist")
    rf = torch.cdist(real, fake, compute_mode="donot_use_mm_for_euclid_dist")
    scale = (rr.square().sum() / (n * (n - 1))).sqrt()
    scale = scale if float(scale) > 0 else scale.new_tensor(1.)
    numerator = rf.mean() - .5 * rr.sum() / (n * (n - 1)) - .5 * ff.sum() / (m * (m - 1))
    value = float(numerator / scale)
    sq = rf.square()
    raw_nn = float((.5 * (sq.min(1).values.mean() + sq.min(0).values.mean()) / sq.median().clamp_min(1e-12)).sqrt())
    return value, raw_nn


def auroc(good, bad):
    if not good or not bad:
        return None
    return sum(float(b > g) + .5 * float(b == g) for g in good for b in bad) / (len(good) * len(bad))


def panel_summary(values):
    return {**scalar_summary(values), "mean": statistics.mean(values),
        "positive_panels": sum(value > 0 for value in values), "panels": len(values)}


def thermal_scale(value):
    u = min(1., max(0., value) / .020)
    return 1e-5 + (1 - 1e-5) * (3 * u * u - 2 * u * u * u)


def thermal_response(values):
    direct = [thermal_scale(value) for value in values]
    ema_values = []
    ema = None
    for value in values:
        value = max(0., value)
        ema = value if ema is None or value > ema else .95 * ema + .05 * value
        ema_values.append(thermal_scale(ema))
    return {"tau": .020, "s_min": 1e-5, "ema_decay": .95,
        "pointwise_scale": {**scalar_summary(direct), "mean": statistics.mean(direct)},
        "full_heat_panels": sum(value >= .020 for value in values),
        "sequential_fixed_state_ema_last_scale": ema_values[-1],
        "sequential_fixed_state_ema_mean_scale": statistics.mean(ema_values),
        "ema_scope": "32 fixed-state saved panel pairs in fixed row order, reset once per scored state; not historical in-training EMA"}


def population_atom_energy(points, real_masses, fake_masses):
    points = torch.tensor(points, dtype=torch.float64)
    real = torch.tensor(real_masses, dtype=torch.float64)
    fake = torch.tensor(fake_masses, dtype=torch.float64)
    distance = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    scale = (real @ distance.square() @ real).sqrt()
    scale = scale if float(scale) > 0 else scale.new_tensor(1.)
    value = float((real @ distance @ fake - .5 * real @ distance @ real - .5 * fake @ distance @ fake) / scale)
    return {"scope": "exact population point-mass fixture, distinct from positive-width task data; no sampled approximation",
        "real_masses": real.tolist(), "fake_masses": fake.tolist(), "mass_tv": float((real - fake).abs().sum() / 2),
        "energy_population_normalized_numerator": value, "unchanged_tau020_scale": thermal_scale(value)}


def run_panels(reference, current):
    values, nearest = [], []
    panels = min(len(reference), len(current)) // 128
    for index in range(panels):
        # A one-panel rotation ensures disjoint samples even for the reference
        # state itself. Across states the already saved draws are independent.
        ref_index = (index + 1) % panels
        value, nn = signal(reference[ref_index * 128:(ref_index + 1) * 128],
                           current[index * 128:(index + 1) * 128])
        values.append(value)
        nearest.append(nn)
    return {**panel_summary(values), "unchanged_controller_response": thermal_response(values)}, panel_summary(nearest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/queue"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    state = json.loads((args.queue_root / "queue/state.json").read_text())
    source = state["submissions"][REQUESTS["control"]]["request"]["source"]
    if source["origin_commit"] != SOURCE_COMMIT or source["digest"] != SOURCE_DIGEST:
        raise ValueError("source binding changed")
    tasks = {}
    artifacts = {}
    for job in state["jobs"].values():
        task = job["definition"]["execution_group"]
        if REQUESTS["control"] not in job["subscribers"] or task not in ("gaussian1d_smoke", "gaussian1d_stability", "vector_overlap"):
            continue
        row, = job["result"]["task_results"]
        evidence = row["evidence"]
        metadata = evidence["saved_observer_outputs"]
        path = Path(evidence.get("artifact_root") or job["attempts"][-1]["path"]) / metadata["path"]
        if digest(path) != metadata["sha256"]:
            raise ValueError("sample artifact differs")
        records = torch.load(path, map_location="cpu", weights_only=False)
        metrics = {item["step"]: item for item in evidence["observations"]}
        # Gaussian continuation also records its restored initial sample set;
        # include it only if the receipt supplied a numerical observation.
        tasks[task] = [{**item, "metrics": metrics[item["step"]]} for item in records if item["step"] in metrics]
        artifacts[task] = {"path": str(path), "sha256": metadata["sha256"]}
    references = {
        "gaussian_stationary": next(item for item in tasks["gaussian1d_smoke"] if gaussian_pass(item["metrics"])),
        "gaussian_shifted": next(item for item in tasks["gaussian1d_stability"] if item["step"] > 4000 and gaussian_pass(item["metrics"])),
        "overlap": next(item for item in tasks["vector_overlap"] if overlap_pass(item["metrics"])),
    }
    result = {"schema_version": 1, "source_commit": SOURCE_COMMIT, "source_digest": SOURCE_DIGEST,
        "signal": "[mean cross distance - .5 mean offdiagonal real distance - .5 mean offdiagonal fake distance] / RMS offdiagonal real distance; realconstant scale1",
        "signal_threshold": None, "sampling_draws_added": 0, "optimizer_updates_added": 0,
        "reference_scope": "first receipt-PASS generated state per phase; approximate target reference, not historical real batches or an oracle training mechanism",
        "artifacts": artifacts,
        "references": {key: {"step": item["step"], "metrics": item["metrics"]} for key, item in references.items()},
        "tasks": {}, "analytical_controls": {}}
    for task, records in tasks.items():
        output = []
        for item in records:
            metrics = item["metrics"]
            if task == "vector_overlap":
                reference, passed = references["overlap"], overlap_pass(metrics)
                severity = max(metrics["mean_error"] / .15, metrics["covariance_error"] / .45, metrics["sw1_normalized"] / .18)
            else:
                reference = references["gaussian_stationary" if item["step"] <= 4000 else "gaussian_shifted"]
                passed = gaussian_pass(metrics)
                severity = max(metrics["mean_error_sigma"] / .2, metrics["cdf_ks"] / .05,
                               abs(metrics["std_ratio"] - 1) / .2)
            value, raw = run_panels(reference["samples"], item["samples"])
            output.append({"step": item["step"], "receipt_pass": passed,
                "max_original_gate_violation_factor": severity, "energy_u": value, "raw_nn": raw})
        good = [item["energy_u"]["mean"] for item in output if item["receipt_pass"]]
        failed = [item["energy_u"]["mean"] for item in output if not item["receipt_pass"]]
        severe = [item["energy_u"]["mean"] for item in output if item["max_original_gate_violation_factor"] >= 2]
        result["tasks"][task] = {"good_state_means": scalar_summary(good), "failed_state_means": scalar_summary(failed),
            "severe_state_means": scalar_summary(severe), "passing_states": len(good), "failed_states": len(failed),
            "severe_states": len(severe), "auroc_pass_vs_fail": auroc(good, failed),
            "auroc_pass_vs_2x_original_gate_violation": auroc(good, severe), "observations": output}
    zeros = torch.zeros(128, 1, dtype=torch.float64)
    for name, real, fake in (("matched_constant", zeros + 3, zeros + 3),
                             ("shifted_constant", zeros + 3, zeros + 4)):
        value, raw = signal(real, fake)
        result["analytical_controls"][name] = {"energy_u": value, "raw_nn": raw,
                                               "unchanged_tau020_scale": thermal_scale(value)}
    for name in ("gaussian_stationary", "overlap"):
        real = references[name]["samples"][:128]
        fake = real.mean(0, keepdim=True).expand_as(real)
        value, raw = signal(real, fake)
        result["analytical_controls"][name + "_collapsed_to_reference_mean"] = {"energy_u": value, "raw_nn": raw,
                                                                               "unchanged_tau020_scale": thermal_scale(value)}
    corners = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]
    result["analytical_controls"]["four_modes_missing_one"] = population_atom_energy(corners, [.25] * 4, [1/3] * 3 + [0])
    result["analytical_controls"]["rare2percent_missing"] = population_atom_energy(corners, [.55, .30, .13, .02], [.57, .30, .13, 0])
    for count in (8, 16):
        angles = torch.arange(count, dtype=torch.float64) * (2 * torch.pi / count)
        points = torch.stack((angles.cos(), angles.sin()), 1).tolist()
        result["analytical_controls"][f"ring{count}_missing_one"] = population_atom_energy(
            points, [1/count] * count, [1/(count - 1)] * (count - 1) + [0])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = {key: value for key, value in result.items() if key != "tasks"}
    summary["tasks"] = {key: {k: v for k, v in item.items() if k != "observations"} for key, item in result["tasks"].items()}
    for task, item in result["tasks"].items():
        responses = {}
        for label, rows in (("PASS", [row for row in item["observations"] if row["receipt_pass"]]),
                            ("FAIL", [row for row in item["observations"] if not row["receipt_pass"]])):
            direct = [row["energy_u"]["unchanged_controller_response"]["pointwise_scale"]["mean"] for row in rows]
            ema = [row["energy_u"]["unchanged_controller_response"]["sequential_fixed_state_ema_last_scale"] for row in rows]
            responses[label] = {"mean_pointwise_scale_across_states": scalar_summary(direct),
                                "fixed_state_32panel_final_ema_scale": scalar_summary(ema)}
        summary["tasks"][task]["unchanged_tau020_response"] = responses
    summary["raw_probe"] = {"path": str(args.output), "sha256": digest(args.output)}
    summary["decision"] = "Do not implement or train the unchanged energy-U observer substitution: retained peak EMA reheats passing overlap states and analytical missing-mode controls cool prematurely."
    if args.summary_output:
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
