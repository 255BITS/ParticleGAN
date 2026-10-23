#!/usr/bin/env python3
"""Reduce final certified saved evidence; never construct a model or sample.

The final publisher owns campaign/source/cost/stream certification. This reducer
requires its complete fifteen-cell full readouts and binds their final hashes.
No partial projection or changing queue-state hash is used as final evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
SOURCE_COMMIT = "a03373da3421303b89c99a39898b091e22910134"
SOURCE_DIGEST = "668914920afda7bb6fc53d9c918aac92a9c9cd5ae582884fb9f068ff1b8249e8"
ARMS = {"original": "bispf-optimized-quintic-v1", "energy": "bispf-output-energy-v1",
        "game": "bispf-bregman-game-v1"}
REGRESSION_CONTROLS = {"gaussian1d_smoke", "ring16_acquisition", "mode_hold", "img_stripes2"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(raw_root):
    queue = raw_root / "queue"
    state = read(queue / "queue/state.json")
    output = {}
    for path in (queue / "queue/requests").glob("*.json"):
        request = read(path)
        if request["candidate"]["id"] in ARMS.values():
            output[request["candidate"]["id"]] = dict(Counter(
                job["status"] for job in state["jobs"].values()
                if request["request_id"] in job["subscribers"]))
    return output


def scalar_summary(values):
    values = [float(v) for v in values]
    assert values and all(math.isfinite(v) for v in values)
    width = max(1, len(values) // 10)
    return {"count": len(values), "first": values[0], "last": values[-1],
            "min": min(values), "max": max(values), "mean": math.fsum(values) / len(values),
            "first_decile_mean": math.fsum(values[:width]) / width,
            "last_decile_mean": math.fsum(values[-width:]) / width}


def update_trace(row):
    receipt = row.get("evidence", {}).get("public_update_trace")
    if receipt is None:
        return {"status": "UNAVAILABLE"}
    path = Path(receipt["path"])
    assert sha(path) == receipt["sha256"], "public trace identity changed"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(records) == receipt["rows"]
    first, last = records[0]["step"], records[-1]["step"]
    assert [r["step"] for r in records] == list(range(first, last + 1))
    fields, invalid = {}, Counter()
    for record in records:
        for section in ("bregman_stats", "penalty_stats"):
            for key, value in record.get(section, {}).items():
                # The campaign publisher owns the unchanged regularizer audit.
                # Retain its thermal scalar and every game diagnostic here.
                if section == "penalty_stats" and key != "s_thermal":
                    continue
                if type(value) in (int, float) and math.isfinite(value):
                    fields.setdefault(section + "." + key, []).append(value)
                elif value is None or (type(value) in (int, float) and not math.isfinite(value)):
                    invalid[section + "." + key] += 1
    game = [r for r in records if "bregman_stats" in r]
    if row["recipe"].get("bispf_bregman_game"):
        assert len(game) == len(records), "game diagnostics missing public updates"
    else:
        assert not game, "inactive reference unexpectedly reports game diagnostics"
    return {"status": "VERIFIED", "path": str(path), "sha256": receipt["sha256"],
            "updates": len(records), "first_step": first, "last_step": last,
            "bregman_diagnostic_updates": len(game),
            "scalar_summary": {k: scalar_summary(v) for k, v in fields.items()},
            "nonfinite_or_null_counts": dict(invalid),
            "scope": "G-phase pre-move batch, returned after completed public update; canonical adversarial mean-loss force before/after thermal, not realized parameter motion; row routes are not modes"}


def image_pixels(path):
    import torch
    records = torch.load(path, map_location="cpu", weights_only=False)
    history = {"output_max": [], "output_min": [], "output_mean": []}
    values = []
    for record in records:
        samples = record["samples"]
        for name, value in (("output_max", samples.max()), ("output_min", samples.min()),
                            ("output_mean", samples.mean())):
            history[name].append(float(value))
        if record is not records[-1]:
            continue
        images, templates = record["samples"].double(), record["targets"].double()
        distances = (images[:, None] - templates[None]).square().mean(tuple(range(2, images.ndim + 1)))
        minimum, nearest = distances.min(1)
        threshold = .06 if len(templates) == 2 and bool((templates.max() < 1)) else .10
        clean = minimum <= threshold**2
        mse = minimum.mean()
        values.append({"step": record["step"], "output_min": float(images.min()),
                       "output_max": float(images.max()), "output_mean": float(images.mean()),
                       "mean_pixel_mse": float(mse), "aggregate_pixel_rmse": float(mse.sqrt()),
                       "near_zero_pixel_fraction": float((images <= 1e-6).double().mean()),
                       "near_one_pixel_fraction": float((images >= 1-1e-6).double().mean()),
                       "exact_endpoint_pixel_fraction": float(((images == 0) | (images == 1)).double().mean()),
                       "nearest_template_counts": nearest.bincount(minlength=len(templates)).tolist(),
                       "clean_template_counts": nearest[clean].bincount(minlength=len(templates)).tolist(),
                       "diagnostic_rmse_threshold": threshold,
                       "original_scored_metrics": record["metrics"]})
    smallest_maximum = min(range(len(records)), key=lambda i: history["output_max"][i])
    return {"artifact": str(path), "sha256": sha(path), "observations": len(records),
            "scope": "retained already-scored samples only; original task gates remain authoritative",
            "pixel_trend": {name: scalar_summary(series) for name, series in history.items()},
            "minimum_observed_output_max": {"step": records[smallest_maximum]["step"],
                                            "value": history["output_max"][smallest_maximum]},
            "endpoint": values[0]}


def saved_reducer():
    path = ROOT / "reports/forge/three-theme-repairs-20261022/fidelity/analyze_saved.py"
    spec = importlib.util.spec_from_file_location("previous_fidelity_saved_reducer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                    "used": ["curve_summary", "width_decomposition", "image_decomposition"],
                    "not_called": "width_center_census (would add a forward)"}


def reduce(raw_root, destination):
    packets, paths = {}, {}
    for name, candidate in ARMS.items():
        path = raw_root / "full-readouts" / (candidate + ".json")
        if not path.is_file():
            raise RuntimeError("final complete full readout not available: " + candidate)
        packet = read(path)
        if packet.get("complete") is not True:
            raise RuntimeError("full readout is still partial: " + candidate)
        assert packet["source"]["origin_commit"] == SOURCE_COMMIT
        assert packet["source"]["digest"] == SOURCE_DIGEST
        assert len(packet["task_results"]) == len(packet["expected_tasks"]) == 15
        assert {r["task_id"] for r in packet["task_results"]} == set(packet["expected_tasks"])
        packets[name], paths[name] = packet, path
    import torch
    torch.set_num_threads(1)
    helper, helper_receipt = saved_reducer()
    from reports.forge.regenerate_technique_inventory import _evaluator_summary
    result = {"schema": 1, "scientific_source_commit": SOURCE_COMMIT,
              "scientific_source_digest": SOURCE_DIGEST,
              "analysis_source_sha256": sha(Path(__file__)), "saved_helper": helper_receipt,
              "scope": "final saved-only research diagnostic; zero extra model forwards, sampling draws, optimizer updates or gate changes; no qualification",
              "campaign_certification_owner": "../publish.py and ../comparison.json",
              "arms": {}}
    for name, packet in packets.items():
        tasks = {}
        receipt_paths = {p["attempt"]: Path(p["local_artifact_root"]) for p in packet["original_receipts"]}
        for row in packet["task_results"]:
            task = row["task_id"]
            cell = {"status": row["gate_status"], "reason": row.get("reason"),
                    "metrics": row.get("metrics", {}),
                    "evaluator_summary": _evaluator_summary(row.get("evaluator_result", {})),
                    "attempt": row.get("_attempt_id")}
            if row.get("_attempt_id") is None:
                assert cell["status"] == "BLOCKED"
                tasks[task] = cell
                continue
            if cell["status"] not in ("PASS", "FAIL"):
                cell["artifact_diagnostics"] = "UNAVAILABLE: no completed scored task law; preserve the original terminal status"
                tasks[task] = cell
                continue
            assert row["recipe"].get("bispf_bregman_game", False) is (name == "game")
            attempt = receipt_paths[row["_attempt_id"]]
            cell["curve"] = helper.curve_summary(row, attempt)
            if not task.startswith(("vector_", "img_")):
                # Aggregate acquisition/continuation criteria are not flat
                # per-observation bounds. Their own evaluator stays authoritative.
                for key in ("per_gate_observation_counts", "joint_passing_checks", "longest_joint_run"):
                    cell["curve"].pop(key, None)
                cell["curve"]["gate_scope"] = "original acquisition/phase/native evaluator summary; no synthetic per-observation joint gate"
            if task in ("grid100", "rotated100", "staggered100"):
                declaration = ROOT / "configs/forge/tasks" / (task + ".json")
                relative = str(declaration.relative_to(ROOT))
                assert sha(declaration) == packet["source"]["files"][relative]
                evaluation = read(declaration)["evaluation"]
                cell["native_gate_contract"] = {
                    "task_path": relative, "task_sha256": sha(declaration),
                    **{k: evaluation[k] for k in ("coverage_thresholds", "accuracy_limits",
                        "minimum_stable_checks", "eval_samples", "holdout_samples", "eval_output_noise")},
                    "accuracy_source_sha256": evaluation["sources"]["benchmarks/toy100/accuracy.py"],
                    "undefined_shape_condition": "original estimator returns null conditional shape/centering if any mode has fewer than two within-3-sigma samples",
                    "diagnostic_not_gate_fields": ["center_max_sigma", "cov_frob_rms"]}
            cell["update_trace"] = update_trace(row)
            if task.startswith("img_"):
                observer = row["evidence"]["saved_observer_outputs"]
                artifact = attempt / observer["path"]
                assert sha(artifact) == observer["sha256"]
                cell["saved_pixels"] = image_pixels(artifact)
                if task == "img_intensity2":
                    cell["intensity_decomposition"] = helper.image_decomposition(artifact)
                    mse = cell["saved_pixels"]["endpoint"]["mean_pixel_mse"]
                    cell["intensity_decomposition"]["mean_pixel_mse"] = mse
                    cell["intensity_decomposition"]["component_pixel_mse"] = {
                        k: value * mse for k, value in
                        cell["intensity_decomposition"]["squared_error_fraction"].items()}
            if task in ("vector_unequal_width", "vector_unequal_mass", "vector_anisotropic"):
                observer = row["evidence"]["saved_observer_outputs"]
                artifact = attempt / observer["path"]
                assert sha(artifact) == observer["sha256"]
                cell["component_tails"] = helper.width_decomposition(artifact, row["evidence"]["host"]["definition"])
            tasks[task] = cell
        result["arms"][name] = {"candidate": ARMS[name], "candidate_revision": packet["candidate_revision"],
                                 "full_readout": {"path": str(paths[name]), "sha256": sha(paths[name])},
                                 "original_immutable_receipts": packet["original_receipts"],
                                 "outcomes": dict(Counter(c["status"] for c in tasks.values())), "tasks": tasks}
    game = result["arms"]["game"]["tasks"]
    result["comparisons"] = {}
    for name in ("original", "energy"):
        reference = result["arms"][name]["tasks"]
        result["comparisons"][name] = {
            "new_passes": [t for t in game if game[t]["status"] == "PASS" and reference[t]["status"] != "PASS"],
            "lost_passes": [t for t in game if game[t]["status"] != "PASS" and reference[t]["status"] == "PASS"],
            "game_passing_controls": sorted(t for t in REGRESSION_CONTROLS if game[t]["status"] == "PASS"),
            "unchanged_gates": sum(game[t]["status"] == reference[t]["status"] for t in game),
            "native_holdout_precision_deltas": {
                t: game[t]["metrics"]["precision"] - reference[t]["metrics"]["precision"]
                for t in ("grid100", "rotated100", "staggered100")
                if type(game[t]["metrics"].get("precision")) in (int, float)
                and type(reference[t]["metrics"].get("precision")) in (int, float)},
            "no_taskwise_recipe_selection": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"output": str(destination), "sha256": sha(destination),
            "outcomes": {name: arm["outcomes"] for name, arm in result["arms"].items()},
            "comparisons": result["comparisons"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("measured-analysis.json"))
    parser.add_argument("--status", action="store_true", help="Print live counts only; do not bind mutable partial files")
    options = parser.parse_args()
    print(json.dumps(progress(options.raw_root) if options.status else reduce(options.raw_root, options.output), indent=2))
