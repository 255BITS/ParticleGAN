"""Deterministic dissection of archived, already-scored ring outputs; no draws or training."""
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
FAILING = ("7ea5954f54ab44a497cd6a303828e836", "9f7ed4bb35be41f0a18e55e291606a4e")
INCUMBENT = "8235957023aa443bb6ddca752b291dbb"
GATE_KEYS = ("sample_count", "modes", "mass_tv", "hq", "component_covariance_error", "component_min_eigen_ratio")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_archive(path, expected, names):
    data = Path(path).read_bytes()
    assert sha(data) == expected
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert len(archive.getnames()) == len(set(archive.getnames()))
        return {name: archive.extractfile(name).read() for name in names}


def bind(data, attempt, durable, raw_prefix):
    envelope, result, certificate = [json.loads(data[durable + name])
        for name in ("request.json", "result.json", "evidence.json")]
    request = envelope["request"]
    raw, grading = [json.loads(data[raw_prefix + name]) for name in ("raw-result.json", "graded-result.json")]
    assert stable(result) == certificate["result_hash"]
    assert request["source"] == certificate["source"] and request["runtime"] == certificate["runtime"]
    assert result["attempt_id"] == attempt and result["candidate_revision"] == request["candidate_revision"]
    assert stable(raw) == grading["raw_hash"] and grading["source_digest"] == request["source"]["digest"]
    assert result["raw"]["grading"] == grading
    row, = result["task_results"]
    task = request["tasks"]["ring16_acquisition"]
    assert row["task_id"] == "ring16_acquisition" and row["raw_status"] == "completed"
    assert row["compatibility_key"] == envelope["job"]["compatibility_key"]
    assert row["gate_status"] == grading["grades"][row["task_id"]]["gate_status"]
    assert row["cost"]["completed_steps"] == task["execution"]["steps"] == 400
    assert row["cost"]["optimizer_updates"] == {"generator": 400, "discriminator": 400, "prior": 400}
    assert row["evidence"]["guards"]["all_finite"] and row["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert len(row["evidence"]["observations"]) == row["evaluator_result"]["convergence"]["observations"] == 24
    for key in ("recipe", "prior", "initializer", "initialization", "rng", "field_ownership"):
        assert row[key] == raw[key]
    for key, value in request["candidate"]["recipe_overrides"].items():
        if key in raw["recipe"]:
            assert raw["recipe"][key] == value, key
    assert raw["evidence"] == row["evidence"]
    assert raw["prior"] == task["execution"]["prior"]
    assert raw["initializer"] == task["execution"]["initializer"]
    assert task["execution"]["host_definition"] == raw["evidence"]["host"]["definition"]
    for observation in raw["evidence"]["observations"]:
        assert all(math.isfinite(observation[key]) for key in GATE_KEYS)
    return request, task, raw, row


@torch.no_grad()
def dissect(points, definition):
    assert points.shape == (4096, 2) and points.device.type == "cpu" and torch.isfinite(points).all()
    means, cov, target = [points.new_tensor(definition[key]) for key in ("means", "covariances", "masses")]
    labels = torch.cdist(points, means).argmin(1)
    delta = points - means[labels]
    mahal = torch.einsum("ni,nij,nj->n", delta, torch.linalg.inv(cov)[labels], delta)
    counts = torch.bincount(labels, minlength=len(means))
    hq_counts = torch.bincount(labels[mahal <= 9], minlength=len(means))
    radial = means / means.norm(dim=1, keepdim=True)
    tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=1)
    rows, centered, core_errors = [], torch.empty_like(points), []
    for k in range(len(means)):
        selected = labels == k
        member = points[selected]
        centered[selected] = member - member.mean(0)
        empirical = centered[selected].T @ centered[selected] / max(1, len(member))
        inverse = torch.linalg.inv(torch.linalg.cholesky(cov[k]))
        eigen = torch.linalg.eigvalsh(inverse @ empirical @ inverse.T)
        core = points[selected & (mahal <= 16)]
        core_x = core - core.mean(0)
        core_cov = core_x.T @ core_x / max(1, len(core))
        core_error = float((core_cov - cov[k]).norm() / cov[k].norm()) if len(core) >= 10 else 1.
        core_errors.append(core_error)
        rows.append({"component": k, "count": int(counts[k]), "hq_fraction": float(hq_counts[k] / max(1, int(counts[k]))),
            "centroid_offset_sigma": float((member.mean(0) - means[k]).norm() / cov[k, 0, 0].sqrt()) if len(member) else 0.,
            "covariance_relative_frobenius_error": float((empirical - cov[k]).norm() / cov[k].norm()) if len(member) >= 10 else 1.,
            "minimum_eigenvalue_ratio": float(eigen.min()) if len(member) >= 10 else 0., "maximum_eigenvalue_ratio": float(eigen.max()),
            "radial_variance_ratio": float(radial[k] @ empirical @ radial[k] / cov[k, 0, 0]),
            "tangential_variance_ratio": float(tangent[k] @ empirical @ tangent[k] / cov[k, 0, 0]),
            "four_sigma_core_covariance_error": core_error,
            "beyond_four_sigma_fraction": float((mahal[selected] > 16).float().mean()) if len(member) else 0.})
    error = sum(row["covariance_relative_frobenius_error"] for row in rows) / len(rows)
    mass = counts / len(points)
    metrics = {"sample_count": len(points), "modes": int((hq_counts / len(points) / target >= .25).sum()),
        "mass_tv": float((mass - target).abs().sum() / 2), "hq": float((mahal <= 9).float().mean()),
        "component_covariance_error": error,
        "component_min_eigen_ratio": min(row["minimum_eigenvalue_ratio"] for row in rows),
        "component_core_covariance_error": sum(core_errors) / len(rows)}
    energy = centered.square().sum(1)
    centered_mahal = torch.einsum("ni,nij,nj->n", centered, torch.linalg.inv(cov)[labels], centered)
    diagnostic = {"components_exceeding_aggregate_covariance_bound_individually": sum(
        row["covariance_relative_frobenius_error"] > .85 for row in rows),
        "mean_radial_variance_ratio": sum(row["radial_variance_ratio"] for row in rows) / len(rows),
        "mean_tangential_variance_ratio": sum(row["tangential_variance_ratio"] for row in rows) / len(rows),
        "mean_maximum_eigenvalue_ratio": sum(row["maximum_eigenvalue_ratio"] for row in rows) / len(rows),
        "mean_centroid_offset_sigma": sum(row["centroid_offset_sigma"] for row in rows) / len(rows),
        "hq_after_empirical_recentering_diagnostic_only": float((centered_mahal <= 9).float().mean()),
        "beyond_four_sigma_fraction": float((mahal > 16).float().mean()),
        "beyond_four_sigma_share_of_centered_covariance_energy": float(energy[mahal > 16].sum() / energy.sum()),
        "target_mean_distance_sigma_quantiles": dict(zip(("p50", "p85", "p90", "p95", "p99"),
            torch.quantile(mahal.sqrt(), points.new_tensor((.5, .85, .9, .95, .99))).tolist()))}
    return metrics, diagnostic, rows


def compact(attempt, request, task, raw, row):
    return {"attempt_id": attempt, "candidate_id": request["candidate"]["id"],
        "candidate_revision": request["candidate_revision"], "source_commit": request["source"]["origin_commit"],
        "source_digest": request["source"]["digest"], "gate_status": row["gate_status"],
        "convergence": row["evaluator_result"]["convergence"],
        "final_metrics": {key: row["metrics"][key] for key in (*GATE_KEYS, "component_core_covariance_error")},
        "recipe_controls": {key: raw["recipe"][key] for key in ("lr", "d_lr_mult", "prior_lr_mult", "reg_coeff",
            "reg_kappa", "input_noise_std", "output_noise_std", "network_lr_horizon_cap", "prior_reg", "total_steps")},
        "nominal_prior_base_lr": raw["recipe"]["lr"] * raw["recipe"]["prior_lr_mult"],
        "prior": raw["prior"], "initializer": raw["initializer"], "sampling_law": raw["evidence"]["sampling_law"],
        "scoring_weights": task["evaluation"]["scoring_weights"], "completed_steps": row["cost"]["completed_steps"],
        "optimizer_updates": row["cost"]["optimizer_updates"], "all_finite": True, "unintended_rng_deviations": 0,
        "selected_checkpoint_metrics": [{key: observation[key] for key in ("step", "hq", "component_covariance_error")}
            for observation in raw["evidence"]["observations"] if observation["step"] in (200, 300, 334, 367, 400)]}


def main():
    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    current_card = json.loads((ROOT / "reports/forge/k3p-global-tier1-v2/archive.json").read_text())
    old_card = json.loads((ROOT / "reports/forge/tier1-refresh/archive.json").read_text())
    current_names = ["inventory.json"]
    for attempt in FAILING:
        current_names += [f"durable/{attempt}/{name}" for name in ("request.json", "result.json", "evidence.json")]
        current_names += [f"raw/{attempt}/{name}" for name in ("raw-result.json", "graded-result.json", "observed-samples.pt")]
    data = read_archive(current_card["primary_archive_path"], current_card["archive_sha256"], current_names)
    assert sha(data["inventory.json"]) == current_card["inventory_sha256"]
    inventory = {entry["path"]: entry for entry in json.loads(data["inventory.json"])["entries"]}
    for name, value in data.items():
        if name != "inventory.json":
            assert sha(value) == inventory[name]["sha256"] and len(value) == inventory[name]["bytes"]
    durable = f"reports/forge/attempts/{INCUMBENT}/"
    raw_prefix = f"queue/{INCUMBENT}/"
    old_names = [durable + name for name in ("request.json", "result.json", "evidence.json")]
    old_names += [raw_prefix + name for name in ("raw-result.json", "graded-result.json")]
    old = read_archive(old_card["archive"]["path"], old_card["archive"]["sha256"], old_names)
    projection = json.loads((ROOT / f"reports/forge/technique-receipts/{INCUMBENT}.json").read_text())
    for record in projection["provenance"]["original_files"].values():
        assert sha(old[record["path"]]) == record["sha256"]
    baseline = bind(old, INCUMBENT, durable, raw_prefix)
    assert stable(json.loads(old[durable + "result.json"])) == projection["provenance"]["canonical_result_hash"]
    report = {"schema_version": 1, "id": "k3p-ring-broadness-forensics-v1",
        "scope": "saved_scored_ring_outputs_forensic_analysis",
        "source_digest": current_card["source_manifests"]["846190389b0e423054fc9912ab1a69d06bd5348c"]["digest"],
        "qualification_input": False, "training_updates_added": 0, "sampling_draws_added": 0,
        "reproduction": "python reports/forge/k3p-global-tier1-v3/ring-analysis.py",
        "archives": [{"path": current_card["primary_archive_path"], "sha256": current_card["archive_sha256"]},
                     old_card["archive"]], "thresholds": baseline[1]["evaluation"]["thresholds"],
        "incumbent": compact(INCUMBENT, *baseline), "failures": []}
    report["incumbent"]["saved_tensor_availability"] = "No scored tensors or ring checkpoints were retained; exact archived observations only."
    for attempt in FAILING:
        request, task, raw, row = bind(data, attempt, f"durable/{attempt}/", f"raw/{attempt}/")
        assert task["execution"] == baseline[1]["execution"]
        assert task["evaluation"]["thresholds"] == baseline[1]["evaluation"]["thresholds"]
        for source in ("benchmarks/toy_audit/ring16_quality.py", "benchmarks/transfer_suite/vector_tasks.py"):
            assert task["evaluation"]["sources"][source] == baseline[1]["evaluation"]["sources"][source]
        for field in ("prior", "initializer", "initialization"):
            assert raw[field] == baseline[2][field]
        assert raw["rng"]["bindings"] == baseline[2]["rng"]["bindings"]
        assert raw["evidence"]["host"]["models"] == baseline[2]["evidence"]["host"]["models"]
        assert raw["evidence"]["sampling_law"] == baseline[2]["evidence"]["sampling_law"]
        desc = raw["evidence"]["saved_observer_outputs"]
        retained = data[f"raw/{attempt}/{desc['path']}"]
        assert sha(retained) == desc["sha256"] and len(retained) == desc["bytes"]
        assert desc["sampling_draws_added"] == desc["optimizer_updates_added"] == 0
        records = torch.load(io.BytesIO(retained), weights_only=True, map_location="cpu")
        assert len(records) == desc["observation_count"] == 24
        largest_error = 0.
        for record, observed in zip(records, raw["evidence"]["observations"]):
            assert record["step"] == observed["step"]
            metrics, diagnostic, components = dissect(record["samples"], raw["evidence"]["host"]["definition"])
            for key, value in metrics.items():
                largest_error = max(largest_error, abs(value - observed[key]))
                assert math.isclose(value, observed[key], abs_tol=2e-6, rel_tol=2e-6), (attempt, key, value, observed[key])
        entry = compact(attempt, request, task, raw, row)
        entry.update(saved_outputs=desc, all_24_recomputed_gate_observations_match=True,
            maximum_recomputed_scalar_absolute_error=largest_error, final_diagnostics=diagnostic, final_components=components,
            actual_recipe_differences_from_incumbent={key: {"incumbent": baseline[2]["recipe"][key], "failed": value}
                for key, value in raw["recipe"].items() if value != baseline[2]["recipe"].get(key)})
        report["failures"].append(entry)
    assert torch.equal(rng_before, torch.get_rng_state())
    report["validation"] = {"exact_archive_and_target_member_hashes": True, "durable_result_certificates": True,
        "raw_grading_source_and_recipe_bindings": True, "same_task_execution_geometry_prior_budget": True,
        "same_thresholds_and_sample_scorer_sources": True, "same_named_initialization_and_model_initial_state_hashes": True,
        "same_initial_rng_bindings": True, "same_clean_live_serving_cohort": True, "cpu_rng_state_unchanged": True}
    report["interpretation"] = [
        "Mode occupancy and mass balance pass; all 16 components are individually too broad by the aggregate covariance bound.",
        "Tangential variance, tails and broad four-sigma cores explain local covariance failure better than centroid offsets alone. Recentring is an analysis diagnostic, not a changed gate.",
        "Both runs improve late in the fixed horizon. A larger generator LR does not improve final local quality; extending the task horizon is not justified by this forensic analysis.",
        "Historical comparison changes prior LR, critic LR and fake-only training output noise together; it does not isolate a causal control. Cap1600 and no cap are equivalent at 400 updates.",
        "No task geometry, prior, initialization, sampling-cohort or scoring implementation misbinding was found. The grading equality revision does not change these >=/<= gates.",
        "No saved ring checkpoint permits attribution to latent-table displacement or generator Jacobians. Original whole-candidate words remain UNKNOWN in these two trials."]
    report["bounded_global_search_recommendation"] = {"hold": {"lr": .006375, "reg_coeff": 1., "reg_kappa": 1.,
        "input_noise_std": .5, "output_noise_std": 0., "network_lr_horizon_cap": None},
        "existing_numeric_axes": {"nominal_prior_base_lr": [.0012, .0031875, .006375], "d_lr_mult": [.5, 1., 1.5]},
        "limits": "Use whole reusable recipes through all five unchanged Tier 1 gates; retain first-fail stops and full 26-task UNKNOWN denominator. Prior2 at LR.006375 previously failed ring, so faster is not presumed better. This suggestion changes no objective, technique, task resource or gate."}
    (REPORT / "ring-analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"report": str(REPORT / "ring-analysis.json"), "attempts": [INCUMBENT, *FAILING],
                      "training_updates_added": 0, "sampling_draws_added": 0, "validation": report["validation"]}))


if __name__ == "__main__":
    main()
