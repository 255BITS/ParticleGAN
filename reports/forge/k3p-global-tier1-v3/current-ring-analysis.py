"""Dissect this frozen factorial study's saved ring samples without draws or training."""
import importlib.util
import json
import math
from pathlib import Path

import torch

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
spec = importlib.util.spec_from_file_location("ring_forensics", REPORT / "ring-analysis.py")
forensics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forensics)


def main():
    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    study_path = ROOT / "reports/forge/configuration-search/k3p-global-tier1-v3.json"
    study = json.loads(study_path.read_text())
    assert study["selection"]["all_trials_terminal"] and len(study["trials"]) == 8
    entries, keyed, source = [], {}, None
    baseline = None
    for trial in study["trials"]:
        ring = next(row for row in trial["tasks"] if row["task"] == "ring16_acquisition")
        attempt = ring["attempt_id"]
        durable = ROOT / "reports/forge/attempts" / attempt
        certificate = json.loads((durable / "evidence.json").read_text())
        local = Path(certificate["local_artifact_root"])
        assert local.resolve().is_relative_to((ROOT / "runs/forge/k3p-global-tier1-v3").resolve())
        data = {"durable/" + name: (durable / name).read_bytes()
                for name in ("request.json", "result.json", "evidence.json")}
        data.update({"raw/" + name: (local / name).read_bytes()
                     for name in ("raw-result.json", "graded-result.json")})
        request, task, raw, row = forensics.bind(data, attempt, "durable/", "raw/")
        assert request["source"]["origin_commit"] == "72da7275034422bc171f9b9b1955c9b150cda045"
        assert row["gate_status"] == ring["gate_status"] == "FAIL"
        assert request["candidate"]["id"] == trial["candidate_id"]
        assert request["candidate_revision"] == trial["candidate_revision"]
        if source is None:
            source = request["source"]
            assert forensics.stable(source["files"]) == source["digest"] == study["source_digest"]
            snapshot = Path(request["queue_root"]) / "snapshots" / source["digest"]
            for relative, expected in source["files"].items():
                assert forensics.sha((snapshot / relative).read_bytes()) == expected, relative
            baseline = request, task, raw
        assert request["source"] == source and request["runtime"] == baseline[0]["runtime"]
        assert {key: value for key, value in task.items() if key != "field_ownership"} == {
            key: value for key, value in baseline[1].items() if key != "field_ownership"}
        for field in ("prior", "initializer", "initialization", "rng"):
            assert raw[field] == baseline[2][field]
        assert raw["evidence"]["host"]["models"] == baseline[2]["evidence"]["host"]["models"]
        assert raw["evidence"]["sampling_law"] == baseline[2]["evidence"]["sampling_law"]
        assert raw["recipe"]["lr"] == .006375 and raw["recipe"]["output_noise_std"] == 0.
        allowed = {"reg_coeff", "d_lr_mult", "prior_lr_mult"}
        assert {name for name, value in raw["recipe"].items()
                if value != baseline[2]["recipe"].get(name)} <= allowed
        ownership = task["field_ownership"]
        reference = baseline[1]["field_ownership"]
        assert {key: value for key, value in ownership.items() if key != "recipe_fields"} == {
            key: value for key, value in reference.items() if key != "recipe_fields"}
        assert {name for name, value in ownership["recipe_fields"].items()
                if value != reference["recipe_fields"].get(name)} <= allowed
        descriptor = raw["evidence"]["saved_observer_outputs"]
        path = local / descriptor["path"]
        retained = path.read_bytes()
        assert forensics.sha(retained) == descriptor["sha256"] and len(retained) == descriptor["bytes"]
        assert descriptor["optimizer_updates_added"] == descriptor["sampling_draws_added"] == 0
        records = torch.load(path, map_location="cpu", weights_only=True)
        assert len(records) == descriptor["observation_count"] == 24
        max_error = 0.
        for record, observed in zip(records, raw["evidence"]["observations"]):
            assert record["step"] == observed["step"]
            metrics, diagnostic, components = forensics.dissect(record["samples"], raw["evidence"]["host"]["definition"])
            for name, value in metrics.items():
                max_error = max(max_error, abs(value - observed[name]))
                assert math.isclose(value, observed[name], abs_tol=2e-6, rel_tol=2e-6), (attempt, name)
        points = records[-1]["samples"]
        means = points.new_tensor(raw["evidence"]["host"]["definition"]["means"])
        assignment = torch.cdist(points, means).argmin(1)
        mahal = ((points - means[assignment]) / .1).square().sum(1)
        radial_offset_sigma = ((points - means[assignment]) * means[assignment] / 3).sum(1) / .1
        diagnostic.update(beyond_four_sigma_radially_inward_count=int(((mahal > 16) & (radial_offset_sigma < 0)).sum()),
            beyond_four_sigma_radially_outward_count=int(((mahal > 16) & (radial_offset_sigma >= 0)).sum()),
            beyond_six_sigma_radially_inward_count=int(((mahal > 36) & (radial_offset_sigma < 0)).sum()),
            beyond_six_sigma_radially_outward_count=int(((mahal > 36) & (radial_offset_sigma >= 0)).sum()),
            minimum_origin_radius=float(points.norm(dim=1).min()), maximum_origin_radius=float(points.norm(dim=1).max()))
        for component in components:
            member = assignment == component["component"]
            component.update(beyond_four_sigma_count=int((mahal[member] > 16).sum()),
                beyond_six_sigma_count=int((mahal[member] > 36).sum()),
                minimum_radial_offset_sigma=float(radial_offset_sigma[member].min()),
                maximum_radial_offset_sigma=float(radial_offset_sigma[member].max()),
                maximum_target_mean_distance_sigma=float(mahal[member].sqrt().max()))
        worst = sorted(components, key=lambda row: row["covariance_relative_frobenius_error"], reverse=True)
        rest_error = (metrics["component_covariance_error"] * 16 - worst[0]["covariance_relative_frobenius_error"]) / 15
        entry = forensics.compact(attempt, request, task, raw, row)
        entry.update(configuration_id=trial["configuration_id"], saved_outputs=descriptor,
            final_diagnostics=diagnostic, worst_three_components=worst[:3],
            covariance_error_of_other_fifteen_components_diagnostic_only=rest_error,
            beyond_four_sigma_count=int((mahal > 16).sum()), beyond_six_sigma_count=int((mahal > 36).sum()),
            full_covariance_gate_rejects=True,
            four_sigma_core_covariance_diagnostic_below_full_gate_bound=metrics["component_core_covariance_error"] <= .85,
            all_24_recomputed_gate_observations_match=True, maximum_recomputed_scalar_absolute_error=max_error,
            provenance=[{"path": str(path.relative_to(ROOT)), "sha256": forensics.sha(path.read_bytes()), "bytes": path.stat().st_size}
                for path in (durable / "request.json", durable / "result.json", durable / "evidence.json", local / "raw-result.json", local / "graded-result.json")])
        entries.append(entry)
        key = tuple(raw["recipe"][field] for field in ("reg_coeff", "d_lr_mult", "prior_lr_mult"))
        assert key not in keyed
        keyed[key] = entry
    assert set(keyed) == {(c, d, p) for c in (.5, 1.) for d in (.5, 1.) for p in (.5, 1.)}
    pairs = []
    axes = ("reg_coeff", "d_lr_mult", "prior_lr_mult")
    for index, axis in enumerate(axes):
        for key, low in sorted(keyed.items()):
            if key[index] != .5:
                continue
            upper = list(key)
            upper[index] = 1.
            high = keyed[tuple(upper)]
            pairs.append({"axis": axis, "low_value": .5, "high_value": 1.,
                "fixed": {name: value for i, (name, value) in enumerate(zip(axes, key)) if i != index},
                "low_configuration": low["configuration_id"], "high_configuration": high["configuration_id"],
                "high_minus_low_final_metrics": {name: high["final_metrics"][name] - low["final_metrics"][name]
                    for name in ("hq", "component_covariance_error", "component_core_covariance_error")}})
    assert torch.equal(rng_before, torch.get_rng_state())
    report = {"schema_version": 1, "id": "k3p-current-ring-tail-forensics-v1",
        "scope": "saved_scored_ring_outputs_forensic_analysis", "source_digest": source["digest"],
        "source_commit": source["origin_commit"], "qualification_input": False,
        "training_updates_added": 0, "sampling_draws_added": 0, "cpu_rng_state_unchanged": True,
        "reproduction": "python reports/forge/k3p-global-tier1-v3/current-ring-analysis.py",
        "reproducer_dependency": {"path": "reports/forge/k3p-global-tier1-v3/ring-analysis.py",
            "sha256": forensics.sha((REPORT / "ring-analysis.py").read_bytes())},
        "study_report": {"path": str(study_path.relative_to(ROOT)), "sha256": forensics.sha(study_path.read_bytes())},
        "thresholds": baseline[1]["evaluation"]["thresholds"], "candidate_count": len(entries),
        "validation": {"durable_certificates_and_raw_grade_hashes": True,
            "complete_executed_source_snapshot_hashes": True, "same_task_geometry_resources_and_gates": True,
            "same_prior_initialization_rng_runtime_and_clean_live_sampling": True,
            "only_declared_three_global_numeric_fields_differ": True, "all_192_saved_check_observations_reproduce": True},
        "candidates": entries, "matched_numeric_contrasts": pairs,
        "interpretation": [
            "All eight global recipes fail the unchanged full-component covariance gate; their word cells remain UNKNOWN.",
            "Core shape is substantially better than full shape in several arms. The full gate measures all served mass, so excluding four-sigma tails would change the experiment's question.",
            "At coeff1/D1/prior1, 200 of4096 points beyond four sigma carry34.34% of centered covariance energy. Worst component12 has full error8.0644 versus core.3297; even the other15 components average1.0478, above.85.",
            "At coeff1/D.5/prior1, 425 points beyond four sigma carry66.66% of energy; worst component1 reaches20.66 sigma and full error33.50 versus core.1612. This is substantial served outlier mass, not an unbound scalar or scorer defect.",
            "Within these exact four matched contrasts, increasing D multiplier.5 to1 improves HQ and full covariance error in every case. Prior multiplier1 improves HQ in all four cases, but full covariance is not strictly monotonic. Coefficient1 improves full covariance in all four pairs while often lowering HQ.",
            "These are one fixed-initialization bounded grid's numerical contrasts, not a general optimizer claim. No ring checkpoints were retained to attribute tails to prior rows or generator derivatives; no new sampling or gate relaxation was performed."]}
    (REPORT / "current-ring-analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"candidate_count": len(entries), "matched_contrasts": len(pairs),
        "source_commit": source["origin_commit"], "source_digest": source["digest"],
        "training_updates_added": 0, "sampling_draws_added": 0, "validation": report["validation"]}))


if __name__ == "__main__":
    main()
