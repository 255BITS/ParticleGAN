"""Verify original Tier 2 receipts, archive bulk artifacts and export summaries."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.decision_contracts import evaluate
from experiments.forge.tier1_media import export_attempt, render
from reports.forge.regenerate_technique_inventory import project_receipt

STUDY = "bcap-tier2-smoothing-v1"
OUTPUT = ROOT / "reports/forge/bcap-tier2"


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_gaussian(evidence):
    import torch
    from benchmarks.toy_audit.gaussian1d_quality import score_samples
    from experiments.forge.gaussian_tasks import bounds
    root = Path(evidence["artifact_root"])
    verify_artifacts(root, evidence["artifact_manifest"])
    descriptor = evidence["saved_observer_outputs"]
    path = root / descriptor["path"]
    assert file_hash(path) == descriptor["sha256"]
    records = torch.load(path, map_location="cpu", weights_only=True)
    assert len(records) == 121
    assert [{"step": row["step"], **row["metrics"]} for row in records[1:]] == evidence["observations"]
    sets = 0
    for record in records:
        spec = deepcopy(evidence["host"]["definition"])
        if record["step"] > 4000:
            spec["means"] = [[3.0]]
        assert score_samples(record["samples"], spec) == record["metrics"]
        sets += 1
        if "frozen_samples" in record:
            assert score_samples(record["frozen_samples"], spec) == record["frozen_metrics"]
            sets += 1
    observations = evidence["observations"]
    return {"saved_sample_sets_recomputed": sets,
            "stationary_passes": sum(not bounds(row) for row in observations[:72]),
            "stationary_checks": 72,
            "first_stationary_failure_step": next(row["step"] for row in observations[:72] if bounds(row)),
            "shift_hold_passes": sum(not bounds(row) for row in observations[96:]),
            "shift_hold_checks": 24,
            "continuity": evidence["continuity"], "endpoint": observations[-1]}


def audit_word(evidence, local):
    import torch
    from experiments.forge.word_tasks import bounds
    descriptor = evidence["saved_observer_outputs"]
    path = local / descriptor["path"]
    assert file_hash(path) == descriptor["sha256"]
    records = torch.load(path, map_location="cpu", weights_only=True)
    assert len(records) == 25
    pairs, hits = 0, []
    for record, observed, confirm in zip(records, evidence["observations"], evidence["confirmations"]):
        assert {"step": record["step"], **record["metrics"]} == observed
        assert record["confirmation"] == confirm
        assert confirm["primary_state_sha256"] == confirm["confirmed_state_sha256"]
        assert confirm["training_state_unchanged"] is True
        pairs += 1
        if not bounds(observed) and not bounds(confirm["metrics"]):
            hits.append(record["step"])
    return {"saved_record_pairs_verified": pairs, "confirmed_checks": len(hits),
            "first_failed_check": next(record["step"] for record in records
                                       if bounds(record["metrics"]) or bounds(record["confirmation"]["metrics"])),
            "continuity": evidence["continuity"], "endpoint": evidence["observations"][-1]}


def main():
    queue_root = ROOT / "runs/forge"
    state_path = queue_root / "queue/state.json"
    before = file_hash(state_path)
    state = read_json(state_path)
    entries = [entry for entry in state["submissions"].values()
               if entry["request"].get("campaign_id") == STUDY]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] not in {"queued", "running", "paused"}
    request = entry["request"]
    campaign = state["campaigns"][STUDY]
    assert campaign["reserved_seconds"] == 0
    rows, attempts, new_attempts, audits, inputs, media = [], [], [], [], [], []
    executable_media_attempts, refused = [], []
    gaussian, word = None, None
    assignments = {a["task"]: a for a in request["view"]["assignments"]}
    for definition in request["jobs"]:
        if not all(assignments[name]["qualification_tier"] <= 2 for name in definition["task_ids"]):
            continue
        job = state["jobs"][definition["compatibility_key"]]
        assert job["status"] == "terminal" and len(job["attempts"]) == 1
        result = job["result"]
        attempt = result["attempt_id"]
        attempts.append(attempt)
        paid_here = job["cost_owner"]["campaign"] == STUDY
        if paid_here:
            new_attempts.append(attempt)
        receipt = project_receipt(ROOT, attempt)
        directory = ROOT / "reports/forge/attempts" / attempt
        envelope = read_json(directory / "request.json")
        producer = envelope["request"]
        assert producer["candidate_revision"] == request["candidate_revision"]
        assert producer["source"]["digest"] == request["source"]["digest"]
        assert producer["protocol"] == request["protocol"] and producer["protocol"]["seed"] == 0
        assert producer["runtime"] == request["runtime"]
        assert producer["compute_profiles"] == request["compute_profiles"]
        assert producer["execution_policy"]["mode"] == "complete_current_tier"
        local = Path(envelope["worker"]["directory"])
        raw = read_json(local / "raw-result.json")
        applied = raw.get("applied", raw)
        if "recipe" not in applied:
            assert raw["error"] == {"message": "dualnorm supports matrix weights and vector/scalar biases only",
                                    "type": "ValueError"}
            assert all(row["gate_status"] == "INCOMPLETE" and row["task_id"].startswith("img_")
                       for row in result["task_results"])
            for compact in receipt["task_results"]:
                name = compact["task_id"]
                rows.append({**compact, "tier": assignments[name]["qualification_tier"],
                             "importance": assignments[name]["importance"], "attempt_id": attempt,
                             "new_training": paid_here, "setup_error": raw["error"]})
                refused.append({"task_id": name, "attempt_id": attempt, "setup_error": raw["error"],
                                "training_started": False, "gif": None})
                audits.append({"task_id": name, "receipt_validated": True, "same_scientific_binding": True,
                               "training_started": False, "recipe_resolved_at_execution": False})
            inputs.extend((directory, local))
            continue
        if paid_here:
            executable_media_attempts.append(attempt)
        recipe = applied["recipe"]
        assert recipe["optimizer_family"] == "dualnorm" and recipe["optimizer_smoothing"] == 1e-5
        assert recipe["lr"] == .012 and recipe["d_lr_mult"] == 1.5 and recipe["prior_lr_mult"] == 2.5
        assert recipe["lr_floor"] == recipe["network_lr_floor"] == 1
        assert recipe.get("optimizer_momentum", 0) == 0
        schedules = applied.get("training_schedules", {}).get("optimizer_groups", {})
        for role in schedules.values():
            assert role["lr"]["minimum"] == role["lr"]["maximum"]
        for row, compact in zip(result["task_results"], receipt["task_results"]):
            name = row["task_id"]
            assert row["gate_status"] in {"PASS", "FAIL"}
            assert row["compatibility_key"] == definition["compatibility_key"]
            assert producer["tasks"][name] == request["tasks"][name]
            evidence = row["evidence"]
            thresholds = request["tasks"][name]["evaluation"].get("thresholds", [])
            from experiments.forge.decision_contracts import OPS
            observed = evidence.get("observations", [])
            failed_checks = [point["step"] for point in observed
                             if any(not OPS[op](point[metric], bound) for metric, op, bound in thresholds)]
            threshold_summary = {"primary_checks": len(observed),
                                 "primary_threshold_passes": len(observed) - len(failed_checks),
                                 "first_primary_threshold_failure_step": min(failed_checks) if failed_checks else None}
            checkpoint = evidence.get("provenance_checkpoint")
            if checkpoint:
                verify_artifacts(Path(checkpoint["artifact_root"]), checkpoint["artifact_manifest"])
            if "artifact_manifest" in evidence:
                verify_artifacts(Path(evidence["artifact_root"]), evidence["artifact_manifest"])
            if name == "gaussian1d_stability":
                gaussian = audit_gaussian(evidence)
            if name == "five_word_joint_hold":
                word = audit_word(evidence, local)
            proof = {key: evidence[key] for key in ("continuity", "continuation", "completed_steps", "guards")
                     if key in evidence}
            rows.append({**compact, "tier": assignments[name]["qualification_tier"],
                         "importance": assignments[name]["importance"], "attempt_id": attempt,
                         "new_training": paid_here, "continuation_proof": proof,
                         "threshold_observations": threshold_summary})
            audits.append({"task_id": name, "receipt_validated": True, "same_scientific_binding": True,
                           "constant_rate_groups": schedules,
                           "recipe_sha256": stable_hash(recipe), "checkpoint_artifacts_verified": bool(checkpoint)})
        inputs.extend((directory, local))
    tier2 = [row for row in rows if row["tier"] == 2 and row["importance"] == "required"]
    assert len(tier2) == len(new_attempts) == 21
    assert len(attempts) == 28 and sum(row["tier"] == 1 for row in rows) == 7
    charges = [charge for charge in state["charges"] if charge["attempt_id"] in new_attempts]
    assert len(charges) == 21 and len({charge["attempt_id"] for charge in charges}) == 21
    assert math.isclose(sum(charge["seconds"] for charge in charges), campaign["spent_seconds"], abs_tol=1e-8)
    assert gaussian is not None and word is not None
    counts = dict(Counter(row["gate_status"] for row in tier2))
    conclusion = (f"Frozen BCAP smoothing=1e-5 retains 6/6 required Tier 1 passes; Tier 2 records {counts}. "
                  "Gaussian retains only 1/72 stationary checks and misses shifted reacquisition/hold. "
                  "Words retain generation but lose inverse reconstruction; four image hosts reject convolution tensors before training. "
                  "No learning-rate annealing, scientific retries, Tier 1 reruns, Tier 3 or default adoption. "
                  "The frozen hypothesis signature misnames cdf_ks as ks; preserve its incomplete decision and report the actual scalar separately.")
    from experiments.forge import knowledge
    original_compile = knowledge.compile_memory
    knowledge.compile_memory = lambda *args, **kwargs: None  # Batch the reporting refresh after registration.
    try:
        record = knowledge.readout(ROOT, request["candidate"]["id"], conclusion,
            "Tier 1 evidence is reused only under identical source, recipe, task, protocol and runtime keys. "
            "This stage tests one frozen selected recipe; the unsmoothed binding is historical motivation rather than a new control.",
            "Keep the Tier 1 selection and these complete Tier 2 failures. Inspect saved retention and inverse-map trajectories; "
            "declare any smaller constant global rates or stronger fixed smoothing as a new bounded comparison. "
            "Convolution support requires a separately declared optimizer adaptation. Do not spend on Tier 3.", study_id=STUDY)
    finally:
        knowledge.compile_memory = original_compile
    full_record_path = queue_root / STUDY / "full-readout.json"
    atomic_json(full_record_path, record)
    record["task_results"] = rows
    record["source"] = {"path": "reports/forge/bcap-tier2/readout.json"}
    record.update(qualification_input=False, qualification_reuse=False,
                  full_readout_sha256=file_hash(full_record_path),
                  summary_curation="Final metrics and exact original receipt hashes retained; per-update evidence is archived locally.")
    record_path = ROOT / "reports/forge/records" / (record["record_id"] + ".json")
    atomic_json(record_path, record)
    before = file_hash(state_path)
    media_root = OUTPUT / "media"
    media_root.mkdir(exist_ok=True)
    guard = helper("saved_media_guard", ROOT / "reports/forge/gaussian-smoke-inventory/export_media.py")
    with guard.forbid_live_execution():
        for attempt in sorted(executable_media_attempts):
            directory = ROOT / "reports/forge/attempts" / attempt
            envelope = read_json(directory / "request.json")
            original = read_json(directory / "result.json")
            native_rows = [row for row in original["task_results"]
                           if envelope["request"]["tasks"][row["task_id"]]["adapter"] == "native100"]
            rendered_rows = [guard.render_native(envelope["request"]["tasks"][row["task_id"]], row,
                             media_root / (row["task_id"] + ".gif")) for row in native_rows]
            spiral_rows = [row for row in original["task_results"] if row["task_id"] == "vector_spiral"]
            for row in spiral_rows:
                # The spiral has a procedural target rather than mixture centers.
                # Use its complete saved numerical learning curve in the standard renderer.
                measured = deepcopy(row)
                measured["evidence"].pop("saved_observer_outputs", None)
                rendered_rows.append(render(envelope["request"]["tasks"][row["task_id"]], measured,
                    Path(envelope["worker"]["directory"]), media_root / (row["task_id"] + ".gif")))
            if not native_rows and not spiral_rows:
                rendered_rows = export_attempt(directory, media_root)
            for rendered in rendered_rows:
                name = rendered["task_id"]
                media.append({"task_id": name, "attempt_id": attempt, "gif": name + ".gif",
                              "gif_sha256": file_hash(media_root / (name + ".gif")),
                              "renderer_receipt": name + ".json",
                              "renderer_receipt_sha256": file_hash(media_root / (name + ".json")),
                              "recorded_grade": rendered["recorded_grade"]})
                print("media", name, rendered["observation_count"], flush=True)
    assert len(media) + len(refused) == 21
    atomic_json(media_root / "index.json", {"schema_version": 1, "study_id": STUDY,
        "candidate_id": request["candidate"]["id"], "tasks": sorted(media, key=lambda row: row["task_id"]),
        "setup_refusals_without_training": refused,
        "optimizer_updates_added": 0, "sampling_draws_added": 0, "qualification_input": False,
        "exporter_sha256": file_hash(Path(__file__))})
    inputs.extend((queue_root / STUDY, queue_root / "queue", queue_root / "events.jsonl",
                   Path(request["source"]["snapshot_path"]), OUTPUT / "run.py", Path(__file__),
                   ROOT / "configs/forge/studies" / (STUDY + ".json")))
    archiver = helper("bcap_archive", ROOT / "reports/forge/bcap-six/publish.py")
    archive = archiver.archive(ROOT, inputs, ROOT / "artifacts/bcap-tier2-smoothing-v1.tar.gz")
    original_decision = evaluate(request, [state["jobs"][j["compatibility_key"]]["result"]["task_results"][0]
                                         for j in request["jobs"]
                                         if state["jobs"][j["compatibility_key"]].get("result")])
    assert file_hash(state_path) == before
    atomic_json(OUTPUT / "readout.json", {"schema_version": 1, "study_id": STUDY,
        "candidate_id": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
        "request_id": request["request_id"], "source_digest": request["source"]["digest"],
        "source_origin_commit": request["source"]["origin_commit"], "runtime": request["runtime"],
        "compute_profiles": request["compute_profiles"], "view_revision": request["view"]["revision"],
        "policy_fingerprint": request["policy_fingerprint"], "execution_policy": request["execution_policy"],
        "campaign": campaign, "new_attempt_count": 21, "scientific_retries": 0,
        "completed_training_attempts": len(executable_media_attempts), "setup_refusals": refused,
        "reused_attempt_count": 7, "reused_cost_added": 0, "tier2_counts": dict(Counter(row["gate_status"] for row in tier2)),
        "tier1_required_counts": {"PASS": 6}, "tier3": "not_requested", "default_adoption": False,
        "tasks": sorted(rows, key=lambda row: (row["tier"], row["task_id"])), "audits": audits,
        "gaussian_saved_sample_audit": gaussian, "word_saved_pair_audit": word, "archive": archive,
        "frozen_study_decision": original_decision,
        "concluded_readout_record": record_path.relative_to(ROOT).as_posix(),
        "full_readout_sha256": file_hash(full_record_path),
        "signature_reporting_correction": {
            "declared_metric": "ks", "actual_metric": "cdf_ks", "declared_study_preserved": True,
            "value": gaussian["endpoint"]["cdf_ks"], "threshold": .05,
            "corrected_scalar_outcome": "falsified" if gaussian["endpoint"]["cdf_ks"] > .05 else "prediction_observed",
            "scope": "Display-only correction to the misnamed hypothesis signature; original task gates and receipts unchanged.",
            "qualification_input": False},
        "qualification_input": False, "qualification_reuse": False,
        "publisher_sha256": file_hash(Path(__file__)), "queue_state_sha256": before})
    print(dict(Counter(row["gate_status"] for row in tier2)), "paid seconds", campaign["spent_seconds"], flush=True)


if __name__ == "__main__":
    main()
