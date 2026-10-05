"""Independently verify the final BCAP paid matrix; no training or selection."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import torch

from benchmarks.toy_audit.gaussian1d_quality import score_samples as gaussian_score
from benchmarks.toy_audit.ring16_quality import score_samples as ring_score
from experiments.forge.contracts import file_hash, read_json, stable_hash


TASKS = {"gaussian1d_acquisition", "ring16_acquisition", "two_pole", "unused_token_hold",
         "ae_gan_hold", "five_word_joint_acquisition", "schedule_contract_audit"}
CAMPAIGNS = {"bcap-tier1-repair-rates-v1": 8, "bcap-tier1-repair-moments-v1": 4,
             "bcap-original-horizon-diagnostics-v1": 1}


def failed(point, thresholds):
    compare = {"<=": lambda x, y: x <= y, ">=": lambda x, y: x >= y, "==": lambda x, y: x == y}
    return [key for key, op, bound in thresholds
            if type(point.get(key)) not in (int, float) or not math.isfinite(point[key])
            or not compare[op](point[key], bound)]


def curve(task, evidence):
    points = evidence["observations"]
    expected = task["evaluation"].get("observation_steps") or sorted({
        math.ceil(i * task["execution"]["steps"] / 24) for i in range(1, 25)})
    assert [p["step"] for p in points] == expected, (task["id"], [p["step"] for p in points], expected)
    assert {b[0] for b in task["evaluation"]["thresholds"]} <= evidence["live"].keys()
    assert all(points[-1].get(key) == value for key, value in evidence["live"].items()), (task["id"], "live/endpoint disagreement")
    assert task["evaluation"]["minimum_stable_checks"] == 5
    failures = [failed(p, task["evaluation"]["thresholds"]) for p in points]
    suffix = 0
    for value in reversed(failures):
        if value:
            break
        suffix += 1
    return "PASS" if suffix >= 5 else "FAIL", suffix, failures


def check_schedule(task, evidence):
    audit = evidence["schedule_contract"]
    assert audit["clockfree_claim"] is False
    assert audit["schedule_observations"] == 20
    assert audit["guard_parameter_checks"] > 0 and audit["guard_released_parameter_checks"] > 0
    assert set(c["condition"] for c in evidence["comparisons"]) == set(task["evaluation"]["conditions"])
    errors = (audit["maximum_schedule_error"] > task["evaluation"]["schedule_tolerance"]
              or audit["maximum_guard_relative_error"] > task["evaluation"]["guard_relative_tolerance"]
              or audit["restart_cadence_failures"] != 0 or audit["schedule_replay_state_mismatches"] != 0)
    return "FAIL" if errors else "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rng = torch.get_rng_state().clone()
    report_path = root / "reports/forge/bcap-tier1-repair/results.json"
    report = read_json(report_path)
    assert report["status"] == "COMPLETE" and report["attempt_count"] == 86
    assert report["qualification_input"] is False and len(report["candidates"]) == 13
    assert Counter(r["campaign"] for r in report["candidates"]) == Counter(CAMPAIGNS)
    costs, counts, observed, by_task, late_failures = defaultdict(float), Counter(), {}, defaultdict(Counter), defaultdict(Counter)
    sources, vector_checks, curve_checks = {}, 0, 0
    receipt_checks, canonical_manifest, row_map = 0, {}, {}
    endpoint_pass_suffix_fail = Counter()
    endpoint_pass_suffix_fail_by_campaign = defaultdict(Counter)
    for row in report["candidates"]:
        assert row["submission_status"] not in {"queued", "running", "paused"}
        cells = row["tasks"]
        assert len(cells) == (2 if row["campaign"] == "bcap-original-horizon-diagnostics-v1" else 7)
        if len(cells) == 7:
            assert {cell["task_id"] for cell in cells} == TASKS
        assert row["counts"] == dict(Counter(t["status"] for t in cells))
        row_map[row["candidate"]] = row
        for cell in cells:
            attempt = cell["attempt_id"]
            assert attempt not in observed
            durable = root / "reports/forge/attempts" / attempt
            envelope, result, certificate = [read_json(durable / name) for name in
                ("request.json", "result.json", "evidence.json")]
            request, job = envelope["request"], envelope["job"]
            local = Path(certificate["local_artifact_root"])
            raw, graded = read_json(local / "raw-result.json"), read_json(local / "graded-result.json")
            canonical_hash = stable_hash(result)
            assert canonical_hash == certificate["result_hash"] == cell["canonical_result_sha256"]
            assert stable_hash(raw) == graded["raw_hash"] and result["raw"]["grading"] == graded
            assert result["attempt_id"] == attempt and result["candidate_revision"] == row["candidate_revision"]
            assert request["candidate_revision"] == row["candidate_revision"]
            assert envelope["request_id"] == row["request_id"] and request["candidate"]["id"] == row["candidate"]
            assert request["candidate"]["recipe_overrides"] == row["recipe_overrides"]
            assert request["source"] == certificate["source"] and request["runtime"] == certificate["runtime"]
            source = request["source"]
            assert source["digest"] == row["source_digest"] == graded["source_digest"] == stable_hash(source["files"])
            assert source["origin_commit"] == row["executed_commit"]
            sources[source["digest"]] = source
            task = request["tasks"][cell["task_id"]]
            saved, = result["task_results"]
            assert saved["task_id"] == cell["task_id"] and saved["compatibility_key"] == job["compatibility_key"]
            assert saved["raw_status"] == "completed" and saved["gate_status"] == cell["status"]
            for key in ("metrics", "cost", "reason"):
                assert saved.get(key) == cell.get(key), key
            for key in ("recipe", "prior", "initializer", "initialization", "rng", "evidence", "applied"):
                if key in raw:
                    assert saved[key] == raw[key], key
            actual = raw.get("applied", raw)
            for key, value in row["recipe_overrides"].items():
                if key in actual["recipe"]:
                    assert actual["recipe"][key] == value, key
            if "prior" in actual:
                assert actual["prior"] == task["execution"]["prior"]
                assert actual["initializer"] == task["execution"]["initializer"]
            assert saved["cost"]["charged_task"] == cell["task_id"]
            assert saved["cost"]["wall_seconds"] == result["raw"]["elapsed_seconds"]
            evidence = raw["evidence"]
            if task["evaluation"]["kind"] == "schedule_contract":
                status = check_schedule(task, evidence)
                for name, identity in evidence["source_audit"]["source_sha256"].items():
                    assert identity == source["files"][name]
            else:
                assert evidence["guards"]["all_finite"] and evidence["guards"]["unintended_rng_deviations"] == 0
                status, suffix, failures = curve(task, evidence)
                curve_checks += len(failures)
                convergence = graded["grades"][task["id"]]["evaluator_result"]["convergence"]
                assert convergence["passing_suffix"] == suffix and convergence["complete"] is True
                if not failures[-1] and status == "FAIL":
                    endpoint_pass_suffix_fail[cell["task_id"]] += 1
                    endpoint_pass_suffix_fail_by_campaign[row["campaign"]][cell["task_id"]] += 1
                for values in failures[-5:]:
                    late_failures[cell["task_id"]].update(values)
                if "saved_observer_outputs" in evidence:
                    descriptor = evidence["saved_observer_outputs"]
                    path = local / descriptor["path"]
                    assert file_hash(path) == descriptor["sha256"] and path.stat().st_size == descriptor["bytes"]
                    draws = torch.load(path, map_location="cpu", weights_only=True)
                    assert len(draws) == len(failures) == descriptor["observation_count"]
                    assert descriptor["optimizer_updates_added"] == descriptor["sampling_draws_added"] == 0
                    if task["adapter"] == "transfer_vector":
                        score = gaussian_score if task["id"].startswith("gaussian") else ring_score
                        for point, draw in zip(evidence["observations"], draws, strict=True):
                            measured = score(draw["samples"], evidence["host"]["definition"], draw["step"])
                            for name, _, _ in task["evaluation"]["thresholds"]:
                                assert measured[name] == point[name]
                            vector_checks += 1
                    else:
                        for point, draw, values in zip(evidence["observations"], draws, failures, strict=True):
                            assert point == {"step": draw["step"], **draw["metrics"]}
                            assert bool(draw["passed"]) == (not values)
            assert status == cell["status"] == graded["grades"][task["id"]]["gate_status"]
            counts[status] += 1
            by_task[cell["task_id"]][status] += 1
            costs[row["campaign"]] += cell["cost"]["wall_seconds"]
            observed[attempt] = cell
            canonical_manifest[attempt] = canonical_hash
            receipt_checks += 1
    assert len(observed) == 86 and counts == Counter(PASS=40, FAIL=46)
    for digest, source in sources.items():
        snapshot = root / "runs/forge/bcap-tier1-repair/queue/snapshots" / digest
        for path, identity in source["files"].items():
            assert file_hash(snapshot / path) == identity, path
    for campaign, cost in costs.items():
        saved = report["campaign_accounting"][campaign]
        assert saved["reserved_seconds"] == 0 and math.isclose(cost, saved["spent_seconds"], abs_tol=1e-6)
        assert cost <= saved["definition"]["budget_seconds"]
    assert math.isclose(sum(costs.values()), report["total_new_paid_seconds"], abs_tol=1e-6)
    search_hashes = {}
    for campaign, expected_count in CAMPAIGNS.items():
        if expected_count == 1:
            continue
        path = root / "reports/forge/configuration-search" / (campaign + ".json")
        search = read_json(path)
        assert len(search["trials"]) == expected_count and search["blocked_count"] == 0
        assert search["progression"]["comparison_complete"] and search["selection"]["all_trials_terminal"]
        assert not search["selection"]["qualified"] and not search["default_adoption"]
        assert search["independent_confirmation"] == "not_performed"
        assert math.isclose(search["cost"]["new_paid_wall_seconds"], costs[campaign], abs_tol=1e-6)
        for trial in search["trials"]:
            row = row_map[trial["candidate_id"]]
            assert trial["candidate_revision"] == row["candidate_revision"] and trial["source_digest"] == row["source_digest"]
            assert trial["recipe_overrides"] == row["recipe_overrides"]
            measured = {t["task"]: t for t in trial["tasks"] if t["qualification_tier"] == 1}
            assert set(measured) == TASKS
            assert set(trial["selected_attempt_ids"]) == {t["attempt_id"] for t in row["tasks"]}
            for cell in row["tasks"]:
                assert measured[cell["task_id"]]["gate_status"] == cell["status"]
                compact_metrics = {k: v for k, v in cell["metrics"].items() if not isinstance(v, (list, dict))}
                assert measured[cell["task_id"]]["metrics"] == compact_metrics, (row["candidate"], cell["task_id"])
            assert all(binding["valid_receipt"] and canonical_manifest[binding["attempt_id"]] == binding["result_hash"]
                       for binding in trial["receipt_bindings"])
        search_hashes[campaign] = file_hash(path)
    assert torch.equal(rng, torch.get_rng_state())
    proof = {"schema_version": 1, "status": "PASS", "qualification_input": False,
        "scope": "independent_final_bcap_repair_readout_verification", "candidate_rows": 13,
        "ordinary_candidates": 12, "ordinary_tasks_per_candidate": 7, "diagnostic_cells": 2,
        "unique_attempts": receipt_checks, "counts": dict(counts), "counts_by_task": dict(by_task),
        "independently_gated_observations": curve_checks, "vector_observations_rescored_from_tensors": vector_checks,
        "endpoint_pass_but_sustained_fail_counts": dict(endpoint_pass_suffix_fail),
        "endpoint_pass_but_sustained_fail_counts_by_campaign": dict(endpoint_pass_suffix_fail_by_campaign),
        "failed_bounds_across_terminal_checks": dict(late_failures), "campaign_paid_seconds": dict(costs),
        "total_new_paid_seconds": sum(costs.values()), "results_sha256": file_hash(report_path),
        "canonical_receipts_manifest_sha256": stable_hash(canonical_manifest), "search_report_sha256": search_hashes,
        "verified_source_snapshots": {d: {"files": len(s["files"]), "executed_commit": s["origin_commit"]} for d, s in sources.items()},
        "cpu_global_rng_unchanged": True, "training_updates_added": 0, "model_sampling_draws_added": 0}
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(proof, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
