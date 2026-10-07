"""Collector fixtures contain JSON metadata only, with no neural execution."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from reports.forge.collect_gaussian_smoke_inventory import collect_saved, validate_attempt, recall_record, saved_state_certificate


class SavedCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.queue = self.root / "runs/metadata-fixture"
        self.origin = "f" * 40
        self.source = {"files": {"metadata.py": "a" * 64}, "origin_commit": self.origin}
        self.source["digest"] = stable_hash(self.source["files"])
        self.round = {"id": "metadata-fixture", "candidate_ids": ["winner", "partial", "refused"],
                      "scientific_retries": 0, "seed": 0, "view": "fixture", "view_revision": 7,
                      "execution_backend": "cuda", "required_denominator_by_tier": [6, 20, 2]}
        self.view = {"id": "fixture", "revision": 7, "assignments": [
            {"task": "gaussian1d_smoke" if tier == 1 and i == 0 else f"tier{tier}-{i}",
             "qualification_tier": tier, "importance": "required", "order": i}
            for tier, total in ((1, 6), (2, 20), (3, 2)) for i in range(total)]}
        self.campaign = {"id": "metadata-fixture", "budget_seconds": 1000}
        self.state = {"campaigns": {"metadata-fixture": {"definition": self.campaign, "reserved_seconds": 0,
                                                        "spent_seconds": 0, "paused": False}},
                      "submissions": {}, "jobs": {}, "charges": []}
        self.preparation = {"round": "metadata-fixture", "stage": "enqueue", "round_sha256": stable_hash(self.round),
                            "candidate_count": 3, "source_digest": self.source["digest"], "through_tier": 2,
                            "execution_backend": "cuda", "campaign": self.campaign, "candidates": []}
        self.launch = {"source_commit": self.origin, "source_digest": self.source["digest"], "seed": 0,
                       "campaign": "metadata-fixture", "through_tier": 2, "workers_per_gpu": 1, "devices": ["0", "1"], "requests": 2}
        for name in ("winner", "partial"):
            self.add_candidate(name)
        self.preparation["candidates"].append({"candidate_id": "refused", "candidate_revision": "refused-revision",
                                               "source_digest": self.source["digest"], "declaration_kind": "idea",
                                               "request_id": None, "submission_status": "DECLARATION_REFUSED",
                                               "submission_blockers": ["frozen unsupported declaration"]})

    def tearDown(self):
        self.temporary.cleanup()

    def add_candidate(self, name):
        jobs = []
        tasks = {}
        for item in self.view["assignments"]:
            task = item["task"]
            key = stable_hash({"name": name, "task": task})
            jobs.append({"task_id": task, "task_ids": [task], "compatibility_key": key,
                         "resources": {"backend": "cuda", "allow_cpu": False, "gpus": 1},
                         "science": {"seed": 0, "compute": {"backend": "cuda"},
                                     "execution": {task: "execution"}, "evaluation": {task: "evaluation"}}})
            tasks[task] = {"adapter": "transfer_vector", "evaluation": {"kind": "transfer_sustained"},
                           "execution": {"initializer": "deterministic_orthogonal", "prior": {"kind": "mog"}}, "dependencies": []}
        request = {"candidate": {"id": name}, "candidate_revision": name + "-revision", "campaign_id": "metadata-fixture",
                   "source": deepcopy(self.source), "runtime": {"scope": "pure_saved_metadata_fixture"},
                   "jobs": jobs, "tasks": tasks, "view": self.view, "policy_fingerprint": stable_hash(self.view),
                   "execution_backend": "cuda", "execution_policy": {"schema_version": 1, "mode": "complete_current_tier"},
                   "requires_independent_grading": True, "protocol": {"seed": 0}, "rng": {"seed": 0}}
        request["request_id"] = stable_hash(request)[:24]
        request_id = request["request_id"]
        self.state["submissions"][request_id] = {"request": request, "status": "blocked"}
        atomic_json(self.queue / "queue/requests" / (request_id + ".json"), request)
        self.preparation["candidates"].append({"candidate_id": name, "candidate_revision": request["candidate_revision"],
                                               "source_digest": self.source["digest"], "declaration_kind": "idea",
                                               "request_id": request_id, "submission_status": "queued", "submission_blockers": []})
        for index, job in enumerate(jobs):
            task = job["task_id"]
            run = index < 6 or name == "winner" and index < 26
            stored = {"definition": job, "subscribers": [request_id], "attempts": [], "result": None, "status": "pending"}
            if run:
                attempt = name + "-" + str(index)
                worker = {"attempt": attempt, "token": "metadata-token", "device": str(index % 2)}
                resolved = {"request": request, "job": job, "worker": worker, "request_id": request_id}
                status = "FAIL" if name == "partial" and index == 0 or name == "winner" and index == 6 else "PASS"
                metrics = {"mean_error_sigma": .01, "trace": list(range(128))}
                grade = {"gate_status": status, "status": status, "metrics": metrics,
                         "evaluator_result": {"status": status, "metrics": [], "passed": status == "PASS"}}
                artifact_root = self.queue / attempt / "provenance"
                artifact_root.mkdir(parents=True)
                # Deliberately plain fixture bytes, never a serialized model.
                (artifact_root / "provenance-state.pt").write_bytes(b"explicit metadata-only fixture")
                manifest = manifest_artifacts(artifact_root)
                identity = manifest["files"]["provenance-state.pt"]
                descriptor = {"schema_version": 1, "purpose": "provenance_only", "prerequisite_eligible": False,
                              "artifact_root": str(artifact_root), "artifact_manifest": manifest,
                              "path": "provenance-state.pt", "sha256": identity["sha256"], "bytes": identity["size"],
                              "state_sha256": "b" * 64, "completed_steps": 1, "named_stream_keys": ["data:fixture"],
                              "named_stream_state_sha256": {"data:fixture": "c" * 64},
                              "optimizer_updates_added": 0, "sampling_draws_added": 0}
                producer = {"execution_path": "public_trainer", "device": "cuda:0",
                            "rng": {"bindings": {"data:fixture": {"family": "data"}}},
                            "evidence": {"provenance_checkpoint": descriptor}}
                raw = {"attempt_status": "completed", "finished_at": "2026-10-06T00:00:00Z",
                       "token": worker["token"], "elapsed_seconds": 1.0,
                       "result": {"scope": "metadata", "task": task, **producer},
                       "telemetry": {"interval": {"device": worker["device"]}}}
                raw["grading"] = {"source_digest": self.source["digest"], "raw_hash": stable_hash(raw["result"]),
                                  "grades": {task: deepcopy(grade)}}
                row = {**producer, **grade, "task_id": task, "compatibility_key": job["compatibility_key"],
                       "raw_status": "completed", "device": "cuda:0", "execution_path": "public_trainer",
                       "cost": {"device": worker["device"], "wall_seconds": 1.0, "completed_steps": 1}, "reason": "saved fixture"}
                result = {"attempt_id": attempt, "candidate_revision": request["candidate_revision"], "raw": raw,
                          "task_results": [row], "cost_owner": {"campaign": "metadata-fixture", "request": request_id,
                                                                "revision": request["candidate_revision"]}}
                directory = self.root / "reports/forge/attempts" / attempt
                atomic_json(directory / "request.json", resolved)
                atomic_json(directory / "result.json", result)
                atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result), "source": request["source"],
                                                         "runtime": request["runtime"]})
                stored.update(status="terminal", result=result, attempts=[{"attempt_id": attempt}])
                self.state["charges"].append({"attempt_id": attempt, "owner": result["cost_owner"], "seconds": 1.0})
                self.state["campaigns"]["metadata-fixture"]["spent_seconds"] += 1
            self.state["jobs"][job["compatibility_key"]] = stored

    def call(self):
        return collect_saved(self.root, self.queue, self.state, self.round, self.preparation, self.launch,
                             origin=self.origin, digest=self.source["digest"], roster_size=3)

    def test_full_roster_exact_metrics_and_newly_eligible_tier2_complete(self):
        result = self.call()
        rows = {row["candidate_id"]: row for row in result["candidate_rows"]}
        self.assertEqual(result["roster_count"], 3)
        self.assertEqual(rows["winner"]["tiers"]["1"]["passed"], 6)
        self.assertEqual(rows["winner"]["tiers"]["2"]["passed"], 19)
        self.assertEqual(rows["partial"]["tiers"]["1"]["passed"], 5)
        self.assertEqual(rows["refused"]["tiers"]["1"]["counts"], {"BLOCKED": 6})
        self.assertEqual(rows["refused"]["tiers"]["2"]["total"], 20)
        self.assertEqual(result["tier2_unlocked_candidates"], ["winner"])
        self.assertTrue(result["all_newly_eligible_tier2_jobs_ran"])
        self.assertEqual(result["cost"]["unique_campaign_paid_seconds"], 32)
        self.assertEqual(rows["winner"]["tasks"][0]["metrics"], {"mean_error_sigma": .01})
        self.assertFalse(result["qualification_input"])

    def test_active_campaign_refused_before_reading_or_writing_receipts(self):
        self.state["campaigns"]["metadata-fixture"]["reserved_seconds"] = 120
        with self.assertRaisesRegex(ValueError, "campaign is active"):
            self.call()

    def test_retry_drift_and_missing_originals_refused(self):
        job = next(job for job in self.state["jobs"].values() if job["result"])
        job["attempts"].append({"attempt_id": "extra-retry"})
        with self.assertRaisesRegex(ValueError, "scientific retry"):
            self.call()
        job["attempts"].pop()
        (self.root / "reports/forge/attempts" / job["result"]["attempt_id"] / "evidence.json").unlink()
        with self.assertRaisesRegex(ValueError, "hydrate original"):
            self.call()

    def test_certificate_mismatch_cpu_fallback_and_source_splice_refused(self):
        directory = self.root / "reports/forge/attempts/winner-0"
        resolved, result = read_json(directory / "request.json"), read_json(directory / "result.json")
        result["task_results"][0]["metrics"]["mean_error_sigma"] = 99
        with self.assertRaisesRegex(ValueError, "independent evaluator certificate"):
            validate_attempt(self.root, resolved, result, origin=self.origin, digest=self.source["digest"])
        result = read_json(directory / "result.json")
        resolved["worker"]["device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "non-CUDA"):
            validate_attempt(self.root, resolved, result, origin=self.origin, digest=self.source["digest"])
        resolved = read_json(directory / "request.json")
        resolved["request"]["source"]["origin_commit"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "frozen source identity"):
            validate_attempt(self.root, resolved, result, origin=self.origin, digest=self.source["digest"])

    def test_missing_newly_eligible_tier2_job_is_explicit(self):
        job = next(job for job in self.state["jobs"].values() if job["definition"]["task_id"] == "tier2-0" and job["result"])
        attempt = job["result"]["attempt_id"]
        job.update(result=None, attempts=[], status="pending")
        self.state["charges"] = [charge for charge in self.state["charges"] if charge["attempt_id"] != attempt]
        self.state["campaigns"]["metadata-fixture"]["spent_seconds"] -= 1
        result = self.call()
        self.assertFalse(result["all_newly_eligible_tier2_jobs_ran"])
        self.assertEqual(result["missing_newly_eligible_tier2_jobs"][0]["task_ids"], ["tier2-0"])

    def test_round_denominator_or_roster_drift_refused(self):
        self.round["candidate_ids"].append("new-recipe")
        with self.assertRaisesRegex(ValueError, "frozen roster"):
            self.call()

    def test_recall_is_nonqualifying_without_aggregate_task_cells(self):
        result = self.call()
        result["provenance"] = {"inputs": {"round": {"sha256": "round-file"}}, "collector_sha256": "collector"}
        record = recall_record(result, "readout-sha")
        self.assertFalse(record["qualification_input"])
        self.assertEqual(record["task_results"], [])
        self.assertEqual(record["evidence_scope"], "source_bound_campaign_readout")
        self.assertEqual(record["source"]["sha256"], "readout-sha")

    def test_saved_state_bytes_and_extra_files_fail_closed(self):
        directory = self.root / "reports/forge/attempts/winner-0"
        row = read_json(directory / "result.json")["task_results"][0]
        task = read_json(directory / "request.json")["request"]["tasks"][row["task_id"]]
        descriptor = row["evidence"]["provenance_checkpoint"]
        path = Path(descriptor["artifact_root"]) / descriptor["path"]
        original = path.read_bytes()
        path.write_bytes(b"changed saved fixture bytes")
        with self.assertRaisesRegex(ValueError, "artifact bytes changed"):
            saved_state_certificate(row, task)
        path.write_bytes(original)
        (path.parent / "extra-receipt.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "file set changed"):
            saved_state_certificate(row, task)

    def test_named_streams_and_non_prerequisite_contract_fail_closed(self):
        row = read_json(self.root / "reports/forge/attempts/winner-0/result.json")["task_results"][0]
        task = {"adapter": "transfer_vector"}
        descriptor = row["evidence"]["provenance_checkpoint"]
        descriptor["named_stream_state_sha256"]["extra_consumed_stream"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "every consumed named stream"):
            saved_state_certificate(row, task)
        descriptor["named_stream_state_sha256"].pop("extra_consumed_stream")
        descriptor["prerequisite_eligible"] = True
        with self.assertRaisesRegex(ValueError, "provenance-only checkpoint contract"):
            saved_state_certificate(row, task)
        descriptor["prerequisite_eligible"] = False
        descriptor["sampling_draws_added"] = 1
        with self.assertRaisesRegex(ValueError, "added scientific work"):
            saved_state_certificate(row, task)

    def test_missing_provenance_only_allowed_in_explicit_interrupted_cut(self):
        row = {"evidence": {}}
        with self.assertRaisesRegex(ValueError, "lacks a certified complete"):
            saved_state_certificate(row, {"adapter": "word_joint"})
        self.assertEqual(saved_state_certificate(row, {"adapter": "word_joint"}, allow_missing=True)["status"], "uncertified")
        self.state["submissions"][next(iter(self.state["submissions"]))]["status"] = "cancelled"
        with self.assertRaisesRegex(ValueError, "refuse finalization"):
            self.call()
        result = collect_saved(self.root, self.queue, self.state, self.round, self.preparation, self.launch,
                               origin=self.origin, digest=self.source["digest"], roster_size=3, partial_cut=True)
        self.assertFalse(result["finalized"])
        self.assertEqual(result["publication_scope"], "closed_partial_cut")

    def test_existing_gaussian_and_clockfree_formats_need_complete_certificates(self):
        directory = self.root / "legacy-saved-metadata"
        directory.mkdir()
        for name in ("state.pt", "initial.pt", "comparisons.pt"):
            (directory / name).write_bytes(b"metadata-only legacy fixture")
        manifest = manifest_artifacts(directory)
        evidence = {"artifact_root": str(directory), "artifact_manifest": manifest,
                    "checkpoint": {"path": "state.pt", "sha256": manifest["files"]["state.pt"]["sha256"], "state_sha256": "a" * 64}}
        row = {"evidence": evidence}
        self.assertEqual(saved_state_certificate(row, {"evaluation": {"kind": "gaussian_smoke"}})["format"], "gaussian_full_context_v1")
        self.assertEqual(saved_state_certificate(row, {"adapter": "clockfree_audit"})["format"], "clockfree_full_context_branches_v1")
        evidence["checkpoint"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "full-context checkpoint"):
            saved_state_certificate(row, {"evaluation": {"kind": "gaussian_smoke"}})

    def test_running_paid_worker_with_removed_subscribers_is_active(self):
        job = next(iter(self.state["jobs"].values()))
        job.update(status="running", subscribers=[], cost_owner={"campaign": "metadata-fixture"})
        with self.assertRaisesRegex(ValueError, "running worker"):
            self.call()

    def test_component_count_is_provenance_and_uses_nested_named_bindings(self):
        row = read_json(self.root / "reports/forge/attempts/winner-0/result.json")["task_results"][0]
        row["applied"] = {"rng": row.pop("rng")}
        row["cost"].pop("completed_steps")
        task = {"adapter": "transfer_behavior", "execution": {"host": "two_pole", "steps": 1000}}
        certificate = saved_state_certificate(row, task)
        self.assertEqual(certificate["completed_steps"], 1)
        self.assertEqual(certificate["completed_steps_semantics"], "minimum_actual_active_role_optimizer_count")
        self.assertFalse(certificate["completion_budget_witness"])


if __name__ == "__main__":
    unittest.main()
