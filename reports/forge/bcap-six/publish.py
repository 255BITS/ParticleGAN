"""Verify and archive completed BCAP attempts, then save compact display evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import report_search
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.queue import Queue
from reports.forge.regenerate_technique_inventory import project_receipt


def audit_gaussian(raw):
    """Recompute both saved draws; never construct or query a learned model."""
    import torch
    from benchmarks.toy_audit.gaussian1d_quality import score_samples
    from experiments.forge.gaussian_tasks import bounds
    evidence = raw["evidence"]
    root = Path(evidence["artifact_root"]).resolve()
    verify_artifacts(root, evidence["artifact_manifest"])
    descriptor = evidence["saved_observer_outputs"]
    path = (root / descriptor["path"]).resolve()
    assert path.is_relative_to(root) and file_hash(path) == descriptor["sha256"]
    records = torch.load(path, map_location="cpu", weights_only=True)
    assert len(records) == 25
    assert [{"step": row["step"], **row["metrics"]} for row in records[1:]] == evidence["observations"]
    assert [row["confirmation"] for row in records[1:]] == evidence["confirmations"]
    confirmed, first_pair = [], None
    for record in records:
        for points, metrics in ((record["samples"], record["metrics"]),
                                (record["confirmation_samples"], record["confirmation"]["metrics"])):
            assert score_samples(points, evidence["host"]["definition"]) == metrics
        proof = record["confirmation"]
        assert proof["primary_state_sha256"] == proof["confirmed_state_sha256"] == record["training_state_sha256"]
        assert proof["training_state_unchanged"] is True
        assert proof["independent_stream"] == "eval/live/smoke_confirmation"
        if record["step"] == 0:
            continue
        if not bounds(record["metrics"]) and not bounds(proof["metrics"]):
            confirmed.append(record["step"])
            if first_pair is None:
                first_pair = {"step": record["step"], "primary": record["metrics"],
                              "confirmation": proof["metrics"], "state_sha256": proof["primary_state_sha256"]}
    assert confirmed == raw["gaussian_grade"]["evaluator_result"]["confirmed_steps"]
    return {"optimizer_smoothing": raw["recipe"]["optimizer_smoothing"], "saved_sample_sets_recomputed": 50,
            "same_state_pairs_verified": 25, "confirmed_scheduled_checks": len(confirmed),
            "first_confirmed_pair": first_pair, "endpoint": evidence["observations"][-1],
            "endpoint_confirmation": evidence["confirmations"][-1]["metrics"]}


def archive(root, inputs, destination):
    if destination.exists():
        raise ValueError("Archives are immutable; choose a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in inputs:
        for path in source.rglob("*") if source.is_dir() else [source]:
            if path.is_symlink():
                raise ValueError("Archive input is a symlink: " + str(path))
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    inventory = {name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                 for name, path in sorted(files.items())}
    payload = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    with tarfile.open(destination, "x:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
        member = tarfile.TarInfo("archive-members.json")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with tarfile.open(destination, "r:gz") as bundle:
        assert {item.name for item in bundle.getmembers()} == set(inventory) | {"archive-members.json"}
        for name, expected in inventory.items():
            member = bundle.getmember(name)
            assert member.isfile() and member.size == expected["bytes"]
            hasher = hashlib.sha256()
            with bundle.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            assert hasher.hexdigest() == expected["sha256"] == file_hash(files[name])
    return {"path": destination.relative_to(root).as_posix(), "bytes": destination.stat().st_size,
            "sha256": file_hash(destination), "original_files": len(inventory),
            "members_digest": stable_hash(inventory), "byte_exact_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--search", default="bcap-six-smoothing-v1")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    queue_root = (args.queue_root or root / "runs/forge").resolve()
    state_path = queue_root / "queue/state.json"
    original_state_hash = file_hash(state_path)
    state = read_json(state_path)
    assert all(item["status"] not in {"queued", "running", "paused"}
               for item in state["submissions"].values())
    queue = Queue(queue_root, report_root=root / "reports/forge", on_completion=None)
    report = report_search(root, queue_root, args.search, queue=queue)
    campaign = state["campaigns"][report["campaign"]["id"]]
    assert campaign["reserved_seconds"] == 0
    jobs = [job for job in state["jobs"].values()
            if job.get("cost_owner") and job["cost_owner"]["campaign"] == report["campaign"]["id"]]
    attempts = sorted(item["attempt_id"] for job in jobs for item in job["attempts"])
    assert all(len(job["attempts"]) <= 1 for job in jobs)
    charges = [row for row in state["charges"] if row["attempt_id"] in attempts]
    assert len(charges) == len(attempts) == len(set(attempts))
    assert math.isclose(sum(row["seconds"] for row in charges), campaign["spent_seconds"], abs_tol=1e-8)
    receipts, slots, raw_by_task, gaussian_audits = [], Counter(), {}, []
    for attempt in attempts:
        receipts.append(project_receipt(root, attempt))
        envelope = read_json(root / "reports/forge/attempts" / attempt / "request.json")
        request = envelope["request"]
        assert request["protocol"]["seed"] == 0
        assert request["execution_policy"]["mode"] == "complete_current_tier"
        assert request["compute_profiles"]["cuda"]["autograd_multithreading_enabled"] is False
        slots[envelope["worker"]["device"]] += 1
        local = Path(envelope["worker"]["directory"])
        raw = read_json(local / "raw-result.json")
        task = next(row["task_id"] for row in read_json(local / "result.json")["task_results"])
        raw_by_task.setdefault(task, []).append((request, raw))
        if task == "gaussian1d_smoke":
            gaussian_audits.append(audit_gaussian(raw))
        checkpoint = raw.get("evidence", {}).get("provenance_checkpoint")
        if checkpoint:
            verify_artifacts(Path(checkpoint["artifact_root"]), checkpoint["artifact_manifest"])
        for row in read_json(local / "result.json")["task_results"]:
            assert row["telemetry"]["memory"]["cuda_peak_allocated_bytes"] > 0
        applied = raw.get("applied", raw)
        recipe = applied["recipe"]
        assert recipe["optimizer_family"] == "dualnorm"
        assert recipe["optimizer_smoothing"] in {1e-5, 1e-4, 1e-3}
        assert recipe["lr_floor"] == recipe["network_lr_floor"] == 1
        assert recipe.get("optimizer_momentum", 0) == 0
        for role in applied.get("training_schedules", {}).get("optimizer_groups", {}).values():
            assert role["lr"]["minimum"] == role["lr"]["maximum"]
    matched = []
    for task, entries in sorted(raw_by_task.items()):
        assert len(entries) == len(report["trials"])
        # Generated ownership receipts include the deliberately different
        # trainer smoothing value; authored task conditions must be identical.
        task_declarations = {stable_hash({key: value for key, value in request["tasks"][task].items()
                                         if key != "field_ownership"}) for request, raw in entries}
        initializations = {stable_hash(raw.get("applied", raw).get("initialization"))
                           for request, raw in entries}
        data_receipts = [raw.get("evidence", {}).get("data_sha256") for request, raw in entries]
        data_hashes = {stable_hash(value) for value in data_receipts}
        stream_receipts = [raw.get("evidence", {}).get("provenance_checkpoint", {})
                           .get("named_stream_state_sha256") for request, raw in entries]
        stream_hashes = {stable_hash(value) for value in stream_receipts}
        assert len(task_declarations) == len(initializations) == len(data_hashes) == 1
        assert len(stream_hashes) == 1
        matched.append({"task": task, "task_declaration_sha256": next(iter(task_declarations)),
                        "initialization_receipt_sha256": next(iter(initializations))
                            if entries[0][1].get("applied", entries[0][1]).get("initialization") else None,
                        "seen_batch_sequence_sha256": next(iter(data_hashes)) if data_receipts[0] else None,
                        "seen_batch_digests": data_receipts[0],
                        "terminal_named_streams_sha256": next(iter(stream_hashes)) if stream_receipts[0] else None,
                        "task_conditions_identical": True})
    trials = []
    for trial in report["trials"]:
        tasks = [row for row in trial["tasks"] if row["qualification_tier"] == 1]
        required = [row for row in tasks if row["importance"] == "required"]
        assert len(required) == 6
        assert all(row["gate_status"] in {"PASS", "FAIL"} for row in tasks)
        trials.append({"candidate_id": trial["candidate_id"], "candidate_revision": trial["candidate_revision"],
                       "settings": trial["settings"], "source_digest": trial["source_digest"],
                       "required_passes": sum(row["gate_status"] == "PASS" for row in required),
                       "required_total": 6, "qualified_tier": trial["qualification"]["qualified_tier"],
                       "tasks": [{key: row[key] for key in ("task", "importance", "gate_status", "metrics", "cost")}
                                 for row in tasks]})
    destination = (args.archive or root / "artifacts" / (args.search + ".tar.gz")).resolve()
    spec = root / "configs/forge/searches" / (args.search + ".json")
    inputs = [queue_root, root / "runs/bcap-six", spec,
              root / "reports/forge/configuration-search" / spec.name,
              *(root / "reports/forge/bcap-six" / name
                for name in ("run.py", "publish.py", "export_media.py", "select.py")),
              *(root / "reports/forge/attempts" / attempt for attempt in attempts),
              *(root / "configs/forge/configurations" / (trial["candidate_id"] + ".json")
                for trial in report["trials"])]
    artifact = archive(root, inputs, destination)
    assert file_hash(state_path) == original_state_hash
    output = {"schema_version": 1, "study_id": args.search, "campaign_id": report["campaign"]["id"],
              "goal": "discriminator_stability", "view_revision": 8, "required_counts": [6, 21, 2],
              "through_tier": 1, "trials": trials, "selection": report["selection"],
              "source_commits": sorted({row["provenance"]["source_origin_commit"] for row in receipts}),
              "source_digests": report["source_digests"], "attempt_count": len(attempts),
              "attempt_receipts": receipts, "matched_conditions": matched,
              "gaussian_saved_sample_audits": gaussian_audits,
              "scientific_retries": 0, "seed": 0, "learning_rates_constant": True,
              "paid_seconds": campaign["spent_seconds"], "reserved_seconds": 0,
              "physical_cuda_slots": dict(slots), "archive": artifact,
              "qualification_input": False, "default_adoption": False,
              "publication_note": "Display evidence. Qualification consumes independently graded original receipts; no cross-candidate task pooling."}
    atomic_json(root / "reports/forge/bcap-six/readout.json", output)
    print(json.dumps({"study": args.search, "attempts": len(attempts), "paid_seconds": campaign["spent_seconds"],
                      "trials": [(row["settings"], row["required_passes"]) for row in trials],
                      "archive": artifact}, indent=2))


if __name__ == "__main__":
    main()
