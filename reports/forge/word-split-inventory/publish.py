"""Certify compact results and archive original receipts without running models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import operator
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from reports.forge.regenerate_technique_inventory import project_receipt

CAMPAIGN = "technique-inventory-word-split-v1"


def archive(root, queue_root, attempts, specs, destination):
    """Hash each original before packing and recheck every archived member."""
    if destination.exists():
        raise ValueError("artifact archives are immutable; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = [queue_root, root / "runs/word-split-tier-results",
              *(root / "reports/forge/attempts" / item for item in attempts), *specs,
              *(root / "reports/forge/configuration-search" / path.name for path in specs),
              root / "reports/forge/word-split-inventory/run.py",
              root / "reports/forge/word-split-inventory/protocol.json", Path(__file__)]
    files = {}
    for source in inputs:
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            if path.is_symlink():
                raise ValueError("archive input cannot be a symlink: " + str(path))
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    inventory = {name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                 for name, path in sorted(files.items())}
    payload = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
    with tarfile.open(destination, "x:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name, recursive=False)
        member = tarfile.TarInfo("archive-members.json")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with tarfile.open(destination, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != len(inventory) + 1 or {item.name for item in members} != set(inventory) | {"archive-members.json"}:
            raise ValueError("archive membership differs")
        for name, expected in inventory.items():
            member = bundle.getmember(name)
            if not member.isfile() or member.size != expected["bytes"]:
                raise ValueError("archive size/type differs: " + name)
            hasher = hashlib.sha256()
            with bundle.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            if hasher.hexdigest() != expected["sha256"] or file_hash(files[name]) != expected["sha256"]:
                raise ValueError("archive bytes differ: " + name)
    return {"path": destination.relative_to(root).as_posix(), "bytes": destination.stat().st_size,
            "sha256": file_hash(destination), "original_files": len(inventory), "members": len(members),
            "members_digest": stable_hash(inventory), "byte_exact_verified": True,
            "qualification_input": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=ROOT / "runs/forge-word-split")
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/word-split-inventory-v1.tar.gz")
    args = parser.parse_args()
    queue_root = args.queue_root.resolve()
    state_path = queue_root / "queue/state.json"
    state_hash = file_hash(state_path)
    state = read_json(state_path)
    if any(item["status"] in {"queued", "running", "paused"} for item in state["submissions"].values()):
        raise ValueError("finish draining the campaign before publishing")
    if set(state["campaigns"]) != {CAMPAIGN} or any(item["status"] == "running" for item in state["jobs"].values()):
        raise ValueError("archive requires one completed campaign")
    campaign = state["campaigns"][CAMPAIGN]
    assert campaign["reserved_seconds"] == 0
    attempts = sorted(item["attempt_id"] for job in state["jobs"].values() for item in job["attempts"])
    charges = state["charges"]
    assert len(charges) == len(attempts) == len(set(attempts))
    assert {item["attempt_id"] for item in charges} == set(attempts)
    assert math.isclose(sum(item["seconds"] for item in charges), campaign["spent_seconds"], abs_tol=1e-8)
    assert all(len(job["attempts"]) <= 1 for job in state["jobs"].values())
    receipts, slots = [], Counter()
    for attempt in attempts:
        compact = project_receipt(ROOT, attempt)
        envelope = read_json(ROOT / "reports/forge/attempts" / attempt / "request.json")
        assert envelope["request"]["protocol"]["seed"] == 0
        assert envelope["request"]["compute_profiles"]["cuda"]["autograd_multithreading_enabled"] is False
        original = read_json(ROOT / "reports/forge/attempts" / attempt / "result.json")
        for row in original["task_results"]:
            if "device" in row:
                assert row["device"].startswith("cuda")
            # Frozen behavioral/clock adapters can omit a device label.
            # Their CUDA-bound execution also records actual allocator use.
            assert row["telemetry"]["memory"]["cuda_peak_allocated_bytes"] > 0
        slots[envelope["worker"]["device"]] += 1
        receipts.append(compact)
    specs = sorted((ROOT / "configs/forge/searches").glob("word-split-inventory-*-v1.json"))
    assert len(specs) == len(state["submissions"]) == 5
    families = []
    for path in specs:
        saved = read_json(ROOT / "reports/forge/configuration-search" / path.name)
        assert saved["campaign_accounting"]["reserved_seconds"] == 0
        assert math.isclose(saved["campaign_accounting"]["spent_seconds"], campaign["spent_seconds"], abs_tol=1e-8)
        assert len(saved["trials"]) == 1
        trial = saved["trials"][0]
        required = [task for task in trial["tasks"] if task["importance"] == "required"]
        smoke = [task for task in required if task["qualification_tier"] == 1]
        assert len(smoke) == 6 and all(task["gate_status"] not in {"UNKNOWN", "NOT_RUN"} for task in smoke)
        assert all(task["gate_status"] not in {"UNKNOWN", "NOT_RUN"} for task in trial["tasks"]
                   if task["qualification_tier"] == 1 and not task["blockers"] and not task["execution_group_blockers"])
        families.append({"trainer_family": saved["trainer_family"], "candidate_id": trial["candidate_id"],
                         "candidate_revision": trial["candidate_revision"], "source_digest": saved["source_digest"],
                         "qualified_tier": trial["qualification"]["qualified_tier"],
                         "tier2_eligible": all(task["gate_status"] == "PASS" for task in smoke),
                         "tiers": trial["qualification"]["tiers"],
                         "tasks": [{key: task[key] for key in ("task", "qualification_tier", "importance", "gate_status", "metrics", "cost", "blockers")}
                                   for task in trial["tasks"]],
                         "search_report": (ROOT / "reports/forge/configuration-search" / path.name).relative_to(ROOT).as_posix()})
    plan = read_json(ROOT / "runs/word-split-tier-results/plan.json")
    blocked = [{"candidate_id": row["candidate"], "blockers": row["submission_blockers"]}
               for row in plan["candidates"] if row["submission_status"] == "BLOCKED"]
    artifact = archive(ROOT, queue_root, attempts, specs, args.archive.resolve())
    assert file_hash(state_path) == state_hash
    output = {"schema_version": 1, "campaign_id": CAMPAIGN, "goal": "discriminator_stability",
              "through_tier": 2, "required_counts": [6, 21, 2], "selected_family_count": 7,
              "executed_family_count": 5, "blocked_families": blocked, "families": families,
              "attempt_count": len(attempts), "attempt_receipts": receipts, "all_actual_attempts_cuda": True,
              "physical_cuda_slots": dict(slots), "scientific_retries": 0, "seed": 0,
              "cuda_verification": "Frozen CUDA compute/request binding, physical worker slot, device label where reported, and positive measured CUDA allocator peak on every task result.",
              "paid_seconds": campaign["spent_seconds"], "reserved_seconds": 0,
              "source_commits": sorted({row["provenance"]["source_origin_commit"] for row in receipts}),
              "source_digests": sorted({row["provenance"]["source_digest"] for row in receipts}),
              "all_runnable_tier1_jobs_ran": True,
              "tier2_eligible_families": [row["trainer_family"] for row in families if row["tier2_eligible"]],
              "archive": artifact, "qualification_input": False, "qualification_reuse": False,
              "collector_sha256": file_hash(Path(__file__)), "reproduction_sha256": file_hash(ROOT / "reports/forge/word-split-inventory/run.py"),
              "publication_note": "Display readout only. Qualification uses independently graded original receipts registered by regenerate_technique_inventory.py."}
    tier2_receipts = [row for receipt in receipts for row in receipt["task_results"]
                      if next(task["qualification_tier"] for family in families for task in family["tasks"] if task["task"] == row["task_id"]) == 2]
    output["tier2_actual_task_results"] = len(tier2_receipts)
    if not output["tier2_eligible_families"]:
        assert not tier2_receipts
        output["all_newly_eligible_tier2_jobs_ran"] = True
        output["tier2_execution_reason"] = "No selected family passes all six required Tier 1 tasks. No ordinary Tier 2 execution is eligible."
    baseline_path = ROOT / "reports/forge/gaussian-smoke-inventory/final-v6/readout.json"
    baseline = read_json(baseline_path)
    comparisons = []
    for family in families:
        previous = next(row for row in baseline["candidate_rows"] if row["candidate_id"] == family["candidate_id"])
        prior_tasks = {row["task_id"]: row for row in previous["tasks"]}
        for task in family["tasks"]:
            if not task["metrics"]:
                continue
            old_id = "five_word_joint_acquisition" if task["task"] == "five_word_joint_smoke" else task["task"]
            if old_id not in prior_tasks:
                continue
            old = prior_tasks[old_id]
            scalars = lambda metrics: {key: value for key, value in metrics.items() if not isinstance(value, (list, dict))}
            before, after = scalars(old["metrics"]), scalars(task["metrics"])
            comparisons.append({"candidate_id": family["candidate_id"], "previous_task": old_id,
                                "current_task": task["task"], "previous_status": old["gate_status"],
                                "current_status": task["gate_status"], "scalar_endpoint_metrics_equal": before == after,
                                "changed_scalar_metrics": sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key)),
                                "scope": "Endpoint scalar comparison only; changed word acquisition policy is not a retention qualification."})
    output["previous_cohort_comparison"] = {"path": baseline_path.relative_to(ROOT).as_posix(),
                                            "sha256": file_hash(baseline_path), "source_commit": baseline["source_commit"],
                                            "source_digest": baseline["source_digest"], "tasks": comparisons,
                                            "old_evidence_is_new_qualification": False}
    gaussian_boundaries = []
    operations = {"==": operator.eq, ">=": operator.ge, "<=": operator.le}
    for job in state["jobs"].values():
        result = job.get("result")
        if not result or job["definition"]["task_id"] != "gaussian1d_smoke":
            continue
        request = state["submissions"][result["cost_owner"]["request"]]["request"]
        if not request["candidate"]["id"].startswith("bcap-dualnorm--"):
            continue
        row = result["task_results"][0]
        thresholds = request["tasks"][row["task_id"]]["evaluation"]["thresholds"]
        evidence = row["evidence"]
        confirmations = {item["step"]: item for item in evidence["confirmations"]}
        for primary in evidence["observations"]:
            if all(operations[op](primary[name], bound) for name, op, bound in thresholds):
                confirmation = confirmations[primary["step"]]
                assert confirmation["training_state_unchanged"] is True
                assert confirmation["primary_state_sha256"] == confirmation["confirmed_state_sha256"]
                gaussian_boundaries.append({"candidate_id": request["candidate"]["id"], "step": primary["step"],
                                            "primary_metrics": primary, "confirmation_metrics": confirmation["metrics"],
                                            "training_state_sha256": confirmation["primary_state_sha256"],
                                            "independent_stream": confirmation["independent_stream"],
                                            "training_state_unchanged": True})
    output["bcap_gaussian_primary_passing_checks"] = gaussian_boundaries
    atomic_json(ROOT / "reports/forge/word-split-inventory/readout.json", output)
    print(json.dumps({key: output[key] for key in ("attempt_count", "paid_seconds", "tier2_eligible_families", "archive")}, indent=2))


if __name__ == "__main__":
    main()
