"""Compact original certificate witnesses for the repair and regression."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.knowledge import _attempts

ROOT = Path(__file__).resolve().parents[4]


def run(archive, output):
    progress = json.loads((archive / "progress.json").read_text())
    state = json.loads((archive / "queue/queue/state.json").read_text())
    assert progress["phase"] == "complete"
    assert not any(e["status"] in {"running", "queued", "paused"} for e in state["submissions"].values())
    spec = importlib.util.spec_from_file_location("repair_workflow", ROOT / "reports/forge/bcap-prior-search/workflow.py")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    attempts = {a["attempt_id"]: a for a in _attempts(ROOT)[0]}
    ids = {"control": progress["control_request"]}
    ids["moment_tether_0.01"] = next(rid for rid in progress["search_requests"]
        if state["submissions"][rid]["request"]["candidate"]["recipe_overrides"]["prior_reg"] == .01)
    witnesses = {}
    for label, rid in ids.items():
        entry = state["submissions"][rid]
        rows = {r["task_id"]: r for r in workflow.validated_submission_rows(state, entry, attempts)}
        witnesses[label] = {}
        for task in ("mode_hold", "five_word_joint_hold"):
            row = rows[task]
            assert row["receipt_binding"]["valid_receipt"]
            result = next(state["jobs"][j["compatibility_key"]]["result"]
                          for j in entry["request"]["jobs"] if task in j.get("task_ids", [j["task_id"]]))
            attempt = attempts[result["attempt_id"]]
            directory = next(Path(a["path"]) for j in state["jobs"].values()
                             for a in j.get("attempts", []) if a["attempt_id"] == result["attempt_id"])
            packet = dict(gate_status=row["gate_status"], receipt_binding=row["receipt_binding"],
                original_files_sha256={name: file_hash(directory / name) for name in
                    ("result.json", "adapter-receipt.json", "graded-result.json", "request.json", "run.log")},
                artifact_root=str(directory), checkpoint=row["evidence"]["provenance_checkpoint"],
                original_valid_receipt=attempt["valid_receipt"])
            if task == "mode_hold":
                observations = row["evidence"]["observations"]
                convergence = row["evaluator_result"]["convergence"]
                suffix = convergence["passing_suffix"]
                point = next(p for p in observations if p["step"] == 1050)
                packet.update(passing_suffix=dict(score=suffix, target=">= 5"),
                    original_observations_sha256=stable_hash(observations), observation_count=len(observations),
                    final_suffix_observation_hashes=[dict(step=p["step"], sha256=stable_hash(p))
                                                    for p in observations[-suffix:]],
                    discriminating_original_observation=dict(step=1050, sha256=stable_hash(point),
                        measurements=[dict(metric="modes", score=point["modes"], target=">= 8"),
                                      dict(metric="hq", score=point["hq"], target=">= 0.9")]))
            else:
                report = json.loads(output.with_name("results.json").read_text())
                role = "control" if label == "control" else str({"prior_reg": .01})
                cell = next(c for c in report["task_measurements"] if c["role"] == role and c["task_id"] == task)
                packet.update(own_continuation=row["evidence"]["continuity"],
                    mandatory_gate=cell["measurements"][0],
                    original_failure_witnesses=[s for s in cell["measurements"] if "Worst mandatory" in s["name"]])
            witnesses[label][task] = packet
    assert witnesses["control"]["mode_hold"]["gate_status"] == "FAIL"
    assert witnesses["moment_tether_0.01"]["mode_hold"]["gate_status"] == "PASS"
    assert witnesses["control"]["five_word_joint_hold"]["gate_status"] == "PASS"
    assert witnesses["moment_tether_0.01"]["five_word_joint_hold"]["gate_status"] == "FAIL"
    atomic_json(output, dict(schema_version=1, scope="saved_original_qualification_witnesses",
        source_commit=progress["source_commit"], source_digest=progress["source_digest"], witnesses=witnesses,
        optimizer_updates_added=0, sampling_draws_added=0, model_executions_added=0,
        note="Original grades and observation packets are retained. Word hold uses each candidate's own certified acquisition state; equal trained starting weights are not claimed."))
    print(json.dumps(dict(event="repair_witness", status="PASS", original_certificates=4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("repair-witness.json"))
    args = parser.parse_args()
    run(args.archive, args.output)
