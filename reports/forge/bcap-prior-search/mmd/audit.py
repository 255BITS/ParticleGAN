"""Read-only verification of completed MMD search and saved provenance.

This launches no training, optimizer update, sampling, scoring change or retry.
Missing original artifacts block the audit instead of authorizing a rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.knowledge import _attempts
from experiments.forge.sources import verify_snapshot
from experiments.forge.state import state_digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recipe_receipt(row):
    return row.get("applied", row)


def task_conditions(tasks):
    """Compare task law while allowing only the declared recipe receipt delta."""
    tasks = deepcopy(tasks)
    for card in tasks.values():
        fields = card.get("field_ownership", {}).get("recipe_fields", {})
        for name in ("prior_reg", "prior_regularizer"):
            for value in ("value", "reference_recipe_value"):
                fields.get(name, {}).pop(value, None)
    return tasks


def verify_provenance(row):
    evidence = row.get("evidence", {})
    verified = []
    for proof in (evidence, evidence.get("provenance_checkpoint", {})):
        if proof.get("artifact_root") and proof.get("artifact_manifest"):
            verify_artifacts(proof["artifact_root"], proof["artifact_manifest"])
            verified.append(proof["artifact_manifest"]["sha256"])
    proof = evidence.get("provenance_checkpoint")
    require(proof is not None, f"{row['task_id']}: missing saved provenance")
    state = torch.load(Path(proof["artifact_root"]) / "provenance-state.pt", map_location="cpu", weights_only=False)
    streams = state.get("streams", {}).get("states", {})
    require(bool(streams), f"{row['task_id']}: missing saved named streams")
    return state, verified


def non_eval_streams(state):
    return {name: value for name, value in state["streams"]["states"].items()
            if json.loads(name)[0] != "eval"}


def initial_models(row):
    evidence = row["evidence"]
    root, manifest = evidence.get("artifact_root"), evidence.get("artifact_manifest", {})
    if not root or "initial-state.pt" not in manifest.get("files", {}):
        return None
    state = torch.load(Path(root) / "initial-state.pt", map_location="cpu", weights_only=False)
    models = state.get("trainer", {}).get("models", state.get("models"))
    require(models is not None, f"{row['task_id']}: initial artifact has no models")
    # Added fixed reference buffers belong to the recipe. Comparing the actual
    # task tensors (including z/sigma/d0) avoids pretending these extra buffers
    # changed the initial prior distribution or network initialization.
    return {role: {name: value for name, value in values.items() if not name.startswith("_initial_mmd_")}
            for role, values in models.items()}


def terminal_failure_witness(row, card, driver):
    """Retain one original failed check from the required terminal window."""
    if card["evaluation"]["kind"] != "transfer_sustained" or row["gate_status"] != "FAIL":
        return None
    points = row["evidence"]["observations"]
    count = card["evaluation"]["minimum_stable_checks"]
    require(len(points) >= count, f"{row['task_id']}: incomplete terminal observations")
    for point in points[-count:]:
        metrics = driver.metric_rows(point, card["evaluation"]["thresholds"])
        if any(metric["status"] != "PASS" for metric in metrics if metric["limits"]):
            return dict(step=point["step"], original_observation_sha256=stable_hash(point),
                        original_observations_sha256=stable_hash(points),
                        required_terminal_checks=count, window_steps=[p["step"] for p in points[-count:]],
                        measurements=metrics)
    raise ValueError(f"{row['task_id']}: failed sustained gate has no terminal-window numerical witness")


def execution_accounting(archive, state, progress):
    """Keep interrupted execution costs without treating them as science."""
    charges = {row["attempt_id"]: row for row in state["charges"]}
    campaigns = {}
    for name, campaign in state["campaigns"].items():
        charged = sum(row["seconds"] for row in charges.values() if row["owner"]["campaign"] == name)
        require(abs(charged - campaign["spent_seconds"]) < 1e-6, "campaign charge total differs")
        require(campaign["reserved_seconds"] == 0 and not campaign["paused"], "campaign is not drained")
        require(charged <= campaign["definition"]["budget_seconds"], "campaign exceeds registered budget")
        campaigns[name] = dict(spent_seconds=charged, reserved_seconds=0,
                               budget_seconds=campaign["definition"]["budget_seconds"])
    repair = json.loads(Path(progress["execution_repair"]).read_text())
    probe = repair["mps_numerical_probe"]
    original, mps = [json.loads(Path(probe[key]).read_text()) for key in ("non_mps_path", "mps_path")]
    require(original["numerical_digests"] == mps["numerical_digests"], "MPS numerical probe differs")
    for key in ("device", "sm_count", "compute_capability", "torch_version", "cuda_version",
                "deterministic", "tf32", "cublas_workspace"):
        require(original[key] == mps[key], f"MPS compute setting differs: {key}")
    cancellations = []
    for attempt in repair["attempts"]:
        if attempt["name"] != "mmd":
            continue
        charge = charges[attempt["attempt_id"]]
        job = state["jobs"][attempt["compatibility_key"]]
        require(job["definition"]["budget_seconds"] == repair["unchanged_allowance_seconds"],
                "repaired word allowance changed")
        cancellations.append(dict(attempt_id=attempt["attempt_id"],
                                  previous_elapsed_seconds=attempt["previous_elapsed_seconds"],
                                  charged_seconds=charge["seconds"],
                                  selected_attempt_id=(job.get("result") or {}).get("attempt_id")))
    return dict(campaigns=campaigns, infrastructure_cancellations=cancellations,
                repair_record_sha256=file_hash(progress["execution_repair"]),
                reason=repair["reason"], unchanged_word_allowance_seconds=repair["unchanged_allowance_seconds"],
                unchanged_word_updates=repair["unchanged_updates"],
                mps_probe_equal_digests=len(mps["numerical_digests"]),
                mps_probe_sha256={key: file_hash(probe[key]) for key in ("non_mps_path", "mps_path")},
                scheduling="MPS; existing five repaired word workers finish unchanged; subsequent new jobs use one per physical GPU across five independent queues",
                driver_sha256=file_hash(OUT.parent / "workflow.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--control-word-audit", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    progress = json.loads((archive / "progress.json").read_text())
    require(progress["phase"] == "complete", "finish the bounded experiment before auditing its conclusion")
    published = json.loads((OUT / "results.json").read_text())
    require(published["source_commit"] == progress["source_commit"]
            and published["source_digest"] == progress["source_digest"], "publication source mismatch")
    queue = Queue(archive / "queue", on_completion=None)
    state = queue.inspect()
    accounting = execution_accounting(archive, state, progress)
    if args.control_word_audit:
        cross = json.loads(args.control_word_audit.read_text())
        require(cross["status"] == "PASS", "cross-control word audit fails")
        comparisons = [row for row in cross["comparisons"] if row["name"] == "mmd"]
        require(len(comparisons) == 2 and all(row["scientific_api_state_equal"]
                                            and row["all_named_streams_equal"]
                                            and row["fixture_data_generator_equal"] for row in comparisons),
                "matched MMD control scientific states differ")
        accounting["cross_control_word_audit"] = dict(
            path=str(args.control_word_audit.resolve()), sha256=file_hash(args.control_word_audit),
            comparisons=comparisons,
            note="Caller global CPU/CUDA RNG snapshots differ; named scientific streams and consumed training data match.")
    control = state["submissions"][progress["control_request"]]
    source = control["request"]["source"]
    require(source["digest"] == progress["source_digest"], "control source mismatch")
    verify_snapshot(archive / "queue/snapshots" / source["digest"], source)
    bundle = subprocess.check_output(["git", "archive", progress["source_commit"], "--", *sorted(source["files"])], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(bundle)) as tree:
        for name, digest in source["files"].items():
            require(hashlib.sha256(tree.extractfile(name).read()).hexdigest() == digest, f"executed Git source mismatch: {name}")
    module = importlib.util.spec_from_file_location("prior_search_readout", OUT.parent / "workflow.py")
    driver = importlib.util.module_from_spec(module)
    module.loader.exec_module(driver)
    originals, _ = _attempts(ROOT)
    attempts = {entry["attempt_id"]: entry for entry in originals}
    controls = {row["task_id"]: row for row in driver.validated_submission_rows(state, control, attempts)}
    checks, artifact_hashes = [], []
    all_entries = [("control", control)] + [(str(state["submissions"][rid]["request"]["candidate"]["recipe_overrides"]["prior_reg"]),
                                            state["submissions"][rid]) for rid in progress["search_requests"]]
    for label, entry in all_entries:
        request = entry["request"]
        require(request["source"]["digest"] == source["digest"], "arms do not share scientific source")
        require(request["protocol"] == control["request"]["protocol"] and request["protocol"]["seed"] == 0,
                "arms do not share seed-0 protocol")
        require(task_conditions(request["tasks"]) == task_conditions(control["request"]["tasks"]),
                "task conditions or undeclared ownership fields differ")
        require(request["execution_policy"]["mode"] == "complete_current_tier", "required tier policy changed")
        rows = driver.validated_submission_rows(state, entry, attempts)
        assignments = {row["task"]: row for row in request["view"]["assignments"]}
        for row in rows:
            name = row["task_id"]
            assignment = assignments[name]
            require(row["receipt_binding"]["valid_receipt"], f"{label}/{name}: invalid original receipt")
            if row.get("raw_status") != "completed":
                checks.append(dict(arm=label, task=name, status=row["gate_status"], completed=False,
                                   qualification_tier=assignment["qualification_tier"], importance=assignment["importance"],
                                   reason=row.get("reason")))
                continue
            evidence = row["evidence"]
            if request["tasks"][name]["adapter"] == "clockfree_audit":
                require(assignments[name]["importance"] == "diagnostic", "clock probe changed qualification scope")
                verify_artifacts(evidence["artifact_root"], evidence["artifact_manifest"])
                artifact_hashes.append(evidence["artifact_manifest"]["sha256"])
                checks.append(dict(arm=label, task=name, status=row["gate_status"], completed=True,
                                   qualification_tier=assignment["qualification_tier"], importance=assignment["importance"],
                                   scope="optional_clock_diagnostic", qualification_credit=False,
                                   paid_seconds=row.get("cost", {}).get("wall_seconds")))
                continue
            guards = evidence.get("guards", {})
            require(guards.get("all_finite") is True and guards.get("hooks_exercised") is True,
                    f"{label}/{name}: state or hook guard fails")
            require(guards.get("unintended_rng_deviations") == 0, f"{label}/{name}: RNG isolation fails")
            saved, verified = verify_provenance(row)
            artifact_hashes.extend(verified)
            same = controls.get(name)
            parity = {}
            if same is not None and label != "control" and same.get("raw_status") == "completed":
                before, after = recipe_receipt(same), recipe_receipt(row)
                left, right = deepcopy(before["recipe"]), deepcopy(after["recipe"])
                for field in ("prior_regularizer", "prior_reg"):
                    left.pop(field, None); right.pop(field, None)
                require(left == right, f"{label}/{name}: undeclared effective recipe delta")
                require(after.get("initialization") == before.get("initialization"), f"{label}/{name}: initial parameter identity differs")
                require(after.get("prior") == before.get("prior"), f"{label}/{name}: actual initial prior law differs")
                initial_before, initial_after = initial_models(same), initial_models(row)
                task = request["tasks"][name]
                own_prefix = bool(task["execution"].get("continuation_of") or name == "five_word_joint_hold")
                if initial_before is not None and initial_after is not None and not own_prefix:
                    require(state_digest(initial_before) == state_digest(initial_after), f"{label}/{name}: initial z/sigma/d0 or model tensors differ")
                    parity["actual_initial_tensors"] = True
                reference_state, _ = verify_provenance(same)
                if name != "five_word_joint_hold":
                    old_updates = same["evidence"]["guards"]["optimizer_updates"]
                    new_updates = guards["optimizer_updates"]
                    if task["adapter"] == "ring_endurance" and old_updates != new_updates:
                        # The original task stops after its own settling/hold
                        # decision plus extension, within a fixed maximum.
                        # Different declared stopping points cannot yield the
                        # same final consumed RNG state.
                        require(row["rng"]["bindings"] == same["rng"]["bindings"],
                                f"{label}/{name}: adaptive task initial streams differ")
                        parity["named_initial_stream_bindings"] = True
                        parity["task_owned_adaptive_stop_updates"] = dict(control=old_updates, trial=new_updates)
                    else:
                        require(old_updates == new_updates, f"{label}/{name}: fixed update counts differ")
                        require(state_digest(non_eval_streams(saved)) == state_digest(non_eval_streams(reference_state)),
                                f"{label}/{name}: consumed training/constructor/data streams differ")
                        parity["non_eval_stream_states"] = True
                if evidence.get("data_sha256") is not None and same["evidence"].get("data_sha256") is not None:
                    require(evidence["data_sha256"] == same["evidence"]["data_sha256"], f"{label}/{name}: target batch sequence differs")
                    parity["target_batch_digests"] = True
                parity["initialization_receipts"] = True
                parity["declared_recipe_delta_only"] = True
            check = dict(arm=label, task=name, status=row["gate_status"], completed=True,
                               qualification_tier=assignment["qualification_tier"], importance=assignment["importance"],
                               paid_seconds=row.get("cost", {}).get("wall_seconds"),
                               optimizer_updates=guards.get("optimizer_updates"), matched_control=parity)
            witness = terminal_failure_witness(row, request["tasks"][name], driver)
            if witness:
                check["original_terminal_failure_witness"] = witness
            if name == "five_word_joint_hold":
                check["continuity"] = deepcopy(evidence["continuity"])
                check["executed_continuation_updates"] = evidence["executed_updates"]
                check["completed_steps"] = evidence["completed_steps"]
            checks.append(check)
    result = dict(scope="read_only_saved_evidence_audit", source_commit=progress["source_commit"],
                  source_digest=source["digest"], source_files_verified=len(source["files"]),
                  execution_accounting=accounting,
                  immutable_artifact_manifests_verified=len(artifact_hashes), checks=checks,
                  recorded_result_status_counts=dict(Counter(check["status"] for check in checks)),
                  declared_readout_cells=len(published["task_measurements"]),
                  readout_status_counts=dict(Counter(row["status"] for row in published["task_measurements"])),
                  note="Word hold restores each recipe's own earliest certified prefix. Ring endurance retains its task-owned adaptive stopping rule and maximum budget; unequal stop lengths require equal initial named bindings rather than equal final stream states. Missing/blocked Tier 2 results provide no repair credit.")
    atomic_json(OUT / "audit.json", result)
    text = ["# Original sustained failure witnesses", "",
            "One original failed check from each required terminal window. Endpoint scores alone do not establish a sustained pass.", "",
            "Full original observation arrays stay in the artifact archive; their hashes and each selected packet hash are retained in [audit.json](audit.json).", ""]
    for check in checks:
        witness = check.get("original_terminal_failure_witness")
        if not witness:
            continue
        text.extend([f"## {check['arm']}: {check['task']}, step {witness['step']}", "",
                     f"Required terminal window: {witness['window_steps']}; {witness['required_terminal_checks']} passing checks required.", "",
                     f"Original observation SHA256: `{witness['original_observation_sha256']}`.", "",
                     "| Measurement | Score | Target | Status |", "| --- | ---: | --- | --- |"])
        text.extend(f"| {metric['metric']} | {driver.fmt(metric['score'])} | {metric['target']} | {metric['status']} |"
                    for metric in witness["measurements"] if metric["limits"])
        text.append("")
    (OUT / "failure-witnesses.md").write_text("\n".join(text))
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main()
