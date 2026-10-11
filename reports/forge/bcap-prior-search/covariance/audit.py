"""Read-only saved-state validation of the completed moment-tether comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import torch

from experiments.forge.api import resolve_public_recipe, task_formulation_context
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.knowledge import _attempts
from experiments.forge.rng import NamedStreams
from experiments.forge.sources import verify_snapshot
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result


ROOT = Path(__file__).resolve().parents[4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b):
    return state_digest(a) == state_digest(b)


def actual_metadata(saved, row):
    if "applied" in saved:
        return saved["applied"]
    if "fixture" in saved:
        # WordFixture has a caller-owned policy and named data stream.
        return dict(recipe=saved["fixture"]["recipe"],
                    initialization=row.get("initialization", {}), prior=row.get("prior"))
    return saved


def moment_references(saved):
    """Find persistent reference buffers without recomputing or sampling them."""
    references = []
    def visit(value, path):
        if not isinstance(value, dict):
            return
        keys = ("moment_tether_mean", "moment_tether_covariance", "moment_tether_count")
        if all(key in value for key in keys):
            require("z" in value and int(value[keys[2]]) == len(value["z"]),
                    "Saved reference count differs from prior capacity")
            mean, covariance = value[keys[0]], value[keys[1]]
            require(torch.isfinite(mean).all() and torch.isfinite(covariance).all(),
                    "Non-finite saved reference moments")
            references.append(dict(path=path, count=int(value[keys[2]]),
                mean_sha256=state_digest(mean), covariance_sha256=state_digest(covariance)))
        for key, child in value.items():
            visit(child, path + "/" + str(key))
    visit(saved, "checkpoint")
    return references


def run(archive, output):
    progress = json.loads((archive / "progress.json").read_text())
    require(progress["phase"] == "complete", "Audit only the completed registered comparison")
    state = json.loads((archive / "queue/queue/state.json").read_text())
    spec = importlib.util.spec_from_file_location("moment_audit_workflow", ROOT / "reports/forge/bcap-prior-search/workflow.py")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    attempts = {a["attempt_id"]: a for a in _attempts(ROOT)[0]}
    registration = json.loads((output / "registration.json").read_text())
    require(progress["source_digest"] == registration["source_digest"], "Registered source changed")
    ids = [progress["control_request"], *progress["search_requests"]]
    entries = {rid: state["submissions"][rid] for rid in ids}
    require(all(entry["status"] not in {"queued", "running", "paused"} for entry in entries.values()),
            "Paused or active submissions are not a completed scientific round")
    baseline = entries[ids[0]]["request"]
    reference_recipe = asdict(resolve_public_recipe(baseline["candidate"]))
    verified_sources, receipts, by_arm = set(), [], {}
    request_checks = []
    for rid, entry in entries.items():
        req = entry["request"]
        require(req["source"]["digest"] == progress["source_digest"], "Cross-source comparison")
        require(req["source"]["origin_commit"] == progress["source_commit"], "Execution commit differs")
        require(req["protocol"] == baseline["protocol"] and req["protocol"]["seed"] == 0, "Protocol changed")
        require(req["runtime"] == baseline["runtime"] and req["compute_profiles"] == baseline["compute_profiles"],
                "Runtime/compute contract changed")
        conditions = lambda request: {name: {k: v for k, v in card.items()
            if k not in {"field_ownership", "preflight_blockers"}}
            for name, card in request["tasks"].items()}
        require(conditions(req) == conditions(baseline), "Frozen task conditions changed")
        require(req["through_tier"] == 2 and req["execution_policy"]["mode"] == "complete_current_tier",
                "Ordinary scope or execution law changed")
        recipe = asdict(resolve_public_recipe(req["candidate"]))
        delta = {k: dict(control=reference_recipe[k], candidate=v)
                 for k, v in recipe.items() if v != reference_recipe[k]}
        if rid == ids[0]:
            require(not delta, "Control recipe changed")
            role = "control"
        else:
            require(set(delta) == {"prior_regularizer", "prior_reg"}
                    and recipe["prior_regularizer"] == "moment_tether"
                    and recipe["prior_reg"] in (.001, .01, .1), "Undeclared trainer delta")
            role = str(recipe["prior_reg"])
        source = req["source"]
        if source["digest"] not in verified_sources:
            verify_snapshot(Path(source["snapshot_path"]), source)
            for path, digest in source["files"].items():
                require(file_hash(ROOT / path) == digest, "Published scientific bytes changed: " + path)
            verified_sources.add(source["digest"])
        request_checks.append(dict(role=role, request_id=rid, candidate_revision=req["candidate_revision"],
            recipe_sha256=stable_hash(recipe), delta=delta,
            task_contracts_sha256=stable_hash(req["tasks"]), source_files_verified=len(source["files"])))
        rows = workflow.validated_submission_rows(state, entry, attempts)
        by_arm[role] = {}
        for row in rows:
            task = req["tasks"][row["task_id"]]
            require(row["receipt_binding"]["valid_receipt"],
                    role + "/" + task["id"] + ": original certificate/result/source binding invalid")
            grade = grade_result(task, row)
            require(grade["status"] == row["gate_status"], role + "/" + task["id"] + ": saved grade differs")
            for field in ("metrics", "evaluator_result"):
                if field in grade:
                    require(same(grade[field], row[field]), task["id"] + ": saved numerical packet differs")
            descriptor = row.get("evidence", {}).get("provenance_checkpoint")
            if descriptor is None:  # Explicit incomplete/software failures retain their absence.
                receipts.append(dict(role=role, task=task["id"], status=row["gate_status"],
                                     checkpoint="unavailable", original_receipt_binding=row["receipt_binding"]))
                continue
            path = Path(descriptor["artifact_root"]) / descriptor["path"]
            verify_artifacts(Path(descriptor["artifact_root"]), descriptor["artifact_manifest"])
            require(file_hash(path) == descriptor["sha256"], task["id"] + ": saved state bytes changed")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            require(state_digest(saved) == descriptor["state_sha256"], task["id"] + ": saved state hash differs")
            streams = saved["streams"]
            probe = NamedStreams(streams["manifest"]["seed"], version=streams["manifest"]["version"])
            probe.validate_state_dict(streams)
            require(streams["manifest"]["seed"] == 0, "Checkpoint protocol seed changed")
            require({k: state_digest(v) for k, v in streams["states"].items()}
                    == descriptor["named_stream_state_sha256"], "Consumed stream hashes differ")
            if task["adapter"] == "clockfree_audit":
                receipts.append(dict(role=role, task=task["id"], status=row["gate_status"],
                    scope="separate_optional_clock_diagnostic", checkpoint_sha256=descriptor["sha256"],
                    numerical_grade_sha256=stable_hash(grade), original_receipt_binding=row["receipt_binding"]))
                continue
            actual = actual_metadata(saved, row)
            expected = asdict(task_formulation_context(req["candidate"], task, req["protocol"], root=ROOT).recipe)
            require(asdict(resolve_public_recipe(dict(recipe_overrides=actual["recipe"]))) == expected,
                    role + "/" + task["id"] + ": actual recipe differs")
            guards = row.get("evidence", {}).get("guards", {})
            require(guards.get("unintended_rng_deviations", 0) == 0, "RNG guard deviation")
            prior_policy = row.get("evidence", {}).get("prior_policy", {})
            if prior_policy:
                require(prior_policy["regularizer"] == expected["prior_regularizer"]
                        and prior_policy["weight"] == expected["prior_reg"], "Actual prior policy differs")
            bindings = streams["manifest"]["bindings"]
            non_eval = {k: v for k, v in streams["states"].items() if bindings[k]["family"] != "eval"}
            initial = actual.get("initialization", row.get("initialization"))
            if not initial:
                initial = row.get("evidence", {}).get("host", {}).get("initialization", {})
            references = moment_references(saved)
            if prior_policy.get("applicability") == "latent_prior":
                require(bool(references) == (expected["prior_regularizer"] == "moment_tether"
                    and expected["prior_reg"] > 0 and expected["prior_update"] == "learned"),
                    task["id"] + ": actual saved reference activation differs")
            receipt = dict(role=role, task=task["id"], status=row["gate_status"],
                compatibility_key=row["compatibility_key"], recipe_sha256=stable_hash(expected),
                initialization_sha256=stable_hash(initial),
                initial_tensor_hashes_available=bool(initial), moment_references=references,
                initial_prior=actual.get("prior", row.get("prior")),
                checkpoint_sha256=descriptor["sha256"], state_sha256=descriptor["state_sha256"],
                completed_steps=descriptor["completed_steps"],
                non_eval_streams_sha256=state_digest(non_eval), optimizer_updates=guards.get("optimizer_updates"),
                numerical_grade_sha256=stable_hash(grade), prior_policy=prior_policy)
            receipt["original_receipt_binding"] = row["receipt_binding"]
            receipts.append(receipt)
            by_arm[role][task["id"]] = dict(receipt=receipt, initial=initial, bindings=bindings,
                                            non_eval=non_eval, row=row)
    pairs = []
    for role, cells in by_arm.items():
        if role == "control":
            continue
        for task, cell in cells.items():
            control = by_arm["control"].get(task)
            if control is None:
                continue
            require(same(cell["initial"], control["initial"]), role + "/" + task + ": initial tensors differ")
            require(cell["receipt"]["initial_prior"] == control["receipt"]["initial_prior"], "Initial prior law differs")
            a, b = cell["receipt"]["completed_steps"], control["receipt"]["completed_steps"]
            bindings_match = same({k: cell["bindings"][k] for k in cell["non_eval"]},
                                  {k: control["bindings"][k] for k in control["non_eval"]})
            states_match = same(cell["non_eval"], control["non_eval"])
            require(bindings_match, "Named training bindings differ")
            if a == b:
                require(states_match, role + "/" + task + ": matched-prefix consumed streams differ")
            pairs.append(dict(role=role, task=task,
                              initial_tensors_equal=True if cell["initial"] else "source-bound explicit fixture; no tensor hash receipt",
                              initial_prior_equal=True,
                              named_non_eval_bindings_equal=bindings_match, consumed_non_eval_streams_equal=states_match,
                              completed_steps=dict(control=b, candidate=a),
                              pairing="equal consumed prefix" if a == b else "unequal own certified prefixes; no equal-consumption claim"))
    reference_pairs = []
    for role, cells in by_arm.items():
        if role == "control":
            continue
        for task, cell in cells.items():
            refs = cell["receipt"]["moment_references"]
            for other_role, other_cells in by_arm.items():
                if other_role == "control" or other_role >= role or task not in other_cells or not refs:
                    continue
                require(same(refs, other_cells[task]["receipt"]["moment_references"]),
                        task + ": fixed initial moment references differ between strengths")
                reference_pairs.append(dict(task=task, strengths=[other_role, role],
                                            fixed_initial_moment_buffers_equal=True))
    continuation_reference_pairs = []
    for role, cells in by_arm.items():
        if role == "control":
            continue
        for child, parent in (("gaussian1d_stability", "gaussian1d_smoke"),
                              ("five_word_joint_hold", "five_word_joint_smoke")):
            if child not in cells or parent not in cells:
                continue
            original = cells[parent]["receipt"]["moment_references"]
            restored = cells[child]["receipt"]["moment_references"]
            require(original and same(original, restored),
                    role + "/" + child + ": own continuation replaced fixed initial references")
            continuation_reference_pairs.append(dict(role=role, child=child, parent=parent,
                original_fixed_moment_buffers_restored=True,
                scope="Same candidate's original initial refs; no equal trained starting-weight claim"))
    history = []
    for job in state["jobs"].values():
        for attempt in job.get("attempts", []):
            path = Path(attempt["path"])
            # Infrastructure cancellations retain exact archived originals;
            # they never inherit the completed attempt's numerical grade.
            files = {p.name: file_hash(p) for p in path.iterdir()
                     if p.is_file() and p.name in {"result.json", "grading.json", "run.log", "cancel.json", "terminal.json"}}
            original = json.loads((path / "result.json").read_text()) if (path / "result.json").exists() else {}
            raw = original.get("raw", original)
            cancellation = json.loads((path / "cancel.json").read_text()) if (path / "cancel.json").exists() else None
            history.append(dict(attempt_id=attempt["attempt_id"], artifact_root=str(path),
                task=job["definition"]["execution_group"], original_files_sha256=files,
                original_status=raw.get("attempt_status"), cancellation=cancellation,
                charged_seconds=sum(c["seconds"] for c in state["charges"]
                                    if c["attempt_id"] == attempt["attempt_id"])))
    repair_path = Path(progress["execution_repair"]) if progress.get("execution_repair") else None
    repair = (dict(path=str(repair_path), sha256=file_hash(repair_path),
                   contract=json.loads(repair_path.read_text())) if repair_path is not None else None)
    result = dict(schema_version=1, status="PASS", scope="read_only_actual_saved_scientific_bindings",
        source_commit=progress["source_commit"], source_digest=progress["source_digest"],
        registration_sha256=file_hash(output / "registration.json"), execution_repair=repair,
        scientific_requests=request_checks,
        saved_states=receipts, paired_states=pairs, fixed_reference_pairs=reference_pairs,
        own_continuation_reference_pairs=continuation_reference_pairs,
        physical_attempts=len(history), attempt_history=history,
        actual_status_counts=dict(Counter(r["status"] for r in receipts)),
        optimizer_updates_added=0, sampling_draws_added=0,
        note="Unmeasured/blocked cells remain explicit in results.json. Own confirmed-prefix differences are reported, never pooled as equal consumed streams.")
    atomic_json(output / "audit.json", result)
    print(json.dumps(dict(status="PASS", saved_states=len(receipts), paired_states=len(pairs))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.archive, args.output)
