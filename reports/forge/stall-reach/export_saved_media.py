"""Export every certified completed stall-reach toy (both arms) from retained training data.

Adapted from reports/forge/bcap-prior-search/export_saved_media.py.

No model is constructed or queried, no optimizer runs, and no sample is drawn.
Final export requires a completed queue. --allow-partial/--task are software
checks over already completed evidence and cannot conclude the research round.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from experiments.forge.knowledge import _attempts
from experiments.forge.queue import Queue
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def export(args):
    root = args.repository.resolve()
    require(root == ROOT, "Run the copied helper in its feature checkout so imports match executed source.")
    output = args.output or root / "reports/forge/stall-reach/media"
    output = output.resolve()
    workflow = module("prior_search_media_workflow", root / "reports/forge/bcap-prior-search/workflow.py")
    saved = module("prior_search_saved_publisher", root / "reports/forge/bcap-develop-integration/publish.py")
    renderer = saved._saved_renderer(root)
    axis = "generator_critic_read"
    progress = read_json(args.archive / "progress.json")
    queue = Queue(args.archive / "queue", report_root=root / "reports/forge", on_completion=None)
    state = queue.inspect()
    active = (any(e["status"] in {"queued", "running", "paused"} for e in state["submissions"].values())
              or any(j["status"] == "running" for j in state["jobs"].values()))
    if not args.allow_partial:
        require(progress["phase"] == "complete" and not active, "Finish all registered work before final media publication.")
        require(not args.task, "A final publication must illustrate every certified completed toy.")
    originals, _ = _attempts(root)
    attempts = {a["attempt_id"]: a for a in originals}
    requests = [("control", progress["requests"]["control"]), ("stall_reach", progress["requests"]["base"])]
    output.mkdir(parents=True, exist_ok=True)
    entries, skipped, seen_tasks = [], [], set()
    with renderer.forbid_live_execution():
        for role, request_id in requests:
            entry = state["submissions"][request_id]
            request = entry["request"]
            require(request["source"]["digest"] == progress["source_digest"], "Media cohort differs from executed source.")
            rows = {r["task_id"]: r for r in workflow.validated_submission_rows(state, entry, attempts)}
            for assignment in request["view"]["assignments"]:
                if assignment["qualification_tier"] > 2:
                    continue
                task_id = assignment["task"]
                seen_tasks.add(task_id)
                if args.task and task_id not in args.task:
                    continue
                row = rows.get(task_id, {})
                status = row.get("gate_status", "UNMEASURED")
                identity = dict(role=role, request_id=request_id, task_id=task_id,
                                qualification_tier=assignment["qualification_tier"],
                                candidate_id=request["candidate"]["id"], candidate_revision=request["candidate_revision"],
                                source_commit=request["source"].get("origin_commit"), source_digest=request["source"]["digest"],
                                gate_status=status, settings={axis: request["candidate"]["resolved_recipe"].get(axis)})
                if status not in {"PASS", "FAIL"}:
                    skipped.append(dict(identity, media_status="NOT_RENDERED", reason=row.get("reason", "No certified completed PASS/FAIL")))
                    continue
                binding = row["receipt_binding"]
                require(binding["valid_receipt"], "Only original certified completions can receive actual-training media.")
                attempt_id = binding["attempt_id"]
                original = saved.certified_attempt(root, attempt_id)
                require(stable_hash(original["result"]) == binding["result_hash"], "Media result differs from original certificate.")
                task = request["tasks"][task_id]
                original_row = next(r for r in original["result"]["task_results"] if r["task_id"] == task_id)
                require(original_row["gate_status"] == status, "Frozen regrade differs from original recorded grade.")
                original_hash = stable_hash(original_row)
                actual = workflow.compact_row(original_row)
                require(actual.get("recipe"), "Actual executed recipe must be recorded for completed media.")
                context_sources = {}
                for key in ("recipe", "initializer", "initialization", "initial_prior", "prior"):
                    context_sources[key] = ("original_row." + key if key in original_row else
                                            "original_row.applied." + key if key in original_row.get("applied", {}) else
                                            "not_recorded")
                context_sources["prior_policy"] = ("original_row.prior_policy" if "prior_policy" in original_row else
                                                   "original_row.evidence.prior_policy" if "prior_policy" in original_row.get("evidence", {}) else
                                                   "not_recorded")
                sampling = {k: original_row.get("evidence", {}).get(k, original_row.get("claim_contract", {}).get(k))
                            for k in ("sampling_contract_version", "sampling_law", "eval_output_noise", "scoring_weights")}
                provenance = renderer.retained_provenance(original_row.get("evidence", {}))
                display = dict(task=task, row=deepcopy(row), attempt=original, request=request)
                alias = None
                if task["adapter"] == "ring_endurance":
                    dense = display["row"]["evidence"].get("dense", [])
                    require(len(dense) >= 2, "Ring media requires multiple actual certified dense observations.")
                    display["row"]["evidence"]["observations"] = deepcopy(dense)
                    alias = dict(display_curve="Exact original certified dense trajectory", original_dense_sha256=stable_hash(dense),
                                 observations_added=0, grading_row_mutated=False)
                gif = output / role / (task_id + ".gif")
                receipt = saved.render_saved(display, gif, renderer)
                with Image.open(gif) as image:
                    frames = image.n_frames
                require(frames >= 2, "Every completed toy needs at least two actual-training frames.")
                require(receipt["optimizer_updates_added"] == receipt["sampling_draws_added"] == 0,
                        "Media must add no training updates or sampling draws.")
                require(stable_hash(original_row) == original_hash, "Rendering changed original grading evidence.")
                certificate_root = root / "reports/forge/attempts" / attempt_id
                receipt.update(identity, attempt_id=attempt_id, frames=frames,
                               task_card_sha256=stable_hash(task), evaluation_sha256=stable_hash(task["evaluation"]),
                               original_result_sha256=binding["result_hash"],
                               evaluator_result_sha256=stable_hash(original_row.get("evaluator_result", {})),
                               certificate_sha256={name: file_hash(certificate_root / (name + ".json"))
                                                   for name in ("request", "result", "evidence")},
                               recipe_sha256=stable_hash(actual["recipe"]),
                               initializer=actual.get("initializer"), initializer_sha256=stable_hash(actual.get("initializer")),
                               initialization_sha256=stable_hash(actual.get("initialization")),
                               initial_prior_sha256=stable_hash(actual.get("initial_prior")),
                               prior_sha256=stable_hash(actual.get("prior")),
                               prior_policy=actual.get("prior_policy"), prior_policy_sha256=stable_hash(actual.get("prior_policy")),
                               context_sources=context_sources,
                               actual_recipe_settings={axis: actual["recipe"].get(axis)},
                               media_cohort="clock_probe_diagnostic" if task["adapter"] == "clockfree_audit" else "original_certified_task",
                               feature_invariance_claim=False,
                               sampling=sampling, sampling_sha256=stable_hash(sampling), retained_provenance=provenance,
                               qualification_input=False, rescored_observations=0,
                               live_execution_guard="Model construction/forward, trainer step/sample and prior sampling forbidden")
                if alias:
                    receipt.update(alias)
                atomic_json(gif.with_suffix(".json"), receipt)
                entries.append(dict(role=role, task_id=task_id, gate_status=status, attempt_id=attempt_id, frames=frames,
                                    gif=str(gif.relative_to(output)), gif_sha256=file_hash(gif),
                                    receipt=str(gif.with_suffix(".json").relative_to(output)),
                                    receipt_sha256=file_hash(gif.with_suffix(".json"))))
                print(json.dumps(dict(event="saved_training_gif", name=args.name, **entries[-1])), flush=True)
    require(not args.task or set(args.task) <= seen_tasks, "Requested media task is outside the frozen through-2 view.")
    index = dict(schema_version=1, name=args.name, scope="every_certified_completed_prior_search_toy",
                 partial=args.allow_partial, requested_tasks=args.task or None, source_commit=progress["source_commit"],
                 source_digest=progress["source_digest"], media=entries, not_rendered=skipped,
                 actual_training_gifs=len(entries), minimum_frames=2, optimizer_updates_added=0, sampling_draws_added=0,
                 rescored_observations=0, qualification_input=False, original_grades_unchanged=True,
                 script_sha256=file_hash(Path(__file__)), shared_publisher_sha256=file_hash(root / "reports/forge/bcap-develop-integration/publish.py"))
    atomic_json(output / "index.json", index)
    lines = ["# Saved actual-training illustrations", "",
             "Every listed GIF uses only original certified scored outputs and recorded numerical observations. "
             "Model construction/forward, training and sampling are forbidden during export; "
             "these illustrations supply no additional qualification credit.", "",
             f"Executed source `{progress['source_commit']}`. " + ("This is an explicitly partial software check." if args.allow_partial else "All certified completed PASS/FAIL toys are illustrated."), "",
             "The [compact index](index.json) binds every GIF and receipt to its original source, recipe, initialization and sampling law.", ""]
    for entry in entries:
        lines.append(f"- [{entry['role']}: {entry['task_id']} — {entry['gate_status']} ({entry['frames']} frames)]({entry['gif']}) ([receipt]({entry['receipt']}))")
    if skipped:
        lines += ["", "Unmeasured, incomplete, blocked or invalid rows receive no substituted training illustration; their states remain in the compact index."]
    atomic_text(output / "README.md", "\n".join(lines) + "\n")
    print(json.dumps(dict(event="saved_media_complete", name=args.name, gifs=len(entries), partial=args.allow_partial)), flush=True)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="stall-reach")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    export(args)


if __name__ == "__main__":
    main()
