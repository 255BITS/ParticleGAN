"""Execute the frozen API-mode research and export compact evidence/GIFs.

Use an unused artifact directory outside Git. --export-only reads completed
receipts without executing, sampling, grading again, or rendering new frames.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.model_toy_adapter import definition, run
from experiments.forge.views import grade_result

PROTOCOL = ROOT / "reports/forge/api-model-toys/protocol.json"
PUBLICATION = ROOT / "reports/toy_audit/api_contract/model_modes/publication.json"


def event(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def execute(task_id, output, prefix=False):
    protocol = read_json(PROTOCOL)
    task = protocol["task_contracts"][task_id]
    destination = Path(output) / task_id
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "stdout.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            adapter = run({"candidate": protocol["candidate_contract"], "protocol": {"seed": 0}},
                          task, destination, "cpu", execution_limit=3 if prefix else None)
            grade = grade_result(task, adapter)
            raw = read_json(destination / "api/receipt.json")
            result = {"task_id": task_id, "grade": grade, "software_prefix": prefix,
                      "adapter_receipt": str(destination / "adapter-receipt.json"),
                      "api_receipt": str(destination / "api/receipt.json"),
                      "elapsed_seconds": raw["elapsed_seconds"]}
        except Exception as error:
            traceback.print_exc()
            result = {"task_id": task_id, "grade": {"gate_status": "ERROR", "reasons": [str(error)]},
                      "software_prefix": prefix, "elapsed_seconds": 0., "error": repr(error)}
        atomic_json(destination / "result.json", result)
    return result


def execute_stage(tasks, output, jobs, *, prefix=False):
    results = {}
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(execute, name, str(output), prefix): name for name in tasks}
        for future in as_completed(pending):
            result = future.result()
            results[result["task_id"]] = result
            event(task=result["task_id"], result=result["grade"]["gate_status"],
                  software_prefix=prefix, elapsed_seconds=result["elapsed_seconds"])
    return results


def export(output, results):
    protocol = read_json(PROTOCOL)
    cases, readouts, runs = [], [], []
    media = PUBLICATION.parent / "media"
    media.mkdir(parents=True, exist_ok=True)
    for name in protocol["task_contracts"]:
        task, result = protocol["task_contracts"][name], results[name]
        case = definition(task)
        # Map each new requirement directly as well as retaining historical IDs.
        case["legacy_ids"] = list(dict.fromkeys(case["legacy_ids"] + [case["id"]]))
        cases.append(case)
        raw = read_json(result["api_receipt"]) if result.get("api_receipt") else {}
        adapter = read_json(result["adapter_receipt"]) if result.get("adapter_receipt") else {}
        last = raw.get("observations", [{}])[-1]
        metric_steps = raw.get("protocol", {}).get("metric_evaluation_steps", [])
        terminal = [row for row in raw.get("observations", []) if row["step"] > 0 and row["step"] in metric_steps][-5:]
        gif = None
        if raw.get("artifacts", {}).get("goal.gif"):
            original = Path(result["api_receipt"]).parent / "goal.gif"
            target = media / (case["id"] + ".gif")
            shutil.copyfile(original, target)
            gif = target.relative_to(PUBLICATION.parent).as_posix()
        grade = result["grade"]["gate_status"]
        verdict = "BLOCKED" if result.get("blocked_by") else grade
        failed = (result.get("blocked_by", []) or raw.get("failed_bounds", [])
                  or result["grade"].get("reasons", []))
        readout = {"id": case["id"], "task_id": name, "tier": protocol["tiers"][name],
                   "goal": case["goal"], "scope": case["scope"], "verdict": verdict,
                   "execution_status": "SOFTWARE_PREFIX_ONLY" if result["software_prefix"] else raw.get("status", "ERROR"),
                   "completed_updates": raw.get("completed_updates", 0), "default_updates": case["default_steps"],
                   "full_protocol_complete": raw.get("default_protocol_complete", False),
                   "terminal_passes": sum(row["passed"] for row in terminal),
                   "terminal_observations": len(terminal), "failed_bounds": failed,
                   "final_metrics": last.get("metrics", {}), "gif": gif,
                   "source_commit": raw.get("source", {}).get("commit"),
                   "task_contract_sha256": stable_hash(task), "qualification_input": False}
        readouts.append(readout)
        recipe = raw.get("recipe") or adapter.get("recipe")
        if recipe is None:
            from experiments.forge.api import task_formulation_context
            recipe = task_formulation_context(protocol["candidate_contract"], task, {"seed": 0}).recipe.to_dict()
        runs.append({"id": case["id"], "task_id": name, "recipe": recipe,
                     "recipe_sha256": stable_hash(recipe), "runtime": raw.get("runtime", {}),
                     "source_identity": {"commit": raw.get("source", {}).get("commit"),
                                         "manifest_sha256": stable_hash(raw.get("source", {}))},
                     "raw_receipt": result.get("api_receipt"),
                     "raw_receipt_sha256": file_hash(result["api_receipt"]) if result.get("api_receipt") else None,
                     "adapter_receipt_sha256": file_hash(result["adapter_receipt"]) if result.get("adapter_receipt") else None,
                     "reproducibility": raw.get("reproducibility", {}),
                     "guards": adapter.get("evidence", {}).get("guards", {}),
                     "artifact_hashes": raw.get("artifacts", {}),
                     "gif_sha256": file_hash(PUBLICATION.parent / gif) if gif else None,
                     "cost_seconds": result["elapsed_seconds"], "qualification_input": False})
    publication = {"schema": "forge_api_model_research_v1", "seed": 0, "qualification_input": False,
                   "protocol_sha256": file_hash(PROTOCOL), "baseline": protocol["candidate_contract"]["id"],
                   "no_cross_cohort_pooling": True, "cases": cases, "readouts": readouts, "runs": runs,
                   "paid_cpu_seconds": sum(row["elapsed_seconds"] for row in results.values()),
                   "cap_seconds": protocol["research_budget_seconds"]}
    atomic_json(PUBLICATION, publication)
    from experiments.forge.model_toy_publication import load_publication
    load_publication(ROOT)
    event(publication=str(PUBLICATION), cases=len(cases),
          outcomes={status: sum(row["verdict"] == status for row in readouts)
                    for status in sorted({row["verdict"] for row in readouts})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or not 1 <= args.jobs <= 4:
        parser.error("use an external artifact directory and one to four worker processes")
    protocol = read_json(PROTOCOL)
    if args.export_only:
        results = read_json(output / "results.json")["results"]
    else:
        from experiments.forge.sources import inspect_source
        if inspect_source(ROOT, extra_paths=protocol["source_extra_paths"])["digest"] != protocol["complete_source_digest"]:
            raise ValueError("the complete frozen scientific source has changed")
        for path, expected in protocol["source_sha256"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("frozen research source changed: " + path)
        output.mkdir(parents=True, exist_ok=False)
        basic = [name for name, tier in protocol["tiers"].items() if tier == 1]
        deeper = [name for name, tier in protocol["tiers"].items() if tier == 2]
        results = execute_stage(basic, output, args.jobs)
        failures = [name for name in basic if results[name]["grade"]["gate_status"] != "PASS"]
        if failures:
            # Explicit software proof only: never launch a failed prerequisite's
            # full-budget Tier 2 training just to populate a display cell.
            prefixes = execute_stage(deeper, output, args.jobs, prefix=True)
            for name, result in prefixes.items():
                result["blocked_by"] = failures
            results.update(prefixes)
        else:
            results.update(execute_stage(deeper, output, args.jobs))
        atomic_json(output / "results.json", {"protocol_sha256": file_hash(PROTOCOL), "results": results})
    export(output, results)


if __name__ == "__main__":
    main()
