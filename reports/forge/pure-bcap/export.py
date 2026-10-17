"""Certify saved metrics and render training observations; never train or sample.

Complete readouts admit whole candidates from their executed source cohorts.
Partial initial readouts retain cancelled work without selecting a winner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.tier1_media import export_attempt
from reports.forge.regenerate_technique_inventory import project_receipt

REPORT = Path("reports/forge/pure-bcap")


def load_plan(root, path=None, *, partial=False):
    root = Path(root).resolve()
    path = Path(path) if path else REPORT / ("plans.json" if partial else "publication-plans.json")
    if not (root / path).is_file() and path == REPORT / "publication-plans.json":
        path = REPORT / "plans.json"
    return read_json(root / path), path


def execution_rounds(plan):
    return plan.get("execution_rounds") or [{"round": plan["round"], "campaign_id": plan["round"],
                                            "campaign_cap_seconds": plan["campaign_cap_seconds"]}]


def paid_attempts(root, plan):
    """Count each collected physical attempt once, including cancelled work."""
    root = Path(root).resolve()
    records, seen = [], set()
    for cohort in execution_rounds(plan):
        queue = root / cohort.get("queue_path", f"runs/forge/{cohort['round']}/queue")
        state = Queue(queue, on_completion=None).inspect()
        campaign = cohort.get("campaign_id", cohort["round"])
        charges = [charge for charge in state["charges"] if charge["owner"]["campaign"] == campaign]
        frozen_ids = cohort.get("paid_attempt_ids_frozen")
        if frozen_ids is not None and ({charge["attempt_id"] for charge in charges} != set(frozen_ids)
                or not math.isclose(sum(charge["seconds"] for charge in charges),
                                    cohort["paid_wall_seconds_frozen"], abs_tol=1e-8)):
            raise ValueError("Original paid attempts changed after publication planning")
        if sum(charge["seconds"] for charge in charges) > cohort["campaign_cap_seconds"]:
            raise ValueError("Execution round exceeds its frozen paid allowance")
        for charge in charges:
            attempt = charge["attempt_id"]
            if attempt in seen:
                raise ValueError("Execution rounds overlap the same paid attempt")
            seen.add(attempt)
            request = state["submissions"][charge["owner"]["request"]]["request"]
            receipt = project_receipt(root, attempt)
            source = receipt["provenance"]
            if (receipt["campaign_id"] != campaign or receipt["candidate_revision"] != charge["owner"]["revision"]
                    or source["source_digest"] != request["source"]["digest"]
                    or source["source_origin_commit"] != request["source"]["origin_commit"]
                    or not math.isfinite(charge["seconds"]) or charge["seconds"] < 0):
                raise ValueError("Paid attempt differs from its frozen cost owner or source")
            records.append({"attempt_id": attempt, "round": cohort["round"], "campaign_id": campaign,
                "candidate_id": receipt["candidate_id"], "candidate_revision": receipt["candidate_revision"],
                "paid_wall_seconds": charge["seconds"], "source_digest": source["source_digest"],
                "source_commit": source["source_origin_commit"], "attempt_status": receipt["attempt_status"],
                "task_statuses": {row["task_id"]: row["gate_status"] for row in receipt["task_results"]},
                "canonical_result_hash": source["canonical_result_hash"]})
    return sorted(records, key=lambda record: record["attempt_id"])


def pure_recipe(recipe):
    if (recipe["optimizer_family"] != "adam" or recipe["reg_arm"] != "b_cap"
            or recipe["lr_floor"] != 1 or recipe["network_lr_floor"] != 1
            or any(recipe[key] for key in ("d_guard_ratio", "reg_anchor_weight", "latent_damping_max_rate",
                    "direct_particle_gain", "prior_reg", "ema_decay", "input_noise_std", "output_noise_std"))
            or recipe["reg_coeff_end"] is not None or recipe["beta2_end"] is not None):
        raise ValueError("Published recipe differs from the declared pure fixed-rate BCAP family")


def export_readout(root=ROOT, *, plan_path=None, partial=False):
    root = Path(root).resolve()
    report = root / REPORT
    plan, path = load_plan(root, plan_path, partial=partial)
    summaries = [read_json(root / "reports/forge/configuration-search" / (Path(spec).stem + ".json"))
                 for spec in plan["specs"]]
    trials = [trial for summary in summaries for trial in summary["trials"]]
    expected = {trial["candidate_id"]: trial["candidate_revision"] for trial in plan["trials"]}
    if (len(trials) != len(expected) or len(expected) != plan["configuration_count"]
            or {trial["candidate_id"]: trial["candidate_revision"] for trial in trials} != expected):
        raise ValueError("Readout differs from its entire frozen candidate roster")
    if not partial and not all(summary["selection"]["all_trials_terminal"] for summary in summaries):
        raise ValueError("Complete publication requires terminal search trials")
    cohorts = plan.get("source_cohorts") or [{"source_digest": plan["source_digest"],
        "source_commit": None, "candidate_ids": list(expected)}]
    sources = {}
    for cohort in cohorts:
        for candidate in cohort["candidate_ids"]:
            if candidate in sources:
                raise ValueError("A whole candidate cannot pool source cohorts")
            sources[candidate] = dict(cohort)
    if set(sources) != set(expected):
        raise ValueError("Executed source cohorts differ from the candidate roster")
    paid = paid_attempts(root, plan)
    if not partial and plan.get("expected_paid_attempt_count", len(paid)) != len(paid):
        raise ValueError("Final readout differs from the declared complete unique-attempt roster")
    by_attempt = {record["attempt_id"]: record for record in paid}
    attempts, candidates, counts, diagnostic = set(), [], Counter(), Counter()
    projected = {}
    for record in paid:
        attempt = record["attempt_id"]
        destination = report / "receipts" / (attempt + ".json")
        atomic_json(destination, project_receipt(root, attempt))
        record.update(receipt=destination.relative_to(root).as_posix(), receipt_sha256=file_hash(destination))
        projected[attempt] = read_json(destination)
    print(f"Certified {len(paid)} unique paid receipts; exporting saved observations", flush=True)
    for trial in trials:
        recipe = trial["declaration"]["resolved_configuration_recipe"]
        pure_recipe(recipe)
        source = sources[trial["candidate_id"]]
        if trial["source_digest"] != source["source_digest"]:
            raise ValueError("Trial report belongs to a different executed source")
        tasks, unmeasured = [], []
        for task in trial["tasks"]:
            if task["qualification_tier"] != 1:
                continue
            if task["gate_status"] not in {"PASS", "FAIL", "INCOMPLETE", "INVALID", "BLOCKED"} or not task.get("attempt_id"):
                if not partial:
                    raise ValueError("Complete readout requires all declared task measurements")
                unmeasured.append(task["task"])
                continue
            if not partial and task["gate_status"] not in {"PASS", "FAIL"}:
                raise ValueError("Complete readout requires PASS or FAIL numerical measurements")
            attempt = task["attempt_id"]
            if attempt in attempts or attempt not in by_attempt:
                raise ValueError("A task must own one disjoint collected paid attempt")
            attempts.add(attempt)
            receipt = projected[attempt]
            provenance = receipt["provenance"]
            if (receipt["candidate_revision"] != trial["candidate_revision"]
                    or provenance["source_digest"] != source["source_digest"]
                    or (source["source_commit"] is not None and provenance["source_origin_commit"] != source["source_commit"])):
                raise ValueError("Candidate receipt differs from its complete executed source cohort")
            source["source_commit"] = provenance["source_origin_commit"]
            target = counts if task["importance"] == "required" else diagnostic
            target[task["gate_status"]] += 1
            rendered = export_attempt(root / "reports/forge/attempts" / attempt,
                                      report / "media" / trial["configuration_id"][:12])
            if not partial and len(rendered) != 1:
                raise ValueError("A complete task requires its saved actual-training GIF")
            if len(rendered) > 1:
                raise ValueError("Task receipt unexpectedly contains several physical task measurements")
            tasks.append({"task_id": task["task"], "importance": task["importance"],
                "status": task["gate_status"], "metrics": task["metrics"], "cost": task["cost"],
                "compatibility_key": task["compatibility_key"], "attempt_id": attempt,
                "receipt": by_attempt[attempt]["receipt"], "receipt_sha256": by_attempt[attempt]["receipt_sha256"],
                "gif_receipt": rendered[0] if rendered else None,
                "reason": task.get("reason")})
        required_total = len(plan["required_task_ids"])
        if not partial and (len(tasks) != len(plan["task_ids"]) or {task["task_id"] for task in tasks} != set(plan["task_ids"])):
            raise ValueError("Whole candidate is missing a declared Tier 1 peer")
        candidates.append({"candidate_id": trial["candidate_id"], "candidate_revision": trial["candidate_revision"],
            "configuration_id": trial["configuration_id"], "loss": recipe.get("loss", "relativistic"), "lr": recipe["lr"],
            "recipe": recipe, "source_digest": source["source_digest"], "source_commit": source["source_commit"],
            "required_passes": sum(task["importance"] == "required" and task["status"] == "PASS" for task in tasks),
            "required_total": required_total, "tasks": tasks, "unmeasured_task_ids": sorted(unmeasured),
            "submission_status": trial["submission_status"], "cost": trial["cost"]})
        print(f"Exported {recipe.get('loss', 'relativistic')} lr={recipe['lr']}: "
              f"{len(tasks)} measured, {len(unmeasured)} unmeasured Tier 1 tasks", flush=True)
    for cohort in cohorts:
        commits = {sources[candidate]["source_commit"] for candidate in cohort["candidate_ids"]}
        if len(commits) != 1 or None in commits:
            raise ValueError("Every executed source cohort requires one certified origin commit")
        cohort["source_commit"] = next(iter(commits))
    cohorts = [{key: cohort[key] for key in ("source_commit", "source_digest", "candidate_ids")} for cohort in cohorts]
    # Candidate costs already include their retries. Prior context belongs to
    # candidates omitted from the final roster, not omitted task-attempt cells.
    unselected = [record for record in paid if record["candidate_id"] not in expected]
    if not partial and (plan.get("selected_measurement_attempt_count", len(attempts)) != len(attempts)
            or plan.get("original_unselected_paid_attempt_count", len(unselected)) != len(unselected)):
        raise ValueError("Final selected measurements or retained original context count differs")
    paid_seconds = sum(record["paid_wall_seconds"] for record in paid)
    selected_cost = sum(candidate["cost"]["new_paid_wall_seconds"] for candidate in candidates)
    prior_cost = sum(record["paid_wall_seconds"] for record in unselected)
    if (paid_seconds > plan["campaign_cap_seconds"]
            or not math.isclose(paid_seconds, selected_cost + prior_cost, abs_tol=1e-5)):
        raise ValueError("Whole-candidate and unique-attempt campaign accounting differ")
    result = {"schema_version": 2, "round": plan["round"],
        "scope": "partial_pure_bcap_initial_readout" if partial else "finite_pure_bcap_initial_readout",
        "qualification_input": False, "default_adoption": False, "source_cohorts": cohorts,
        "view_revision": plan["view_revision"], "required_counts": dict(counts), "diagnostic_counts": dict(diagnostic),
        "unique_paid_attempts": len(paid), "selected_measurement_attempts": len(attempts),
        "paid_wall_seconds": paid_seconds, "maximum_paid_seconds": plan["campaign_cap_seconds"],
        "candidates": sorted(candidates, key=lambda candidate: (candidate["loss"], candidate["lr"])),
        "paid_attempts": paid, "original_unselected_paid_attempt_ids": [record["attempt_id"] for record in unselected],
        "original_unselected_paid_wall_seconds": prior_cost, "execution_rounds": execution_rounds(plan),
        "plan": path.as_posix(), "plan_sha256": file_hash(root / path),
        "cost_note": "Unique physical paid attempts, including cancelled original work, count once; overlapping per-search campaign totals are not added.",
        "scoring_note": "Original sustained task criteria; final metrics and actual saved training GIFs project certified receipts. Partial cells grant no whole-candidate selection."}
    if not partial:
        result["selection"] = select_configuration(trials, 1)
    result["input_digest"] = stable_hash(result)
    output = report / ("original-initial-readout.json" if partial else "readout.json")
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--partial-initial", action="store_true")
    args = parser.parse_args()
    result = export_readout(args.root, plan_path=args.plan, partial=args.partial_initial)
    print(f"Certified {result['unique_paid_attempts']} unique paid attempts; required {result['required_counts']}, "
          f"diagnostic {result['diagnostic_counts']}; {result['paid_wall_seconds']:.3f} paid seconds; {result['scope']}", flush=True)


if __name__ == "__main__":
    main()
