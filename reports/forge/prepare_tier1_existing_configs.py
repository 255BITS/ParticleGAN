"""Plan or enqueue the exact registered Tier 1 roster; never launch training."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.__main__ import queue_location
from experiments.forge.configuration_search import enqueue_search, plan_search
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.planning import discover_candidate_ids
from experiments.forge.queue import Queue
from experiments.forge.technique_inventory import discover_techniques, enqueue_inventory, plan_inventory
from experiments.forge.views import load_view


ROUND = Path("configs/forge/rounds/tier1-existing-configs-v1.json")


def _progress(stage, scope):
    print(json.dumps({"event": "tier1_preparation", "stage": stage, "scope": scope}),
          file=sys.stderr, flush=True)


def _row(candidate, *, study=None):
    return {
        "candidate_id": candidate.get("candidate_id", candidate.get("candidate")),
        "declaration_kind": "configuration" if study else "idea",
        "study_id": study,
        **{key: candidate[key] for key in (
            "candidate_revision", "source_digest", "submission_status",
            "submission_blockers", "declared_worst_case_seconds")},
        "unreused_worst_case_seconds": candidate.get(
            "unreused_worst_case_seconds", candidate.get("worst_case_seconds")),
        "request_id": candidate.get("request_id"),
    }


def _verify_plans(round_definition, campaign, inventory, searches):
    """Reject roster, source, policy or ceiling drift before queue admission."""
    if sorted(searches) != sorted(round_definition["studies"]):
        raise ValueError("prepared studies differ from the frozen round")
    rows = [_row(row) for row in inventory["candidates"]]
    rows += [_row(row, study=study) for study, plan in searches.items() for row in plan["trials"]]
    names = [row["candidate_id"] for row in rows]
    if len(names) != len(set(names)) or sorted(names) != sorted(round_definition["candidate_ids"]):
        raise ValueError("prepared candidates differ from the exact frozen roster")
    for kind, field in (("idea", "idea_ids"), ("configuration", "configuration_ids")):
        if sorted(row["candidate_id"] for row in rows if row["declaration_kind"] == kind) != sorted(round_definition[field]):
            raise ValueError(f"prepared {kind} declarations differ from the frozen round")
    plans = [inventory, *searches.values()]
    if any(plan["campaign"] != campaign for plan in plans):
        raise ValueError("all studies and ideas must share the exact common campaign")
    if any(plan["view"] != round_definition["view"] or
           plan["execution_backend"] != round_definition["execution_backend"] for plan in plans):
        raise ValueError("prepared view/runtime differs from the frozen round")
    if inventory["through_tier"] != 1 or any(plan["tuning_through_tier"] != 1 for plan in searches.values()):
        raise ValueError("this round authorizes Tier 1 only")
    if not inventory["source_digest"] or len({plan["source_digest"] for plan in plans}) != 1 or any(
            row["source_digest"] != inventory["source_digest"] for row in rows):
        raise ValueError("all candidates require one frozen source digest")
    if len({row["policy_fingerprint"] for row in inventory["candidates"]} |
           {plan["policy_fingerprint"] for plan in searches.values()}) != 1:
        raise ValueError("all candidates require one exact view policy")
    ceiling = sum(row["declared_worst_case_seconds"] for row in rows)
    per_candidate = round_definition["candidate_reservation_seconds"]
    if (ceiling != round_definition["worst_case_campaign_reservation_seconds"] or
            campaign["budget_seconds"] != ceiling or
            campaign["candidate_budget_seconds"] != per_candidate or
            any(row["declared_worst_case_seconds"] != per_candidate for row in rows)):
        raise ValueError("campaign must cover the complete frozen per-candidate task allowances")
    return sorted(rows, key=lambda row: row["candidate_id"])


def prepare(root, queue_root, *, stage="plan"):
    if stage not in {"plan", "enqueue"}:
        raise ValueError("only plan or enqueue is supported; execution is separate")
    root, queue_root = Path(root).resolve(), Path(queue_root).resolve()
    round_definition = read_json(root / ROUND)
    if discover_candidate_ids(root) != sorted(round_definition["candidate_ids"]):
        raise ValueError("current checkout declarations differ from the frozen 47-candidate roster")
    if discover_techniques(root) != sorted(round_definition["idea_ids"]):
        raise ValueError("current idea roster differs from the frozen round")
    view = load_view(root, round_definition["view"])
    denominator = [sum(row["importance"] == "required" and row["qualification_tier"] == tier
                       for row in view["assignments"]) for tier in (1, 2, 3)]
    if view["revision"] != round_definition["view_revision"] or denominator != round_definition["required_denominator_by_tier"]:
        raise ValueError("current view revision/denominator differs from the frozen round")
    campaign = read_json(root / round_definition["campaign"])
    options = dict(view_id=round_definition["view"], through_tier=1,
                   execution_backend=round_definition["execution_backend"],
                   cuda_model=round_definition["cuda_model"], campaign=campaign)
    queue = Queue(queue_root, report_root=root / "reports/forge")
    _progress("plan", "15 ideas")
    inventory = plan_inventory(root, queue_root, queue=queue, **options)
    searches = {}
    for study in round_definition["studies"]:
        _progress("plan", study)
        searches[study] = plan_search(root, queue_root, study, queue=queue)
    rows = _verify_plans(round_definition, campaign, inventory, searches)
    initial_source = inventory["source_digest"]
    if stage == "enqueue":
        # Every declaration has passed global planning before the first admission.
        # Existing enqueue APIs freeze/verify source and materialize exact cards.
        searches = {}
        for study in round_definition["studies"]:
            _progress("enqueue", study)
            searches[study] = enqueue_search(root, queue_root, study, queue=queue)
        _progress("enqueue", "15 ideas")
        inventory = enqueue_inventory(root, queue_root, queue=queue, **options)
        rows = _verify_plans(round_definition, campaign, inventory, searches)
        if inventory["source_digest"] != initial_source:
            raise ValueError("source changed during admission; do not drain this round")
    return {
        "schema_version": 1, "round": round_definition["id"], "stage": stage,
        "round_sha256": stable_hash(round_definition), "source_digest": initial_source,
        "view": view["id"], "view_revision": view["revision"],
        "required_denominator_by_tier": denominator, "through_tier": 1,
        "execution_backend": round_definition["execution_backend"], "cuda_model": round_definition["cuda_model"],
        "campaign": campaign, "queue_root": str(queue_root), "logs": str(queue_root / "events.jsonl"),
        "candidate_count": len(rows), "status_counts": dict(sorted(Counter(row["submission_status"] for row in rows).items())),
        "declared_worst_case_seconds": sum(row["declared_worst_case_seconds"] for row in rows),
        "unreused_worst_case_seconds": sum(row["unreused_worst_case_seconds"] for row in rows if not row["submission_blockers"]),
        "candidates": rows, "training_launched": False,
        "execution": "Drain this common campaign separately only after reviewing all prepared admissions.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "enqueue"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--output", type=Path, help="save the compact preparation receipt")
    args = parser.parse_args(argv)
    result = prepare(args.root, queue_location(args.root, args.queue_root), stage=args.stage)
    output = args.output or Path(result["queue_root"]) / result["round"] / f"preparation-{args.stage}.json"
    atomic_json(output, result)
    summary = {key: result[key] for key in (
        "round", "stage", "candidate_count", "status_counts", "source_digest",
        "declared_worst_case_seconds", "unreused_worst_case_seconds", "training_launched", "logs")}
    summary["output"] = str(output.resolve())
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
