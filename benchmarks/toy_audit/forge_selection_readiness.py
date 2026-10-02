"""Read-only config options from Forge's actual current qualification predicates.

PR228 inventories declared tests. Forge's knowledge board independently grades
compatible evidence; calibrated selection and default adoption remain separate.
This command neither starts work nor chooses a metric winner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from experiments.forge import calibration, knowledge, planning, views
from experiments.forge.contracts import file_hash, stable_hash
from experiments.forge.tier_report import build_report


ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("pinned_rows", "calibration_rows", "historical_rows")
TOOL_FILES = (
    "benchmarks/toy_audit/forge_selection_readiness.py",
    "experiments/forge/tier_report.py", "experiments/forge/knowledge.py",
    "experiments/forge/views.py", "experiments/forge/planning.py",
    "experiments/forge/calibration.py", "experiments/forge/sources.py",
    "experiments/forge/board_filters.py", "experiments/forge/promotion.py",
)


def _commit(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def _resolve(root, view_id, row):
    runtime = row["runtime_cohort"]
    backend = runtime["execution_backend"]
    model = runtime.get("compute_profiles", {}).get("cuda", {}).get("model")
    request = planning.resolve_idea(root, row["candidate_id"], view_id=view_id,
        through_tier=3, freeze_source=False, execution_backend=backend,
        cuda_model=model if backend == "cuda" else None)
    if (request["candidate_revision"] != row["candidate_revision"]
            or request["source"]["digest"] != row["source_digest"]
            or knowledge._runtime_cohort(request) != runtime):
        raise ValueError("candidate/source/runtime changed during config readout")
    return request


def build_readiness(root=ROOT, *, view_id="quality_coverage", backend=None, candidate_id=None):
    """Select whole current rows; never shrink their required denominator."""
    root = Path(root).resolve()
    if backend not in (None, "cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    inventory = build_report(root, view_id)
    view = views.load_view(root, view_id)
    idea_paths = sorted((root / "configs/forge/ideas").glob("*.json"))
    ideas = {json.loads(path.read_text())["id"]: path for path in idea_paths}
    if candidate_id is not None and candidate_id not in ideas:
        raise ValueError(f"unknown candidate {candidate_id!r}")
    paths = {name: root / name for name in inventory["input_hashes"]}
    paths.update({str(path.relative_to(root)): path for path in idea_paths})
    criteria = root / "configs/forge/calibration/criteria-v1.json"
    if criteria.is_file(): paths[str(criteria.relative_to(root))] = criteria
    before = {name: file_hash(path) for name, path in paths.items()}
    result = knowledge.board(root, view_id, include_bindings=True)
    if result["policy_fingerprint"] != views.view_fingerprint(view):
        raise ValueError("view policy changed during config readout")
    required = {a["task"] for a in view["assignments"] if a["importance"] == "required"}
    records, recipes, rows, reasons = [], {}, [], {}
    for row in result["current_rows"]:
        if row.get("evidence_scope") != "current":
            raise ValueError("noncurrent evidence in current config rows")
        if backend and row["runtime_cohort"]["execution_backend"] != backend: continue
        if candidate_id and row["candidate_id"] != candidate_id: continue
        q = row.get("qualification", {})
        task_rows = q.get("tasks", [])
        if q and {t["task_id"] for t in task_rows if t["importance"] == "required"} != required:
            raise ValueError("qualification omitted a required declared task")
        screen_pass = row["status"] == "PASS" and q.get("status") == "PASS" and q.get("eligible") is True
        request = _resolve(root, view_id, row) if row.get("candidate_revision") is not None else None
        if screen_pass and request is None:
            raise ValueError("qualified option lacks an exact resolved source identity")
        try:
            if request is None: raise ValueError("candidate resolution is blocked")
            verified = calibration.verify_calibration(root, request)
            calibrated = {"verified": True, "binding": verified}
        except (ValueError, KeyError, TypeError, OSError) as error:
            calibrated = {"verified": False, "reason": str(error)}
        status_map = {t["task_id"]: t["status"] for t in task_rows}
        for name in required: status_map.setdefault(name, "BLOCKED" if row["status"] == "BLOCKED" else "NOT_RUN")
        reason_keys = {}
        for task in task_rows:
            if task["status"] != "PASS" and task.get("reasons"):
                digest = stable_hash(task["reasons"])
                reasons[digest] = task["reasons"]
                reason_keys[task["task_id"]] = digest
        bindings = row.get("scientific_bindings", {})
        recipe_key = bindings.get("recipe_sha256")
        if recipe_key is not None: recipes[recipe_key] = bindings.get("recipe")
        item = {"candidate_id": row["candidate_id"], "candidate_revision": row.get("candidate_revision"),
            "cohort": row.get("cohort"), "runtime_cohort": row["runtime_cohort"],
            "status": row["status"], "qualified_tier": row["qualified_tier"],
            "screen_qualified": screen_pass, "calibration": calibrated,
            "config": str(ideas[row["candidate_id"]].relative_to(root)),
            "config_sha256": file_hash(ideas[row["candidate_id"]]),
            "bindings": {k: deepcopy(bindings.get(k)) for k in (
                "recipe_sha256", "prior", "initializer", "claim_contract", "rng_sha256",
                "source_digest", "source_origin_commit")},
            "protocol_sha256": stable_hash(bindings.get("protocol")),
            "task_statuses": status_map, "tiers": deepcopy(q.get("tiers", [])),
            "status_reason_refs": reason_keys, "preflight_blockers": row.get("preflight_blockers", row.get("blockers", [])),
            "attempt_ids": row.get("attempt_ids", [])}
        rows.append(item)
        if screen_pass:
            records.append({"candidate_id": item["candidate_id"], "candidate_revision": item["candidate_revision"],
                "cohort": item["cohort"], "runtime_cohort": item["runtime_cohort"],
                "config": item["config"], "config_sha256": item["config_sha256"],
                "recipe_sha256": recipe_key, "calibration_verified": calibrated["verified"]})
    if before != {name: file_hash(path) for name, path in paths.items()}:
        raise ValueError("declarations changed during config readout")
    calibrated_options = [r for r in records if r["calibration_verified"]]
    summary = {
        "all_current_cohorts": len(result["current_rows"]), "shown_current_cohorts": len(rows),
        "current_statuses": dict(sorted(Counter(r["status"] for r in rows).items())),
        "qualified_tiers": dict(sorted(Counter(r["qualified_tier"] for r in rows).items())),
        "required_cell_statuses": dict(sorted(Counter(r["task_statuses"][name] for r in rows for name in required).items())),
        "required_cells": len(rows) * len(required),
        "excluded_evidence_cohorts": {scope: len(result[scope]) for scope in SCOPES},
    }
    conflicts = deepcopy(result.get("conflicts", []))
    return {"schema": "particlegan_forge_selection_readiness_v1", "source_commit": _commit(root),
        "view": view_id, "view_revision": view["revision"], "backend_filter": backend,
        "candidate_filter": candidate_id, "policy_fingerprint": result["policy_fingerprint"],
        "decision": "CONFLICTS_REQUIRE_REVIEW" if conflicts else "QUALIFIED_OPTIONS_AVAILABLE" if records else "NO_QUALIFIED_OPTION",
        "qualified_options": records, "calibrated_options": calibrated_options,
        "selected_config": None, "default_adoption": "NOT_ASSESSED_REQUIRES_SEPARATE_REGISTERED_ROBUSTNESS",
        "declared_calibration": view.get("calibration"), "conflicts": conflicts,
        "summary": summary, "rows": rows, "recipe_contracts": recipes, "status_reasons": reasons,
        "criteria_inventory": inventory, "declaration_hashes": before,
        "source_board_sha256": stable_hash(result),
        "tool_sources": {name: file_hash(ROOT / name) for name in TOOL_FILES},
        "training_updates": 0, "queue_writes": 0, "default_changes": 0,
        "notes": [
            "PR228 inventories requirements; eligibility uses Forge's actual independently graded current board.",
            "Qualified options are provisional screen passes until calibration verifies their exact cohort.",
            "No aggregate metric objective is declared, so no best config is selected by name, cost or partial metrics.",
            "Whole-row filters retain every required task, including missing, blocked, invalid and incomplete cells.",
            "Pinned, historical and calibration diagnostic outcomes cannot fill current qualification cells.",
            "Standalone PR239–243 toy tests are not registered Forge tasks and supply no automatic qualification.",
            "New source bytes change Forge's broad source digest; earlier receipts remain under their own identities.",
            "Even calibrated qualification does not register, execute or pass the separate default-promotion robustness stage.",
        ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--view", default="quality_coverage")
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--candidate", help="exact declared candidate ID; filters whole cohorts")
    parser.add_argument("--output", type=Path, help="explicit new JSON report path; default stdout only")
    args = parser.parse_args(argv)
    try:
        report = build_readiness(args.root, view_id=args.view, backend=args.backend, candidate_id=args.candidate)
        encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream: stream.write(encoded)
        else: print(encoded, end="")
        return 0 if report["qualified_options"] and not report["conflicts"] else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"READINESS ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
