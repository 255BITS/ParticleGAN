"""Single-command Forge readout checker and refresher when no code changes are needed.

Usage:
  python ops/refresh_forge_readouts.py --check
  python ops/refresh_forge_readouts.py --gpus 0,1 --scaling-gpu 0
  python ops/refresh_forge_readouts.py --reports-only

Exit codes:
  0  All Forge readouts are current (or were refreshed cleanly to current).
  1  (--check only) Out-of-date Forge readouts remain and can be refreshed automatically.
  2  Code or declaration changes are required (preflight error, broken task/adapter,
     unrecoverable stale queue state, or non-gate task execution failure such as ERROR/OOM).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.forge import attempt_coverage, history, knowledge, planning, queue as forge_queue
from experiments.forge import scaling, scaling_export, scaling_publication, studies
from experiments.forge.contracts import atomic_json, file_hash, identifier, read_json, stable_hash
from experiments.forge.sources import inspect_source, snapshot_source
from experiments.forge.trainer_families import CURRENT_SELECTION, current_family_candidates
from experiments.forge.views import (
    diagnostic_evidence_scope,
    load_tasks,
    load_view,
    task_evaluation_fingerprint,
    task_execution_fingerprint,
)

CURRENT_EXIT = 0
NEEDS_REFRESH_EXIT = 1
CODE_CHANGE_EXIT = 2

DEFAULT_CUDA_MODEL = "NVIDIA RTX A6000"
ORDINARY_LEADERBOARD_VIEWS = (
    "adaptation",
    "clockfree_continuous",
    "formulation_comparison",
    "host_profile_transfer",
    "quality_coverage",
)
SCOPED_ATTEMPT_KEYS = ("research_scope", "repair_scope", "policy_scope", "cohort_scope")
OP_COMPLEMENT = {">=": "<", "<=": ">", ">": "<=", "<": ">=", "==": "<"}


class CodeChangeRequiredError(RuntimeError):
    """Raised when refreshing readouts cannot proceed without code or card changes."""


def _log(message: str, *, json_mode: bool = False) -> None:
    stream = sys.stderr if json_mode else sys.stdout
    print(f"[refresh-forge] {message}", file=stream, flush=True)


def _is_scoped_attempt(record: dict[str, Any]) -> bool:
    return any(record.get(key) for key in SCOPED_ATTEMPT_KEYS)


def catalog_source_manifest(root: Path, idea: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the exact source manifest (including task evaluation/hostprofile extras) used by planning.resolve_idea."""
    from experiments.forge.hostprofiles import profile_source_paths

    root = Path(root).resolve()
    extra_sources = set((idea or {}).get("source_files", []))
    for task in load_tasks(root).values():
        extra_sources.update(
            relative
            for relative in task.get("evaluation", {}).get("sources", {})
            if (root / relative).is_file()
        )
        try:
            profile_sources = profile_source_paths(task)
        except (KeyError, TypeError, ValueError):
            profile_sources = {}
        extra_sources.update(
            relative for relative in profile_sources if (root / relative).is_file()
        )
    return inspect_source(root, sorted(extra_sources))


def _request_matches_worktree(root: Path, req: dict[str, Any]) -> bool:
    expected = catalog_source_manifest(root, req.get("candidate"))["digest"]
    return req.get("source", {}).get("digest") == expected


def scaling_source_files_sha256(root: Path) -> dict[str, str]:
    """Compute the exact code-file digest map used by scaling_benchmark._source()."""
    root = Path(root).resolve()
    paths = sorted(
        set((root / "particlegan").rglob("*.py"))
        | {
            root / "experiments/forge/scaling_benchmark.py",
            root / "experiments/forge/api.py",
            root / "experiments/forge/rng.py",
            root / "experiments/forge/priors.py",
            root / "experiments/forge/state.py",
            root / "experiments/forge/contracts.py",
            root / "benchmarks/toy_audit/reproducibility.py",
        }
    )
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise CodeChangeRequiredError(f"missing scaling benchmark source files: {missing}")
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def _git_task_names_at_commit(root: Path, commit: str) -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, "configs/forge/tasks"],
            cwd=root,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise CodeChangeRequiredError(f"cannot inspect frozen commit {commit}: {exc}") from exc
    names = set()
    for line in output.splitlines():
        path = Path(line.strip())
        if path.suffix == ".json" and path.parent.as_posix() == "configs/forge/tasks":
            names.add(path.stem)
    return names


def _selected_family_source_commit(root: Path, publication: dict[str, Any], family: str, selection: dict[str, Any]) -> str:
    sources = publication.get("evidence_sources", {})
    pub_key = selection.get("publication_key")
    commit = sources.get(pub_key, {}).get("source_commit") if pub_key else None
    if not commit:
        commit = selection.get("bindings", {}).get("source_origin_commit")
    if not commit:
        preflight = root / "reports/forge/tier-coverage-20261014/preflight.json"
        if preflight.is_file():
            pins = {item["family"]: item for item in read_json(preflight).get("families", [])}
            commit = pins.get(family, {}).get("source_commit") or pins.get(family, {}).get("origin_commit")
    if not commit:
        raise CodeChangeRequiredError(f"selected family {family} has no recorded frozen source commit")
    return commit


def _selected_family_preserved_absent_paths(
    root: Path, publication: dict[str, Any], selection: dict[str, Any]
) -> list[str]:
    pub_key = selection.get("publication_key")
    snap_rel = publication.get("evidence_sources", {}).get(pub_key, {}).get("snapshot") if pub_key else None
    if not snap_rel:
        return []
    snap_path = root / snap_rel
    if not snap_path.is_file():
        return []
    closure = read_json(snap_path).get("frozen_catalog_support_closure") or {}
    return sorted(set(closure.get("preserved_absent_paths") or []))


def check_queue_status(root: Path, queue_root: Path) -> dict[str, Any]:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return {"current": True, "active_submissions": [], "awaiting_readout_submissions": []}
    state = read_json(state_path)
    active, awaiting = [], []
    for req_id, entry in sorted(state.get("submissions", {}).items()):
        status = entry.get("status")
        lifecycle = entry.get("lifecycle")
        cand = entry.get("request", {}).get("candidate", {}).get("id", "unknown")
        study_id = studies.selected_study_id(entry.get("request", {}))
        if status in {"queued", "running", "paused"}:
            active.append({"request_id": req_id, "candidate_id": cand, "study_id": study_id, "status": status})
        elif lifecycle == "awaiting_readout":
            req = entry.get("request", {})
            keys = [job["compatibility_key"] for job in req.get("jobs", [])]
            has_results = any(state.get("jobs", {}).get(k, {}).get("result") is not None for k in keys)
            if not has_results and not _request_matches_worktree(root, req):
                raise CodeChangeRequiredError(
                    f"queue submission {req_id} ({cand}) is awaiting_readout on superseded source "
                    f"with zero durable attempts; supersede or clean its lifecycle explicitly"
                )
            awaiting.append({"request_id": req_id, "candidate_id": cand, "study_id": study_id, "status": status})
    return {
        "current": not active and not awaiting,
        "active_submissions": active,
        "awaiting_readout_submissions": awaiting,
    }


def check_scaling_performance(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    try:
        evidence = scaling_publication.load_performance(root)
    except ValueError as exc:
        raise CodeChangeRequiredError(f"invalid scaling-performance.json: {exc}") from exc
    expected_choices = current_family_candidates(root)
    expected_perf_pairs = {
        (item.get("trainer_family") or item["family"], item["candidate_id"])
        for item in expected_choices
    }
    expected_reg_pairs = {(item["family"], item["candidate_id"]) for item in expected_choices}
    expected_tasks = [item["task"] for item in load_view(root, "dcgan_performance")["assignments"]]
    expected_cells = {(fam, cid, tid) for fam, cid in expected_perf_pairs for tid in expected_tasks}
    if evidence is None:
        return {"current": False, "reason": "missing_manifest", "missing_cells": len(expected_cells)}

    unavailable = [
        row for row in evidence.get("unavailable", [])
        if (row["trainer_family"], row["candidate_id"]) in expected_perf_pairs
    ]
    records = [
        row for row in evidence.get("records", [])
        if (row["trainer_family"], row["candidate_id"]) in expected_perf_pairs
    ]
    complete_cells = set()
    non_complete = []
    cohorts = set()
    expected_files_sha = scaling_source_files_sha256(root)
    stale_source_cells = []

    for row in records:
        cell = (row["trainer_family"], row["candidate_id"], row["task_id"])
        result = row["result"]
        status = scaling_publication.performance_status(result)
        if status == "COMPLETE":
            complete_cells.add(cell)
        else:
            non_complete.append({"cell": cell, "status": status})
        cohorts.add(
            (
                result["identity"]["source"]["sha256"],
                result["identity"]["runtime_sha256"],
                stable_hash(scaling_publication.hardware_key(result)),
            )
        )
        if result["identity"]["source"].get("files_sha256") != expected_files_sha:
            stale_source_cells.append(cell)

    declaration_mismatches = []
    reg_id = (evidence.get("export") or {}).get("registration")
    if reg_id:
        reg_path = scaling.manifest_path(root, reg_id)
        if reg_path.is_file():
            reg_doc = read_json(reg_path)
            by_pair = {
                (item["family"], item["candidate_id"]): item["candidate_declaration_sha256"]
                for item in reg_doc.get("family_selections", [])
            }
            for fam, cid in sorted(expected_reg_pairs):
                idea_path = root / "configs/forge/ideas" / f"{cid}.json"
                if by_pair.get((fam, cid)) != file_hash(idea_path):
                    declaration_mismatches.append((fam, cid))
        else:
            declaration_mismatches.append(("registration_missing", reg_id))

    missing = sorted(expected_cells - complete_cells)
    current = (
        not unavailable
        and not non_complete
        and not missing
        and len(cohorts) == 1
        and not stale_source_cells
        and not declaration_mismatches
    )
    return {
        "current": current,
        "expected_cells": len(expected_cells),
        "complete_cells": len(complete_cells),
        "missing_cells": len(missing),
        "unavailable_cells": len(unavailable),
        "non_complete_cells": non_complete,
        "table_cohorts": len(cohorts),
        "stale_source_cells": len(stale_source_cells),
        "declaration_mismatches": declaration_mismatches,
    }


def check_attempt_coverage(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    pub_path = root / "reports/forge/technique-inventory.json"
    cov_path = root / "reports/forge/attempt-coverage.json"
    if not pub_path.is_file() or not cov_path.is_file():
        return {"current": False, "reason": "missing_report"}
    publication = read_json(pub_path)
    coverage = read_json(cov_path)
    summary = coverage.get("summary", {})
    pending = int(summary.get("pending", 0))
    running = int(summary.get("running", 0))

    selected = attempt_coverage.selected_tasks(publication)
    expected_selected: dict[str, tuple[str, str, str, list[str]]] = {}
    commit_tasks_cache: dict[str, set[str]] = {}
    for family, selection, tasks in selected:
        commit = _selected_family_source_commit(root, publication, family, selection)
        if commit not in commit_tasks_cache:
            commit_tasks_cache[commit] = _git_task_names_at_commit(root, commit)
        available = [task for task in tasks if task in commit_tasks_cache[commit]]
        expected_selected[family] = (
            selection["candidate_id"],
            selection["candidate_revision"],
            selection["bindings"]["source_digest"],
            available,
        )

    superseded_records = []
    covered_by_family: dict[str, set[str]] = {family: set() for family in expected_selected}
    for record in coverage.get("records", []):
        if _is_scoped_attempt(record):
            continue
        family = record.get("family")
        spec = expected_selected.get(family)
        if spec is None or (
            record.get("candidate_id"),
            record.get("candidate_revision"),
            record.get("source_digest"),
        ) != spec[:3]:
            superseded_records.append(
                {
                    "family": family,
                    "candidate_id": record.get("candidate_id"),
                    "task": record.get("task"),
                }
            )
            continue
        if record.get("attempted") and record.get("status") in attempt_coverage.DONE:
            covered_by_family[family].add(record["task"])

    missing_by_family = {}
    for family, (cid, _, _, available) in sorted(expected_selected.items()):
        missing = sorted(set(available) - covered_by_family.get(family, set()))
        if missing:
            missing_by_family[family] = {"candidate_id": cid, "missing_tasks": missing}

    current = pending == 0 and running == 0 and not superseded_records and not missing_by_family
    return {
        "current": current,
        "pending": pending,
        "running": running,
        "superseded_record_count": len(superseded_records),
        "missing_by_family": missing_by_family,
    }


def _concluded_ordinary_records(root: Path) -> list[dict[str, Any]]:
    records_dir = Path(root) / "reports/forge/records"
    if not records_dir.is_dir():
        return []
    result = []
    for path in sorted(records_dir.glob("readout-*.json")):
        doc = read_json(path)
        if doc.get("lifecycle") == "concluded" and doc.get("evidence_scope") == "current":
            result.append(doc)
    return result


def check_selected_family_readouts(root: Path, *, cuda_model: str = DEFAULT_CUDA_MODEL) -> dict[str, Any]:
    root = Path(root).resolve()
    concluded = _concluded_ordinary_records(root)
    leaderboard_texts = {}
    for view_id in ORDINARY_LEADERBOARD_VIEWS:
        lb_path = root / "reports/forge/leaderboards" / f"{view_id}.md"
        leaderboard_texts[view_id] = lb_path.read_text() if lb_path.is_file() else ""

    missing_families = []
    stale_leaderboards = []
    with planning.resolution_cache(root):
        for choice in current_family_candidates(root):
            family, cid = choice["family"], choice["candidate_id"]
            try:
                req = planning.resolve_idea(
                    root,
                    cid,
                    view_id="discriminator_stability",
                    through_tier=2,
                    freeze_source=False,
                    execution_backend="cuda",
                    cuda_model=cuda_model,
                )
            except Exception as exc:
                raise CodeChangeRequiredError(f"cannot resolve selected candidate {cid}: {exc}") from exc

            t1_assignments = [
                item for item in req["view"]["assignments"] if item["qualification_tier"] == 1
            ]
            t1_runnable = [
                item["task"]
                for item in t1_assignments
                if not req["tasks"][item["task"]].get("preflight_blockers")
            ]
            if not t1_runnable:
                continue

            rev = req["candidate_revision"]
            matching = [
                rec
                for rec in concluded
                if rec.get("candidate_id") == cid
                and rec.get("candidate_revision") == rev
                and (rec.get("study", {}).get("scope", {}).get("view") or rec.get("goal"))
                == "discriminator_stability"
                and rec.get("study", {}).get("scope", {}).get("execution_backend", "cuda") == "cuda"
            ]
            measured_statuses: dict[str, str] = {}
            unresolved_errors: dict[str, str] = {}
            for rec in matching:
                for row in rec.get("task_results", []):
                    st = row.get("gate_status") or row.get("status")
                    tid = str(row.get("task_id", ""))
                    if st in {"PASS", "FAIL", "BLOCKED"}:
                        measured_statuses[tid] = st
                        unresolved_errors.pop(tid, None)
                    elif st in {"ERROR", "INVALID", "INCOMPLETE"} and tid not in measured_statuses:
                        unresolved_errors[tid] = st
            if unresolved_errors:
                bad_tid, bad_st = sorted(unresolved_errors.items())[0]
                raise CodeChangeRequiredError(
                    f"selected candidate {cid} has {bad_st} on {bad_tid} "
                    f"in current-source readout"
                )

            unmeasured_t1 = [task for task in t1_runnable if task not in measured_statuses]
            unmeasured_t2: list[str] = []
            t1_required_all_pass = all(
                not req["tasks"][item["task"]].get("preflight_blockers")
                and measured_statuses.get(item["task"]) == "PASS"
                for item in t1_assignments
                if item.get("importance") == "required"
            )
            if not unmeasured_t1 and t1_required_all_pass:
                t2_runnable = [
                    item["task"]
                    for item in req["view"]["assignments"]
                    if item["qualification_tier"] == 2
                    and not req["tasks"][item["task"]].get("preflight_blockers")
                ]
                unmeasured_t2 = [
                    task
                    for task in t2_runnable
                    if all(
                        measured_statuses.get(dep["task"] if isinstance(dep, dict) else dep) == "PASS"
                        for dep in req["tasks"][task].get("dependencies", [])
                    )
                    and task not in measured_statuses
                ]

            if not matching or unmeasured_t1 or unmeasured_t2:
                missing_families.append(
                    {
                        "family": family,
                        "candidate_id": cid,
                        "candidate_revision": rev,
                        "unmeasured_tier1": unmeasured_t1,
                        "unmeasured_tier2": unmeasured_t2,
                    }
                )
                continue

            rev_short = rev[:12]
            needle = f"/ {rev_short} / cuda ({cuda_model}) | current |"
            for view_id, text in leaderboard_texts.items():
                matched_lines = [line for line in text.splitlines() if needle in line]
                if not matched_lines or any("| unknown |" in line for line in matched_lines):
                    stale_leaderboards.append({"family": family, "candidate_id": cid, "view": view_id})

    return {
        "current": not missing_families and not stale_leaderboards,
        "missing_families": missing_families,
        "stale_leaderboards": stale_leaderboards,
    }


def check_compiled_reports(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    mem_status = knowledge.freshness(root)
    inv_cov = history.validate_inventory(root)
    comp_path = root / "reports/forge/compilation.json"
    comp = read_json(comp_path) if comp_path.is_file() else {}
    clean_checkout_safe = not any(
        key.startswith("reports/forge/attempts/") for key in comp.get("input_hashes", {})
    )
    selected_cids = {choice["candidate_id"] for choice in current_family_candidates(root)}
    raw_pending = comp.get("pending_readout") if "pending_readout" in comp else None
    selected_pending_readout = (
        ["missing"]
        if raw_pending is None
        else [cid for cid in raw_pending if cid in selected_cids]
    )

    inv_json_path = root / "reports/forge/technique-inventory.json"
    inv_md_path = root / "reports/forge/technique-inventory.md"
    perf_path = root / "reports/forge/scaling-performance.json"
    cov_path = root / "reports/forge/attempt-coverage.json"

    perf_synced = False
    unknown_perf_rows = -1
    diagnostic_rows_present = False
    attempt_summary_synced = False

    if inv_json_path.is_file() and inv_md_path.is_file():
        inv_json = read_json(inv_json_path)
        inv_md = inv_md_path.read_text()
        if perf_path.is_file():
            perf_synced = (
                inv_json.get("scaling_performance", {}).get("manifest_sha256") == file_hash(perf_path)
            )
        if scaling_publication.START in inv_md and scaling_publication.END in inv_md:
            block = inv_md.split(scaling_publication.START, 1)[1].split(scaling_publication.END, 1)[0]
            unknown_perf_rows = sum(1 for line in block.splitlines() if "| UNKNOWN |" in line)
        diagnostic_rows_present = (
            "Selected recipe 1-for-1 baseline-35 diagnostics (atlas-particle-normal-v4); seed 0" in inv_md
            and "Selected recipe declared-suite Tier 1–3 diagnostics (atlas-particle-normal-v4); seed 0"
            in inv_md
        )
        if cov_path.is_file():
            cov_sum = read_json(cov_path).get("summary", {})
            expected_phrase = (
                f"**{cov_sum.get('attempted')}/{cov_sum.get('total')} distinct recipe/test pairs attempted "
                "(selected and research)**"
            )
            attempt_summary_synced = expected_phrase in inv_md

    current = (
        bool(mem_status.get("fresh"))
        and bool(inv_cov.get("valid"))
        and clean_checkout_safe
        and selected_pending_readout == []
        and perf_synced
        and unknown_perf_rows == 0
        and diagnostic_rows_present
        and attempt_summary_synced
    )
    return {
        "current": current,
        "memory_fresh": bool(mem_status.get("fresh")),
        "inventory_valid": bool(inv_cov.get("valid")),
        "clean_checkout_safe": clean_checkout_safe,
        "selected_pending_readout": selected_pending_readout,
        "scaling_performance_synced": perf_synced,
        "unknown_performance_rows": unknown_perf_rows,
        "selected_diagnostic_rows_present": diagnostic_rows_present,
        "attempt_summary_synced": attempt_summary_synced,
    }


def check_status(
    root: Path,
    *,
    queue_root: Path | None = None,
    cuda_model: str = DEFAULT_CUDA_MODEL,
) -> dict[str, Any]:
    root = Path(root).resolve()
    queue_root = Path(queue_root or (root / "runs/forge")).resolve()
    stages = {
        "queue": check_queue_status(root, queue_root),
        "scaling_performance": check_scaling_performance(root),
        "attempt_coverage": check_attempt_coverage(root),
        "selected_family_readouts": check_selected_family_readouts(root, cuda_model=cuda_model),
        "compiled_reports": check_compiled_reports(root),
    }
    stale = [name for name, info in stages.items() if not info.get("current")]
    return {
        "status": "CURRENT" if not stale else "NEEDS_REFRESH",
        "source_digest": catalog_source_manifest(root)["digest"],
        "stale_stages": stale,
        "stages": stages,
    }


def _conclude_queue_submissions(
    root: Path,
    queue_root: Path,
    *,
    campaign_id: str | None = None,
    json_mode: bool = False,
) -> list[str]:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return []
    state = read_json(state_path)
    concluded_records = []
    for req_id, entry in sorted(state.get("submissions", {}).items()):
        if entry.get("lifecycle") != "awaiting_readout":
            continue
        req = entry["request"]
        if campaign_id is not None and req.get("campaign_id") != campaign_id:
            continue
        if not _request_matches_worktree(root, req):
            raise CodeChangeRequiredError(
                f"submission {req_id} has stale source digest {req.get('source', {}).get('digest')}; "
                "cannot auto-conclude"
            )
        req_digest = req["source"]["digest"]
        cid = req["candidate"]["id"]
        study_id = studies.selected_study_id(req)
        view_id = req.get("view", {}).get("id", "unknown")
        is_diag = bool(diagnostic_evidence_scope(req))
        statuses = Counter()
        for job_def in req.get("jobs", []):
            job = state.get("jobs", {}).get(job_def["compatibility_key"], {})
            res = job.get("result")
            if res:
                for row in res.get("task_results", []):
                    st = row.get("gate_status", "UNKNOWN")
                    statuses[st] += 1
                    if st in {"ERROR", "INVALID", "INCOMPLETE"}:
                        raise CodeChangeRequiredError(
                            f"task {row.get('task_id')} for {cid} ({study_id or req_id}) ended with {st}: "
                            f"{row.get('reason', 'see run log')}"
                        )
        summary_counts = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())) or "no runnable tasks"
        scope_note = "Diagnostic performance/research scope only; no qualification credit." if is_diag else (
            "Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact."
        )
        _log(f"Recording readout for {cid} (study={study_id or 'none'}, {summary_counts})", json_mode=json_mode)
        rec = knowledge.readout(
            root,
            cid,
            conclusion=f"Completed {view_id} seed-0 CUDA execution for {cid} ({summary_counts}). {scope_note}",
            comparison=f"Measured unchanged declaration {cid} at source {req_digest[:12]} under protocol seed 0.",
            next_action="Retain source-bound readout and compiled Forge reports.",
            study_id=study_id,
        )
        concluded_records.append(rec["record_id"])
    return concluded_records


def _select_or_register_scaling_campaign(
    root: Path,
    queue_root: Path,
    *,
    cuda_model: str,
    json_mode: bool = False,
) -> dict[str, Any]:
    expected_pairs = {(c["family"], c["candidate_id"]) for c in current_family_candidates(root)}
    perf_dir = root / "configs/forge/performance"
    state_path = Path(queue_root) / "queue/state.json"
    q_state = read_json(state_path) if state_path.is_file() else {"submissions": {}}
    current_digest = catalog_source_manifest(root)["digest"]

    if perf_dir.is_dir():
        for path in sorted(perf_dir.glob("*.json"), reverse=True):
            try:
                manifest = scaling._load(root, path.stem)
            except ValueError:
                continue
            pairs = {(s["family"], s["candidate_id"]) for s in manifest.get("family_selections", [])}
            if (
                manifest.get("view") != "dcgan_performance"
                or manifest.get("execution_backend") != "cuda"
                or manifest.get("cuda_model") != cuda_model
                or pairs != expected_pairs
            ):
                continue
            existing_subs = [
                item
                for item in q_state.get("submissions", {}).values()
                if item["request"].get("campaign_id") == manifest["campaign"]["id"]
            ]
            if any(not _request_matches_worktree(root, item["request"]) for item in existing_subs):
                continue
            _log(f"Using registered scaling campaign {manifest['id']}", json_mode=json_mode)
            return manifest

    src12 = current_digest[:12]
    reg_id = f"dcgan-performance-cloud-{src12}"
    idx = 2
    while scaling.manifest_path(root, reg_id).exists():
        reg_id = f"dcgan-performance-cloud-{src12}-r{idx}"
        idx += 1
    _log(f"Registering new unified scaling campaign {reg_id}", json_mode=json_mode)
    scaling.register(root, reg_id, view_id="dcgan_performance", execution_backend="cuda", cuda_model=cuda_model)
    return scaling._load(root, reg_id)


def refresh_scaling_performance(
    root: Path,
    queue_root: Path,
    *,
    scaling_gpu: str,
    cuda_model: str,
    json_mode: bool = False,
) -> dict[str, Any]:
    manifest = _select_or_register_scaling_campaign(root, queue_root, cuda_model=cuda_model, json_mode=json_mode)
    reg_id = manifest["id"]
    campaign_id = manifest["campaign"]["id"]
    queue = forge_queue.Queue(queue_root, report_root=root / "reports/forge")

    enqueued = scaling.enqueue(root, queue_root, reg_id, queue=queue)
    blocked = [item for item in enqueued["submissions"] if item["status"] == "BLOCKED"]
    if blocked:
        raise CodeChangeRequiredError(f"scaling campaign {reg_id} has preflight blockers: {blocked}")

    _log(
        f"Draining scaling campaign {campaign_id} on single GPU cuda:{scaling_gpu} (workers_per_gpu=1). "
        f"Tail logs: tail -f {queue_root / 'events.jsonl'}",
        json_mode=json_mode,
    )
    forge_queue.drain(
        queue,
        devices=[str(scaling_gpu)],
        workers_per_gpu=1,
        allow_sharing=False,
        campaign=campaign_id,
    )
    _conclude_queue_submissions(root, queue_root, campaign_id=campaign_id, json_mode=json_mode)
    published = scaling_export.publish(root, queue_root, reg_id, allow_partial=False, refresh=False)
    verify = check_scaling_performance(root)
    if not verify["current"]:
        raise CodeChangeRequiredError(f"scaling performance remained incomplete after execution: {verify}")
    return published


def _ensure_frozen_checkout(
    root: Path,
    commit: str,
    checkout_root: Path,
    *,
    preserved_absent_paths: list[str] | tuple[str, ...] = (),
) -> Path:
    import shutil

    absent_list = sorted(set(preserved_absent_paths))
    dir_name = f"{commit}-{stable_hash(absent_list)[:8]}" if absent_list else commit
    target = checkout_root / dir_name
    expected_stamp = json.dumps({"commit": commit, "preserved_absent_paths": absent_list}, sort_keys=True)
    stamp = target / ".git-archive-commit"
    if stamp.is_file() and stamp.read_text().strip() == expected_stamp:
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(["git", "archive", commit], cwd=root, stdout=subprocess.PIPE)
    assert archive.stdout is not None
    subprocess.run(["tar", "-x", "-C", str(target)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise CodeChangeRequiredError(f"git archive failed for frozen commit {commit}")
    for rel in absent_list:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise CodeChangeRequiredError(f"unsafe preserved_absent_path: {rel}")
        (target / rel_path).unlink(missing_ok=True)
    stamp.write_text(f"{expected_stamp}\n")
    return target


def _prepare_combined_attempt_request(
    frozen_checkout: Path,
    candidate_id: str,
    available_tasks: list[str],
    state_dir: Path,
    family: str,
) -> Path:
    req_dir = state_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    output_path = req_dir / f"{family}.json"
    all_ordinary_views = (
        "clockfree_continuous",
        "discriminator_stability",
        "adaptation",
        "formulation_comparison",
        "host_profile_transfer",
        "quality_coverage",
    )
    view_names = [
        name
        for name in all_ordinary_views
        if (frozen_checkout / "configs/forge/views" / f"{name}.json").is_file()
    ]
    partial_specs = []
    partial_outputs = []
    for view_name in view_names:
        part_out = req_dir / f"_{family}-{view_name}.json"
        partial_outputs.append(part_out)
        partial_specs.append(
            {
                "candidate_id": candidate_id,
                "view_id": view_name,
                "tasks": available_tasks,
                "output": str(part_out),
            }
        )
    spec_path = req_dir / f"_{family}-prepare.json"
    atomic_json(spec_path, {"source_root": str(frozen_checkout), "requests": partial_specs})
    subprocess.run(
        [sys.executable, str(attempt_coverage.WORKER), "prepare", str(spec_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=600,
    )
    merged = read_json(partial_outputs[0])
    seen_job_keys = {job["compatibility_key"] for job in merged.get("jobs", [])}
    for part_out in partial_outputs[1:]:
        part = read_json(part_out)
        merged["tasks"].update(part.get("tasks", {}))
        for job in part.get("jobs", []):
            if job["compatibility_key"] not in seen_job_keys:
                merged["jobs"].append(job)
                seen_job_keys.add(job["compatibility_key"])
    for part_out in [*partial_outputs, spec_path]:
        part_out.unlink(missing_ok=True)
    atomic_json(output_path, merged)
    return output_path


def _record_to_state_job(record: dict[str, Any]) -> dict[str, Any]:
    """Invert coverage_value()'s new_attempt evidence rewrite for lossless round-tripping."""
    job = deepcopy(record)
    if job.get("origin") == "new_attempt" and "original_attempt" in job:
        job["evidence"] = job.pop("original_attempt")
    return job


def _certified_parent_from_queue(
    queue_root: Path,
    candidate_revision: str,
    task_id: str,
    compatibility_key: str,
) -> dict[str, Any] | None:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return None
    q_state = read_json(state_path)
    job = q_state.get("jobs", {}).get(compatibility_key)
    if not job or job.get("status") != "terminal" or not job.get("result"):
        return None
    res = job["result"]
    if res.get("candidate_revision") != candidate_revision:
        return None
    row = next((item for item in res.get("task_results", []) if item.get("task_id") == task_id), None)
    if not row or row.get("gate_status") != "PASS":
        return None
    ev = row.get("evidence", {})
    cp, manifest = ev.get("checkpoint"), ev.get("artifact_manifest")
    if not cp or not manifest or not Path(cp).is_file() or not Path(manifest).is_file():
        return None
    return {
        "attempt_id": res["attempt_id"],
        "result_hash": stable_hash(res),
        "result": row,
        "candidate_revision": candidate_revision,
        "compatibility_key": compatibility_key,
    }


def _is_cross_gpu_rng_mismatch(job: dict[str, Any]) -> bool:
    return (
        job.get("status") == "ERROR"
        and "trainer/named RNG checkpoint mismatch" in str(job.get("reason", ""))
    )


def _frozen_preflight_blockers(frozen_checkout: Path, candidate_id: str) -> dict[str, list[str]]:
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from experiments.forge import planning\n"
        "co = Path(sys.argv[1])\n"
        "cid = sys.argv[2]\n"
        "views = ('clockfree_continuous', 'discriminator_stability', 'adaptation', "
        "'formulation_comparison', 'host_profile_transfer', 'quality_coverage')\n"
        "out = {}\n"
        "for name in views:\n"
        "    vp = co / 'configs/forge/views' / f'{name}.json'\n"
        "    if not vp.is_file():\n"
        "        continue\n"
        "    v = json.loads(vp.read_text())\n"
        "    max_tier = max(item['qualification_tier'] for item in v['assignments'])\n"
        "    req = planning.resolve_idea(co, cid, view_id=name, through_tier=max_tier, "
        "freeze_source=False, execution_backend='cuda')\n"
        "    for tid, card in req.get('tasks', {}).items():\n"
        "        blockers = card.get('preflight_blockers') or []\n"
        "        if blockers:\n"
        "            out.setdefault(tid, []).extend(blockers)\n"
        "print(json.dumps({k: sorted(set(v)) for k, v in out.items()}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(frozen_checkout)
    proc = subprocess.run(
        [sys.executable, "-c", script, str(frozen_checkout), candidate_id],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return json.loads(proc.stdout)


def _is_expected_preflight_refusal(
    job: dict[str, Any],
    preflight_by_family_task: dict[tuple[str, str], list[str]],
) -> bool:
    if job.get("status") != "ERROR":
        return False
    reason = str(job.get("reason", ""))
    blockers = preflight_by_family_task.get((str(job.get("family", "")), str(job.get("task", ""))), [])
    for blocker in blockers:
        raw = str(blocker).strip()
        suffix = raw.split(": ", 1)[-1].strip()
        if (raw and raw in reason) or (suffix and suffix in reason):
            return True
    return False


def _pin_checkpoint_pair_gpus(
    jobs: dict[str, dict[str, Any]],
    family: str,
    pair_gpus: list[str],
) -> None:
    family_jobs = {
        job["task"]: job
        for job in jobs.values()
        if not _is_scoped_attempt(job) and job.get("family") == family
    }
    allowed_gpus = {str(g) for g in pair_gpus}
    default_gpu = str(pair_gpus[0])
    changed = True
    while changed:
        changed = False
        for _, child in sorted(family_jobs.items()):
            for dep in child.get("dependencies", []):
                if not (isinstance(dep, dict) and dep.get("kind") == "checkpoint"):
                    continue
                parent = family_jobs.get(dep["task"])
                if parent is None:
                    continue
                target_gpu: str | None = None
                if parent.get("required_gpu") is not None and str(parent["required_gpu"]) in allowed_gpus:
                    target_gpu = str(parent["required_gpu"])
                elif parent.get("gpu") is not None and str(parent["gpu"]) in allowed_gpus:
                    target_gpu = str(parent["gpu"])
                elif parent.get("certified_parent"):
                    device = str(parent["certified_parent"].get("result", {}).get("device", ""))
                    if device.startswith("cuda:"):
                        cand_gpu = device.split(":", 1)[1].strip()
                        if cand_gpu in allowed_gpus:
                            target_gpu = cand_gpu
                if target_gpu is None:
                    target_gpu = default_gpu
                if parent.get("status") == "PENDING" and parent.get("required_gpu") != target_gpu:
                    parent["required_gpu"] = target_gpu
                    changed = True
                if child.get("status") == "PENDING" and child.get("required_gpu") != target_gpu:
                    child["required_gpu"] = target_gpu
                    changed = True


def refresh_attempt_coverage(
    root: Path,
    queue_root: Path,
    attempt_state_root: Path,
    *,
    gpus: list[str],
    json_mode: bool = False,
) -> dict[str, Any]:
    if len(gpus) < 2 or len(set(gpus[:2])) != 2:
        raise CodeChangeRequiredError(
            "selected-family attempt_coverage.drain requires two distinct GPUs (--gpus 0,1)"
        )
    pair_gpus = [str(gpus[0]), str(gpus[1])]
    root = Path(root).resolve()
    attempt_state_root = Path(attempt_state_root).resolve()
    attempt_state_root.mkdir(parents=True, exist_ok=True)

    publication = read_json(root / "reports/forge/technique-inventory.json")
    existing_cov = (
        read_json(root / "reports/forge/attempt-coverage.json")
        if (root / "reports/forge/attempt-coverage.json").is_file()
        else {"records": []}
    )
    previous_state = (
        read_json(attempt_state_root / "state.json")
        if (attempt_state_root / "state.json").is_file()
        else {"jobs": {}}
    )

    selected = attempt_coverage.selected_tasks(publication)
    expected_selected: dict[str, tuple[dict[str, Any], str, list[str]]] = {}
    commit_tasks_cache: dict[str, set[str]] = {}
    for family, selection, tasks in selected:
        commit = _selected_family_source_commit(root, publication, family, selection)
        if commit not in commit_tasks_cache:
            commit_tasks_cache[commit] = _git_task_names_at_commit(root, commit)
        available = [task for task in tasks if task in commit_tasks_cache[commit]]
        expected_selected[family] = (selection, commit, available)

    jobs: dict[str, dict[str, Any]] = {}
    # 1. Preserve all scoped records and all still-current selected-family records.
    for record in existing_cov.get("records", []):
        if _is_scoped_attempt(record):
            job = _record_to_state_job(record)
            jobs[job["key"]] = job
            continue
        family = record.get("family")
        if family not in expected_selected:
            continue
        selection = expected_selected[family][0]
        if (
            record.get("candidate_id") == selection["candidate_id"]
            and record.get("candidate_revision") == selection["candidate_revision"]
            and record.get("source_digest") == selection["bindings"]["source_digest"]
        ):
            job = _record_to_state_job(record)
            jobs[job["key"]] = job

    # 2. Merge any in-progress or completed jobs from attempt_state_root/state.json that match current targets.
    for key, prev_job in previous_state.get("jobs", {}).items():
        if _is_scoped_attempt(prev_job):
            jobs.setdefault(key, deepcopy(prev_job))
            continue
        if _is_cross_gpu_rng_mismatch(prev_job):
            continue
        family = prev_job.get("family")
        if family in expected_selected:
            selection = expected_selected[family][0]
            if (
                prev_job.get("candidate_id") == selection["candidate_id"]
                and prev_job.get("candidate_revision") == selection["candidate_revision"]
                and prev_job.get("source_digest") == selection["bindings"]["source_digest"]
            ):
                if key not in jobs or jobs[key].get("status") == "PENDING":
                    jobs[key] = deepcopy(prev_job)

    retained = attempt_coverage.retained_attempts(root)
    checkout_root = Path(queue_root) / "frozen-checkouts"
    newly_scheduled_keys: set[str] = set()
    preflight_by_family_task: dict[tuple[str, str], list[str]] = {}

    for family, (selection, commit, available) in expected_selected.items():
        covered_tasks = {
            job["task"]
            for job in jobs.values()
            if not _is_scoped_attempt(job)
            and job.get("family") == family
            and job.get("status") in attempt_coverage.DONE
        }
        if set(available).issubset(covered_tasks):
            continue

        _log(
            f"Preparing frozen attempt-coverage request for {family} ({selection['candidate_id']} @ {commit[:12]})",
            json_mode=json_mode,
        )
        absent_paths = _selected_family_preserved_absent_paths(root, publication, selection)
        frozen_dir = _ensure_frozen_checkout(
            root, commit, checkout_root, preserved_absent_paths=absent_paths
        )
        for tid, blockers in _frozen_preflight_blockers(frozen_dir, selection["candidate_id"]).items():
            preflight_by_family_task[(family, tid)] = blockers
        request_path = _prepare_combined_attempt_request(
            frozen_dir, selection["candidate_id"], available, attempt_state_root, family
        )
        request = read_json(request_path)
        if (
            request["source"]["digest"] != selection["bindings"]["source_digest"]
            or request["candidate_revision"] != selection["candidate_revision"]
        ):
            raise CodeChangeRequiredError(
                f"{family}: frozen request mismatch with selected row bindings"
            )

        recorded_statuses = {
            item["task_id"]: item["status"]
            for item in selection.get("tasks", []) + selection.get("nonrequired_tasks", [])
            if item.get("status") in {"PASS", "FAIL"}
        }
        job_compat_by_task = {
            tid: job_def["compatibility_key"]
            for job_def in request.get("jobs", [])
            for tid in job_def.get("task_ids", [job_def["task_id"]])
        }

        # Identify checkpoint parents required by unrecorded children so passing parents
        # without local checkpoint files run once to materialize certified state.
        checkpoint_parents_needed: set[str] = set()
        for task_name in available:
            if task_name in covered_tasks or task_name in recorded_statuses:
                continue
            card = request["tasks"][task_name]
            for dep in card.get("dependencies", []):
                if isinstance(dep, dict) and dep.get("kind") == "checkpoint":
                    checkpoint_parents_needed.add(dep["task"])

        req_sha = file_hash(request_path)
        for task_name in available:
            card = request["tasks"][task_name]
            execution = task_execution_fingerprint(card)
            evaluation = task_evaluation_fingerprint(card)
            binding = selection["bindings"]["task_contracts"].get(task_name)
            pin = publication.get("task_contracts", {}).get(binding) if binding else None
            if pin and (execution != pin["execution_sha256"] or evaluation != pin["evaluation_sha256"]):
                raise CodeChangeRequiredError(
                    f"{family}/{task_name}: frozen task contract differs from published binding"
                )
            key = attempt_coverage.identity(
                request["source"]["digest"], request["candidate_revision"], task_name, execution, evaluation
            )
            existing = jobs.get(key)
            if existing and existing.get("status") in attempt_coverage.DONE:
                continue

            job: dict[str, Any] = {
                "key": key,
                "family": family,
                "candidate_id": selection["candidate_id"],
                "candidate_revision": request["candidate_revision"],
                "source_digest": request["source"]["digest"],
                "task": task_name,
                "execution_sha256": execution,
                "evaluation_sha256": evaluation,
                "request_path": str(request_path),
                "source_root": request["source"]["snapshot_path"],
                "request_sha256": req_sha,
                "timeout_seconds": card["resources"]["timeout_seconds"],
                "resources": card["resources"],
                "dependencies": card.get("dependencies", []),
                "status": "PENDING",
            }
            if key in retained:
                job.update(retained[key])
            elif task_name in recorded_statuses:
                rec_status = recorded_statuses[task_name]
                if rec_status == "PASS" and task_name in checkpoint_parents_needed:
                    cert = _certified_parent_from_queue(
                        queue_root,
                        request["candidate_revision"],
                        task_name,
                        job_compat_by_task.get(task_name, ""),
                    )
                    if cert is not None:
                        job.update(
                            status="PASS",
                            attempted=True,
                            origin="retained_selected_row",
                            diagnostic=False,
                            evidence="reports/forge/technique-inventory.json",
                            certified_parent=cert,
                        )
                    else:
                        newly_scheduled_keys.add(key)
                else:
                    job.update(
                        status=rec_status,
                        attempted=True,
                        origin="retained_selected_row",
                        diagnostic=False,
                        evidence="reports/forge/technique-inventory.json",
                    )
            else:
                newly_scheduled_keys.add(key)
            jobs[key] = job

        _pin_checkpoint_pair_gpus(jobs, family, pair_gpus)

    state = {
        "schema_version": 1,
        "jobs": jobs,
        "maximum_tests": 4,
        "maximum_per_gpu": 2,
        "include_research": True,
    }
    atomic_json(attempt_state_root / "state.json", state)
    pending_count = sum(1 for item in jobs.values() if item.get("status") in {"PENDING", "RUNNING"})
    _log(
        f"Draining attempt coverage ({pending_count} pending/running of {len(jobs)} records) on GPUs "
        f"{','.join(pair_gpus)}. Tail logs: tail -f {attempt_state_root / 'events.jsonl'}",
        json_mode=json_mode,
    )
    result_summary = attempt_coverage.drain(root, attempt_state_root, gpus=pair_gpus)
    final_state = read_json(attempt_state_root / "state.json")
    bad_new = [
        {
            "family": item["family"],
            "candidate_id": item["candidate_id"],
            "task": item["task"],
            "status": item["status"],
            "reason": item.get("reason", ""),
        }
        for key, item in final_state.get("jobs", {}).items()
        if key in newly_scheduled_keys
        and item.get("status") in {"ERROR", "INVALID", "INCOMPLETE", "INTERRUPTED"}
        and not _is_expected_preflight_refusal(item, preflight_by_family_task)
    ]
    if bad_new:
        raise CodeChangeRequiredError(f"newly executed diagnostic attempts failed with non-gate errors: {bad_new}")
    return result_summary


def _signature_for_request(req: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    t1_tasks = [
        item["task"]
        for item in req["view"]["assignments"]
        if item["qualification_tier"] == 1 and not req["tasks"][item["task"]].get("preflight_blockers")
    ]
    search_order = t1_tasks + [name for name in sorted(req["tasks"]) if name not in t1_tasks]
    for task_id in search_order:
        evaluation = req["tasks"][task_id].get("evaluation", {})
        thresholds = evaluation.get("thresholds") or []
        for entry in thresholds:
            if (
                isinstance(entry, list)
                and len(entry) == 3
                and isinstance(entry[0], str)
                and entry[1] in OP_COMPLEMENT
                and isinstance(entry[2], (int, float))
            ):
                metric, op, threshold = entry[0], entry[1], float(entry[2])
                base = {"task_id": task_id, "metric": metric, "threshold": threshold, "phase": "final"}
                return {**base, "op": op}, {**base, "op": OP_COMPLEMENT[op]}
    raise CodeChangeRequiredError(f"no numerical final-metric threshold found for view {req['view']['id']}")


def _ensure_ordinary_requal_study(
    root: Path,
    family: str,
    candidate_id: str,
    req: dict[str, Any],
    campaign: dict[str, Any],
    *,
    cuda_model: str,
) -> dict[str, Any]:
    src12 = req["source"]["digest"][:12]
    study_id = identifier(f"requal-{src12}-{family}-ordinary", "study")
    path = studies.study_path(root, study_id)
    if path.is_file():
        return studies.load_study(root, study_id)

    control_id = (
        "bispf-optimized-quintic-v1"
        if candidate_id != "bispf-optimized-quintic-v1"
        else "atlas-particle-normal-v4"
    )
    prediction, falsifier = _signature_for_request(req)
    study = studies.scaffold(
        study_id,
        candidate_id,
        control_id,
        "discriminator_stability",
        hypothesis=(
            f"Ordinary re-qualification of the unchanged selected {family} recipe ({candidate_id}) on "
            f"current develop source {src12} under protocol seed 0 on {cuda_model} records current-source "
            "discriminator_stability and shared Tier 1 goal-view verdicts."
        ),
        execution_backend="cuda",
    )
    study["status"] = "ready"
    study["scope"]["through_tier"] = 2
    study["scope"]["cuda_model"] = cuda_model
    study["campaign"] = deepcopy(campaign)
    study["prior_evidence"] = [
        {
            "path": CURRENT_SELECTION.as_posix(),
            "selector": [],
            "identity": {"scope": "whole_candidate_family_current", "default_adoption": False},
            "use": "motivation_only",
        }
    ]
    study["prediction"] = prediction
    study["falsifier"] = falsifier
    study["competing_explanation"] = (
        "Source updates since the prior family pin could alter unguarded numerical paths or runtime "
        "bindings; seed-0 public-API execution on fixed hardware isolates current-source behavior."
    )
    studies.validate_study(study)
    atomic_json(path, study)
    return study


def refresh_selected_family_readouts(
    root: Path,
    queue_root: Path,
    *,
    gpus: list[str],
    cuda_model: str,
    json_mode: bool = False,
) -> list[str]:
    status = check_selected_family_readouts(root, cuda_model=cuda_model)
    missing = status["missing_families"]
    if not missing:
        return []

    src12 = catalog_source_manifest(root)["digest"][:12]
    campaign_id = identifier(f"stale-family-requal-{src12}", "campaign")
    prepared: list[tuple[str, str, dict[str, Any]]] = []
    max_allowance = 0
    for item in missing:
        family, cid = item["family"], item["candidate_id"]
        base_req = planning.resolve_idea(
            root,
            cid,
            view_id="discriminator_stability",
            through_tier=2,
            freeze_source=False,
            execution_backend="cuda",
            cuda_model=cuda_model,
        )
        allowance = sum(job["budget_seconds"] for job in base_req["jobs"])
        max_allowance = max(max_allowance, allowance)
        prepared.append((family, cid, base_req))

    campaign_path = root / "configs/forge/campaigns" / f"{campaign_id}.json"
    if campaign_path.is_file():
        campaign = read_json(campaign_path)
    else:
        campaign = {
            "id": campaign_id,
            "candidate_budget_seconds": max(max_allowance, 100000),
            "budget_seconds": max(max_allowance * max(len(prepared), 1) * 2, 600000),
        }
        atomic_json(campaign_path, campaign)

    queue = forge_queue.Queue(queue_root, report_root=root / "reports/forge")
    for family, cid, base_req in prepared:
        study = _ensure_ordinary_requal_study(
            root, family, cid, base_req, campaign, cuda_model=cuda_model
        )
        request = planning.resolve_idea(root, cid, study=study["id"])
        if request["preflight_blockers"]:
            raise CodeChangeRequiredError(
                f"ordinary requalification study {study['id']} is blocked: {request['preflight_blockers']}"
            )
        request["source"]["snapshot_path"] = str(
            snapshot_source(Path(root), Path(queue_root), request["source"])
        )
        entry = queue.submit(request, deepcopy(campaign))
        _log(
            f"Submitted {study['id']} ({cid}) -> request {entry['request']['request_id']} ({entry['status']})",
            json_mode=json_mode,
        )

    drain_devices = [str(g) for g in gpus]
    _log(
        f"Draining ordinary requalification campaign {campaign_id} across GPUs {','.join(drain_devices)} "
        f"(workers_per_gpu=1). Tail logs: tail -f {queue_root / 'events.jsonl'}",
        json_mode=json_mode,
    )
    forge_queue.drain(
        queue,
        devices=drain_devices,
        workers_per_gpu=1,
        allow_sharing=False,
        campaign=campaign_id,
    )
    return _conclude_queue_submissions(root, queue_root, campaign_id=campaign_id, json_mode=json_mode)


def _retry_shared_gpu_timeouts(
    queue: forge_queue.Queue,
    queue_root: Path,
    root: Path,
    *,
    json_mode: bool = False,
) -> list[str]:
    state_path = Path(queue_root) / "queue/state.json"
    if not state_path.is_file():
        return []
    state = read_json(state_path)
    retried_keys: list[str] = []
    seen_keys: set[str] = set()
    for _, entry in sorted(state.get("submissions", {}).items()):
        if entry.get("lifecycle") not in {"ready", "running", "awaiting_readout"}:
            continue
        req = entry["request"]
        if not _request_matches_worktree(root, req):
            continue
        for job_def in req.get("jobs", []):
            key = job_def["compatibility_key"]
            if key in seen_keys:
                continue
            job = state.get("jobs", {}).get(key, {})
            res = job.get("result") or {}
            raw = res.get("raw") or {}
            if (
                job.get("status") == "terminal"
                and raw.get("attempt_status") == "timeout"
                and len(job.get("attempts", [])) == 1
            ):
                seen_keys.add(key)
                _log(
                    f"Retrying single-attempt timeout job {job_def.get('task_id')} for "
                    f"{req['candidate']['id']} on exclusive GPU slot (workers_per_gpu=1)",
                    json_mode=json_mode,
                )
                queue.retry(
                    key,
                    reason=(
                        "rerun single-attempt wall-time timeout on exclusive GPU slot "
                        "(workers_per_gpu=1) without GPU-sharing contention"
                    ),
                )
                retried_keys.append(key)
    return retried_keys


def refresh_compiled_reports(root: Path, *, json_mode: bool = False) -> dict[str, Any]:
    from reports.forge import regenerate_technique_inventory

    root = Path(root).resolve()
    _log("Refreshing technique-inventory.json/.md, family pages, problems.md, test-performance.md", json_mode=json_mode)
    pub_summary = regenerate_technique_inventory.refresh_publication(root)

    if not history.validate_inventory(root).get("valid"):
        _log("Updating Forge history catalog coverage", json_mode=json_mode)
        history.import_history(root)

    _log("Compiling live goal leaderboards and automation telemetry", json_mode=json_mode)
    knowledge.compile_memory(root, summaries_only=False)

    _log("Sealing clean-checkout-safe EXPERIMENT_MEMORY.md and compilation.json", json_mode=json_mode)
    mem_summary = knowledge.compile_memory(root, summaries_only=True)
    return {"publication": pub_summary, "memory": mem_summary}


def refresh_all(
    root: Path,
    *,
    queue_root: Path | None = None,
    attempt_state_root: Path | None = None,
    gpus: list[str] | None = None,
    scaling_gpu: str = "0",
    cuda_model: str = DEFAULT_CUDA_MODEL,
    reports_only: bool = False,
    json_mode: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    queue_root = Path(queue_root or (root / "runs/forge")).resolve()
    attempt_state_root = Path(
        attempt_state_root or (queue_root / "attempt-coverage-refresh")
    ).resolve()
    gpu_list = [str(g).strip() for g in (gpus or ["0", "1"]) if str(g).strip()]

    initial = check_status(root, queue_root=queue_root, cuda_model=cuda_model)
    if initial["status"] == "CURRENT":
        _log("All Forge readouts are already CURRENT; nothing to run.", json_mode=json_mode)
        return {"action": "NOOP", **initial}

    actions_executed: list[str] = []
    if not reports_only:
        q_info = initial["stages"]["queue"]
        if not q_info["current"]:
            queue = forge_queue.Queue(queue_root, report_root=root / "reports/forge")
            retried = _retry_shared_gpu_timeouts(queue, queue_root, root, json_mode=json_mode)
            if q_info["active_submissions"] or retried:
                _log(
                    f"Draining active queue submissions across GPUs {','.join(gpu_list)} (workers_per_gpu=1). "
                    f"Tail logs: tail -f {queue_root / 'events.jsonl'}",
                    json_mode=json_mode,
                )
                forge_queue.drain(queue, devices=gpu_list, workers_per_gpu=1, allow_sharing=False)
            _conclude_queue_submissions(root, queue_root, json_mode=json_mode)
            actions_executed.append("queue")

        if not check_scaling_performance(root)["current"]:
            refresh_scaling_performance(
                root,
                queue_root,
                scaling_gpu=str(scaling_gpu),
                cuda_model=cuda_model,
                json_mode=json_mode,
            )
            actions_executed.append("scaling_performance")

        if not check_attempt_coverage(root)["current"]:
            refresh_attempt_coverage(
                root,
                queue_root,
                attempt_state_root,
                gpus=gpu_list,
                json_mode=json_mode,
            )
            actions_executed.append("attempt_coverage")

        if not check_selected_family_readouts(root, cuda_model=cuda_model)["current"]:
            records = refresh_selected_family_readouts(
                root,
                queue_root,
                gpus=gpu_list,
                cuda_model=cuda_model,
                json_mode=json_mode,
            )
            if records:
                actions_executed.append("selected_family_readouts")

    if actions_executed or not check_compiled_reports(root)["current"]:
        refresh_compiled_reports(root, json_mode=json_mode)
        actions_executed.append("compiled_reports")

    final = check_status(root, queue_root=queue_root, cuda_model=cuda_model)
    if final["status"] != "CURRENT":
        if reports_only:
            return {"action": "REPORTS_ONLY_PARTIAL", "executed_stages": actions_executed, **final}
        raise CodeChangeRequiredError(
            f"Forge readouts still report stale stages after full refresh: {final['stale_stages']}"
        )
    return {"action": "REFRESHED", "executed_stages": actions_executed, **final}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT, help="Repository root path")
    parser.add_argument("--queue-root", type=Path, default=None, help="Forge queue root (default: runs/forge)")
    parser.add_argument(
        "--attempt-state-root",
        type=Path,
        default=None,
        help="Attempt-coverage state directory (default: runs/forge/attempt-coverage-refresh)",
    )
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU indices for parallel stages")
    parser.add_argument("--scaling-gpu", default="0", help="Single GPU index for uncontended DCGAN scaling table")
    parser.add_argument("--cuda-model", default=DEFAULT_CUDA_MODEL, help="Named CUDA hardware cohort")
    parser.add_argument("--check", action="store_true", help="Read-only check; exit 0 if current, 1 if stale, 2 if broken")
    parser.add_argument("--reports-only", action="store_true", help="Regenerate compiled reports without launching GPU runs")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON summary on stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gpus = [token.strip() for token in args.gpus.split(",") if token.strip()]
    try:
        if args.check:
            report = check_status(args.root, queue_root=args.queue_root, cuda_model=args.cuda_model)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                _log(f"Status: {report['status']} (stale stages: {report['stale_stages'] or 'none'})")
                if report["stale_stages"]:
                    print(json.dumps(report, indent=2, sort_keys=True))
            return CURRENT_EXIT if report["status"] == "CURRENT" else NEEDS_REFRESH_EXIT

        result = refresh_all(
            args.root,
            queue_root=args.queue_root,
            attempt_state_root=args.attempt_state_root,
            gpus=gpus,
            scaling_gpu=args.scaling_gpu,
            cuda_model=args.cuda_model,
            reports_only=args.reports_only,
            json_mode=args.json,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            _log(
                f"Completed ({result['action']}): executed stages={result.get('executed_stages', [])}, "
                f"status={result['status']}"
            )
        if args.reports_only and result["status"] != "CURRENT":
            return NEEDS_REFRESH_EXIT
        return CURRENT_EXIT
    except CodeChangeRequiredError as exc:
        payload = {"status": "CODE_CHANGE_REQUIRED", "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"[refresh-forge] CODE_CHANGE_REQUIRED: {exc}", file=sys.stderr, flush=True)
        return CODE_CHANGE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
