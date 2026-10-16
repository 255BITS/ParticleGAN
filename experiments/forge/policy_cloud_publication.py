"""Durable publication of the scoped Atlas/E22 policy-cloud cohorts.

  python -m experiments.forge.policy_cloud_publication --state-root RUN_DIRECTORY [--refresh]

The command reads one drained ``attempt_coverage --intake policy-cloud`` state
directory and writes this cohort's committed report directory:

* ``records.json``: every cohort record of both scoring laws (clean and as
  served), in the ordinary attempt-record format. A law-independent record
  (one run for both laws) names the as-served row in ``shared_scopes``.
* ``results.json``: the sha256 of every local receipt (attempt request,
  result, raw evidence, run log and training GIF), the clean/as-served
  training-state comparison and the recipe identity mapping.
* ``README.md``: generated per-test tables with both laws side by side.
* ``grouped-evidence/``: verified ring-extension regrading certificates.

The live attempt-fill service rewrites ``reports/forge/attempt-coverage.json``
and ``ATTEMPT_COVERAGE.md`` from its own state on every publish, so cohort
records never enter those files. The leaderboard rows (``research_tiers``)
and the Atlas/E22 family pages read ``records.json`` instead. Raw receipts,
logs, checkpoints and GIFs stay in the state directory. Output depends only
on the state directory and the repository, so a rerun writes identical bytes.
``--refresh`` then regenerates the leaderboard and family pages.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from .contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from .policy_coverage import (BLOCKED_HOSTS, COHORT, COVERAGE, DIAGNOSTIC_CONTINUATIONS, FAMILIES, LABEL,
                              LAW_INDEPENDENT, PARENTS, SERVED, SERVED_COHORT, SERVED_LABEL, SERVED_PARENTS,
                              SERVED_VIEW, TIERS, VIEW, reporting_tier, variant_id)

REPORT_DIRECTORY = Path("reports/forge/atlas-e22-policy-cloud-20261015")
RECORDS = REPORT_DIRECTORY / "records.json"
RESULTS = REPORT_DIRECTORY / "results.json"
README = REPORT_DIRECTORY / "README.md"
GROUPED = REPORT_DIRECTORY / "grouped-evidence"
SCHEMA = "forge_policy_cloud_records_v1"
RESULTS_SCHEMA = "forge_policy_cloud_results_v1"
VIEWS = (VIEW, SERVED_VIEW)
COHORTS = {COHORT: COVERAGE, SERVED_COHORT: SERVED}
LAW_NAMES = {"clean": "clean", "as_served": "as served", "shared": "shared"}
FINAL = frozenset({"PASS", "FAIL", "INCOMPLETE", "INVALID", "ERROR", "TIMEOUT", "BLOCKED", "INTERRUPTED"})
RECEIPT_FILES = ("request.json", "result.json", "raw.json", "run.log", "training.gif")


# --- Validation and projection (read by research_tiers and family_reports) --

def _scope(family, source, revision, cohort):
    from .policy_cloud_attempts import cohort_scope
    return cohort_scope(family, source, revision, cohort)


def validate_records(value):
    """Check one complete, unchanged-formulation cohort publication."""
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("qualification_input") is not False or value.get("qualification_reuse") is not False
            or value.get("formulations_modified") is not False):
        raise ValueError("policy-cloud records must remain an unchanged-formulation diagnostic")
    records, recipes = value.get("records"), value.get("recipes")
    if not isinstance(records, list) or not records or not isinstance(recipes, dict):
        raise ValueError("policy-cloud publication needs its records and recipe identity mapping")
    seen, members, identities = set(), defaultdict(lambda: defaultdict(set)), defaultdict(set)
    for record in records:
        cohort = COHORTS.get(record.get("cohort_id"))
        family, parent = record.get("owning_family"), record.get("parent_task")
        if cohort is None or family not in FAMILIES or parent not in cohort.parents:
            raise ValueError("policy-cloud record names another cohort, family or test")
        law = "as_served" if cohort is SERVED else "shared" if parent in LAW_INDEPENDENT else "clean"
        if (record.get("measurement_law") != law or record.get("task") != variant_id(parent, cohort)
                or record.get("reporting_tier") != reporting_tier(parent) or record.get("diagnostic") is not True
                or record.get("status") not in FINAL
                or record.get("label") != (SERVED_LABEL if cohort is SERVED else LABEL)):
            raise ValueError(f"{record.get('task')}: policy-cloud record has another law, tier, label or status")
        identity = record["source_digest"], record["candidate_revision"]
        if record.get("cohort_scope") != _scope(family, *identity, cohort.id):
            raise ValueError(f"{record['task']}: cohort scope differs from its family, source, recipe and law")
        shared = [_scope(family, *identity, SERVED_COHORT)] if law == "shared" else None
        if record.get("shared_scopes") != shared:
            raise ValueError(f"{record['task']}: only a law-independent record is shared with the as-served row")
        if (family, cohort.id, parent) in seen:
            raise ValueError(f"{record['task']}: repeated policy-cloud record")
        seen.add((family, cohort.id, parent))
        members[family][cohort.id].add(parent)
        identities[family].add(identity)
    for family in members:
        if members[family][COHORT] != set(PARENTS) or members[family][SERVED_COHORT] != set(SERVED_PARENTS):
            raise ValueError(f"{family}: a publication covers every test of both laws")
        mapping = recipes.get(family) or {}
        if (len(identities[family]) != 1 or mapping.get("verified") is not True
                or (mapping.get("cohort_source_digest"), mapping.get("cohort_candidate_revision"))
                != next(iter(identities[family]))):
            raise ValueError(f"{family}: records need one frozen source and a verified recipe mapping")
    return value


def load_records(root, load=None):
    """The validated committed publication, or None before the first one."""
    path = Path(root) / RECORDS
    if not path.is_file():
        return None
    return validate_records(load(RECORDS) if load is not None else read_json(path))


def cohort_rows(value):
    """One row per family and law (clean first); shared records enter both."""
    rows = []
    for family in FAMILIES:
        mine = [r for r in value["records"] if r["owning_family"] == family]
        if not mine:
            continue
        for cohort in (COVERAGE, SERVED):
            own = [r for r in mine if r["cohort_id"] == cohort.id]
            shared = ([r for r in mine if r["cohort_id"] == COHORT and r["measurement_law"] == "shared"]
                      if cohort is SERVED else [])
            records = {r["parent_task"]: r for r in own + shared}
            if len(records) != len(own) + len(shared):
                raise ValueError("policy cohort rows cannot repeat a test")
            rows.append({"family": family, "law": "as_served" if cohort is SERVED else "clean",
                         "label": SERVED_LABEL if cohort is SERVED else LABEL, "scope": own[0]["cohort_scope"],
                         "records": records, "shared": len(shared),
                         "requirements": {str(tier): {name for name in records if reporting_tier(name) == tier}
                                          for tier in (1, 2, 3)}})
    return rows


def row_label(row):
    if row["shared"]:
        return f"{row['label']} · {row['shared']} law-independent verdicts shared with the clean row"
    return row["label"]


def anchor(family):
    return README.as_posix() + "#" + family


def view_counts(records, required):
    """Family-page counts that equal the leaderboard row's counts."""
    from collections import Counter
    from .research_tiers import attempt_count
    value = attempt_count({}, records, required, fill=True)
    statuses = Counter(records[name]["status"] if name in records else "UNKNOWN" for name in required)
    return {**value, "incomplete": any(status not in {"PASS", "FAIL"} for status in statuses.elements()),
            "counts": dict(sorted(statuses.items()))}


def family_overlay(value, family_id, row, task_fingerprint):
    """Cohort cells for one family's current selected row, keyed by task id.

    Records attach only to the selected row whose recipe the mapping verified
    (same candidate, candidate revision and source); other runtime cohorts of
    the family keep their own (unrun) cells.
    """
    if value is None or family_id not in value["recipes"]:
        return None
    mapping = value["recipes"][family_id]
    if (row.get("candidate_id"), row.get("candidate_revision"), row.get("bindings", {}).get("source_digest")) != (
            mapping["selected_candidate_id"], mapping["selected_candidate_revision"], mapping["selected_source_digest"]):
        return None
    cells = {}
    for record in value["records"]:
        if record["owning_family"] != family_id:
            continue
        execution, evaluation = task_fingerprint(record["task"])
        changed = [label for label, now, then in (
            ("execution (host, recipe binding, prior, initialization or budget)", execution, record["execution_sha256"]),
            ("evaluation (gates or sampling law)", evaluation, record["evaluation_sha256"])) if now != then]
        law = LAW_NAMES[record["measurement_law"]]
        cells[record["task"]] = {
            "status": record["status"], "execution_recorded": record["status"] in {"PASS", "FAIL"},
            "attempt_id": record.get("attempt_id"), "current_contract": "CHANGED" if changed else "matches",
            "changed_contract_fields": changed, "reason": plain_cause(record),
            "coverage_reason": (f"Separately scoped policy-cloud diagnostic ({law} law"
                                + ("; one run supplies both laws" if record["measurement_law"] == "shared" else "")
                                + f"); frozen source {record['source_digest'][:12]}, candidate revision "
                                f"{record['candidate_revision'][:12]}. "
                                + ("Test definition changed since this run: " + "; ".join(changed) + "."
                                   if changed else "Test definition matches the recorded run.")),
            "run_attempt": {key: deepcopy(record[key]) for key in ("status", "attempted", "origin", "reason", "evidence",
                                                                    "block_kind", "attempt_id", "measurement_law")
                            if key in record}}
    return {"cells": cells, "publication": {
        "readme": anchor(family_id), "records": RECORDS.as_posix(), "source_digest": mapping["cohort_source_digest"],
        "candidate_revision": mapping["cohort_candidate_revision"], "shared": len(LAW_INDEPENDENT)}}


# --- Plain-language causes ---------------------------------------------------

def _number(value):
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3g}" if math.isfinite(value) else str(value)
    return str(value)


def _checks(checks):
    return ", ".join(f"{check['metric']} {_number(check['value'])} (needs {check['op']} {_number(check['threshold'])})"
                     for check in checks)


def plain_cause(record):
    """A short readable cause from the recorded grade; display only."""
    status, reason = record["status"], str(record.get("reason") or "").strip()
    if status == "BLOCKED":
        if record.get("block_kind") == "incompatible":
            prefix = record["task"] + ": "
            text = reason.removeprefix(prefix).replace("; " + prefix, "; ") or "ownership boundary"
            return "Not run (incompatible): " + text
        return "Not run (prerequisite): " + (reason or "a required producer is unavailable")
    if status == "INTERRUPTED":
        return "Worker stopped without any receipt (infrastructure interruption); no outcome was measured."
    if status == "TIMEOUT":
        return f"The task's fixed wall-time limit ({_number(record.get('timeout_seconds'))} s) ran out; no verdict."
    if status in {"ERROR", "INCOMPLETE", "INVALID"}:
        return f"{status.capitalize()}: {reason[:300]}"
    grade = record.get("grade") or {}
    result, metrics = grade.get("evaluator_result") or {}, grade.get("metrics") or {}
    if isinstance(result.get("metrics"), list) and isinstance(result.get("convergence"), dict):
        convergence = result["convergence"]
        observed = (f"{_number(convergence.get('passing_observations'))}/{_number(convergence.get('observations'))} "
                    "observations passed")
        failing = [check for check in result["metrics"] if check.get("status") != "PASS"]
        if status == "PASS":
            return f"Stable passing suffix from step {_number(convergence.get('stable_from_step'))}; all final live checks pass ({observed})."
        if failing:
            return f"Final live checks fail: {_checks(failing)}; {observed}."
        return (f"Final live checks pass but no stable passing suffix ({_number(convergence.get('passing_suffix'))} "
                f"consecutive, needs {_number(convergence.get('minimum_stable_checks'))}); {observed}.")
    if "coverage_status" in result:
        terminal = result.get("terminal_checks") or []
        holdout = result.get("holdout_metrics") or {}
        holdout_pass = bool(holdout.get("frozen_pass")) and bool(holdout.get("accuracy_pass"))
        return (f"Coverage gate {result['coverage_status']}; final live accuracy checks "
                f"{sum(bool(c.get('passed')) for c in terminal)}/{len(terminal)} passed; independent holdout "
                f"{'passes' if holdout_pass else 'fails'} (precision {_number(holdout.get('precision'))}, "
                f"mass TV {_number(holdout.get('mass_tv'))}).")
    if "stationary_checks" in result:
        return (f"Stationary hold {result['stationary_passes']}/{result['stationary_checks']}, deadline reacquisition "
                f"{result['reacquisition_status']}, shifted hold {result['shift_hold_passes']}/{result['shift_hold_checks']} "
                f"(PASS needs every check); frozen control {result['frozen_passes']}/{result['frozen_checks']}.")
    if "confirmed_steps" in result:
        if status == "PASS":
            return (f"First independently confirmed full pass at step {_number(result['first_confirmed_step'])} "
                    f"({result['passing_observations']}/{result['observations']} scheduled observations passed).")
        return f"No independently confirmed full pass ({result['passing_observations']}/{result['observations']} observations passed)."
    if metrics.get("name") == "first_convergence_then_hold_v1":
        if metrics.get("status") == "NOT_CONVERGED":
            text = (f"No qualifying window: {metrics.get('settling_failures')}/{metrics.get('settling_checks')} settling "
                    f"checks failed (needs {metrics.get('confirmation_checks')} consecutive checks with "
                    f"{metrics.get('modes')} modes and hq >= {metrics.get('min_hq')}).")
        elif metrics.get("status") == "POST_CONVERGENCE_FAIL":
            text = (f"Converged at step {_number(metrics.get('converged_step'))}, then the hold failed at step "
                    f"{_number(metrics.get('first_hold_failure'))}.")
        else:
            text = (f"Converged at step {_number(metrics.get('converged_step'))}; held "
                    f"{metrics.get('hold_checks')}/{metrics.get('hold_budget')} checks.")
        if "extension_checks" in metrics:
            text += f" Extension {metrics['extension_passing']}/{metrics['extension_checks']} checks passed."
        return text
    if "live_deadline" in metrics:
        live, frozen = metrics["live_deadline"], metrics.get("frozen_deadline") or {}
        return (f"Recovery window after the shift: live {live.get('passing_checks')}/{live.get('checks')} checks passed "
                f"(min modes {_number(live.get('min_modes'))}, min hq {_number(live.get('min_hq'))}); frozen control "
                f"{frozen.get('passing_checks')}/{frozen.get('checks')}.")
    reasons = grade.get("reasons") or ([reason] if reason else [])
    return "; ".join(str(item) for item in reasons) or status


# --- Publication -------------------------------------------------------------

def _hashes(directory, job):
    files = {name: file_hash(directory / name) for name in RECEIPT_FILES if (directory / name).is_file()}
    expected = {"result.json": job.get("result_sha256"), "run.log": job.get("log_sha256"),
                "training.gif": job.get("training_gif_sha256")}
    if (directory / "result.json").is_file():
        expected["raw.json"] = read_json(directory / "result.json").get("raw_sha256")
    for name, digest in expected.items():
        if digest is not None and files.get(name) != digest:
            raise ValueError(f"{job['family']}/{job['task']}: local {name} differs from its recorded receipt")
    return files


def _training_state(directory):
    raw = directory / "raw.json"
    if not raw.is_file():
        return None
    value = read_json(raw).get("evidence", {}).get("training_state")
    return {key: value.get(key) for key in ("sha256", "completed_steps")} if isinstance(value, dict) else None


def receipts_and_states(jobs):
    receipts, states = [], {}
    for job in jobs:
        entry = {key: job.get(key) for key in ("family", "owning_family", "measurement_law", "parent_task", "task",
                                                "status", "attempt_id", "updates")}
        if job.get("origin") == "reused_execution":
            entry["reused_execution"] = deepcopy(job.get("reused_execution"))
        elif job.get("directory"):
            directory = Path(job["directory"])
            entry.update(directory=str(directory), files=_hashes(directory, job))
            states[job["key"]] = _training_state(directory)
        if job.get("retry_history"):
            entry["retry_history"] = deepcopy(job["retry_history"])
        receipts.append(entry)
    return receipts, states


def training_state_pairs(jobs, states):
    """Compare the final training state of each clean/as-served pair.

    Evaluation is audited as pure, so a pair should train identically. GPU
    kernels can be nondeterministic: a mismatch is reported, never graded.
    """
    index = {(job["owning_family"], job["cohort_id"], job["parent_task"]): job for job in jobs}
    pairs = []
    for family in FAMILIES:
        for parent in SERVED_PARENTS:
            clean, served = index.get((family, COHORT, parent)), index.get((family, SERVED_COHORT, parent))
            if clean is None or served is None:
                continue
            first, second = states.get(clean["key"]), states.get(served["key"])
            if "reused_execution" in (clean.get("origin"), served.get("origin")):
                comparison = "no_own_training"
            elif not first or not second or not first.get("sha256") or not second.get("sha256"):
                comparison = "unavailable"
            elif first["completed_steps"] != second["completed_steps"]:
                comparison = "different_stop_steps"
            else:
                comparison = "match" if first["sha256"] == second["sha256"] else "mismatch"
            pairs.append({"family": family, "parent_task": parent, "comparison": comparison,
                          "clean": {"status": clean["status"], "attempt_id": clean.get("attempt_id"), **(first or {})},
                          "as_served": {"status": served["status"], "attempt_id": served.get("attempt_id"),
                                        **(second or {})}})
    return pairs


def _same(value, default):
    normalize = (lambda item: tuple(item) if isinstance(item, list) else item)
    return normalize(value) == normalize(default)


def recipe_identity(root, state_root, records):
    """Map each family's frozen cohort recipe onto its selected recipe row.

    The cohort ran the complete family Recipe at a newer source. Restricted to
    the fields the selected row recorded, it must hash to that row's recipe;
    every field added to Recipe since then must hold its registered default.
    """
    from particlegan.recipe_compat import SEARCH_RECIPE_DEFAULTS
    from .attempt_coverage import selected_tasks
    publication = read_json(Path(root) / "reports/forge/technique-inventory.json")
    selected = {family: row for family, row, _ in selected_tasks(publication)}
    mapping = {}
    for family in FAMILIES:
        identities = {(r["source_digest"], r["candidate_revision"]) for r in records if r["owning_family"] == family}
        if len(identities) != 1:
            raise ValueError(f"{family}: cohort records must share one frozen source and recipe")
        source, revision = identities.pop()
        requests = [read_json(Path(state_root) / "requests" / (prefix + family + ".json"))
                    for prefix in ("policy-cloud-", "policy-cloud-as-served-")]
        if any((request["source"]["digest"], request["candidate_revision"]) != (source, revision)
               or request["protocol"]["seed"] != 0 or request["candidate"].get("recipe_overrides") != {}
               for request in requests):
            raise ValueError(f"{family}: frozen requests differ from their records, seed 0 or the unchanged recipe")
        recipe = requests[0]["candidate"]["resolved_recipe"]
        if stable_hash(requests[1]["candidate"]["resolved_recipe"]) != stable_hash(recipe):
            raise ValueError(f"{family}: both laws must run one resolved recipe")
        row = selected[family]
        recorded = row["bindings"]["recipe_sha256"]
        pointer = publication["evidence_sources"][row["publication_key"]]
        contract = read_json(Path(root) / pointer["snapshot"])["recipe_contracts"][recorded]
        if stable_hash(contract) != recorded:
            raise ValueError(f"{family}: selected recipe contract differs from its recorded hash")
        added = {key: recipe[key] for key in sorted(set(recipe) - set(contract))}
        removed = sorted(set(contract) - set(recipe))
        changed = sorted(key for key in set(contract) & set(recipe) if contract[key] != recipe[key])
        restricted = stable_hash({key: recipe[key] for key in contract if key in recipe})
        nondefault = sorted(key for key, value in added.items()
                            if key not in SEARCH_RECIPE_DEFAULTS or not _same(value, SEARCH_RECIPE_DEFAULTS[key]))
        if removed or changed or nondefault or restricted != recorded:
            raise ValueError(f"{family}: cohort recipe differs from the selected recipe beyond registered defaults: "
                             f"removed {removed}, changed {changed}, non-default additions {nondefault}")
        mapping[family] = {
            "selected_candidate_id": row["candidate_id"], "selected_candidate_revision": row["candidate_revision"],
            "selected_source_digest": row["bindings"]["source_digest"], "selected_recipe_sha256": recorded,
            "selected_recipe_contract": pointer["snapshot"], "cohort_candidate_revision": revision,
            "cohort_source_digest": source, "cohort_source_origin_commit": requests[0]["source"].get("origin_commit"),
            "cohort_recipe_sha256": stable_hash(recipe), "restricted_to_selected_fields_sha256": restricted,
            "added_fields_at_registered_defaults": added, "verified": True,
            "inert_because": {key: recipe.get(key) for key in ("prior_regularizer", "prior_reg", "prior_proximal_strength")}}
    return mapping


def summary(value):
    from .family_reports import score
    from .research_tiers import attempt_count
    result = {}
    for row in cohort_rows(value):
        counts = {tier: score(attempt_count({}, row["records"], sorted(names), fill=True))
                  for tier, names in row["requirements"].items()}
        counts["total"] = score(attempt_count({}, row["records"], sorted(row["records"]), fill=True))
        result.setdefault(row["family"], {})[row["law"]] = {"label": row_label(row), "scores": counts,
                                                           "shared_verdicts": row["shared"]}
    return result


def publish(root, state_root):
    """Write records.json, results.json, README.md and grouped evidence."""
    from .attempt_coverage import coverage_value
    from .policy_cloud_attempts import settings
    root, state_root = Path(root).resolve(), Path(state_root).resolve()
    state = read_json(state_root / "state.json")
    if state.get("intake") != "policy-cloud":
        raise ValueError("publication reads a policy-cloud intake state directory")
    configured = settings(state_root)
    jobs = sorted(state["jobs"].values(), key=lambda job: (job["family"], job["task"]))
    unfinished = [job["task"] for job in jobs if job["status"] not in FINAL]
    if unfinished:
        raise ValueError(f"publication requires a drained run; {len(unfinished)} jobs are pending or running")
    records = coverage_value(state, evidence=README.as_posix())["records"]
    certificates = []
    for record in records:
        if record.get("origin") != "reused_execution":
            continue
        source = Path(record["evidence"])
        if (source.parent != state_root / "grouped-evidence" or not source.is_file()
                or file_hash(source) != record["result_sha256"]):
            raise ValueError(f"{record['task']}: grouped evidence differs from its recorded certificate")
        record["evidence"] = (GROUPED / source.name).as_posix()
        certificates.append((root / GROUPED / source.name, source.read_text()))
    value = {"schema": SCHEMA, "qualification_input": False, "qualification_reuse": False,
             "formulations_modified": False, "publication": "cohort_report_directory",
             "families": list(FAMILIES), "laws": deepcopy(configured["laws"]),
             "shared_tests": sorted(LAW_INDEPENDENT), "diagnostic_continuations": sorted(DIAGNOSTIC_CONTINUATIONS),
             "recipes": recipe_identity(root, state_root, records), "records": records}
    validate_records(value)
    value["summary"] = summary(value)
    receipts, states = receipts_and_states(jobs)
    pairs = training_state_pairs(jobs, states)
    for path, text in certificates:
        atomic_text(path, text)
    atomic_json(root / RECORDS, value)
    results = {"schema": RESULTS_SCHEMA, "qualification_input": False, "qualification_reuse": False,
               "state_root": str(state_root), "source_root": configured["source_root"],
               "records_sha256": file_hash(root / RECORDS), "recipes": deepcopy(value["recipes"]),
               "summary": deepcopy(value["summary"]), "training_state_pairs": pairs,
               "diagnostic_continuations": continuation_producers(jobs),
               "grouped_evidence": [{"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
                                    for path, _ in certificates],
               "receipts": receipts}
    atomic_json(root / RESULTS, results)
    atomic_text(root / README, render_readme(value, results))
    return {"records": str(root / RECORDS), "results": str(root / RESULTS), "readme": str(root / README),
            "summary": value["summary"],
            "training_state_pairs": {name: sum(p["comparison"] == name for p in pairs)
                                     for name in sorted({p["comparison"] for p in pairs})}}


# --- README ------------------------------------------------------------------

def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def _result(record):
    if record is None:
        return "—"
    text = record["status"] + (f" ({record['block_kind']})" if record["status"] == "BLOCKED" and record.get("block_kind") else "")
    if record["measurement_law"] == "shared":
        return text + " · shared run"
    if record.get("origin") == "reused_execution":
        return text + " · regraded, no new training"
    updates = record.get("updates")
    return text + (f" · {_number(updates)} updates" if isinstance(updates, int) and updates else "")


def continuation_producers(jobs):
    """Each diagnostic continuation's own same-law producer and its gate."""
    rows = []
    for job in jobs:
        if not job.get("diagnostic_continuation"):
            continue
        for dependency in job["dependencies"]:
            if not isinstance(dependency, dict) or dependency.get("kind") != "checkpoint":
                continue
            producer = next((other for other in jobs if other["family"] == job["family"]
                             and other["task"] == dependency["task"]), None)
            rows.append({"family": job["owning_family"], "law": job["measurement_law"], "task": job["parent_task"],
                         "producer": producer["parent_task"] if producer else dependency["task"],
                         "producer_status": producer["status"] if producer else "unavailable",
                         "status": job["status"]})
    return sorted(rows, key=lambda row: (row["family"], row["task"], row["law"]))


def _notes(family, parent, continuations):
    notes = []
    if parent in LAW_INDEPENDENT:
        notes.append("Law-independent: " + LAW_INDEPENDENT[parent] + "; one run supplies both columns")
    if parent in DIAGNOSTIC_CONTINUATIONS:
        parts = [f"{LAW_NAMES[row['law']]} parent {row['producer']} {row['producer_status']}"
                 for row in continuations if row["family"] == family and row["task"] == parent]
        failed = any(row["producer_status"] == "FAIL" for row in continuations
                     if row["family"] == family and row["task"] == parent)
        notes.append("Diagnostic continuation of its own same-law saved state"
                     + (" after a FAIL parent gate" if failed else "") + (": " + ", ".join(parts) if parts else ""))
    if parent == "ring_extension":
        notes.append("Regraded only from its own passing uninterrupted ring_hold run; after a failed hold it stays "
                     "BLOCKED (prerequisite)")
    return "; ".join(notes)


def _scores_table(value):
    lines = ["| Family | Law | Tier 1 tests | Tier 2 tests | Tier 3 tests | Total distinct tests |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for family in FAMILIES:
        for law in ("clean", "as_served"):
            entry = value["summary"].get(family, {}).get(law)
            if entry is None:
                continue
            scores = entry["scores"]
            label = LAW_NAMES[law] + (f" ({entry['shared_verdicts']} shared with clean)" if entry["shared_verdicts"] else "")
            lines.append(f"| [{family}](#{family}) | {label} | {scores['1']} | {scores['2']} | {scores['3']} | "
                         f"{scores['total']} |")
    return lines


def render_readme(value, results):
    records = value["records"]
    index = {(r["owning_family"], r["cohort_id"], r["parent_task"]): r for r in records}
    lines = ["# Atlas/E22 policy-cloud scoped tests: clean and as-served scoring laws", "",
             "Generated by `python -m experiments.forge.policy_cloud_publication` from one drained run. "
             "Do not edit by hand.", "",
             "Atlas and E22 refused every selected-view test before training because ordinary test declarations "
             "certify a clean/live mixture-prior measurement, not a policy-controlled served sampler. These "
             "separately scoped variants run each compatible test with an explicit equal-mass particle cloud "
             "(sigma 0, with a stated reason), the complete family recipe unchanged (no overrides or tuning) and "
             "protocol seed 0. Each test keeps its architecture, data, update budget, timeout, evaluation cadence "
             "and numerical gates. Scores use the state-selected served model, the actual served latent policy and "
             "isolated evaluation streams. This is a diagnostic cohort: it changes no qualification, no other "
             "family's row and no archived verdict, and the original live/MoG tests receive no credit.", "",
             "**Two scoring laws.** *Clean* removes the learned output noise from the measured samples. *As served* "
             "keeps it exactly as GANTrainer serves it (the learned sigma with its policy floor). Atlas and E22 "
             "supply per-mode width through that noise, so the clean law alone can fail width gates for that reason "
             "only. Each law trains in its own run; the recipe, training, gates, budget, data, cadence and seed are "
             "identical, and only the evaluation draw differs. Tests whose measurement draws no output samples "
             "(the two-pole parameter measurement, the clock audits) and tests refused by an ownership blocker are "
             "*law-independent*: they run once and the same verdict appears in both columns, marked *shared*.", "",
             "**Diagnostic continuation.** Checkpoint dependents (Gaussian stability and the three 14k natives) "
             "continue their own same-law saved state, also after the parent's quality gate failed; this is "
             "user-authorized and marked below. Ring extension is different: it is regraded only from its own "
             "passing uninterrupted ring_hold run and stays BLOCKED (prerequisite) after a failed hold, matching "
             "the precedent.", "",
             "[Records](records.json) · [Receipt hashes and comparisons](results.json) · View declarations: "
             "[clean](../../../configs/forge/views/policy_cloud_coverage.json), "
             "[as served](../../../configs/forge/views/policy_cloud_coverage_as_served.json) · Variant declarations: "
             "[clean](../../../configs/forge/task-variants/policy_cloud_coverage_v1/), "
             "[as served](../../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/)", "",
             "## Scores", "",
             "Passes / distinct tests in each row. Every row covers all 35 tests; the as-served row shows the 12 "
             "shared law-independent verdicts too. BLOCKED (incompatible) tests stay in the denominator.", "",
             *_scores_table(value), ""]
    for family in FAMILIES:
        if not any(r["owning_family"] == family for r in records):
            continue
        lines += [f'<a name="{family}"></a>', "", f"## {family}", "",
                  "| Tier | Test | Clean (learned noise removed) | As served (learned noise kept) | Notes | Cause |",
                  "| ---: | --- | --- | --- | --- | --- |"]
        for tier in (1, 2, 3):
            for parent in TIERS[tier]:
                clean = index.get((family, COHORT, parent))
                served = index.get((family, SERVED_COHORT, parent)) if parent in SERVED_PARENTS else clean
                if clean is None:
                    continue
                if parent in LAW_INDEPENDENT:
                    cause = plain_cause(clean)
                else:
                    cause = f"Clean: {plain_cause(clean)}<br>As served: {plain_cause(served)}" if served else plain_cause(clean)
                lines.append(f"| {tier} | {parent} | {_cell(_result(clean))} | {_cell(_result(served))} | "
                             f"{_cell(_notes(family, parent, results['diagnostic_continuations']))} | {_cell(cause)} |")
        lines.append("")
    lines += ["## Recipe identity", "",
              "The cohort ran the complete family recipe from its frozen source. Its resolved recipe hash differs from "
              "the selected row's only because Recipe gained fields after that row was recorded; each added field "
              "holds its registered default (`particlegan/recipe_compat.py`). Restricted to the selected row's "
              "fields, the cohort recipe hashes exactly to the selected recipe.", "",
              "| Family | Selected recipe (row) | Cohort recipe | Restricted to selected fields | Added fields at defaults | Why the additions are unused |",
              "| --- | --- | --- | --- | --- | --- |"]
    for family, mapping in value["recipes"].items():
        added = ", ".join(f"{key}={json.dumps(item)}" for key, item in mapping["added_fields_at_registered_defaults"].items()) or "none"
        inert = mapping["inert_because"]
        lines.append(f"| {family} | `{mapping['selected_recipe_sha256'][:12]}` (revision `{mapping['selected_candidate_revision'][:12]}`, "
                     f"source `{mapping['selected_source_digest'][:12]}`) | `{mapping['cohort_recipe_sha256'][:12]}` (revision "
                     f"`{mapping['cohort_candidate_revision'][:12]}`, source `{mapping['cohort_source_digest'][:12]}`"
                     + (f", commit `{mapping['cohort_source_origin_commit'][:12]}`" if mapping.get("cohort_source_origin_commit") else "")
                     + f") | `{mapping['restricted_to_selected_fields_sha256'][:12]}` (equal) | {_cell(added)} | "
                     f"prior_regularizer {inert['prior_regularizer']} with prior_reg {_number(inert['prior_reg'])} (Coulomb and "
                     f"SIGReg settings apply only to those regularizers); prior_proximal_strength "
                     f"{_number(inert['prior_proximal_strength'])} (no proximal restore) |")
    pairs = results["training_state_pairs"]
    counts = {name: sum(p["comparison"] == name for p in pairs) for name in
              ("match", "mismatch", "different_stop_steps", "no_own_training", "unavailable")}
    lines += ["", "## Clean and as-served training states", "",
              "Evaluation is audited as pure, so a clean/as-served pair should end in the same training state. The "
              "digest covers every model, optimizer, policy controller, named training stream and counter, and "
              "excludes only the isolated evaluation streams, process-global RNG snapshots and wall-clock "
              "diagnostics. GPU kernels can be nondeterministic, so a mismatch is reported here and never changes a "
              "verdict. A ring_hold pair can stop at different updates because each law's own gate decides; such "
              "pairs are not comparable.", "",
              f"Pairs: **{counts['match']} match**, {counts['mismatch']} mismatch, {counts['different_stop_steps']} "
              f"stopped at different updates, {counts['no_own_training']} without own training (regraded), "
              f"{counts['unavailable']} unavailable (no completed receipt).", "",
              "| Family | Test | Clean digest @ updates | As-served digest @ updates | Comparison |",
              "| --- | --- | --- | --- | --- |"]
    for pair in pairs:
        def digest(side):
            return (f"`{side['sha256'][:12]}` @ {_number(side.get('completed_steps'))}" if side.get("sha256")
                    else side["status"])
        lines.append(f"| {pair['family']} | {pair['parent_task']} | {digest(pair['clean'])} | {digest(pair['as_served'])} | "
                     f"{pair['comparison'].replace('_', ' ')} |")
    lines += ["", "## Reproduce", "",
              f"Local state directory (raw receipts, logs, checkpoints, GIFs; outside Git): `{results['state_root']}`. "
              f"Frozen execution source: `{results['source_root']}`.", "",
              "```sh",
              "python -m experiments.forge.attempt_coverage plan --intake policy-cloud --source-root FROZEN --state-root STATE",
              "python -m experiments.forge.attempt_coverage run --intake policy-cloud --source-root FROZEN --state-root STATE --gpus 0,1",
              "python -m experiments.forge.attempt_coverage status --state-root STATE",
              "python -m experiments.forge.policy_cloud_publication --state-root STATE --refresh",
              "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true",
                        help="regenerate the leaderboard and family pages after writing the cohort report")
    args = parser.parse_args(argv)
    value = publish(args.root, args.state_root)
    if args.refresh:
        subprocess.run([sys.executable, "reports/forge/regenerate_technique_inventory.py", "--refresh-publication"],
                       cwd=args.root.resolve(), check=True, timeout=1800)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
