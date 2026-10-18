"""Durable publication of the scoped Atlas/E22 policy-cloud cohorts.

  python -m experiments.forge.policy_cloud_publication --state-root RUN_DIRECTORY [--refresh]

The command reads one drained ``attempt_coverage --intake policy-cloud`` state
directory and writes this cohort's committed report directory:

* ``records.json``: every cohort record of both scoring laws (clean and as
  served), in the ordinary attempt-record format. A law-independent record
  (one run for both laws) names the as-served row in ``shared_scopes``.
  ``adaptation_windows`` holds every window the target-shift grader checks,
  recomputed from each raw receipt.
* ``results.json``: the sha256 of every local receipt (attempt request,
  result, raw evidence, run log and training GIF), the clean/as-served
  training-state comparison, the recipe identity mapping and the source
  identity (with a git-checked rebased equivalent of a local frozen commit).
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
import re
import subprocess
import sys

from .contracts import atomic_json, atomic_text, file_hash, read_json, stable_hash
from .policy_coverage import (COHORT, COVERAGE, DIAGNOSTIC_CONTINUATIONS, FAMILIES, LABEL,
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
                                          for tier in sorted(TIERS)}})
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
            "changed_contract_fields": changed,
            "reason": plain_cause(record, windows=(value.get("adaptation_windows") or {}).get(_window_key(record))),
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


# Fixed native fidelity limits (benchmarks/toy100/accuracy.py LIMITS); display only.
NATIVE_LIMITS = (("mass_tv", "mass TV", 0.06), ("center_rms_sigma", "center RMS (sigma)", 0.20),
                 ("abs_cov_trace_bias", "|cov trace bias|", 0.10), ("radial_ks", "radial KS", 0.04))


def _native_holdout(holdout):
    parts = []
    for key, name, limit in NATIVE_LIMITS:
        if isinstance(holdout.get(key), (int, float)):
            value = holdout[key]
            parts.append(f"{name} {_number(value)} (needs <= {_number(limit)})" + (" FAIL" if value > limit else ""))
    text = f"precision {_number(holdout.get('precision'))}" + (", " + ", ".join(parts) if parts else
                                                              f", mass TV {_number(holdout.get('mass_tv'))}")
    if isinstance(holdout.get("accuracy_score"), (int, float)):
        text += f"; accuracy score {_number(holdout['accuracy_score'])} (mean ratio to those limits)"
    return text


WIDTH_LIMITS = tuple(item for item in NATIVE_LIMITS if item[0] in {"abs_cov_trace_bias", "radial_ks"})


def native_width_summary(records):
    """Ranges of the native holdout precision and width metrics, per law and family, read from the records."""
    summary = {}
    for record in records:
        result = (record.get("grade") or {}).get("evaluator_result") or {}
        holdout = result.get("holdout_metrics")
        if record["status"] not in {"PASS", "FAIL"} or "coverage_status" not in result or not isinstance(holdout, dict):
            continue
        for scope in ("all", record["owning_family"]):
            entry = summary.setdefault(record["measurement_law"], {}).setdefault(
                scope, {"tests": 0, "within_width_limits": 0})
            entry["tests"] += 1
            entry["within_width_limits"] += all(holdout[key] <= limit for key, _, limit in WIDTH_LIMITS)
            for key in ("precision", "abs_cov_trace_bias", "radial_ks"):
                low, high = entry.get(key, (math.inf, -math.inf))
                entry[key] = (min(low, holdout[key]), max(high, holdout[key]))
    return summary


def _span(pair, digits):
    low, high = (f"{value:.{digits}f}" for value in pair)
    return low if low == high else f"{low}-{high}"


def _width_text(records):
    """The README's native width paragraph, with every range computed from the records."""
    summary = native_width_summary(records)
    clean, served = (summary.get(law, {}) for law in ("clean", "as_served"))
    if not clean.get("all"):
        return ""
    limit = {key: _number(value) for key, _, value in WIDTH_LIMITS}
    whole = clean["all"]
    per_family = ", ".join(f"{FAMILY_NAMES[family]} {_span(clean[family]['radial_ks'], 2)}"
                           for family in FAMILIES if family in clean)
    text = (f"The native 100-mode tests show this most clearly. Their {whole['tests']} clean holdouts (both families, "
            f"7k and 14k) have precision {_span(whole['precision'], 3)} but fail the fixed width limits: |cov trace "
            f"bias| {_span(whole['abs_cov_trace_bias'], 2)} against {limit['abs_cov_trace_bias']}, radial KS "
            f"{_span(whole['radial_ks'], 2)} against {limit['radial_ks']} ({per_family}); "
            f"{whole['within_width_limits']} of {whole['tests']} are within both width limits.")
    if served.get("all"):
        other = served["all"]
        text += (f" As served, the same weights give |cov trace bias| {_span(other['abs_cov_trace_bias'], 3)} and "
                 f"radial KS {_span(other['radial_ks'], 3)}; {other['within_width_limits']} of {other['tests']} are "
                 f"within both width limits.")
    return text + " These ranges are computed from the holdout metrics in [records.json](records.json)."


def _thresholds(metrics, thresholds):
    """``metric value (needs op bound)`` for flat final metrics with declared bounds."""
    bounds = {}
    for metric, op, bound in thresholds or ():
        value = metrics.get(metric) if isinstance(metrics, dict) else None
        if metric in {"sample_count", "finite_fraction"} or not isinstance(value, (int, float)):
            continue
        bounds.setdefault(metric, []).append((op, bound))
    parts = []
    for metric, rules in bounds.items():
        value = metrics[metric]
        ok = all({"<=": value <= bound, ">=": value >= bound, "==": value == bound}.get(op, True) for op, bound in rules)
        ops = dict(rules)
        need = (f"{_number(ops['>='])} to {_number(ops['<='])}" if set(ops) == {">=", "<="}
                else ", ".join(f"{op} {_number(bound)}" for op, bound in rules))
        parts.append(f"{metric} {_number(value)} (needs {need})" + ("" if ok else " FAIL"))
    return ", ".join(parts)


ADAPTATION_RULE = ("PASS needs every live check in the stationary window, the continued hold up to the shift and "
                   "the recovery window from the deadline to the end, with the frozen control passing none of its "
                   "(at least 5) recovery-window checks")


def _steps(window):
    return f"steps {window['first_step']:,}-{window['last_step']:,}"


def plain_cause(record, thresholds=None, windows=None):
    """A short readable cause from the recorded grade; display only.

    ``thresholds`` (the task's declared ``[metric, op, bound]`` list) adds the
    final metrics against their bounds where the grade records flat metrics.
    ``windows`` (one ``adaptation_windows`` entry) adds the stationary and
    continued-hold windows that the paired-adaptation grader also checks.
    """
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
            return (f"Stable passing suffix from step {_number(convergence.get('stable_from_step'))}; all final live "
                    f"checks pass: {_checks(result['metrics'])}; {observed}.")
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
                f"{'passes' if holdout_pass else 'fails'} ({_native_holdout(holdout)}). PASS needs the coverage gate, "
                f"every final live check and the holdout.")
    if "stationary_checks" in result:
        final = _thresholds(metrics, thresholds)
        return (f"Stationary hold {result['stationary_passes']}/{result['stationary_checks']}, deadline reacquisition "
                f"{result['reacquisition_status']}, shifted hold {result['shift_hold_passes']}/{result['shift_hold_checks']} "
                f"(PASS needs every check); frozen control {result['frozen_passes']}/{result['frozen_checks']}."
                + (f" Final: {final}." if final else ""))
    if "confirmed_steps" in result:
        final = _thresholds(metrics, thresholds)
        final = f" Final: {final}." if final else ""
        if status == "PASS":
            return (f"First independently confirmed full pass at step {_number(result['first_confirmed_step'])} "
                    f"({result['passing_observations']}/{result['observations']} scheduled observations passed).{final}")
        return (f"No independently confirmed full pass ({result['passing_observations']}/{result['observations']} "
                f"observations passed).{final}")
    if metrics.get("name") == "first_convergence_then_hold_v1":
        if metrics.get("status") == "NOT_CONVERGED":
            text = (f"No qualifying window: {metrics.get('settling_failures')}/{metrics.get('settling_checks')} settling "
                    f"checks failed (needs {metrics.get('confirmation_checks')} consecutive checks with "
                    f"{metrics.get('modes')} modes and hq >= {metrics.get('min_hq')}).")
        elif metrics.get("status") == "POST_CONVERGENCE_FAIL":
            text = (f"Converged at step {_number(metrics.get('converged_step'))} ({_number(metrics.get('confirmation_checks'))} "
                    f"consecutive checks with {metrics.get('modes')} modes and hq >= {metrics.get('min_hq')}), then the "
                    f"hold failed at step {_number(metrics.get('first_hold_failure'))} after "
                    f"{_number(metrics.get('hold_checks'))}/{_number(metrics.get('hold_budget'))} hold checks (needs all).")
        else:
            text = (f"Converged at step {_number(metrics.get('converged_step'))}; held "
                    f"{metrics.get('hold_checks')}/{metrics.get('hold_budget')} checks.")
        if "extension_checks" in metrics:
            text += f" Extension {metrics['extension_passing']}/{metrics['extension_checks']} checks passed."
        return text
    if "live_deadline" in metrics:
        live, frozen = metrics["live_deadline"], metrics.get("frozen_deadline") or {}
        recovery = (f"live {live.get('passing_checks')}/{live.get('checks')} checks passed (min modes "
                    f"{_number(live.get('min_modes'))}, min hq {_number(live.get('min_hq'))}); frozen control "
                    f"{frozen.get('passing_checks')}/{frozen.get('checks')}")
        if windows:
            stationary, hold, late = windows["stationary"], windows["continued_hold"], windows["live_recovery"]
            return (f"Live run: stationary window ({_steps(stationary)}) {stationary['passing_checks']}/"
                    f"{stationary['checks']} checks passed, continued hold ({_steps(hold)}) {hold['passing_checks']}/"
                    f"{hold['checks']}, recovery window after the deadline ({_steps(late)}) {recovery} ({ADAPTATION_RULE}).")
        return f"Recovery window after the deadline: {recovery} ({ADAPTATION_RULE})."
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
    evidence = read_json(raw).get("evidence", {})
    value = evidence.get("training_state")
    if not isinstance(value, dict):
        return None
    state = {key: value.get(key) for key in ("sha256", "completed_steps")}
    checkpoint = evidence.get("provenance_checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("path") and checkpoint.get("sha256"):
        state["provenance_checkpoint"] = {"path": (directory / "provenance" / checkpoint["path"]).as_posix(),
                                          "sha256": checkpoint["sha256"]}
    return state


def receipts_and_states(jobs):
    receipts, states = [], {}
    for job in jobs:
        entry = {key: job.get(key) for key in ("family", "owning_family", "measurement_law", "parent_task", "task",
                                                "status", "attempt_id", "updates", "gpu", "seconds")}
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


DEVICE_LABEL = re.compile(r"cuda:\d+")


def _without_device_labels(value):
    """Replace every ``cuda:<index>`` label in state keys and strings with ``cuda``."""
    if isinstance(value, dict):
        return {(DEVICE_LABEL.sub("cuda", key) if isinstance(key, str) else key): _without_device_labels(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_without_device_labels(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_device_labels(item) for item in value)
    if isinstance(value, str):
        return DEVICE_LABEL.sub("cuda", value)
    return value


def device_independent_digest(state):
    """Recompute one receipt's training-state digest without the CUDA device index.

    The receipt digest (``adapters.training_state_receipt``) records the named
    stream bindings with their device, for example ``cuda:0`` versus
    ``cuda:1``, so runs on different GPUs can never match. This loads the
    run's saved final state (its provenance checkpoint, the exact object the
    receipt digested), proves that by recomputing the receipt digest, then
    applies the same exclusions after replacing the device label. Returns a
    dict with ``verified`` (file and receipt digest reproduced) and
    ``sha256`` (device-independent digest), or None without a saved state.
    """
    checkpoint = (state or {}).get("provenance_checkpoint")
    if not checkpoint or not Path(checkpoint["path"]).is_file():
        return None
    import torch
    from .adapters import training_state_receipt  # read-only use of the pinned digest
    path = Path(checkpoint["path"])
    if file_hash(path) != checkpoint["sha256"]:
        return {"verified": False, "sha256": None, "reason": "saved state differs from its receipt"}
    saved = torch.load(path, map_location="cpu", weights_only=False)
    steps = state["completed_steps"]
    reproduced = training_state_receipt(saved, steps)["sha256"]
    if reproduced != state["sha256"]:
        return {"verified": False, "sha256": None, "reason": "saved state does not reproduce the receipt digest"}
    return {"verified": True, "sha256": training_state_receipt(_without_device_labels(saved), steps)["sha256"]}


def training_state_pairs(jobs, states):
    """Compare the final training state of each clean/as-served pair.

    Evaluation is audited as pure, so a pair should train identically. The
    receipt digests are compared first (method ``receipt_digest``). The receipt
    digest includes the CUDA device index, so when two digests differ and the
    runs used different GPUs, the saved final states are compared with the
    device label normalized and the digest's own exclusions applied (method
    ``device_normalized_saved_state``). A remaining mismatch is reported, never
    graded.
    """
    index = {(job["owning_family"], job["cohort_id"], job["parent_task"]): job for job in jobs}
    pairs = []
    for family in FAMILIES:
        for parent in SERVED_PARENTS:
            clean, served = index.get((family, COHORT, parent)), index.get((family, SERVED_COHORT, parent))
            if clean is None or served is None:
                continue
            first, second = states.get(clean["key"]), states.get(served["key"])
            method, normalized = "receipt_digest", None
            if "reused_execution" in (clean.get("origin"), served.get("origin")):
                comparison, method = "no_own_training", None
            elif not first or not second or not first.get("sha256") or not second.get("sha256"):
                comparison, method = "unavailable", None
            elif first["completed_steps"] != second["completed_steps"]:
                comparison = "different_stop_steps"
            elif first["sha256"] == second["sha256"]:
                comparison = "match"
            elif clean.get("gpu") is not None and served.get("gpu") is not None and clean["gpu"] != served["gpu"]:
                method = "device_normalized_saved_state"
                normalized = [device_independent_digest(first), device_independent_digest(second)]
                comparison = ("match" if all(item and item["verified"] for item in normalized)
                              and normalized[0]["sha256"] == normalized[1]["sha256"] else "mismatch")
            else:
                comparison = "mismatch"

            def side(job, state, extra):
                value = {"status": job["status"], "attempt_id": job.get("attempt_id"), "gpu": job.get("gpu"),
                         **{key: (state or {}).get(key) for key in ("sha256", "completed_steps") if state}}
                if extra is not None:
                    value["device_normalized"] = extra
                return value
            pairs.append({"family": family, "parent_task": parent, "comparison": comparison, "method": method,
                          "clean": side(clean, first, normalized[0] if normalized else None),
                          "as_served": side(served, second, normalized[1] if normalized else None)})
    return pairs


def adaptation_windows(root, jobs, records):
    """Every window the paired-adaptation grader checks, recomputed from each raw receipt.

    ``views._adaptation`` passes target-shift recovery only when the active run
    passes every check of the stationary window (steps 1000-1200, every 50),
    the continued hold (step 1210 up to the shift, every 10) and the recovery
    window (from shift + recovery_deadline to the end, every 10), and the frozen
    control passes none of its (at least 5) recovery-window checks. The grade
    records only the two recovery windows. This recomputes all of them with the
    grader's own window function and refuses to publish unless the recovery
    windows reproduce the recorded ones.
    """
    from benchmarks.toy100.continuous_probe import _window
    by_key = {record["key"]: record for record in records}
    directories = {COHORT: COVERAGE.directory, SERVED_COHORT: SERVED.directory}
    found = {}
    for job in jobs:
        record = by_key.get(job["key"])
        metrics = ((record or {}).get("grade") or {}).get("metrics") or {}
        if "live_deadline" not in metrics or not job.get("directory"):
            continue
        raw = Path(job["directory"]) / "raw.json"
        evidence = read_json(raw).get("evidence", {}) if raw.is_file() else {}
        active, frozen = evidence.get("active"), evidence.get("frozen")
        if not isinstance(active, dict) or not isinstance(frozen, dict):
            continue
        evaluation = read_json(Path(root) / directories[record["cohort_id"]] / (record["task"] + ".json"))["evaluation"]
        shift = evaluation["shift_step"]
        spans = {"stationary": (1000, 1200, 50), "continued_hold": (1210, shift, 10),
                 "recovery": (shift + evaluation["recovery_deadline"], evaluation["steps"], 10)}

        def windows(run):
            by_step = {point["step"]: point for point in run["diagnostic"]}
            return {name: dict(_window([by_step[step] for step in range(first, last + 1, every)]),
                               first_step=first, last_step=last, every=every)
                    for name, (first, last, every) in spans.items()}
        live, control = windows(active), windows(frozen)
        for window, recorded in ((live["recovery"], metrics["live_deadline"]),
                                 (control["recovery"], metrics.get("frozen_deadline") or {})):
            if any(window[key] != recorded.get(key) for key in ("checks", "passing_checks", "min_modes", "min_hq")):
                raise ValueError(f"{record['task']}: recomputed recovery window differs from the recorded grade")
        keep = ("first_step", "last_step", "every", "checks", "passing_checks", "min_modes", "min_hq", "pass_all")
        found[_window_key(record)] = {
            "attempt_id": record.get("attempt_id"), "raw_sha256": file_hash(raw),
            **{name: {key: window[key] for key in keep} for name, window in (
                ("stationary", live["stationary"]), ("continued_hold", live["continued_hold"]),
                ("live_recovery", live["recovery"]), ("frozen_recovery", control["recovery"]))},
            "frozen_minimum_checks": 5, "rule": ADAPTATION_RULE + "."}
    return dict(sorted(found.items()))


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


def _declared_thresholds(root, records):
    """Each record's declared ``[metric, op, bound]`` list from its committed variant."""
    directories = {COHORT: COVERAGE.directory, SERVED_COHORT: SERVED.directory}
    found = {}
    for record in records:
        path = Path(root) / directories[record["cohort_id"]] / (record["task"] + ".json")
        if path.is_file():
            thresholds = read_json(path).get("evaluation", {}).get("thresholds")
            if thresholds:
                found[record["task"]] = thresholds
    return found


def _utc(seconds):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_costs(state_root, jobs):
    """Summed attempt seconds, optimizer updates and coordinator wall time."""
    attempts = [job for job in jobs if job.get("directory") and job.get("origin") != "reused_execution"]
    trained = [job for job in attempts if job["status"] in {"PASS", "FAIL"} and (job.get("updates") or 0) > 0]
    by_law = {}
    for job in attempts:
        entry = by_law.setdefault(job["measurement_law"], {"attempts": 0, "seconds": 0.0, "updates": 0})
        entry["attempts"] += 1
        entry["seconds"] = round(entry["seconds"] + float(job.get("seconds") or 0.0), 3)
        entry["updates"] += int(job.get("updates") or 0)
    gpus = {}
    for job in trained:
        gpus[str(job.get("gpu"))] = gpus.get(str(job.get("gpu")), 0) + 1
    seconds = sum(float(job.get("seconds") or 0.0) for job in attempts)
    value = {"attempts": len(attempts), "trained_attempts": len(trained), "attempt_seconds": round(seconds, 3),
             "gpu_hours": round(seconds / 3600.0, 2), "optimizer_updates": sum(int(j.get("updates") or 0) for j in trained),
             "by_law": dict(sorted(by_law.items())), "trained_attempts_by_gpu": dict(sorted(gpus.items()))}
    events = Path(state_root) / "events.jsonl"
    if events.is_file():
        rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
        starts = [row["time"] for row in rows if row.get("event") == "started"]
        ends = [row["time"] for row in rows if row.get("event") == "finished"]
        if starts and ends:
            value.update(first_start=_utc(min(starts)), last_finish=_utc(max(ends)),
                         wall_seconds=round(max(ends) - min(starts), 1))
    return value


# The frozen execution commit, which every request records unchanged, exists
# only in the local checkout that froze the run. This branch carries its
# rebased equivalent; ``source_equivalent`` checks the stated file differences
# with git before the publication states them.
SOURCE_EQUIVALENTS = {
    "90569733974b73a26d869ff9dd1a114f1184a390": {
        "commit": "de17a8556184b7675596404ca215873ca6186340",
        "branch": "task/atlas-e22-fill-20261015",
        "differing_files": {
            "experiments/forge/knowledge.py": (
                "from develop's #430: imports deepcopy and calls the cached _attempts with one argument inside "
                "compile_memory, the memory-compilation step run by `forge compile`, `forge readout` and the forge "
                "queue's post-completion refresh; the policy-cloud runner, its workers and the graders never call it"),
        },
    },
}


def _in_snapshot_scope(path):
    """Whether ``sources.source_files`` would snapshot this repository path."""
    from .sources import SOURCE_DIRS, SOURCE_SUFFIXES
    parts = path.split("/")
    if Path(path).suffix not in SOURCE_SUFFIXES or {"__pycache__", ".venv", "runs"} & set(parts):
        return False
    return parts[0] in SOURCE_DIRS or (parts[0] == "configs" and "forge" not in parts)


def source_equivalent(root, manifest):
    """The rebased equivalent of a local frozen commit, checked file by file with git.

    Every file in the frozen snapshot must be byte-identical in the
    equivalent commit except the declared ones, and the commit may add no file
    the snapshot would have included. Returns None when no equivalent is
    declared for the manifest's origin commit.
    """
    import hashlib
    entry = SOURCE_EQUIVALENTS.get(manifest.get("origin_commit"))
    if entry is None:
        return None
    commit, files = entry["commit"], manifest["files"]
    paths = sorted(files)
    try:
        batch = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"], check=True, capture_output=True,
                               input="".join(f"{commit}:{path}\n" for path in paths).encode())
        listed = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--name-only", commit], check=True,
                                capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot check the rebased source equivalent {commit[:12]} with git: {error}") from error
    output, offset, differing = batch.stdout, 0, []
    for path in paths:
        end = output.index(b"\n", offset)
        header = output[offset:end].split()
        if header[-1] == b"missing":
            differing.append(path)
            offset = end + 1
            continue
        size = int(header[2])
        content = output[end + 1:end + 1 + size]
        offset = end + 1 + size + 1
        if hashlib.sha256(content).hexdigest() != files[path]:
            differing.append(path)
    added = sorted(path for path in listed if _in_snapshot_scope(path) and path not in files)
    if differing != sorted(entry["differing_files"]) or added:
        raise ValueError(f"rebased source {commit[:12]} differs from the frozen snapshot in {differing} "
                         f"(declared {sorted(entry['differing_files'])}) and adds {added}")
    return {"origin_commit": manifest["origin_commit"], "origin_commit_location": "local checkout only (not pushed to GitHub)",
            "source_digest": manifest["digest"], "snapshot_files": len(paths), "equivalent_commit": commit,
            "equivalent_branch": entry["branch"], "differing_files": dict(entry["differing_files"]),
            "files_added_in_snapshot_scope": added,
            "check": ("git cat-file of every snapshot path at the equivalent commit, sha256 compared with the "
                      "request's source manifest; git ls-tree for files the snapshot rules would add")}


def source_identity(state_root, jobs, root=None):
    """The frozen execution source, recipes, seed and runtime shared by every attempt."""
    requests = {}
    for job in jobs:
        if job.get("request_path") and Path(job["request_path"]).is_file():
            requests.setdefault(job["request_path"], read_json(job["request_path"]))
    origins = sorted({request["source"].get("origin_commit") or "" for request in requests.values()})
    digests = sorted({request["source"]["digest"] for request in requests.values()})
    runtimes = sorted({json.dumps(job.get("runtime"), sort_keys=True) for job in jobs if job.get("runtime")})
    equivalents = []
    if root is not None:
        manifests = {request["source"]["digest"]: request["source"] for request in requests.values()}
        equivalents = [item for item in (source_equivalent(root, manifests[digest]) for digest in sorted(manifests))
                       if item is not None]
    return {"origin_commits": origins, "source_digests": digests, "rebased_equivalents": equivalents,
            "protocol_seeds": sorted({request["protocol"]["seed"] for request in requests.values()}),
            "candidate_revisions": {job["owning_family"]: job["candidate_revision"] for job in jobs},
            "requests": {Path(path).name: file_hash(path) for path in sorted(requests)},
            "runtimes": [json.loads(item) for item in runtimes]}


def publish(root, state_root):
    """Write records.json, results.json, README.md, GIF contact sheets and grouped evidence."""
    from .attempt_coverage import coverage_value
    from .policy_cloud_attempts import settings
    from .policy_cloud_media import render_cohort_media
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
    value["adaptation_windows"] = adaptation_windows(root, jobs, records)
    value["summary"] = summary(value)
    receipts, states = receipts_and_states(jobs)
    pairs = training_state_pairs(jobs, states)
    trained = {(job["owning_family"], job["measurement_law"], job["parent_task"]): job for job in jobs
               if job.get("directory") and job["status"] in {"PASS", "FAIL"} and job.get("origin") != "reused_execution"}
    order = [name for tier in sorted(TIERS) for name in TIERS[tier]]
    for path, text in certificates:
        atomic_text(path, text)
    media = render_cohort_media(root, REPORT_DIRECTORY, trained, families=FAMILIES, laws=("clean", "as_served"),
                                order=order)
    atomic_json(root / RECORDS, value)
    results = {"schema": RESULTS_SCHEMA, "qualification_input": False, "qualification_reuse": False,
               "state_root": str(state_root), "source_root": configured["source_root"],
               "records_sha256": file_hash(root / RECORDS), "recipes": deepcopy(value["recipes"]),
               "summary": deepcopy(value["summary"]), "source_identity": source_identity(state_root, jobs, root),
               "costs": run_costs(state_root, jobs), "training_state_pairs": pairs,
               "training_state_method": TRAINING_STATE_METHOD,
               "diagnostic_continuations": continuation_producers(jobs),
               "grouped_evidence": [{"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
                                    for path, _ in certificates],
               "media": media, "receipts": receipts}
    atomic_json(root / RESULTS, results)
    atomic_text(root / README, render_readme(value, results, _declared_thresholds(root, records)))
    return {"records": str(root / RECORDS), "results": str(root / RESULTS), "readme": str(root / README),
            "summary": value["summary"],
            "training_state_pairs": {name: sum(p["comparison"] == name for p in pairs)
                                     for name in sorted({p["comparison"] for p in pairs})},
            "training_state_methods": {name: sum(p["method"] == name for p in pairs)
                                       for name in sorted({p["method"] for p in pairs if p["method"]})}}


TRAINING_STATE_METHOD = {
    "receipt_digest": ("the two receipts' training_state.sha256 (adapters.training_state_receipt) compared directly"),
    "device_normalized_saved_state": (
        "the receipt digest includes the CUDA device index (named-stream bindings such as cuda:0 versus cuda:1), so "
        "for a pair whose runs used different GPUs each run's saved final state (provenance/provenance-state.pt) is "
        "loaded, verified by reproducing its receipt digest byte for byte, every cuda:<index> label in keys and "
        "strings is replaced with cuda, and the same digest (same exclusions: isolated evaluation streams, global "
        "CPU/CUDA RNG snapshots, birth/death eval_seconds) is recomputed and compared"),
}


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
    order = {"clean": 0, "as_served": 1}
    return sorted(rows, key=lambda row: (row["family"], row["task"], order.get(row["law"], 2)))


def _notes(family, parent, continuations, records=()):
    notes = []
    retried = [LAW_NAMES[record["measurement_law"]] for record in records
               if record is not None and record.get("retry_history")]
    if retried:
        notes.append("Re-run after an infrastructure interruption (" + ", ".join(dict.fromkeys(retried)) +
                     "); every earlier record is kept in results.json")
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


def _metric(record, name):
    """One final metric from a recorded grade (curve checks or flat metrics), else None."""
    if record is None:
        return None
    grade = record.get("grade") or {}
    result = grade.get("evaluator_result") or {}
    checks = result.get("metrics") if isinstance(result.get("metrics"), list) else []
    for check in checks:
        if check.get("metric") == name:
            return check.get("value")
    metrics = grade.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(name), (int, float)):
        return metrics[name]
    holdout = result.get("holdout_metrics")
    if isinstance(holdout, dict) and isinstance(holdout.get(name), (int, float)):
        return holdout[name]
    return None


def _status_pair(index, family, parent):
    clean = index.get((family, COHORT, parent))
    served = index.get((family, SERVED_COHORT, parent)) if parent in SERVED_PARENTS else clean
    return clean, served


def _verdicts(index, parent):
    parts = []
    for family in FAMILIES:
        clean, served = _status_pair(index, family, parent)
        if clean is None:
            continue
        if parent in LAW_INDEPENDENT:
            parts.append(f"{FAMILY_NAMES[family]} {clean['status']} (shared)")
        else:
            parts.append(f"{FAMILY_NAMES[family]} {clean['status']} clean / {served['status'] if served else '—'} as served")
    return "; ".join(parts)


FAMILY_NAMES = {"atlas": "Atlas", "e22": "E22"}

# Earlier Atlas/E22 measurements of the same questions (static history; the
# "now" column is computed from this publication's records).
EARLIER = (
    ("img_stripes2", "[Unchanged public policy, 600 updates, selected serving state](../actual-missing-diagnostics-20261014/README.md)",
     "Atlas PASS; E22 PASS", None),
    ("img_bars4", "same image diagnostic", "Atlas FAIL; E22 FAIL", None),
    ("img_blobs4", "same image diagnostic", "Atlas FAIL; E22 FAIL", None),
    ("img_intensity2", "same image diagnostic", "Atlas FAIL; E22 FAIL (4 consecutive passing observations, 5 needed)", None),
    ("two_pole", "[Selected-cloud two-pole, 80 updates](../actual-missing-diagnostics-20261014/README.md#what-the-atlase22-policy-attempts-uncovered)",
     "Atlas INCOMPLETE, E22 INCOMPLETE (receipt lacked recipe-owned prior-policy metadata); mean_abs 0.002586, needs >= 0.3",
     "mean_abs"),
    ("gaussian1d_smoke", "[Tier 1 selected-cloud Gaussian acquisition](../atlas-compatible-coverage-20261013/README.md) "
     "(a different gate: a stable terminal passing suffix)",
     "first ERROR/INCOMPLETE (checkpoint NaN export; final CDF KS 0.029), then FAIL after the "
     "[export repair](../ATTEMPT_COVERAGE.md#checkpoint-recovery), Atlas and E22", "cdf_ks"),
    ("ring16_acquisition", "[Tier 1 selected-cloud Ring16 acquisition](../atlas-compatible-coverage-20261013/README.md)",
     "first ERROR/INCOMPLETE (checkpoint NaN export), then FAIL after the export repair; logged component covariance "
     "error Atlas 1.812, E22 1.139 (needs <= 0.85)", "component_covariance_error"),
    ("clockfree_audit_measurement_v1", "[Tier 1 selected-cloud clock audit](../atlas-compatible-coverage-20261013/README.md); "
     "[E22 policy clock audit](../actual-missing-diagnostics-20261014/README.md)",
     "Atlas FAIL; E22 FAIL (step label and restart changed the state)", None),
)


def _now(index, parent, metric):
    text = _verdicts(index, parent)
    if metric:
        values = []
        for family in FAMILIES:
            clean, served = _status_pair(index, family, parent)
            got = [(law, _metric(record, metric)) for law, record in (("clean", clean), ("as served", served))
                   if record is not None and _metric(record, metric) is not None]
            if parent in LAW_INDEPENDENT:
                got = got[:1]
            if got:
                values.append(FAMILY_NAMES[family] + " " + ", ".join(
                    (f"{law} " if parent not in LAW_INDEPENDENT else "") + _number(value) for law, value in got))
        if values:
            text += f". {metric}: " + "; ".join(values)
    return text


def _pair_counts(pairs):
    counts = {name: sum(p["comparison"] == name for p in pairs) for name in
              ("match", "mismatch", "different_stop_steps", "no_own_training", "unavailable")}
    counts["normalized"] = sum(p["comparison"] == "match" and p["method"] == "device_normalized_saved_state"
                               for p in pairs)
    return counts


def _differing(index):
    rows = []
    for family in FAMILIES:
        for parent in SERVED_PARENTS:
            clean, served = _status_pair(index, family, parent)
            if clean and served and clean["status"] != served["status"]:
                rows.append((family, parent, clean["status"], served["status"]))
    return rows


def _window_key(record):
    return f"{record['owning_family']}/{record['task']}"


def _cause(record, thresholds, windows):
    return plain_cause(record, thresholds.get(record["task"]), windows.get(_window_key(record)))


def render_readme(value, results, thresholds=None):
    thresholds = thresholds or {}
    windows = value.get("adaptation_windows") or {}
    records = value["records"]
    index = {(r["owning_family"], r["cohort_id"], r["parent_task"]): r for r in records}
    statuses = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    pairs = results["training_state_pairs"]
    counts = _pair_counts(pairs)
    comparable = counts["match"] + counts["mismatch"]
    differing = _differing(index)
    lines = ["# Atlas and E22 on the leaderboard tests: scoped policy-cloud runs, clean and as served", "",
             "Generated by `python -m experiments.forge.policy_cloud_publication` from one drained run. "
             "Do not edit by hand.", "",
             "## What was asked", "",
             "Fill the Atlas and E22 rows of the Forge family leaderboard with actual PASS/FAIL results wherever a test "
             "is compatible with those formulations and a scoped run makes scientific sense, by actually running the "
             "experiments. Score every compatible test under two laws, *clean* and *as served*, and report them as two "
             "separate rows per family.", "",
             "Before this run the selected Atlas and E22 rows showed 0/152 view requirements (143 incompatible, 9 "
             "blocked) over 35 distinct tests. No numerical failure caused those labels: one blanket guard refused "
             "every attempt before training because the original tests declare a clean/live mixture-prior "
             "measurement and no policy-control evidence or served-sampling contract, and the checkpoint and gate "
             "dependents were then blocked as prerequisites.", "",
             "## Result", "",
             f"{len(records)} records in [records.json](records.json): " + ", ".join(
                 f"{statuses[name]} {name}" for name in sorted(statuses)) + ". Each row below covers all 35 tests; "
             "the as-served row repeats the 12 law-independent verdicts (two-pole, both clock audits and the 9 "
             "incompatible hosts), which come from one run each. BLOCKED tests stay in the denominator.", "",
             *_scores_table(value), ""]
    if differing:
        lines += ["The two laws give different verdicts on " + ", ".join(
            f"{FAMILY_NAMES[f]} {p} ({c} clean, {s} as served)" for f, p, c, s in differing) + ". "
            + ("Every clean/as-served pair trained identically (see [training states](#training-states)), so each of "
               "these differences is caused by the scoring law alone." if comparable and not counts["mismatch"]
               and not counts["different_stop_steps"] else
               "See [training states](#training-states) for which pairs trained identically."), ""]
    lines += ["## The scoped law, and why it is the closest analog", "",
              "The other families take the clean/live tests: the trained generator's own samples, scored with live "
              "weights, from the task's declared prior, without added noise. Atlas and E22 cannot be measured that way "
              "without changing them. Their recipes own an equal-mass table of latent rows with row controls "
              "(birth/death, copying and rebalancing rows), a policy that selects which weights are served, and a "
              "learned output noise. These scoped variants change only what has to change, and say so:", "",
              "- an explicit equal-mass particle-cloud prior (sigma 0) with a stated reason, instead of the task's "
              "mixture prior;",
              "- the complete family recipe unchanged (no overrides, no tuning), schedule-free, protocol seed 0;",
              "- the task's architecture, data law, seen batch sequence, capacity, update budget, timeout, evaluation "
              "cadence and numerical gates unchanged;",
              "- scoring with the state-selected served weights and the latent policy exactly as GANTrainer serves "
              "it, drawn from isolated evaluation streams; every sample-measuring receipt attests its measured draws.", "",
              "This is the closest runnable analog of the clean/live tests: the same question, gates and budget, "
              "measured on what these recipes actually serve. It is a separate diagnostic cohort. It changes no "
              "qualification numerator, no other family's row and no archived verdict, and the original live/MoG "
              "tests receive no credit. Lower-tier failures did not stop higher-tier attempts in this "
              "user-authorized lane. Checkpoint dependents (Gaussian stability and the three 14k natives) continue "
              "their own same-law saved state, also after the parent's gate failed (marked below). Ring extension "
              "is regraded only from its own passing uninterrupted ring_hold run and stays BLOCKED (prerequisite) "
              "after a failed hold.", "",
              "[Records](records.json) · [Receipt hashes, costs and comparisons](results.json) · View declarations: "
              "[clean](../../../configs/forge/views/policy_cloud_coverage.json), "
              "[as served](../../../configs/forge/views/policy_cloud_coverage_as_served.json) · Variant declarations: "
              "[clean](../../../configs/forge/task-variants/policy_cloud_coverage_v1/), "
              "[as served](../../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/)", "",
              "## Why the two laws can differ", "",
              "Atlas and E22 learn a set of points (latent rows mapped through the generator) plus one learned output "
              "noise level. GANTrainer serves a sample as a point plus that noise; here the learned noise is about "
              "0.029, close to the data's per-mode width of 0.03. The *clean* law removes the noise before scoring, "
              "so each mode becomes a tight cluster: positions and masses can be right while the width is far too "
              "narrow. The *as-served* law keeps the noise, so the same trained model is scored with the width it "
              "actually produces. Recipe, training, gates, budget, data, cadence and seed are identical; only the "
              "evaluation draw differs.", "",
              _width_text(records), "",
              "Noise cannot help where a test needs a different shape per component (Ring16 and the "
              "unequal-mass/unequal-width vectors), and it does not change tests that fail on modes or on holding a "
              "learned state.", ""]
    for family in FAMILIES:
        if not any(r["owning_family"] == family for r in records):
            continue
        lines += [f'<a name="{family}"></a>', "", f"## {family}", "",
                  "| Tier | Test | Clean (learned noise removed) | As served (learned noise kept) | Notes | Key metrics against bounds |",
                  "| ---: | --- | --- | --- | --- | --- |"]
        for tier in sorted(TIERS):
            for parent in TIERS[tier]:
                clean, served = _status_pair(index, family, parent)
                if clean is None:
                    continue
                if clean.get("block_kind") == "incompatible":
                    cause = "Not run (incompatible); the reason is under [Incompatible tests](#incompatible-tests)."
                elif parent in LAW_INDEPENDENT:
                    cause = _cause(clean, thresholds, windows)
                else:
                    cause = (f"Clean: {_cause(clean, thresholds, windows)}<br>As served: "
                             f"{_cause(served, thresholds, windows)}" if served else _cause(clean, thresholds, windows))
                notes = _notes(family, parent, results["diagnostic_continuations"],
                               (clean, served) if parent in SERVED_PARENTS else (clean,))
                lines.append(f"| {tier} | {parent} | {_cell(_result(clean))} | {_cell(_result(served))} | "
                             f"{_cell(notes)} | {_cell(cause)} |")
        lines.append("")
    lines += ["## Incompatible tests", "",
              "These nine tests are refused by an ownership boundary before any host setup, for both families and "
              "both laws (one refusal fills both rows). Atlas and E22 control an independent table of "
              "interchangeable sample rows; each of these tests needs rows bound to something else.", "",
              "| Tier | Test | Why Atlas/E22 cannot run it faithfully |", "| ---: | --- | --- |"]
    for tier in sorted(TIERS):
        for parent in TIERS[tier]:
            if parent in COVERAGE.blocked_hosts:
                reason = COVERAGE.blocked_hosts[parent]
                lines.append(f"| {tier} | {parent} | {_cell(reason[:1].upper() + reason[1:])}. |")
    lines += ["", "## The selected rows stay unchanged", "",
              "The bold selected Atlas and E22 rows at the top of the leaderboard still show their recorded "
              "qualification (0/152 with 143 incompatible and 9 blocked). The generator labels a selected-row cell a "
              "*diagnostic attempt* only when the selected attempt registry (`reports/forge/attempt-coverage.json`) "
              "holds an attempt of that same test contract. These runs are a different, explicitly scoped contract "
              "(particle-cloud prior, state-selected served weights, a declared measurement law); the original "
              "clean/live contract really is incompatible, and the cohort declaration gives its parent tests no "
              "credit, as the `tier1_policy_selected_cloud_v1` precedent did. The registry also belongs to the live "
              "attempt-fill service, which rewrites it from its own state, so cohort records never enter it. The "
              "results therefore appear as two separate rows per family under *Research recipes and separate "
              "diagnostics*, and per test on the family pages.", ""]
    media = results.get("media")
    if media:
        lines += ["## Actual training GIFs", "",
                  "Each tile is the attempt's own `training.gif`, written during its run from its saved scoring "
                  "observations; the contact sheets only resample and tile those frames (zero optimizer updates, "
                  "zero sampling draws). Shared law-independent attempts appear on the clean sheet. Provenance, "
                  "source GIF hashes and frame indices are in [results.json](results.json) under `media`.", ""]
        for sheet in media["sheets"]:
            name = Path(sheet["path"]).name
            lines.append(f"- [{FAMILY_NAMES[sheet['family']]}, {LAW_NAMES[sheet['law']]}](gifs/{name}) "
                         f"({len(sheet['tiles'])} tests)")
        lines.append("")
    costs = results.get("costs") or {}
    if costs:
        lines += ["## Costs", ""]
        text = (f"{costs['attempts']} attempts ({costs['trained_attempts']} trained) used "
                f"{costs['attempt_seconds']:,.0f} attempt-seconds, about **{costs['gpu_hours']} GPU-hours** summed "
                f"over attempts on shared GPUs, for {_number(costs['optimizer_updates'])} optimizer updates.")
        if costs.get("wall_seconds"):
            text += (f" The coordinator ran from {costs['first_start']} to {costs['last_finish']} "
                     f"({_number(round(costs['wall_seconds'] / 3600.0, 2))} h wall) with at most two tests per GPU.")
        lines += [text, "", "| Law | Attempts | Attempt-seconds | Optimizer updates |", "| --- | ---: | ---: | ---: |"]
        for law, entry in costs["by_law"].items():
            lines.append(f"| {LAW_NAMES.get(law, law)} | {entry['attempts']} | {entry['seconds']:,.0f} | "
                         f"{_number(entry['updates'])} |")
        lines += ["", "Nothing was retried, re-run or tuned unless a note above says so; incompatible refusals cost "
                  "about one second each.", ""]
    identity = results.get("source_identity") or {}
    if identity:
        runtime = identity["runtimes"][0] if len(identity["runtimes"]) == 1 else None
        lines += ["## Source identity", "",
                  f"- Frozen execution source: commit `{', '.join(identity['origin_commits'])}`, source digest "
                  f"`{', '.join(d[:16] for d in identity['source_digests'])}`; every request names the same source "
                  "and records that commit unchanged."]
        for item in identity.get("rebased_equivalents") or ():
            lines += [f"- Commit `{item['origin_commit'][:12]}` exists only in the local checkout that froze the run and "
                      "was not pushed to GitHub. The durable identity is the source digest "
                      f"`{item['source_digest']}` over the frozen snapshot's {item['snapshot_files']:,} files, and "
                      f"its rebased equivalent on `{item['equivalent_branch']}` is commit `{item['equivalent_commit']}`.",
                      f"- Of the {item['snapshot_files']:,} snapshot files, `{item['equivalent_commit'][:12]}` differs only "
                      "in " + "; ".join(f"`{path}` ({why})" for path, why in item["differing_files"].items())
                      + ". The publication checks this with git: every other snapshot file is byte-identical at that "
                      "commit, and it adds no file the snapshot rules would include."]
        lines += [
                  "- Candidate revisions: " + ", ".join(f"{FAMILY_NAMES[f]} `{r[:16]}`"
                                                        for f, r in sorted(identity["candidate_revisions"].items())) + ".",
                  f"- Protocol seed: {', '.join(str(s) for s in identity['protocol_seeds'])}; recipe overrides: none.",
                  (f"- Runtime: Python {runtime['python']}, torch {runtime['packages'].get('torch')}, numpy "
                   f"{runtime['packages'].get('numpy')}." if runtime else "- Runtime: see results.json."),
                  "", "The cohort ran the complete family recipe from its frozen source. Its resolved recipe hash differs "
                  "from the selected row's only because Recipe gained fields after that row was recorded; each added "
                  "field holds its registered default (`particlegan/recipe_compat.py`). Restricted to the selected "
                  "row's fields, the cohort recipe hashes exactly to the selected recipe.", ""]
    else:
        lines += ["## Recipe identity", ""]
    lines += ["| Family | Selected recipe (row) | Cohort recipe | Restricted to selected fields | Added fields at defaults | Why the additions are unused |",
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
    lines += ["", '<a name="training-states"></a>', "", "## Clean and as-served training states", "",
              "Evaluation is audited as pure, so a clean/as-served pair should end in the same training state. The "
              "receipt digest covers every model, optimizer, policy controller, named training stream and counter, "
              "and excludes only the isolated evaluation streams, process-global RNG snapshots and wall-clock "
              "diagnostics. It also records the CUDA device index inside the named-stream bindings (`cuda:0` versus "
              "`cuda:1`), so two runs on different GPUs can never have equal receipt digests. For such pairs the "
              "publication loads each run's saved final state, proves it is the digested state by reproducing the "
              "receipt digest exactly, replaces the device label and recomputes the same digest. The method used is "
              "recorded per pair.", "",
              f"Pairs: **{counts['match']} match** ({counts['match'] - counts['normalized']} by receipt digest on the "
              f"same GPU, {counts['normalized']} after device-label normalization across GPUs), {counts['mismatch']} "
              f"mismatch, {counts['different_stop_steps']} stopped at different updates, {counts['no_own_training']} "
              f"without own training (regraded), {counts['unavailable']} unavailable (no training receipt).", ""]
    if comparable and not counts["mismatch"] and not counts["different_stop_steps"]:
        lines += [f"All {counts['match']} trained pairs trained identically: every model, optimizer, policy "
                  "controller, training stream and counter is equal. No verdict difference in this report comes from "
                  "GPU nondeterminism; each comes from the scoring law.", ""]
    lines += ["| Family | Test | Clean digest @ updates (GPU) | As-served digest @ updates (GPU) | Comparison | Method |",
              "| --- | --- | --- | --- | --- | --- |"]
    for pair in pairs:
        def digest(side):
            if not side.get("sha256"):
                return side["status"]
            text = f"`{side['sha256'][:12]}` @ {_number(side.get('completed_steps'))} (GPU {side.get('gpu')})"
            normalized = side.get("device_normalized")
            if normalized:
                text += (f"; device-normalized `{normalized['sha256'][:12]}`" if normalized.get("sha256")
                         else f"; {normalized.get('reason', 'no saved state')}")
            return text
        method = {"receipt_digest": "receipt digest", "device_normalized_saved_state": "saved state, device label "
                  "normalized"}.get(pair["method"], "—")
        lines.append(f"| {pair['family']} | {pair['parent_task']} | {digest(pair['clean'])} | {digest(pair['as_served'])} | "
                     f"{pair['comparison'].replace('_', ' ')} | {method} |")
    lines += ["", "## Comparison with earlier Atlas/E22 diagnostics", "",
              "Earlier runs measured some of these questions under other contracts. Their receipts and verdicts are "
              "unchanged; this table only places them next to the new results.", "",
              "| Test | Earlier measurement | Earlier result | Now |", "| --- | --- | --- | --- |"]
    for parent, where, earlier, metric in EARLIER:
        if any(_status_pair(index, family, parent)[0] for family in FAMILIES):
            lines.append(f"| {parent} | {where} | {_cell(earlier)} | {_cell(_now(index, parent, metric))} |")
    lines += ["", "Ring hold, ring extension, target-shift recovery, the natives, mode hold and the vector transfers "
              "had no earlier Atlas/E22 measurement under Forge; these are their first results.", "",
              "## Recommended next experiments", "",
              "1. Make `adapters.training_state_receipt` device-independent (drop the CUDA index from the named-stream "
              "bindings before hashing), so the clean/as-served pairing check needs no saved-state fallback. It is a "
              "source-pinned file, so this needs a new frozen source; it changes no verdict.",
              "2. Decide which scoring law the Atlas/E22 cards declare. The as-served law measures what GANTrainer "
              "serves; the clean law isolates where the rows sit. Keep both rows until the owners decide, and treat "
              "clean-only native failures (high precision with width far too narrow) as a property of the "
              "measurement, not of training.",
              "3. Image Bars4 and Blobs4 find 0 modes under both laws and both families at the declared 600 updates, "
              "as in the earlier unchanged-policy diagnostic. Use the saved observations and policy-control receipts "
              "to see whether the rows ever leave their initialization before changing anything.",
              "4. Per-component shape (Ring16, unequal mass, unequal width) fails under both laws; one scalar output "
              "noise cannot supply a different covariance per component. A per-row or per-mode width study would "
              "test whether learned per-row width fixes it.",
              "5. Holding and recovering a learned state fails under both laws with identical training: Atlas "
              "ring_hold converges and then loses the hold, E22 never acquires the ring window, Gaussian stability "
              "and target-shift recovery miss their windows. These are training-dynamics failures; a one-seed, "
              "fixed-budget ablation of the late-phase row controls (birth/death and handover) on ring_hold is the "
              "cheapest informative next run.",
              "6. The clock audits fail by construction (absolute step stamps in policy diagnostics, KA2's call-800 "
              "switch, the critic-guard release count). Passing needs clock-free versions of those controls.",
              "7. The nine incompatible tests need a routed (conditional) row contract before any Atlas/E22 binding "
              "is faithful.", "",
              "## Reproduce", "",
              f"Local state directory (raw receipts, logs, checkpoints, in-run GIFs; outside Git): `{results['state_root']}`. "
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
