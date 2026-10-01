"""Retrospective gate calibration from pinned receipts; never launches training.

The initial and proposed profiles share a reference that excludes both smoke
predicates. Imported PASS stamps describe their archived protocols, not current
Forge qualification. Cross-cohort totals are descriptive and cannot adopt a gate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import gzip
import json
import math

from .contracts import atomic_json, atomic_text, file_hash, identifier, read_json, stable_hash
from .history import Source, LRFREE_REVISION, LRFREE_ROOT

SCHEMA_VERSION = 1
FOLLOWUP_REVISION = "bb9036e33557b23b30293811dd95829a04112388"
PACKAGE_KEYS = {
    "dv12-ams-rc3-c22": "ca2feb43713a0eb8000fbdcf6a640d83d47ce275bbb3228a34ed271756f72506",
    "st-10-c22": "7effbb0f0083c0e532581a4cc8763e60d106ab4944930337c128aef7014825e4",
    "row-em-renew-all22": "1fc12d29e2c5964f91c9523006e569cba743eba2e3e9851a68afa063c08cd324",
    "pr215_exact_host_init": "eb988dd9f211df8af9c3ade4ba90edc438eec6f71251545b467e6dc91ade184b",
    "pr215_qr_adapter": "0a5aae9479ab741dd0f9f0ebde86137fbde43109e33e128d4839c7ad6a1d9686",
    "pr217_qr_native_adapter": "387a1265838966581b7e1858593ce9f292f376ba76d04f5f2c64e0666bf814c4",
}


def _seconds(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 else None)


def _task_cost(root, task, sources):
    """Enrich cost only through an exactly hashed snapshot, never a label join."""
    cost = task.get("cost", {})
    seconds = next((_seconds(cost[k]) for k in ("wall_seconds", "seconds")
                    if _seconds(cost.get(k)) is not None), None)
    receipt = task.get("raw_result", {}).get("snapshot_receipt")
    if seconds is not None or not receipt:
        return {"wall_seconds": seconds, "source": "normalized_card" if seconds is not None else None,
                "flops": None, "flops_status": "unavailable"}
    try:
        rev = receipt["revision"]
        if rev not in sources:
            sources[rev] = Source(root, rev, ("reports",))
        source = sources[rev]
        raw = source.read(receipt["path"])
        if sha256(raw).hexdigest() != receipt["sha256"]:
            raise ValueError("snapshot hash mismatch")
        payload = gzip.decompress(raw) if receipt["path"].endswith(".gz") else raw
        artifact_hash = task.get("raw_result", {}).get("artifact_sha256")
        if artifact_hash and sha256(payload).hexdigest() != artifact_hash:
            raise ValueError("decoded artifact hash mismatch")
        saved = json.loads(payload)
        seconds = _seconds(saved.get("seconds"))
        return {"wall_seconds": seconds, "source": receipt,
                "flops": None, "flops_status": "unavailable"}
    except (ValueError, KeyError, OSError) as error:
        return {"wall_seconds": None, "source": receipt, "gap": str(error),
                "flops": None, "flops_status": "unavailable"}


def _decision(tasks, required):
    statuses = {name: tasks.get(name, {}).get("gate_status", "NOT_RUN") for name in required}
    if "FAIL" in statuses.values():
        decision = "FAIL"
    elif statuses and all(s == "PASS" for s in statuses.values()):
        decision = "PASS"
    else:
        decision = "UNKNOWN"
    unknown = [name for name, status in statuses.items() if status not in {"PASS", "FAIL"}]
    costs = {name: tasks.get(name, {}).get("calibration_cost", {}).get("wall_seconds")
             for name in required}
    known = [value for value in costs.values() if value is not None]
    # An uninterrupted multi-task job reports its whole runtime on each row.
    # Count that attempt once within a screen/reference, including paid retries.
    units = {}
    for name in required:
        receipt = tasks.get(name, {}).get("calibration_cost", {})
        for key, value in receipt.get("cost_units", {name: costs[name]}).items():
            units[key] = value if key not in units or units[key] == value else None
    known = [value for value in units.values() if value is not None]
    complete_cost = all(value is not None for value in costs.values()) and all(value is not None for value in units.values())
    return {"decision": decision, "task_statuses": statuses,
            "unknown_tasks": unknown,
            "blocked_tasks": [n for n, s in statuses.items() if s == "BLOCKED"],
            "error_or_incomplete_tasks": [n for n, s in statuses.items() if s in {"INVALID", "INCOMPLETE"}],
            "failed_tasks": [n for n, s in statuses.items() if s == "FAIL"],
            "cost": {"known_wall_seconds": sum(known), "wall_seconds": sum(known) if complete_cost else None,
                     "known_task_count": sum(v is not None for v in costs.values()), "required_task_count": len(required),
                     "missing_tasks": [n for n, v in costs.items() if v is None],
                     "flops": None, "note": "Recorded runtime; concurrent execution and archived budgets differ."}}


def _lineage(root, spec, cards, profile, sources):
    matching = [(path, card) for path, card in cards
                if card.get("candidate_id") == spec["candidate_id"]
                and card.get("source", {}).get("path") == spec["source_path"]]
    tasks, conflicts = {}, []
    packages = sorted({c.get("provenance", {}).get("package_sha256") for _, c in matching
                       if c.get("provenance", {}).get("package_sha256")})
    expected = PACKAGE_KEYS.get(spec["candidate_id"])
    if expected and packages != [expected]:
        conflicts.append("Frozen package does not match preregistered lineage")
    revisions = sorted({c.get("source", {}).get("revision") for _, c in matching})
    if len(revisions) > 1:
        conflicts.append("Multiple source revisions in the same lineage")
    for _, card in matching:
        for task in card.get("task_results", []):
            name = task["task_id"]
            if name in tasks and stable_hash(tasks[name]) != stable_hash(task):
                conflicts.append("Conflicting duplicate task: " + name)
            else:
                tasks[name] = task
    if conflicts:
        tasks = {name: {"task_id": name, "gate_status": "INVALID", "reason": "; ".join(conflicts)}
                 for name in tasks}
    tasks = {name: {**task, "calibration_cost": _task_cost(root, task, sources)}
             for name, task in tasks.items()}
    smoke = _decision(tasks, profile["smoke_tasks"])
    reference = _decision(tasks, profile["reference_tasks"])
    classification = {("PASS", "PASS"): "true_accept", ("PASS", "FAIL"): "false_accept",
                      ("FAIL", "PASS"): "false_reject", ("FAIL", "FAIL"): "true_reject"}.get(
                          (smoke["decision"], reference["decision"]), "unknown")
    return {**spec, "record_ids": [c["record_id"] for _, c in matching],
            "candidate_revisions": sorted({c["candidate_revision"] for _, c in matching}),
            "sources": [c["source"] for _, c in matching], "package_sha256": packages,
            "frozen_config_and_fixture_location": "Bound input cards preserve exact recipe, fixture, initializer, sampling and package headers.",
            "input_cards": [{"path": str(p.relative_to(root)), "sha256": file_hash(p)} for p, _ in matching],
            "conflicts": conflicts, "smoke": smoke, "reference": reference,
            "classification": classification,
            "cost_receipts": {name: task["calibration_cost"] for name, task in tasks.items()
                              if name in profile["smoke_tasks"] + profile["reference_tasks"]},
            "evidence_scope": "historical", "current_qualification_reuse": False}


def _stats(rows):
    counts = Counter(row["classification"] for row in rows)
    pos = counts["true_accept"] + counts["false_reject"]
    neg = counts["true_reject"] + counts["false_accept"]
    return {"lineages": len(rows), "paired": pos + neg,
            "paired_fraction": (pos + neg) / len(rows) if rows else 0,
            "reference_positives": pos, "reference_negatives": neg,
            "all_reference_positives": sum(r["reference"]["decision"] == "PASS" for r in rows),
            "all_reference_negatives": sum(r["reference"]["decision"] == "FAIL" for r in rows),
            **{name: counts[name] for name in ("true_accept", "false_accept", "true_reject", "false_reject", "unknown")},
            "false_accept_fraction": counts["false_accept"] / neg if neg else None,
            "false_reject_fraction": counts["false_reject"] / pos if pos else None,
            "blocked_lineages": sum(bool(r["smoke"]["blocked_tasks"] + r["reference"]["blocked_tasks"]) for r in rows),
            "incomplete_lineages": sum(bool(r["smoke"]["error_or_incomplete_tasks"] + r["reference"]["error_or_incomplete_tasks"]) for r in rows),
            "reference_unknown_lineages": sum(r["reference"]["decision"] == "UNKNOWN" for r in rows),
            "smoke_unknown_lineages": sum(r["smoke"]["decision"] == "UNKNOWN" for r in rows)}


def _criteria(rows, criteria):
    stats = _stats(rows)
    checks = {}
    for name in ("paired_lineages", "reference_positives", "reference_negatives", "paired_fraction"):
        actual = stats["paired" if name == "paired_lineages" else name]
        checks["minimum_" + name] = {"actual": actual, "required": criteria["minimum_" + name],
                                    "status": "PASS" if actual >= criteria["minimum_" + name] else "BLOCKED"}
    for name in ("false_accept_fraction", "false_reject_fraction"):
        actual = stats[name]
        checks["maximum_" + name] = {"actual": actual, "required": criteria["maximum_" + name],
            "status": "BLOCKED" if actual is None else "PASS" if actual <= criteria["maximum_" + name] else "FAIL"}
    for part in ("smoke", "reference"):
        missing = [r["id"] for r in rows if r[part]["cost"]["wall_seconds"] is None]
        checks["require_complete_" + part + "_cost"] = {"missing_lineages": missing,
            "status": "PASS" if rows and not missing else "BLOCKED"}
    smoke = [r["smoke"]["cost"]["wall_seconds"] for r in rows]
    complete = bool(smoke) and all(v is not None for v in smoke)
    max_smoke = max(smoke) if complete else None
    ratios = [r["smoke"]["cost"]["wall_seconds"] / r["reference"]["cost"]["wall_seconds"]
              for r in rows if r["smoke"]["cost"]["wall_seconds"] is not None
              and r["reference"]["cost"]["wall_seconds"] not in {None, 0}]
    for name, actual in (("smoke_wall_seconds", max_smoke),
                         ("smoke_to_reference_wall_ratio", max(ratios) if len(ratios) == len(rows) and ratios else None)):
        checks["maximum_" + name] = {"actual": actual, "required": criteria["maximum_" + name],
            "status": "BLOCKED" if actual is None else "PASS" if actual <= criteria["maximum_" + name] else "FAIL"}
    return {"stats": stats, "checks": checks,
            "adoption": "PASS" if all(v["status"] == "PASS" for v in checks.values()) else "BLOCKED"}


def _followup_audit(root):
    """Bind the later correction separately from the original pinned import."""
    source = Source(root, FOLLOWUP_REVISION, (LRFREE_ROOT,))
    base = LRFREE_ROOT + "/prior-ema-relaxation/"
    manifest = source.json(base + "manifest.json")
    return {"schema_version": 1, "revision": source.revision,
            "pinned_original_revision": LRFREE_REVISION,
            "searched_scope": "Tracked PR155 report Markdown and latest PR metadata at this revision; no missing claim is treated as disproven.",
            "unbound_user_reported_claims": ["dt075 14k continuation", "EMA .995 D-tracking native 3/3", "sub-.03 sigma sensitivity"],
            "new_distinct_lineage": {"name": "prior-ema-relaxation", "package_sha256": manifest["package_sha256"],
                "diagnostic_host": {"status": "PASS", "native_passes": 3, "native_total": 3,
                    "host": manifest["protocol"]["native_host"], "current_qualification_reuse": False,
                    "reason": "Fresh callback batches change training inputs relative to the frozen cached-callback host."},
                "canonical_host": manifest["canonical_staggered100"],
                "default_promotion": "BLOCKED", "scoring_weights": "live noisy; prior-EMA is a training mechanism, not output-EMA scoring"},
            "evidence": [source.receipt(base + path) for path in ("README.md", "manifest.json", "overrides.json",
                "canonical-staggered100/native-noisy-verdict.json", "canonical-staggered100/job-header.json")],
            "next_action": "Obtain exact missing-claim receipts; keep corrected diagnostic and canonical host cohorts separate."}


def next_missing_runs(root: Path) -> list:
    """Small bounded proposal; package restoration/parity must precede execution."""
    root = Path(root)
    source = Source(root, LRFREE_REVISION, (LRFREE_ROOT,))
    receipts = [source.receipt(LRFREE_ROOT + p) for p in (
        "/harness/tasks/custom22_specs.json", "/harness/hosts/custom/MANIFEST.json")]
    runs = []
    for candidate in ("pr217_qr_native_adapter", "pr215_qr_adapter"):
        for task, updates in (("two_pole", 80), ("unused_token_hold", 200), ("ae_gan_hold", 250)):
            runs.append({"id": candidate + "--" + task, "candidate_id": candidate, "task_id": task,
                         "source_revision": LRFREE_REVISION, "package_sha256": PACKAGE_KEYS[candidate],
                         "fixture_spec_receipts": receipts, "fixed_fixture": True, "seed_sweep": False,
                         "fixture_parameter_sha256": None,
                         "prerequisites": ["Restore exact archived package and harness", "Freeze task parameter/RNG fixture and verify custom-host parity before spending training budget"],
                         "status": "BLOCKED", "reason": "Task-specific parameter receipt is absent; declarations are not an executed fixture.",
                         "maximum_updates": updates, "wall_timeout_seconds": 300,
                         "purpose": "Resolve initial-smoke unknown for an already observed independent quality failure.",
                         "prior_cohort": "historical_particle_cloud", "scoring_weights": "live"})
    return runs


def _promotion_template():
    return {"schema_version": 1, "kind": "public_default_promotion_contract_template_pointer",
            "status": "UNREGISTERED_TEMPLATE", "candidate_id": None, "candidate_revision": None,
            "is_candidate_claim": False, "execution_authorized": False,
            "authoritative_template": "configs/forge/promotion-template.json",
            "eligible_lifecycle": "finished_candidate_with_concluded_tier3_readout",
            "registration_required_before_execution": True,
            "seed_policy": "Finished-candidate robustness only. No screening sweeps, replacement seeds, or best-seed selection; publish every result.",
            "training_allowance_seconds": 0,
            "next_action": "Register the canonical template against one finished candidate before any robustness run."}


def calibration_cohort(request: dict, task_ids: list[str]) -> dict:
    """Scientific comparison cohort, independent of candidate knobs and view policy."""
    from .views import task_execution_fingerprint, task_evaluation_fingerprint
    jobs = {name: job for job in request["jobs"] for name in job.get("task_ids", [job["task_id"]])}
    tasks = {}
    for name in sorted(task_ids):
        task = request["tasks"][name]
        tasks[name] = {"execution": task_execution_fingerprint(task),
                       "evaluation": task_evaluation_fingerprint(task),
                       "execution_group": task["execution"].get("execution_group", name),
                       "compute": jobs[name].get("science", {}).get("compute"),
                       "scoring_weights": task["evaluation"].get("scoring_weights", "live")}
        if tasks[name]["compute"] is None:
            raise ValueError(f"calibration needs a bound compute profile for {name}")
    identity = {"source_sha256": request["source"]["digest"], "protocol": request["protocol"],
                "rng": request["rng"], "runtime": request["runtime"],
                "initializer": request["candidate"].get("initializer", "deterministic_orthogonal"),
                "prior": request["candidate"]["prior"], "tasks": tasks}
    return {"sha256": stable_hash(identity), "identity": deepcopy(identity)}


def profile_task_ids(config: dict) -> list[str]:
    """Bind optional diagnostics without adding them to either decision set."""
    return config["smoke_tasks"] + config["reference_tasks"] + config.get("diagnostic_tasks", [])


_IMPORT_FIELDS = {"attempt_id", "registration_id", "registration_sha256", "candidate_id",
                  "candidate_revision", "tasks", "files", "qualification_reuse"}
_IMPORT_FILES = {"request.json", "result.json", "evidence.json"}


def _diagnostic(request):
    return bool(request.get("calibration_lane") or request.get("view", {}).get("evidence_scope") == "calibration_diagnostic"
                or any(job.get("science", {}).get("evidence_use") == "calibration_diagnostic"
                       or "qualification_compatibility_key" in job for job in request.get("jobs", [])))


def _diagnostic_binding(root, attempt, names):
    """Bind existing original bytes; never manufacture a second receipt."""
    from .calibration_lane import verify_request
    request = attempt["request"]
    lane = verify_request(root, request)
    if not attempt["valid_receipt"]:
        raise ValueError("diagnostic import has an invalid certified receipt: " + str(attempt["receipt_error"]))
    selected = {a["task"] for a in request["view"]["assignments"]}
    rows = [row["task_id"] for row in attempt["task_results"]]
    if len(rows) != len(set(rows)) or not set(names).issubset(selected & set(rows)):
        raise ValueError("diagnostic import tasks must have unique, registered measured receipts")
    jobs = {name: job for job in request["jobs"] for name in job.get("task_ids", [job["task_id"]])}
    directory = root / "reports/forge/attempts" / attempt["attempt_id"]
    binding = {"attempt_id": attempt["attempt_id"], "registration_id": lane["registration_id"],
        "registration_sha256": lane["registration_sha256"], "candidate_id": request["candidate"]["id"],
        "candidate_revision": request["candidate_revision"], "qualification_reuse": False,
        "tasks": {name: {"diagnostic_key": jobs[name]["compatibility_key"],
                         "qualification_key": jobs[name]["qualification_compatibility_key"]} for name in sorted(names)},
        "files": {name: file_hash(directory / name) for name in sorted(_IMPORT_FILES)}}
    return binding, lane


def diagnostic_imports(root: Path, registration_id: str, tasks_by_lineage: dict[str, list[str]]) -> list[dict]:
    """Read-only bindings for a future profile's explicit diagnostic_imports.

    Include every recorded outcome and paid retry for the selected cells, never
    just successful attempts. This authorizes no execution or qualification.
    The consuming profile still verifies its complete scientific cohort.
    """
    from .calibration_lane import _load
    from .knowledge import _attempts
    root = Path(root).resolve()
    artifact = _load(root, registration_id)
    if not isinstance(tasks_by_lineage, dict) or not tasks_by_lineage:
        raise ValueError("diagnostic imports require explicit lineage/task selections")
    for name, tasks in tasks_by_lineage.items():
        subject = artifact["subjects"].get(name)
        if (not subject or not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks)
                or not set(tasks).issubset(subject["selection"]["tasks"])):
            raise ValueError("diagnostic imports must select original registered lineage/tasks")
    attempts, issues = _attempts(root)
    result, measured = [], set()
    for attempt in attempts:
        marker = attempt["request"].get("calibration_lane", {})
        if marker.get("registration_id") != registration_id:
            continue
        lineage = marker.get("lineage_id")
        names = set(tasks_by_lineage.get(lineage, [])) & {r["task_id"] for r in attempt["task_results"]}
        if names:
            binding, _ = _diagnostic_binding(root, attempt, names)
            result.append(binding)
            measured.update((lineage, name) for name in names)
    expected = {(lineage, name) for lineage, names in tasks_by_lineage.items() for name in names}
    if measured != expected:
        raise ValueError("diagnostic imports require already-recorded evidence for every selected cell")
    selected_ids = {row["attempt_id"] for row in result}
    for issue in issues:
        path = root / "reports/forge/attempts" / issue["attempt_id"] / "request.json"
        resolved = read_json(path) if path.is_file() else {}
        marker = resolved.get("request", resolved).get("calibration_lane", {})
        if (issue["attempt_id"] in selected_ids or
                (marker.get("registration_id") == registration_id and marker.get("lineage_id") in tasks_by_lineage)):
            raise ValueError("diagnostic imports contain unresolved receipt/retry issues")
    return result


def _validate_imports(config):
    imports = config.get("diagnostic_imports", [])
    if not isinstance(imports, list):
        raise ValueError("diagnostic_imports must be an explicit list of frozen attempt bindings")
    seen = set()
    lineages = {(r["candidate_id"], r["candidate_revision"]) for r in config["lineages"]}
    names = set(profile_task_ids(config))
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    for row in imports:
        if not isinstance(row, dict) or set(row) != _IMPORT_FIELDS or row["qualification_reuse"] is not False:
            raise ValueError("diagnostic import must bind an original attempt without qualification reuse")
        identifier(row["attempt_id"], "diagnostic import attempt")
        identifier(row["registration_id"], "diagnostic import registration")
        if row["attempt_id"] in seen or (row["candidate_id"], row["candidate_revision"]) not in lineages:
            raise ValueError("diagnostic imports require unique attempts and an exact declared candidate")
        seen.add(row["attempt_id"])
        if (not digest(row["registration_sha256"]) or not isinstance(row["files"], dict)
                or set(row["files"]) != _IMPORT_FILES or not all(digest(v) for v in row["files"].values())):
            raise ValueError("diagnostic import must hash its original registration and all receipt files")
        tasks = row["tasks"]
        if not isinstance(tasks, dict) or not tasks or not set(tasks).issubset(names):
            raise ValueError("diagnostic import tasks must belong to the target profile")
        for keys in tasks.values():
            if (not isinstance(keys, dict) or set(keys) != {"diagnostic_key", "qualification_key"}
                    or not all(digest(v) for v in keys.values()) or keys["diagnostic_key"] == keys["qualification_key"]):
                raise ValueError("diagnostic imports must preserve separate exact diagnostic and qualification keys")


def _foreign_unlisted(request, config, attempt_id):
    """Unselected foreign diagnostics cannot contribute or poison this profile."""
    marker = request.get("calibration_lane", {})
    return (isinstance(marker, dict) and marker.get("profile_sha256") is not None
            and marker["profile_sha256"] != config.get("_profile_sha256")
            and attempt_id not in {row["attempt_id"] for row in config.get("diagnostic_imports", [])})


def _current_profile(config, criteria):
    """Check the frozen current-evidence contract before reading outcomes."""
    if config.get("schema_version") != 1 or config.get("evidence_scope") != "current":
        raise ValueError("current calibration needs schema_version 1 and current evidence scope")
    if not isinstance(config.get("id"), str) or not config["id"] or type(config.get("revision")) is not int or config["revision"] < 1:
        raise ValueError("current calibration needs an explicit profile id and positive revision")
    if not isinstance(config.get("reference_scope"), str) or not config["reference_scope"].strip():
        raise ValueError("declare the independent reference scope before reducing calibration")
    for key in ("smoke_tasks", "reference_tasks"):
        values = config.get(key)
        if not isinstance(values, list) or not values or any(not isinstance(n, str) for n in values) or len(values) != len(set(values)):
            raise ValueError(f"{key} must be a nonempty unique task list")
    diagnostics = config.get("diagnostic_tasks", [])
    if (not isinstance(diagnostics, list) or any(not isinstance(n, str) or not n for n in diagnostics)
            or len(diagnostics) != len(set(diagnostics))):
        raise ValueError("diagnostic_tasks must be an explicit unique task list")
    if set(diagnostics) & set(config["smoke_tasks"] + config["reference_tasks"]):
        raise ValueError("diagnostic tasks may not enter smoke or independent reference decisions")
    if set(config["smoke_tasks"]) & set(config["reference_tasks"]):
        raise ValueError("Reference may not include smoke predicates")
    if set(config.get("reference_exclusions", [])) & set(config["reference_tasks"]):
        raise ValueError("Reference includes a preregistered excluded predicate")
    if config.get("training_allowance_seconds") != 0:
        raise ValueError("calibration reduction never authorizes training")
    if config.get("scoring_weights") != "live":
        raise ValueError("current calibration supports the declared live evaluator only")
    cohort = config.get("cohort", {})
    identity = cohort.get("identity", {})
    if not identity or cohort.get("sha256") != stable_hash(identity):
        raise ValueError("calibration requires the complete frozen scientific cohort identity")
    required = set(profile_task_ids(config))
    if set(identity.get("tasks", {})) != required:
        raise ValueError("cohort task definitions differ from declared smoke/reference/diagnostic tasks")
    groups = lambda names: {identity["tasks"][n]["execution_group"] for n in names}
    if groups(config["smoke_tasks"]) & groups(config["reference_tasks"]):
        raise ValueError("independent reference cannot share a smoke execution group")
    if groups(diagnostics) & groups(config["smoke_tasks"] + config["reference_tasks"]):
        raise ValueError("diagnostics cannot share a smoke/reference execution group or its cost")
    fingerprints = lambda names: {(identity["tasks"][n]["execution"], identity["tasks"][n]["evaluation"]) for n in names}
    if fingerprints(config["smoke_tasks"]) & fingerprints(config["reference_tasks"]):
        raise ValueError("independent reference cannot rename an identical smoke measurement")
    if fingerprints(diagnostics) & fingerprints(config["smoke_tasks"] + config["reference_tasks"]):
        raise ValueError("diagnostics cannot rename an identical smoke/reference measurement")
    if any(identity["tasks"][n].get("scoring_weights") != config["scoring_weights"] for n in required):
        raise ValueError("cohort and profile scoring policies differ")
    prior = identity.get("prior", {})
    if criteria.get("require_new_mog_cohort") and not (
            prior.get("kind") == "mog" and prior.get("learnable") is True
            and type(prior.get("sigma")) in (int, float) and math.isfinite(prior["sigma"]) and prior["sigma"] > 0):
        raise ValueError("adoption requires a measured learned-MoG cohort with explicit positive width")
    protocol = identity.get("protocol", {})
    if type(protocol.get("seed")) is not int or protocol["seed"] != 0 or "promotion" in protocol:
        raise ValueError("calibration fixes screening seed 0; finished-candidate robustness is a separate protocol")
    lineages = config.get("lineages")
    if not isinstance(lineages, list) or not lineages:
        raise ValueError("freeze the positive/negative reference candidate lineages before calibration")
    for row in lineages:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k] for k in ("id", "candidate_id", "candidate_revision")):
            raise ValueError("each calibration lineage requires exact candidate identity and revision")
    if len({r["id"] for r in lineages}) != len(lineages) or len({r["candidate_revision"] for r in lineages}) != len(lineages):
        raise ValueError("renamed duplicate candidate revisions are not independent calibration lineages")
    _validate_imports(config)
    # Cost completeness and unknown handling are mandatory, not optional flags.
    if (criteria.get("require_complete_smoke_cost") is not True
            or criteria.get("require_complete_reference_cost") is not True
            or criteria.get("unknowns_count_as_failures") is not False):
        raise ValueError("calibration must retain unknowns and require complete smoke/reference costs")
    for key in ("minimum_paired_lineages", "minimum_reference_positives", "minimum_reference_negatives"):
        if type(criteria.get(key)) is not int or criteria[key] < 1:
            raise ValueError("calibration criteria require positive and negative references")
    for key in ("minimum_paired_fraction", "maximum_false_accept_fraction", "maximum_false_reject_fraction"):
        value = criteria.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid calibration criterion {key}")
    for key in ("maximum_smoke_wall_seconds", "maximum_smoke_to_reference_wall_ratio"):
        value = criteria.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid calibration cost criterion {key}")


def _current_lineage(root, spec, config, attempts):
    from .views import grade_result
    names = profile_task_ids(config)
    indexed, inputs, problems = defaultdict(list), [], []
    imports = {row["attempt_id"]: row for row in config.get("diagnostic_imports", [])
               if (row["candidate_id"], row["candidate_revision"]) == (spec["candidate_id"], spec["candidate_revision"])}
    import_tasks = defaultdict(set)
    for binding in imports.values():
        import_tasks[binding["registration_id"]].update(binding["tasks"])
    present = {attempt["attempt_id"] for attempt in attempts}
    for identity, binding in imports.items():
        if identity not in present:
            problems.append(f"{identity}: imported diagnostic receipt is missing or incomplete")
            inputs.append({"attempt_id": identity, "diagnostic_import": binding, "missing": True})
            for name in binding["tasks"]:
                indexed[name].append(({"gate_status": "INVALID"}, None, identity, False))
    for attempt in attempts:
        request = attempt["request"]
        identity = attempt["attempt_id"]
        binding = imports.get(identity)
        same_candidate = (request.get("candidate", {}).get("id") == spec["candidate_id"]
                          and request.get("candidate_revision") == spec["candidate_revision"])
        if not same_candidate and binding is None:
            continue
        marker = request.get("calibration_lane", {})
        history_names = import_tasks.get(marker.get("registration_id"), set()) if isinstance(marker, dict) else set()
        history_names = history_names & {row["task_id"] for row in attempt["task_results"]}
        if _foreign_unlisted(request, config, identity) and not history_names:
            continue
        try:
            compatible = calibration_cohort(request, names) == config["cohort"]
        except (KeyError, TypeError, ValueError):
            compatible = False
        if not compatible and binding is None and not history_names:
            continue
        lane_error = None
        authorized = set(binding["tasks"]) if binding else set(names)
        if binding:
            try:
                actual, lane = _diagnostic_binding(root, attempt, authorized)
                if actual != binding:
                    raise ValueError("diagnostic import differs from its frozen registration, task keys or receipt bytes")
                if not same_candidate or not compatible:
                    raise ValueError("diagnostic import differs from the target candidate or complete scientific cohort")
                if lane["criteria_sha256"] != config["criteria_sha256"]:
                    raise ValueError("diagnostic import changes the frozen calibration criteria policy")
                neighbors = [attempt.get("superseded_by"), (attempt.get("retry_of") or {}).get("attempt_id")]
                if any(neighbor and neighbor not in imports for neighbor in neighbors):
                    raise ValueError("diagnostic imports must retain the entire certified retry history")
            except (ImportError, KeyError, TypeError, ValueError, OSError) as error:
                lane_error = str(error)
        elif history_names:
            authorized = history_names
            lane_error = "diagnostic imports omitted recorded outcomes or costs from a selected registration/task"
        elif _diagnostic(request):
            try:
                from .calibration_lane import verify_request
                lane = verify_request(root, request)
                for key in ("profile_sha256", "criteria_sha256", "cohort_sha256"):
                    expected = {"profile_sha256": config.get("_profile_sha256"),
                                "criteria_sha256": config["criteria_sha256"],
                                "cohort_sha256": config["cohort"]["sha256"]}[key]
                    if lane.get(key) != expected:
                        raise ValueError(f"diagnostic lane {key} differs from calibration registration")
            except (ImportError, KeyError, TypeError, ValueError, OSError) as error:
                lane_error = str(error)
        directory = root / "reports/forge/attempts" / identity
        inputs.append({"attempt_id": identity, "result_sha256": attempt["result_hash"],
                       **({"diagnostic_import": binding} if binding else {}),
                       "files": {name: file_hash(directory / name) for name in ("request.json", "result.json", "evidence.json")
                                 if (directory / name).is_file()}})
        if not attempt["valid_receipt"]:
            problems.append(f"{attempt['attempt_id']}: {attempt['receipt_error']}")
        if lane_error:
            problems.append(f"{attempt['attempt_id']}: invalid diagnostic registration: {lane_error}")
        seen = set()
        selected = {a["task"] for a in request.get("view", {}).get("assignments", [])}
        for row in attempt["task_results"]:
            if row["task_id"] in authorized:
                if row["task_id"] in seen or (request.get("calibration_lane") and row["task_id"] not in selected):
                    lane_error = "duplicate task receipt or task outside registered diagnostic selection"
                    problems.append(f"{attempt['attempt_id']}: {lane_error}")
                seen.add(row["task_id"])
                grade = grade_result(request["tasks"][row["task_id"]], row) if attempt["valid_receipt"] and not lane_error else {"gate_status": "INVALID"}
                cost = row.get("cost", {})
                seconds = _seconds(cost.get("wall_seconds", cost.get("seconds")))
                indexed[row["task_id"]].append((grade, seconds, attempt["attempt_id"], bool(attempt.get("superseded_by"))))
        if binding:
            for name in authorized - seen:
                indexed[name].append(({"gate_status": "INVALID"}, None, identity, False))
    tasks = {}
    for name, matches in indexed.items():
        active = [grade for grade, _, _, superseded in matches if not superseded]
        signatures = {stable_hash(grade) for grade in active}
        status = active[0]["gate_status"] if len(signatures) == 1 else "INVALID"
        if len(signatures) != 1:
            problems.append(f"{name}: conflicting compatible outcomes")
        costs = {attempt_id: cost for _, cost, attempt_id, _ in matches}
        tasks[name] = {"gate_status": status, "calibration_cost": {
            "wall_seconds": sum(costs.values()) if all(c is not None for c in costs.values()) else None,
            "cost_units": costs}}
    smoke, reference = _decision(tasks, config["smoke_tasks"]), _decision(tasks, config["reference_tasks"])
    classification = {("PASS", "PASS"): "true_accept", ("PASS", "FAIL"): "false_accept",
                      ("FAIL", "PASS"): "false_reject", ("FAIL", "FAIL"): "true_reject"}.get(
                          (smoke["decision"], reference["decision"]), "unknown")
    return {**spec, "cohort_sha256": config["cohort"]["sha256"], "smoke": smoke, "reference": reference,
            **({"diagnostics": {**_decision(tasks, config["diagnostic_tasks"]),
                "reference_credit": False, "current_qualification_reuse": False}}
               if "diagnostic_tasks" in config else {}),
            "classification": classification, "conflicts": problems, "inputs": inputs,
            "incompatible_evidence_policy": "Excluded without cross-cohort or cross-seed credit.", "evidence_scope": "current",
            "current_qualification_reuse": False,
            "scope": "Calibration diagnoses the screen; it never bypasses failed qualification gates."}


def _evaluate_current(root, profile, config, config_path, criteria, criteria_path):
    from .knowledge import _attempts
    _current_profile(config, criteria)
    if config.get("criteria_sha256") != file_hash(criteria_path):
        raise ValueError("current calibration criteria changed after profile registration")
    attempts, issues = _attempts(root)
    reduction_config = {**config, "_profile_sha256": file_hash(config_path)}
    matrix = [_current_lineage(root, row, reduction_config, attempts) for row in config["lineages"]]
    cohort = {**config["cohort"], **_criteria(matrix, criteria)}
    relevant_ids = {entry["attempt_id"] for row in matrix for entry in row["inputs"]}
    # In-progress/malformed durable attempts must not disappear behind an older
    # complete result for the same preregistered lineage.
    selected = {(row["candidate_id"], row["candidate_revision"]) for row in config["lineages"]}
    for issue in issues:
        path = root / "reports/forge/attempts" / issue["attempt_id"] / "request.json"
        if path.is_file():
            resolved = read_json(path)
            request = resolved.get("request", resolved)
            marker = request.get("calibration_lane", {})
            imported_history = any(
                row["registration_id"] == marker.get("registration_id")
                and row["candidate_id"] == request.get("candidate", {}).get("id")
                and row["candidate_revision"] == request.get("candidate_revision")
                for row in config.get("diagnostic_imports", [])) if isinstance(marker, dict) else False
            if _foreign_unlisted(request, reduction_config, issue["attempt_id"]) and not imported_history:
                continue
            try:
                matches = ((request["candidate"]["id"], request["candidate_revision"]) in selected
                           and calibration_cohort(request, profile_task_ids(config)) == config["cohort"])
            except (KeyError, TypeError, ValueError):
                matches = False
            if matches or imported_history:
                relevant_ids.add(issue["attempt_id"])
    relevant_issues = [issue for issue in issues if issue.get("attempt_id") in relevant_ids]
    complete = all(not row["smoke"]["unknown_tasks"] and not row["reference"]["unknown_tasks"] for row in matrix)
    accepted = cohort["adoption"] == "PASS" and complete and not relevant_issues and not any(row["conflicts"] for row in matrix)
    return {"schema_version": 1, "profile": profile, "profile_path": str(config_path.relative_to(root)),
            "profile_sha256": file_hash(config_path), "criteria_path": str(criteria_path.relative_to(root)),
            "criteria_sha256": file_hash(criteria_path), "criteria": criteria,
            "reducer_sha256": file_hash(Path(__file__)), "evidence_scope": "current",
            "training_seconds_spent": 0, "current_qualification_reuse": False,
            "diagnostic_import_policy": "Only frozen original receipt bindings with unchanged science and criteria; all selected outcomes and retry costs retained; no qualification credit.",
            "sampling_limit": "Fixed selected lineages diagnose this screen; these fractions are not unbiased population error estimates.",
            "smoke_tasks": config["smoke_tasks"], "reference_tasks": config["reference_tasks"],
            **({"diagnostic_tasks": config["diagnostic_tasks"],
                "diagnostic_scope": "Separate outcomes and paid costs; no smoke/reference decision, criterion or qualification credit."}
               if "diagnostic_tasks" in config else {}),
            "reference_scope": config.get("reference_scope"), "cohorts": [cohort], "matrix": matrix,
            "receipt_issues": relevant_issues, "complete": complete,
            "adoption": "PASS" if accepted else "BLOCKED",
            "recommendation": "Accept only this bound cohort/profile if all frozen criteria pass; no inherited historical-cloud or cross-seed credit."}


def verify_calibration(root: Path, request: dict) -> dict:
    """Recompute the accepted report and bind it to this scientific cohort."""
    root = Path(root).resolve()
    binding = request.get("view", {}).get("calibration", {})
    if binding.get("status") not in ("accepted", "PASS"):
        raise ValueError("promotion requires accepted view calibration")
    def inside(relative):
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("accepted calibration paths must be repository-relative")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("accepted calibration artifact is missing or outside the repository")
        return path
    path = inside(binding.get("report"))
    if binding.get("report_sha256") != file_hash(path):
        raise ValueError("accepted calibration report hash mismatch")
    saved = read_json(path)
    config_path, criteria_path = inside(saved.get("profile_path")), inside(saved.get("criteria_path"))
    config, criteria = read_json(config_path), read_json(criteria_path)
    if (binding.get("profile_sha256") != file_hash(config_path)
            or binding.get("criteria_sha256") != file_hash(criteria_path)):
        raise ValueError("accepted calibration profile or criteria hash mismatch")
    rebuilt = _evaluate_current(root, saved["profile"], config, config_path, criteria, criteria_path)
    if rebuilt != saved or rebuilt["adoption"] != "PASS":
        raise ValueError("accepted calibration lacks unchanged complete passing evidence")
    expected = calibration_cohort(request, profile_task_ids(saved))
    if expected != config["cohort"] or binding.get("cohort_sha256") != expected["sha256"]:
        raise ValueError("accepted calibration does not cover this source/task/prior/RNG/runtime cohort")
    required_smoke = {row["task"] for row in request["view"]["assignments"]
                      if row["qualification_tier"] == 1 and row["importance"] == "required"}
    if required_smoke != set(saved["smoke_tasks"]):
        raise ValueError("accepted calibration does not cover this required Tier 1 screen")
    return {key: binding[key] for key in ("report", "report_sha256", "profile_sha256", "criteria_sha256", "cohort_sha256")}


def calibrate(root: Path, profile: str = "initial") -> dict:
    """Publish a deterministic saved-evidence matrix and proposed bounded work."""
    root = Path(root).resolve()
    identifier(profile, "calibration profile")
    config_path = root / "configs/forge/calibration" / (profile + ".json")
    config = read_json(config_path)
    criteria_path = config_path.parent / identifier(config["criteria"], "criteria file")
    criteria = read_json(criteria_path)
    if set(config["smoke_tasks"]) & set(config["reference_tasks"]):
        raise ValueError("Reference may not include smoke predicates")
    if set(config.get("reference_exclusions", [])) & set(config["reference_tasks"]):
        raise ValueError("Reference includes a preregistered excluded predicate")
    if config.get("training_allowance_seconds") != 0:
        raise ValueError("Historical calibration never authorizes training")
    if config.get("evidence_scope") == "current":
        output = _evaluate_current(root, profile, config, config_path, criteria, criteria_path)
        directory = root / "reports/forge/calibration"
        atomic_json(directory / (profile + ".json"), output)
        lines = [f"# Forge calibration: {profile}", "", f"Adoption: **{output['adoption']}**. This reduction launches no training.", "",
                 f"Bound source/prior/runtime cohort: `{config['cohort']['sha256']}`.", "",
                 "| Exact lineage | Smoke | Independent reference | Classification | Smoke seconds | Reference seconds |",
                 "| --- | --- | --- | --- | ---: | ---: |"]
        for row in output["matrix"]:
            costs = [row[k]["cost"]["wall_seconds"] for k in ("smoke", "reference")]
            lines.append("| " + " | ".join([row["id"], row["smoke"]["decision"], row["reference"]["decision"], row["classification"],
                *("unknown" if value is None else f"{value:.3f}" for value in costs)]) + " |")
        if output.get("diagnostic_tasks"):
            lines += ["", "Separate diagnostics provide no independent reference or qualification credit.", "",
                      "| Exact lineage | Diagnostic task | Outcome | Diagnostic seconds |",
                      "| --- | --- | --- | ---: |"]
            for row in output["matrix"]:
                diagnostic = row["diagnostics"]
                for name, status in diagnostic["task_statuses"].items():
                    lines.append(f"| {row['id']} | {name} | {status} | See lineage cost total below |")
                value = diagnostic["cost"]["wall_seconds"]
                seconds = "unknown" if value is None else f"{value:.3f}"
                lines.append(f"| {row['id']} | All declared diagnostics (grouped/retry costs counted once) | — | {seconds} |")
        lines += ["", output["recommendation"], "", "Complete raw evidence and costs are required; unknown/incompatible cells remain in the denominator."]
        atomic_text(directory / (profile + ".md"), "\n".join(lines) + "\n")
        return output
    cards = [(p, read_json(p)) for p in sorted((root / "reports/forge/records").glob("history-*.json"))]
    sources = {}
    matrix = [_lineage(root, spec, cards, config, sources) for spec in config["lineages"]]
    grouped = defaultdict(list)
    for row in matrix:
        grouped[(row["prior_cohort"], row["runtime_cohort"])].append(row)
    cohorts = [{"prior_cohort": key[0], "runtime_cohort": key[1], **_criteria(rows, criteria)}
               for key, rows in sorted(grouped.items())]
    # A historical cloud is not evidence for the new learned-MoG public default.
    new_mog = {"prior_cohort": "new_learned_mog_sigma_0.025", "runtime_cohort": "unmeasured",
               **_criteria([], criteria), "reason": "No compatible new-MoG calibration receipts exist in the archived sources."}
    if criteria["require_new_mog_cohort"]:
        cohorts.append(new_mog)
    output = {"schema_version": 1, "profile": profile, "profile_sha256": file_hash(config_path),
              "criteria": criteria, "criteria_sha256": file_hash(criteria_path),
              "reducer_sha256": file_hash(Path(__file__)), "evidence_scope": "historical",
              "training_seconds_spent": 0, "current_qualification_reuse": False,
              "sampling_limit": "Retrospectively selected saved lineages; these fractions are descriptive diagnostics, not unbiased population error estimates.",
              "profile_updates": config.get("smoke_updates"), "reference_scope": config["reference_scope"],
              "descriptive_totals_not_pooled_adoption": _stats(matrix), "cohorts": cohorts, "matrix": matrix,
              "adoption": "PASS" if cohorts and all(c["adoption"] == "PASS" for c in cohorts) else "BLOCKED",
              "interpretation": "Historical outcomes assess rejection risk; unknowns and host/prior differences prevent a validated default. Passing custom smoke does not predict native100 quality.",
              "recommendation": "Keep initial gates provisional. Neither profile satisfies frozen adoption criteria. Preserve independent native quality gates; repair evidence/parity before the small proposed campaign, then calibrate new MoG separately."}
    directory = root / "reports/forge/calibration"
    atomic_json(directory / (profile + ".json"), output)
    lines = [f"# Forge calibration: {profile}", "", f"Adoption: **{output['adoption']}**. Training spent: 0 seconds.", "",
             "Historical totals are descriptive; prior/runtime cohorts remain separate. Unknowns never become scientific failures.", "",
             "| Lineage | Smoke | Independent quality | Classification | Smoke seconds | Reference seconds |",
             "|---|---|---|---|---:|---:|"]
    for row in matrix:
        values = [row["id"], row["smoke"]["decision"], row["reference"]["decision"], row["classification"]]
        values += [f"{row[part]['cost']['wall_seconds']:.2f}" if row[part]["cost"]["wall_seconds"] is not None
                   else f"unknown ({row[part]['cost']['known_task_count']}/{row[part]['cost']['required_task_count']} timed)"
                   for part in ("smoke", "reference")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "```json", json.dumps(output["descriptive_totals_not_pooled_adoption"], indent=2), "```", "",
              output["recommendation"], "", "Exact cards, fixture/package links, missing-cost lists and per-cohort acceptance checks are in the adjacent JSON."]
    atomic_text(directory / (profile + ".md"), "\n".join(lines) + "\n")
    atomic_json(directory / "followup-source-audit.json", _followup_audit(root))
    missing = next_missing_runs(root)
    proposal = {"schema_version": 1, "status": "PROPOSED_BLOCKED_NOT_AUTHORIZED", "training_allowance_seconds": 0,
                "maximum_training_wall_seconds": sum(r["wall_timeout_seconds"] for r in missing),
                "maximum_jobs": len(missing), "retry_budget": 0, "runs": missing,
                "stop_rule": "Stop each candidate at first scientific smoke FAIL; host or fixture refusal is BLOCKED before training.",
                "scope": "Six missing historical cloud smoke receipts only. This does not establish new-MoG adoption.",
                "new_mog_next_step": "Use the existing bounded Forge pilot to bind a finished baseline/challenger cohort; do not relabel cloud receipts or launch all reference experiments."}
    atomic_json(root / "configs/forge/calibration/missing-fixed-fixture-campaign.json", proposal)
    atomic_json(directory / "missing-runs.json", proposal)
    atomic_json(root / "reports/forge/promotion-contract-template.json", _promotion_template())
    return output
