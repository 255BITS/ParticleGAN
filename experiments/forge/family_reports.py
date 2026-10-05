"""Generated navigation over recorded results, never a qualification reducer.

One selected scientific row supplies every view in a runtime group. Views share
task results, so aggregate counts describe view requirements, not unique runs.
Earlier task verdicts retain their contracts and cannot qualify newer tasks.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import os
from pathlib import Path

from .contracts import file_hash, identifier, read_json, stable_hash
from . import family_documentation
from .views import task_evaluation_fingerprint, task_execution_fingerprint


FAMILY_DIRECTORY = Path("reports/forge/families")
COMPLETE = {"PASS", "FAIL"}
TIERS = ("1", "2", "3")


FIRST_FULL_ATLAS_RESULT = Path("reports/forge/common26-first-two-pole-full-atlas-20261004/results.json")
FIRST_FULL_ATLAS_RESULT_SHA256 = "66551a0018a22cc96ffe88bd27754634ddb5768c5fd896cf6f5dd262e72aa38b"
FULL_ORIGINAL_ATLAS_CONFIG_SHA256 = "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"


# These immutable public reports supply navigation only, never selected-row cells.
ATLAS_EVIDENCE_REPORTS = (
    ("live_clean_common26", "reports/forge/common26-full-original-diagnostic-20261005/results.json",
     "02cee5ca6877a4fd8c8b7bca78bcc245d905b0482fee2626a0c8b1220480b6e5", 922104,
     "pg_common26_full_original_diagnostic_checkpoint_v1"),
    ("native_restoration", "reports/forge/atlas-native-restoration-checkpoint-20261005/results.json",
     "8d1b91ea5326731176edc68df06ed36d38007e62d4ebf704fde1d5bf35e4b698", 34254,
     "pg_atlas_restoration_public_report_v1"),
    ("ae_sourceguard", "reports/forge/atlas-ae-sourceguard-checkpoint-20261005/results.json",
     "e6d93f4a850e5c4afd2ab2c4fa18d5b9dab4b34968b5c89e320fd16626d622a6", 15492,
     "pg_ae633_sourceguard_public_checkpoint_v1"),
)


def _full_original_atlas_first_case(root, load):
    """Navigation over one immutable accepted result; never a selected-row grade."""
    if not (root / FIRST_FULL_ATLAS_RESULT).is_file():
        return None
    if file_hash(root / FIRST_FULL_ATLAS_RESULT) != FIRST_FULL_ATLAS_RESULT_SHA256:
        raise ValueError("retained first Full Atlas result changed")
    record = load(FIRST_FULL_ATLAS_RESULT)
    result = record.get("result", {})
    certificate = result.get("certificate", {})
    grade = certificate.get("grade", {})
    candidate = record.get("candidate", {})
    binding = record.get("binding", {})
    source = record.get("source", {})
    repeat = record.get("protocol", {}).get("scientific_repeat", {})
    convergence = grade.get("evaluator_result", {}).get("convergence", {})
    if (record.get("schema") != "pg_canonical_two_pole_first_case_public_v1"
            or record.get("task_id") != "two_pole" or result.get("task_id") != "two_pole"
            or candidate.get("id") != "atlas-full-original-common26-ember552"
            or candidate.get("trainer_family") != "atlas"
            or binding.get("config_sha256") != FULL_ORIGINAL_ATLAS_CONFIG_SHA256
            or repeat.get("config_sha256") != FULL_ORIGINAL_ATLAS_CONFIG_SHA256
            or repeat.get("source_digest") != source.get("digest")
            or repeat.get("source_origin_commit") != source.get("origin_commit")
            or type(repeat.get("execution_updates")) is not int or repeat["execution_updates"] != 80
            or type(repeat.get("seed")) is not int or repeat["seed"] != 0
            or certificate.get("full_protocol_complete") is not True
            or result.get("status") not in COMPLETE
            or grade.get("status") != result["status"] or grade.get("gate_status") != result["status"]
            or grade.get("evaluator_result", {}).get("status") != result["status"]
            or convergence.get("complete") is not True
            or type(convergence.get("observations")) is not int or convergence["observations"] != 24):
        raise ValueError("retained first Full Atlas result has conflicting scope or completeness")
    checks = grade.get("evaluator_result", {}).get("metrics", [])
    if {check.get("metric") for check in checks} != {"mean_abs", "grad_med"} or len(checks) != 2:
        raise ValueError("retained first Full Atlas result has incomplete metric receipts")
    return {"schema": "pg_full_original_atlas_common26_navigation_v1", "qualification_input": False,
            "configuration_id": candidate["id"], "config_sha256": binding["config_sha256"],
            "readout": FIRST_FULL_ATLAS_RESULT.with_name("README.md").as_posix(),
            "public_result": FIRST_FULL_ATLAS_RESULT.as_posix(), "public_result_sha256": FIRST_FULL_ATLAS_RESULT_SHA256,
            "source": {key: source[key] for key in ("origin_commit", "digest")},
            "first_case": {"task_id": "two_pole", "status": result["status"], "completed_updates": 80,
                           "observations": 24, "seed": 0, "metric_receipts": [
                               {key: deepcopy(check[key]) for key in ("metric", "value", "op", "threshold", "status")}
                               for check in checks]},
            "completed": 1, "required": 26, "remaining_not_run": 25,
            "campaign": None, "campaign_status": "PENDING_ADAPTER_AND_BUDGET",
            "current_selected_configuration_credit": False, "prerequisite_credit": False,
            "default_adoption": False, "speed_ranking": False}


def _atlas_evidence_navigation(root, load):
    """Read separate byte-bound public scopes without importing grades into a row."""
    cohorts = []
    for scope, relative, digest, size, schema in ATLAS_EVIDENCE_REPORTS:
        path = Path(relative)
        if not (root / path).is_file():
            continue  # Historical fixtures need not contain later publications.
        if (root / path).stat().st_size != size or file_hash(root / path) != digest:
            raise ValueError("Atlas evidence navigation public report changed: " + scope)
        record = load(path)
        if record.get("schema") != schema:
            raise ValueError("Atlas evidence navigation has an unsupported public schema")
        entry = {"scope_id": scope, "public_schema": schema, "qualification_input": False,
                 "selected_row_credit": False, "readout": path.with_name("README.md").as_posix(),
                 "public_result": relative, "public_result_sha256": digest, "public_result_bytes": size}
        if scope == "live_clean_common26":
            if (record.get("config_sha256") != FULL_ORIGINAL_ATLAS_CONFIG_SHA256
                    or any(record.get(key) is not False for key in
                           ("qualification_credit", "default_adoption", "speed_ranking",
                            "same_source_whole26_claim", "measured_whole26_ranking"))
                    or not isinstance(record.get("campaign_status"), str)
                    or type(record.get("halt_required")) is not bool):
                raise ValueError("retained common26 navigation changed its scope")
            entry.update(label="Retained live-clean common-26 diagnostic", status=record["campaign_status"],
                         halt_required=record["halt_required"], counts=deepcopy(record["counts"]),
                         source=deepcopy(record["current_source"]),
                         retained_first_case_source=deepcopy(record["retained_first_case_source"]))
        elif scope == "native_restoration":
            cases = record.get("native_cases", [])
            if (not isinstance(cases, list) or [case.get("task_id") for case in cases]
                    != ["grid100", "rotated100", "staggered100"]
                    or any(record.get("claims", {}).get(key) is not False for key in
                           ("qualification_credit", "original_common26_credit", "default_adoption", "speed_ranking"))):
                raise ValueError("native restoration navigation changed its roster or scope")
            for case in cases:
                source = case.get("source", {})
                grade = case.get("grade", {})
                if (case.get("candidate_id") != "atlas-native-original-representation-selected-seed0-v1"
                        or case.get("recipe_sha256") != "152ee607b4a2b18d99e9f440985b920363a481a14236ae12409747e9c283738a"
                        or source.get("origin_commit") != "9da0e927a0a4242ee813ceb0340e9fee70ccdc40"
                        or source.get("execution_digest") != "00c19ef01f68e3222888a9b0edc6fcc5f635a384b982260c4e005832c1ed09ae"
                        or grade.get("status") not in COMPLETE or grade.get("full_protocol_complete") is not True):
                    raise ValueError("native restoration navigation changed its binding or completion")
            entry.update(label="Restored native Atlas — raw particles, public selected/noisy",
                         candidate_id=cases[0]["candidate_id"], recipe_sha256=cases[0]["recipe_sha256"],
                         source={"origin_commit": cases[0]["source"]["origin_commit"],
                                 "digest": cases[0]["source"]["execution_digest"]},
                         task_statuses=[{"task_id": case["task_id"], "status": case["grade"]["status"]} for case in cases])
        else:
            grade = record.get("accepted_evidence", {}).get("grade", {})
            if (record.get("candidate_id") != "atlas-full-original-ae-sourceguard-repair-v1"
                    or record.get("task_id") != "ae_gan_hold"
                    or record.get("recipe_sha256") != "a26eb000fbb7616c90265a4bf31c5126e75f86a4cbd04912d71dc0292167e250"
                    or record.get("source") != {"origin_commit": "36718f9ecbffe6a217914de7a9c22cff1349079e",
                                               "digest": "2f71661f3407a3af3b166493fa0cb1f6730e8ac4689c2b37a8c8a9e424c5c233"}
                    or record.get("complete_protocol") is not True or record.get("completed_updates") != 250
                    or record.get("observation_count") != 24 or record.get("status") not in COMPLETE
                    or grade.get("status") != record["status"] or grade.get("gate_status") != record["status"]
                    or any(record.get(key) is not False for key in
                           ("qualification_credit", "default_adoption", "speed_ranking"))):
                raise ValueError("repaired AE navigation changed its binding or completion")
            entry.update(label="Repaired Atlas AE — auxiliary MoG, live scheduled-noise",
                         candidate_id=record["candidate_id"], recipe_sha256=record["recipe_sha256"],
                         source=deepcopy(record["source"]),
                         task_statuses=[{"task_id": record["task_id"], "status": record["status"]}])
        cohorts.append(entry)
    if not cohorts:
        return None
    return {"schema": "pg_atlas_evidence_navigation_v1", "qualification_input": False,
            "selected_row_credit": False, "cohorts": cohorts}


def _full_original_atlas_status(root, page, progress):
    lines = []
    navigation = progress.get("atlas_evidence_navigation")
    if navigation is not None:
        links = []
        for cohort in navigation["cohorts"]:
            outcome = (", ".join(row["task_id"] + " " + row["status"] for row in cohort["task_statuses"])
                       if "task_statuses" in cohort else cohort["status"])
            links.append(link(root, page, cohort["label"] + ": " + outcome, cohort["readout"]))
        if navigation.get("next_steps"):
            links.append(link(root, page, "Atlas inventory gaps and bounded next steps", navigation["next_steps"]["readout"]))
        lines += ["**Atlas measured evidence — separate configurations and contracts.** " + " · ".join(links) + ". " +
                  "These records do not add cells to the selected Atlas row or grant prerequisite credit, "
                  "default adoption or speed ranking.", ""]
    context = progress.get("full_original_atlas_common26")
    if context is None:
        return lines
    first = context["first_case"]
    metrics = "; ".join(cell(check["metric"]) + " " + number(check["value"]) + " " + cell(check["op"]) + " " +
                        number(check["threshold"]) + " (" + cell(check["status"]) + ")"
                        for check in first["metric_receipts"])
    return lines + ["**Archived first Full Atlas case: two_pole " + first["status"] + ".** " + metrics + ". " +
                    "This accepted first case completed 80 updates and 24 ordinary live observations at seed 0. " +
                    link(root, page, "Archived first-case result and goal GIF", context["readout"]) + " · " +
                    link(root, page, "Pinned result, full Recipe and source", context["public_result"]) + ". " +
                    "Its first-case status does not describe the later diagnostic scopes or the canonical Atlas "
                    "configuration selected in the recorded table. No selected-table cells, prerequisite credit, "
                    "default adoption or speed ranking are awarded.", ""]

def cell(value):
    return str(value if value is not None else "unavailable").replace("|", "\\|").replace("\n", " ")


def number(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return cell(value)


def link(root, page, label, target, anchor=None):
    relative = os.path.relpath(root / target, page.parent)
    return f"[{cell(label)}]({relative}" + (f"#{anchor}" if anchor else "") + ")"


def evaluation_rows(evaluation):
    """Readable bounds from the declaration, without an invented combined score."""
    rows = [(name, op, bound) for name, op, bound in evaluation.get("thresholds", [])]
    for name, bound in evaluation.get("coverage_thresholds", {}).items():
        op = ">=" if name.startswith("min_") else "<=" if name.startswith("max_") else "=="
        rows.append(("coverage." + name, op, bound))
    rows += [("accuracy." + name, "<=", bound) for name, bound in evaluation.get("accuracy_limits", {}).items()]
    return rows


def evaluation_notes(evaluation):
    notes = []
    if evaluation.get("minimum_stable_checks"):
        notes.append(f"At least {evaluation['minimum_stable_checks']} consecutive passing terminal observations.")
    if evaluation.get("kind") == "transfer_sustained":
        notes.append(f"All {evaluation.get('observations', 24)} declared observations and final live metrics are required.")
    if evaluation.get("kind") == "native_accuracy":
        notes.append("Both sustained coverage and independent holdout accuracy must pass.")
    if evaluation.get("conditions"):
        notes.append("Exact state/output parity for: " + ", ".join(evaluation["conditions"]) + "; bound source audit required.")
    for key, label in (("confirmation_checks", "Confirmation checks"), ("hold_budget", "Hold updates"),
                       ("extension_steps", "Extension updates"), ("recovery_deadline", "Recovery deadline in updates")):
        if key in evaluation:
            notes.append(f"{label}: {evaluation[key]}.")
    if evaluation.get("kind") == "paired_adaptation":
        notes.append("Active quality must hold before the shift and throughout the post-deadline window; "
                     "the matched frozen control must have zero passing post-deadline checks.")
    guards = evaluation.get("guards", {})
    if guards:
        notes.append("Execution guards: " + "; ".join(
            key.replace("_", " ") + " = " + (", ".join(value) if isinstance(value, list) else str(value))
            for key, value in guards.items()) + ".")
    return notes


def _execution_recorded(status, receipt):
    """Distinguish a recorded run from preflight rejection or missing evidence."""
    if status in COMPLETE:
        return True  # The immutable selected row already records a measurement.
    result = receipt.get("result", {})
    cost = result.get("cost", {})
    return (result.get("raw_status") == "completed"
            or any(isinstance(cost.get(key), (int, float)) and cost[key] > 0
                   for key in ("execution_seconds", "wall_seconds")))


def _count(tasks):
    statuses = Counter(task["status"] for task in tasks)
    return {"passed": statuses.get("PASS", 0), "total": len(tasks),
            "incomplete": any(not task["execution_recorded"] for task in tasks),
            "counts": dict(sorted(statuses.items()))}


def _sum(counts):
    statuses = Counter()
    for count in counts:
        statuses.update(count["counts"])
    return {"passed": sum(c["passed"] for c in counts), "total": sum(c["total"] for c in counts),
            "incomplete": any(c["incomplete"] for c in counts), "counts": dict(sorted(statuses.items()))}


def score(count, *, marker=True):
    return f"{count['passed']}" + ("(*)" if marker and count["incomplete"] else "") + f"/{count['total']}"


def _definition_label(result):
    return {"matches": "matches recorded run", "CHANGED": "changed since run",
            "unbound": "recorded definition unavailable"}[result["current_contract"]]


def _same_active_runtime(left, right):
    """Match served hardware/software while retaining separate cohort identities."""
    backend = left.get("execution_backend")
    runtime = left.get("runtime")
    profile = left.get("compute_profiles", {}).get(backend)
    return (backend in {"cpu", "cuda"} and backend == right.get("execution_backend")
            and isinstance(runtime, dict) and bool(runtime) and runtime == right.get("runtime")
            and isinstance(profile, dict) and bool(profile)
            and profile == right.get("compute_profiles", {}).get(backend))


def build_progress(root: Path, publication: dict) -> dict:
    """Project current view placement over immutable selected task outcomes."""
    root = Path(root)
    inputs = {}

    def load(path):
        inputs[path.as_posix()] = file_hash(root / path)
        return read_json(root / path)

    views = [load(path.relative_to(root)) for path in sorted((root / "configs/forge/views").glob("*.json"))]
    task_files = sorted([*(root / "configs/forge/tasks").glob("*.json"),
                         *(root / "configs/forge/task-variants").rglob("*.json")])
    tasks = {path.stem: load(path.relative_to(root)) for path in task_files}
    if len(tasks) != len(task_files):
        raise ValueError("family report task declarations have duplicate ids")
    task_paths = {path.stem: path.relative_to(root).as_posix() for path in task_files}
    from .scoped_publications import load_publications
    attachments = load_publications(root, load)
    scoped_publications, media = attachments["cohorts"], attachments["media"]
    inputs.update({item["gif"]: item["gif_sha256"] for item in media.values()})
    for view in views:
        identifier(view["id"], "view")
        names = [assignment["task"] for assignment in view["assignments"]]
        if len(names) != len(set(names)):
            raise ValueError("family report view has duplicate assignments")
        if tasks and set(names) - tasks.keys():
            raise ValueError("family report view references a missing task")
    scoped = [view for view in views if view.get("reporting", {}).get("family_totals") is False]
    ordinary = [view for view in views if view not in scoped
                and view.get("evidence_scope") not in {"research_diagnostic", "calibration_diagnostic"}
                and any(a["importance"] == "required" for a in view["assignments"])]
    diagnostics = [view for view in views if view not in ordinary and view not in scoped]
    families = {}
    selected_rows = publication["rows"] + publication.get("historical_family_rows", [])
    for row_index, selected in enumerate(selected_rows):
        family_id = identifier(selected.get("trainer_family", selected["candidate_id"]), "trainer family")
        family = families.setdefault(family_id, {"id": family_id, "label": selected["technique"],
                                                "page": (FAMILY_DIRECTORY / (family_id + ".md")).as_posix(), "cohorts": [],
                                                "historical_only": row_index >= len(publication["rows"])})
        backend = selected.get("runtime_cohort", {}).get("execution_backend", "unrecorded")
        cohort_anchor = "cohort-" + backend + "-" + stable_hash(selected.get("runtime_cohort", {}))[:12]
        if any(cohort["anchor"] == cohort_anchor for cohort in family["cohorts"]):
            raise ValueError("family report cannot pool multiple selected configurations in one runtime")
        recorded = {task["task_id"]: task for task in selected.get("tasks", [])}
        recorded.update({task["task_id"]: task for task in selected.get("nonrequired_tasks", [])})
        bindings = dict(selected.get("bindings", {}).get("task_contracts", {}))
        catalogs = dict(publication.get("task_contracts", {}))
        reason_catalog = dict(publication.get("status_reasons", {}))
        separate = [item for item in scoped_publications if item["family"] == family_id
                    and item["row"]["candidate_id"] == selected["candidate_id"]
                    and item["row"].get("candidate_revision") == selected.get("candidate_revision")
                    and _same_active_runtime(item["row"].get("runtime_cohort", {}), selected.get("runtime_cohort", {}))
                    and item["row"]["bindings"]["source_digest"] == selected.get("bindings", {}).get("source_digest")]
        scoped_names = {assignment["task"] for view in scoped for assignment in view["assignments"]}
        scoped_attempts = []
        final_measurements = {task: measured for (family, task), measured in attachments["final_measurements"].items()
                              if family == family_id and measured["candidate_id"] == selected["candidate_id"]
                              and measured["candidate_revision"] == selected.get("candidate_revision")
                              and measured["source_digest"] == selected.get("bindings", {}).get("source_digest")}
        for item in separate:
            measured = item["row"].get("tasks", []) + item["row"].get("nonrequired_tasks", [])
            if any(result["task_id"] not in scoped_names or result["task_id"] in recorded for result in measured):
                raise ValueError("separate cohort results cannot replace selected parent cells")
            recorded.update({result["task_id"]: result for result in measured})
            bindings.update(item["row"]["bindings"].get("task_contracts", {}))
            catalogs.update(item["task_contracts"])
            reason_catalog.update(item["status_reasons"])
            scoped_attempts.extend(item["row"].get("attempt_ids", []))
        receipts = {}
        for attempt in selected.get("attempt_ids", []) + scoped_attempts:
            identifier(attempt, "attempt")
            path = Path("reports/forge/technique-receipts") / (attempt + ".json")
            summary = load(path)
            if (summary.get("candidate_id") != selected["candidate_id"]
                    or summary.get("candidate_revision") != selected.get("candidate_revision")
                    or summary.get("provenance", {}).get("source_digest") != selected.get("bindings", {}).get("source_digest")):
                raise ValueError("family report receipt differs from the selected configuration/source")
            for result in summary.get("task_results", []):
                # Grades belong to the selected scientific row. Compact metrics
                # enrich its navigation only; a receipt never creates a pass.
                if result["task_id"] in recorded:
                    final = final_measurements.get(result["task_id"])
                    if final and final["attempt_id"] != attempt:
                        continue
                    prior = receipts.get(result["task_id"])
                    if prior and prior["result"] != result:
                        raise ValueError("family report has conflicting compact task results")
                    receipts[result["task_id"]] = {"path": path.as_posix(), "attempt_id": attempt, "result": result}

        results = {}
        assigned = {assignment["task"] for view in views for assignment in view["assignments"]}
        for name in sorted(assigned):
            previous = recorded.get(name, {})
            contract_hash = bindings.get(name)
            contract = catalogs.get(contract_hash, {})
            task = tasks.get(name, {})
            match = "unbound"
            changed = []
            if task and contract:
                if contract.get("execution_sha256") != task_execution_fingerprint(task):
                    changed.append("execution (host, recipe binding, prior, initialization or budget)")
                if contract.get("evaluation_sha256") != task_evaluation_fingerprint(task):
                    changed.append("evaluation (gates or sampling law)")
                if contract.get("timeout_seconds") != task.get("resources", {}).get("timeout_seconds"):
                    changed.append("timeout reservation")
                match = "CHANGED" if changed else "matches"
            receipt = receipts.get(name, {})
            reasons = reason_catalog.get(previous.get("reasons_sha256"), [])
            status = previous.get("status", "UNKNOWN")
            reason = ("; ".join(previous.get("reasons", reasons)) or previous.get("reason")
                      or receipt.get("result", {}).get("reason"))
            if not reason:
                reason = ("; ".join(selected.get("blockers", [])) if status == "BLOCKED" else None)
            reason = reason or {"UNKNOWN": "No recorded result for this selected configuration and source.",
                                "NOT_RUN": "This task was not executed for the selected configuration and source.",
                                "BLOCKED": "The frozen selected cohort lacks task support; no scientific measurement was admitted.",
                                "INVALID": "The recorded execution violated its frozen contract; it supplies no valid measurement.",
                                "ERROR": "The recorded attempt encountered an execution error; see its receipt for the cause.",
                                "INCOMPLETE": "The recorded execution or required observations did not complete."}.get(
                                    status, "Recorded scientific gate " + status + ".")
            coverage = ("Test definition changed since this run: " + "; ".join(changed) + ". Earlier verdict preserved." if changed else
                        "Recorded test definition unavailable; compatibility with today's declaration cannot be checked." if match == "unbound" else
                        "Test definition matches the recorded conditions.")
            results[name] = {"task_id": name, "status": previous.get("status", "UNKNOWN"),
                             "execution_recorded": _execution_recorded(status, receipt),
                             "attempt_id": receipt.get("attempt_id"),
                             "current_contract": match, "receipt": receipt.get("path"),
                             "metrics": deepcopy(receipt.get("result", {})), "recorded_contract": contract_hash,
                             "recorded_conditions": {key: deepcopy(contract.get(key, {})) for key in ("prior", "sampling")},
                             "reason": reason, "coverage_reason": coverage, "changed_contract_fields": changed,
                             "device": receipt.get("result", {}).get("device", receipt.get("result", {}).get("cost", {}).get("device")),
                             "policy_parent": deepcopy(task.get("policy_parent"))}
            matching_media = [item for (family, task_id, attempt), item in media.items()
                              if family == family_id and task_id == name
                              and attempt in selected.get("attempt_ids", []) + scoped_attempts
                              and (name not in final_measurements or attempt == final_measurements[name]["attempt_id"])
                              and item["recorded_grade"] == status]
            if matching_media:
                results[name]["training_media"] = matching_media
            audit = attachments["clock_audits"].get((family_id, name))
            if (audit and name in final_measurements
                    and audit["attempt_id"] in selected.get("attempt_ids", []) + scoped_attempts
                    and audit["recorded_grade"] == status):
                results[name]["clock_audit"] = audit

        def view_row(view):
            tiers = {}
            assignments = sorted(view["assignments"], key=lambda a: (a["qualification_tier"], a.get("order", 0), a["task"]))
            for tier in TIERS:
                required = [results[a["task"]] for a in assignments if str(a["qualification_tier"]) == tier
                            and a["importance"] == "required"]
                tiers[tier] = _count(required)
            return {"id": view["id"], "revision": view["revision"], "tiers": tiers,
                    "total": _sum(list(tiers.values()))}

        view_rows = [view_row(view) for view in ordinary]
        family["cohorts"].append({"anchor": cohort_anchor, "backend": backend,
                                   "row_index": row_index, "tasks": results, "views": view_rows,
                                   "scoped_views": [view_row(view) for view in scoped],
                                   "scoped_publications": [{**{key: item[key] for key in ("path", "view", "source_commit")},
                                                            "cohort": item["row"].get("cohort"),
                                                            "runtime_cohort_sha256": stable_hash(item["row"].get("runtime_cohort", {}))}
                                                           for item in separate],
                                   "tiers": {tier: _sum([view["tiers"][tier] for view in view_rows]) for tier in TIERS},
                                   "total": _sum([view["total"] for view in view_rows])})
    documented, tag_definitions = family_documentation.load_family_documentation(root, families, load)
    for family in families.values():
        family.update(documented.get(family["id"], {"tags": [], "documentation": None}))
        illustration = (family.get("documentation") or {}).get("illustration")
        if illustration is not None:
            inputs[illustration["path"]] = file_hash(root / illustration["path"])
    atlas_navigation = _atlas_evidence_navigation(root, load)
    next_steps = Path("reports/forge/atlas-inventory-next-steps-20261005.md")
    if atlas_navigation is not None and (root / next_steps).is_file():
        inputs[next_steps.as_posix()] = file_hash(root / next_steps)
        atlas_navigation["next_steps"] = {"readout": next_steps.as_posix(),
                                          "sha256": inputs[next_steps.as_posix()]}
    first_full_atlas = _full_original_atlas_first_case(root, load)
    progress = {"schema_version": 1, "scope": "recorded_view_progress", "qualification_input": False,
                "qualification_reuse": False, "views": views, "diagnostic_views": [view["id"] for view in diagnostics],
                "scoped_views": [view["id"] for view in scoped],
                "families": [family for family in families.values() if not family["historical_only"]],
                "historical_families": [family for family in families.values() if family["historical_only"]],
                "tag_definitions": tag_definitions,
                "task_paths": task_paths,
                "input_hashes": dict(sorted(inputs.items())),
                "renderer_sha256": file_hash(Path(__file__)),
                "documentation_loader_sha256": file_hash(Path(family_documentation.__file__))}
    if first_full_atlas is not None:
        progress["full_original_atlas_common26"] = {**first_full_atlas, "archived_snapshot": True,
                                                   "counts_scope": "AT_FIRST_ACCEPTED_CASE"}
    if atlas_navigation is not None:
        progress["atlas_evidence_navigation"] = atlas_navigation
    return progress


def _table_header():
    return ["| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |",
            "| --- | ---: | ---: | ---: | ---: |"]


def _score_cells(root, page, family, cohort, counts, *, view_id=None, bold=False):
    base = cohort["anchor"] + ("-" + view_id if view_id else "")
    cells = []
    for tier in (*TIERS, "total"):
        count = counts["total"] if tier == "total" else counts["tiers"][tier]
        target = base + ("-tier-" + tier if tier != "total" else "")
        text = link(root, page, score(count), family["page"], target)
        cells.append("**" + text + "**" if bold else text)
    return cells


def render_leaderboard(root: Path, publication: dict, page: Path) -> str:
    progress = publication["family_progress"]
    lines = ["# Forge family leaderboard", "",
             "Recorded passes / required experiments, grouped by family and view. Click a family for its technique, "
             "pseudocode and training details; click any count for the experiment results. "
             "Each family/runtime uses one complete selected configuration and source.", "",
             "Family totals sum the view rows. A shared experiment counts once per view requiring it; "
             "these totals measure requirements across views, not unique training runs or scientific rank.", "",
             *_table_header()]
    for family in progress["families"]:
        for cohort in family["cohorts"]:
            name = family["label"]
            if cohort["backend"] != "cuda" or len(family["cohorts"]) > 1:
                name += " (" + cohort["backend"] + ")"
            cells = ["**" + link(root, page, name, family["page"]) + "**"]
            cells += _score_cells(root, page, family, cohort, cohort, bold=True)
            lines.append("| " + " | ".join(cells) + " |")
            for view in cohort["views"]:
                label = "↳ " + link(root, page, view["id"], family["page"], cohort["anchor"] + "-" + view["id"])
                lines.append("| " + " | ".join([label, *_score_cells(root, page, family, cohort, view, view_id=view["id"])]) + " |")
    lines += ["", "Runtime cohorts and actual per-task devices are recorded on the family pages and in receipt provenance.", "",
              "(*) means at least one required experiment has no recorded execution for the selected configuration and source, "
              "including preflight-blocked tests. Recorded PASS and FAIL are both executed results. Attempted ERROR, INVALID "
              "or INCOMPLETE results show their cause and execution evidence on the family page and earn no pass credit. "
              "Zero recorded passes always displays as 0, including unrun families.", "",
              "Counts retain recorded verdicts under their original recipe, prior, initialization, budget, serving law "
              "and source. Changes to today's test definition and unavailable recorded definitions are shown separately "
              "on the family pages and do not add (*); recorded passes grant no "
              "new qualification. Required lower tiers must pass before later work is eligible. "
              "The declared calibration and eligibility requirements appear on each family page.", "",
              "Separately scoped cohort views, diagnostic-only views and historical/API studies are available on the "
              "family pages and excluded from totals.", ""]
    if {family["id"] for family in progress["families"]} & {"atlas", "e22"}:
        lines += ["Atlas/E22 retain (*) for blocked, unrun tests in their selected main-table cohort. "
                  "The family pages list the host, prior, serving and parameter-ownership blockers. "
                  "Their separately scoped policy measurements keep their own results and do not fill these cells.", ""]
    lines += _tag_directory(root, page, progress)
    lines += ["## Refresh", "", "```sh", "python reports/forge/regenerate_technique_inventory.py", "```", "",
              "This regenerates the leaderboard, family pages and experiments-by-tier report from committed evidence "
              "and declarations. Register newly measured evidence with `--source-commit <executed-commit>`; "
              "advancing the recorded view policy also requires `--advance-policy`.", "",
              link(root, page, "Experiments, criteria and tier assignments", "reports/forge/EXPERIMENTS_BY_TIER.md") + " · " +
              link(root, page, "Complete numerical publication and provenance", "reports/forge/technique-inventory.json"), "",
              f"Publication input digest `{publication['provenance']['input_digest']}`.", ""]
    return "\n".join(lines)


def _tag_directory(root, page, progress):
    families = progress["families"] + progress.get("historical_families", [])
    used = {tag for family in families for tag in family.get("tags", [])}
    if not used:
        return []
    lines = ["## Browse by technique tag", "",
             "Tags describe properties shared by a family's supported variants, not performance or qualification. "
             "Configuration-dependent behavior is explained on each family page. Historical pages retain their own scope.", ""]
    for tag in sorted(used):
        lines += [f'<a name="tag-{tag}"></a>', "", f"### `{tag}`", "",
                  progress["tag_definitions"][tag], ""]
        for family in sorted(families, key=lambda item: (item["label"], item["id"])):
            if tag in family.get("tags", []):
                lines.append("- " + link(root, page, family["label"], family["page"]) +
                             (" (historical cohort)" if family.get("historical_only") else ""))
        lines.append("")
    return lines


def _technique_description(root, page, family):
    documentation = family.get("documentation")
    if documentation is None:
        return ["## Technique overview", "",
                "Maintained technique documentation is unavailable in this historical or synthetic report source. "
                "Consult the recorded recipe and source bindings below; no training behavior is inferred from its name.", ""]
    lines = ["**Tags:** " + " · ".join(link(root, page, tag, "reports/forge/technique-inventory.md", "tag-" + tag)
                                       for tag in family["tags"]), "",
             "## Technique overview", "", documentation["overview"], ""]
    illustration = documentation.get("illustration")
    if illustration is not None:
        alt = illustration["alt"].replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")
        target = os.path.relpath(root / illustration["path"], page.parent)
        if any(character.isspace() for character in target):
            target = "<" + target + ">"
        lines += [f"![{alt}]({target})", "", illustration["caption"], ""]
    equations = documentation.get("equations", [])
    if equations:
        lines += ["## Mathematical formulation", "", documentation["symbols"], ""]
        for equation in equations:
            lines += ["**" + equation["label"] + "**", "", "$$", equation["latex"], "$$", "",
                      equation["explanation"], ""]
        lines += ["## Simplified pseudocode", ""]
    else:
        lines += ["## Simplified pseudocode", "", documentation["symbols"], ""]
    lines += ["```text",
             *documentation["pseudocode"], "```", "",
             "## Training details", "", "| Characteristic | Behavior |", "| --- | --- |"]
    lines += ["| " + label + " | " + cell(documentation["training_details"][key]) + " |"
              for key, label in family_documentation.TRAINING_DETAILS]
    lines += ["", "## Configuration differences", ""]
    lines += ["- " + note for note in documentation["configuration_notes"]]
    lines += ["", "<details>", "<summary>Implementation and recipe sources</summary>", "",
              "These links support the explanation. Recorded results below remain bound to their own executed source.", ""]
    for source in documentation["sources"]:
        lines.append("- " + (f"[{source}]({source})" if "://" in source else link(root, page, source, source)))
    lines += ["", "</details>", ""]
    return lines


def _technique_references(family):
    documentation = family.get("documentation")
    if documentation is None:
        return []
    lines = ["## References", ""]
    if not documentation["references"]:
        lines += ["This technique is repository-specific; no dedicated paper reference is declared. "
                  "Its implementation and recipe sources are linked above.", ""]
    for reference in documentation["references"]:
        lines.append("- [" + reference["title"] + "](" + reference["url"] + "). " + reference["component"])
    if documentation["references"]:
        lines.append("")
    return lines


def _existing_link(root, page, label, target):
    return link(root, page, label, target) if target and (root / target).is_file() else cell(label)


def _current_gate(evaluation):
    bounds = evaluation_rows(evaluation)
    lines = ["| Metric | Required bound |", "| --- | --- |"] if bounds else []
    lines += [f"| {cell(name)} | {cell(op)} {number(bound)} |" for name, op, bound in bounds]
    if lines:
        lines.append("")
    lines += evaluation_notes(evaluation)
    return lines


def _heading(level, title, anchor):
    return [f'<a name="{anchor}"></a>', "", "#" * level + " " + title, ""]


def _metric_values(metrics, prefix=""):
    for name, value in sorted(metrics.items()):
        key = prefix + name
        if isinstance(value, dict):
            yield from _metric_values(value, key + ".")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            yield key, value


def render_family(root: Path, publication: dict, family: dict) -> str:
    page = root / family["page"]
    progress = publication["family_progress"]
    views = {view["id"]: view for view in progress["views"]}
    lines = ["<!-- Generated Forge family report -->", "", f"# {family['label']}", "",
             link(root, page, "← Family leaderboard", "reports/forge/technique-inventory.md"), "",
             *_technique_description(root, page, family),
             "Generated from one selected configuration per runtime. Recorded verdicts retain their original scientific "
             "contracts; grouping them under current views grants no new qualification.", ""]
    if family.get("historical_only"):
        lines += ["**Historical cohort navigation.** This original recipe/prior row is retained for existing links. " +
                  link(root, page, "Current GAN v3 solution family", FAMILY_DIRECTORY / "release07-gan-v3.md") +
                  " uses one whole selected configuration and task-declared priors; these historical cells are not pooled into it.", ""]
    if family["id"] == "atlas":
        lines += _full_original_atlas_status(root, page, progress)
    for cohort in family["cohorts"]:
        row = (publication["rows"] + publication.get("historical_family_rows", []))[cohort["row_index"]]
        base = cohort["anchor"]
        config = Path("configs/forge/configurations") / (row["candidate_id"] + ".json")
        if not (root / config).is_file():
            config = Path("configs/forge/ideas") / (row["candidate_id"] + ".json")
        label = row["candidate_id"].rsplit("--", 1)
        configuration_label = " · ".join([label[0], label[1][:12]]) if len(label) == 2 else label[0]
        source = row.get("bindings", {}).get("source_digest")
        pointer = publication.get("evidence_sources", {}).get(row.get("publication_key"), {})
        # publish_current validates this snapshot's complete identity before
        # rendering, then writes a pending snapshot alongside the pages. Its
        # first-write link must not depend on that later filesystem mutation.
        frozen_evidence = (link(root, page, "Frozen numerical evidence", pointer["snapshot"])
                           if pointer.get("snapshot") and pointer.get("json_sha256") else
                           _existing_link(root, page, "Frozen numerical evidence", pointer.get("snapshot")))
        lines += [*_heading(2, cohort["backend"].upper() + " results", base), f"Runtime: **{cohort['backend']}**. Selected configuration: " +
                  _existing_link(root, page, configuration_label, config) + ".", "",
                  f"Recorded qualification: **tier {row.get('qualified_tier', 0)}**, "
                  f"{publication['view']} revision {publication['view_revision']}. "
                  "Other view rows below are navigation over recorded task evidence, not recomputed qualification.", "",
                  "<details>", "<summary>Configuration, source and runtime provenance</summary>", "",
                  f"Source digest: `{source or 'unbound'}`. Candidate revision: `{row.get('candidate_revision') or 'unbound'}`. "
                  f"Runtime cohort: `{row.get('cohort') or 'unbound'}`.", "",
                  frozen_evidence + " · " +
                  link(root, page, "Complete recipe, prior, initialization and sampling bindings", "reports/forge/technique-inventory.json"), "",
                  "Selection: " + cell(row.get("selection", {}).get("selection_kind", "canonical fallback")) + ". " +
                  cell(row.get("selection", {}).get("reason", "No qualified configuration has been selected.")), "",
                  "</details>", "", *_table_header()]
        if row.get("selection", {}).get("measurement_complete"):
            lines[-2:-2] = ["Complete current Tier 1 measurement in: " +
                            ", ".join(row["selection"]["measurement_views"]) +
                            ("; additional scoped probes: " + ", ".join(row["selection"]["measurement_tasks"])
                             if row["selection"].get("measurement_tasks") else "") +
                            ". PASS and FAIL are both measured outcomes; other cohorts retain their own required cells.", ""]
        for view in cohort["views"]:
            lines.append("| " + " | ".join([link(root, page, view["id"], family["page"], base + "-" + view["id"]),
                                            *_score_cells(root, page, family, cohort, view, view_id=view["id"])]) + " |")
        if cohort.get("scoped_views"):
            lines += ["", "Separate cohort coverage (excluded from family totals):", "", *_table_header()]
            for view in cohort["scoped_views"]:
                lines.append("| " + " | ".join([link(root, page, view["id"], family["page"], base + "-" + view["id"]),
                                                *_score_cells(root, page, family, cohort, view, view_id=view["id"])]) + " |")
            for item in cohort.get("scoped_publications", []):
                lines += ["", link(root, page, "Frozen separate-cohort numerical evidence", item["path"]) +
                          "; source commit `" + item["source_commit"] + "`; exact scoped cohort `" +
                          cell(item.get("cohort")) + "` (runtime SHA256 `" + cell(item.get("runtime_cohort_sha256")) +
                          "`). These cells give no parent-cohort credit."]
        lines += ["", "(*) means at least one required experiment has no recorded execution, including preflight blockers. "
                  "PASS and FAIL both count as executed. Attempted errors retain their status and cause; "
                  "test-definition compatibility is shown separately and does not add (*).", ""]
        for tier in TIERS:
            members = {}
            for definition in progress["views"]:
                if definition["id"] in progress["diagnostic_views"] + progress.get("scoped_views", []):
                    continue
                for assignment in definition["assignments"]:
                    if str(assignment["qualification_tier"]) == tier and assignment["importance"] == "required":
                        members.setdefault(assignment["task"], []).append(definition["id"])
            lines += [*_heading(3, "Tier " + tier + " across views", base + "-tier-" + tier),
                      "Shared experiments appear once in this list; the family numerator/denominator count their view requirements.", "",
                      "| Experiment | Required by | Recorded result | Test definition |", "| --- | --- | --- | --- |"]
            for name, memberships in sorted(members.items()):
                result = cohort["tasks"][name]
                lines.append("| " + " | ".join([
                    link(root, page, name, family["page"], base + "-experiment-" + name),
                    ", ".join(link(root, page, view_id, family["page"], base + "-" + view_id + "-tier-" + tier)
                              for view_id in memberships), result["status"], _definition_label(result)]) + " |")
            lines.append("")
        for definition in progress["views"]:
            view_id = definition["id"]
            anchor = base + "-" + view_id
            diagnostic = view_id in progress["diagnostic_views"]
            separate_cohort = view_id in progress.get("scoped_views", [])
            lines += [*_heading(2, view_id, anchor), "**" + view_id + f" — revision {definition['revision']}**. " +
                      _existing_link(root, page, "View declaration", Path("configs/forge/views") / (view_id + ".json")) + ".", ""]
            if diagnostic:
                lines += ["Diagnostic-only view; its outcomes are excluded from the family totals and grant no qualification.", ""]
            else:
                if separate_cohort:
                    lines += ["Separately scoped cohort. This ordinary lane retains its own required gates and execution "
                              "policy; its measurements are excluded from family totals and give no parent-cohort credit.", ""]
                requirements = [sum(a["importance"] == "required" and str(a["qualification_tier"]) == tier
                                    for a in definition["assignments"]) for tier in TIERS]
                lines += ["Qualification requires every required experiment to pass, with all lower tiers and "
                          f"task dependencies passed first. Required counts (Tier 1 / 2 / 3): **{' / '.join(map(str, requirements))}**.", ""]
            calibration = definition.get("calibration", {})
            lines += ["Calibration: **" + cell(calibration.get("status", "undeclared")) + "**. " +
                      cell(calibration.get("adoption_blocker", "Calibration and robustness are separate from recorded task passes.")), ""]
            eligibility = definition.get("eligibility", {})
            if eligibility:
                lines += ["Additional eligibility requirements:", ""]
                lines += ["- Capability: " + cell(name) for name in eligibility.get("requires_capabilities", [])]
                lines += [f"- Claim {cell(name)}: {cell(value)}" for name, value in eligibility.get("claim_contract", {}).items()]
                lines.append("")
            for tier in TIERS:
                assignments = sorted((a for a in definition["assignments"] if str(a["qualification_tier"]) == tier),
                                     key=lambda a: (a.get("order", 0), a["task"]))
                lines += _heading(3, "Tier " + tier, anchor + "-tier-" + tier)
                if not assignments:
                    lines += ["No experiments assigned.", ""]
                    continue
                lines += ["| Experiment | Role | Recorded result | Test definition |", "| --- | --- | --- | --- |"]
                for assignment in assignments:
                    name = assignment["task"]
                    result = cohort["tasks"][name]
                    lines.append("| " + " | ".join([
                        link(root, page, name, family["page"], base + "-experiment-" + name), assignment["importance"],
                        result["status"], _definition_label(result)]) + " |")
                lines.append("")
        lines += [*_heading(2, "Experiment metrics and pass criteria", base + "-experiments"), "One evidence entry per experiment is shared by its view rows. "
                  "Test-definition changes describe differences from the recorded run, independently of whether it was executed. "
                  "Earlier verdicts are preserved.", ""]
        for name, result in cohort["tasks"].items():
            task_path = Path(progress.get("task_paths", {}).get(name, "configs/forge/tasks/" + name + ".json"))
            task = read_json(root / task_path) if (root / task_path).is_file() else {}
            lines += [*_heading(3, name, base + "-experiment-" + name), f"**{name}: {result['status']}**. " +
                      _existing_link(root, page, "Current experiment declaration", task_path) + ".", "",
                      "Test definition: **" + _definition_label(result) + "**. " + cell(result["coverage_reason"]) +
                      " " + cell(result["reason"]), ""]
            if result["status"] not in COMPLETE:
                lines += ["Execution: **" + ("recorded" if result["execution_recorded"] else "no recorded execution (*)") + "**."]
                if result["attempt_id"]:
                    lines.append("Attempt: `" + result["attempt_id"] + "`; raw outcome: " +
                                 cell(result["metrics"].get("raw_status", "unavailable")) + ".")
                    cost = result["metrics"].get("cost", {})
                    lines.append("Recorded execution seconds: " + number(cost.get("execution_seconds")) +
                                 "; charged wall seconds: " + number(cost.get("wall_seconds")) +
                                 " (shared execution is charged once to " + cell(cost.get("charged_task", name)) + ").")
                lines.append("")
            if result["device"]:
                lines += ["Actual task device: `" + cell(result["device"]) + "` (recorded execution receipt).", ""]
            if result["policy_parent"]:
                parent = result["policy_parent"]
                parent_id = parent.get("id", "unbound")
                lines += ["Policy-cohort variant of " + _existing_link(root, page, parent_id,
                           Path("configs/forge/tasks") / (parent_id + ".json")) +
                          "; parent task SHA256 `" + cell(parent.get("task_sha256")) +
                          "`. This measurement supplies no cells to the parent clean cohort.", ""]
            memberships = [link(root, page, view["id"] + " / Tier " + str(a["qualification_tier"]), family["page"],
                                base + "-" + view["id"] + "-tier-" + str(a["qualification_tier"]))
                           for view in progress["views"] for a in view["assignments"] if a["task"] == name]
            lines += ["Used by: " + " · ".join(memberships) + ".", ""]
            measured = result["metrics"]
            checks = measured.get("evaluator_summary", {}).get("metric_checks", {})
            if checks:
                lines += ["Recorded final metric checks:", "", "| Metric | Measured | Recorded bound | Recorded check |",
                          "| --- | ---: | --- | --- |"]
                lines += [f"| {cell(metric)} | {number(check.get('value'))} | {cell(check.get('op'))} "
                          f"{number(check.get('threshold'))} | {cell(check.get('status'))} |" for metric, check in sorted(checks.items())]
                lines.append("")
            elif measured.get("metrics"):
                values = measured["metrics"]
                lines += ["Recorded final metrics:", "", "| Metric | Measured |", "| --- | ---: |"]
                lines += [f"| {cell(metric)} | {number(value)} |" for metric, value in _metric_values(values)]
                lines.append("")
            convergence = measured.get("evaluator_summary", {}).get("convergence", {})
            if "passing_suffix" in convergence:
                lines += [f"Recorded terminal passing observations: **{convergence['passing_suffix']}**; "
                          f"required: {convergence.get('minimum_stable_checks', 'unavailable')}.", ""]
            if result["receipt"]:
                lines += [link(root, page, "Compact metrics and receipt provenance", result["receipt"]), ""]
            for artifact in result.get("training_media", []):
                lines += [link(root, page, "Actual-training GIF", artifact["gif"]) +
                          "; " + str(artifact["observation_count"]) + " saved observations; "
                          "no new optimizer updates or sampling draws.", ""]
            if result.get("clock_audit"):
                audit = result["clock_audit"]
                lines += ["Recorded clock parity diagnostics:", "", "| Condition | Exact state digest equality |",
                          "| --- | --- |"]
                lines += ["| " + cell(condition) + " | " + ("equal" if check["digest_equal"] else "different") + " |"
                          for condition, check in audit["comparisons"].items()]
                lines += ["", "Recorded unexplained clock dependencies: **" + str(audit["clock_dependency_count"]) + "**.", ""]
                lines += ["- " + cell(dependency) for dependency in audit["unexplained_clock_dependencies"]]
                lines += ["", link(root, page, "Certified parity digests and source audit", audit["path"]) +
                          ". These display diagnostics preserve the recorded gate " + cell(audit["recorded_grade"]) + ".", ""]
            if result["recorded_contract"]:
                old = publication.get("task_contracts", {}).get(result["recorded_contract"], result.get("recorded_conditions", {}))
                sampling = old.get("sampling", {})
                prior = old.get("prior", {})
                lines += ["Recorded conditions: " + cell(prior.get("kind", "unbound")) +
                          f" prior (sigma {number(prior.get('sigma'))}); " + cell(sampling.get("sampling_law", "unbound")) +
                          "; weights " + cell(sampling.get("scoring_weights", "unbound")) +
                          "; output noise " + cell(sampling.get("eval_output_noise", "unbound")) + ".", ""]
            execution = task.get("execution", {})
            budget = execution.get("max_total_steps", execution.get("steps", "unavailable"))
            evaluation = task.get("evaluation", {})
            prior = execution.get("prior", {})
            lines += ["Current pass criteria:", "", *_current_gate(task.get("evaluation", {})), "",
                      f"Declared budget: {budget} updates; timeout {task.get('resources', {}).get('timeout_seconds', 'unavailable')} seconds.", "",
                      "Current measurement: " + cell(prior.get("kind", "unbound")) +
                      f" prior (sigma {number(prior.get('sigma'))}); " + cell(evaluation.get("sampling_law", "unbound")) +
                      "; weights " + cell(evaluation.get("scoring_weights", "unbound")) +
                      "; output noise " + cell(evaluation.get("eval_output_noise", "unbound")) + ".", ""]
            if task.get("dependencies"):
                lines += ["Dependencies: " + ", ".join(dep["task"] + " (" + dep["kind"] + ")" for dep in task["dependencies"]) + ".", ""]
            explanation = task.get("research_artifacts", {}).get("readout")
            if explanation:
                lines += [_existing_link(root, page, "Explanation and existing training artifacts", explanation), ""]
    lines += ["## Historical and diagnostic evidence", "",
              "Separate configurations, API variants and serving laws retain their own scopes and supply no cells above.", ""]
    if family["id"] == "release07-gan-v3":
        for previous in progress.get("historical_families", []):
            if previous["id"].startswith("release07-gan-v3-"):
                lines.append("- " + link(root, page, previous["label"] + " original cohort", previous["page"]))
    if family["id"] == "atlas":
        original = publication.get("original_pr223_atlas", {})
        for label, record in [("Original Atlas recipe and serving-law evidence", original), *[(name, original.get(name, {})) for name in
                              ("fresh_retest", "native3_continuation", "native3_repaired_continuation")]]:
            if record.get("readout"):
                lines.append("- " + _existing_link(root, page, label, record["readout"]))
        progress_atlas = publication.get("atlas_unblocking_progress", {})
        for record in [progress_atlas.get("baseline", {}), *progress_atlas.get("adaptations", []), progress_atlas.get("word", {})]:
            if record.get("readout"):
                lines.append("- " + _existing_link(root, page, "Atlas diagnostic", record["readout"]))
        for key, label in (("baseline_debugging", "C6 baseline selection and retained diagnosis"),
                           ("word_half_base_diagnostic", "Word half-base rate contrast")):
            record = publication.get(key, {})
            if record.get("readout"):
                lines.append("- " + _existing_link(root, page, label, record["readout"]))
    for study in publication.get("completed_api_studies", {}).get("rows", []):
        if study.get("family_id", study.get("family")) == family["id"] or family["id"] == "atlas":
            if study.get("readout"):
                lines.append("- " + _existing_link(root, page, study.get("id", "Completed study"), study["readout"]))
    for study in publication.get("standalone_api_scores", []):
        if study.get("trainer_family") == family["id"]:
            label = "Standalone API evidence: " + family["id"] + " · " + study["case"]["title"]
            lines.append("- " + _existing_link(root, page, label, study["readout"]) + " · " +
                         _existing_link(root, page, "actual-training GIF", study["gif"]))
    lines += ["- " + link(root, page, "Other configurations and original evidence bindings", "reports/forge/technique-inventory.json"),
              "- " + link(root, page, "Compiled experiment memory", "reports/forge/EXPERIMENT_MEMORY.md"), "",
              "## Refresh", "", "```sh", "python reports/forge/regenerate_technique_inventory.py", "```", "",
              "This page is generated alongside the leaderboard. Register new source evidence before refreshing; "
              "editing a page cannot change a verdict or earn qualification.", ""]
    lines += _technique_references(family)
    return "\n".join(lines)


def generated_pages(root: Path, publication: dict) -> dict[Path, str]:
    return {root / family["page"]: render_family(root, publication, family)
            for family in (publication["family_progress"]["families"] +
                           publication["family_progress"].get("historical_families", []))}
