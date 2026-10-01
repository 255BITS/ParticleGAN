"""Display-only family and evidence provenance filters for qualification boards.

These labels never enter scientific identities, alter denominators, regrade old
results, sort rows or grant eligibility. A durable receipt certificate is not a
scientific PASS, a complete measurement, or historical protocol equivalence.
"""
from copy import deepcopy
from pathlib import Path

from .contracts import read_json, stable_hash


FAMILIES = {"transfer_behavior": "behavioral", "transfer_vector": "vector",
            "transfer_image": "image", "native100": "native100",
            "native100_continuation": "native100", "ring_endurance": "endurance",
            "paired_adaptation": "adaptation", "clockfree_audit": "clockfree"}
EVIDENCE_QUALITIES = frozenset({"certified_current", "certified_pinned", "certified_diagnostic",
                              "imported_recorded", "unmeasured", "invalid_receipt", "unverified"})
CATEGORIES = ("current_rows", "pinned_rows", "calibration_rows", "historical_rows")


def _identity(row):
    return stable_hash({key: value for key, value in row.items() if key != "display_metadata"})


def _categories(result):
    """Keep category entries referencing the same rows as the combined board."""
    indexed = {_identity(row): row for row in result["rows"]}
    for name in CATEGORIES:
        if name in result:
            result[name] = [indexed[_identity(row)] for row in result[name] if _identity(row) in indexed]


def _task_family(task, basis):
    execution = task.get("execution", {})
    declaration = execution.get("host_definition", {})
    family = FAMILIES.get(task.get("adapter"), "unknown")
    explicit = task.get("family") or declaration.get("family")
    if not isinstance(explicit, str) or not explicit.strip():
        explicit = None
    return {"family": family, "subfamily": explicit, "basis": basis,
            "adapter": task.get("adapter"), "host": execution.get("host"),
            "protocol_equivalence_asserted": False}


def _record_index(root, records):
    if records is None:
        records = [read_json(path) for path in sorted((root / "reports/forge/records").glob("*.json"))]
    return {record["record_id"]: record for record in records if record.get("record_id")}


def _annotate(row, catalog, attempts, records):
    scope = row.get("evidence_scope", "unknown")
    task_rows = row.get("qualification", {}).get("tasks", row.get("task_results", []))
    names = {item["task_id"] for item in task_rows if isinstance(item.get("task_id"), str)}
    selected = [attempts[name] for name in row.get("attempt_ids", []) if name in attempts]
    missing = sorted(set(row.get("attempt_ids", [])) - set(attempts))
    definitions, measured = {}, set()
    for attempt in selected:
        for task in attempt["task_results"]:
            name = task["task_id"]
            if name not in names:
                continue
            measured.add(name)
            if name in attempt["request"].get("tasks", {}):
                definitions.setdefault(name, []).append(attempt["request"]["tasks"][name])
    if scope == "historical":
        measured = {item["task_id"] for item in task_rows if item.get("gate_status") != "NOT_RUN"}
    task_families = {}
    for name in sorted(names | measured):
        saved = definitions.get(name, [])
        if saved:
            candidates = {_identity(_task_family(task, "recorded_request")): _task_family(task, "recorded_request")
                          for task in saved}
            if len(candidates) == 1:
                task_families[name] = next(iter(candidates.values()))
            else:
                task_families[name] = _task_family({}, "conflicting_recorded_declarations")
        elif name in catalog:
            # A historical task name may be classified for browsing, but this
            # match does not establish matching budgets, priors or evaluators.
            basis = "current_catalog" if scope == "current" else "catalog_name_only"
            task_families[name] = _task_family(catalog[name], basis)
        else:
            task_families[name] = _task_family({}, "unknown")
    families = {value["family"] for value in task_families.values()}
    subfamilies = {value["subfamily"] for value in task_families.values() if value["subfamily"]}
    record = records.get(row.get("record_id"), {})
    source = row.get("source") or record.get("source") or {}
    source_family = "unknown"
    if isinstance(source.get("path"), str):
        from .history import classify
        source_family = classify(source["path"])["family"]
        if source_family == "unclassified":
            source_family = "unknown"
    labels = set()
    valid = sorted(a["attempt_id"] for a in selected if a.get("valid_receipt") is True)
    invalid = sorted(a["attempt_id"] for a in selected if a.get("valid_receipt") is not True)
    if scope == "historical":
        if measured:
            labels.add("imported_recorded")
    else:
        certification = {"current": "certified_current", "pinned": "certified_pinned",
                         "calibration_diagnostic": "certified_diagnostic"}.get(scope)
        if valid and certification:
            labels.add(certification)
        if invalid:
            labels.add("invalid_receipt")
        if missing or (measured and not valid and not invalid) or (valid and not certification):
            labels.add("unverified")
    unmeasured = sorted(names - measured)
    if unmeasured or not measured:
        labels.add("unmeasured")
    # A rendered PASS with no backing attempts must not become certified.
    if not row.get("attempt_ids") and scope != "historical" and any(
            item.get("gate_status", item.get("status")) not in {None, "NOT_RUN", "BLOCKED"} for item in task_rows):
        labels.add("unverified")
    return {"schema_version": 1, "families": sorted(families or {"unknown"}),
            "subfamilies": sorted(subfamilies), "task_families": task_families,
            "source_family": source_family, "source_family_basis": "source_path_classification",
            "evidence_quality": sorted(labels or {"unverified"}),
            "valid_attempt_ids": valid, "invalid_attempt_ids": invalid, "missing_attempt_ids": missing,
            "unmeasured_task_ids": unmeasured,
            "certification_scope": "Durable result/source/runtime identity only; scientific verdicts and full qualification stay unchanged.",
            "historical_verification": "Imported recorded outcomes, without fresh verification or qualification." if scope == "historical" else None}


def annotate_rows(root: Path, board_result: dict, *, attempts=None, records=None) -> dict:
    """Attach display metadata; caller can reuse its already-validated attempts."""
    root = Path(root)
    result = deepcopy(board_result)
    if attempts is None:
        from .knowledge import _attempts
        attempts, _ = _attempts(root)
    indexed = {attempt["attempt_id"]: attempt for attempt in attempts}
    catalog = {task["id"]: task for path in sorted((root / "configs/forge/tasks").glob("*.json"))
               if (task := read_json(path)).get("id")}
    record_index = _record_index(root, records)
    for row in result["rows"]:
        row["display_metadata"] = _annotate(row, catalog, indexed, record_index)
    _categories(result)
    result["display_filter_options"] = {
        "families": sorted({label for row in result["rows"] for label in row["display_metadata"]["families"]}),
        "subfamilies": sorted({label for row in result["rows"] for label in row["display_metadata"]["subfamilies"]}),
        "source_families": sorted({row["display_metadata"]["source_family"] for row in result["rows"]}),
        "evidence_quality": sorted(EVIDENCE_QUALITIES)}
    return result


def _selection(value):
    if value is None:
        return set()
    values = {value} if isinstance(value, str) else set(value)
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError("display filters must be nonempty label strings")
    return values


def filter_rows(board_result: dict, *, family=None, evidence_quality=None) -> dict:
    """Filter whole rows, OR within a dimension and AND between dimensions.

    Family accepts a broad family, explicitly declared subfamily or classified
    source family. Filtering never trims task results or recomputes a score.
    """
    families, qualities = _selection(family), _selection(evidence_quality)
    if qualities - EVIDENCE_QUALITIES:
        raise ValueError("unknown evidence-quality label: " + ", ".join(sorted(qualities - EVIDENCE_QUALITIES)))
    result = deepcopy(board_result)
    original = result["rows"]
    if any("display_metadata" not in row for row in original):
        raise ValueError("annotate rows before applying family/evidence filters")
    def keep(row):
        meta = row["display_metadata"]
        labels = set(meta["families"] + meta["subfamilies"])
        if meta["source_family"] != "unknown":
            labels.add(meta["source_family"])
        return (not families or bool(labels & families)) and (not qualities or bool(set(meta["evidence_quality"]) & qualities))
    result["rows"] = [row for row in original if keep(row)]
    _categories(result)
    result["display_filter"] = {"families": sorted(families), "evidence_quality": sorted(qualities),
        "input_rows": len(original), "visible_rows": len(result["rows"]), "hidden_rows": len(original) - len(result["rows"]),
        "qualification_scope": "Full original view and task denominator; row filtering never confers eligibility or changes ordering."}
    return result
