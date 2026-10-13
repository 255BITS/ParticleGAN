"""Read-only test-by-recipe projection of Forge's published evidence.

No planning, grading, sampling, hydration or model code is invoked. Protocols
are exact comparison keys, never task-name joins or best-of-recipe scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re

from .contracts import atomic_text, file_hash, stable_hash
from .views import task_evaluation_fingerprint, task_execution_fingerprint

PUBLICATION = "reports/forge/technique-inventory.json"
PREFIX = "reports/forge/test-performance"
WITNESSES = "reports/forge/test-witnesses.json"
SUPPLEMENTAL = "reports/forge/test-performance-evidence/index.json"
STATUSES = {"PASS", "FAIL", "ERROR", "INCOMPLETE", "NOT RUN", "BLOCKED", "INVALID", "UNASSESSED", "UNCONFIRMED"}
LOCAL_PATH = re.compile(r"(?<![\w:/])/(?:ml2|home|tmp|mnt|var|Users|private)(?:/[^\s\"'<>;,)]*)?")


def portable(value):
    """Keep scientific metadata, omitting machine-local provenance strings."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return LOCAL_PATH.sub("[local reference omitted]", value)
    return value


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}
        self.cache = {}

    def read(self, name, *, sha256=None):
        import hashlib
        path = PurePosixPath(name)
        if (not isinstance(name, str) or path.is_absolute() or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("test performance: unsafe evidence path")
        file = self.root / path
        if any(parent.is_symlink() for parent in (file, *file.parents) if parent != self.root
               and parent.is_relative_to(self.root)) or not file.resolve().is_relative_to(self.root):
            raise ValueError("test performance: evidence symlink or path escape")
        if name not in self.cache:
            data = file.read_bytes()
            self.hashes[name] = hashlib.sha256(data).hexdigest()
            self.cache[name] = json.loads(data, parse_constant=lambda val: (_ for _ in ()).throw(ValueError(val)))
        if sha256 is not None and self.hashes[name] != sha256:
            raise ValueError("test performance: evidence bytes changed: " + name)
        return deepcopy(self.cache[name])


def recorded_status(outcome, receipt=None):
    receipt = receipt or {}
    raw = str(receipt.get("raw_status", "")).upper()
    if raw in {"ERROR", "EXECUTION_ERROR"}:
        return "ERROR"
    status = outcome.get("gate_status", outcome.get("status", "NOT_RUN"))
    if status in {None, "UNKNOWN", "NOT_RUN", "NOT RUN"}:
        return "NOT RUN"
    if status not in STATUSES:
        raise ValueError("test performance: unknown recorded status " + str(status))
    return status


def cell_status(records):
    statuses = {record["status"] for record in records}
    return next(iter(statuses)) if len(statuses) == 1 else "MIXED" if statuses else "NOT RUN"


def coverage(test, recipes, records):
    """Count compatible recipe cells, not summed view requirements or runs."""
    cells = [[records[key] for key in test["cells"].get(recipe, [])] for recipe in recipes]
    statuses = Counter(cell_status(cell) for cell in cells)
    passing = sum(any(record["status"] == "PASS" for record in cell) for cell in cells)
    graded = sum(any(record["status"] in {"PASS", "FAIL"} for record in cell) for cell in cells)
    witnesses = Counter(record["kind"] for record in test["witnesses"] if record["status"] == "PASS")
    return {"recipes": len(recipes), "counts": dict(sorted(statuses.items())),
            "optimizer_passes": passing, "zero_optimizer_pass": passing == 0,
            "graded_recipes": graded,
            "oracle_passes": witnesses["oracle"], "representational_passes": witnesses["representational"],
            "no_indexed_witness": passing == 0 and sum(witnesses.values()) == 0}


def _digest_check(document):
    copy = deepcopy(document)
    claimed = copy.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(copy):
        raise ValueError("test performance: publication digest mismatch")


def _receipt(inputs, row, source):
    from .family_reports import certified_retry_successors
    summaries = {}
    for attempt in row.get("attempt_ids", []):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", attempt):
            raise ValueError("test performance: unsafe attempt id")
        name = "reports/forge/technique-receipts/" + attempt + ".json"
        summary = inputs.read(name)
        provenance = summary.get("provenance", {})
        proof = source.get("provenance", {}).get("qualified_receipts", {}).get(attempt)
        actual = {"canonical_result_hash": provenance.get("canonical_result_hash"),
                  "source_digest": provenance.get("source_digest"),
                  "original_file_sha256": {key: value.get("sha256") for key, value in provenance.get("original_files", {}).items()}}
        if summary.get("retry_of"):
            actual["retry_of"] = summary["retry_of"]
        scoped = source.get("schema") == "forge_tier1_scoped_evidence_v1"
        proof_matches = (source.get("receipt_result_hashes", {}).get(attempt) == actual["canonical_result_hash"]
                         if scoped else proof == actual)
        if (not proof_matches or summary.get("certificate_validated") is not True
                or summary.get("candidate_id") != row["candidate_id"]
                or summary.get("candidate_revision") != row.get("candidate_revision")
                or provenance.get("source_digest") != row.get("bindings", {}).get("source_digest")):
            raise ValueError("test performance: receipt proof/recipe/source mismatch: " + attempt)
        summaries[attempt] = summary
    superseded = certified_retry_successors(summaries)
    final = {item["task_id"]: item["attempt_id"] for item in source.get("final_measurements", [])
             if item["candidate_id"] == row["candidate_id"]
             and item.get("candidate_revision") == row.get("candidate_revision")}
    for measured in source.get("final_measurements", []):
        if measured["task_id"] not in final or final[measured["task_id"]] != measured["attempt_id"]:
            continue
        summary = summaries[measured["attempt_id"]]
        results = [result for result in summary["task_results"] if result["task_id"] == measured["task_id"]]
        if (len(results) != 1 or results[0]["gate_status"] != measured["status"]
                or results[0]["compatibility_key"] != measured["compatibility_key"]
                or summary["provenance"]["canonical_result_hash"] != measured["canonical_result_hash"]):
            raise ValueError("test performance: scoped final measurement mismatch")
    return [(attempt, result) for attempt, summary in summaries.items() if attempt not in superseded
            for result in summary.get("task_results", [])
            if result["task_id"] not in final or final[result["task_id"]] == attempt]


def _scoped_rows(inputs):
    """Use existing scoped-publications contracts without loading their GIFs."""
    from .scoped_publications import REGISTRY, SCHEMA
    from .trainer_families import scientific_row_hash
    if not (inputs.root / REGISTRY).is_file():
        return []
    registry = inputs.read(REGISTRY.as_posix())
    if registry.get("schema") != SCHEMA or registry.get("qualification_input") is not False:
        raise ValueError("test performance: invalid scoped publication registry")
    rows = []
    for entry in registry["cohorts"]:
        document = inputs.read(entry["path"], sha256=entry["sha256"])
        if (document.get("schema") != "forge_tier1_scoped_evidence_v1" or document.get("qualification_input") is not False
                or document.get("parent_qualification_credit") is not False or document["view"] != entry["view"]):
            raise ValueError("test performance: scoped evidence cannot grant parent credit")
        for family, row in document["rows"].items():
            if (row["trainer_family"] != family or scientific_row_hash(row) != document["scientific_rows_sha256"][family]
                    or row["bindings"]["source_digest"] != document["source_digest"]):
                raise ValueError("test performance: scoped scientific row drift")
            rows.append((row, document, {"snapshot": entry["path"], "json_sha256": entry["sha256"]}))
    return rows


def build_report(root, *, publication=None):
    import hashlib
    inputs = Inputs(root)
    if publication is None:
        publication = inputs.read(PUBLICATION)
    else:
        wire = json.dumps(publication, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        inputs.hashes[PUBLICATION] = hashlib.sha256(wire.encode()).hexdigest()
    _digest_check(publication)
    recipes, tests, records, evidence_catalogs = {}, {}, {}, {}
    declarations = {}
    for path in sorted([*(inputs.root / "configs/forge/tasks").glob("*.json"),
                        *(inputs.root / "configs/forge/task-variants").glob("*/*.json")]):
        name = path.relative_to(inputs.root).as_posix()
        task = inputs.read(name)
        if task["id"] in declarations:
            raise ValueError("test performance: duplicate task declaration")
        declarations[task["id"]] = (name, task)

    def add_recipe(identity, label, selected, lane, recipe):
        key = stable_hash(identity)
        entry = recipes.setdefault(key, {"id": key, "label": label, "identity": portable(identity),
                                         "recipe": portable(recipe), "selected": False, "lane": lane})
        entry["selected"] |= selected
        return key

    def add_test(task_id, conditions, lane, contract, definition_match, definition=None):
        key = stable_hash({"task_id": task_id, "conditions": conditions})
        test = tests.setdefault(key, {"id": key, "task_id": task_id, "lane": lane,
                                     "conditions": portable(conditions), "contract": portable(contract),
                                     "definition_match": definition_match, "definition": definition,
                                     "cells": {}, "witnesses": []})
        return test

    def add_record(test, recipe, record):
        record = portable(record)
        if "original_evidence" in record:
            original = record["original_evidence"]
            digest = stable_hash(original)
            evidence_catalogs[digest] = original
            record["original_evidence"] = {"catalog_sha256": digest, "availability": original["availability"]}
        key = stable_hash(record)
        records[key] = record
        cell = test["cells"].setdefault(recipe, [])
        if key not in cell:
            cell.append(key)

    sources = {}
    rows = [(row, True) for row in publication.get("rows", [])]
    rows += [(row, False) for key in ("evidence_rows", "archived_evidence_rows", "configuration_rows", "historical_family_rows")
             for row in publication.get(key, []) if row.get("attempt_ids")]
    scoped = _scoped_rows(inputs)
    scoped_sources = {stable_hash(row): (document, pointer) for row, document, pointer in scoped}
    rows += [(row, False) for row, _, _ in scoped]
    seen = set()
    for row, selected in rows:
        bindings = row.get("bindings", {})
        source = publication
        pointer = publication.get("evidence_sources", {}).get(row.get("publication_key"))
        if stable_hash(row) in scoped_sources:
            source, pointer = scoped_sources[stable_hash(row)]
        elif pointer:
            name = pointer["snapshot"]
            if name not in sources:
                sources[name] = inputs.read(name, sha256=pointer["json_sha256"])
                _digest_check(sources[name])
            source = sources[name]
        catalogs = {key: {**publication.get(key, {}), **source.get(key, {})}
                    for key in ("recipe_contracts", "protocol_contracts", "task_contracts", "status_reasons")}
        recipe = catalogs["recipe_contracts"].get(bindings.get("recipe_sha256"))
        protocol = catalogs["protocol_contracts"].get(bindings.get("protocol_sha256"))
        for digest, contract in [(bindings.get("recipe_sha256"), recipe), (bindings.get("protocol_sha256"), protocol)]:
            if contract is not None and stable_hash(contract) != digest:
                raise ValueError("test performance: scientific catalog digest mismatch")
        identity = {"candidate_id": row["candidate_id"], "candidate_revision": row.get("candidate_revision"),
                    "configuration_id": row.get("configuration_id", bindings.get("configuration_id")),
                    "bindings": {key: bindings.get(key) for key in ("recipe_sha256", "source_digest", "source_origin_commit", "prior", "claim_contract", "initializer")},
                    "runtime": row.get("runtime_cohort", {})}
        recipe_key = add_recipe(identity, row.get("technique", row["candidate_id"]), selected, "Forge", recipe)
        row_digest = stable_hash(row)
        if row_digest in seen:
            continue
        seen.add(row_digest)
        receipt_rows = _receipt(inputs, row, source)
        for outcome in row.get("tasks", []) + row.get("nonrequired_tasks", []):
            task_id = outcome["task_id"]
            contract_digest = bindings.get("task_contracts", {}).get(task_id)
            contract = catalogs["task_contracts"].get(contract_digest)
            if contract is not None and stable_hash(contract) != contract_digest:
                raise ValueError("test performance: task contract digest mismatch")
            definition_path, definition = declarations.get(task_id, (None, None))
            match = "unavailable"
            if definition and contract:
                match = "matches" if (contract.get("execution_sha256") == task_execution_fingerprint(definition)
                    and contract.get("evaluation_sha256") == task_evaluation_fingerprint(definition)
                    and contract.get("timeout_seconds") == definition.get("resources", {}).get("timeout_seconds")) else "CHANGED"
            conditions = {"task_contract_sha256": contract_digest, "protocol_sha256": bindings.get("protocol_sha256"),
                          "protocol": protocol, "rng_sha256": bindings.get("rng_sha256"),
                          "initializer": bindings.get("initializer"), "source_digest": bindings.get("source_digest"),
                          "runtime": row.get("runtime_cohort", {})}
            if contract is None or protocol is None:
                conditions["unbound_recipe"] = recipe_key  # Missing bindings cannot establish comparability.
            test = add_test(task_id, conditions, "Forge", contract, match, definition_path)
            receipts = [(attempt, result) for attempt, result in receipt_rows if result["task_id"] == task_id]
            reasons = catalogs["status_reasons"].get(outcome.get("reasons_sha256"), [])
            # The board owns the grade. Receipts expose original metrics and raw
            # execution errors, never manufacture a passing gate from an endpoint.
            for attempt, result in receipts or [(None, {})]:
                status = recorded_status(outcome, result)
                add_record(test, recipe_key, {"status": status, "recorded_status": deepcopy(outcome),
                    "attempt_id": attempt, "receipt": "reports/forge/technique-receipts/" + attempt + ".json" if attempt else None,
                    "publication": pointer["snapshot"] if pointer else PUBLICATION,
                    "receipt_sha256": inputs.hashes.get("reports/forge/technique-receipts/" + attempt + ".json") if attempt else None,
                    "result": {key: result[key] for key in ("metrics", "evaluator_summary", "raw_status", "reason", "cost", "sampling", "compatibility_key") if key in result},
                    "reasons": reasons or outcome.get("reasons", []) or row.get("blockers", []),
                    "evidence_available": bool(receipts), "kind": "optimizer"})

    # New declarations with no compatible original host/evaluator stay visible.
    for task_id, (path, task) in declarations.items():
        if not any(test["task_id"] == task_id and test["definition_match"] == "matches" for test in tests.values()):
            add_test(task_id, {"declared_task_sha256": stable_hash(task), "execution_sha256": task_execution_fingerprint(task),
                              "evaluation_sha256": task_evaluation_fingerprint(task), "no_recorded_protocol": True},
                     "Unmeasured declaration", task, "matches", path)

    _add_api(inputs, publication, add_recipe, add_test, add_record)
    _add_supplemental(inputs, add_recipe, add_test, add_record)
    _add_witnesses(inputs, tests)
    ordered_recipes = sorted(recipes.values(), key=lambda row: (not row["selected"], row["label"], row["id"]))
    selected_ids = [row["id"] for row in ordered_recipes if row["selected"]]
    for test in tests.values():
        test["coverage"] = coverage(test, list(recipes), records)
        test["selected_coverage"] = coverage(test, selected_ids, records)
    result = {"schema": "forge_test_performance_v1", "qualification_input": False, "qualification_reuse": False,
              "training_launched": False, "source_publication_digest": publication["provenance"]["input_digest"],
              "recipes": ordered_recipes, "tests": sorted(tests.values(), key=lambda row: (row["coverage"]["optimizer_passes"], row["task_id"], row["id"])),
              "records": records, "evidence_catalogs": evidence_catalogs, "inputs": dict(sorted(inputs.hashes.items())),
              "provenance": {"projector_sha256": file_hash(Path(__file__)),
                             "renderer_sha256": file_hash(Path(__file__).with_name("test_performance_render.py"))},
              "witness_scope": "Explicit byte-pinned witnesses only; unindexed evidence is unknown, not disproven.",
              "notes": ["Rows compare exact task, protocol, seeds/RNG, initializer, source and runtime bindings.",
                        "Empty cells mean no compatible indexed evidence. BLOCKED/INVALID retain original reasons.",
                        "Zero passing recipes is a research smell, never proof that a test is impossible.",
                        "Oracle/representational passes are feasibility witnesses, never optimizer or qualification passes.",
                        "Historical and API protocols never fill another row; no per-test recipe selection becomes a suite score."]}
    result["input_digest"] = stable_hash(result)
    return result


def _add_api(inputs, publication, add_recipe, add_test, add_record):
    """Reuse the public-API contract publications already linked by tier reports."""
    base = "reports/toy_audit/api_contract/"
    if not (inputs.root / (base + "cases.json")).is_file():
        return
    definitions = {row["id"]: row for row in inputs.read(base + "cases.json")["cases"]}
    for filename in ("runs.json", "failed-runs.json"):
        document = inputs.read(base + filename)
        for run in document["cases"]:
            definition = definitions[run["id"]]
            recipe = run.get("recipe")
            source = document.get("source_identities", {}).get(run.get("source_identity"), {})
            conditions = {"case_sha256": stable_hash(definition), "protocol": run.get("protocol"),
                          "seed": run.get("seed"), "source_identity": run.get("source_identity"),
                          "runtime": run.get("runtime"), "sampling": run.get("sampling"),
                          "original_gates": run.get("thresholds")}
            key = add_recipe({"recipe_sha256": stable_hash(recipe), "source_identity": run.get("source_identity"),
                              "source_commit": source.get("commit"), "runtime": run.get("runtime")},
                             "API · " + str((recipe or {}).get("name", "unbound")), False, "API", recipe)
            test = add_test(run["id"], conditions, "API", definition, "separate API protocol", base + "cases.json")
            status = run.get("verdict", "UNASSESSED")
            if status in {"PASS", "FAIL"} and run.get("default_protocol_complete") is not True:
                status = "INCOMPLETE"
            status = recorded_status({"status": status}, run)
            add_record(test, key, {"kind": "optimizer", "status": status, "recorded_status": run.get("verdict"),
                "receipt": base + filename, "receipt_sha256": inputs.hashes[base + filename],
                "raw_receipt_sha256": run.get("raw_receipt_sha256"), "evidence_available": True,
                "result": {name: run[name] for name in ("protocol", "completed_updates", "default_protocol_complete", "sustained_metric_passed", "final_metrics", "failed_bounds", "thresholds") if name in run}})


def _add_witnesses(inputs, tests):
    if not (inputs.root / WITNESSES).exists():
        return
    registry = inputs.read(WITNESSES)
    if registry.get("schema") != "forge_test_witnesses_v1":
        raise ValueError("test performance: unsupported witness registry")
    seen = set()
    for witness in registry["witnesses"]:
        if witness["id"] in seen or witness["kind"] not in {"oracle", "representational"}:
            raise ValueError("test performance: duplicate or untyped witness")
        seen.add(witness["id"])
        key = stable_hash({"task_id": witness["task_id"], "conditions": witness["conditions"]})
        if key not in tests:
            raise ValueError("test performance: witness has no exact protocol row")
        pin = witness["evidence"]
        evidence = inputs.read(pin["path"], sha256=pin["sha256"])
        for token in witness["status_pointer"].strip("/").split("/"):
            evidence = evidence[int(token)] if isinstance(evidence, list) else evidence[token.replace("~1", "/").replace("~0", "~")]
        if evidence not in {"PASS", "FAIL", "INCOMPLETE", "ERROR"}:
            raise ValueError("test performance: witness has no recorded numerical verdict")
        tests[key]["witnesses"].append({"id": witness["id"], "kind": witness["kind"], "status": evidence,
                                       "evidence": pin, "status_pointer": witness["status_pointer"], "scope": portable(witness["scope"])})


def _add_supplemental(inputs, add_recipe, add_test, add_record):
    """Compact reviewed local publications remain independent diagnostic rows."""
    if not (inputs.root / SUPPLEMENTAL).is_file():
        return
    registry = inputs.read(SUPPLEMENTAL)
    if registry.get("schema") != "forge_test_display_registry_v1" or registry.get("qualification_input") is not False:
        raise ValueError("test performance: invalid supplemental registry")
    seen = set()
    for pin in registry["publications"]:
        document = inputs.read(pin["path"], sha256=pin["sha256"])
        if (document.get("schema") != "forge_test_display_evidence_v1" or document.get("qualification_input") is not False
                or document.get("qualification_reuse") is not False or portable(document) != document):
            raise ValueError("test performance: invalid or nonportable diagnostic publication")
        for cell in document["cells"]:
            if cell["id"] in seen:
                raise ValueError("test performance: duplicate diagnostic cell")
            seen.add(cell["id"])
            recipe = cell.get("complete_recipe", document.get("recipe_contracts", {}).get(cell["recipe_sha256"]))
            if not isinstance(recipe, dict) or not recipe or stable_hash(recipe) != cell["recipe_sha256"]:
                raise ValueError("test performance: diagnostic full recipe digest mismatch")
            conditions = cell.get("conditions", document.get("condition_contracts", {}).get(cell.get("conditions_sha256")))
            if conditions is None or ("conditions_sha256" in cell and stable_hash(conditions) != cell["conditions_sha256"]):
                raise ValueError("test performance: diagnostic conditions digest mismatch")
            if not all(key in conditions for key in ("task", "seed", "source_digest", "runtime", "budget", "original_gates")):
                raise ValueError("test performance: diagnostic protocol binding incomplete")
            status = recorded_status({"status": cell["status"]})
            if status in {"PASS", "FAIL"} and cell["full_protocol_complete"] is not True:
                raise ValueError("test performance: incomplete diagnostic claims full gate")
            result = cell["result"]
            if "effective_recipe" in result:
                effective = result["effective_recipe"]
                declared = recipe.get("declared_global_recipe", {})
                owned = result.get("task_owned_recipe_fields", {})
                if (stable_hash(effective) != result.get("effective_recipe_sha256")
                        or set(effective) != set(declared)
                        or any(value != (owned[field]["value"] if field in owned else declared[field])
                               for field, value in effective.items())
                        or result.get("original_grade", {}).get("gate_status") != cell["status"]):
                    raise ValueError("test performance: executed diagnostic recipe or grade differs from binding")
            identity = {"candidate_id": cell["candidate_id"], "recipe_sha256": cell["recipe_sha256"],
                        "source_digest": conditions["source_digest"], "runtime": conditions["runtime"]}
            if cell.get("candidate_revision") is not None:
                identity["candidate_revision"] = cell["candidate_revision"]
            key = add_recipe(identity, cell["label"], False, "Forge", recipe)
            test = add_test(cell["task_id"], conditions, "Scoped diagnostic", conditions["task"], "separate scoped protocol")
            add_record(test, key, {"kind": "optimizer", "status": status, "recorded_status": cell["status"],
                "receipt": pin["path"], "receipt_sha256": pin["sha256"], "evidence_available": True,
                "result": cell["result"], "full_protocol_complete": cell["full_protocol_complete"],
                "cell_id": cell["id"], "candidate_revision": cell.get("candidate_revision"),
                "original_evidence": document["original_evidence"], "scope": document["scope"]})


def generated_reports(root, *, publication=None):
    from .test_performance_render import render_html, render_markdown
    result = build_report(root, publication=publication)
    return {Path(root) / (PREFIX + ".json"): json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            Path(root) / (PREFIX + ".md"): render_markdown(result), Path(root) / (PREFIX + ".html"): render_html(result)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true", help="fail on stale generated reports, writing nothing")
    args = parser.parse_args(argv)
    outputs = generated_reports(args.root)
    stale = []
    for path, content in outputs.items():
        if not path.is_file() or path.read_text() != content:
            stale.append(path.name)
            if not args.check:
                atomic_text(path, content)
    if args.check and stale:
        parser.exit(1, "Stale test performance reports: " + ", ".join(stale) + "\n")
    print(json.dumps({"reports": [path.name for path in outputs], "changed": stale, "training_launched": False}))


if __name__ == "__main__":
    main()
