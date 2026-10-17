"""Pure projection of three committed result publications, never qualification.

Only registry-pinned JSON, archive cards, Markdown and GIF bytes are read. Raw
paths inside a report are provenance strings; they are never opened. This module
imports no training, scoring, sampling, hydration, Git or queue implementation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote

REGISTRY = "reports/forge/completed-studies.json"
REGISTRY_SCHEMA = "forge_completed_studies_registry_v1"
SECTION_SCHEMA = "forge_completed_studies_projection_v1"
KINDS = ("atlas19_hold", "critic_balance", "generator_step")
FOLDERS = dict(zip(KINDS, ("continuous-baseline-20261003", "critic-balance-20261003", "generator-step-20261003")))
REPORT_SCHEMAS = dict(zip(KINDS, ("continuous_original_atlas19_and_c6_hold_publication_v1",
                                "particlegan_critic_balance_publication_v1", "particlegan_generator_step_publication_v1")))
ARCHIVE_SCHEMAS = dict(zip(KINDS, ("particlegan-continuous-archive-card-v1",
                                 "particlegan-critic-balance-archive-card-v1", "particlegan-generator-step-archive-card-v1")))
HEX = re.compile(r"[0-9a-f]{64}")
STATUSES = {"PASS", "FAIL", "UNKNOWN", "INCOMPLETE", "BLOCKED", "ERROR", "INVALID", "UNASSESSED"}
FAMILIES = ("atlas", "e22")
CASE_HORIZONS = {
    "image-develop-img_intensity2-source-transpose12": 600,
    "api-vector-two-broad": 1200, "api-grid100": 7000,
    "api-rotated100": 7000, "api-staggered100": 7000,
    "api-vector-unequal-mass": 1200, "api-vector-anisotropic": 1200,
    "image-develop-img_bars4-source-transpose12": 600,
}


def _require(condition, message):
    if not condition:
        raise ValueError("completed studies: " + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(data):
    def invalid(value):
        raise ValueError("completed studies: nonfinite JSON value " + value)
    return json.loads(data, object_pairs_hook=_object, parse_constant=invalid)


def _relative(value):
    _require(isinstance(value, str) and value and "\\" not in value and "\0" not in value,
             "safe POSIX relative path required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/")),
             "unsafe or noncanonical relative path")
    return path


def _file(root, name):
    path = root.joinpath(*_relative(name).parts)
    for part in (path, *path.parents):
        if part == root:
            break
        _require(not part.is_symlink(), "symlinked committed input")
    _require(path.is_file() and path.resolve().is_relative_to(root), "missing committed input")
    return path


def _pin(root, name):
    path = _file(root, name)
    data = path.read_bytes()
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _read_pin(root, pin):
    _require(isinstance(pin, dict) and set(pin) == {"path", "sha256", "bytes"}, "exact committed pin required")
    _require(isinstance(pin["sha256"], str) and HEX.fullmatch(pin["sha256"]), "SHA256 required")
    _require(type(pin["bytes"]) is int and pin["bytes"] >= 0, "exact byte count required")
    data = _file(root, pin["path"]).read_bytes()
    _require(len(data) == pin["bytes"] and hashlib.sha256(data).hexdigest() == pin["sha256"], "pinned input drift")
    return data


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite nonnegative cost required")
    return value


def _same_cost(a, b):
    _require(math.isclose(_number(a), _number(b), rel_tol=1e-9, abs_tol=1e-8), "inconsistent paid/reserved cost")


def _cost(cost):
    for key in ("paid_seconds", "reserved_seconds", "charged_seconds"):
        _number(cost[key])
    _same_cost(cost["charged_seconds"], cost["paid_seconds"] + cost["reserved_seconds"])
    if "previous_paid_seconds" in cost:
        _same_cost(cost["combined_charged_seconds"], cost["charged_seconds"] + cost["previous_paid_seconds"])
    return deepcopy(cost)


def _counts(rows, field, stated, *, unavailable=False):
    actual = Counter((row.get(field) or "UNAVAILABLE") if unavailable else row[field] for row in rows)
    allowed = STATUSES | ({"UNAVAILABLE"} if unavailable else set())
    _require(set(actual) <= allowed and isinstance(stated, dict) and set(stated) <= allowed, "unknown status/count")
    _require(all(type(count) is int and count >= 0 for count in stated.values()), "integer status counts required")
    _require(dict(actual) == {key: val for key, val in stated.items() if val}, "inconsistent status denominator")
    return dict(sorted(actual.items()))


def _source(source):
    _require(isinstance(source, dict) and re.fullmatch(r"[0-9a-f]{40}", source.get("commit", "")), "exact source commit required")
    _require(isinstance(source.get("manifest_sha256"), str) and HEX.fullmatch(source["manifest_sha256"]), "source manifest digest required")
    return {key: deepcopy(value) for key, value in source.items() if key != "snapshot_path"}


def _archive_source(source, archive):
    matches = [row for row in archive["source_cohorts"] if row.get("origin_commit") == source["commit"]]
    _require(len(matches) == 1, "source is not a distinct archived cohort")
    if "execution_digest" in source:
        _require(matches[0].get("execution_digest", matches[0].get("digest")) == source["execution_digest"], "archived source/snapshot digest changed")


def _no_credit(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"qualification_input", "qualification_changed", "default_adoption", "speed_ranking", "prior_qualification_credit", "speed_eligible"}:
                _require(child is False, "publication cannot grant qualification/default/speed credit")
            if key in {"speed_winner", "fair_speed_winner", "eligible_shipping_winner"}:
                _require(child is None, "publication cannot declare a winner")
            if key == "fully_qualified_ids":
                _require(child == [], "publication cannot declare a qualified configuration")
            _no_credit(child)
    elif isinstance(value, list):
        for child in value:
            _no_credit(child)


def _report_media(report, kind):
    if kind == "atlas19_hold":
        values = [row["media"] for key in ("baseline", "hold_continuations") for row in report[key]["rows"]]
        values += [row["gif"] for row in report.get("supplemental_media", [])]
    else:
        values = [row["media"] for row in report["cases"] if row.get("media")]
    result = []
    for value in values:
        if value is None:
            continue
        _relative(value["path"])
        _require(value["path"].endswith(".gif") and type(value.get("frames")) is int and value["frames"] > 0,
                 "recorded actual GIF metadata required")
        result.append({key: value[key] for key in ("path", "sha256", "bytes")})
    _require(len({pin["path"] for pin in result}) == len(result), "duplicate goal media path")
    return result


def build_registry(publication_roots):
    """Return the fixed registry from committed publication trees; no writes."""
    _require(set(publication_roots) == set(KINDS), "exactly the three assigned publications required")
    studies = []
    for kind in KINDS:
        root = Path(publication_roots[kind]).resolve()
        directory = "reports/forge/" + FOLDERS[kind]
        report_pin = _pin(root, directory + "/results.json")
        report = _json(_read_pin(root, report_pin))
        media = [{**pin, "path": directory + "/" + pin["path"]} for pin in _report_media(report, kind)]
        for pin in media:
            _read_pin(root, pin)
        studies.append({"id": kind, "report": report_pin, "archive": _pin(root, directory + "/archive.json"),
                        "readout": _pin(root, directory + "/README.md"), "archive_readout": _pin(root, directory + "/ARCHIVE.md"),
                        "media": media})
    return {"schema": REGISTRY_SCHEMA, "qualification_input": False, "reuse": False,
            "cross_cohort_pooling": False, "studies": studies}


def _media(value, directory):
    return None if value is None else {key: deepcopy(value[key]) for key in ("sha256", "bytes", "frames")} | {
        "path": directory + "/" + value["path"], "caption": value.get("caption")}


def _baseline(report, archive, directory):
    baseline = report["baseline"]
    _archive_source(baseline["source"], archive)
    holds = report["hold_continuations"]
    rows = baseline["rows"]
    _require(baseline["required_questions"] == 19 and len(rows) == 19 and len({row["id"] for row in rows}) == 19,
             "all 19 original questions required")
    _require(baseline["completed"] == sum(row["full_protocol_complete"] is True for row in rows)
             and baseline["media_completed"] == sum(row.get("media") is not None for row in rows), "original completeness count")
    execution = _counts(rows, "execution_status", baseline["execution_counts"])
    science = _counts(rows, "scientific_status", baseline["scientific_counts"])
    _require(execution == science == {"PASS": 19} and baseline["required_evidence_complete"] is True,
             "assigned original19 publication must retain its complete original PASS scope")
    for row in rows:
        definition = row["definition"]
        _require(definition["id"] == row["id"] and definition["original_requirements"] and definition["observation_steps"],
                 "original question/gate/cadence missing")
        _require(type(definition["original_host"]["steps"]) is int and definition["original_host"]["steps"] > 0, "original full budget missing")
        if row["group"] == "native":
            _require(row["native_gates"].get("noisy") == {"coverage": "PASS", "accuracy": "PASS"}, "native noisy joint gate missing")
            _require(definition["original_host"].get("evaluation_samples") == 20000
                     and definition["original_host"].get("holdout_samples") == 100000
                     and len(definition["observation_steps"]) == 34, "original native20k/100k/cadence scope missing")
    _require(baseline["declared_updates"] == sum(row["definition"]["original_host"]["steps"] for row in rows), "original update denominator")
    _same_cost(baseline["cost"]["paid_seconds"], sum(_cost(row["cost"])["paid_seconds"] for row in rows))
    hold_rows = holds["rows"]
    _require(holds["required_questions"] == 2 and len(hold_rows) == 2 and {row["family"] for row in hold_rows} == set(FAMILIES), "two distinct hold questions required")
    hold_counts = _counts(hold_rows, "scientific_status", holds["scientific_counts"])
    _counts(hold_rows, "execution_status", holds["execution_counts"])
    _require(hold_counts == {"FAIL": 2} and holds["completed"] == 2, "assigned hold failures must remain distinct")
    for row in hold_rows:
        _archive_source(row["source"], archive)
        hold = row["compound_hold"]
        _require(row["full_protocol_complete"] is True and row["new_updates"] == 150 and row["completed_steps"] == 1350
                 and row["original_gate"] == "PASS" and row["original_study_gate"] == "INCOMPLETE"
                 and hold["status"] == "FAIL" and hold["passed"] is False
                 and hold["hold_checks"] == hold["required_hold_checks"] == 5 and hold["hold_passed"] == 3,
                 "original C6 grade or distinct appended hold changed")
    hold_cost = _cost(holds["cost"])
    _same_cost(hold_cost["paid_seconds"], sum(_cost(row["cost"])["paid_seconds"] for row in hold_rows))
    engineering = holds["engineering_startup"]
    _same_cost(hold_cost["previous_paid_seconds"], engineering["paid_seconds"])
    _same_cost(engineering["paid_seconds"], sum(row["paid_wall_seconds"] for row in engineering["records"]))
    _require(len(engineering["records"]) == 2 and all(row["status"] == "INCOMPLETE" and row["scientific_updates"] == 0 for row in engineering["records"]), "startup is engineering, not scientific FAIL")
    _same_cost(report["cost"]["paid_seconds"], baseline["cost"]["paid_seconds"] + hold_cost["combined_charged_seconds"])
    _cost(report["cost"])
    _require(archive["baseline"] == {"PASS": 19, "required": 19} and archive["hold_continuations"] == {"FAIL": 2, "required": 2}
             and archive["cost"] == report["cost"], "archive/result count or cost disagreement")
    cells = [{key: deepcopy(row[key]) for key in ("id", "question", "execution_status", "scientific_status", "full_protocol_complete", "definition", "clean_diagnostic_status", "native_gates", "cost")}
             | {"media": _media(row["media"], directory)} for row in rows]
    continuation = [{key: deepcopy(row[key]) for key in ("id", "family", "question", "execution_status", "scientific_status", "full_protocol_complete", "definition", "original_gate", "original_study_gate", "compound_hold", "runtime", "original_source", "cost")}
                    | {"source": _source(row["source"]), "media": _media(row["media"], directory)} for row in hold_rows]
    return {"id": "atlas19_original", "label": "Historical Atlas19 original protocols", "required_cells": 19,
            "counts": science, "source": _source(baseline["source"]), "runtime": deepcopy(baseline["runtime"]),
            "recipe": deepcopy(baseline["configuration"]), "law": "Original noisy selected-policy law; clean diagnostics separate; native joint coverage+accuracy includes independent 100k",
            "declared_updates": baseline["declared_updates"], "cost": _cost(baseline["cost"]), "cells": cells}, {
            "id": "c6_hold", "label": "Named C6 broad hold extension", "required_cells": 2,
            "counts": hold_counts, "source": [row["source"] for row in continuation], "cells": continuation,
            "law": "Original 8021 public served sampler; output_noise=False; 150 appended updates, no ordinary eight-case credit",
            "best_original_baseline": "Broad checkpoint only: both original 1200 PASS; original study INCOMPLETE; new 1350 hold FAIL (3/5); no full eight-case configuration grade", "cost": hold_cost}


def _contrast(report, archive, kind, directory):
    _archive_source(report["source"], archive)
    rows = report["cases"]
    _require(report["required_configurations"] == 2 and report["required_cases_per_configuration"] == 8
             and report["required_cells"] == len(rows) == 16, "whole-config16-cell denominator required")
    pairs = {(row["family"], row["id"]) for row in rows}
    _require(len(pairs) == 16 and {row["family"] for row in rows} == set(FAMILIES)
             and all(len([row for row in rows if row["family"] == family]) == 8 for family in FAMILIES), "duplicate/missing family case")
    _require({row["id"] for row in rows if row["family"] == "atlas"} == {row["id"] for row in rows if row["family"] == "e22"}, "different family denominators")
    _require({row["id"] for row in rows} == set(CASE_HORIZONS), "exact eight declared question IDs required")
    configuration_ids = [{row["config_id"] for row in rows if row["family"] == family} for family in FAMILIES]
    _require(all(len(ids) == 1 and all(isinstance(value, str) and value for value in ids) for ids in configuration_ids)
             and configuration_ids[0].isdisjoint(configuration_ids[1]), "one distinct unchanged configuration per family required")
    selection = report["selection"]
    _require(selection["required_configurations"] == 2 and selection["required_cells"] == 16
             and selection["required_cases_per_config"] == 8 and selection["attempts_concluded"] is True, "unfinished or changed selection denominator")
    counts = report["counts"]
    _counts(rows, "status", counts["execution"])
    _counts(rows, "original_gate", counts["original_gates"], unavailable=True)
    _counts(rows, "study_gate", counts["study_gates"], unavailable=True)
    cold = Counter(row["capacity"]["status"] for row in rows)
    _require(dict(cold) == counts["capacity"] == {"SUPPORTED": 16}, "assigned cold capacity denominator")
    _require(all(type(row["capacity"]["ordinary_training_updates"]) is int and row["capacity"]["ordinary_training_updates"] == 0 for row in rows), "capacity cannot count as training")
    _require(counts["goal_gifs"] == sum(row.get("media") is not None for row in rows), "GIF denominator mismatch")
    expected = {"critic_balance": {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 2.25},
                "generator_step": {"lr": .00265625, "prior_lr_mult": 3., "d_lr_mult": 4.5}}[kind]
    for row in rows:
        _require(all(isinstance(row[key], str) and HEX.fullmatch(row[key]) for key in ("case_sha256", "resolved_recipe_sha256"))
                 and row["capacity"]["case_sha256"] == row["case_sha256"]
                 and row["capacity"]["recipe_sha256"] == row["resolved_recipe_sha256"], "capacity/case/recipe identity changed")
        _require(row["recipe_overrides"] == expected and row["requirements"]["thresholds"]
                 and row["requirements"]["sampling"] and row["requirements"]["evaluation_observations"] == 24,
                 "whole recipe/gate/sampling/cadence metadata changed")
        _require(row["requirements"]["default_steps"] == CASE_HORIZONS[row["id"]], "original full horizon changed")
        sibling = next(value for value in rows if value["id"] == row["id"] and value["family"] != row["family"])
        _require(row["requirements"] == sibling["requirements"] and row["case_sha256"] == sibling["case_sha256"], "different family gates/host/sampling")
        if row["status"] == "UNKNOWN":
            _require(row["original_gate"] is None and row["study_gate"] is None and row["completed_updates"] is None
                     and row["full_protocol_complete"] is False and row["media"] is None
                     and row["paid_seconds"] == row["conservative_reserved_seconds"] == 0., "unknown cannot acquire grade/time/media credit")
        if row["full_protocol_complete"]:
            _require(row["completed_updates"] == row["requirements"]["default_steps"] and row["original_gate"] in {"PASS", "FAIL"}
                     and row["study_gate"] in {"PASS", "FAIL", "INCOMPLETE"}
                     and row["acquisition_hold"]["status"] == row["study_gate"], "original/hold/full-horizon discrepancy")
    costs = deepcopy(report["costs"])
    _same_cost(costs["ordinary_paid_seconds"], sum(_number(row["paid_seconds"]) for row in rows))
    _same_cost(costs["ordinary_reserved_seconds"], sum(_number(row["conservative_reserved_seconds"]) for row in rows))
    prior = costs.get("prior_total_paid_seconds", costs.get("engineering_paid_seconds", 0.))
    _same_cost(costs["combined_charged_seconds"], costs["ordinary_paid_seconds"] + costs["ordinary_reserved_seconds"] + prior)
    _require(costs["combined_cap_seconds"] == 15360. and costs["combined_charged_seconds"] <= 15360., "original campaign allowance changed")
    _require(archive["costs"] == costs and archive["learning"] == {**counts["execution"], "required": 16}
             and archive["capacity"] == {**counts["capacity"], "required": 16}, "archive/result denominator or cost changed")
    cells = [{key: deepcopy(row[key]) for key in ("id", "family", "question", "requirements", "recipe_overrides", "resolved_recipe_sha256", "case_sha256", "runtime", "status", "original_gate", "study_gate", "full_protocol_complete", "completed_updates", "acquisition_hold", "paid_seconds", "conservative_reserved_seconds")}
             | {"capacity": {key: deepcopy(row["capacity"][key]) for key in ("status", "ordinary_training_updates", "case_sha256", "recipe_sha256")},
                "media": _media(row["media"], directory)} for row in rows]
    return {"id": kind, "label": "Critic-rate contrast" if kind == "critic_balance" else "Generator-half contrast",
            "required_cells": 16, "required_configurations": 2, "required_cases_per_configuration": 8,
            "study_id": report["study_id"], "counts": deepcopy(counts), "source": _source(report["source"]),
            "runtime": deepcopy(report["runtime_cohorts"]), "recipe_overrides": expected, "cells": cells, "cost": costs,
            "law": "Public selected-policy serving; image/vector output-noise-off primary, native noisy primary; 24×20k native checks, no independent 100k"}


def load_completed_studies(root):
    """Return optional additive evidence; absence preserves existing behavior."""
    root = Path(root).resolve()
    path = root / REGISTRY
    if not path.exists() and not path.is_symlink():
        return {}
    registry = _json(_file(root, REGISTRY).read_bytes())
    _require(registry.get("schema") == REGISTRY_SCHEMA, "unknown registry schema")
    _require(all(registry.get(key) is False for key in ("qualification_input", "reuse", "cross_cohort_pooling")), "registry cannot grant qualification/reuse/pooling")
    studies = registry.get("studies", [])
    _require(len(studies) == 3 and [row.get("id") for row in studies] == list(KINDS), "exact ordered three distinct assigned cohorts required")
    rows = []
    pins = []
    cost_groups = {}
    for study in studies:
        kind = study["id"]
        directory = "reports/forge/" + FOLDERS[kind]
        for role, suffix in (("report", "results.json"), ("archive", "archive.json"), ("readout", "README.md"), ("archive_readout", "ARCHIVE.md")):
            _require(study[role]["path"] == directory + "/" + suffix, "assigned committed report path required")
            _read_pin(root, study[role])
            pins.append(deepcopy(study[role]))
        report = _json(_read_pin(root, study["report"]))
        archive = _json(_read_pin(root, study["archive"]))
        _require(report["schema"] == REPORT_SCHEMAS[kind] and archive["schema"] == ARCHIVE_SCHEMAS[kind], "wrong report/archive cohort")
        _no_credit(report)
        _no_credit(archive)
        _require(archive["availability"] == "LOCAL_ONLY" and archive["remote_replication"] == "NOT_PERFORMED"
                 and archive["all_member_hashes_verified"] is True and HEX.fullmatch(archive["sha256"]), "raw archive availability is not hosted evidence")
        wanted = [{**pin, "path": directory + "/" + pin["path"]} for pin in _report_media(report, kind)]
        _require(study["media"] == wanted, "all original/supplemental committed media must remain pinned")
        for pin in wanted:
            data = _read_pin(root, pin)
            _require(data[:6] in (b"GIF87a", b"GIF89a"), "pinned goal GIF required")
            pins.append(deepcopy(pin))
        projected = list(_baseline(report, archive, directory)) if kind == "atlas19_hold" else [_contrast(report, archive, kind, directory)]
        cost_groups[kind] = deepcopy(report.get("cost", report.get("costs")))
        for row in projected:
            row.update(publication=deepcopy(study["report"]), readout=study["readout"]["path"],
                       archive_readout=study["archive_readout"]["path"], archive={"card": deepcopy(study["archive"]),
                       **{key: deepcopy(archive[key]) for key in ("availability", "remote_replication", "sha256", "bytes")}},
                       qualification_input=False, reuse=False, cross_cohort_pooling=False)
        rows += projected
    critic, generator = rows[-2:]
    _same_cost(generator["cost"]["prior_total_paid_seconds"], critic["cost"]["combined_charged_seconds"])
    _same_cost(generator["cost"]["prior_scientific_paid_seconds"], critic["cost"]["ordinary_paid_seconds"])
    _same_cost(generator["cost"]["prior_engineering_paid_seconds"], critic["cost"]["engineering_paid_seconds"])
    return {"schema": SECTION_SCHEMA, "qualification_input": False, "reuse": False, "cross_cohort_pooling": False,
            "default_adoption": False, "speed_winner": None, "ordinary_rows_changed": False,
            "registry": _pin(root, REGISTRY), "inputs": pins, "rows": rows, "cost_groups": cost_groups,
            "cost_scope": "Historical Atlas19+hold cost is disjoint. Generator cumulative cost already includes critic science and startup once; never add both cumulative figures.",
            "verification_scope": "Registry-pinned committed publications only; original certification remains authoritative; no raw archive/sampler/scorer replay.",
            "new_training_updates": 0, "new_sampler_calls": 0, "metric_rescoring": False}


def render_completed_studies(section, root, markdown_path):
    """Render additive navigation; no writes and no ordinary selection changes."""
    if not section:
        return ""
    root = Path(root).resolve()
    _require(section == load_completed_studies(root), "section changed or committed inputs drifted before rendering")
    destination = Path(markdown_path)
    if not destination.is_absolute():
        destination = root / Path(*_relative(str(markdown_path)).parts)
    _require(destination.resolve().is_relative_to(root), "Markdown output must be inside checkout")
    def link(label, relative):
        target = root / Path(*_relative(relative).parts)
        href = quote(os.path.relpath(target, destination.parent).replace(os.sep, "/"), safe="/.-_")
        return "[" + label + "](" + href + ")"
    lines = ["", "## Completed source-bound studies", "",
             "These separate publications add evidence navigation. They do not change the ordinary table, ranks, tiers or qualification. Historical and current cells are not pooled; no default or fair speed winner is established.", "",
             "| Study | Result | Goal and readout |",
             "| --- | --- | --- |"]
    details = ["<details>", "<summary>Exact protocols, sources, costs and archive availability</summary>", ""]
    for row in section["rows"]:
        cost = row["cost"]
        if row["id"] == "atlas19_original":
            verdict = f"19/19 original PASS; {row['declared_updates']:,} original updates; native final-five 20k + independent 100k; clean native diagnostics separate"
            compact = "19/19 original PASS"
            clock = f"{cost['paid_seconds']:.6f} / {cost['reserved_seconds']:.6f}"
            source = row["source"]["commit"][:8] + "; original noisy law"
        elif row["id"] == "c6_hold":
            verdict = "2/2 new hold FAIL; old 1200 PASS/study INCOMPLETE retained; 150 appended each; 3/5 later checks"
            compact = "2/2 new hold FAIL"
            clock = f"{cost['paid_seconds']:.6f} new + {cost['previous_paid_seconds']:.6f} startup / {cost['reserved_seconds']:.6f}"
            source = row["cells"][0]["original_source"]["commit"][:8] + " → " + row["cells"][0]["source"]["commit"][:8]
        else:
            counts = row["counts"]
            verdict = "2 configs × 8 = 16; 16 cold SUPPORTED; " + ", ".join(f"{n} {status}" for status, n in counts["execution"].items())
            verdict += "; original " + ", ".join(f"{n} {status}" for status, n in counts["original_gates"].items())
            verdict += "; hold " + ", ".join(f"{n} {status}" for status, n in counts["study_gates"].items()) + "; 24 observations; native 20k, no 100k"
            compact = "; ".join(f"{n} {status}" for status, n in counts["execution"].items())
            clock = f"{cost['ordinary_paid_seconds']:.6f} new / {cost['ordinary_reserved_seconds']:.6f}; cumulative {cost['combined_charged_seconds']:.6f}"
            source = row["source"]["commit"][:8] + "; selected-policy; LR/prior/D " + "/".join(str(row["recipe_overrides"][key]) for key in ("lr", "prior_lr_mult", "d_lr_mult"))
        media = next((cell["media"] for cell in row["cells"] if cell.get("media")), None)
        evidence = link("readout + exact gates", row["readout"])
        if media:
            evidence = link("actual goal GIF", media["path"]) + "; " + evidence
        lines.append(f"| {row['label']} | {compact} | {evidence} |")
        details += [f"### {row['label']}", "",
                    f"- **Required evidence and result:** {verdict}",
                    f"- **Source and recipe:** `{source}`",
                    f"- **Serving law:** {row['law']}",
                    f"- **Paid / reserve (seconds):** {clock}",
                    "- **Raw availability:** " + link("archive card", row["archive"]["card"]["path"]) + "; " +
                    link("raw resolver (LOCAL_ONLY)", row["archive_readout"]), ""]
    lines += [""] + details + ["</details>"]
    costs = section["cost_groups"]
    lines += ["", "Original GIF verdicts and added hold verdicts remain separate in each readout. Capacity uses zero optimizer updates and provides no learned PASS. UNKNOWN means unreached; engineering failures remain distinct from numerical FAIL.", "",
              "The historical intensity host uses residual16, seed 0 and enumeration of 32 rows without latent perturbation; the new intensity host uses transpose12, seed 24002 and 1024 public noise-off selected-policy draws with latent perturbation. These are different protocols. The C6 extension preserves only the original broad checkpoint's PASS/INCOMPLETE grades, not a full eight-case qualification.", "",
              "Costs are summed supervised child intervals, with conservative reserves separate. "
              f"Generator cumulative {costs['generator_step']['combined_charged_seconds']:.6f} seconds already includes critic {costs['critic_balance']['combined_charged_seconds']:.6f} seconds once. "
              f"Historical Atlas19+holds {costs['atlas19_hold']['charged_seconds']:.6f} seconds is a disjoint study; "
              "these figures are not elapsed-time or FLOPs rankings. Raw archives remain LOCAL_ONLY; this projection does not hydrate or independently recertify them.", ""]
    return "\n".join(lines)
