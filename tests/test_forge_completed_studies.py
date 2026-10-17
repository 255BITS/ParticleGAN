"""Self-contained publication controls: no raw archive, Torch or models."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

try:
    from experiments.forge import completed_studies as module
except ImportError:
    spec = importlib.util.spec_from_file_location("external_completed_studies", Path(__file__).with_name("completed_studies.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

GIF = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b")


def source(letter):
    return {"commit": letter * 40, "manifest_sha256": letter * 64,
            "execution_digest": letter * 64, "snapshot_path": "/unavailable/raw/snapshot"}


def cost(paid):
    return {"paid_seconds": paid, "reserved_seconds": 0., "charged_seconds": paid}


def media(name):
    return {"path": f"media/{name}/goal.gif", "sha256": hashlib.sha256(GIF).hexdigest(),
            "bytes": len(GIF), "frames": 1, "caption": "Original gate FAIL; added hold FAIL; no acquisition"}


def publications():
    runtime = {"python": "3.12", "torch": "recorded-only", "device": "cuda:0", "torch_threads": 1}
    baseline_rows = []
    for index in range(19):
        native = index >= 16
        definition = {"id": f"original-{index}", "original_host": {"steps": 7000 if native else 600},
                      "original_requirements": [["coverage", ">=", .99]], "observation_steps": list(range(34 if native else 24))}
        if native:
            definition["original_host"].update(evaluation_samples=20000, holdout_samples=100000)
        baseline_rows.append({"id": definition["id"], "question": "Frozen original question",
                              "definition": definition, "group": "native" if native else "portability",
                              "execution_status": "PASS", "scientific_status": "PASS", "full_protocol_complete": True,
                              "clean_diagnostic_status": "FAIL" if native else None,
                              "native_gates": {"noisy": {"coverage": "PASS", "accuracy": "PASS"},
                                               "clean": {"coverage": "FAIL", "accuracy": "FAIL"}} if native else {},
                              "cost": cost(1.), "media": media(f"original-{index}")})
    hold_rows = [{"id": "hold-" + family, "family": family, "question": "Named persistence extension",
                  "execution_status": "FAIL", "scientific_status": "FAIL", "full_protocol_complete": True,
                  "new_updates": 150, "completed_steps": 1350, "original_gate": "PASS", "original_study_gate": "INCOMPLETE",
                  "compound_hold": {"status": "FAIL", "passed": False, "hold_checks": 5,
                                    "required_hold_checks": 5, "hold_passed": 3, "first_confirmed_step": 1100, "speed_eligible": False},
                  "runtime": runtime, "source": source("b"), "original_source": {"commit": "8" * 40},
                  "definition": {"original_case": "api-vector-two-broad", "seed": 24002,
                                 "original_recipe": {"lr": .0053125, "prior_lr_mult": 1.5}, "new_steps": [1250, 1300, 1350]},
                  "cost": cost(1.), "media": media("hold-" + family)} for family in module.FAMILIES]
    original = {"schema": module.REPORT_SCHEMAS["atlas19_hold"], "qualification_input": False,
                "default_adoption": False, "speed_ranking": False, "new_training_updates": 0, "new_sampler_calls": 0,
                "baseline": {"rows": baseline_rows, "required_questions": 19, "completed": 19, "media_completed": 19,
                             "required_evidence_complete": True, "execution_counts": {"PASS": 19}, "scientific_counts": {"PASS": 19},
                             "declared_updates": sum(row["definition"]["original_host"]["steps"] for row in baseline_rows),
                             "cost": cost(19.), "source": source("a"), "runtime": runtime,
                             "configuration": {"overrides": {"lr": .00425}, "source": "immutable original"}},
                "hold_continuations": {"rows": hold_rows, "required_questions": 2, "completed": 2,
                                       "execution_counts": {"FAIL": 2}, "scientific_counts": {"FAIL": 2},
                                       "cost": {**cost(2.), "previous_paid_seconds": 2., "combined_charged_seconds": 4.},
                                       "engineering_startup": {"paid_seconds": 2., "records": [
                                           {"status": "INCOMPLETE", "scientific_updates": 0, "paid_wall_seconds": .5},
                                           {"status": "INCOMPLETE", "scientific_updates": 0, "paid_wall_seconds": 1.5}]}},
                "cost": cost(23.), "supplemental_media": [], "external_raw_provenance": "/unavailable/raw/study.json"}
    original_archive = {"schema": module.ARCHIVE_SCHEMAS["atlas19_hold"], "availability": "LOCAL_ONLY",
                        "remote_replication": "NOT_PERFORMED", "all_member_hashes_verified": True,
                        "sha256": "f" * 64, "bytes": 123, "qualification_changed": False,
                        "baseline": {"PASS": 19, "required": 19}, "hold_continuations": {"FAIL": 2, "required": 2},
                        "cost": original["cost"], "source_cohorts": [
                            {"origin_commit": letter * 40, "execution_digest": letter * 64} for letter in ("a", "b")]}
    reports = {"atlas19_hold": original}
    archives = {"atlas19_hold": original_archive}
    for kind, letter, paid_values, prior in (("critic_balance", "c", (3., 4.), 2.), ("generator_step", "d", (5., 6.), 9.)):
        overrides = {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 2.25} if kind == "critic_balance" else {
            "lr": .00265625, "prior_lr_mult": 3., "d_lr_mult": 4.5}
        cases = []
        for family, paid in zip(module.FAMILIES, paid_values):
            for index, (case_id, steps) in enumerate(module.CASE_HORIZONS.items()):
                executed = index == 0
                cases.append({"id": case_id, "family": family, "question": "Exact declared API question",
                              "config_id": family + "--" + kind, "case_sha256": "1" * 64,
                              "recipe_overrides": overrides, "resolved_recipe_sha256": "2" * 64,
                              "requirements": {"default_steps": steps, "evaluation_observations": 24,
                                               "thresholds": {"coverage": .99}, "sampling": {"law": "selected-policy"}},
                              "runtime": runtime if executed else None,
                              "capacity": {"status": "SUPPORTED", "ordinary_training_updates": 0,
                                           "case_sha256": "1" * 64, "recipe_sha256": "2" * 64},
                              "status": "FAIL" if executed else "UNKNOWN", "original_gate": "FAIL" if executed else None,
                              "study_gate": "FAIL" if executed else None, "full_protocol_complete": executed,
                              "completed_updates": steps if executed else None,
                              "acquisition_hold": {"status": "FAIL", "reason": "no five-observation window"} if executed else None,
                              "paid_seconds": paid if executed else 0., "conservative_reserved_seconds": 0.,
                              "media": media(kind + "-" + family) if executed else None})
        costs = {"ordinary_paid_seconds": sum(paid_values), "ordinary_reserved_seconds": 0.,
                 "combined_charged_seconds": sum(paid_values) + prior, "combined_cap_seconds": 15360.}
        costs.update({"engineering_paid_seconds": prior} if kind == "critic_balance" else {
            "prior_total_paid_seconds": prior, "prior_scientific_paid_seconds": 7., "prior_engineering_paid_seconds": 2.})
        reports[kind] = {"schema": module.REPORT_SCHEMAS[kind], "source": source(letter), "study_id": kind,
                         "runtime_cohorts": {family: runtime for family in module.FAMILIES}, "costs": costs,
                         "required_configurations": 2, "required_cases_per_configuration": 8, "required_cells": 16,
                         "cases": cases, "counts": {"capacity": {"SUPPORTED": 16}, "execution": {"FAIL": 2, "UNKNOWN": 14},
                                                      "original_gates": {"FAIL": 2, "UNAVAILABLE": 14},
                                                      "study_gates": {"FAIL": 2, "UNAVAILABLE": 14}, "goal_gifs": 2},
                         "selection": {"required_configurations": 2, "required_cases_per_config": 8, "required_cells": 16,
                                       "attempts_concluded": True, "fully_qualified_ids": [], "default_adoption": False, "speed_winner": None}}
        archives[kind] = {"schema": module.ARCHIVE_SCHEMAS[kind], "availability": "LOCAL_ONLY",
                          "remote_replication": "NOT_PERFORMED", "all_member_hashes_verified": True, "sha256": "e" * 64,
                          "bytes": 234, "costs": costs, "learning": {"FAIL": 2, "UNKNOWN": 14, "required": 16},
                          "capacity": {"SUPPORTED": 16, "required": 16}, "qualification_changed": False,
                          "source_cohorts": [{"origin_commit": letter * 40, "digest": letter * 64}]}
    return reports, archives


def install(root, reports, archives):
    for kind in module.KINDS:
        directory = root / "reports/forge" / module.FOLDERS[kind]
        directory.mkdir(parents=True, exist_ok=True)
        for name, obj in (("results.json", reports[kind]), ("archive.json", archives[kind])):
            (directory / name).write_text(json.dumps(obj, sort_keys=True, allow_nan=False))
        (directory / "README.md").write_text("Exact question/gate context; original and hold verdicts separate.\n")
        (directory / "ARCHIVE.md").write_text("Raw archive LOCAL_ONLY; independent hydration not performed.\n")
        for pin in module._report_media(reports[kind], kind):
            path = directory / pin["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(GIF)
    registry = module.build_registry({kind: root for kind in module.KINDS})
    (root / module.REGISTRY).write_text(json.dumps(registry, sort_keys=True))
    return registry


@pytest.fixture
def publication(tmp_path):
    reports, archives = publications()
    registry = install(tmp_path, reports, archives)
    return tmp_path, reports, archives, registry


def test_optional_absence_preserves_existing_behavior(tmp_path):
    assert module.load_completed_studies(tmp_path) == {}
    assert module.render_completed_studies({}, tmp_path, "reports/forge/technique-inventory.md") == ""


def test_pure_idempotent_projection_does_not_change_ordinary_ranking(publication):
    root, _, _, _ = publication
    ordinary = {"rows": [{"candidate": "existing", "tier": 1, "rank": 1}], "selection": {"winner": None}}
    before = deepcopy(ordinary)
    section = module.load_completed_studies(root)
    assert section == module.load_completed_studies(root)
    assert ordinary == before
    assert section["ordinary_rows_changed"] is False
    assert section["qualification_input"] is section["reuse"] is section["cross_cohort_pooling"] is False
    assert section["new_training_updates"] == section["new_sampler_calls"] == 0
    assert section["metric_rescoring"] is False
    rows = section["rows"]
    assert [row["id"] for row in rows] == ["atlas19_original", "c6_hold", "critic_balance", "generator_step"]
    assert rows[0]["counts"] == {"PASS": 19}
    assert rows[1]["counts"] == {"FAIL": 2}
    assert all(cell["original_gate"] == "PASS" and cell["original_study_gate"] == "INCOMPLETE" for cell in rows[1]["cells"])
    assert all(len(row["cells"]) == 16 and row["counts"]["execution"] == {"FAIL": 2, "UNKNOWN": 14} for row in rows[2:])
    assert rows[3]["cost"]["combined_charged_seconds"] == 20.
    assert rows[3]["cost"]["prior_total_paid_seconds"] == rows[2]["cost"]["combined_charged_seconds"] == 9.
    assert section["cost_groups"]["atlas19_hold"]["charged_seconds"] == 23.
    rendered = module.render_completed_studies(section, root, "reports/forge/technique-inventory.md")
    assert rendered == module.render_completed_studies(section, root, "reports/forge/technique-inventory.md")
    assert "19/19 original PASS" in rendered and "2/2 new hold FAIL" in rendered
    assert "14 UNKNOWN" in rendered and "16 cold SUPPORTED" in rendered
    assert "without latent perturbation" in rendered and "native 20k, no 100k" in rendered
    assert "LOCAL_ONLY" in rendered and "already includes critic" in rendered
    # Verdicts stay visible in a narrow table; detailed provenance remains
    # available in an expandable section beside the source-pinned readouts.
    table = [line for line in rendered.splitlines() if line.startswith("|")]
    assert len(table) == 6 and all(line.count("|") == 4 for line in table)
    assert "<summary>Exact protocols, sources, costs and archive availability</summary>" in rendered
    assert all(f"### {row['label']}" in rendered for row in rows)
    assert not (root / "reports/forge/technique-inventory.md").exists()


def test_only_committed_inputs_are_read_without_raw_archive_dependencies(publication, monkeypatch):
    root, _, _, registry = publication
    allowed = {root / module.REGISTRY}
    for row in registry["studies"]:
        allowed.update(root / row[role]["path"] for role in ("report", "archive", "readout", "archive_readout"))
        allowed.update(root / pin["path"] for pin in row["media"])
    original = Path.read_bytes
    reads = []
    def pinned_only(path):
        assert path in allowed
        reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", pinned_only)
    section = module.load_completed_studies(root)
    module.render_completed_studies(section, root, "reports/forge/technique-inventory.md")
    assert reads and len(section["inputs"]) == len(allowed) - 1
    assert not (root / "unavailable/raw").exists()


@pytest.mark.parametrize("role", ["report", "archive", "readout", "archive_readout", "gif"])
@pytest.mark.parametrize("mutation", ["missing", "drift"])
def test_all_committed_inputs_are_hash_bound(publication, role, mutation):
    root, _, _, registry = publication
    pin = registry["studies"][0]["media"][0] if role == "gif" else registry["studies"][0][role]
    path = root / pin["path"]
    path.unlink() if mutation == "missing" else path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError):
        module.load_completed_studies(root)


@pytest.mark.parametrize("name", ["/tmp/report.json", "../report.json", "reports/../results.json", "reports//file", "./file", "reports\\file"])
def test_unsafe_registry_path_rejected(publication, name):
    root, _, _, registry = publication
    registry["studies"][0]["report"]["path"] = name
    (root / module.REGISTRY).write_text(json.dumps(registry))
    with pytest.raises(ValueError):
        module.load_completed_studies(root)


def test_symlinked_input_rejected_even_when_bytes_match(publication):
    root, _, _, registry = publication
    target = root / registry["studies"][0]["readout"]["path"]
    copied = root / "same-bytes.md"
    copied.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(copied)
    with pytest.raises(ValueError, match="symlink"):
        module.load_completed_studies(root)


def test_duplicate_cohort_missing_media_and_section_tamper_rejected(publication):
    root, reports, archives, registry = publication
    section = module.load_completed_studies(root)
    section["rows"][2]["counts"]["execution"]["PASS"] = 16
    with pytest.raises(ValueError, match="section changed"):
        module.render_completed_studies(section, root, "reports/forge/technique-inventory.md")
    registry["studies"][1] = deepcopy(registry["studies"][0])
    (root / module.REGISTRY).write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="three distinct"):
        module.load_completed_studies(root)
    registry = install(root, reports, archives)
    registry["studies"][0]["media"].pop()
    (root / module.REGISTRY).write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="media must remain pinned"):
        module.load_completed_studies(root)


@pytest.mark.parametrize("defect", ["counter", "duplicate", "denominator", "unknown_pass", "unknown_media", "capacity_updates", "capacity_recipe", "config_identity", "family_gate", "horizon", "cadence", "knobs", "source", "native", "hold", "cost", "cost_bool", "winner", "default", "double_prior"])
def test_rebound_metadata_cannot_create_scientific_or_cost_credit(publication, defect):
    root, reports, archives, _ = publication
    critic = reports["critic_balance"]
    if defect == "counter":
        critic["counts"]["execution"] = {"PASS": 2, "UNKNOWN": 14}
    elif defect == "duplicate":
        critic["cases"][1] = deepcopy(critic["cases"][2])
    elif defect == "denominator":
        critic["required_cells"] = 2
    elif defect == "unknown_pass":
        critic["cases"][1]["original_gate"] = "PASS"
        critic["counts"]["original_gates"] = {"FAIL": 2, "PASS": 1, "UNAVAILABLE": 13}
    elif defect == "unknown_media":
        critic["cases"][1]["media"] = media("unknown")
        critic["counts"]["goal_gifs"] = 3
    elif defect == "capacity_updates":
        critic["cases"][1]["capacity"]["ordinary_training_updates"] = 1
    elif defect == "capacity_recipe":
        critic["cases"][1]["capacity"]["recipe_sha256"] = "9" * 64
    elif defect == "config_identity":
        critic["cases"][1]["config_id"] = "per-toy-winner"
    elif defect == "family_gate":
        critic["cases"][1]["requirements"] = {**critic["cases"][1]["requirements"], "thresholds": {"coverage": .01}}
    elif defect == "horizon":
        critic["cases"][0]["requirements"] = {**critic["cases"][0]["requirements"], "default_steps": 5}
    elif defect == "cadence":
        critic["cases"][0]["requirements"] = {**critic["cases"][0]["requirements"], "evaluation_observations": 5}
    elif defect == "knobs":
        critic["cases"][0]["recipe_overrides"] = {"lr": .006, "prior_lr_mult": 1.5, "d_lr_mult": 2.25}
    elif defect == "source":
        critic["source"]["commit"] = "9" * 40
    elif defect == "native":
        reports["atlas19_hold"]["baseline"]["rows"][-1]["definition"]["original_host"]["holdout_samples"] = 0
    elif defect == "hold":
        reports["atlas19_hold"]["hold_continuations"]["rows"][0]["original_study_gate"] = "PASS"
    elif defect == "cost":
        critic["costs"]["ordinary_paid_seconds"] = 0.
    elif defect == "cost_bool":
        critic["cases"][0]["paid_seconds"] = True
    elif defect == "winner":
        critic["selection"]["fully_qualified_ids"] = ["atlas"]
    elif defect == "default":
        critic["selection"]["default_adoption"] = True
    elif defect == "double_prior":
        generator = reports["generator_step"]
        generator["costs"]["prior_total_paid_seconds"] *= 2
        generator["costs"]["combined_charged_seconds"] += 9.
    install(root, reports, archives)
    with pytest.raises(ValueError):
        module.load_completed_studies(root)


def test_registry_and_json_reject_ambiguous_numeric_inputs(publication):
    root, _, _, registry = publication
    for text in ('{"schema": "x", "schema": "y"}', '{"value": NaN}', '{"value": Infinity}'):
        with pytest.raises(ValueError):
            module._json(text)
    registry["studies"][0]["report"]["bytes"] = True
    (root / module.REGISTRY).write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="byte count"):
        module.load_completed_studies(root)


def test_no_training_or_external_execution_imports():
    tree = ast.parse(Path(module.__file__).read_text())
    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imports |= {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imports <= {"__future__", "collections", "copy", "hashlib", "json", "math", "os", "pathlib", "re", "urllib"}
    assert not {"torch", "numpy", "subprocess", "importlib", "experiments", "particlegan"} & imports
