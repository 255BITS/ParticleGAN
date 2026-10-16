"""Scientific display controls: exact cohorts, missing evidence, and provenance."""
from copy import deepcopy
import json

import pytest

from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.test_performance import (Inputs, build_report, cell_status, coverage,
                                               generated_reports, portable, recorded_status)
from experiments.forge.test_performance_render import render_html, render_markdown
from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint


def seal(publication):
    publication.setdefault("provenance", {}).pop("input_digest", None)
    publication["provenance"]["input_digest"] = stable_hash(publication)
    return publication


@pytest.fixture
def evidence(tmp_path):
    task = {"id": "test_a", "schema_version": 1, "adapter": "fixture", "execution": {"steps": 80},
            "evaluation": {"kind": "transfer_sustained", "thresholds": [["score", ">=", .5]],
                           "observations": 24, "minimum_stable_checks": 5}, "resources": {"timeout_seconds": 60}}
    atomic_json(tmp_path / "configs/forge/tasks/test_a.json", task)
    contract = {"execution_sha256": task_execution_fingerprint(task), "evaluation_sha256": task_evaluation_fingerprint(task),
                "steps": 80, "timeout_seconds": 60, "sampling": {"scoring_weights": "live", "eval_output_noise": False}}
    recipe = {"name": "method", "lr": .01, "prior_lr_mult": 2, "prior_reg": 0, "optimizer_family": "adam"}
    protocol = {"seed": 0, "revision": 1, "scoring": {"weights": "live"}}
    row = {"candidate_id": "whole_recipe", "candidate_revision": "revision", "technique": "Method",
           "bindings": {"recipe_sha256": stable_hash(recipe), "protocol_sha256": stable_hash(protocol),
                        "rng_sha256": "rng", "source_digest": "source", "initializer": "orthogonal",
                        "task_contracts": {"test_a": stable_hash(contract)}},
           "runtime_cohort": {"execution_backend": "cpu"}, "tasks": [{"task_id": "test_a", "gate_status": "PASS"}],
           "attempt_ids": ["attempt"]}
    summary = {"attempt_id": "attempt", "candidate_id": "whole_recipe", "candidate_revision": "revision",
               "certificate_validated": True, "provenance": {"source_digest": "source", "canonical_result_hash": "result",
                    "original_files": {"result": {"sha256": "original"}}},
               "task_results": [{"task_id": "test_a", "gate_status": "PASS", "raw_status": "completed",
                    "metrics": {"score": .75}, "evaluator_summary": {"convergence": {"complete": True, "observations": 24}}}]}
    atomic_json(tmp_path / "reports/forge/technique-receipts/attempt.json", summary)
    publication = {"rows": [row], "recipe_contracts": {stable_hash(recipe): recipe},
                   "protocol_contracts": {stable_hash(protocol): protocol}, "task_contracts": {stable_hash(contract): contract},
                   "provenance": {"qualified_receipts": {"attempt": {"source_digest": "source", "canonical_result_hash": "result",
                                                               "original_file_sha256": {"result": "original"}}}}}
    return tmp_path, seal(publication), summary


def test_projection_is_read_only_and_preserves_whole_recipe(evidence, monkeypatch):
    import torch
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *a, **kw: pytest.fail("report launched an optimizer"))
    root, publication, _ = evidence
    before = deepcopy(publication)
    first = build_report(root, publication=publication)
    assert publication == before and first == build_report(root, publication=publication)
    assert len(first["tests"]) == len(first["recipes"]) == 1
    assert first["recipes"][0]["recipe"]["prior_lr_mult"] == 2
    assert first["tests"][0]["coverage"]["optimizer_passes"] == 1
    assert first["qualification_input"] is first["training_launched"] is False


@pytest.mark.parametrize("field,value", [("seed", 1), ("revision", 2), ("scoring", {"weights": "ema"})])
def test_unmatched_protocols_never_pool(evidence, field, value):
    root, publication, _ = evidence
    row = deepcopy(publication["rows"][0])
    protocol = deepcopy(next(iter(publication["protocol_contracts"].values())))
    protocol[field] = value
    publication["protocol_contracts"][stable_hash(protocol)] = protocol
    row["bindings"]["protocol_sha256"] = stable_hash(protocol)
    row["tasks"][0]["gate_status"] = "FAIL"
    publication["rows"].append(row)
    report = build_report(root, publication=seal(publication))
    assert len(report["tests"]) == 2
    assert sorted(t["coverage"]["optimizer_passes"] for t in report["tests"]) == [0, 1]


@pytest.mark.parametrize("field,value", [("rng_sha256", "other-rng"), ("source_digest", "new-source"), ("initializer", "zero")])
def test_seed_stream_source_and_initializer_changes_split_rows(evidence, field, value):
    root, publication, _ = evidence
    row = deepcopy(publication["rows"][0])
    row["attempt_ids"] = []
    row["bindings"][field] = value
    row["tasks"][0]["gate_status"] = "NOT_RUN"
    publication["rows"].append(row)
    report = build_report(root, publication=seal(publication))
    assert len(report["tests"]) == 2
    assert any(t["coverage"]["zero_optimizer_pass"] for t in report["tests"])


def test_changed_gate_keeps_old_pass_and_new_definition_unmeasured(evidence):
    root, publication, _ = evidence
    path = root / "configs/forge/tasks/test_a.json"
    task = json.loads(path.read_text())
    task["evaluation"]["thresholds"] = [["score", ">=", .9]]
    atomic_json(path, task)
    report = build_report(root, publication=publication)
    old = next(t for t in report["tests"] if t["definition_match"] == "CHANGED")
    new = next(t for t in report["tests"] if t["lane"] == "Unmeasured declaration")
    assert old["coverage"]["optimizer_passes"] == 1
    assert new["coverage"]["counts"] == {"NOT RUN": 1}
    assert new["coverage"]["no_indexed_witness"] is True


def test_error_before_training_is_not_scientific_fail(evidence):
    root, publication, summary = evidence
    publication["rows"][0]["tasks"][0]["gate_status"] = "BLOCKED"
    summary["task_results"][0].update(raw_status="error", reason="failed import in /home/user/private/run", metrics={})
    atomic_json(root / "reports/forge/technique-receipts/attempt.json", summary)
    report = build_report(root, publication=seal(publication))
    record = next(iter(report["records"].values()))
    assert record["status"] == "ERROR" and record["recorded_status"]["gate_status"] == "BLOCKED"
    assert report["tests"][0]["coverage"]["counts"] == {"ERROR": 1}
    assert "/home/" not in json.dumps(report)


def test_statuses_missing_cells_and_conflicts_are_visible():
    for status in ("PASS", "FAIL", "ERROR", "INCOMPLETE", "BLOCKED", "INVALID", "UNCONFIRMED"):
        assert recorded_status({"status": status}) == status
    assert recorded_status({"status": "UNKNOWN"}) == "NOT RUN"
    assert cell_status([]) == "NOT RUN"
    assert cell_status([{"status": "PASS"}, {"status": "FAIL"}]) == "MIXED"
    with pytest.raises(ValueError):
        recorded_status({"status": "SUCCEEDED"})
    result = coverage({"cells": {"recipe": ["pass", "fail"]}, "witnesses": []}, ["recipe"],
                      {"pass": {"status": "PASS"}, "fail": {"status": "FAIL"}})
    assert result["counts"] == {"MIXED": 1}
    assert result["optimizer_passes"] == 1 and result["zero_optimizer_pass"] is False


def test_pinned_feasibility_witness_cannot_become_optimizer_pass(evidence):
    root, publication, _ = evidence
    publication["rows"][0]["tasks"][0]["gate_status"] = "FAIL"
    report = build_report(root, publication=seal(publication))
    test = report["tests"][0]
    path = root / "reports/forge/feasibility.json"
    atomic_json(path, {"verdict": "PASS"})
    witness = {"id": "oracle-a", "task_id": "test_a", "kind": "oracle", "conditions": test["conditions"],
               "status_pointer": "/verdict", "scope": "Known target sampler, no optimizer run",
               "evidence": {"path": "reports/forge/feasibility.json", "sha256": file_hash(path)}}
    atomic_json(root / "reports/forge/test-witnesses.json", {"schema": "forge_test_witnesses_v1", "witnesses": [witness]})
    result = build_report(root, publication=publication)["tests"][0]["coverage"]
    assert result["optimizer_passes"] == 0 and result["oracle_passes"] == 1
    assert result["zero_optimizer_pass"] is True and result["no_indexed_witness"] is False
    witness["conditions"]["protocol"]["seed"] = 1
    atomic_json(root / "reports/forge/test-witnesses.json", {"schema": "forge_test_witnesses_v1", "witnesses": [witness]})
    with pytest.raises(ValueError, match="exact protocol"):
        build_report(root, publication=publication)


def test_provenance_drift_and_unsafe_files_fail_closed(evidence):
    root, publication, summary = evidence
    bad = deepcopy(publication)
    bad["rows"][0]["tasks"][0]["gate_status"] = "FAIL"
    with pytest.raises(ValueError, match="publication digest"):
        build_report(root, publication=bad)
    summary["candidate_revision"] = "wrong"
    atomic_json(root / "reports/forge/technique-receipts/attempt.json", summary)
    with pytest.raises(ValueError, match="receipt proof"):
        build_report(root, publication=publication)
    reader = Inputs(root)
    for path in ("../secret.json", "/tmp/secret.json", "reports\\secret.json"):
        with pytest.raises(ValueError, match="unsafe"):
            reader.read(path)
    with pytest.raises(ValueError, match="bytes changed"):
        reader.read("reports/forge/technique-receipts/attempt.json", sha256="wrong")
    (root / "escape.json").symlink_to(root / "reports/forge/technique-receipts/attempt.json")
    with pytest.raises(ValueError, match="symlink"):
        reader.read("escape.json")


def test_api_instantaneous_pass_cannot_fill_a_full_protocol(evidence):
    root, publication, _ = evidence
    base = root / "reports/toy_audit/api_contract"
    atomic_json(base / "cases.json", {"cases": [{"id": "api-a", "thresholds": {"score_min": .5}}]})
    atomic_json(base / "runs.json", {"cases": [{"id": "api-a", "recipe": {"name": "api-method"},
        "seed": 24002, "verdict": "PASS", "default_protocol_complete": False, "protocol": {"updates": 10}}]})
    atomic_json(base / "failed-runs.json", {"cases": []})
    result = build_report(root, publication=publication)
    api = next(t for t in result["tests"] if t["lane"] == "API")
    assert api["coverage"]["optimizer_passes"] == 0 and api["coverage"]["counts"]["INCOMPLETE"] == 1


def test_renderer_is_offline_accessible_and_escapes_data(evidence):
    root, publication, _ = evidence
    publication["rows"][0]["technique"] = '</script><script>alert("x")</script>'
    result = build_report(root, publication=seal(publication))
    rendered = render_html(result)
    payload = rendered.split('<script id="evidence" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload) == result and '</script>' not in payload
    assert "aria-live" in rendered and 'type="search"' in rendered and '<dialog' in rendered
    assert 'fetch(' not in rendered and 'https://' not in rendered
    markdown = render_markdown(result)
    assert 'NO INDEXED WITNESS' in markdown and 'technique-receipts/attempt.json' in markdown
    assert '/ml2/' not in portable({"path": "/ml2/private/file"})["path"]


def test_unbound_metadata_cannot_establish_shared_protocol(evidence):
    root, publication, _ = evidence
    for i in range(2):
        row = deepcopy(publication["rows"][0])
        row.update(candidate_id="unbound" + str(i), attempt_ids=[])
        row["bindings"].pop("protocol_sha256")
        row["tasks"][0]["gate_status"] = "NOT_RUN"
        publication["rows"].append(row)
    report = build_report(root, publication=seal(publication))
    assert len(report["tests"]) == 3
    assert sum("unbound_recipe" in test["conditions"] for test in report["tests"]) == 2


def test_supplemental_publications_fail_closed_on_scope_and_completeness(evidence):
    root, publication, _ = evidence
    recipe = {"native_recipe": {"name": "method"}, "extension": {"rule": "training_packet_only"}}
    cell = {"id": "screen", "candidate_id": "whole-extension", "label": "Scoped screen", "task_id": "test_a",
            "complete_recipe": recipe, "recipe_sha256": stable_hash(recipe), "status": "PASS", "full_protocol_complete": True,
            "conditions": {"task": {"steps": 80}, "seed": 0, "source_digest": "other-source", "runtime": {"device": "cpu"},
                           "budget": {"steps": 80}, "original_gates": [["score", ">=", .5]]}, "result": {"score": .75}}
    document = {"schema": "forge_test_display_evidence_v1", "qualification_input": False, "qualification_reuse": False,
                "scope": "Only the screening gate", "original_evidence": {"availability": "LOCAL_ONLY_NOT_PUBLISHED"}, "cells": [cell]}
    path = root / "reports/forge/test-performance-evidence/screen.json"
    registry = root / "reports/forge/test-performance-evidence/index.json"
    def write():
        atomic_json(path, document)
        atomic_json(registry, {"schema": "forge_test_display_registry_v1", "qualification_input": False,
                              "publications": [{"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}]})
    write()
    result = build_report(root, publication=publication)
    assert sum(test["lane"] == "Scoped diagnostic" for test in result["tests"]) == 1
    cell["full_protocol_complete"] = False
    write()
    with pytest.raises(ValueError, match="incomplete diagnostic"):
        build_report(root, publication=publication)
    cell["full_protocol_complete"] = True
    document["qualification_reuse"] = True
    write()
    with pytest.raises(ValueError, match="diagnostic publication"):
        build_report(root, publication=publication)


def test_existing_regenerator_emits_same_matrix_without_mutating_publication(evidence):
    import importlib.util
    from pathlib import Path
    root, publication, _ = evidence
    source = Path(__file__).resolve().parents[1] / "reports/forge/regenerate_technique_inventory.py"
    spec = importlib.util.spec_from_file_location("performance_regeneration", source)
    regenerator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regenerator)
    before = deepcopy(publication)
    assert regenerator._family_pages(root, publication) == generated_reports(root, publication=publication)
    assert publication == before


def test_reviewed_diagnostics_keep_failed_confirmation_and_all_original_cells():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    reader = Inputs(root)
    registry = reader.read("reports/forge/test-performance-evidence/index.json")
    documents = [reader.read(pin["path"], sha256=pin["sha256"]) for pin in registry["publications"]]
    cells = {cell["id"]: {**cell, "conditions": document["condition_contracts"][cell["conditions_sha256"]]}
             for document in documents for cell in document["cells"]}
    # 267 reviewed diagnostic cells plus PR399's 42 retained attempts (experiments/forge/pr399_scores.py PINS).
    assert len(cells) == 267 + 42
    assert cells["atlas-location-first-error"]["status"] == "ERROR"
    assert cells["atlas-location-first-error"]["result"]["work"]["new_GAN_updates"] == 0
    assert cells["atlas-location-treatment-seed0"]["status"] == "PASS"
    for seed, expected in ((1, "FAIL"), (2, "PASS")):
        for arm in ("control", "treatment"):
            cell = cells[f"atlas-location-confirmation-seed{seed}_{arm}"]
            assert cell["status"] == expected
            assert cell["conditions"]["budget"]["steps"] == 80
            assert cell["conditions"]["seed"] == seed
            assert cell["result"]["local_confirmation_accepted"] is False
    assert cells["g-angle-width-seed0"]["status"] == cells["g-angle-width-seed1"]["status"] == "PASS"
    assert cells["g-angle-anisotropic-seed0"]["status"] == "FAIL"
    for document in documents:
        assert document["qualification_input"] is document["qualification_reuse"] is False
        assert document["original_evidence"]["availability"] == "LOCAL_ONLY_NOT_PUBLISHED"
        assert portable(document) == document


@pytest.fixture
def nonatlas(evidence):
    from pathlib import Path
    root, publication, _ = evidence
    repository = Path(__file__).resolve().parents[1]
    directory = "reports/forge/test-performance-evidence/"
    documents = []
    pins = []
    for name in ("nonatlas-cell-moment-full-runs", "nonatlas-finite-realization-prefix64"):
        relative = directory + name + ".json"
        document = json.loads((repository / relative).read_text())
        atomic_json(root / relative, document)
        pins.append({"path": relative, "sha256": file_hash(root / relative)})
        documents.append(document)
    registry = {"schema": "forge_test_display_registry_v1", "qualification_input": False, "publications": pins}
    atomic_json(root / (directory + "index.json"), registry)
    return root, publication, documents, registry


def test_nonatlas_preserves_executed_recipes_and_nine_distinct_full_protocols(nonatlas):
    from collections import Counter
    root, publication, (full, prefix), _ = nonatlas
    before = deepcopy(publication)
    report = build_report(root, publication=publication)
    assert publication == before
    tests = [test for test in report["tests"] if test["lane"] == "Scoped diagnostic"]
    complete = [test for test in tests if test["task_id"] != "staggered100"]
    assert len(complete) == 9
    assert {test["task_id"] for test in complete} == {
        "vector_anisotropic", "vector_unequal_width", "vector_unequal_mass", "vector_two_broad",
        "vector_overlap", "vector_spiral", "vector_unequal_width_published", "gaussian1d_smoke", "ring16_acquisition"}
    for test in complete:
        assert len(test["cells"]) == 5 and test["conditions"]["seed"] == 0
        assert test["witnesses"] == []
        for ids in test["cells"].values():
            assert len(ids) == 1
            record = report["records"][ids[0]]
            actual = record["result"]
            assert record["full_protocol_complete"] is True
            assert actual["completed_steps"] == test["conditions"]["budget"]["steps"]
            assert len(actual["effective_recipe"]) == 110
            assert stable_hash(actual["effective_recipe"]) == actual["effective_recipe_sha256"]
            assert actual["actual_Popen_to_exit_seconds"] >= actual["runner_reported_seconds"]
            assert actual["ordinary_qualification"] is actual["promotion"] is False
    counts = Counter(("control" if c["result"]["mode"] == "per_kernel" else "aggregate", c["status"])
                     for c in full["cells"])
    assert counts == {("aggregate", "PASS"): 35, ("aggregate", "FAIL"): 1,
                      ("control", "PASS"): 5, ("control", "FAIL"): 4}
    ring = next(test for test in complete if test["task_id"] == "ring16_acquisition")
    assert ring["conditions"]["budget"]["steps"] == 1600
    assert ring["conditions"]["budget"]["observations"] == 96
    assert ring["conditions"]["budget"]["original_schedule_horizon"] == 400
    assert next(c for c in full["cells"] if c["task_id"] == "ring16_acquisition"
                and c["result"]["mode"] == "aggregate_bures")["status"] == "FAIL"
    assert sum(c["result"]["initial_prior"] is None for c in full["cells"]) == 5
    assert full["original_evidence"]["retained_errors_and_exclusions"]
    assert prefix["original_evidence"]["initial_CPU_validation_error_retained"] is True


def test_nonatlas_prefix_mechanism_pass_is_incomplete_quality(nonatlas):
    root, publication, (_, prefix), _ = nonatlas
    report = build_report(root, publication=publication)
    rows = [test for test in report["tests"] if test["task_id"] == "staggered100"]
    assert len(rows) == 1 and len(rows[0]["cells"]) == 2
    assert rows[0]["conditions"]["budget"]["steps"] == 7000
    assert rows[0]["coverage"]["optimizer_passes"] == 0
    for cell in prefix["cells"]:
        assert cell["status"] == "INCOMPLETE" and cell["full_protocol_complete"] is False
        assert cell["result"]["completed_updates"] == 64
        assert cell["result"]["original_four_full_FAILs_preserved"] is True
        assert len(prefix["recipe_contracts"][cell["recipe_sha256"]]["native_recipe"]) == 109
    receipts = [cell["result"]["actual_sequence_receipt"] for cell in prefix["cells"]]
    assert receipts[0] == receipts[1]


@pytest.mark.parametrize("field,value", [("aggregate_passed", 36), ("full_round_charge_seconds", 1)])
def test_nonatlas_shared_navigation_validates_counts_and_costs(nonatlas, field, value):
    from experiments.forge.test_performance_render import render_discovery_navigation
    root, _, (full, _), registry = nonatlas
    text = render_discovery_navigation(root, root / "reports/forge/technique-inventory.md")
    assert "35/36 PASS" in text and "5/9 PASS" in text
    assert "Bures ring FAIL" in text and "INCOMPLETE" in text and "2243.186445957" in text
    assert "/ml2/" not in text
    # Even a repinned document must not turn incorrect discovery totals into a leaderboard claim.
    full["discovery_summary"][field] = value
    pin = registry["publications"][0]
    atomic_json(root / pin["path"], full)
    pin["sha256"] = file_hash(root / pin["path"])
    atomic_json(root / "reports/forge/test-performance-evidence/index.json", registry)
    with pytest.raises(ValueError, match="overall summary"):
        render_discovery_navigation(root, root / "reports/forge/technique-inventory.md")


def test_nonatlas_rejects_repinned_executed_recipe_change(nonatlas):
    root, publication, (full, _), registry = nonatlas
    cell = full["cells"][0]
    cell["result"]["effective_recipe"]["lr"] *= 2
    cell["result"]["effective_recipe_sha256"] = stable_hash(cell["result"]["effective_recipe"])
    pin = registry["publications"][0]
    atomic_json(root / pin["path"], full)
    pin["sha256"] = file_hash(root / pin["path"])
    atomic_json(root / "reports/forge/test-performance-evidence/index.json", registry)
    with pytest.raises(ValueError, match="executed diagnostic recipe"):
        build_report(root, publication=publication)


@pytest.fixture
def reconciled(nonatlas):
    from pathlib import Path
    root, publication, _, registry = nonatlas
    name = "reports/forge/test-performance-evidence/nonatlas-reconciled-204.json"
    document = json.loads((Path(__file__).resolve().parents[1] / name).read_text())
    atomic_json(root / name, document)
    registry["publications"].append({"path": name, "sha256": file_hash(root / name)})
    atomic_json(root / "reports/forge/test-performance-evidence/index.json", registry)
    return root, publication, document, registry


def test_reconciled_packet_keeps_every_full_outcome_without_duplicate_producers(reconciled):
    from collections import Counter
    root, publication, document, _ = reconciled
    report = build_report(root, publication=publication)
    rows = [r for r in report["records"].values() if r.get("cell_id", "").startswith("nonatlas-")]
    full = [r for r in rows if r["full_protocol_complete"] is True]
    assert len(full) == len({r["result"]["attempt_id"] for r in full}) == 204
    assert Counter(r["status"] for r in full) == {"PASS": 107, "FAIL": 97}
    assert sum(r["result"].get("new_optimizer_updates", r["result"]["completed_steps"]) for r in full) == 434400
    continued = [r for r in full if r["result"].get("continuity")]
    assert len(continued) == 51
    by_attempt = {r["result"]["attempt_id"]:r for r in full}
    for row in continued:
        value = row["result"]
        assert value["completed_steps"] == 6000 and value["new_optimizer_updates"] == 5000
        dependency = value["producer_dependency"]
        parent = by_attempt[dependency["attempt_id"]]
        assert dependency["candidate_revision"] == parent["candidate_revision"]
        assert parent["status"] == "PASS" and parent["result"]["completed_steps"] == 1000
        assert dependency["cost_counted_in_original_producer_only"] is True
    assert all(r["result"]["ordinary_qualification"] is False for r in full)
    assert document["discovery_summary"]["global_survivor"] is False


def test_reconciled_errors_and_cpu_diagnostics_cannot_fill_full_quality(reconciled):
    root, publication, document, _ = reconciled
    report = build_report(root, publication=publication)
    errors = [r for r in report["records"].values() if r.get("cell_id", "").startswith("nonatlas-pre-runtime-error-")]
    assert len(errors) == 12
    for record in errors:
        result = record["result"]
        assert record["status"] == "ERROR" and record["full_protocol_complete"] is False
        assert result["original_grade"]["gate_status"] == "INCOMPLETE"
        assert result["raw_status"] == "error" and result["started_training"] is False
        assert result["new_optimizer_updates"] == 0
    assert sum(r["result"]["actual_Popen_to_exit_seconds"] for r in errors) == pytest.approx(30.168105392011057)
    summary = document["discovery_summary"]
    assert len(summary["cpu_preflight_rejections"]) == 3
    assert all(r["scientific_full_run"] is False and r["GPU_children"] == 0 for r in summary["cpu_preflight_rejections"])
    assert summary["frozen_gradient_probe"]["decision"] == "REJECT"
    assert summary["frozen_gradient_probe"]["new_training_updates"] == 0
    followup = summary["cpu_reference_followup"]
    assert followup["reference_ablation"]["decision"] == "REJECT"
    assert followup["quota_confirmation"]["decision"] == "SUPPORT"
    assert followup["reference_summary"]["global_qualification"] is False
    assert followup["reference_summary"]["timed_CPU_wall_seconds"] == pytest.approx(4.349590721001732)
    assert followup["quota_errata"]["original_registration_preserved"] is True
    assert not any("probe" in r.get("cell_id", "") for r in report["records"].values())


def test_reconciled_sample_gaps_and_complete_role_recipes_are_visible(reconciled):
    root, publication, document, _ = reconciled
    report = build_report(root, publication=publication)
    gaps = [r for r in report["records"].values() if r["result"].get("audit_scope", {}).get("sample_panels_available") is False]
    assert len(gaps) == 6
    assert all(r["status"] == "FAIL" and r["full_protocol_complete"] is True
               and r["result"]["saved_outputs_sha256"] is None for r in gaps)
    assert all(r["result"]["audit_scope"]["grade_basis"] == "ORIGINAL_LOGGED_METRICS_REGRADING" for r in gaps)
    recipes = [r["recipe"].get("declared_global_recipe", {}) for r in report["recipes"]]
    assert any(r.get("optimizer_role_families") for r in recipes)
    assert any(r.get("optimizer_signal_roles") for r in recipes)
    assert any(r.get("cell_cubature_rule") == "sobol128" for r in recipes)
    for record in report["records"].values():
        if "original_evidence" not in record: continue
        pin = record["original_evidence"]
        catalog = report["evidence_catalogs"][pin["catalog_sha256"]]
        assert stable_hash(catalog) == pin["catalog_sha256"]
        assert catalog["availability"] == "LOCAL_ONLY_NOT_PUBLISHED"
    assert len(report["evidence_catalogs"]) == 3
    html = render_html(report)
    assert "Audit gap: saved sample panels unavailable" in html
    assert "Pinned source catalogs, receipt hashes and audit limitations" in html


@pytest.mark.parametrize("field,value", [("unique_recipe_ids", 204), ("source_bound_revisions", 66),
                                          ("total_charged_attempt_wall_seconds", 0), ("global_survivor", True)])
def test_reconciled_summary_rejects_repinning_incorrect_identity_or_charge(reconciled, field, value):
    from experiments.forge.test_performance_render import render_discovery_navigation
    root, _, document, registry = reconciled
    text = render_discovery_navigation(root, root / "reports/forge/technique-inventory.md")
    assert "204 full runs" in text and "75 source-bound revisions" in text and "7 source cohorts" in text
    assert "21044.534383269" in text and "Six mode-hold" in text
    document["discovery_summary"][field] = value
    pin = registry["publications"][-1]
    atomic_json(root / pin["path"], document)
    pin["sha256"] = file_hash(root / pin["path"])
    atomic_json(root / "reports/forge/test-performance-evidence/index.json", registry)
    with pytest.raises(ValueError, match="test discoveries:"):
        render_discovery_navigation(root, root / "reports/forge/technique-inventory.md")


@pytest.fixture
def astra(reconciled):
    from pathlib import Path
    root, publication, _, registry = reconciled
    name = "reports/forge/test-performance-evidence/astra-cumulative-24.json"
    document = json.loads((Path(__file__).resolve().parents[1] / name).read_text())
    atomic_json(root / name, document)
    registry["publications"].append({"path": name, "sha256": file_hash(root / name)})
    atomic_json(root / "reports/forge/test-performance-evidence/index.json", registry)
    return root, publication, document, registry


def test_astra_cumulative_rows_and_paid_controls_are_counted_once(astra):
    from collections import Counter
    root, publication, document, _ = astra
    report = build_report(root, publication=publication)
    records = [r for r in report["records"].values() if r.get("cell_id", "").startswith("astra-")]
    assert len(records) == len({r["result"]["attempt_id"] for r in records}) == 24
    assert Counter(r["status"] for r in records) == {"PASS": 5, "FAIL": 19}
    interim = document["discovery_summary"]["interim_attempts_reused_once"]
    assert len(interim) == 6
    assert all(sum(r["result"]["attempt_id"] == key for r in records) == 1 for key in interim)
    candidates = [r for r in records if r["result"]["round"].startswith("astra-") and r["result"]["role"] == "candidate"]
    controls = [r for r in records if r["result"]["round"].startswith("astra-") and r["result"]["role"] == "control"]
    assert len(candidates) == 12 and len(controls) == 4
    assert all(r["status"] == "FAIL" and r["full_protocol_complete"] is True for r in candidates + controls)
    assert len({r["candidate_revision"] for r in controls}) == 4
    assert sum(r["result"]["new_optimizer_updates"] for r in records) == 36000
    assert sum(r["result"]["actual_Popen_to_exit_seconds"] for r in records) == pytest.approx(1206.4819039370632)
    assert sum(r["result"]["queue_charged_seconds"] for r in records) == pytest.approx(1157.1356888279988)


def test_astra_submetric_passes_and_predictor_work_never_override_full_gate(astra):
    root, publication, document, _ = astra
    report = build_report(root, publication=publication)
    records = [r for r in report["records"].values() if r.get("cell_id", "").startswith("astra-")]
    submetric_pass = []
    for cell in document["cells"]:
        grade = cell["result"]["original_grade"]["evaluator_result"]
        if cell["task_id"] != "vector_unequal_mass": continue
        assert cell["status"] == "FAIL"
        metric = next(m for m in grade["metrics"] if m["metric"] == "component_covariance_error")
        if metric["status"] == "PASS":
            submetric_pass.append(cell["candidate_id"])
            assert grade["convergence"]["passing_suffix"] == 1
            assert grade["convergence"]["minimum_stable_checks"] == 5
    assert set(submetric_pass) == {"cell-quota-aggregate-swap-v1", "cell-slow-covariance-v1", "cell-extragradient-map-v1"}
    lookahead = next(r for r in records if r["result"]["effective_recipe"].get("cell_extragradient_map"))
    assert lookahead["result"]["new_optimizer_updates"] == 1200
    assert lookahead["result"]["additional_temporary_predictor_evaluations"] == 1200
    assert lookahead["result"]["optimizer_updates_total_clock"] == {"generator": 1200, "discriminator": 1200, "prior": 1200}
    assert sum(r["result"]["additional_temporary_predictor_evaluations"] for r in records) == 1200
    fractional = next(r for r in records if r["result"]["effective_recipe"].get("cell_transport_policy") == "fractional_exact")
    assert fractional["result"]["assignment_label_semantics"] == "transport_row_argmax_not_hard_quota"
    assert fractional["result"]["trace_counters"]["fractional_steps"] == 1193
    assert all(f["elapsed_seconds"] is None and f["new_training_updates"] == 0
               for f in document["discovery_summary"]["preparation_failures"])


def test_duplicate_interim_attempt_is_rejected_even_with_a_new_cell_id(astra):
    root, publication, document, registry = astra
    duplicate = deepcopy(document)
    duplicate["cells"] = [deepcopy(document["cells"][-1])]
    duplicate["cells"][0]["id"] = "interim-copy-with-another-cell-id"
    path = "reports/forge/test-performance-evidence/duplicate-interim.json"
    atomic_json(root / path, duplicate)
    registry["publications"].append({"path": path, "sha256": file_hash(root / path)})
    atomic_json(root / "reports/forge/test-performance-evidence/index.json", registry)
    with pytest.raises(ValueError, match="duplicate diagnostic attempt/task"):
        build_report(root, publication=publication)


def test_reader_presentation_preserves_every_scientific_field_and_receipt(astra):
    from experiments.forge.test_performance_render import campaign_overview, diagnostic_group
    root, publication, _, _ = astra
    report = build_report(root, publication=publication)
    before = deepcopy(report)
    markdown = render_markdown(report)
    rendered = render_html(report)
    payload = rendered.split('<script id="evidence" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload) == before and report == before
    assert campaign_overview(report) == {"full_runs": 228, "passes": 112, "fails": 116}
    assert '228 full research runs' in rendered and 'No global winner' in rendered
    assert 'How to read a row' in markdown and 'NOT RUN' in markdown and 'MIXED' in markdown
    assert 'No graded result' in rendered and 'Failures only' in rendered
    for record in report["records"].values():
        if not record.get("cell_id"): continue
        assert record["receipt"].split('/')[-1] in markdown
    assert diagnostic_group('reports/forge/test-performance-evidence/astra-cumulative-24.json') == 'Quota transition and Astra portfolio'
    assert 'cohort' in rendered and 'metricSummary' in rendered


@pytest.mark.parametrize("field,value", [("family_full_runs", 234), ("additional_temporary_predictor_evaluations", 0),
                                        ("all_24_supervisor_wall_seconds", 0)])
def test_astra_summary_checks_cumulative_counts_predictors_and_separate_clocks(astra, field, value):
    from experiments.forge.test_performance_render import render_discovery_navigation
    root, _, document, registry = astra
    text = render_discovery_navigation(root, root / "reports/forge/technique-inventory.md")
    assert "228 full runs" in text and "470,400 new accepted updates" in text
    assert "12 Astra hypotheses FAIL" in text and "1,200 temporary predictor evaluations" in text
    document["discovery_summary"][field] = value
    pin = registry["publications"][-1]
    atomic_json(root / pin["path"], document)
    pin["sha256"] = file_hash(root / pin["path"])
    atomic_json(root / "reports/forge/test-performance-evidence/index.json", registry)
    with pytest.raises(ValueError, match="Astra cumulative"):
        render_discovery_navigation(root, root / "reports/forge/technique-inventory.md")


def test_existing_scoped_policy_registry_stays_separate_from_parent(evidence):
    from experiments.forge.trainer_families import scientific_row_hash
    root, publication, summary = evidence
    row = deepcopy(publication["rows"][0])
    row.update(candidate_id="scoped_recipe", candidate_revision="scoped_revision", trainer_family="method", attempt_ids=["scoped-attempt"])
    row["bindings"]["source_digest"] = "scoped-source"
    row["tasks"][0]["task_id"] = "test_a_policy"
    row["bindings"]["task_contracts"] = {"test_a_policy": next(iter(publication["task_contracts"]))}
    summary.update(candidate_id=row["candidate_id"], candidate_revision=row["candidate_revision"], attempt_id="scoped-attempt")
    summary["provenance"].update(source_digest="scoped-source", canonical_result_hash="scoped-result")
    summary["task_results"][0]["task_id"] = "test_a_policy"
    atomic_json(root / "reports/forge/technique-receipts/scoped-attempt.json", summary)
    document = {"schema": "forge_tier1_scoped_evidence_v1", "qualification_input": False, "parent_qualification_credit": False,
                "view": "policy", "source_digest": "scoped-source", "rows": {"method": row},
                "scientific_rows_sha256": {"method": scientific_row_hash(row)}, "receipt_result_hashes": {"scoped-attempt": "scoped-result"},
                "task_contracts": publication["task_contracts"]}
    path = root / "reports/forge/scoped-evidence.json"
    atomic_json(path, document)
    atomic_json(root / "reports/forge/scoped-publications.json", {"schema": "forge_scoped_publications_v1", "qualification_input": False,
        "cohorts": [{"path": "reports/forge/scoped-evidence.json", "sha256": file_hash(path), "view": "policy"}]})
    result = build_report(root, publication=publication)
    scoped = next(test for test in result["tests"] if test["task_id"] == "test_a_policy")
    assert scoped["coverage"]["optimizer_passes"] == 1
    assert scoped["selected_coverage"]["optimizer_passes"] == 0
    assert len(publication["rows"]) == 1


def test_confidence_lr_complete_screen_keeps_own_retention_failure_separate():
    from collections import Counter
    from pathlib import Path
    from experiments.forge.test_performance import _add_supplemental
    from experiments.forge.test_performance_render import render_discovery_navigation

    root = Path(__file__).resolve().parents[1]
    records = []
    _add_supplemental(Inputs(root),
                      lambda identity, label, selected, lane, recipe: label,
                      lambda task, conditions, lane, contract, match: task,
                      lambda task, recipe, record: records.append(record))
    screen = [r for r in records if r["cell_id"].startswith("pr399-confidence-lr-tier1-6-")]
    retention = [r for r in records if r["cell_id"].startswith("pr399-confidence-lr-own-retention-2-")]
    assert len(screen) == 6 and len(retention) == 2
    assert Counter(r["status"] for r in screen) == {"PASS": 6}
    assert Counter(r["status"] for r in retention) == {"PASS": 1, "FAIL": 1}
    assert len({r["result"]["attempt_id"] for r in screen + retention}) == 8
    assert all(r["full_protocol_complete"] is True and r["result"]["ordinary_qualification"] is False
               for r in screen + retention)
    assert {r["result"]["original_grade"]["gate_status"] for r in retention} == {"PASS", "FAIL"}
    gaussian = next(r for r in retention if r["status"] == "FAIL")
    assert gaussian["result"]["original_grade"]["evaluator_result"]["stationary_passes"] == 8
    assert gaussian["result"]["original_grade"]["evaluator_result"]["shift_hold_passes"] == 0
    rendered = render_discovery_navigation(root, root / "reports/forge/test-performance.md")
    assert "74 unique attempts" in rendered and "full Martyn" in rendered


def test_later_gaussian_corrections_retain_failure_and_unstarted_dependency():
    from collections import Counter
    from pathlib import Path
    from experiments.forge.test_performance_render import render_discovery_navigation

    root = Path(__file__).resolve().parents[1]
    reader = Inputs(root)
    registry = reader.read("reports/forge/test-performance-evidence/index.json")
    pins = {Path(pin["path"]).name: pin for pin in registry["publications"]}
    finite = reader.read(pins["pr399-gaussian-finite-cap-1.json"]["path"],
                         sha256=pins["pr399-gaussian-finite-cap-1.json"]["sha256"])
    orientation = reader.read(pins["pr399-critic-orientation-2.json"]["path"],
                              sha256=pins["pr399-critic-orientation-2.json"]["sha256"])
    assert len(finite["cells"]) == 1 and finite["cells"][0]["status"] == "FAIL"
    assert finite["cells"][0]["task_id"] == "gaussian1d_smoke"
    blocked = finite["discovery_summary"]["unstarted_retention"]
    assert (blocked["status"], blocked["attempts"], blocked["new_updates"], blocked["charges"]) == (
        "BLOCKED_UNSTARTED", 0, 0, 0)
    assert len(orientation["cells"]) == 2
    assert Counter(c["status"] for c in orientation["cells"]) == {"PASS": 1, "FAIL": 1}
    assert {c["task_id"] for c in orientation["cells"]} == {"gaussian1d_smoke", "gaussian1d_stability"}
    retention = next(c for c in orientation["cells"] if c["task_id"] == "gaussian1d_stability")
    original = retention["result"]["original_grade"]["evaluator_result"]
    assert (original["stationary_passes"], original["shift_hold_passes"], original["frozen_passes"]) == (5, 1, 0)
    assert len({c["result"]["attempt_id"] for c in finite["cells"] + orientation["cells"]}) == 3
    assert all(c["result"]["ordinary_qualification"] is False
               for c in finite["cells"] + orientation["cells"])
    rendered = render_discovery_navigation(root, root / "reports/forge/test-performance.md")
    assert "77 unique attempts" in rendered and "BLOCKED/UNSTARTED" in rendered


def test_signed_and_positive_responses_keep_endpoint_success_below_full_retention_gate():
    from pathlib import Path
    from experiments.forge.test_performance_render import render_discovery_navigation

    root = Path(__file__).resolve().parents[1]
    reader = Inputs(root)
    registry = reader.read("reports/forge/test-performance-evidence/index.json")
    pins = {Path(pin["path"]).name: pin for pin in registry["publications"]}
    documents = []
    for name in ("pr399-signed-confidence-2.json", "pr399-positive-evidence-2.json"):
        pin = pins[name]
        documents.append(reader.read(pin["path"], sha256=pin["sha256"]))
    sources = set()
    attempts = set()
    for document, expected_stationary, expected_shifted in zip(documents, (7, 14), (0, 2)):
        cells = document["cells"]
        assert [(c["task_id"], c["status"]) for c in cells] == [
            ("gaussian1d_smoke", "PASS"), ("gaussian1d_stability", "FAIL")]
        assert all(c["full_protocol_complete"] is True and c["result"]["ordinary_qualification"] is False
                   for c in cells)
        observed = cells[1]["result"]["original_grade"]["evaluator_result"]
        assert (observed["stationary_passes"], observed["shift_hold_passes"],
                observed["frozen_passes"], observed["reacquisition_status"]) == (
                expected_stationary, expected_shifted, 0, "FAIL")
        sources.update(document["condition_contracts"][c["conditions_sha256"]]["source_digest"] for c in cells)
        attempts.update(c["result"]["attempt_id"] for c in cells)
    assert len(sources) == 2 and len(attempts) == 4
    positive = documents[1]["cells"][1]["result"]["original_grade"]
    assert positive["gate_status"] == "FAIL" and positive["metrics"]["cdf_ks"] < .05
    rendered = render_discovery_navigation(root, root / "reports/forge/test-performance.md")
    assert "81 unique attempts" in rendered and "endpoint bounds" in rendered
