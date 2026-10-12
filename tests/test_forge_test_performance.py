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
    assert len(cells) == 10
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
