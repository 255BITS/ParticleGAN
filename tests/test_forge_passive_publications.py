"""Committed-summary controls only; no model, sampler, scorer or raw inputs."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from experiments.forge import knowledge, publication_memory as memory
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION

ROOT = Path(__file__).resolve().parents[1]


def make_cut(root, directory="reports/forge/synthetic-prefix", *, completed=5, failed=(), metadata_paid=2.):
    """Private synthetic flags authenticate software, never numerical science."""
    protocol = read_json(ROOT / memory.PR223_PROTOCOL)
    destination = root / memory.PR223_PROTOCOL
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((ROOT / memory.PR223_PROTOCOL).read_bytes())
    rows = []
    for i, declared in enumerate(protocol["rows"]):
        definition = declared["original_definition"]
        accepted = i < completed
        status = ("FAIL" if i in failed else "PASS") if accepted else "NOT_RUN"
        media = None
        if accepted:
            gif = f"gifs/{declared['group']}-{declared['task']}.gif"
            path = root / directory / gif
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"GIF89a-private-pinned-byte-fixture")
            media = {"file": gif, "bytes": path.stat().st_size, "sha256": file_hash(path),
                     "frames": len(declared["media_steps"]), "actual_steps": declared["media_steps"],
                     "original_gate": status}
        paid = 1. if accepted else 0.
        cost = {"allowance_seconds": declared["proposed_inclusive_allowance_seconds"],
                "paid_wall_seconds": paid, "reserved_seconds": 0., "unmeasured_interrupt_reserved_seconds": 0.,
                "charged_seconds": paid, "overrun_seconds": 0., "certified": accepted,
                "terminal_status": "completed" if accepted else "missing", "completed_terminal": accepted}
        row = {"id": declared["id"], "group": declared["group"], "task": declared["task"],
               "case_sha256": stable_hash(definition), "required_host": definition["original_host"],
               "complete_recipe": declared["resolved_recipe"], "execution_status": status,
               "accepted_original_gate": status if accepted else None,
               "raw_reported_status": status if accepted else None, "full_protocol_complete": accepted,
               "cost": cost, "media": media}
        row.update({key: definition[key] for key in ("original_requirements", "observation_steps", "original_options", "sampling")})
        if accepted:
            row.update(completed_steps=definition["original_host"]["steps"],
                       metric_observations=len(definition["observation_steps"]),
                       final_metrics={"synthetic_recorded_flag": status})
            if declared["group"] == "native":
                row["native_gates"] = {"noisy": {"coverage": "PASS", "accuracy": status},
                                       "clean": {"coverage": "FAIL", "accuracy": "FAIL"}}
        rows.append(row)
    counts = {key: Counter(row["execution_status"] for row in rows).get(key, 0)
              for key in ("PASS", "FAIL", "NOT_RUN", "UNKNOWN", "BLOCKED", "INVALID", "INCOMPLETE", "BUDGET_EXCEEDED")}
    metadata = {"paid_wall_seconds": metadata_paid, "reserved_seconds": 0., "charged_seconds": metadata_paid,
                "overrun_seconds": 0., "blocked": False}
    costs = {"aggregate_cap_seconds": 10800, "case_caps_sum_seconds": 9810, "metadata_cap_seconds": 180,
             "export_grace_seconds": 0, "retries": 0, "case_paid_wall_seconds": float(completed),
             "case_reserved_seconds": 0., "case_charged_seconds": float(completed), "metadata": metadata,
             "charged_seconds": completed + metadata_paid, "remaining_seconds": 10800 - completed - metadata_paid,
             "halt_required": False}
    report = {"schema": memory.PR223_SCHEMA, "family": "atlas", "required": 19, "completed": completed,
              "source": {"origin_commit": "a" * 40, "digest": "b" * 64},
              "trusted_terminal_card": {"sha256": "c" * 64, "bytes": 100}, "rows": rows, "counts": counts,
              "accepted_original_gate_counts": {"PASS": counts["PASS"], "FAIL": counts["FAIL"], "UNAVAILABLE": 19 - completed},
              "all_required_case_evidence_complete": completed == 19,
              "raw_full_protocol_gate": ("FAIL" if failed else "PASS") if completed == 19 else "UNAVAILABLE",
              "accepted_full_retest_status": "PENDING_PUBLICATION_COST_FINALIZATION",
              "cost_snapshot_before_publication": {**deepcopy(costs), "metadata": {**metadata, "charged_seconds": 1.}},
              "metadata_phase_count_before_publication": 1,
              "authoritative_metadata_ledger_path": "/unavailable-original-machine/metadata-ledger.json",
              "claims": {key: False for key in ("historical_passes_are_current_credit", "current26_qualification",
                        "default_adoption", "speed_ranking", "named10500_cost_pooling")}}
    final = {"schema": "pg_pr223_final_publication_cost_v1", "costs": costs,
             "status": report["raw_full_protocol_gate"] if completed == 19 else "INCOMPLETE",
             "raw_full_protocol_gate": report["raw_full_protocol_gate"], "metadata_phase_count": 3,
             "authoritative_final_metadata_ledger": {"path": report["authoritative_metadata_ledger_path"],
                   "sha256": "d" * 64, "bytes": 100},
             "trusted_terminal_card_sha256": "c" * 64, "original_case_verdicts_unchanged": True,
             "no_old_cost_or_qualification_pooling": True}
    proof = {"schema": "pg_pr223_passive_verification_v1", "required": 19, "counts": counts,
             "accepted_original_gifs": completed, "trusted_card_sha256": "c" * 64,
             **{key: 0 for key in ("models", "draws", "scorer_calls", "grader_calls", "rendered_frames")}}
    value = dict(root=root, directory=directory, report=report, final=final, proof=proof)
    refresh(value)
    return value


def refresh(cut):
    root, directory = cut["root"], cut["directory"]
    atomic_json(root / directory / "results.json", cut["report"])
    sha = file_hash(root / directory / "results.json")
    cut["final"]["original_terminal_cut_results_sha256"] = sha
    cut["proof"]["results_sha256"] = sha
    for name, value in (("FINAL_COST.json", cut["final"]), ("verification.json", cut["proof"])):
        atomic_json(root / directory / name, value)
    (root / directory / "README.md").write_text("Private recorded metadata fixture; no scientific claim.\n")
    entry = memory.passive_publication_entry(root, directory)
    atomic_json(root / memory.PASSIVE_REGISTRY, {
        "schema": "forge_passive_publications_registry_v1", "qualification_input": False,
        "reuse": False, "cross_cohort_pooling": False, "publications": [entry],
        "latest": {memory.PR223_SCHEMA: entry["id"]}})
    return entry


def test_optional_absence_keeps_existing_memory_empty(tmp_path):
    assert memory.load_passive_publications(tmp_path) == {}
    assert memory.normalize(tmp_path) == []


def test_closed_prefix_preserves_full_denominator_and_final_cost(tmp_path, monkeypatch):
    cut = make_cut(tmp_path)
    monkeypatch.setattr(knowledge, "_attempts", lambda *a: pytest.fail("No raw hydration/regrading"))
    projection = memory.load_passive_publications(tmp_path)
    latest = projection["latest"][memory.PR223_SCHEMA]
    assert latest["counts"] == {"PASS": 5, "NOT_RUN": 14} and latest["status"] == "INCOMPLETE"
    assert latest["cost"]["charged_seconds"] == 7. and latest["cost"]["metadata"]["charged_seconds"] == 2.
    assert latest["actual_runtime_in_compact_result"] is False
    record = memory.normalize(tmp_path)[0]
    assert record["lifecycle"] == "closed_partial_cut" and len(record["task_results"]) == 19
    assert record["record_type"] == "published_study" and "trial_ids" not in record
    assert record["qualification_input"] is record["qualification_reuse"] is False
    assert not knowledge._records(tmp_path)[0]
    assert knowledge.recall(tmp_path, cut["directory"].split("/")[-1], "discriminator_stability")
    assert memory.PASSIVE_REGISTRY in {p.relative_to(tmp_path).as_posix() for p in memory.input_paths(tmp_path)}
    knowledge.compile_memory(tmp_path, summaries_only=True)
    assert knowledge.freshness(tmp_path)["fresh"]
    assert "## Compact publications and closed partial cuts" in (tmp_path / "reports/forge/EXPERIMENT_MEMORY.md").read_text()


@pytest.mark.parametrize("completed,failed,status", [(19, (), "PASS"), (19, (18,), "FAIL"), (5, (4,), "INCOMPLETE")])
def test_full_and_partial_original_verdicts_are_separate(tmp_path, completed, failed, status):
    make_cut(tmp_path, completed=completed, failed=failed)
    latest = memory.load_passive_publications(tmp_path)["latest"][memory.PR223_SCHEMA]
    assert latest["status"] == status
    assert latest["accepted_counts"]["FAIL"] == len(failed)
    assert latest["accepted_counts"]["UNAVAILABLE"] == 19 - completed


@pytest.mark.parametrize("change", ["counts", "gate", "recipe", "typed_recipe", "host", "cadence", "sampling",
    "partial_pass", "unknown_credit", "default", "speed", "cost", "cap", "final_status", "ledger_path",
    "metadata_rewind", "private_token", "native_clean_credit"])
def test_coherent_repin_cannot_change_protocol_or_accepted_scope(tmp_path, change):
    cut = make_cut(tmp_path, completed=19 if change == "native_clean_credit" else 5)
    report, final = cut["report"], cut["final"]
    row = report["rows"][0]
    if change == "counts": report["counts"]["PASS"] += 1
    elif change == "gate": row["original_requirements"] = []
    elif change == "recipe": row["complete_recipe"]["lr"] = .0053125
    elif change == "typed_recipe": row["complete_recipe"]["prior_reg"] = False
    elif change == "host": row["required_host"]["seed"] = 1234
    elif change == "cadence": row["observation_steps"] = [600]
    elif change == "sampling": row["sampling"] = "clean live"
    elif change == "partial_pass": final["status"] = "PASS"
    elif change == "unknown_credit": report["rows"][5]["accepted_original_gate"] = "PASS"
    elif change in {"default", "speed"}: report["claims"]["default_adoption" if change == "default" else "speed_ranking"] = True
    elif change == "cost": final["costs"]["case_paid_wall_seconds"] = 0.
    elif change == "cap": final["costs"]["aggregate_cap_seconds"] = 20000
    elif change == "final_status": final["trusted_terminal_card_sha256"] = "e" * 64
    elif change == "ledger_path": final["authoritative_final_metadata_ledger"]["path"] = "/foreign/ledger.json"
    elif change == "metadata_rewind": final["costs"]["metadata"]["charged_seconds"] = 0.
    elif change == "private_token": report["rows"][0]["attempt_token"] = "private-synthetic-value"
    else: report["rows"][-1]["native_gates"]["noisy"]["coverage"] = "FAIL"
    refresh(cut)
    with pytest.raises(ValueError): memory.load_passive_publications(tmp_path)


@pytest.mark.parametrize("role", ["result", "final_cost", "verification", "readout", "protocol", "gif"])
def test_missing_or_changed_pinned_inputs_fail_closed(tmp_path, role):
    cut = make_cut(tmp_path)
    entry = read_json(tmp_path / memory.PASSIVE_REGISTRY)["publications"][0]
    pin = entry["media"][0] if role == "gif" else entry[role]
    (tmp_path / pin["path"]).write_bytes(b"changed bytes")
    with pytest.raises(ValueError): memory.load_passive_publications(tmp_path)


def test_unsafe_path_and_duplicate_registration_rejected(tmp_path):
    make_cut(tmp_path)
    registry = read_json(tmp_path / memory.PASSIVE_REGISTRY)
    registry["publications"][0]["final_cost"]["path"] = "../outside/FINAL_COST.json"
    atomic_json(tmp_path / memory.PASSIVE_REGISTRY, registry)
    with pytest.raises(ValueError): memory.load_passive_publications(tmp_path)
    cut = make_cut(tmp_path)
    registry = read_json(tmp_path / memory.PASSIVE_REGISTRY)
    registry["publications"].append(deepcopy(registry["publications"][0]))
    atomic_json(tmp_path / memory.PASSIVE_REGISTRY, registry)
    with pytest.raises(ValueError): memory.load_passive_publications(tmp_path)


def test_later_full_cut_updates_latest_without_summing_prefix_cost(tmp_path):
    first = make_cut(tmp_path, "reports/forge/synthetic-prefix")
    old_entry = read_json(tmp_path / memory.PASSIVE_REGISTRY)["publications"][0]
    old_bytes = (tmp_path / old_entry["result"]["path"]).read_bytes()
    make_cut(tmp_path, "reports/forge/synthetic-full", completed=19, metadata_paid=3.)
    registry = read_json(tmp_path / memory.PASSIVE_REGISTRY)
    registry["publications"].insert(0, old_entry)
    atomic_json(tmp_path / memory.PASSIVE_REGISTRY, registry)
    projection = memory.load_passive_publications(tmp_path)
    assert len(projection["cuts"]) == 2
    assert projection["latest"][memory.PR223_SCHEMA]["cost"]["charged_seconds"] == 22.
    assert (tmp_path / old_entry["result"]["path"]).read_bytes() == old_bytes
    registry["latest"][memory.PR223_SCHEMA] = old_entry["id"]
    atomic_json(tmp_path / memory.PASSIVE_REGISTRY, registry)
    with pytest.raises(ValueError, match="rewinds"): memory.load_passive_publications(tmp_path)


def test_one_table_context_preserves_historical_and_all_ordinary_values(tmp_path):
    make_cut(tmp_path)
    spec = importlib.util.spec_from_file_location("private_passive_score_renderer", ROOT / "reports/forge/regenerate_technique_inventory.py")
    renderer = importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    report = read_json(ROOT / "reports/forge/technique-inventory.json")
    before = deepcopy(report)
    report["passive_publications"] = memory.load_passive_publications(tmp_path)
    report["original_pr223_atlas"] = renderer._original_pr223_atlas(report)
    text = renderer._current_markdown(report, tmp_path, tmp_path / "reports/forge/technique-inventory.md")
    table = [line for line in text.splitlines() if line.startswith("|")]
    assert text.count("| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |") == 1
    pins = read_json(ROOT / CURRENT_SELECTION)["selections"]
    families = {pin["trainer_family"] for pin in pins}
    assert len(pins) == len(families)
    represented = {row["trainer_family"] for row in report["rows"]}
    assert len(represented) == len(report["rows"]) and families <= represented
    for row in report["rows"]:
        if row["trainer_family"] not in families:
            assert not row["attempt_ids"] and row["qualified_tier"] == 0
            assert row["selection"]["qualified"] is False
    from experiments.forge.trainer_families import load_families
    registry = load_families(ROOT)
    visible = {name for name in families if registry[name].get("inventory_visible", True)
               and registry[name].get("reporting_family", name) == name}
    assert len([line for line in table if line.startswith("| **[")]) == len(visible)
    # Hidden configurations remain in the unchanged scientific rows while each
    # visible reporting family owns one table row.
    assert {row["trainer_family"] for row in report["rows"]} == represented
    assert "19/19" not in text and "5/19 PASS" not in text
    fresh = report["original_pr223_atlas"]["fresh_retest"]
    assert fresh["counts"] == {"PASS": 5, "NOT_RUN": 14} and fresh["status"] == "INCOMPLETE"
    from experiments.forge.family_reports import generated_pages
    family = generated_pages(tmp_path, report)[tmp_path / "reports/forge/families/atlas.md"]
    assert "Original Atlas recipe and serving-law evidence" in family and "fresh_retest" in family
    assert report["rows"] == before["rows"] and report["evidence_sources"] == before["evidence_sources"]
    assert report["original_pr223_atlas"]["counts"] == {"PASS": 19}
    for scope in ("recorded", "quality"):
        original = deepcopy(before)
        if scope == "recorded": original["recorded_policy"] = "configs/forge/view-history/discriminator_stability-v2.json"
        else: original["view"] = "quality_coverage"
        changed = deepcopy(original);changed["passive_publications"] = report["passive_publications"]
        changed["original_pr223_atlas"] = report["original_pr223_atlas"]
        assert renderer._current_markdown(changed, ROOT, ROOT / "reports/forge/older.md") == renderer._current_markdown(original, ROOT, ROOT / "reports/forge/older.md")
