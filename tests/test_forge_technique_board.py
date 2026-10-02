"""Technique tier cells preserve denominator, scientific cohort and receipt scope."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge import knowledge, technique_board as techniques
from experiments.forge.contracts import atomic_json, read_json, stable_hash


def task(name, *, prior="mog", sampling="public_prior_without_output_noise"):
    return {"schema_version": 1, "id": name, "adapter": "test",
            "execution": {"steps": 24, "prior": {"kind": prior, "sigma": .025 if prior == "mog" else 0,
                                                 "standardize": False, "learnable": True,
                                                 **({"exception_reason": "Finite-cloud fixture"} if prior == "particle_cloud" else {})},
                          "host_definition": {"initialization": {"method": "named_v1"}}},
            "evaluation": {"kind": "transfer_sustained", "thresholds": [["error", "<=", 1.]],
                           "scoring_weights": "live", "sampling_law": sampling},
            "resources": {"timeout_seconds": 300}, "dependencies": [], "requires_capabilities": []}


def view():
    return {"schema_version": 1, "id": "stability", "goal": "stability", "revision": 1,
            "eligibility": {}, "assignments": [
                {"task": "cheap", "qualification_tier": 1, "importance": "required", "order": 0},
                {"task": "quality", "qualification_tier": 2, "importance": "required", "order": 1},
                {"task": "hold", "qualification_tier": 3, "importance": "required", "order": 2},
                {"task": "diagnostic", "qualification_tier": 2, "importance": "diagnostic", "order": 3}]}


def current(name="k3p", backend="cuda", statuses=("PASS", "NOT_RUN", "NOT_RUN", "PASS")):
    rows = [{"task_id": name, "status": status} for name, status in zip(
        ("cheap", "quality", "hold", "diagnostic"), statuses)]
    return {"candidate_id": name, "candidate_revision": "revision-" + name,
            "cohort": backend + "-cohort-" + name, "evidence_scope": "current",
            "runtime_cohort": {"execution_backend": backend,
                               "compute_profiles": {backend: {"model": "hardware-" + backend}}},
            "qualification": {"tasks": rows}, "qualified_tier": 1,
            "status": "INCOMPLETE", "cost": {"wall_seconds": 2.5}, "attempt_ids": [backend]}


def board(*rows, archives=()):
    return {"view": "stability", "view_revision": 1, "policy_fingerprint": "view-hash",
            "current_rows": list(rows), "rows": [*rows, *archives], "calibration": {"status": "provisional"},
            "conflicts": []}


def test_cells_keep_full_tier_denominator_and_nonrequired_results_separate():
    source = board(current(statuses=("FAIL", "BLOCKED", "NOT_RUN", "PASS")))
    before = deepcopy(source)
    result = techniques.reduce_board(source, view())
    row = result["rows"][0]
    assert source == before
    assert row["tiers"] == {
        "1": {"passed": 0, "total": 1, "counts": {"FAIL": 1}},
        "2": {"passed": 0, "total": 1, "counts": {"BLOCKED": 1}},
        "3": {"passed": 0, "total": 1, "counts": {"UNKNOWN": 1}}}
    assert row["tasks"][2]["gate_status"] == "NOT_RUN"
    assert row["nonrequired_tasks"][0]["status"] == "PASS"
    text = techniques.render_markdown(result)
    assert "| Tier 1 | Tier 2 | Tier 3 |" in text
    assert "UNKNOWN means missing or unrun evidence" in text
    assert "0/1" in text


def test_incomplete_and_invalid_are_not_unknown_or_failed():
    result = techniques.reduce_board(board(current(statuses=("INVALID", "INCOMPLETE", "NOT_RUN", "PASS"))), view())
    assert result["rows"][0]["tiers"]["1"]["counts"] == {"INVALID": 1}
    assert result["rows"][0]["tiers"]["2"]["counts"] == {"INCOMPLETE": 1}
    assert result["rows"][0]["tiers"]["3"]["counts"] == {"UNKNOWN": 1}


def test_missing_qualification_and_unresolvable_candidates_keep_requirements():
    missing, blocked = current("missing"), current("unsupported")
    missing["qualification"]["tasks"] = []
    blocked.pop("qualification")
    blocked.update(status="BLOCKED", qualified_tier=0, blockers=[{"reason": "Unsupported host"}])
    result = techniques.reduce_board(board(missing, blocked), view())
    assert all(cell["counts"] == {"UNKNOWN": 1} for cell in result["rows"][0]["tiers"].values())
    assert all(cell["counts"] == {"BLOCKED": 1} for cell in result["rows"][1]["tiers"].values())
    assert result["rows"][1]["blockers"] == [{"reason": "Unsupported host"}]


def test_runtime_filter_keeps_scopes_separate_and_never_reuses_archive_pass():
    archives = [{"candidate_id": "k3p", "candidate_revision": "old-" + scope,
                 "cohort": scope, "evidence_scope": scope, "counts": {"PASS": 22},
                 "task_results": [{"task_id": "quality", "gate_status": "PASS"}],
                 "cost": {}} for scope in techniques.ARCHIVED_SCOPES]
    result = techniques.reduce_board(board(current(backend="cpu"), current(backend="cuda"), archives=archives),
                                     view(), execution_backend="cuda")
    assert len(result["rows"]) == 1
    assert result["rows"][0]["tiers"]["2"]["passed"] == 0
    for scope in techniques.ARCHIVED_SCOPES:
        assert result["archived_rows"][scope][0]["qualified_tier"] is None
        assert result["archived_rows"][scope][0]["qualification_reuse"] is False
        assert "tiers" not in result["archived_rows"][scope][0]
    assert "hardware-cuda" in techniques.render_markdown(result)
    assert "hardware-cpu" not in techniques.render_markdown(result)


def request(tasks):
    return {"candidate": {"id": "k3p", "resolved_recipe": {"reg_arm": "k3p"},
                          "prior": {"kind": "mog", "sigma": .025}, "initializer": "named_v1",
                          "claim_contract": {"sampling_law": "task_declared"}},
            "source": {"digest": "source"}, "protocol": {"seed": 0, "id": "screening"},
            "rng": {"stream": "fixed"}, "tasks": tasks,
            "jobs": [{"task_id": name, "compatibility_key": "key-" + name} for name in tasks]}


def test_bindings_keep_prior_initialization_full_budget_and_clean_noisy_laws():
    clean = request({"quality": task("quality")})
    noisy = deepcopy(clean)
    noisy["tasks"]["quality"]["execution"]["prior"] = task("quality", prior="particle_cloud")["execution"]["prior"]
    noisy["tasks"]["quality"]["execution"]["host_definition"]["initialization"]["method"] = "different"
    noisy["tasks"]["quality"]["evaluation"]["sampling_law"] = "noisy_served"
    noisy["tasks"]["quality"]["resources"]["timeout_seconds"] = 900
    a, b = current("clean"), current("noisy")
    a["scientific_bindings"], b["scientific_bindings"] = map(techniques.request_bindings, (clean, noisy))
    result = techniques.reduce_board(board(a, b), view())
    refs = [row["bindings"]["task_contracts"]["quality"] for row in result["rows"]]
    assert refs[0] != refs[1]
    contracts = [result["task_contracts"][ref] for ref in refs]
    assert contracts[0]["prior"] == clean["tasks"]["quality"]["execution"]["prior"]
    assert contracts[1]["initialization"] == {"method": "different"}
    assert contracts[1]["timeout_seconds"] == 900
    assert contracts[0]["steps"] == 24
    assert contracts[0]["sampling"]["sampling_law"] == "public_prior_without_output_noise"
    assert contracts[1]["sampling"]["sampling_law"] == "noisy_served"
    assert result["recipe_contracts"][result["rows"][0]["bindings"]["recipe_sha256"]] == {"reg_arm": "k3p"}
    assert result["rows"][0]["bindings"]["source_digest"] == "source"
    assert result["protocol_contracts"][result["rows"][0]["bindings"]["protocol_sha256"]]["seed"] == 0


def test_labels_never_merge_distinct_current_rows_or_change_order():
    result = techniques.reduce_board(board(current("a"), current("b")), view(), labels={"a": "Same", "b": "Same"})
    assert [row["candidate_id"] for row in result["rows"]] == ["a", "b"]
    assert [row["technique"] for row in result["rows"]] == ["Same", "Same"]


@pytest.fixture
def repository(tmp_path, monkeypatch):
    tasks = {name: task(name) for name in ("cheap", "quality", "hold", "diagnostic")}
    for name, declaration in tasks.items():
        atomic_json(tmp_path / f"configs/forge/tasks/{name}.json", declaration)
    atomic_json(tmp_path / "configs/forge/views/stability.json", view())
    atomic_json(tmp_path / "configs/forge/ideas/k3p.json", {"id": "k3p"})
    resolved = request(tasks)
    resolved.update(candidate_revision="exact-revision", runtime={"python": "fixed"}, view=view(),
                    execution_backend="cuda", compute_profiles={"cuda": {"model": "gpu"}})
    monkeypatch.setattr(knowledge, "_current_request", lambda *args: deepcopy(resolved))
    receipt = {"task_id": "cheap", "compatibility_key": "key-cheap", "gate_status": "PASS",
               "evidence": {"observations": [{"step": step, "error": 2.} for step in range(1, 25)],
                            "live": {"error": 2.}}, "cost": {"wall_seconds": 2.5}}
    result = {"attempt_id": "attempt", "candidate_revision": "exact-revision", "task_results": [receipt]}
    directory = tmp_path / "reports/forge/attempts/attempt"
    atomic_json(directory / "request.json", {"request": resolved})
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "evidence.json", {"result_hash": stable_hash(result),
                                             "source": resolved["source"], "runtime": resolved["runtime"]})
    return tmp_path, directory


def test_report_regrades_a_receipt_pass_using_current_metrics(repository):
    root, directory = repository
    before = (directory / "result.json").read_bytes()
    result = techniques.technique_board(root, "stability", execution_backend="cuda")
    assert result["rows"][0]["tiers"]["1"] == {"passed": 0, "total": 1, "counts": {"FAIL": 1}}
    assert result["recipe_contracts"][result["rows"][0]["bindings"]["recipe_sha256"]] == {"reg_arm": "k3p"}
    assert (directory / "result.json").read_bytes() == before
    assert not (root / "runs").exists()
    assert "scientific_bindings" not in knowledge.board(root, "stability")["current_rows"][0]


def test_tampered_receipts_get_no_pass_credit(repository):
    root, directory = repository
    result = read_json(directory / "result.json")
    result["task_results"][0]["evidence"]["live"]["error"] = .1
    atomic_json(directory / "result.json", result)
    report = techniques.technique_board(root, "stability")
    assert report["conflicts"]
    assert report["rows"][0]["tiers"]["1"]["passed"] == 0
    assert report["rows"][0]["tiers"]["1"]["counts"] == {"BLOCKED": 1}


def test_custom_report_prefix_is_deterministic_and_keeps_metrics_out(repository):
    root, _ = repository
    first = techniques.write_report(root, "stability", execution_backend="cuda", output_prefix="reports/inventory")
    markdown_path = Path(first["report"])
    before, modified = markdown_path.read_bytes(), markdown_path.stat().st_mtime_ns
    second = techniques.write_report(root, "stability", execution_backend="cuda", output_prefix="reports/inventory")
    assert first == second
    assert markdown_path.read_bytes() == before
    assert markdown_path.stat().st_mtime_ns == modified
    assert "(inventory.json)" in markdown_path.read_text()
    result = read_json(Path(first["json"]))
    assert result["provenance"]["input_digest"]
    assert "observations" not in Path(first["json"]).read_text()
    assert "evidence" not in result["rows"][0]


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="execution_backend"):
        techniques.reduce_board(board(current()), view(), execution_backend="tpu")
