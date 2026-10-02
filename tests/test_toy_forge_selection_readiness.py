"""Software fixtures for current qualification, identity and read-only selection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.toy_audit import forge_selection_readiness as select
from experiments.forge import views
from experiments.forge.contracts import stable_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def task(name):
    return {"schema_version": 1, "id": name, "adapter": "software_fixture",
        "execution": {"steps": 24, "initializer": "deterministic_orthogonal",
                      "prior": {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}},
        "evaluation": {"kind": "transfer_sustained", "thresholds": [["error", "<=", 1.]], "scoring_weights": "live"},
        "resources": {"timeout_seconds": 60}, "requires_capabilities": [], "dependencies": []}


@pytest.fixture
def fixture(tmp_path):
    tasks = {name: task(name) for name in ("smoke", "quality", "diagnostic")}
    for name, value in tasks.items(): write(tmp_path / f"configs/forge/tasks/{name}.json", value)
    view = {"schema_version": 1, "id": "quality_coverage", "goal": "quality_coverage", "revision": 1,
        "eligibility": {}, "calibration": {"status": "provisional"},
        "assignments": [
            {"task": "smoke", "qualification_tier": 1, "importance": "required", "order": 0},
            {"task": "quality", "qualification_tier": 2, "importance": "required", "order": 0},
            {"task": "diagnostic", "qualification_tier": 2, "importance": "diagnostic", "order": 1}]}
    write(tmp_path / "configs/forge/views/quality_coverage.json", view)
    for name in ("alpha", "beta"): write(tmp_path / f"configs/forge/ideas/{name}.json", {"id": name})
    return tmp_path, tasks, view


def row(fixture, candidate="alpha", statuses=("PASS", "PASS", "PASS"), backend="cpu", model=None):
    _, tasks, view = fixture
    receipts = []
    for name, status in zip(tasks, statuses):
        if status == "NOT_RUN": continue
        if status == "BLOCKED": receipts.append({"task_id": name, "gate_status": "BLOCKED", "reason": "Unsupported software fixture"}); continue
        error = .5 if status == "PASS" else 2.
        receipts.append({"task_id": name, "evidence": {
            "observations": [{"step": step, "error": error} for step in range(1, 25)],
            "live": {"error": error}}})
    q = views.qualify(view, tasks, receipts)
    profiles = {backend: {"backend": backend, "model": model}}
    runtime = {"execution_backend": backend, "runtime": {"software_fixture": True}, "compute_profiles": profiles}
    recipe = {"software_fixture": True}
    return {"candidate_id": candidate, "candidate_revision": stable_hash(candidate),
        "evidence_scope": "current", "cohort": stable_hash([candidate, backend, model]),
        "runtime_cohort": runtime, "source_digest": "f" * 64,
        "status": q["status"], "qualified_tier": q["qualified_tier"], "qualification": q,
        "scientific_bindings": {"recipe": recipe, "recipe_sha256": stable_hash(recipe),
            "source_digest": "f" * 64, "protocol": {"software_fixture": True}}, "cost": {"wall_seconds": 0}}


def install(monkeypatch, fixture, rows, *, archived=None, conflicts=None):
    view = fixture[2]
    data = {"view": view["id"], "policy_fingerprint": views.view_fingerprint(view),
        "current_rows": deepcopy(rows), "pinned_rows": [], "calibration_rows": [], "historical_rows": [],
        "conflicts": conflicts or []}
    if archived: data.update(archived)
    monkeypatch.setattr(select.knowledge, "board", lambda *args, **kwargs: deepcopy(data))
    calls = []
    def resolve(root, candidate_id, **kwargs):
        calls.append(kwargs)
        r = next(r for r in rows if r["candidate_id"] == candidate_id
            and r["runtime_cohort"]["execution_backend"] == kwargs["execution_backend"])
        runtime = r["runtime_cohort"]
        return {"candidate_revision": r["candidate_revision"], "source": {"digest": r["source_digest"]},
            **deepcopy(runtime), "view": deepcopy(view), "candidate": {"id": candidate_id}}
    monkeypatch.setattr(select.planning, "resolve_idea", resolve)
    return data, calls


def test_true_screen_pass_remains_provisional_and_never_promotes_default(fixture, monkeypatch):
    root, _, _ = fixture
    _, calls = install(monkeypatch, fixture, [row(fixture)])
    result = select.build_readiness(root)
    assert result["decision"] == "QUALIFIED_OPTIONS_AVAILABLE"
    assert len(result["qualified_options"]) == 1 and not result["calibrated_options"]
    assert result["selected_config"] is None and "SEPARATE" in result["default_adoption"]
    assert result["summary"]["required_cells"] == 2
    assert result["rows"][0]["calibration"]["verified"] is False
    assert all(c["freeze_source"] is False and c["through_tier"] == 3 for c in calls)


@pytest.mark.parametrize("status", ["NOT_RUN", "BLOCKED", "FAIL"])
def test_every_missing_blocked_or_failed_required_task_stays_in_denominator(fixture, monkeypatch, status):
    install(monkeypatch, fixture, [row(fixture, statuses=("PASS", status, "PASS"))])
    result = select.build_readiness(fixture[0])
    assert result["decision"] == "NO_QUALIFIED_OPTION"
    assert not result["qualified_options"]
    assert result["summary"]["required_cells"] == 2
    assert result["summary"]["required_cell_statuses"] == {"PASS": 1, status: 1}


def test_diagnostic_failure_does_not_veto_required_pass_or_supply_its_place(fixture, monkeypatch):
    install(monkeypatch, fixture, [row(fixture, statuses=("PASS", "PASS", "FAIL"))])
    result = select.build_readiness(fixture[0])
    assert len(result["qualified_options"]) == 1
    assert result["summary"]["required_cell_statuses"] == {"PASS": 2}
    install(monkeypatch, fixture, [row(fixture, statuses=("NOT_RUN", "NOT_RUN", "PASS"))])
    assert select.build_readiness(fixture[0])["decision"] == "NO_QUALIFIED_OPTION"


def test_pinned_historical_and_calibration_passes_never_fill_current_cells(fixture, monkeypatch):
    archived = {name: [row(fixture)] for name in select.SCOPES}
    install(monkeypatch, fixture, [row(fixture, statuses=("NOT_RUN", "NOT_RUN", "NOT_RUN"))], archived=archived)
    result = select.build_readiness(fixture[0])
    assert not result["qualified_options"]
    assert result["summary"]["excluded_evidence_cohorts"] == {name: 1 for name in select.SCOPES}
    assert result["summary"]["required_cell_statuses"] == {"NOT_RUN": 2}


def test_conflicts_are_visible_and_block_cli_success_without_regrading_pass(fixture, monkeypatch, capsys):
    install(monkeypatch, fixture, [row(fixture)], conflicts=[{"reason": "conflicting durable receipts"}])
    assert select.main(["--root", str(fixture[0])]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "CONFLICTS_REQUIRE_REVIEW"
    assert result["qualified_options"] and result["selected_config"] is None


def test_backend_and_candidate_filters_keep_whole_exact_cohorts(fixture, monkeypatch):
    rows = [row(fixture, statuses=("PASS", "NOT_RUN", "PASS")),
        row(fixture, backend="cuda", model="software GPU"), row(fixture, candidate="beta")]
    install(monkeypatch, fixture, rows)
    result = select.build_readiness(fixture[0], backend="cpu", candidate_id="alpha")
    assert result["summary"]["all_current_cohorts"] == 3
    assert result["summary"]["shown_current_cohorts"] == 1
    assert result["summary"]["required_cells"] == 2 and not result["qualified_options"]
    assert result["rows"][0]["runtime_cohort"]["execution_backend"] == "cpu"


def test_multiple_qualified_configs_are_options_without_name_or_cost_winner(fixture, monkeypatch):
    a, b = row(fixture), row(fixture, candidate="beta")
    a["cost"]["wall_seconds"] = 100
    install(monkeypatch, fixture, [b, a])
    result = select.build_readiness(fixture[0])
    assert [r["candidate_id"] for r in result["qualified_options"]] == ["beta", "alpha"]
    assert result["selected_config"] is None


def test_configuration_trials_are_resolvable_options_with_their_own_declarations(fixture, monkeypatch):
    root = fixture[0]
    declaration = root / "configs/forge/configurations/alpha.json"
    declaration.parent.mkdir(parents=True)
    (root / "configs/forge/ideas/alpha.json").rename(declaration)
    install(monkeypatch, fixture, [row(fixture)])
    result = select.build_readiness(root, candidate_id="alpha")
    assert result["qualified_options"][0]["config"] == "configs/forge/configurations/alpha.json"
    assert result["selected_config"] is None


def test_claimed_accepted_calibration_needs_actual_cohort_bound_evidence(fixture, monkeypatch):
    fixture[2]["calibration"]["status"] = "accepted"
    write(fixture[0] / "configs/forge/views/quality_coverage.json", fixture[2])
    install(monkeypatch, fixture, [row(fixture)])
    result = select.build_readiness(fixture[0])
    assert result["declared_calibration"]["status"] == "accepted"
    assert not result["calibrated_options"] and not result["rows"][0]["calibration"]["verified"]


@pytest.mark.parametrize("field", ["candidate_revision", "source", "runtime"])
def test_stale_candidate_source_or_runtime_cannot_publish_selection(fixture, monkeypatch, field):
    install(monkeypatch, fixture, [row(fixture)])
    original = select.planning.resolve_idea
    def changed(*args, **kwargs):
        request = original(*args, **kwargs)
        request[field] = {"digest": "e" * 64} if field == "source" else "different"
        return request
    monkeypatch.setattr(select.planning, "resolve_idea", changed)
    with pytest.raises(ValueError, match="changed during"): select.build_readiness(fixture[0])


def test_omitted_required_task_and_wrong_scope_fail_closed(fixture, monkeypatch):
    r = row(fixture); r["qualification"]["tasks"].pop(0)
    install(monkeypatch, fixture, [r])
    with pytest.raises(ValueError, match="omitted"): select.build_readiness(fixture[0])
    r = row(fixture); r["evidence_scope"] = "pinned"
    install(monkeypatch, fixture, [r])
    with pytest.raises(ValueError, match="noncurrent"): select.build_readiness(fixture[0])


def test_actual_empty_board_stdout_creates_no_queue_or_source_files(fixture, monkeypatch, capsys):
    from experiments.forge.queue import Queue
    from experiments.forge import sources, runtime
    root = fixture[0]
    for path in (root / "configs/forge/ideas").glob("*.json"): path.unlink()
    def forbidden(*args, **kwargs): pytest.fail("readiness may not enqueue, initialize a queue, freeze sources or execute work")
    monkeypatch.setattr(Queue, "__init__", forbidden)
    monkeypatch.setattr(sources, "snapshot_source", forbidden)
    monkeypatch.setattr(runtime, "execute", forbidden)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert select.main(["--root", str(root)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["shown_current_cohorts"] == 0
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not (root / "runs").exists()


def test_unknown_candidate_or_view_rejected_without_writes(fixture, monkeypatch):
    root = fixture[0]
    install(monkeypatch, fixture, [row(fixture)])
    with pytest.raises(ValueError, match="unknown candidate"): select.build_readiness(root, candidate_id="typo")
    with pytest.raises(ValueError, match="unknown Forge view"): select.build_readiness(root, view_id="typo")
