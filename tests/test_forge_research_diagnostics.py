"""Bounded research diagnostics retain gates, costs and recall without qualification."""
from copy import deepcopy

import pytest

from experiments.forge import knowledge, telemetry, views
from experiments.forge.api import CapabilityError
from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.planning import resolve_idea
from experiments.forge.promotion import validate_screening_submission
from experiments.forge.queue import Queue
from test_forge_decision_contracts import _ready, campaign
from test_forge_calibration import current as calibration_current, current_attempt
from test_forge_knowledge import setup as knowledge_setup, save_attempt
from test_forge_planning import checkout
from test_forge_views import simple_task, simple_view, receipt


@pytest.fixture
def diagnostic(checkout):
    root, ordinary, path = _ready(checkout)
    original_card = path.read_bytes()
    view = read_json(root / "configs/forge/views/stability.json")
    view.update(id="research_probe", evidence_scope="research_diagnostic")
    view["assignments"] = [{"task": "t1", "qualification_tier": 1, "importance": "diagnostic", "order": 0}]
    atomic_json(root / "configs/forge/views/research_probe.json", view)
    declaration = read_json(path)
    contract = declaration["decision_contract"]
    contract["status"] = "draft"
    contract["scope"]["view"] = view["id"]
    draft = resolve_idea(root, "successor", view_id=view["id"], declaration=declaration, execution_backend="cpu")
    expected = draft["decision_review"]["expected"]
    contract.update(status="ready", candidate_binding_sha256=expected["candidate_binding_sha256"],
                    substantive_delta=expected["substantive_delta"])
    contract["control"].update(task_map=expected["task_map"], binding_sha256=expected["control_binding_sha256"])
    contract["scope"].update({key: expected[key] for key in (
        "task_ids", "protocol_sha256", "source_digest", "execution_backend", "runtime_cohort_sha256", "jobs_sha256")})
    request = resolve_idea(root, "successor", view_id=view["id"], declaration=declaration,
        execution_backend="cpu", freeze_source=True, queue_root=root / "queue")
    assert path.read_bytes() == original_card
    return root, request, ordinary


def test_planner_isolates_diagnostic_job_and_ready_scope_admits_without_mutating_solution(diagnostic):
    root, request, ordinary = diagnostic
    validate_screening_submission(request)
    diagnostic_job = request["jobs"][0]
    original_job = ordinary["jobs"][0]
    assert diagnostic_job["science"] == {**original_job["science"], "evidence_use": "research_diagnostic"}
    assert diagnostic_job["compatibility_key"] != original_job["compatibility_key"]
    assert request["candidate_revision"] == ordinary["candidate_revision"]
    queue = Queue(root / "queue", report_root=root / "reports/forge", on_completion=None)
    queue.submit(request, campaign())
    assert len(queue.inspect()["jobs"]) == 1


@pytest.mark.parametrize("mutation", [
    lambda r: r["candidate"].pop("decision_contract"),
    lambda r: r["candidate"].update(schema_version=1),
    lambda r: r["candidate"]["decision_contract"].update(status="draft"),
    lambda r: r.pop("decision_admission"),
    lambda r: r["decision_review"].update(status="BLOCKED"),
    lambda r: r["view"].pop("evidence_scope"),
    lambda r: r["jobs"][0]["science"].pop("evidence_use"),
    lambda r: r["jobs"][0]["science"].update(evidence_use="calibration_diagnostic"),
    lambda r: r["jobs"][0].update(qualification_compatibility_key="ordinary"),
    lambda r: r["jobs"][0].update(compatibility_key="ordinary"),
    lambda r: r["protocol"].update(seed=1),
    lambda r: r["jobs"][0]["science"].update(seed=1),
    lambda r: r.update(calibration_lane={"registration_id": "forged"}),
])
def test_incomplete_or_forged_diagnostic_authority_cannot_enter_screening(diagnostic, mutation):
    _, request, _ = diagnostic
    request = deepcopy(request)
    mutation(request)
    with pytest.raises((CapabilityError, ValueError)):
        validate_screening_submission(request)


@pytest.mark.parametrize("importance,tier", [("required", 1), ("ranking", 1), ("diagnostic", 2)])
def test_research_views_cannot_create_qualification_tiers(importance, tier):
    task = simple_task("probe")
    view = simple_view(("probe", tier, importance))
    view["evidence_scope"] = "research_diagnostic"
    with pytest.raises(ValueError, match="only diagnostic tasks in Tier 1"):
        views.validate_view(view, {"probe": task})


def test_passing_research_evidence_retains_numerical_gate_but_never_qualifies():
    task = simple_task("probe")
    view = simple_view(("probe", 1, "diagnostic"))
    view["evidence_scope"] = "research_diagnostic"
    verdict = views.qualify(view, {"probe": task}, [receipt(task)])
    assert verdict["task_statuses"] == {"probe": "PASS"}
    assert verdict["status"] == "DIAGNOSTIC" and verdict["diagnostic_complete"]
    assert verdict["qualified_tier"] == 0
    assert not any(verdict[key] for key in ("eligible", "qualification_input", "qualification_reuse", "current_qualification_reuse"))
    assert verdict["cost_seconds"] == 2.


def test_saved_diagnostic_cannot_fill_an_ordinary_cell_even_if_keys_are_forged_equal(knowledge_setup, monkeypatch):
    root, request = knowledge_setup
    diagnostic = deepcopy(request)
    diagnostic["view"]["evidence_scope"] = "research_diagnostic"
    save_attempt(root, diagnostic)
    board = knowledge.board(root, "stability")
    assert board["current_rows"][0]["qualified_tier"] == 0
    assert not board["pinned_rows"] and not board["calibration_rows"]
    row, = board["research_rows"]
    assert row["evidence_scope"] == "research_diagnostic" and row["status"] == "DIAGNOSTIC"
    assert row["counts"] == {"PASS": 1} and row["cost"]["wall_seconds"] == 2.5
    assert row["qualified_tier"] == 0 and not row["qualification_reuse"] and not row["qualification_input"]
    compiled = []
    monkeypatch.setattr(knowledge, "compile_memory", lambda root, **kw: compiled.append(kw))
    record = knowledge.readout(root, "idea", "Diagnostic PASS only", "Original scope unchanged", "Inspect existing controls")
    assert record["evidence_scope"] == "research_diagnostic" and record["lifecycle"] == "concluded"
    assert not record["qualification_reuse"] and not record["qualification_input"] and not record["eligible"]
    assert "evidence" not in record["task_results"][0]
    assert compiled == [{"summaries_only": True}]
    assert knowledge.recall(root, "idea")[0]["qualification_input"] is False


def test_diagnostic_spend_is_counted_without_speed_or_qualification_credit(tmp_path):
    from test_forge_telemetry import request, save, state
    req = request()
    req["view"]["evidence_scope"] = "research_diagnostic"
    result = save(tmp_path, req, "diagnostic", tasks=("cheap", "quality"))
    report = telemetry.summarize_automation(tmp_path, state(tmp_path, [req], [result]))
    assert report["spend"]["wall_seconds"] == 4.
    assert not report["qualification_units"]
    assert report["cost_to_qualify"]["measured_outcomes"] == 0


def test_readout_rejects_combining_diagnostic_and_ordinary_results(knowledge_setup):
    root, request = knowledge_setup
    save_attempt(root, request, name="ordinary")
    diagnostic = deepcopy(request)
    diagnostic["view"]["evidence_scope"] = "research_diagnostic"
    save_attempt(root, diagnostic, name="diagnostic")
    with pytest.raises(ValueError, match="cannot combine"):
        knowledge.readout(root, "idea", "Mixed", "Different scopes", "Stop")


@pytest.mark.parametrize("summaries_only", [False, True])
def test_compile_keeps_diagnostic_recall_without_another_generated_goal_board(knowledge_setup, summaries_only):
    root, request = knowledge_setup
    diagnostic = deepcopy(request)
    diagnostic["view"]["evidence_scope"] = "research_diagnostic"
    diagnostic["view"]["id"] = "research_probe"
    atomic_json(root / "configs/forge/views/research_probe.json", diagnostic["view"])
    save_attempt(root, diagnostic)
    knowledge.readout(root, "idea", "Diagnostic PASS only", "Original scope unchanged", "Inspect existing controls")
    knowledge.compile_memory(root, summaries_only=summaries_only)
    manifest = read_json(root / "reports/forge/compilation.json")
    assert manifest["view_count"] == len(list((root / "configs/forge/views").glob("*.json")))
    assert not list((root / "reports/forge/leaderboards").glob("research_probe.*"))
    memory = (root / "reports/forge/EXPERIMENT_MEMORY.md").read_text()
    assert "research_diagnostic" in memory and "Diagnostic PASS only" in memory
    assert "[research_probe]" not in memory
    assert knowledge.recall(root, "idea")[0]["qualification_input"] is False


def test_matching_research_pass_cannot_become_a_calibration_control(calibration_current):
    from experiments.forge import calibration
    root, _, requests = calibration_current
    diagnostic = deepcopy(requests[0])
    diagnostic["view"]["evidence_scope"] = "research_diagnostic"
    current_attempt(root, diagnostic, "positive", score=1.)
    report = calibration.calibrate(root, "current")
    positive = report["matrix"][0]
    assert report["adoption"] == "BLOCKED"
    assert positive["classification"] == "unknown" and not positive["inputs"]
    assert positive["smoke"]["unknown_tasks"] == ["cheap"]
    assert positive["reference"]["unknown_tasks"] == ["quality"]
