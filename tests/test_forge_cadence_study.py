"""Cadence-study protocol, cost and evidence-scope contracts; no training."""
from copy import deepcopy
import math
from pathlib import Path

import pytest

from experiments.forge import cadence_study as cadence
from experiments.forge import configuration_search as search
from experiments.forge.contracts import atomic_json, read_json, stable_hash
from experiments.forge.queue import Queue
from experiments.forge.promotion import validate_screening_submission
from test_forge_configuration_search import checkout, complete


@pytest.fixture
def cadence_spec(checkout):
    return {
        "schema_version": 1,
        "id": "penalty-cadence",
        "base_candidate": "base",
        "trainer_family": "family",
        "view": "stability",
        "through_tier": 3,
        "execution_backend": "cpu",
        "protocol": "screening",
        "intervals": [1, 4, 8, 16],
        "control_interval": 1,
        "hypothesis": "Lazy penalty application preserves the control's numerical gates.",
        "campaign": {
            "id": "penalty-cadence",
            "budget_seconds": 120,
            "candidate_budget_seconds": 30,
        },
    }


def _files(root):
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _coverage_view(checkout, cadence_spec, *, lanes=("coverage",)):
    view = read_json(checkout / "configs/forge/views/stability.json")
    view.update(id="stability-coverage", evidence_scope="research_diagnostic")
    for assignment in view["assignments"]:
        assignment.update(qualification_tier=1, importance="diagnostic")
    atomic_json(checkout / "configs/forge/views/stability-coverage.json", view)
    cadence_spec.update(coverage_view="stability-coverage", lanes=list(lanes))
    cadence_spec["campaign"].update(
        budget_seconds=120 * len(lanes), candidate_budget_seconds=30 * len(lanes))
    return view


def test_plan_is_readonly_and_retains_full_original_denominators(checkout, cadence_spec):
    before = _files(checkout)
    report = cadence.plan_study(checkout, checkout / "queue", cadence_spec)
    assert report["declared_worst_case_seconds"] == 120
    assert set(report["lanes"]) == {"ordinary"}
    ordinary = report["lanes"]["ordinary"]
    assert len(ordinary["trials"]) == 4
    assert ordinary["declared_worst_case_seconds"] == 120
    assert {trial["settings"]["reg_every"] for trial in ordinary["trials"]} == {1, 4, 8, 16}
    assert all(len(trial["tasks"]) == 3 for trial in ordinary["trials"])
    assert all(task["gate_status"] == "UNKNOWN"
               for trial in ordinary["trials"] for task in trial["tasks"])
    for name in ("t1", "t2", "t3"):
        assert len({task["comparison_binding"] for trial in ordinary["trials"]
                    for task in trial["tasks"] if task["task"] == name}) == 1
    assert before == _files(checkout)


@pytest.mark.parametrize("changed", ["evaluation", "resources"])
def test_comparison_binding_freezes_task_scoring_and_allowance(checkout, changed):
    task = read_json(checkout / "configs/forge/tasks/t1.json")
    original = search._comparison_binding(task)
    task["field_ownership"] = {"recipe_fields": {"reg_every": {"value": 16}}}
    assert search._comparison_binding(task) == original
    if changed == "evaluation":
        task["evaluation"]["thresholds"] = [["score", ">=", 2]]
    else:
        task["resources"]["timeout_seconds"] += 1
    assert search._comparison_binding(task) != original


def test_generated_axis_preserves_strength_prior_and_other_recipe_fields(checkout, cadence_spec):
    path = checkout / "configs/forge/ideas/base.json"
    base = read_json(path)
    base["recipe_overrides"].update(reg_coeff=.7, prior_reg=.01)
    atomic_json(path, base)
    spec = cadence.search_spec(checkout, cadence_spec)
    assert spec["schema_version"] == 2
    assert spec["grid"] == {"reg_every": [1, 4, 8, 16]}
    declarations = search._declarations(checkout, spec)
    control = next(card["resolved_configuration_recipe"] for card, settings in declarations
                   if settings["reg_every"] == 1)
    for card, settings in declarations:
        recipe = card["resolved_configuration_recipe"]
        assert recipe["reg_coeff"] == .7
        assert recipe["prior_reg"] == .01
        assert {key for key in recipe if recipe[key] != control[key]} <= {"reg_every"}
        assert settings == {"reg_every": recipe["reg_every"]}


@pytest.mark.parametrize("intervals", [[], [0, 1], [1, -4], [1, 4.0], [1, True],
                                       [1, 4, 4], [4, 8, 16], "1,4,8,16"])
def test_invalid_or_missing_control_intervals_rejected_before_writes(checkout, cadence_spec, intervals):
    cadence_spec["intervals"] = intervals
    before = _files(checkout)
    with pytest.raises(ValueError):
        cadence.plan_study(checkout, checkout / "queue", cadence_spec)
    assert before == _files(checkout)


@pytest.mark.parametrize("field,value", [
    ("grid", {"reg_coeff": [4, 8, 16]}),
    ("reg_coeff", 16),
    ("recipe_overrides", {"reg_every": 16, "reg_coeff": 16}),
    ("seed", 1),
    ("initializer", "identity"),
])
def test_no_hidden_strength_or_protocol_axes(checkout, cadence_spec, field, value):
    cadence_spec[field] = value
    with pytest.raises(ValueError):
        cadence.load_spec(checkout, cadence_spec)


def test_nonzero_protocol_seed_is_rejected(checkout, cadence_spec):
    path = checkout / "configs/forge/protocols/screening.json"
    protocol = read_json(path)
    protocol["seed"] = 7
    atomic_json(path, protocol)
    with pytest.raises(ValueError, match="seed"):
        cadence.plan_study(checkout, checkout / "queue", cadence_spec)


@pytest.mark.parametrize("budget", [
    {"budget_seconds": 119, "candidate_budget_seconds": 30},
    {"budget_seconds": 120, "candidate_budget_seconds": 29},
    {"budget_seconds": 0, "candidate_budget_seconds": 30},
])
def test_full_candidate_and_campaign_reservations_are_required(checkout, cadence_spec, budget):
    cadence_spec["campaign"].update(budget)
    with pytest.raises(ValueError):
        cadence.plan_study(checkout, checkout / "queue", cadence_spec)


def test_alternative_existing_technique_uses_same_cadence_workflow(checkout, cadence_spec):
    path = checkout / "configs/forge/ideas/base.json"
    base = read_json(path)
    base.update(trainer_family="r1r2")
    base["recipe_overrides"].update(critic_formulation="k3p", reg_arm="a_r1r2")
    atomic_json(path, base)
    cadence_spec["trainer_family"] = "r1r2"
    report = cadence.plan_study(checkout, checkout / "queue", cadence_spec)
    assert {trial["resolved_recipe"]["reg_arm"]
            for trial in report["lanes"]["ordinary"]["trials"]} == {"a_r1r2"}
    assert report["declared_worst_case_seconds"] == 120


def test_coverage_only_retains_original_tasks_and_separate_execution_roles(checkout, cadence_spec):
    _coverage_view(checkout, cadence_spec)
    source_view = read_json(checkout / "configs/forge/views/stability.json")
    before = _files(checkout)
    report = cadence.plan_study(checkout, checkout / "queue", cadence_spec)
    assert set(report["lanes"]) == {"coverage"}
    assert report["declared_worst_case_seconds"] == 120
    assert cadence.search_spec(checkout, cadence_spec, lane="coverage")["tuning_through_tier"] == 1
    for trial in report["lanes"]["coverage"]["trials"]:
        assert {task["task"] for task in trial["tasks"]} == {"t1", "t2", "t3"}
        assert {task["importance"] for task in trial["tasks"]} == {"diagnostic"}
        assert {task["qualification_tier"] for task in trial["tasks"]} == {1}
    assert source_view == read_json(checkout / "configs/forge/views/stability.json")
    assert before == _files(checkout)


def test_both_lanes_require_additive_reservations_and_cannot_alias_jobs(checkout, cadence_spec):
    _coverage_view(checkout, cadence_spec, lanes=("ordinary", "coverage"))
    report = cadence.plan_study(checkout, checkout / "queue", cadence_spec)
    assert report["declared_worst_case_seconds"] == 240
    trials = {lane: {trial["settings"]["reg_every"]: trial for trial in value["trials"]}
              for lane, value in report["lanes"].items()}
    for interval in (1, 4, 8, 16):
        ordinary, diagnostic = (trials[lane][interval] for lane in ("ordinary", "coverage"))
        assert ordinary["candidate_id"] == diagnostic["candidate_id"]
        assert {task["compatibility_key"] for task in ordinary["tasks"]}.isdisjoint(
            {task["compatibility_key"] for task in diagnostic["tasks"]})
    cadence_spec["campaign"].update(budget_seconds=120, candidate_budget_seconds=30)
    with pytest.raises(ValueError):
        cadence.plan_study(checkout, checkout / "queue", cadence_spec)


@pytest.mark.parametrize("mutation", ["missing", "foreign", "ordinary_scope", "required_role", "later_tier"])
def test_coverage_cannot_change_the_selected_question_or_claim_qualification(checkout, cadence_spec, mutation):
    view = _coverage_view(checkout, cadence_spec)
    if mutation == "missing":
        view["assignments"].pop()
    elif mutation == "foreign":
        task = read_json(checkout / "configs/forge/tasks/t3.json")
        task["id"] = "foreign"
        atomic_json(checkout / "configs/forge/tasks/foreign.json", task)
        view["assignments"][-1]["task"] = "foreign"
    elif mutation == "ordinary_scope":
        view.pop("evidence_scope")
    elif mutation == "required_role":
        view["assignments"][0]["importance"] = "required"
    else:
        view["assignments"][-1]["qualification_tier"] = 3
    atomic_json(checkout / "configs/forge/views/stability-coverage.json", view)
    with pytest.raises(ValueError):
        cadence.plan_study(checkout, checkout / "queue", cadence_spec)


@pytest.mark.parametrize("lanes", [[], ["unknown"], ["ordinary", "ordinary"], "coverage"])
def test_invalid_lane_rosters_are_rejected(checkout, cadence_spec, lanes):
    cadence_spec["lanes"] = lanes
    with pytest.raises(ValueError):
        cadence.load_spec(checkout, cadence_spec)


def test_coverage_requires_an_explicit_authored_view(checkout, cadence_spec):
    cadence_spec["lanes"] = ["coverage"]
    with pytest.raises(ValueError):
        cadence.load_spec(checkout, cadence_spec)


def test_manifest_id_and_explicit_path_resolve_the_same_study(checkout, cadence_spec):
    path = checkout / "configs/forge/cadence-studies/penalty-cadence.json"
    atomic_json(path, cadence_spec)
    expected = cadence.load_spec(checkout, cadence_spec)
    assert cadence.load_spec(checkout, "penalty-cadence") == expected
    assert cadence.load_spec(checkout, path) == expected


def test_enqueue_freezes_original_display_policy_without_starting_training(checkout, cadence_spec):
    _coverage_view(checkout, cadence_spec)
    source = read_json(checkout / "configs/forge/views/stability.json")
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    cadence.enqueue_study(checkout, queue.root, cadence_spec, queue=queue)
    registration = read_json(checkout / "reports/forge/cadence-studies/penalty-cadence-registration.json")
    assert registration["source_assignments"] == source["assignments"]
    state = queue.inspect()
    assert len(state["submissions"]) == 4
    assert len(state["jobs"]) == 12
    assert all(not job["attempts"] for job in state["jobs"].values())
    for submission in state["submissions"].values():
        request = submission["request"]
        assert request["view"]["evidence_scope"] == "research_diagnostic"
        assert request["protocol"]["seed"] == 0
        assert all(a["importance"] == "diagnostic" and a["qualification_tier"] == 1
                   for a in request["view"]["assignments"])


def test_registered_study_rejects_new_intervals_but_report_keeps_frozen_source(checkout, cadence_spec):
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    cadence.enqueue_study(checkout, queue.root, cadence_spec, queue=queue)
    (checkout / "particlegan/fixture.py").write_text("mechanism = 999\n")
    # A readout uses original request bindings even after the working source changes.
    report = cadence.report_study(checkout, queue.root, cadence_spec, queue=queue)
    assert report["stage"] == "reported"
    assert report["default_adoption"] is False
    changed = {**cadence_spec, "intervals": [1, 2, 4, 8]}
    with pytest.raises(ValueError, match="immutable|new id"):
        cadence.plan_study(checkout, queue.root, changed, queue=queue)


def test_coverage_source_policy_cannot_change_after_registration(checkout, cadence_spec):
    _coverage_view(checkout, cadence_spec)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    cadence.enqueue_study(checkout, queue.root, cadence_spec, queue=queue)
    path = checkout / "configs/forge/views/stability.json"
    source = read_json(path)
    source["assignments"][-1]["qualification_tier"] = 2
    atomic_json(path, source)
    with pytest.raises(ValueError, match="source view changed|new study id"):
        cadence.plan_study(checkout, queue.root, cadence_spec, queue=queue)


def test_coverage_runs_independent_deeper_peer_but_keeps_failed_parent_dependency(checkout, cadence_spec):
    path = checkout / "configs/forge/tasks/t2.json"
    dependent = read_json(path)
    dependent["dependencies"] = [{"task": "t1", "kind": "gate"}]
    atomic_json(path, dependent)
    _coverage_view(checkout, cadence_spec)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    cadence.enqueue_study(checkout, queue.root, cadence_spec, queue=queue)
    attempted = []
    for _ in range(8):
        # Synthetic certified envelopes test the queue; no worker/model starts.
        claimed = complete(queue, lambda claim: 0. if claim["job"]["task_id"] == "t1" else 1.)
        attempted.append(claimed["job"]["task_id"])
    assert attempted.count("t1") == attempted.count("t3") == 4
    assert "t2" not in attempted
    assert queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}]) is None
    report = cadence.report_study(checkout, queue.root, cadence_spec, queue=queue)
    lane = report["lanes"]["coverage"]
    assert lane["diagnostic_scope"] is True
    assert lane["qualification_credit"] is False
    assert all(not trial["full_view_qualified"] for trial in lane["trials"])


def test_diagnostic_search_plan_requires_actual_repository_registration(checkout, cadence_spec):
    _coverage_view(checkout, cadence_spec)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    requests, _ = search._freeze_search(
        checkout, queue.root, cadence.search_spec(checkout, cadence_spec, lane="coverage"), queue=queue)
    request, _ = requests[0]
    assert request["search_plan"]["declaration"]["schema_version"] == 2
    with pytest.raises(ValueError, match="registration"):
        validate_screening_submission(request, root=checkout, campaign=cadence_spec["campaign"])
    assert not queue.inspect()["submissions"]


@pytest.mark.parametrize("mutation", ["missing_repository", "search_plan", "source", "campaign", "namespace"])
def test_registered_diagnostic_search_cannot_forge_its_admission(checkout, cadence_spec, mutation):
    _coverage_view(checkout, cadence_spec)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    cadence.enqueue_study(checkout, queue.root, cadence_spec, queue=queue)
    request = deepcopy(next(iter(queue.inspect()["submissions"].values()))["request"])
    campaign = deepcopy(cadence_spec["campaign"])
    if mutation == "missing_repository":
        with pytest.raises(ValueError, match="repository-bound"):
            validate_screening_submission(request)
        return
    if mutation == "search_plan":
        request["search_plan"]["declaration"]["hypothesis"] = "A forged comparison"
    elif mutation == "source":
        request["source"]["digest"] = "0" * 64
    elif mutation == "campaign":
        campaign["budget_seconds"] += 1
    else:
        request["jobs"][0]["science"].pop("evidence_use")
    with pytest.raises(ValueError):
        validate_screening_submission(request, root=checkout, campaign=campaign)


def test_certified_fixture_metrics_and_cadence_reach_generated_comparison(checkout, cadence_spec):
    cadence_spec.update(through_tier=1)
    cadence_spec["campaign"].update(budget_seconds=40, candidate_budget_seconds=10)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    cadence.enqueue_study(checkout, queue.root, cadence_spec, queue=queue)
    for _ in range(4):
        claim = queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}])
        assert claim is not None
        interval = claim["request"]["candidate"]["resolved_recipe"]["reg_every"]
        score = float(interval)
        audit = {"interval": interval, "calls": 80, "applied": 80 // interval}
        # Software fixture only: certify a synthetic evaluator envelope so the
        # report must retrieve metrics through the normal original-receipt path.
        evidence = {
            "curve": [{"step": math.ceil(i * 80 / 24), "score": score} for i in range(1, 25)],
            "live": {"score": score},
            "sampling_contract_version": 1,
            "sampling_law": "public_prior_without_output_noise",
            "eval_output_noise": "clean",
            "guards": {"mechanism_audit": {"penalty_cadence": audit}},
        }
        raw = {"evidence": evidence, "metrics": {"score": score}}
        atomic_json(Path(claim["worker"]["directory"]) / "terminal.json", {
            "token": claim["worker"]["token"], "attempt_status": "completed", "elapsed_seconds": 2,
            "result": raw,
            "grading": {"raw_hash": stable_hash(raw), "source_digest": claim["request"]["source"]["digest"],
                        "grades": {"t1": {"gate_status": "PASS"}}},
        })
        assert queue.collect() == 1
    report = cadence.report_study(checkout, queue.root, cadence_spec, queue=queue)
    comparison = report["lanes"]["ordinary"]
    assert comparison["complete"]
    for trial in comparison["trials"]:
        interval = trial["value"]
        pair = next(pair for pair in trial["pairs"] if pair["task"] == "t1")
        assert pair["status"] == "AVAILABLE"
        assert {delta["metric"]: delta["delta"] for delta in pair["metric_deltas"]}["score"] == interval - 1
        assert pair["cadence"] == {"interval": interval, "calls": 80, "applied": 80 // interval}
    assert report["default_adoption"] is False
