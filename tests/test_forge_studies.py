"""Plan/queue/readout integration using synthetic receipts; never train models."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.forge import decision_contracts as decisions, studies, configuration_search as search, knowledge
from experiments.forge.__main__ import main
from experiments.forge.contracts import atomic_json, file_hash, read_json, validate_idea
from experiments.forge.planning import new_idea, plan_summary, resolve_idea
from experiments.forge.queue import Queue
from test_forge_configuration_search import checkout, complete, spec


def ready_study(root, name="question-a", *, tier=1, cap=None):
    candidate_path = root / "configs/forge/ideas/reusable.json"
    if not candidate_path.exists():
        path = new_idea(root, "reusable", "base", goal="stability", hypothesis="A software fixture question")
        candidate = read_json(path)
        candidate.update(recipe_overrides={"lr": .006}, changed_factors=["exercised learning rate"])
        atomic_json(path, candidate)
    atomic_json(root / "reports/prior.json", {"record_id": "original-failure", "gate_status": "FAIL"})
    study = studies.scaffold(name, "reusable", "base", "stability", hypothesis="Compare one frozen recipe", execution_backend="cpu")
    study.update(status="ready", prior_evidence=[{"path": "reports/prior.json", "selector": [],
                 "identity": {"record_id": "original-failure"}, "use": "motivation_only"}],
                 prediction={"task_id": "t1", "metric": "score", "op": ">=", "threshold": .8, "phase": "final"},
                 falsifier={"task_id": "t1", "metric": "score", "op": "<", "threshold": .5, "phase": "final"},
                 competing_explanation="Endpoint quality can conceal late failure")
    study["scope"]["through_tier"] = tier
    study["campaign"].update(candidate_budget_seconds=cap or tier * 10, budget_seconds=40)
    atomic_json(studies.study_path(root, name), study)
    return study


def test_new_writes_recipe_and_human_study_without_technical_hashes(checkout):
    path = new_idea(checkout, "new", "base", goal="stability", hypothesis="A bounded question")
    candidate = read_json(path)
    study = studies.load_study(checkout, "new-study")
    assert candidate["schema_version"] == 3
    assert not {"hypothesis", "goal", "decision_contract", "prior", "lifecycle"} & candidate.keys()
    assert study["hypothesis"] == "A bounded question"
    assert "sha256" not in str(study)
    draft = resolve_idea(checkout, "new", study="new-study")
    assert draft["study_review"]["status"] == "BLOCKED"
    assert plan_summary(draft)["study_binding"]["actual_bindings"]["candidate"]["prior"]["t1"]["kind"] == "mog"
    with pytest.raises(ValueError, match="submission blocked"):
        resolve_idea(checkout, "new", study="new-study", freeze_source=True, queue_root=checkout / "queue")
    assert not (checkout / "queue").exists()


@pytest.mark.parametrize("field,value", [("decision_contract", {}), ("scope", {}), ("hypothesis", "Question"),
                                           ("goal", "stability"), ("prior", {}), ("campaign", {}), ("prediction", {})])
def test_recipe_rejects_experiment_owned_fields(checkout, field, value):
    ready_study(checkout)
    candidate = read_json(checkout / "configs/forge/ideas/reusable.json")
    candidate[field] = value
    with pytest.raises(ValueError, match="study/task-owned"):
        validate_idea(candidate)


def test_one_recipe_two_studies_generated_bindings_reuse_and_finite_budget(checkout):
    first = ready_study(checkout)
    before = (checkout / "configs/forge/ideas/reusable.json").read_bytes()
    second = ready_study(checkout, "question-b", tier=2)
    second.update(hypothesis="The same recipe also sustains a second task")
    second["prediction"]["task_id"] = "t2"
    atomic_json(studies.study_path(checkout, second["id"]), second)
    requests = [resolve_idea(checkout, "reusable", study=s["id"], freeze_source=True, queue_root=checkout / "queue")
                for s in (first, second)]
    a, b = requests
    assert a["candidate"] == b["candidate"]
    assert a["candidate_revision"] == b["candidate_revision"]
    assert a["jobs"][0] == b["jobs"][0]
    assert a["study_admission"] != b["study_admission"]
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    entries = [queue.submit(r, s["campaign"]) for r, s in zip(requests, (first, second))]
    assert entries[0]["request"]["request_id"] != entries[1]["request"]["request_id"]
    state = queue.inspect()
    assert len(state["jobs"][a["jobs"][0]["compatibility_key"]]["subscribers"]) == 2
    assert len(state["studies"]) == len(state["decision_rounds"]) == 2
    key = a["jobs"][0]["compatibility_key"]
    with queue.state() as state:
        state["jobs"][key]["attempts"].append({"attempt_id": "paid-infrastructure-failure"})
        state["charges"].append({"attempt_id": "paid-infrastructure-failure", "seconds": 1})
        assert decisions.available_round_budget(state, b, 10, job_key=key)[0] is False
        assert decisions.available_round_budget(state, b, 9, job_key=key)[0] is True
    assert (checkout / "configs/forge/ideas/reusable.json").read_bytes() == before


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("study"), lambda r: r.pop("study_admission"), lambda r: r.pop("study_review"),
    lambda r: r["study"].update(status="draft"),
    lambda r: r["candidate"].update(schema_version=1),
    lambda r: r["candidate"].pop("schema_version"),
    lambda r: r["tasks"]["t1"]["evaluation"].update(thresholds=[["score", ">=", 0]]),
    lambda r: r["tasks"]["t1"]["execution"].update(steps=1),
    lambda r: r["jobs"][0].update(budget_seconds=1),
    lambda r: r["view"]["assignments"][0].update(importance="diagnostic"),
    lambda r: r["source"].update(digest="a" * 64),
    lambda r: r["runtime"].update(changed="cohort"),
    lambda r: r["study_admission"].update(contract_sha256="b" * 64),
])
def test_missing_or_forged_authority_blocks_before_queue_mutation(checkout, mutation):
    study = ready_study(checkout)
    request = resolve_idea(checkout, "reusable", study=study["id"])
    mutation(request)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    with pytest.raises((ValueError, KeyError)):
        queue.submit(request, study["campaign"])
    assert not (checkout / "queue/queue/state.json").exists()


@pytest.mark.parametrize("change", ["recipe", "task", "evidence", "control", "study", "source"])
def test_changed_authoritative_inputs_block_stale_requests(checkout, change):
    study = ready_study(checkout)
    request = resolve_idea(checkout, "reusable", study=study["id"])
    paths = {"recipe": "configs/forge/ideas/reusable.json", "control": "configs/forge/ideas/base.json",
             "task": "configs/forge/tasks/t1.json", "evidence": "reports/prior.json",
             "study": "configs/forge/studies/question-a.json"}
    if change == "source":
        (checkout / "particlegan/fixture.py").write_text("mechanism = 2\n")
    else:
        path = checkout / paths[change]
        value = read_json(path)
        if change in {"recipe", "control"}:
            value["recipe_overrides"]["lr"] = .007
        elif change == "task":
            value["execution"]["prior"]["sigma"] = .03
        elif change == "evidence":
            value["gate_status"] = "INCOMPLETE"
        else:
            value["prediction"]["threshold"] = .9
        atomic_json(path, value)
    with pytest.raises(ValueError):
        decisions.validate_admission(request, study["campaign"], root=checkout)


def test_study_identity_and_round_caps_cannot_be_reset(checkout):
    study = ready_study(checkout)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    queue.submit(resolve_idea(checkout, "reusable", study=study["id"]), study["campaign"])
    study["hypothesis"] = "Changed prose"
    atomic_json(studies.study_path(checkout, study["id"]), study)
    with pytest.raises(ValueError, match="immutable after enqueue"):
        queue.submit(resolve_idea(checkout, "reusable", study=study["id"]), study["campaign"])
    study.update(id="new-question")
    study["campaign"].update(id="new-campaign", candidate_budget_seconds=11)
    atomic_json(studies.study_path(checkout, study["id"]), study)
    with pytest.raises(ValueError, match="round budget is immutable"):
        queue.submit(resolve_idea(checkout, "reusable", study=study["id"]), study["campaign"])


def test_prediction_is_separate_from_sustained_scientific_gate_and_saved_readout(checkout, monkeypatch):
    study = ready_study(checkout)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    request = resolve_idea(checkout, "reusable", study=study["id"], freeze_source=True, queue_root=queue.root)
    queue.submit(request, study["campaign"])
    claimed = complete(queue, .9)  # Prediction observed; unchanged task threshold >=1 FAIL.
    state = queue.inspect()
    result = state["jobs"][request["jobs"][0]["compatibility_key"]]["result"]
    assert result["task_results"][0]["gate_status"] == "FAIL"
    assert decisions.evaluate(claimed["request"], result["task_results"])["outcome"] == "prediction_observed"
    monkeypatch.setattr(knowledge, "compile_memory", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="--study"):
        knowledge.readout(checkout, "reusable", "Observed", "Control", "Stop")
    record = knowledge.readout(checkout, "reusable", "Observed", "Control", "Stop", study_id=study["id"])
    assert record["study_id"] == study["id"]
    assert record["decision_outcomes"][0]["qualification_input"] is False
    assert record["decision_outcomes"][0]["outcome"] == "prediction_observed"
    # Readout consults frozen requests, independent of current plan declarations.
    atomic_json(studies.study_path(checkout, study["id"]), {"unrelated": "changed after execution"})
    again = knowledge.readout(checkout, "reusable", "Observed", "Control", "Stop", study_id=study["id"])
    assert again == record


def test_v2_search_strips_legacy_contract_and_keeps_trials_reusable(checkout, spec):
    base_path = checkout / "configs/forge/ideas/base.json"
    base = read_json(base_path)
    base.update(schema_version=2, decision_contract=decisions.scaffold("base", "stability"))
    atomic_json(base_path, base)
    spec.update(schema_version=2, hypothesis="Compare existing active hyperparameters")
    spec.pop("protocol_hash")
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    report = search.enqueue_search(checkout, queue.root, spec, queue=queue)
    assert report["submitted_count"] == 4
    assert all(t["declaration"]["schema_version"] == 3 for t in report["trials"])
    assert all(not {"decision_contract", "hypothesis", "goal", "search_study_id", "search_report", "prior"}
               & t["declaration"].keys() for t in report["trials"])
    spec.update(id="second-search", hypothesis="A second independently bounded question")
    spec["campaign"]["id"] = "second-search"
    second = search.enqueue_search(checkout, queue.root, spec, queue=queue)
    assert [t["declaration"] for t in second["trials"]] == [t["declaration"] for t in report["trials"]]
    assert second["protocol_hash"] == report["protocol_hash"]
    assert len(queue.inspect()["jobs"]) == 12  # 3 tier jobs per recipe; no duplicated training work.


def test_search_readout_uses_frozen_study_hypothesis_not_recipe_metadata(checkout, spec, monkeypatch):
    spec.update(schema_version=2, hypothesis="Search study owns this hypothesis", grid={"lr": [.003]})
    spec.pop("protocol_hash")
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    summary = search.enqueue_search(checkout, queue.root, spec, queue=queue)
    complete(queue, 1.)
    candidate = summary["trials"][0]["candidate_id"]
    monkeypatch.setattr(knowledge, "compile_memory", lambda *args, **kwargs: {})
    record = knowledge.readout(checkout, candidate, "Observed", "Control", "Stop", study_id=spec["id"])
    assert record["hypothesis"] == spec["hypothesis"]
    assert record["goal"] == spec["view"]
    assert record["search_plan"]["declaration"] == spec
    assert "hypothesis" not in read_json(checkout / f"configs/forge/configurations/{candidate}.json")


@pytest.mark.parametrize("mutation", [lambda r: r.pop("search_plan"),
    lambda r: r["search_plan"]["declaration"].update(hypothesis="Forged question"),
    lambda r: r["view"]["assignments"][0].update(importance="diagnostic"),
    lambda r: r["source"]["files"].update(forged="0" * 64)])
def test_new_search_registration_cannot_authorize_forged_membership_or_source(checkout, spec, mutation):
    spec.update(schema_version=2, hypothesis="Bounded software fixture", grid={"lr": [.003]})
    spec.pop("protocol_hash")
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    summary = search.enqueue_search(checkout, queue.root, spec, queue=queue)
    request = deepcopy(queue.inspect()["submissions"][summary["trials"][0]["request_id"]]["request"])
    mutation(request)
    with pytest.raises(ValueError):
        queue.submit(request, spec["campaign"])


def test_cli_explicit_study_plan_enqueue_owns_campaign(checkout, capsys):
    study = ready_study(checkout)
    args = ["--root", str(checkout), "--queue-root", str(checkout / "queue")]
    assert main(args + ["plan", "reusable", "--study", study["id"]]) == 0
    assert '"study_binding"' in capsys.readouterr().out
    assert main(args + ["enqueue", "reusable", "--study", study["id"]]) == 0
    state = Queue(checkout / "queue").inspect()
    assert list(state["campaigns"]) == [study["campaign"]["id"]]


def test_two_study_readouts_share_original_receipt_without_rewriting_it(checkout, monkeypatch):
    first = ready_study(checkout)
    second = ready_study(checkout, "question-b")
    second["prediction"]["threshold"] = 1.2
    second["hypothesis"] = "Another prediction on the same unchanged science"
    atomic_json(studies.study_path(checkout, second["id"]), second)
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    for study in (first, second):
        request = resolve_idea(checkout, "reusable", study=study["id"], freeze_source=True, queue_root=queue.root)
        queue.submit(request, study["campaign"])
    complete(queue, 1.)
    originals = {path: path.read_bytes() for path in (checkout / "reports/forge/attempts").rglob("*.json")}
    monkeypatch.setattr(knowledge, "compile_memory", lambda *args, **kwargs: {})
    records = [knowledge.readout(checkout, "reusable", "Observed", "Control", "Stop", study_id=s["id"])
               for s in (first, second)]
    assert records[0]["record_id"] != records[1]["record_id"]
    assert records[0]["attempt_ids"] == records[1]["attempt_ids"]
    assert [r["decision_outcomes"][0]["outcome"] for r in records] == ["prediction_observed", "inconclusive"]
    assert all(r["task_results"][0]["gate_status"] == "PASS" for r in records)
    assert records[1]["decision_outcomes"][0]["provenance"][0]["producer_study_id"] == first["id"]
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("kind", ["unchanged", "small_cap", "bad_evidence", "bad_signature"])
def test_ready_label_cannot_authorize_an_unbounded_or_unreviewable_study(checkout, kind):
    study = ready_study(checkout)
    if kind == "unchanged":
        study["control"]["candidate_id"] = "reusable"
    elif kind == "small_cap":
        study["campaign"]["candidate_budget_seconds"] = 9
    elif kind == "bad_evidence":
        study["prior_evidence"][0]["identity"] = {"record_id": "invented"}
    else:
        study["prediction"]["task_id"] = "t3"
    atomic_json(studies.study_path(checkout, study["id"]), study)
    request = resolve_idea(checkout, "reusable", study=study["id"])
    assert request["study_review"]["status"] == "BLOCKED"
    with pytest.raises(ValueError, match="submission blocked"):
        resolve_idea(checkout, "reusable", study=study["id"], freeze_source=True, queue_root=checkout / "queue")
    assert not (checkout / "queue").exists()


def test_v3_research_diagnostic_preserves_ordinary_gates_and_evidence_isolation(checkout):
    study = ready_study(checkout)
    ordinary = resolve_idea(checkout, "reusable", study=study["id"])
    view = read_json(checkout / "configs/forge/views/stability.json")
    view.update(id="probe", evidence_scope="research_diagnostic")
    view["assignments"] = [{"task": "t1", "qualification_tier": 1, "importance": "diagnostic", "order": 0}]
    atomic_json(checkout / "configs/forge/views/probe.json", view)
    study["scope"]["view"] = "probe"
    atomic_json(studies.study_path(checkout, study["id"]), study)
    diagnostic = resolve_idea(checkout, "reusable", study=study["id"])
    assert diagnostic["jobs"][0]["compatibility_key"] != ordinary["jobs"][0]["compatibility_key"]
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    queue.submit(diagnostic, study["campaign"])
    assert diagnostic["tasks"]["t1"]["evaluation"] == ordinary["tasks"]["t1"]["evaluation"]


def test_task_local_binding_refusal_does_not_remove_runnable_peers(checkout):
    study = ready_study(checkout, tier=2)
    path = checkout / "configs/forge/tasks/t2.json"
    task = read_json(path)
    task["execution"]["host"] = "ae_gan_hold"
    task["execution"]["execution_path"] = "public_components"
    atomic_json(path, task)
    candidate_path = checkout / "configs/forge/ideas/reusable.json"
    candidate = read_json(candidate_path)
    candidate["recipe_overrides"]["prior_reg"] = 0.
    atomic_json(candidate_path, candidate)
    request = resolve_idea(checkout, "reusable", study=study["id"])
    assert request["study_review"]["status"] == "READY", request["study_review"]["blockers"]
    assert request["tasks"]["t2"]["preflight_blockers"]
    binding = request["study_review"]["actual_bindings"]["candidate"]["recipe"]["t2"]
    assert binding["status"] == "BLOCKED"
    assert request["study_review"]["expected"]["task_ids"] == ["t1", "t2"]
    queue = Queue(checkout / "queue", report_root=checkout / "reports/forge")
    request = resolve_idea(checkout, "reusable", study=study["id"], freeze_source=True, queue_root=queue.root)
    queue.submit(request, study["campaign"])
    assert queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}])["job"]["task_id"] == "t1"


def test_pure_bcap_recipe_matches_pinned_draft_training_definition():
    root = Path(__file__).resolve().parents[1]
    candidate = read_json(root / "configs/forge/ideas/bcap-pure-adam-example-v3.json")
    original = read_json(root / "tests/fixtures/forge/pure-bcap-b509c065.json")
    assert file_hash(root / "tests/fixtures/forge/pure-bcap-b509c065.json") == "f9e291389fb2207fb701696b4f3ae17c185f4eab60864f69a92977fa5979d7f6"
    assert candidate["recipe_overrides"] == original["recipe_overrides"]
    assert candidate["requires_capabilities"] == original["requires_capabilities"]
    assert candidate["claim_contract"] == original["claim_contract"]
    assert candidate["schema_version"] == 3 and "decision_contract" not in candidate
