"""Inventory integration invariants without launching training."""
from pathlib import Path
import shutil

import pytest

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.queue import Queue
from experiments.forge import technique_inventory as inventory
from forge_legacy_fixtures import pin_legacy


@pytest.fixture
def checkout(tmp_path):
    prior = {"kind": "mog", "sigma": .025, "standardize": False, "learnable": True}
    atomic_json(tmp_path / "configs/forge/defaults.json", {"protocol": "screening", "prior": prior})
    atomic_json(tmp_path / "configs/forge/protocols/screening.json", {
        "schema_version": 1, "id": "screening", "seed": 0,
        "rng": {"version": "forge-rng-v1"}, "scoring": {"weights": "live"}})
    atomic_json(tmp_path / "configs/forge/ideas/base.json", {
        "schema_version": 1, "id": "base", "goal": "stability", "hypothesis": "improve critic stability",
        "changed_factors": ["critic penalty"], "mechanism_class": "structural", "recipe_overrides": {},
        "claim_contract": {"sampling_law": "task_declared"}})
    assignments = []
    for tier in (1, 2, 3):
        atomic_json(tmp_path / f"configs/forge/tasks/t{tier}.json", {
            "schema_version": 1, "id": f"t{tier}", "adapter": "transfer_behavior",
            "execution": {"initializer": "deterministic_orthogonal", "steps": 80, "prior": prior, "host": "mode_hold"},
            "evaluation": {"kind": "transfer_sustained", "thresholds": [["score", ">=", 1]],
                           "sampling_contract_version": 1, "sampling_law": "public_prior_without_output_noise",
                           "eval_output_noise": "clean"},
            "resources": {"gpus": 0, "gpu_memory_mb": 10, "cpu_threads": 1, "timeout_seconds": 10},
            "requires_capabilities": ["named_rng"], "dependencies": []})
        assignments.append({"task": f"t{tier}", "qualification_tier": tier, "importance": "required", "order": 0})
    atomic_json(tmp_path / "configs/forge/views/stability.json", {
        "schema_version": 1, "id": "stability", "revision": 1,
        "goal": "stability", "assignments": assignments, "eligibility": {}})
    (tmp_path / "particlegan").mkdir()
    (tmp_path / "particlegan/fixture.py").write_text("mechanism = 1\n")
    pin_legacy(tmp_path)
    return tmp_path


CAMPAIGN = {"id": "inventory-test", "budget_seconds": 100, "candidate_budget_seconds": 30}


def options(**changes):
    return {"view_id": "stability", "execution_backend": "cpu", "campaign": CAMPAIGN, **changes}


def alias(root, name):
    idea = read_json(root / "configs/forge/ideas/base.json")
    idea.update(id=name, hypothesis="same mechanism with another label")
    atomic_json(root / f"configs/forge/ideas/{name}.json", idea)
    pin_legacy(root)  # These metadata fixtures explicitly retain their v1 controls.


def test_default_inventory_uses_one_current_family_config_and_explicit_rosters_remain_reproducible(checkout):
    from experiments.forge.trainer_families import CURRENT_SELECTION
    alias(checkout, "archived")
    atomic_json(checkout / "configs/forge/trainer-families.json", {"schema_version": 1, "families": [
        {"id": "baseline", "label": "Baseline", "candidates": ["base", "archived"], "canonical_candidate": "base"}]})
    atomic_json(checkout / CURRENT_SELECTION, {"schema_version": 1, "scope": "whole_candidate_family_current",
        "default_adoption": False, "view": "stability", "policy_fingerprint": "fixture", "selections": [
            {"trainer_family": "baseline", "candidate_id": "base", "execution_backend": "cpu",
             "selection_kind": "historical_incumbent", "reason": "Current whole configuration."}]})
    plan = inventory.plan_inventory(checkout, checkout / "runs", **options())
    assert plan["technique_count"] == 1
    assert plan["selection_scope"] == "one_current_configuration_per_family"
    assert [row["candidate"] for row in plan["candidates"]] == ["base"]
    assert plan["unrequested_technique_ids"] == ["archived"]
    assert not (checkout / "runs").exists()
    explicit = inventory.plan_inventory(checkout, checkout / "runs", technique_ids=["base", "archived"], **options())
    assert explicit["technique_count"] == 2
    assert "selection_scope" not in explicit


def test_plan_discovers_new_cards_preserves_denominators_and_writes_nothing(checkout):
    alias(checkout, "added")
    before = {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}
    result = inventory.plan_inventory(checkout, checkout / "runs", **options(through_tier=1))
    assert [r["candidate"] for r in result["candidates"]] == ["added", "base"]
    assert result["declared_worst_case_seconds"] == 20
    for row in result["candidates"]:
        assert row["required_tier_totals"] == {"1": 1, "2": 1, "3": 1}
        assert [t["permitted_by_tier_cap"] for t in row["tasks"]] == [True, False, False]
        assert [t["evidence_status"] for t in row["tasks"]] == ["UNKNOWN"] * 3
    assert before == {p: p.read_bytes() for p in checkout.rglob("*") if p.is_file()}


def test_known_blocked_first_task_spends_nothing_and_creates_no_attempt(checkout):
    task = read_json(checkout / "configs/forge/tasks/t1.json")
    task["requires_capabilities"].append("unsupported_inventory_probe")
    atomic_json(checkout / "configs/forge/tasks/t1.json", task)
    result = inventory.enqueue_inventory(checkout, checkout / "runs", **options())
    assert result["blocked_count"] == 1 and result["submitted_count"] == 0
    row = result["candidates"][0]
    assert row["submission_status"] == "BLOCKED"
    assert "required capability unavailable: unsupported_inventory_probe" in row["submission_blockers"][0]
    assert row["required_tier_totals"] == {"1": 1, "2": 1, "3": 1}
    assert all(t["evidence_status"] == "UNKNOWN" for t in row["tasks"])
    assert not (checkout / "runs").exists()
    assert not (checkout / "reports").exists()


def test_blocked_first_task_does_not_block_independent_current_tier(checkout):
    task = read_json(checkout / "configs/forge/tasks/t1.json")
    task["requires_capabilities"].append("unsupported_inventory_probe")
    atomic_json(checkout / "configs/forge/tasks/t1.json", task)
    view = read_json(checkout / "configs/forge/views/stability.json")
    view["assignments"][1]["qualification_tier"] = 1
    view["assignments"][2]["qualification_tier"] = 2
    atomic_json(checkout / "configs/forge/views/stability.json", view)
    result = inventory.enqueue_inventory(checkout, checkout / "runs", **options())
    assert result["submitted_count"] == 1 and result["blocked_count"] == 0
    row = result["candidates"][0]
    assert row["tasks"][0]["preflight_status"] == "BLOCKED"
    queue = Queue(checkout / "runs")
    claim = queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}])
    assert claim["job"]["task_id"] == "t2"
    assert sum(len(job["attempts"]) for job in queue.inspect()["jobs"].values()) == 1


def test_frozen_preflight_recomputes_forged_readiness_and_validates_healthy_peers(checkout):
    # Include real boundary modules in this otherwise small checkout. Admission
    # must recompute a removed cache against the frozen declarations/source.
    root = Path(__file__).resolve().parents[1]
    for name in ("preflight", "sampling", "hostprofiles", "initialization"):
        relative = f"experiments/forge/{name}.py"
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    view = read_json(checkout / "configs/forge/views/stability.json")
    view["assignments"][1]["qualification_tier"] = 1
    view["assignments"][2]["qualification_tier"] = 2
    atomic_json(checkout / "configs/forge/views/stability.json", view)
    task = read_json(checkout / "configs/forge/tasks/t1.json")
    task["evaluation"].pop("sampling_contract_version")
    atomic_json(checkout / "configs/forge/tasks/t1.json", task)
    req = inventory.resolve_idea(checkout, "base", view_id="stability", through_tier=3,
                                 execution_backend="cpu", freeze_source=True, queue_root=checkout / "runs")
    req["tasks"]["t1"]["preflight_blockers"] = []
    queue = Queue(checkout / "runs", report_root=checkout / "reports/forge")
    entry = queue.submit(req, CAMPAIGN)
    assert "sampling_contract_version" in entry["request"]["tasks"]["t1"]["preflight_blockers"][0]
    assert req["tasks"]["t1"]["preflight_blockers"] == []  # Caller was not mutated.
    assert queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}])["job"]["task_id"] == "t2"


def test_enqueue_is_restartable_and_duplicate_formulations_share_jobs(checkout):
    alias(checkout, "added")
    first = inventory.enqueue_inventory(checkout, checkout / "runs", **options())
    second = inventory.enqueue_inventory(checkout, checkout / "runs", **options())
    assert [r["request_id"] for r in first["candidates"]] == [r["request_id"] for r in second["candidates"]]
    state = Queue(checkout / "runs").inspect()
    assert len(state["submissions"]) == 2
    assert len(state["jobs"]) == 3
    assert all(len(job["subscribers"]) == 2 for job in state["jobs"].values())
    assert state["campaigns"][CAMPAIGN["id"]]["spent_seconds"] == 0
    assert all(job["attempts"] == [] for job in state["jobs"].values())
    assert len(first["source_digests"]) == 1


def test_all_rows_freeze_before_any_submission_and_source_drift_fails(checkout, monkeypatch):
    alias(checkout, "added")
    real = inventory.resolve_idea
    frozen = 0

    def changing_source(*args, **kwargs):
        nonlocal frozen
        if kwargs.get("freeze_source"):
            frozen += 1
            if frozen == 2:
                (checkout / "particlegan/fixture.py").write_text("mechanism = 2\n")
        return real(*args, **kwargs)

    monkeypatch.setattr(inventory, "resolve_idea", changing_source)
    with pytest.raises(ValueError, match="changed before submission"):
        inventory.enqueue_inventory(checkout, checkout / "runs", **options())
    assert Queue(checkout / "runs").inspect()["submissions"] == {}


def test_candidate_specific_extra_sources_keep_distinct_real_identities(checkout):
    alias(checkout, "extra")
    (checkout / "support.py").write_text("support = 1\n")
    path = checkout / "configs/forge/ideas/extra.json"
    idea = read_json(path)
    idea["source_files"] = ["support.py"]
    atomic_json(path, idea)
    pin_legacy(checkout)
    result = inventory.enqueue_inventory(checkout, checkout / "runs", **options())
    assert result["submitted_count"] == 2
    assert len(result["source_digests"]) == 2 and result["source_digest"] is None
    assert len(Queue(checkout / "runs").inspect()["jobs"]) == 6


def test_budget_is_explicit_and_immutable_and_queue_enforces_full_reservation(checkout):
    small = {**CAMPAIGN, "budget_seconds": 5, "candidate_budget_seconds": 5}
    plan = inventory.plan_inventory(checkout, checkout / "runs", **options(campaign=small))
    assert plan["campaign_budget_covers_declared_ceiling"] is False
    assert plan["candidates"][0]["candidate_budget_covers_declared_ceiling"] is False
    inventory.enqueue_inventory(checkout, checkout / "runs", **options(campaign=small))
    queue = Queue(checkout / "runs")
    assert queue.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}]) is None
    assert next(iter(queue.inspect()["submissions"].values()))["status"] == "blocked"
    with pytest.raises(ValueError, match="immutable"):
        inventory.enqueue_inventory(checkout, checkout / "runs", **options())
    with pytest.raises(ValueError, match="positive"):
        inventory.plan_inventory(checkout, checkout / "runs", **options(campaign={**CAMPAIGN, "budget_seconds": 0}))


def test_run_uses_ordinary_failed_smoke_stop_and_only_drains_inventory_campaign(checkout, monkeypatch):
    queue = Queue(checkout / "runs", grader=lambda task, raw: {"gate_status": "FAIL"})
    calls = []

    def drain_smoke(received, devices, *, campaign):
        calls.append((received, devices, campaign))
        claim = received.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}], campaign_filter=campaign)
        assert claim["job"]["task_id"] == "t1"
        atomic_json(Path(claim["worker"]["directory"]) / "terminal.json", {
            "token": claim["worker"]["token"], "attempt_status": "completed",
            "elapsed_seconds": 1, "result": {}})
        received.collect()
        assert received.claim([{"device": "cpu", "slot": 0, "memory_mb": 100}], campaign_filter=campaign) is None

    monkeypatch.setattr(inventory, "drain", drain_smoke)
    result = inventory.run_inventory(checkout, checkout / "runs", queue=queue, devices=["cpu"], **options())
    assert calls == [(queue, ["cpu"], CAMPAIGN["id"])]
    assert result["stage"] == "drained"
    assert result["candidates"][0]["submission_status"] == "blocked"
    state = queue.inspect()
    assert sum(len(job["attempts"]) for job in state["jobs"].values()) == 1
    assert state["campaigns"][CAMPAIGN["id"]]["spent_seconds"] == 1
    assert sum(job["status"] == "pending" for job in state["jobs"].values()) == 2
    with pytest.raises(ValueError, match="agree"):
        inventory.run_inventory(checkout, checkout / "runs", devices=["0"], **options())


def test_queue_validation_errors_propagate_without_fake_blocked_receipts(checkout, monkeypatch):
    queue = Queue(checkout / "runs")

    def refuse(*args):
        raise ValueError("scientific identity mismatch")

    monkeypatch.setattr(queue, "submit", refuse)
    with pytest.raises(ValueError, match="scientific identity mismatch"):
        inventory.enqueue_inventory(checkout, checkout / "runs", queue=queue, **options())
    assert queue.inspect()["submissions"] == {}


def test_grouped_endurance_budget_is_counted_once(checkout):
    path = checkout / "configs/forge/views/stability.json"
    view = read_json(path)
    view["assignments"].append({"task": "t4", "qualification_tier": 3, "importance": "required", "order": 1})
    atomic_json(path, view)
    task = read_json(checkout / "configs/forge/tasks/t3.json")
    task["id"] = "t4"
    atomic_json(checkout / "configs/forge/tasks/t4.json", task)
    for name in ("t3", "t4"):
        path = checkout / f"configs/forge/tasks/{name}.json"
        task = read_json(path)
        task["execution"]["execution_group"] = "endurance"
        task["execution"]["produces_state"] = True
        if name == "t4":
            task["dependencies"] = [{"task": "t3", "kind": "checkpoint"}]
            task["execution"]["continuation_of"] = "t3"
        atomic_json(path, task)
    result = inventory.plan_inventory(checkout, checkout / "runs", **options())
    assert result["declared_worst_case_seconds"] == 30
    assert result["candidates"][0]["required_tier_totals"] == {"1": 1, "2": 1, "3": 2}


@pytest.mark.parametrize('names,message', [([], 'nonempty'),(['base','base'],'unique'),
                                          (['unknown'],'undeclared'),('base','nonempty'),
                                          ([1],'nonempty')])
def test_explicit_idea_roster_rejects_invalid_or_undeclared_ids(checkout,names,message):
    before={p:p.read_bytes() for p in checkout.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match=message):
        inventory.plan_inventory(checkout,checkout/'runs',technique_ids=names,**options())
    assert before=={p:p.read_bytes() for p in checkout.rglob('*') if p.is_file()}
    assert not (checkout/'runs').exists()


def test_explicit_idea_roster_preserves_default_discovery_and_admission(checkout):
    alias(checkout,'added')
    full=inventory.plan_inventory(checkout,checkout/'runs',**options())
    selected=inventory.plan_inventory(checkout,checkout/'runs',technique_ids=['base'],**options())
    assert [row['candidate'] for row in full['candidates']]==['added','base']
    assert [row['candidate'] for row in selected['candidates']]==['base']
    assert selected['explicit_technique_ids']==['base']
    assert selected['unrequested_technique_ids']==['added']
    assert full['candidates'][1]==selected['candidates'][0]
    # The selected declaration remains subject to its own preflight/admission.
    task=read_json(checkout/'configs/forge/tasks/t1.json')
    task['requires_capabilities'].append('unsupported_inventory_probe')
    atomic_json(checkout/'configs/forge/tasks/t1.json',task)
    queued=inventory.enqueue_inventory(checkout,checkout/'runs',technique_ids=['base'],**options())
    assert queued['blocked_count']==1 and queued['submitted_count']==0
    assert queued['candidates'][0]['submission_status']=='BLOCKED'
    assert not (checkout/'runs').exists()
