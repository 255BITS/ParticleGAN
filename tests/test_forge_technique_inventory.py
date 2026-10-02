"""Inventory integration invariants without launching training."""
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.queue import Queue
from experiments.forge import technique_inventory as inventory


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
            "execution": {"steps": 80, "prior": prior, "host": "mode_hold"},
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
    return tmp_path


CAMPAIGN = {"id": "inventory-test", "budget_seconds": 100, "candidate_budget_seconds": 30}


def options(**changes):
    return {"view_id": "stability", "execution_backend": "cpu", "campaign": CAMPAIGN, **changes}


def alias(root, name):
    idea = read_json(root / "configs/forge/ideas/base.json")
    idea.update(id=name, hypothesis="same mechanism with another label")
    atomic_json(root / f"configs/forge/ideas/{name}.json", idea)


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
    assert "missing capability unsupported_inventory_probe" in row["submission_blockers"][0]
    assert row["required_tier_totals"] == {"1": 1, "2": 1, "3": 1}
    assert all(t["evidence_status"] == "UNKNOWN" for t in row["tasks"])
    assert not (checkout / "runs").exists()
    assert not (checkout / "reports").exists()


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
