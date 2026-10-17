"""Global preparation must reject drift before bounded queue admission."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tier1_preparation", ROOT / "reports/forge/prepare_tier1_existing_configs.py")
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


@pytest.fixture
def inputs():
    campaign = {"id": "common", "budget_seconds": 20, "candidate_budget_seconds": 10}
    common = dict(view="main", execution_backend="cuda", campaign=campaign,
                  source_digest="source", policy_fingerprint="policy")
    row = dict(candidate_revision="revision", source_digest="source",
               submission_status="READY", submission_blockers=[],
               declared_worst_case_seconds=10, policy_fingerprint="policy")
    inventory = dict(common, through_tier=1,
                     candidates=[dict(row, candidate="idea", worst_case_seconds=10)])
    searches = {"study": dict(common, tuning_through_tier=1,
                              trials=[dict(row, candidate_id="config", unreused_worst_case_seconds=10)])}
    round_definition = dict(studies=["study"], candidate_ids=["config", "idea"],
                            configuration_ids=["config"], idea_ids=["idea"],
                            view="main", execution_backend="cuda", candidate_reservation_seconds=10,
                            worst_case_campaign_reservation_seconds=20)
    return round_definition, campaign, inventory, searches


def test_preparation_preserves_complete_roster_and_explicit_blockers(inputs):
    _, _, inventory, _ = inputs
    inventory["candidates"][0].update(submission_status="BLOCKED", submission_blockers=["unsupported host"])
    rows = preparation._verify_plans(*inputs)
    assert [row["candidate_id"] for row in rows] == ["config", "idea"]
    assert rows[1]["submission_blockers"] == ["unsupported host"]
    assert sum(row["declared_worst_case_seconds"] for row in rows) == 20


@pytest.mark.parametrize("change, message", [
    (lambda r, c, i, s: s["study"]["trials"][0].update(candidate_id="new"), "roster"),
    (lambda r, c, i, s: s["study"].update(source_digest="other"), "source"),
    (lambda r, c, i, s: s["study"]["trials"][0].update(source_digest="other"), "source"),
    (lambda r, c, i, s: s["study"].update(campaign={**c, "id": "other"}), "common campaign"),
    (lambda r, c, i, s: s["study"].update(tuning_through_tier=2), "Tier 1"),
    (lambda r, c, i, s: s["study"].update(policy_fingerprint="other"), "view policy"),
    (lambda r, c, i, s: c.update(budget_seconds=19), "allowances"),
    (lambda r, c, i, s: s["study"]["trials"][0].update(declared_worst_case_seconds=9), "allowances"),
])
def test_preparation_refuses_global_drift(inputs, change, message):
    changed = deepcopy(inputs)
    change(*changed)
    with pytest.raises(ValueError, match=message):
        preparation._verify_plans(*changed)


@pytest.mark.parametrize("drift, message", [
    ("roster", "roster"), ("source", "source"),
    ("policy", "view policy"), ("campaign", "common campaign"),
])
def test_global_drift_blocks_enqueue_before_any_admission(inputs, monkeypatch, tmp_path, drift, message):
    round_definition, campaign, inventory, searches = deepcopy(inputs)
    round_definition.update(id="round", campaign="campaign.json", cuda_model="gpu", view_revision=3,
                            required_denominator_by_tier=[1, 1, 1])
    if drift == "roster":
        searches["study"]["trials"][0]["candidate_id"] = "different"
    elif drift == "source":
        searches["study"]["source_digest"] = "different"
    elif drift == "policy":
        searches["study"]["policy_fingerprint"] = "different"
    else:
        searches["study"]["campaign"] = {**campaign, "id": "different"}
    monkeypatch.setattr(preparation, "read_json", lambda path:
                        campaign if path.name == "campaign.json" else round_definition)
    monkeypatch.setattr(preparation, "discover_candidate_ids", lambda root: ["config", "idea"])
    monkeypatch.setattr(preparation, "discover_techniques", lambda root: ["idea"])
    monkeypatch.setattr(preparation, "load_view", lambda root, view: {
        "id": "main", "revision": 3, "assignments": [
            {"task": str(tier), "importance": "required", "qualification_tier": tier}
            for tier in (1, 2, 3)]})
    monkeypatch.setattr(preparation, "Queue", lambda *args, **kwargs: object())
    monkeypatch.setattr(preparation, "plan_inventory", lambda *args, **kwargs: inventory)
    monkeypatch.setattr(preparation, "plan_search", lambda *args, **kwargs: searches["study"])
    admissions = []
    monkeypatch.setattr(preparation, "enqueue_inventory", lambda *args, **kwargs: admissions.append("idea"))
    monkeypatch.setattr(preparation, "enqueue_search", lambda *args, **kwargs: admissions.append("config"))
    with pytest.raises(ValueError, match=message):
        preparation.prepare(tmp_path, tmp_path / "queue", stage="enqueue")
    assert admissions == []
    assert list(tmp_path.iterdir()) == []
