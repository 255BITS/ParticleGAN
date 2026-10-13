"""A newly authorized bounded study cannot expand the closed registrations."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("five_track_workflow", Path(__file__).with_name("workflow.py"))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def manifest():
    return workflow.shared.read_json(Path(__file__).with_name("manifest.json"))


@pytest.mark.parametrize("field,value", [
    ("round", 2), ("max_candidates", 7), ("paid_cap_seconds", 5401),
    ("baseline_role", "replacement"),
])
def test_new_campaign_cannot_expand_or_replace_the_cached_control(field, value):
    value_manifest = manifest()
    value_manifest[field] = value
    with pytest.raises(ValueError):
        workflow.validate_manifest(value_manifest)


def test_missing_recipe_or_shortened_reservation_is_rejected():
    value = manifest()
    value["candidates"].pop()
    with pytest.raises(ValueError):
        workflow.validate_manifest(value)
    value = manifest()
    value["campaign"]["budget_seconds"] -= 1
    with pytest.raises(ValueError):
        workflow.validate_manifest(value)


def test_storage_rejects_a_new_queue_before_spending(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "ARCHIVE", tmp_path)
    (tmp_path / "queue").mkdir()
    with pytest.raises(ValueError, match="Reuse the original queue"):
        workflow.storage(tmp_path)


def test_saved_charges_still_enter_the_new_paid_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow.shared.Queue, "_available_budget", lambda *_a, **_kw: (True, None))
    queue = workflow.shared.BoundedQueue(tmp_path, paid_cap_seconds=5400)
    state = dict(charges=[dict(seconds=1805), dict(seconds=2400)],
                 jobs={"running": dict(reserved_seconds=300)})
    assert not queue._available_budget(state, {}, 900)[0]
    assert queue._available_budget(state, {}, 895)[0]


def test_changed_cached_baseline_binding_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "original_requests", lambda *_a:
                        ({"baseline": {"projection": {"source_digest": "changed"}}}, {}))
    monkeypatch.setattr(workflow.shared, "read_json", lambda _p:
                        {"candidates": {"baseline": {"card": {}, "source_digest": "original"}}})
    with pytest.raises(ValueError, match="must remain exact"):
        workflow.requests(tmp_path, tmp_path, {})


def test_missing_cached_attempt_cannot_launch(tmp_path, monkeypatch):
    projection = {"task_keys": {"test": "key"}}
    monkeypatch.setattr(workflow, "original_requests", lambda *_a:
                        ({"baseline": {"projection": projection}}, {}))
    monkeypatch.setattr(workflow.shared, "read_json", lambda p:
                        {"jobs": {}} if str(p).endswith("state.json") else
                        {"candidates": {"baseline": {"card": {}, **projection}}})
    with pytest.raises(ValueError, match="retained result"):
        workflow.requests(tmp_path, tmp_path, {})
