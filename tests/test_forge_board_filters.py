"""Display filters preserve scientific state, denominators and board ordering."""
from copy import deepcopy

import pytest

from experiments.forge.board_filters import annotate_rows, filter_rows
from experiments.forge.contracts import atomic_json, stable_hash


def task(name, adapter, subfamily=None):
    return {"id": name, "adapter": adapter, "execution": {"host": name,
            "host_definition": {"family": subfamily} if subfamily else {}}}


@pytest.fixture
def fixture(tmp_path):
    catalog = {"image": task("image", "transfer_image", "intensity"),
               "vector": task("vector", "transfer_vector"), "old": task("old", "native100")}
    for name, declaration in catalog.items():
        atomic_json(tmp_path / f"configs/forge/tasks/{name}.json", declaration)
    current = {"candidate_id": "unfinished", "candidate_revision": "c", "evidence_scope": "current",
        "qualified_tier": 1, "status": "NOT_RUN", "counts": {"PASS": 1, "NOT_RUN": 1}, "attempt_ids": ["current"],
        "rank": None, "qualification": {"required_total": 2, "required_passed": 1, "eligible": False,
            "tasks": [{"task_id": "image", "gate_status": "PASS", "metrics": {"error": .1}},
                      {"task_id": "vector", "gate_status": "NOT_RUN"}]}, "cost": {"wall_seconds": 3}}
    pinned = {"candidate_id": "pinned", "candidate_revision": "p", "evidence_scope": "pinned",
        "qualified_tier": None, "status": "FAIL", "attempt_ids": ["pinned"], "cost": {"wall_seconds": 5},
        "task_results": [{"task_id": "old", "gate_status": "FAIL", "metrics": {"error": 2}}]}
    diagnostic = {"candidate_id": "diagnostic", "candidate_revision": "d", "evidence_scope": "calibration_diagnostic",
        "qualified_tier": None, "status": "DIAGNOSTIC", "attempt_ids": ["diagnostic"],
        "task_results": [{"task_id": "vector", "gate_status": "PASS"}]}
    history = {"candidate_id": "history", "record_id": "archived", "evidence_scope": "historical", "rank": None,
        "qualified_tier": None, "source": {"path": "reports/toy100/lrfree-search/archived/result.json"},
        "task_results": [{"task_id": "image", "gate_status": "PASS"},
                         {"task_id": "unknown-task", "gate_status": "FAIL"}]}
    proposed = {"candidate_id": "new", "evidence_scope": "current", "status": "BLOCKED", "qualified_tier": 0,
                "attempt_ids": [], "qualification": {"required_total": 1, "required_passed": 0,
                    "eligible": False, "tasks": [{"task_id": "vector", "gate_status": "BLOCKED"}]}}
    board = {"schema_version": 1, "view": "stability", "policy_fingerprint": "frozen-view",
        "rows": [current, pinned, diagnostic, history, proposed], "current_rows": [current, proposed],
        "pinned_rows": [pinned], "calibration_rows": [diagnostic], "historical_rows": [history],
        "ranking_note": "Tier only, raw metrics, no aggregate", "conflicts": [], "import_gaps": {"gaps": ["unknown"]}}
    attempts = [{"attempt_id": name, "valid_receipt": True,
                 "request": {"tasks": {host: catalog[host]}}, "task_results": [{"task_id": host, "gate_status": status}]}
                for name, host, status in (("current", "image", "PASS"), ("pinned", "old", "FAIL"),
                                           ("diagnostic", "vector", "PASS"))]
    return tmp_path, board, attempts


def annotate(fixture):
    root, board, attempts = fixture
    return annotate_rows(root, board, attempts=attempts, records=[])


def test_annotations_keep_science_untouched_and_category_references_shared(fixture):
    _, board, _ = fixture
    before = deepcopy(board)
    result = annotate(fixture)
    assert board == before
    for actual, original in zip(result["rows"], before["rows"]):
        assert {key: value for key, value in actual.items() if key != "display_metadata"} == original
    assert result["rows"][0] is result["current_rows"][0]
    assert result["rows"][3] is result["historical_rows"][0]
    result["rows"][0]["qualification"]["required_total"] = 99
    assert board["rows"][0]["qualification"]["required_total"] == 2


def test_loaded_json_board_relinks_separately_serialized_category_rows(fixture):
    import json
    root, board, attempts = fixture
    loaded = json.loads(json.dumps(board))
    assert loaded["rows"][0] is not loaded["current_rows"][0]
    annotated = annotate_rows(root, loaded, attempts=attempts, records=[])
    assert annotated["rows"][0] is annotated["current_rows"][0]
    filtered = filter_rows(annotated, evidence_quality="certified_pinned")
    assert filtered["rows"][0] is filtered["pinned_rows"][0]


def test_evidence_provenance_distinguishes_certificates_imports_and_unmeasured(fixture):
    rows = annotate(fixture)["rows"]
    assert rows[0]["display_metadata"]["evidence_quality"] == ["certified_current", "unmeasured"]
    assert rows[0]["display_metadata"]["unmeasured_task_ids"] == ["vector"]
    assert rows[1]["display_metadata"]["evidence_quality"] == ["certified_pinned"]
    assert rows[2]["display_metadata"]["evidence_quality"] == ["certified_diagnostic"]
    assert rows[3]["display_metadata"]["evidence_quality"] == ["imported_recorded"]
    assert rows[4]["display_metadata"]["evidence_quality"] == ["unmeasured"]
    assert rows[1]["status"] == "FAIL"  # Certification is not a passing result.


def test_recorded_task_family_wins_over_changed_live_catalog(fixture):
    root, _, _ = fixture
    atomic_json(root / "configs/forge/tasks/old.json", task("old", "transfer_image"))
    pinned = annotate(fixture)["pinned_rows"][0]
    assert pinned["display_metadata"]["families"] == ["native100"]
    assert pinned["display_metadata"]["task_families"]["old"]["basis"] == "recorded_request"


def test_historical_names_classify_without_inventing_verification_or_equivalence(fixture):
    row = annotate(fixture)["historical_rows"][0]
    metadata = row["display_metadata"]
    assert metadata["families"] == ["image", "unknown"]
    assert metadata["source_family"] == "lrfree/archived"
    assert metadata["task_families"]["image"]["basis"] == "catalog_name_only"
    assert not metadata["task_families"]["image"]["protocol_equivalence_asserted"]
    assert metadata["task_families"]["unknown-task"]["basis"] == "unknown"
    assert row["qualified_tier"] is None and row["rank"] is None


def test_family_filter_keeps_entire_incomplete_row_and_full_denominator(fixture):
    full = annotate(fixture)
    before = stable_hash(full)
    result = filter_rows(full, family="image", evidence_quality="certified_current")
    assert stable_hash(full) == before
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["qualification"]["required_total"] == 2
    assert row["qualification"]["required_passed"] == 1
    assert len(row["qualification"]["tasks"]) == 2
    assert not row["qualification"]["eligible"] and row["status"] == "NOT_RUN" and row["rank"] is None
    assert row["counts"] == {"PASS": 1, "NOT_RUN": 1} and row["cost"]["wall_seconds"] == 3
    assert result["current_rows"][0] is row and result["historical_rows"] == []
    assert result["policy_fingerprint"] == full["policy_fingerprint"]
    assert result["display_filter"]["hidden_rows"] == 4


def test_explicit_subfamily_and_source_family_filters_do_not_rank(fixture):
    board = annotate(fixture)
    images = filter_rows(board, family="intensity")
    assert [row["candidate_id"] for row in images["rows"]] == ["unfinished", "history"]
    history = filter_rows(board, family="lrfree/archived")
    assert [row["candidate_id"] for row in history["rows"]] == ["history"]
    assert images["ranking_note"] == board["ranking_note"]
    assert filter_rows(board, family="unknown")["rows"] == [board["historical_rows"][0]]


def test_invalid_missing_or_merely_stamped_receipts_never_become_certified(fixture):
    _, board, attempts = fixture
    attempts[0]["valid_receipt"] = False
    board["rows"][1]["attempt_ids"] = ["missing"]
    board["rows"][2]["attempt_ids"] = []
    rows = annotate(fixture)["rows"]
    assert "invalid_receipt" in rows[0]["display_metadata"]["evidence_quality"]
    assert "certified_current" not in rows[0]["display_metadata"]["evidence_quality"]
    for index in (1, 2):
        assert "unverified" in rows[index]["display_metadata"]["evidence_quality"]
        assert not any(label.startswith("certified_") for label in rows[index]["display_metadata"]["evidence_quality"])


def test_filters_are_or_within_and_between_dimensions_and_preserve_order(fixture):
    result = filter_rows(annotate(fixture), family=["image", "vector"],
                         evidence_quality=["certified_current", "certified_diagnostic"])
    assert [row["candidate_id"] for row in result["rows"]] == ["unfinished", "diagnostic"]
    assert result["current_rows"][0] is result["rows"][0]
    assert result["calibration_rows"][0] is result["rows"][1]


def test_explicit_attempts_avoid_certificate_reads(fixture, monkeypatch):
    def forbidden(_):
        raise AssertionError("already-validated attempts must not be read twice")
    monkeypatch.setattr("experiments.forge.knowledge._attempts", forbidden)
    assert len(annotate(fixture)["rows"]) == 5


def test_unknown_evidence_filter_and_unannotated_boards_fail_explicitly(fixture):
    _, board, _ = fixture
    with pytest.raises(ValueError, match="annotate"):
        filter_rows(board, family="image")
    with pytest.raises(ValueError, match="unknown evidence"):
        filter_rows(annotate(fixture), evidence_quality="verified_historical")
    empty = filter_rows(annotate(fixture), family="not-declared")
    assert empty["rows"] == [] and empty["display_filter"]["visible_rows"] == 0
    assert empty["import_gaps"] == board["import_gaps"]
