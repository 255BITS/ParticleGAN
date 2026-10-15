"""The full leaderboard renderer must preserve scores under original receipts."""
from copy import deepcopy
from collections import Counter
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.forge.confidence_coherence_scores import (
    ORIGINAL, PAIRED, main_board_rows, validate_original_rows, validate_paired_results,
)


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def test_paired_scores_retain_each_original_source_and_failed_gate():
    data = read(PAIRED)
    before = deepcopy(data)
    pairs = validate_paired_results(data)
    assert data == before
    assert [p["candidate"]["stationary_passes"] for p in pairs] == [44, 57, 66, 58, 65, 56]
    assert [p["candidate"]["shift_hold_passes"] for p in pairs] == [20, 22, 12, 11, 9, 6]
    text = "\n".join(main_board_rows(ROOT, ROOT / "reports/forge/technique-inventory.md"))
    assert "35/72 → 44/72" in text and "11/24 → 6/24" in text
    assert text.count("**FAIL / FAIL**") == 6
    assert "1/2 tasks per arm" in text and "three and regress in three" in text
    assert "3e6ab1939a8f" in text and "cb1671ed7174" in text and "10df0bea444a" in text
    assert "/ml2/" not in text


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(qualification_input=True), "scope"),
    (lambda d: d["records"][0].update(source_digest="other"), "source"),
    (lambda d: d["records"][0]["gate"].update(status="FAIL"), "original grade"),
    (lambda d: d["paired_comparisons"][0]["candidate"].update(stationary_passes=72), "summary"),
    (lambda d: d["records"][3]["continuity"].update(history_reset=True), "continuation"),
])
def test_paired_publication_rejects_relabelled_or_unbound_evidence(mutation, match):
    data = read(PAIRED)
    mutation(data)
    with pytest.raises(ValueError, match=match):
        validate_paired_results(data)


def test_seed_zero_machine_rows_are_bound_to_original_receipts():
    board = read("reports/forge/source-bound-board.json")
    validate_original_rows(ROOT, board)
    board["rows"][0]["retention_phases"]["stationary"]["passed"] = 72
    with pytest.raises(ValueError, match="original observation"):
        validate_original_rows(ROOT, board)


def test_original_receipt_byte_change_is_not_silently_regraded(tmp_path):
    path = tmp_path / ORIGINAL
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(read(ORIGINAL)))
    with pytest.raises(ValueError, match="original receipt changed"):
        validate_original_rows(tmp_path, read("reports/forge/source-bound-board.json"))


def test_confidence_scores_keep_all_fifteen_attempts_and_source_cohorts():
    packets = [read("reports/forge/test-performance-evidence/pr399-" + suffix + ".json")
               for suffix in ("confidence-lr-tier1-6", "confidence-lr-own-retention-2",
                              "gaussian-finite-cap-1", "critic-orientation-2",
                              "signed-confidence-2", "positive-evidence-2")]
    assert Counter(c["status"] for p in packets for c in p["cells"]) == {"PASS": 10, "FAIL": 5}
    assert len({c["result"]["attempt_id"] for p in packets for c in p["cells"]}) == 15
    assert len({p["condition_contracts"][c["conditions_sha256"]]["source_digest"]
                for p in packets for c in p["cells"]}) == 4
    assert len(packets[0]["cells"]) == 6
    assert all(c["status"] == "PASS" for c in packets[0]["cells"])
    by_task = {c["task_id"]: c for c in packets[1]["cells"]}
    word = by_task["five_word_joint_hold"]["result"]["original_grade"]["evaluator_result"]
    assert word["confirmed_checks"] == word["observations"] == 25
    gaussian = by_task["gaussian1d_stability"]["result"]["original_grade"]
    assert gaussian["status"] == "FAIL"
    assert gaussian["evaluator_result"]["stationary_passes"] == 8
    assert gaussian["evaluator_result"]["shift_hold_passes"] == 0


def test_full_regenerator_preserves_confidence_coherence_and_selected_scores():
    path = ROOT / "reports/forge/regenerate_technique_inventory.py"
    spec = importlib.util.spec_from_file_location("regenerate_confidence_coherence_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    publication = read("reports/forge/technique-inventory.json")
    before = deepcopy(publication)
    page = ROOT / "reports/forge/technique-inventory.md"
    first = driver._current_markdown(publication, ROOT, page)
    assert first == driver._current_markdown(publication, ROOT, page)
    assert publication == before
    row = next(line for line in first.splitlines() if line.startswith("| ↳ †") and "Confidence LR;" in line)
    assert "[6/6]" in row and "[1(*)/21]" in row and "[0(*)/2]" in row and "[7(*)/29]" in row
    from experiments.forge.family_reports import render_family
    family = next(f for f in publication["family_progress"]["families"] if f["id"] == "bcap-pure")
    detail = render_family(ROOT, publication, family)
    assert detail.count("## G-only coherence: registered paired research") == 1
    assert detail.count("## Confidence-LR correction: six-task screen and own retention") == 1
    assert "**6/6 PASS**" in detail and "**PASS 25/25**" in detail
    assert "8/72 stationary checks" in detail and "0/24 shifted-hold checks" in detail
    assert "69 of 72 observation checks" in detail and "35 of 72 observation checks" in detail
    assert "35/72 → 44/72" in detail and "11/24 → 6/24" in detail
    assert "BCAP initial Adam study" in detail
