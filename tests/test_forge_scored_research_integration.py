"""The canonical regeneration path must retain each published research score."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_board_keeps_scored_research_and_existing_navigation():
    source = ROOT / "reports/forge/regenerate_technique_inventory.py"
    spec = importlib.util.spec_from_file_location("scored_research_regenerator", source)
    regenerator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regenerator)
    publication = json.loads((ROOT / "reports/forge/technique-inventory.json").read_text())
    original = deepcopy(publication)
    page = ROOT / "reports/forge/technique-inventory.md"
    first = regenerator._current_markdown(publication, ROOT, page)
    second = regenerator._current_markdown(publication, ROOT, page)
    assert publication == original
    assert first == second == page.read_text()
    for heading in (
        "## Preserved Atlas scores — separate original cohorts",
        "## Preserved catalog scores — original CPU geometry comparisons",
        "## Source-bound benchmark rows",
        "## G-only coherence: registered paired research",
        "## Confidence-LR correction: six-task screen and own retention",
    ):
        assert first.count(heading) == 1, heading
    assert "0 PASS / 1 FAIL / 2 INCOMPLETE / 4 BLOCKED across 7 questions" in first
    assert "FAIL → PASS | 14 → 15 | 3 → 5" in first
    assert "69 of 72 observation checks" in first
    assert "**1/2 tasks per arm**" in first
    assert "Stationary checks improve in all six pairs" in first
    assert "original word hold **PASS 25/25**" in first
    assert "BCAP initial Adam study" in first
    assert first.count("The [BCAP leader prior searches]") == 1
    assert first.count("## Scoped non-Atlas discoveries") == 1
    assert first.count("## Quota-transition and Astra portfolio follow-up") == 1
