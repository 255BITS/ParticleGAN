"""Integrated future sources do not rewrite or qualify archived research."""
import json
from pathlib import Path
from pr_integration_compat import before_pr_integration

ROOT = Path(__file__).resolve().parents[1]


def test_every_integration_binding_preserves_the_exact_parent_contract():
    receipts = list((ROOT / "reports/forge/pr-integration-20261108").glob("pr-*.json"))
    assert receipts
    for path in receipts:
        receipt = json.loads(path.read_text())
        for row in receipt["tasks"]:
            relative = row["previous_task_path"]
            current = json.loads((ROOT / relative).read_text())
            assert before_pr_integration(ROOT, current, relative) != current
