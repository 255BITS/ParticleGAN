"""Pin the declared whole-recipe winner after original-source qualification."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, read_json
from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin, select_family_rows
from reports.forge.regenerate_technique_inventory import publish_current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    study = read_json(root / "reports/forge/bcap-six/readout.json")
    selection = study["selection"]
    assert selection["selection_complete"] and selection["qualified"]
    assert selection["required_pass_count"] == selection["required_total"] == 6
    board = read_json(root / "reports/forge/technique-inventory.json")
    rows = [row for row in board["configuration_rows"]
            if row["candidate_id"] == selection["selected_candidate_id"]
            and row["bindings"]["source_digest"] == selection["source_digest"]
            and row["runtime_cohort"]["execution_backend"] == "cuda"]
    assert len(rows) == 1
    row = rows[0]
    assert row["qualified_tier"] == 1 and len(row["attempt_ids"]) == 7
    path = root / CURRENT_SELECTION
    original = path.read_bytes()
    card = read_json(path)
    previous = next(pin for pin in card["selections"] if pin["trainer_family"] == row["trainer_family"])
    pin = family_row_pin(row, selection_kind="configured_standard",
        reason="All six unchanged revision-8 Tier1 requirements pass on this single constant-rate BCAP recipe. Selected by the bounded bcap-six-smoothing-v1 PASS-count/content-hash objective; later tiers remain unmeasured and public defaults unchanged.")
    if previous != pin:
        card.setdefault("historical_selections", []).append(deepcopy(previous))
        card["selections"] = [pin if item["trainer_family"] == row["trainer_family"] else item
                              for item in card["selections"]]
    select_family_rows(root, board["configuration_rows"], board, view_id=board["view"],
                       policy_fingerprint=board["policy_fingerprint"], selection_card=card)
    atomic_json(path, card)
    try:
        result = publish_current(root)
    except Exception:
        path.write_bytes(original)
        raise
    print(result)


if __name__ == "__main__":
    main()
