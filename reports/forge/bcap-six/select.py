"""Pin the declared whole-recipe winner after original-source qualification."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json
from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin, select_family_rows, scientific_row_hash
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
    original_hash = file_hash(path)
    card = read_json(path)
    previous = next(pin for pin in card["selections"] if pin["trainer_family"] == row["trainer_family"])
    pin = family_row_pin(row, selection_kind="configured_standard",
        reason="All six unchanged revision-8 Tier1 requirements pass on this single constant-rate BCAP recipe. Selected by the bounded bcap-six-smoothing-v1 PASS-count/content-hash objective; later tiers remain unmeasured and public defaults unchanged.")
    if previous != pin:
        old_rows = [item for item in board["configuration_rows"] if family_row_pin(item,
            selection_kind=previous["selection_kind"], reason=previous["reason"],
            measurement_views=previous.get("measurement_views"),
            measurement_tasks=previous.get("measurement_tasks")) == previous]
        assert len(old_rows) == 1
        # Preserve the exact retired pin in the change receipt. Its scientific
        # row remains an unranked alternative in registered source evidence;
        # do not add a second selected row for this same runtime.
        card["selections"] = [pin if item["trainer_family"] == row["trainer_family"] else item
                              for item in card["selections"]]
    preserved_declarations = []
    pinned_families = {item["trainer_family"] for item in card["selections"]}
    for existing in board["rows"]:
        if (existing["trainer_family"] not in pinned_families
                and existing.get("selection", {}).get("selection_kind") == "unmeasured_declaration"):
            assert not existing["attempt_ids"] and existing["qualified_tier"] == 0
            # The publisher's implicit unanimous-source fallback no longer
            # applies once measured families select different sources. Keep
            # the exact already-published declaration without granting credit.
            retained = family_row_pin(existing, selection_kind="historical_incumbent",
                reason="Preserve the exact existing unmeasured CUDA declaration when measured family selections use different sources. No attempt, measured gate, qualification or default-adoption credit.")
            card["selections"].append(retained)
            preserved_declarations.append(retained)
    current_hashes = {scientific_row_hash(item) for item in board["configuration_rows"]}
    historical_rows = [item for item in board.get("historical_family_rows", [])
                       if scientific_row_hash(item) not in current_hashes]
    select_family_rows(root, board["configuration_rows"], board, view_id=board["view"],
                       policy_fingerprint=board["policy_fingerprint"], selection_card=card,
                       historical_rows=historical_rows)
    atomic_json(path, card)
    try:
        result = publish_current(root)
    except Exception:
        path.write_bytes(original)
        raise
    if previous != pin:
        atomic_json(root / "reports/forge/bcap-six/selection-change.json", {
            "schema_version": 1, "previous_card_sha256": original_hash, "previous_pin": deepcopy(previous),
            "selected_pin": pin, "study_id": study["study_id"], "default_adoption": False,
            "preserved_unmeasured_declarations": preserved_declarations,
            "selection_script_sha256": file_hash(Path(__file__)), "qualification_input": False})
    print(result)


if __name__ == "__main__":
    main()
