"""Print proposed whole-row measurement pins after the finite screen completes.

This reads an independently regraded source snapshot and the nine final search
reports. It never writes the selection card or promotes a default. Register the
source on CUDA first, then review this output before replacing the current card.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.configuration_search import select_configuration
from experiments.forge.contracts import read_json, stable_hash
from experiments.forge.trainer_families import (
    CURRENT_SELECTION, _current_pin, family_for_candidate, family_row_pin,
)
from experiments.forge.views import load_view, view_fingerprint
from reports.forge.regenerate_technique_inventory import _validate_published_row


FAMILIES = {
    "bcap-sgda", "bcap-nsgda-global", "bcap-nsgda-layer", "bcap-ada-nsgda",
    "bcap-dualnorm", "bcap-dualnorm-d-only", "bcap-particle-rownorm-only",
}


def proposed_selection(root, snapshot, source_commit):
    """Bind each selected configuration to verified ordinary science in full."""
    root = Path(root)
    if (snapshot.get("publication_scope") != "frozen_source"
            or snapshot.get("frozen_source", {}).get("commit") != source_commit):
        raise ValueError("selection requires the independently regraded exact executed source snapshot")
    claimed = snapshot.get("provenance", {}).get("input_digest")
    unsigned = deepcopy(snapshot)
    unsigned.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(unsigned):
        raise ValueError("source snapshot publication digest mismatch")
    view = load_view(root, "discriminator_stability")
    if snapshot.get("policy_fingerprint") != view_fingerprint(view):
        raise ValueError("source snapshot differs from the current view policy")
    card = read_json(root / CURRENT_SELECTION)
    if card.get("policy_fingerprint") != snapshot["policy_fingerprint"]:
        raise ValueError("retained selection card differs from the source policy")
    original_incumbent = next(pin for pin in card["selections"] if pin["trainer_family"] == "bcap-pure")
    by_family, by_candidate = {}, {}
    paths = sorted((root / "reports/forge/configuration-search").glob("bcap-optim-*-tier1-v1.json"))
    if len(paths) != 9:
        raise ValueError("the initial screen requires all nine recorded search reports")
    allowed_sources = set(snapshot["frozen_source"]["source_digests"])
    for path in paths:
        report = read_json(path)
        if report.get("input_digest") != stable_hash({key: value for key, value in report.items() if key != "input_digest"}):
            raise ValueError("search report digest mismatch")
        if (set(report.get("source_digests", [])) != allowed_sources
                or report.get("policy_fingerprint") != snapshot["policy_fingerprint"]
                or not report.get("selection", {}).get("selection_complete")):
            raise ValueError("all searches must finish under the exact executed source and policy")
        for trial in report["trials"]:
            tier = [task for task in trial["tasks"] if task["qualification_tier"] == 1]
            if any(task["gate_status"] not in {"PASS", "FAIL", "INVALID", "INCOMPLETE", "BLOCKED"} for task in tier):
                raise ValueError("complete every independent current-tier peer before selecting measurements")
            if trial["candidate_id"] in by_candidate:
                raise ValueError("the finite screen contains duplicate candidate identities")
            by_candidate[trial["candidate_id"]] = trial
            by_family.setdefault(report["trainer_family"], []).append(trial)
    if len(by_candidate) != 41 or FAMILIES - by_family.keys():
        raise ValueError("the initial screen must contain its exact 41 configurations and seven new families")
    proposed = deepcopy(card)
    proposed["selections"] = [pin for pin in proposed["selections"] if pin["trainer_family"] not in FAMILIES]
    for family in sorted(FAMILIES):
        # The dualnorm family includes both zero- and positive-momentum studies.
        selection = select_configuration(by_family[family], 1)
        candidate_id = selection["selected_candidate_id"]
        matches = [row for row in snapshot["rows"] if row["candidate_id"] == candidate_id
                   and row.get("runtime_cohort", {}).get("execution_backend") == "cuda"
                   and row.get("bindings", {}).get("source_digest") in allowed_sources]
        if len(matches) != 1:
            raise ValueError("selected configuration lacks one exact independently regraded CUDA row")
        row = deepcopy(matches[0])
        _validate_published_row(root, snapshot, row)
        if family_for_candidate(root, candidate_id, {"trainer_family": family})["id"] != family:
            raise ValueError("selected candidate differs from the registered family")
        row["trainer_family"] = family
        trial = by_candidate[candidate_id]
        if (row.get("candidate_revision") != trial["candidate_revision"]
                or row.get("runtime_cohort") != trial["runtime_cohort"]):
            raise ValueError("selected row differs from its frozen search revision or runtime")
        reason = ("Finite optimizer-only BCAP screen: required Tier 1 PASS count descending, then "
                  "configuration hash ascending, over complete whole recipes under one executed source. "
                  "This pins a current measurement; calibration, confirmation and default adoption remain separate.")
        pin = family_row_pin(row, selection_kind="current_measurement", reason=reason,
                             measurement_views=["discriminator_stability"])
        _current_pin(root, family, [row], pin, view_id="discriminator_stability", catalogs=snapshot)
        proposed["selections"].append(pin)
    assert next(pin for pin in proposed["selections"] if pin["trainer_family"] == "bcap-pure") == original_incumbent
    return proposed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-commit", required=True, help="full executed Git commit")
    args = parser.parse_args()
    snapshot_path = args.snapshot if args.snapshot.is_absolute() else args.root / args.snapshot
    print(json.dumps(proposed_selection(args.root, read_json(snapshot_path), args.source_commit),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
