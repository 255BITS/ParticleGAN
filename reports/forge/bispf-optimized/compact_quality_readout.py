"""Reproduce the compact ordinary readout from its byte-exact local original."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json
from reports.forge.regenerate_technique_inventory import project_receipt

READOUT = "readout-10ed0ff3f18514315a760f4a.json"
CURATION = ("Final metrics, numerical verdicts, decision outcomes and exact original receipt hashes "
            "retained through project_receipt; per-update evidence is archived byte-exactly locally. "
            "Qualification uses the original independently certified attempt envelopes, not this display summary.")


def project(root, archive):
    original = read_json(archive)
    record = deepcopy(original)
    originals = {(r["_attempt_id"], r["task_id"]): r for r in original["task_results"]}
    rows = []
    for attempt in original["attempt_ids"]:
        summary = project_receipt(root, attempt)
        for row in summary["task_results"]:
            actual = originals[attempt, row["task_id"]]
            if row["gate_status"] != actual["gate_status"]:
                raise ValueError("Readout differs from independently certified numerical verdict")
            rows.append({**row, "attempt_id": attempt, "metrics": actual["metrics"]})
    if len(rows) != 30:
        raise ValueError("Readout must preserve all 30 ordinary task cells")
    record["task_results"] = sorted(rows, key=lambda r: r["task_id"])
    record["original_source"] = record["source"]
    record["source"] = {"path": "reports/forge/bispf-optimized/README.md"}
    record.update(qualification_input=False, full_readout_sha256=file_hash(archive),
                  full_readout_archive=dict(path=str(archive), sha256=file_hash(archive), bytes=archive.stat().st_size),
                  summary_curation=CURATION)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, default=Path(
        "/home/martyn/dev/ParticleGAN/runs/forge/bispf-optimized/ordinary-quality-full-readout.json"))
    parser.add_argument("--write", action="store_true", help="Replace an uncurated original with its display summary")
    args = parser.parse_args()
    record = project(args.root, args.archive)
    destination = args.root / "reports/forge/records" / READOUT
    if args.write:
        current = read_json(destination)
        if current != read_json(args.archive) and current != record:
            raise ValueError("Refuse to replace an unrelated or altered concluded readout")
        atomic_json(destination, record)
    elif read_json(destination) != record:
        raise ValueError("Committed compact readout differs from exact original receipt projection")
    print(f"verified {destination.name}: {file_hash(destination)}")
