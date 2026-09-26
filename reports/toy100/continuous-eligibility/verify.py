"""Audit retained decisions and source receipts without importing training code."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", action="store_true", help="also verify original local candidate sources")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    repo = folder.parents[2]
    audit = read(folder / "audit.json")
    state = read(folder.parent / "continuous-search-state.json")
    receipts = read(folder / "source-receipts.json")
    assert audit["qualified_winner"] is state["selected_candidate"] is state["current_lead"] is None
    assert state["candidate_eligibility"] == audit["entries"]
    assert audit["deadline_all_81_required"] is False
    assert audit["finite_evaluation_duration_is_disqualification"] is False
    assert audit["initialization_or_smoothing_window_is_disqualification"] is False
    counts = {}
    for entry in audit["entries"].values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        for link in entry["evidence"]:
            assert (folder / link).is_file(), link
    assert counts == {"DISQUALIFIED_CURRENT_CONFIGURATION": 18,
                      "REJECTED_MEASURED_QUALITY": 3, "UNVERIFIED_INCOMPLETE": 1}
    for name, receipt in receipts["common_sources"].items():
        path = repo / name
        assert sha(path) == receipt["sha256"], name
        lines = path.read_text().splitlines()
        for excerpt in receipt["excerpts"]:
            start = excerpt["first_line"] - 1
            expected = excerpt["text"].splitlines()
            assert lines[start:start + len(expected)] == expected, name
    common = "reports/toy100/gap-fill-20260925/sources/k3p/"
    for name, receipt in receipts["candidates"].items():
        for filename in ("config.json", "shift.py"):
            assert receipt["sha256"][filename] == receipts["common_sources"][common + filename]["sha256"]
        if args.originals:
            for filename, expected in receipt["sha256"].items():
                assert sha(Path(receipt["source_directory"]) / filename) == expected, (name, filename)
    print("PASS: 22 eligibility decisions, five shared-source hashes/excerpts, six candidate receipts; no training.")


if __name__ == "__main__":
    main()
