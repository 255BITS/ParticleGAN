#!/usr/bin/env python3
"""Independently audit and archive the 22 completed resumed research screens."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    a = p.parse_args()
    scope = json.loads((HERE / "retest-closure/coverage-scope.json").read_text())
    queue = {r["id"]: r for r in json.loads((HERE / "retest-queue.json").read_text())["rows"]}
    selected = [r for r in scope["reviewed_research_cases"]
                if queue[r["queue_row"]]["execution_status"] == "NOT_RUN_USER_STOP"]
    assert len(selected) == 22
    assert all((a.runs / row["case_id"] / "result.json").is_file() for row in selected)
    for number, row in enumerate(selected, 1):
        case = row["case_id"]
        identifier = "research-" + case + "-new-init"
        audit = HERE / "research-mode-hold-review/completed" / (case + "-runtime-audit.json")
        archive = HERE / "research-evidence" / identifier / "archive-manifest.json"
        if archive.exists():
            print(f"{number}/22 {case}: already archived", flush=True)
            continue
        subprocess.run([sys.executable, str(HERE / "research-mode-hold-review/runtime_audit.py"),
                        "--source", str(a.runs / case), "--queue-row", row["queue_row"],
                        "--output", str(audit)], check=True)
        reviewed = json.loads(audit.read_text())
        assert reviewed["status"] == "PASS"
        subprocess.run([sys.executable, str(HERE / "archive_research_screen.py"),
                        "--audit", str(audit), "--id", identifier,
                        "--eligibility", reviewed["historical_eligibility"]], check=True)
        print(f"{number}/22 {case}: archived", flush=True)


if __name__ == "__main__":
    main()
