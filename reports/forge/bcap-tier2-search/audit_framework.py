"""Measure execution reliability and scheduling intervals after both drains."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json

OUTPUT = ROOT / "reports/forge/bcap-tier2-search"


def main():
    combined = read_json(OUTPUT / "combined-readout.json")
    assert combined["selection"]["selection_complete"] and combined["configurations"] == 96
    attempts, campaigns = [], []
    for report_path in (OUTPUT / "readout.json", OUTPUT / "overnight/readout.json"):
        report = read_json(report_path)
        queue = Path(report["queue_root"])
        local = ROOT / "runs/software" / ("bcap-overnight-search" if "overnight" in report["study_id"] else "bcap-tier2-search")
        assert (local / "completed.json").is_file()
        assert report["campaign"]["reserved_seconds"] == 0
        ids = sorted({attempt for trial in report["trials"] for attempt in trial["attempt_ids"]})
        assert len(ids) == report["attempt_count"]
        for attempt in ids:
            envelope = read_json(ROOT / "reports/forge/attempts" / attempt / "request.json")
            directory = Path(envelope["worker"]["directory"])
            terminal_path = directory / "terminal.json"
            terminal = read_json(terminal_path)
            interval = terminal.get("telemetry", {}).get("interval")
            attempts.append({"attempt_id": attempt, "study_id": report["study_id"],
                             "task_id": envelope["job"]["task_id"],
                             "device": envelope["worker"]["device"],
                             "attempt_status": terminal["attempt_status"],
                             "elapsed_seconds": terminal["elapsed_seconds"],
                             "interval": interval, "terminal_sha256": file_hash(terminal_path)})
        campaigns.append({"study_id": report["study_id"], "configurations": len(report["trials"]),
                          "attempts": len(ids), "paid_worker_seconds": report["campaign"]["spent_seconds"],
                          "queue_state_bytes": (queue / "queue/state.json").stat().st_size,
                          "queue_state_sha256": file_hash(queue / "queue/state.json"),
                          "submission_status": dict(Counter(t["submission_status"] for t in report["trials"])),
                          "scientific_retries": report["scientific_retries"]})
    assert len(attempts) == len({a["attempt_id"] for a in attempts}) == combined["attempt_count"]
    intervals = [a for a in attempts if a["interval"]]
    missing = [a["attempt_id"] for a in attempts if not a["interval"]]
    if intervals:
        start = min(a["interval"]["started_at"] for a in intervals)
        finish = max(a["interval"]["finished_at"] for a in intervals)
        window = finish - start
        devices = {}
        for device in sorted({a["device"] for a in intervals}):
            rows = sorted((a["interval"] for a in intervals if a["device"] == device),
                          key=lambda row: row["started_at"])
            gaps = []
            for previous, current in zip(rows, rows[1:]):
                gap = current["started_at"] - previous["finished_at"]
                assert gap >= -1e-3, "A declared single-device slot has overlapping supervisors"
                gaps.append(max(0., gap))
            occupied = sum(row["finished_at"] - row["started_at"] for row in rows)
            devices[device] = {"attempts": len(rows), "supervised_seconds": occupied,
                               "fraction_of_full_window": occupied / window,
                               "seconds_outside_supervision": window - occupied,
                               "between_attempt_gaps_seconds": {
                                   "count": len(gaps), "total": sum(gaps),
                                   "median": statistics.median(gaps) if gaps else None,
                                   "maximum": max(gaps) if gaps else None}}
        timing = {"started_at": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                  "finished_at": datetime.fromtimestamp(finish, timezone.utc).isoformat(),
                  "wall_seconds": window, "devices": devices}
    else:
        timing = None
    statuses = dict(Counter(a["attempt_status"] for a in attempts))
    atomic_json(OUTPUT / "framework-readout.json", {
        "schema_version": 1, "configurations": 96, "campaigns": campaigns,
        "attempt_count": len(attempts), "execution_status": statuses,
        "scientific_retries": combined["scientific_retries"], "timing": timing,
        "missing_supervisor_intervals": missing,
        "measurement_scope": "Supervisor intervals include process startup and independent grading. Gaps include admission, source verification, Python startup, queue bookkeeping, campaign transition and dependency scheduling; they do not isolate coordinator cost. These fractions are not CUDA kernel utilization.",
        "status_note": "Scientific gate failures can leave a completed submission labelled blocked. Execution status is measured separately from numerical PASS/FAIL.",
        "terminal_receipt_digest_inputs": [{"attempt_id": a["attempt_id"], "sha256": a["terminal_sha256"]}
                                            for a in attempts],
        "optimizer_updates_added": 0, "sampling_draws_added": 0, "qualification_input": False})
    print({"stage": "framework_audit", "attempts": len(attempts), "execution_status": statuses,
           "wall_seconds": timing["wall_seconds"] if timing else None}, flush=True)


if __name__ == "__main__":
    main()
