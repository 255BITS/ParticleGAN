"""Publication checks only; full source runs carry their own exact prefix controls."""
import json
from pathlib import Path

from benchmarks.paired_error_2d import run as source_run, task as source_task
from benchmarks.toy_audit.source_family_report import forbid_inline_streams


def test_original_stability_gate_requires_the_complete_budget_and_three_checks():
    passing = dict(nmse=.001, p95_distance=.1)
    row = dict(complete=False, history=[dict(validation=passing, live_validation=passing)] * 3)
    assert not source_run.stable(row, "validation")
    row["complete"] = True
    assert source_run.stable(row, "validation")
    row["history"][-1] = dict(validation=dict(nmse=.001, p95_distance=.201))
    assert not source_run.stable(row, "validation")


def test_paired_publication_is_two_registered_default_jobs_with_real_frames():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/toy_audit/paired_sources/coverage.json").read_text())
    records = report["records"]
    assert len(records) == report["target_laws"] == report["scientific_jobs"] == 2
    assert {r["catalog_id"] for r in records} == {"source-family-08", "source-family-09"}
    assert report["attempts_per_family"] == 1
    assert report["measured_job_seconds"] == sum(r["seconds"] for r in records)
    for row in records:
        assert row["arm"] == "baseline" and row["cloud"] == "movable"
        assert row["fresh_execution_status"] == "COMPLETE"
        assert row["original_scientific_status"] == row["added_gate_status"] == "PASS"
        assert row["expected_updates"] == row["last_saved_step"] == source_task.PROTOCOL["steps"] == 6000
        assert row["observer_prefix_parity"]["exact"] and row["observer_state_purity"]["exact"]
        assert row["actual_step_indices"] == list(range(0, 6001, 500))
        assert row["media_receipt"]["actual_step_indices"] == row["actual_step_indices"]
        assert row["observation_count"] == row["media_receipt"]["media"]["frames"] == 13
        assert not row["media_receipt"]["interpolation"]
        assert row["media"].startswith("media/")
        assert (root / "reports/toy_audit/paired_sources" / row["media"]).is_file()
        assert row["curve_archive"]["path"].endswith("captured-metrics.json")
        assert row["test_evaluation_status"].startswith("NOT_RUN")
    forbid_inline_streams(report)
