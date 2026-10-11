"""Read-only regression checks for compact prior-search publication helpers.

Replays software fixtures and reads immutable completed result/media evidence.
Never constructs a model, samples a prior, trains, grades anew, or admits work.
Run with the public repository Python; raw observations stay in the archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import workflow as report


def validate(pool: Path):
    checks, evidence = {}, []

    def check(name, condition):
        report.require(condition, name)
        checks[name] = "PASS"

    fixture_attempts = {
        ("proximal", "trajectory"): "718d4f9f87b744b3b3a2818f7765599b",
        ("proximal", "gaussian1d_stability"): "32b9be01e213404d93c32e5fc1304d7e",
        ("coulomb", "grid100"): "04450a6861f34600a21252ef317222ca",
        ("coulomb", "five_word_joint_hold"): "2857789ca7094abb88dd299a324803c0",
    }

    def saved(name, task, status=None):
        state = report.read_json(pool / name / "queue/queue/state.json")
        for job in state["jobs"].values():
            result = job.get("result") or {}
            if result.get("attempt_id") != fixture_attempts[(name, task)]:
                continue
            for row in result.get("task_results", []):
                if row["task_id"] == task and (status is None or row.get("gate_status") == status):
                    request = next(entry["request"] for entry in state["submissions"].values()
                                   if any(definition["compatibility_key"] == row["compatibility_key"]
                                          for definition in entry["request"]["jobs"]))
                    evidence.append(dict(archive=name, task=task, attempt_id=result["attempt_id"],
                                         result_sha256=report.stable_hash(result),
                                         row_sha256=report.stable_hash(row),
                                         source_digest=request["source"]["digest"],
                                         task_sha256=report.stable_hash(request["tasks"][task])))
                    return row, request["tasks"][task]
        raise ValueError(f"Required completed saved fixture missing: {name}/{task}/{status}")

    rows = report.metric_rows({"diagnostic": {"array": [1, None]}}, [("missing", ">=", 5)])
    check("missing_scores_keep_targets_and_none_diagnostic_leaves", rows[0]["score"] is None
          and rows[0]["target"] == ">= 5" and rows[0]["status"] == "UNAVAILABLE"
          and rows[-1]["metric"] == "diagnostic.array[1]" and rows[-1]["status"] == "UNAVAILABLE")
    cell = dict(gate_status="UNKNOWN", qualification_tier=2)
    trial = dict(submission_status="blocked", submission_reason="required Tier1 failed",
                 tasks=[dict(task="word", qualification_tier=1, importance="required",
                             gate_status="FAIL", attempt_id="saved-attempt")])
    check("blocked_higher_tier_is_display_only", report.display_outcome({}, cell, trial)
          == ("BLOCKED", "required Tier1 failed") and cell["gate_status"] == "UNKNOWN")
    check("unexecuted_current_tier_peer_remains_unmeasured", report.display_outcome(
        {}, {**cell, "qualification_tier": 1}, trial) == ("UNMEASURED", None))
    check("explicit_blocked_reason_and_measured_status_preserved", report.display_outcome(
        {}, dict(gate_status="BLOCKED", reason="original prerequisite"), trial)
        == ("BLOCKED", "original prerequisite") and report.display_outcome(
        dict(gate_status="ERROR", reason="original error"), cell, trial) == ("ERROR", "original error"))

    row, card = saved("proximal", "trajectory", "PASS")
    before = report.stable_hash(row)
    compact = report.compact_row(row)
    expected = {key: value for key, value in row["raw"].items()
                if key not in row["metrics"] and (value is None or isinstance(value, (int, float, bool)))}
    check("scalar_raw_endpoint_diagnostics_no_duplicate_gate_or_trace", len(expected) == 7
          and compact["raw_endpoint_diagnostics"] == expected
          and all(section.get("qualification_credit") is False for section in report.sections(compact, card)
                  if section["name"].startswith("Additional scalar")))
    check("actual_applied_recipe_initializer_and_evidence_prior_policy", compact["recipe"] == row["applied"]["recipe"]
          and compact["initializer"] == row["applied"]["initializer"]
          and compact["initialization"] == row["applied"]["initialization"]
          and compact["prior_policy"] == row["evidence"]["prior_policy"])
    check("trajectory_projection_preserves_original_row_and_grade", before == report.stable_hash(row)
          and compact["gate_status"] == "PASS")

    row, card = saved("proximal", "gaussian1d_stability", "FAIL")
    compact = report.compact_row(row)
    sections = report.sections(compact, card)
    diagnostics = next(section for section in sections if section["name"].startswith("Recorded evaluator"))
    check("gaussian_frozen_check_counts_retained_as_diagnostics", any(
        metric["metric"] == "frozen_passes" and metric["score"] == 0 for metric in diagnostics["rows"])
        and any(metric["metric"] == "frozen_checks" and metric["score"] == 48 for metric in diagnostics["rows"])
        and diagnostics["qualification_credit"] is False)
    compact["evaluator_result"]["missing_diagnostic"] = None
    check("none_evaluator_diagnostic_retained", any(metric["metric"] == "missing_diagnostic"
          and metric["status"] == "UNAVAILABLE" for section in report.sections(compact, card)
          for metric in section["rows"]))

    row, card = saved("coulomb", "grid100", "FAIL")
    before = report.stable_hash(row)
    compact = report.compact_row(row)
    sections = report.sections(compact, card)
    evaluator_section = next(section for section in sections if section["name"].startswith("Recorded evaluator"))
    check("native_terminal_packet_compacted_and_gate_cohorts_not_duplicated", "terminal_checks" not in compact["evaluator_result"]
          and compact["evaluator_result"]["terminal_summary"]["checks"] == 5
          and not any(metric["metric"].startswith(("final_metrics.", "holdout_metrics.", "holdout_ema_metrics.", "oracle_metrics."))
                      for metric in evaluator_section["rows"]))
    cohorts = [section for section in sections if section.get("cohort") in {"holdout_ema_metrics", "oracle_metrics"}]
    check("native_ema_oracle_separate_diagnostics_keep_original_accuracy_targets", len(cohorts) == 2
          and all(section["qualification_credit"] is False and any(
              metric["metric"] == "center_rms_sigma" and metric["limits"] == [("<=", card["evaluation"]["accuracy_limits"]["center_rms_sigma"])]
              for metric in section["rows"]) for section in cohorts)
          and before == report.stable_hash(row))

    row, card = saved("coulomb", "five_word_joint_hold", "FAIL")
    before = report.stable_hash(row)
    witnesses = report.word_hold_failure_witnesses(row, card)
    check("word_hold_primary_and_confirmation_failure_witnesses", len(witnesses) == 2
          and all(section["observation_count"] == 25 and section["numerical_failed_checks"] == 1
                  and section["qualification_credit"] is False for section in witnesses))
    for section in witnesses:
        points = row["evidence"]["observations" if section["cohort"] == "primary" else "confirmations"]
        check("word_hold_" + section["cohort"].replace(" ", "_") + "_source_hashes_and_worst_scores",
              section["original_observations_sha256"] == report.stable_hash(points)
              and all(any(metric["witness_step"] == point["step"]
                          and metric["witness_observation_sha256"] == report.stable_hash(point)
                          and metric["score"] == point.get("metrics", point).get(metric["metric"])
                          for point in points) for metric in section["rows"])
              and all(metric["witness_step"] == 5001 for metric in section["rows"] if metric["status"] == "FAIL"))
    check("word_hold_witnesses_do_not_mutate_or_extend_acquisition_smoke", report.stable_hash(row) == before
          and report.word_hold_failure_witnesses(row, {**card, "id": "five_word_joint_smoke"}) == [])

    media_validation = report.read_json(report.ROOT / "reports/forge/bcap-prior-search/saved-media-validation.json")
    exported = 0
    for cohort in media_validation["cohorts"]:
        index_path = Path(cohort["archive_index"])
        check("media_index_" + index_path.parent.parent.name + "_" + index_path.parent.name,
              report.file_hash(index_path) == cohort["archive_index_sha256"])
        for media in cohort["media"]:
            directory = index_path.parent
            receipt = report.read_json(directory / media["receipt"])
            check("media_" + media["attempt_id"] + "_" + index_path.parent.name,
                  report.file_hash(directory / media["gif"]) == media["gif_sha256"]
                  and report.file_hash(directory / media["receipt"]) == media["receipt_sha256"]
                  and receipt["frames"] >= 2 and receipt["sampling_draws_added"] == 0
                  and receipt["optimizer_updates_added"] == 0 and receipt["rescored_observations"] == 0
                  and receipt["recorded_grade"] == receipt["gate_status"])
            exported += 1
    directory = pool / "proximal/software-context-media-check/control"
    trajectory = report.read_json(directory / "trajectory.json")
    clock = report.read_json(directory / "clockfree_audit_measurement_v1.json")
    row, _ = saved("proximal", "trajectory", "PASS")
    check("exported_media_uses_actual_applied_context", trajectory["recipe_sha256"] == report.stable_hash(row["applied"]["recipe"])
          and trajectory["initializer_sha256"] == report.stable_hash(row["applied"]["initializer"])
          and trajectory["prior_policy_sha256"] == report.stable_hash(row["evidence"]["prior_policy"]))
    check("clock_probe_explicitly_smaller_and_no_feature_invariance_claim", clock["media_cohort"] == "clock_probe_diagnostic"
          and clock["feature_invariance_claim"] is False and clock["actual_recipe_settings"]["prior_proximal_strength"] is None)
    return dict(schema_version=1, scope="software_only_saved_report_regression_checks",
                qualification_input=False, training_runs_added=0, optimizer_updates_added=0,
                sampling_draws_added=0, rescored_observations=0, checks=checks,
                status="PASS", checked_saved_media=exported, evidence=evidence,
                source_sha256=report.file_hash(Path(__file__)),
                workflow_sha256=report.file_hash(Path(report.__file__)),
                exporter_sha256=report.file_hash(Path(__file__).with_name("export_saved_media.py")),
                saved_media_validation_sha256=report.file_hash(report.ROOT / "reports/forge/bcap-prior-search/saved-media-validation.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, default=Path("/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("report-helper-validation.json"))
    args = parser.parse_args()
    receipt = validate(args.pool_root)
    report.atomic_json(args.output, receipt)
    print(f"{receipt['status']}: {len(receipt['checks'])} report checks; {receipt['checked_saved_media']} saved GIFs; zero training/draws")
