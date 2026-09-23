"""The toy100 leaderboard must be earned by recorded live evidence."""

import json

import numpy as np

from benchmarks.toy100.gate import evaluate_suite, score_run
from experiments import leaderboard as legacy_leaderboard


STEPS = [0, 1, 10, 25, 50, 100, 250, 500, 750, 1000]


def good_metrics():
    return {"n": 20_000, "valid_n": 20_000, "modes": 100,
            "hq": .98, "precision": .98, "min_hq_count": 160,
            "min_hq_mode_mass": .008, "max_mode_mass": .013,
            "mass_tv": .03, "min_cov_eig_ratio": .7,
            "max_cov_eig_ratio": 1.2, "min_radial_median_ratio": .9,
            "max_radial_median_ratio": 1.1, "passed": True}


def write_run(output, problem="grid100", *, passing_steps=STEPS,
              event_steps=STEPS, snapshots=STEPS, budget=1000):
    folder = output / problem
    folder.mkdir(parents=True)
    config = {"problem": problem, "steps": budget, "eval_interval": 250,
              "early_eval_steps": [0, 1, 10, 25, 50, 100]}
    summary = {"problem": problem, "status": "complete", "budget_steps": budget,
               "completed_steps": budget,
               "config": config,
               "eval_steps": STEPS, "snapshot_steps": STEPS,
               "passed": True, "final_metrics": good_metrics()}
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "summary.json").write_text(json.dumps(summary))
    rows = []
    for step in event_steps:
        values = good_metrics()
        if step not in passing_steps:
            values.update(modes=72, min_hq_count=0, min_hq_mode_mass=0., passed=True)
        rows.append({"event": "eval", "step": step, "model": "live",
                     "elapsed": float(step), "metrics": values})
    (folder / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (folder / "snapshots").mkdir()
    for step in snapshots:
        samples = np.array([[float(step) / 1000, 0.], [0., .1]], dtype=np.float32)
        np.savez(folder / "snapshots" / f"step_{step:06d}.npz",
                 live=samples, ema=samples, target=samples)
    return folder


def test_summary_pass_stamp_and_initial_success_cannot_certify(tmp_path):
    write_run(tmp_path, passing_steps=[0])
    row = score_run(tmp_path / "grid100", "grid100")
    assert row["status"] == "FAIL"
    assert row["first_full_quality_step"] is None
    assert row["stable_checks"] == 0


def test_complete_terminal_suffix_is_required(tmp_path):
    write_run(tmp_path, passing_steps=[100, 250, 500, 750, 1000])
    row = score_run(tmp_path / "grid100", "grid100")
    assert row["status"] == "PASS"
    assert row["first_full_coverage_step"] == 100
    assert row["first_full_quality_step"] == 100
    assert row["stable_from_step"] == 100
    assert row["confirmed_step"] == 1000


def test_missing_checkpoint_and_missing_snapshot_are_invalid(tmp_path):
    write_run(tmp_path / "missing-event", event_steps=[step for step in STEPS if step != 500])
    write_run(tmp_path / "missing-snapshot", snapshots=[step for step in STEPS if step != 0])
    corrupt = write_run(tmp_path / "corrupt-snapshot")
    (corrupt / "snapshots" / "step_000000.npz").write_bytes(b"not an NPZ")
    assert score_run(tmp_path / "missing-event" / "grid100", "grid100")["status"] == "INVALID"
    assert score_run(tmp_path / "missing-snapshot" / "grid100", "grid100")["status"] == "INVALID"
    assert score_run(corrupt, "grid100")["status"] == "INVALID"


def test_final_collapse_fails_even_if_earlier_checks_pass(tmp_path):
    write_run(tmp_path, passing_steps=[0, 1, 10, 25, 50, 100, 250, 500, 750])
    row = score_run(tmp_path / "grid100", "grid100")
    assert row["status"] == "FAIL"
    assert row["first_full_quality_step"] == 1
    assert row["stable_checks"] == 0


def test_all_problem_gate_cannot_ignore_missing_problems(tmp_path):
    write_run(tmp_path, passing_steps=[100, 250, 500, 750, 1000])
    suite = evaluate_suite(tmp_path)
    assert suite["status"] == "FAIL"
    assert suite["passed_problems"] == 1
    assert suite["required_problems"] == 3
    assert suite["problems"]["rotated100"]["status"] == "MISSING"
    assert "rotated100 | MISSING" in (tmp_path / "leaderboard.md").read_text()


def test_tiny_budget_cannot_pass_even_with_forged_receipt(tmp_path):
    write_run(tmp_path, budget=1)
    row = score_run(tmp_path / "grid100", "grid100")
    assert row["status"] == "INVALID"
    assert "steps must be an integer" in row["reason"]


def test_legacy_leaderboard_dispatches_toy100_without_erasing_missing_rows(tmp_path):
    write_run(tmp_path, passing_steps=[100, 250, 500, 750, 1000])
    assert legacy_leaderboard.main(["--toy100-output", str(tmp_path)]) == 1
    assert "rotated100 | MISSING" in (tmp_path / "leaderboard.md").read_text()
    for name in ("rotated100", "staggered100"):
        write_run(tmp_path, problem=name, passing_steps=[100, 250, 500, 750, 1000])
    assert legacy_leaderboard.main(["--toy100-output", str(tmp_path)]) == 0
