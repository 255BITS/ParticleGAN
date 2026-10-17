"""The toy100 leaderboard must be earned by recorded live evidence."""

import hashlib
import json

import numpy as np
import torch

from benchmarks.toy100 import metrics, problems
from benchmarks.toy100.config import resolve_problem_config
from benchmarks.toy100.gate import evaluate_suite, score_run
from benchmarks.toy100.train import resolve_config
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


def write_manifested_suite(output):
    """A complete small receipt whose terminal rows can be audited from draws."""
    declared = {"name": "toy100_receipt_test", "steps": 1000, "seed": 1234,
                "snapshot_samples": 2,
                "problem_overrides": {"staggered100": {"batch_size": 1024}}}
    source = output / "recipe.json"
    output.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(declared) + "\n"
    source.write_text(contents)
    flats = {name: resolve_problem_config(declared, name) for name in problems.PROBLEM_NAMES}
    receipt = {"config_path": str(source),
               "config_sha256": hashlib.sha256(contents.encode()).hexdigest(),
               "config_contents": contents, "declared_manifest": declared,
               "resolved_problem_configs": flats,
               "selected_problems": list(problems.PROBLEM_NAMES),
               "command_overrides": {"steps": None, "device": None}}
    (output / "run_manifest.json").write_text(json.dumps(receipt))
    for index, name in enumerate(problems.PROBLEM_NAMES):
        folder = write_run(output, problem=name)
        executed, _ = resolve_config(flats[name])
        executed = json.loads(json.dumps(executed))
        (folder / "config.json").write_text(json.dumps(executed))
        draw = problems.sample_real(name, 20_000,
                                    generator=torch.Generator().manual_seed(1234 + index))
        observed = metrics.evaluate_samples(draw, name)
        assert metrics.passes(name, observed)
        np.savez_compressed(folder / "final_samples.npz", live=draw.numpy(),
                            ema=draw.numpy(), target=draw.numpy())
        summary = json.loads((folder / "summary.json").read_text())
        summary.update(config=executed, final_samples_file="final_samples.npz",
                       final={"live": observed})
        (folder / "summary.json").write_text(json.dumps(summary))
        rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        for row in rows:
            row["metrics"] = observed
        (folder / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return receipt


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


def test_manifest_binds_each_executed_config_and_all_three_selected_runs(tmp_path):
    receipt = write_manifested_suite(tmp_path)
    good = evaluate_suite(tmp_path, write=False)
    assert good["status"] == "PASS"
    assert all(row["audited_final_samples"] for row in good["problems"].values())
    assert evaluate_suite(tmp_path, problem="grid100", write=False)["status"] == "PASS"

    folder = tmp_path / "staggered100"
    config = json.loads((folder / "config.json").read_text())
    config["batch_size"] = 512
    (folder / "config.json").write_text(json.dumps(config))
    summary = json.loads((folder / "summary.json").read_text())
    summary["config"] = config
    (folder / "summary.json").write_text(json.dumps(summary))
    assert score_run(folder, "staggered100")["status"] == "PASS"
    bound = evaluate_suite(tmp_path, write=False)
    assert bound["status"] == "FAIL"
    assert bound["problems"]["staggered100"]["status"] == "INVALID"
    assert "executed config differs" in bound["problems"]["staggered100"]["reason"]

    receipt["selected_problems"] = ["grid100", "staggered100"]
    (tmp_path / "run_manifest.json").write_text(json.dumps(receipt))
    scoped = evaluate_suite(tmp_path, write=False)
    assert scoped["status"] == "FAIL"
    assert all(row["status"] == "INVALID" for row in scoped["problems"].values())


def test_manifest_source_hash_and_resolved_declaration_are_verified(tmp_path):
    receipt = write_manifested_suite(tmp_path)
    receipt["resolved_problem_configs"]["grid100"]["fourier"] = 99
    (tmp_path / "run_manifest.json").write_text(json.dumps(receipt))
    result = evaluate_suite(tmp_path, write=False)
    assert result["status"] == "FAIL"
    assert "resolved config differs" in result["problems"]["grid100"]["reason"]

    receipt["resolved_problem_configs"]["grid100"].pop("fourier")
    receipt["config_contents"] += " "
    (tmp_path / "run_manifest.json").write_text(json.dumps(receipt))
    result = evaluate_suite(tmp_path, write=False)
    assert "SHA256 disagrees" in result["problems"]["grid100"]["reason"]


def test_final_event_cannot_claim_pass_against_saved_collapsed_draw(tmp_path):
    write_manifested_suite(tmp_path)
    folder = tmp_path / "grid100"
    collapsed = np.zeros((20_000, 2), dtype=np.float32)
    np.savez_compressed(folder / "final_samples.npz", live=collapsed,
                        ema=collapsed, target=collapsed)
    row = score_run(folder, "grid100")
    assert row["status"] == "INVALID"
    assert "disagree with saved final evaluation samples" in row["reason"]
    assert evaluate_suite(tmp_path, write=False)["status"] == "FAIL"


def test_manifested_run_cannot_omit_final_draw_receipt(tmp_path):
    write_manifested_suite(tmp_path)
    folder = tmp_path / "grid100"
    summary = json.loads((folder / "summary.json").read_text())
    summary.pop("final_samples_file")
    (folder / "summary.json").write_text(json.dumps(summary))
    result = evaluate_suite(tmp_path, write=False)
    assert result["problems"]["grid100"]["status"] == "INVALID"
    assert "lacks saved final evaluation draw" in result["problems"]["grid100"]["reason"]
