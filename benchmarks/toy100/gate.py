"""Fail-closed, live-weight gate for the three 100-Gaussian toys.

The runner's summary is a receipt, not a verdict. This module reconstructs the
verdict from the complete sequence of independently recorded evaluation events.
Step zero is reported but cannot contribute to convergence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import metrics, problems
from .train import evaluation_steps


MIN_BUDGET_STEPS = 1000
MAX_EVAL_INTERVAL = 250
MIN_STABLE_CHECKS = 5
MANDATORY_EARLY_STEPS = (0, 1, 10, 25, 50, 100)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _finite_tree(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_tree(item) for key, item in value.items())
    return False


def _failure(problem: str, status: str, reason: str, **details):
    return {"problem": problem, "status": status, "reason": reason,
            "passed": False, "stable_checks": 0, **details}


def _expected_steps(config: dict):
    budget = config.get("steps")
    if type(budget) is not int or budget < MIN_BUDGET_STEPS:
        raise ValueError(f"steps must be an integer >= {MIN_BUDGET_STEPS}")
    interval = config.get("eval_interval", 250)
    if type(interval) is not int or not 1 <= interval <= MAX_EVAL_INTERVAL:
        raise ValueError(f"eval_interval must be an integer in [1, {MAX_EVAL_INTERVAL}]")
    early = config.get("early_eval_steps", list(MANDATORY_EARLY_STEPS))
    if not isinstance(early, (list, tuple)) or any(type(step) is not int for step in early):
        raise ValueError("early_eval_steps must be integer checkpoints")
    if not set(MANDATORY_EARLY_STEPS).issubset(early):
        raise ValueError("mandatory initialization and early checkpoints are missing")
    if any(step < 0 or step > budget for step in early):
        raise ValueError("early_eval_steps must fit the training budget")
    expected = evaluation_steps(budget, eval_interval=interval, early_eval_steps=tuple(early))
    if not expected or expected[0] != 0 or expected[-1] != budget:
        raise ValueError("evaluation schedule must include initialization and the full budget")
    if len([step for step in expected if step > 0]) < MIN_STABLE_CHECKS:
        raise ValueError("evaluation schedule has too few postinitial observations")
    return budget, list(expected)


def _events(path: Path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank event line {number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid event line {number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"event line {number} must be an object")
            if row.get("event") == "train":
                continue
            if row.get("model") == "live":
                rows.append(row)
    return rows


def _check_snapshot(path: Path, expected_n: int | None):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"live", "ema", "target"}.issubset(archive.files):
                raise ValueError(f"{path}: missing live, EMA, or target samples")
            for key in ("live", "ema", "target"):
                points = archive[key]
                if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
                        or (expected_n is not None and len(points) != expected_n)
                        or not np.issubdtype(points.dtype, np.number)
                        or not np.isfinite(points).all()):
                    raise ValueError(f"{path}: invalid {key} samples")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid snapshot: {exc}") from exc


def score_run(run_dir: Path | str, problem: str) -> dict:
    """Recompute one verdict; malformed or absent evidence never earns a pass."""
    run_dir = Path(run_dir)
    if problem not in problems.PROBLEM_NAMES:
        raise ValueError(f"unknown problem: {problem}")
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    event_path = run_dir / "events.jsonl"
    missing = [p.name for p in (summary_path, config_path, event_path) if not p.is_file()]
    if missing:
        return _failure(problem, "MISSING", "missing evidence: " + ", ".join(missing))

    try:
        summary, config = _read_json(summary_path), _read_json(config_path)
        if not isinstance(summary, dict) or not isinstance(config, dict):
            raise ValueError("summary and config must be objects")
        if summary.get("error") or summary.get("status") == "error":
            return _failure(problem, "ERROR", str(summary.get("error") or "training error"))
        if summary.get("status") != "complete":
            raise ValueError("training summary does not report a completed run")
        if summary.get("problem") != problem or config.get("problem") != problem:
            raise ValueError("problem identity differs between directory, summary, and config")
        if summary.get("config") != config:
            raise ValueError("summary config differs from executed config")
        budget, expected = _expected_steps(config)
        if summary.get("budget_steps") != budget:
            raise ValueError("summary budget differs from executed config")
        if summary.get("completed_steps") != budget:
            raise ValueError("run did not complete the full training budget")
        if summary.get("eval_steps") != expected:
            raise ValueError("summary evaluation schedule differs from protocol")
        snapshots = summary.get("snapshot_steps")
        if (not isinstance(snapshots, list)
                or any(type(step) is not int for step in snapshots)
                or snapshots != sorted(set(snapshots))
                or any(step < 0 or step > budget for step in snapshots)
                or not set(expected).issubset(snapshots)):
            raise ValueError("summary snapshot schedule omits scored checkpoints")
        missing_snapshots = [step for step in snapshots
                             if not (run_dir / "snapshots" / f"step_{step:06d}.npz").is_file()]
        if missing_snapshots:
            raise ValueError(f"missing saved sample snapshots at {missing_snapshots}")
        expected_snapshot_n = config.get("snapshot_samples")
        if expected_snapshot_n is not None and (type(expected_snapshot_n) is not int or expected_snapshot_n < 1):
            raise ValueError("snapshot_samples must be a positive integer")
        for step in snapshots:
            _check_snapshot(run_dir / "snapshots" / f"step_{step:06d}.npz", expected_snapshot_n)
        rows = _events(event_path)
        actual = [row.get("step") for row in rows]
        if actual != expected:
            raise ValueError(f"live evaluation checkpoints do not match protocol: {actual}")
        last_elapsed = -1.0
        passing = []
        for row in rows:
            elapsed, values = row.get("elapsed"), row.get("metrics")
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
                raise ValueError(f"missing or invalid elapsed time at step {row['step']}")
            if elapsed < last_elapsed or elapsed < 0:
                raise ValueError("evaluation times must be nonnegative and monotonic")
            last_elapsed = elapsed
            if not isinstance(values, dict) or not _finite_tree(values):
                raise ValueError(f"missing or nonfinite metrics at step {row['step']}")
            expected_eval_n = config.get("eval_samples")
            if expected_eval_n is not None and values.get("n") != expected_eval_n:
                raise ValueError(f"evaluation draw count differs from config at step {row['step']}")
            passing.append(bool(metrics.passes(problem, values)))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return _failure(problem, "INVALID", str(exc))

    # Keep initialization visible, but never certify convergence at step zero.
    postinitial = list(zip(rows[1:], passing[1:]))
    first = next((row for row, ok in postinitial if ok), None)
    full_coverage = next((row for row, _ in postinitial
                          if row["metrics"].get("modes") == 100), None)
    suffix = []
    for row, ok in reversed(postinitial):
        if not ok:
            break
        suffix.append(row)
    suffix.reverse()
    stable = len(suffix) >= MIN_STABLE_CHECKS
    final = rows[-1]
    result = {
        "problem": problem,
        "status": "PASS" if stable else "FAIL",
        "reason": "terminal live metrics sustained" if stable else
                  f"only {len(suffix)}/{MIN_STABLE_CHECKS} terminal postinitial checks pass",
        "passed": stable,
        "budget_steps": budget,
        "observations": len(rows),
        "stable_checks": len(suffix),
        "required_stable_checks": MIN_STABLE_CHECKS,
        "first_full_coverage_step": full_coverage["step"] if full_coverage else None,
        "first_full_coverage_seconds": full_coverage["elapsed"] if full_coverage else None,
        "first_full_quality_step": first["step"] if first else None,
        "first_full_quality_seconds": first["elapsed"] if first else None,
        "stable_from_step": suffix[0]["step"] if stable else None,
        "stable_from_seconds": suffix[0]["elapsed"] if stable else None,
        "confirmed_step": suffix[MIN_STABLE_CHECKS - 1]["step"] if stable else None,
        "confirmed_seconds": suffix[MIN_STABLE_CHECKS - 1]["elapsed"] if stable else None,
        "final_step": final["step"],
        "final_seconds": final["elapsed"],
        "final_elapsed": final["elapsed"],
        "initial_metrics": rows[0]["metrics"],
        "final_metrics": final["metrics"],
        "run_dir": str(run_dir),
    }
    return result


def _cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _step_time(result: dict, prefix: str) -> str:
    step, seconds = result.get(prefix + "_step"), result.get(prefix + "_seconds")
    return "—" if step is None else f"{step:,} ({seconds:.1f}s)" if seconds is not None else f"{step:,}"


def _range(values: dict, minimum: str, maximum: str) -> str:
    low, high = values.get(minimum), values.get(maximum)
    return "—" if low is None or high is None else f"{low:.3f}–{high:.3f}"


def leaderboard_markdown(gate: dict) -> str:
    """A compact report that preserves failures, time, and the evaluated budget."""
    lines = [f"# 100-Gaussian toy gate: {gate['status']}", "",
             f"Scope: **{gate['scope']}**. Verdicts use complete live-weight curves at the stated training budget. "
             "A PASS needs five consecutive passing postinitial checks at the end; all 100 modes, "
             "sample quality, mode mass, and per-mode shape are evaluated by the toy metrics.", "",
             "| Problem | Status | Final modes | Final HQ | Mass TV | Cov eig ratio | Radial ratio | First 100 modes | First full quality | Stable from | Confirmed | Budget / elapsed | Reason |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, result in gate["problems"].items():
        final = result.get("final_metrics") or {}
        values = (name, result["status"], _cell(final.get("modes")), _cell(final.get("hq")),
                  _cell(final.get("mass_tv")),
                  _range(final, "min_cov_eig_ratio", "max_cov_eig_ratio"),
                  _range(final, "min_radial_median_ratio", "max_radial_median_ratio"),
                  _step_time(result, "first_full_coverage"),
                  _step_time(result, "first_full_quality"),
                  _step_time(result, "stable_from"), _step_time(result, "confirmed"),
                  _step_time(result, "final") if result.get("budget_steps") is not None else "—",
                  result["reason"].replace("|", "/"))
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "The initial step is displayed in the raw events and animation but does not count toward convergence.", ""]
    return "\n".join(lines)


def evaluate_suite(output: Path | str, *, problem: str | None = None, write: bool = True) -> dict:
    """Grade all declared problems or a visibly scoped individual deep dive."""
    output = Path(output)
    if problem is not None and problem not in problems.PROBLEM_NAMES:
        raise ValueError(f"unknown problem: {problem}")
    names = (problem,) if problem else problems.PROBLEM_NAMES
    verdicts = {name: score_run(output / name, name) for name in names}
    status = "PASS" if all(row["passed"] for row in verdicts.values()) else "FAIL"
    result = {"status": status, "scope": "individual" if problem else "all declared problems",
              "protocol": "toy100-v1", "requirements": metrics.REQUIREMENTS,
              "all_problem_coverage": problem is None,
              "passed_problems": sum(row["passed"] for row in verdicts.values()),
              "required_problems": len(problems.PROBLEM_NAMES) if problem is None else 1,
              "problems": verdicts}
    if write:
        output.mkdir(parents=True, exist_ok=True)
        suffix = f"-{problem}" if problem else ""
        (output / f"gate{suffix}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (output / f"leaderboard{suffix}.md").write_text(leaderboard_markdown(result))
    return result
