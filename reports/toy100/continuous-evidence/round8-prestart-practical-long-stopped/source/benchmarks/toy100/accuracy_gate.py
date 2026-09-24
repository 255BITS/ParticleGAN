"""Independently grade sustained fidelity and an untouched final holdout draw."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .accuracy import LIMITS, PROTOCOL, evaluate_accuracy, passes_accuracy
from .gate import MIN_STABLE_CHECKS, evaluate_suite as coverage_suite
from .metrics import EVAL_N
from .problems import PROBLEM_NAMES


HOLDOUT_N = 100_000
HOLDOUT_SEED_OFFSETS = {"target": 1601, "noise": 1602, "latent": 1603}


def _read(path):
    return json.loads(path.read_text())


def _audit_cloud(path: Path, problem: str, expected_n: int) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if not {"live", "ema", "target"}.issubset(archive.files):
            raise ValueError(f"{path.name} lacks live, EMA, or target samples")
        for name in ("live", "ema", "target"):
            points = archive[name]
            if (points.shape != (expected_n, 2)
                    or not np.issubdtype(points.dtype, np.floating)
                    or not np.isfinite(points).all()):
                raise ValueError(f"invalid {name} samples in {path.name}")
        return {name: evaluate_accuracy(archive[name], problem)
                for name in ("live", "ema", "target")}


def _same_accuracy(recorded: dict, recomputed: dict) -> bool:
    if not isinstance(recorded, dict) or recorded.keys() != recomputed.keys():
        return False
    for key, value in recomputed.items():
        old = recorded[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (isinstance(old, bool) or not isinstance(old, (int, float))
                    or not math.isfinite(old)
                    or not math.isclose(old, value, rel_tol=1e-7, abs_tol=1e-9)):
                return False
        elif old != value:
            return False
    return True


def score_run(run_dir: Path | str, problem: str, coverage: dict) -> dict:
    """Require five complete final clouds plus a separate 100,000-draw audit."""
    directory = Path(run_dir)
    row = {"problem": problem, "status": "FAIL", "passed": False,
           "coverage_status": coverage["status"], "terminal_checks": []}
    try:
        summary = _read(directory / "summary.json")
        declaration = summary.get("accuracy")
        if not isinstance(declaration, dict) or declaration.get("protocol") != PROTOCOL:
            raise ValueError("missing accuracy-protocol evidence")
        steps = summary["eval_steps"][-MIN_STABLE_CHECKS:]
        if (len(steps) != MIN_STABLE_CHECKS or steps[0] <= 0
                or declaration.get("check_steps") != steps):
            raise ValueError("accuracy checks must be the final five postinitial evaluations")
        n = summary["config"]["eval_samples"]
        if type(n) is not int or n < EVAL_N or declaration.get("sample_count") != n:
            raise ValueError("accuracy draws differ from the declared evaluation count")
        if (declaration.get("holdout_samples") != HOLDOUT_N
                or declaration.get("holdout_seed_offsets") != HOLDOUT_SEED_OFFSETS):
            raise ValueError("holdout protocol differs from the fixed independent draw")
        recorded = {}
        for line in (directory / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "eval" and event.get("model") in ("live", "ema"):
                key = (event["step"], event["model"])
                if key in recorded:
                    raise ValueError("duplicate recorded accuracy evaluation")
                recorded[key] = event.get("accuracy")
        terminal = []
        for step in steps:
            path = directory / "quality_checks" / f"step_{step:06d}.npz"
            metrics = _audit_cloud(path, problem, n)
            for model in ("live", "ema"):
                if not _same_accuracy(recorded.get((step, model)), metrics[model]):
                    raise ValueError(f"recorded accuracy differs from saved {model} samples at {step}")
            if not metrics["target"]["passed"]:
                raise ValueError(f"saved target reference fails the oracle audit at {step}")
            terminal.append({"step": step, "passed": bool(metrics["live"]["frozen_pass"]
                              and passes_accuracy(metrics["live"])),
                             "metrics": metrics["live"], "ema_metrics": metrics["ema"]})
        holdout = _audit_cloud(directory / "holdout_samples.npz", problem, HOLDOUT_N)
        for model in ("live", "ema", "target"):
            if not _same_accuracy(summary.get("holdout", {}).get(model), holdout[model]):
                raise ValueError(f"recorded holdout differs from saved {model} samples")
        if not holdout["target"]["passed"]:
            raise ValueError("saved holdout target reference fails the oracle audit")
        passed = (coverage["passed"] and all(check["passed"] for check in terminal)
                  and holdout["live"]["frozen_pass"] and passes_accuracy(holdout["live"]))
        row.update(status="PASS" if passed else "FAIL", passed=bool(passed),
                   terminal_checks=terminal, holdout_metrics=holdout["live"],
                   holdout_ema_metrics=holdout["ema"], oracle_metrics=holdout["target"],
                   final_metrics=terminal[-1]["metrics"],
                   reason="sustained live accuracy and independent holdout pass" if passed else
                   "coverage, sustained live accuracy, or independent holdout failed")
    except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, BadZipFile) as exc:
        row.update(status="INVALID", reason=str(exc))
    return row


def evaluate_suite(output: Path | str, *, problem: str | None = None,
                   write: bool = True) -> dict:
    output = Path(output)
    coverage = coverage_suite(output, problem=problem, write=False)
    names = (problem,) if problem else PROBLEM_NAMES
    rows = {name: score_run(output / name, name, coverage["problems"][name]) for name in names}
    passed = sum(row["passed"] for row in rows.values())
    result = {"protocol": PROTOCOL, "status": "PASS" if passed == len(names) else "FAIL",
              "scope": "individual" if problem else "all declared problems",
              "limits": LIMITS, "holdout_samples": HOLDOUT_N,
              "passed_problems": passed, "required_problems": len(names), "problems": rows}
    if write:
        output.mkdir(parents=True, exist_ok=True)
        suffix = f"-{problem}" if problem else ""
        (output / f"accuracy-gate{suffix}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n")
        lines = [f"# 100-Gaussian accuracy gate: {result['status']}", "",
                 "Original coverage criteria, five final 20k-draw fidelity checks, and a separate "
                 "100k-draw holdout are required. EMA is diagnostic.", "",
                 "| Problem | Status | Terminal checks | Holdout mass TV | Center RMS / σ | "
                 "Covariance trace bias | Radial KS | Reason |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for name, row in rows.items():
            m = row.get("holdout_metrics", {})
            def cell(key):
                value = m.get(key)
                return f"{value:.4f}" if isinstance(value, (int, float)) else "—"
            lines.append("| " + " | ".join([
                name, row["status"], str(sum(c["passed"] for c in row["terminal_checks"])),
                cell("mass_tv"), cell("center_rms_sigma"), cell("cov_trace_bias"),
                cell("radial_ks"), row["reason"],
            ]) + " |")
        (output / f"accuracy-leaderboard{suffix}.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem", choices=PROBLEM_NAMES)
    args = parser.parse_args()
    result = evaluate_suite(args.output, problem=args.problem)
    print(json.dumps({key: result[key] for key in
                      ("protocol", "status", "passed_problems", "required_problems")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
