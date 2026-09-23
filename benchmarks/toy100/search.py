"""Run a declared configuration search while retaining every trial and failure.

This is a research command, not the acceptance gate. The gate and its thresholds
are unchanged by a search; a ranked failure remains a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import resolve_problem_config
from .gate import evaluate_suite
from .problems import PROBLEM_NAMES
from .train import train


def run_search(base: dict, candidates: list[dict], output: Path, *,
               problems: tuple[str, ...] = PROBLEM_NAMES, device: str | None = None):
    if not candidates or len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("candidate names must be nonempty and unique")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"search output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"base": base, "candidates": candidates, "problems": list(problems),
            "device": device, "selection": "frozen gate; no seed sweep"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for candidate in candidates:
        name = candidate["name"]
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"invalid candidate name: {name!r}")
        config = {**base, **candidate.get("overrides", {}), "name": name}
        if config.get("seed") != base.get("seed"):
            raise ValueError("search candidates must preserve the declared seed")
        if device is not None:
            config["device"] = device
        directory = output / name
        print(json.dumps({"event": "candidate_start", "candidate": name,
                          "overrides": candidate.get("overrides", {})}), flush=True)
        scores = {}
        for problem in problems:
            try:
                train(resolve_problem_config(config, problem), directory / problem)
            except Exception as error:
                # Numerical/configuration failures stay visible in the search.
                # The runner retains partial evidence when training has begun.
                folder = directory / problem
                folder.mkdir(parents=True, exist_ok=True)
                if not (folder / "summary.json").exists():
                    (folder / "summary.json").write_text(json.dumps({
                        "status": "error", "problem": problem,
                        "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
                print(json.dumps({"event": "candidate_error", "candidate": name,
                                  "problem": problem, "error": str(error)}), flush=True)
            gate = evaluate_suite(directory, problem=problem)
            scores[problem] = gate["problems"][problem]
        if set(problems) == set(PROBLEM_NAMES):
            evaluate_suite(directory)
        result = {"candidate": name, "config": config, "problems": scores,
                  "passed_problems": sum(row["passed"] for row in scores.values()),
                  "required_problems": len(problems)}
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"candidate": name,
                "passed_problems": result["passed_problems"],
                "final": {p: v.get("final_metrics", {}) for p, v in scores.items()}}, allow_nan=False) + "\n")
        print(json.dumps({"event": "candidate_complete", "candidate": name,
                          "passed_problems": result["passed_problems"]}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem", choices=PROBLEM_NAMES, action="append")
    parser.add_argument("--device")
    args = parser.parse_args()
    run_search(json.loads(args.base.read_text()), json.loads(args.candidates.read_text()),
               args.output, problems=tuple(args.problem or PROBLEM_NAMES), device=args.device)


if __name__ == "__main__":
    main()
