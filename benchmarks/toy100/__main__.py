"""Run, grade, and inspect the frozen 100-Gaussian suite.

Examples:
  python -u -m benchmarks.toy100 run --output artifacts/toy100/recommended
  python -u -m benchmarks.toy100 run --config configs/toy100/baseline.json --output artifacts/toy100/baseline
  python -u -m benchmarks.toy100 run --output artifacts/toy100/grid-deep --problem grid100 --steps 14000
  python -m benchmarks.toy100 gate --output artifacts/toy100/recommended
  python -m benchmarks.toy100 render --output artifacts/toy100/recommended
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from .gate import evaluate_suite
from .config import resolve_problem_config
from .problems import PROBLEM_NAMES
from .render import render_progress
from .train import load_config, train


def _parser():
    parser = argparse.ArgumentParser(description="100-Gaussian training and evidence gate")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="train then gate all problems, or one named problem")
    run.add_argument("--config", type=Path, default=Path("configs/toy100/recommended.json"),
                     help="frozen JSON/TOML recipe (default: configs/toy100/recommended.json)")
    run.add_argument("--output", type=Path, required=True, help="new run directory")
    run.add_argument("--problem", choices=PROBLEM_NAMES, help="individual deep dive")
    run.add_argument("--steps", type=int, help="override training budget for a deep dive")
    run.add_argument("--device", help="override device from the config")
    run.add_argument("--no-render", action="store_true", help="skip diagnostic GIF rendering")
    for name in ("gate", "render"):
        command = commands.add_parser(name, help=f"{name} existing recorded runs")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--problem", choices=PROBLEM_NAMES, help="inspect one problem")
    return parser


def _run(args):
    config_bytes = args.config.read_bytes()
    manifest = load_config(args.config)
    # Preflight every declared problem, even for an individual deep dive. An
    # invalid hidden override cannot be ignored just because it was unselected.
    configs = {name: resolve_problem_config(manifest, name, steps=args.steps, device=args.device)
               for name in PROBLEM_NAMES}
    names = (args.problem,) if args.problem else PROBLEM_NAMES
    args.output.mkdir(parents=True, exist_ok=True)
    # A new CLI invocation must not mix old rows with new training evidence.
    for name in names:
        folder = args.output / name
        if (folder / "summary.json").exists() or (folder / "events.jsonl").exists():
            raise FileExistsError(f"existing run evidence at {folder}; choose a new output directory")
    declaration = {"config_path": str(args.config),
                   "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                   "config_contents": config_bytes.decode("utf-8"),
                   "declared_manifest": manifest, "resolved_problem_configs": configs,
                   "command_overrides": {"steps": args.steps, "device": args.device},
                   "selected_problems": names}
    (args.output / "run_manifest.json").write_text(json.dumps(declaration, indent=2, allow_nan=False) + "\n")
    for name in names:
        folder = args.output / name
        config = configs[name]
        print(json.dumps({"event": "problem_start", "problem": name,
                          "steps": config["steps"], "output": str(folder)}), flush=True)
        try:
            summary = train(config, folder)
            print(json.dumps({"event": "problem_complete", "problem": name,
                              "status": summary.get("status"), "output": str(folder)}), flush=True)
        except Exception as exc:
            # Keep the attempted problem visible in the aggregate gate. Preserve
            # any runner-written receipt and all partial events for debugging.
            folder.mkdir(parents=True, exist_ok=True)
            config_path = folder / "config.json"
            if not config_path.exists():
                config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
            summary_path = folder / "summary.json"
            if not summary_path.exists():
                summary_path.write_text(json.dumps({"problem": name, "status": "error",
                                                    "error": repr(exc)}, indent=2) + "\n")
            print(json.dumps({"event": "problem_error", "problem": name,
                              "error": repr(exc)}), flush=True)
            traceback.print_exc()

    gate = evaluate_suite(args.output, problem=args.problem)
    print(json.dumps({"event": "gate", "status": gate["status"],
                      "passed": gate["passed_problems"],
                      "required": gate["required_problems"],
                      "leaderboard": str(args.output / (f"leaderboard-{args.problem}.md" if args.problem
                                                          else "leaderboard.md"))}), flush=True)
    render_error = None
    if not args.no_render:
        try:
            for path in render_progress(args.output, problem=args.problem):
                print(json.dumps({"event": "gif", "path": str(path)}), flush=True)
        except (OSError, ValueError, ImportError) as exc:
            render_error = exc
            print(json.dumps({"event": "render_error", "error": str(exc)}), flush=True)
    return 0 if gate["status"] == "PASS" and render_error is None else 1


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "gate":
            verdict = evaluate_suite(args.output, problem=args.problem)
            print(json.dumps({"status": verdict["status"],
                              "scope": verdict["scope"],
                              "passed": verdict["passed_problems"],
                              "required": verdict["required_problems"]}))
            return 0 if verdict["status"] == "PASS" else 1
        for path in render_progress(args.output, problem=args.problem):
            print(path)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"toy100: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
