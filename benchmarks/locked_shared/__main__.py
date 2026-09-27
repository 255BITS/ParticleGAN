"""python -m benchmarks.locked_shared --output runs/locked_shared

Trains the three locked_shared host toys (two-pole, trajectory, ring) and
writes results.json and README.md under --output. One ``start``/``done``
line per toy goes to stdout, so the run is easy to tail.
"""

import argparse
import json
from pathlib import Path
import platform

import torch

from .run import markdown, run_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/locked_shared"))
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    report = {"python": platform.python_version(), "torch": torch.__version__, "rows": run_all()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    table = markdown(report)
    (args.output / "README.md").write_text(table)
    print(table)
    return 0 if all(r["verdict"] == "PASS" for r in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
