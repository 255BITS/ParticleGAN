#!/usr/bin/env python
"""Run a grid and stream tagged per-run progress into one tail-friendly log.

Usage: python -u experiments/follow_grid.py --root results/denoising/screen \
    --log results/denoising/screen.log -- [run_grid.py arguments]
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--runner-log", help="Separate scheduler log; defaults to LOG with .runner.log suffix")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path = Path(args.runner_log) if args.runner_log else log_path.with_suffix(".runner.log")
    offsets = {}
    with log_path.open("a", buffering=1) as combined, runner_path.open("w") as runner_log:
        def emit(label, line):
            combined.write(f"{datetime.now().isoformat(timespec='seconds')} [{label}] {line}\n")

        command = [sys.executable, "-u", "experiments/run_grid.py", *forwarded]
        emit("launch", repr(command))
        process = subprocess.Popen(command, stdout=runner_log, stderr=subprocess.STDOUT)
        while True:
            done = process.poll() is not None
            for path in [runner_path, *sorted(Path(args.root).glob("*/log.txt"))]:
                with path.open(errors="replace") as source:
                    source.seek(offsets.get(path, 0))
                    for line in source:
                        line = line.rstrip()
                        if line and not line.startswith('{"final":'):
                            emit("grid" if path == runner_path else path.parent.name, line)
                    offsets[path] = source.tell()
            if done:
                emit("exit", f"runner return code {process.returncode}")
                return process.returncode
            time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
