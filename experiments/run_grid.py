#!/usr/bin/env python
"""
run_grid.py

Parallel runner for the regularizer study.

Takes a glob of YAML configs and executes

    .venv/bin/python experiments/train_arm.py --config <path>

for each one, from the worktree root, with a bounded number of concurrent
runs per GPU. Each worker slot pins ``CUDA_VISIBLE_DEVICES`` to exactly one
physical GPU, so the child process always sees a single device as cuda:0.

Design notes:
  * Resume support: a config whose ``out_dir`` already holds a *valid*
    summary.json (parses as JSON and has a "final" block) is skipped, so
    re-running the same command after a crash only fills in the gaps.
  * A run's stdout and stderr are merged and streamed to ``out_dir/log.txt``
    as it goes, so a run can be tailed while the grid is in flight.
  * A nonzero exit does not stop the grid: the failure is appended to
    ``results/failures.txt`` and the remaining runs continue.
  * Concurrency uses threads rather than processes; the worker body is a
    blocking ``subprocess`` wait, so the GIL is never the bottleneck.

Usage:

    python experiments/run_grid.py --configs 'configs/*.yaml' \
        --workers_per_gpu 5 --gpus 0,1
"""

import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

DEFAULT_PYTHON = ".venv/bin/python"
DEFAULT_TRAINER = "experiments/train_arm.py"
FAILURES_PATH = Path("results/failures.txt")


def load_out_dir(config_path: str) -> Optional[str]:
    """Read ``out_dir`` from a config YAML; returns None if unreadable."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001 - a bad config must not kill the grid
        print(f"[run_grid] WARN could not parse {config_path}: {exc}")
        return None
    if not isinstance(cfg, dict) or not cfg.get("out_dir"):
        print(f"[run_grid] WARN {config_path} has no out_dir")
        return None
    return str(cfg["out_dir"])


def has_valid_summary(out_dir: str) -> bool:
    """
    True if out_dir contains a summary.json that parses and looks complete.

    "Complete" is deliberately loose: a "final" block is enough. A run that
    died mid-write leaves truncated JSON, which fails to parse and is
    therefore re-run.
    """
    path = Path(out_dir) / "summary.json"
    if not path.is_file():
        return False
    try:
        with open(path) as f:
            summary = json.load(f)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(summary, dict) and isinstance(summary.get("final"), dict)


def record_failure(config_path: str, out_dir: str, returncode: int, lock: threading.Lock) -> None:
    """Append one line to results/failures.txt (thread-safe)."""
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}\tconfig={config_path}\tout_dir={out_dir}\treturncode={returncode}\n"
    with lock:
        with open(FAILURES_PATH, "a") as f:
            f.write(line)


def run_one(
    config_path: str,
    out_dir: str,
    gpu: str,
    python_bin: str,
    trainer: str,
    echo_last_line: bool = False,
) -> Tuple[int, float]:
    """
    Execute a single training run, streaming its output to out_dir/log.txt.

    Returns (returncode, wall_clock_seconds).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / "log.txt"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # Keep the child's prints flowing into log.txt rather than sitting in a
    # 4KB pipe buffer until the run ends.
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [python_bin, trainer, "--config", config_path]
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(f"# cmd: {' '.join(cmd)}\n# CUDA_VISIBLE_DEVICES={gpu}\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.getcwd(),
            )
            returncode = proc.wait()
        except FileNotFoundError as exc:
            log.write(f"\n[run_grid] launch failed: {exc}\n")
            returncode = 127
    if echo_last_line:
        # Surface the run's own final line (its [done] summary, or the
        # traceback tail) in the grid's stdout so one log tells the story.
        try:
            lines = [ln.rstrip() for ln in log_path.read_text().splitlines() if ln.strip()]
            if lines:
                print(f"[run_grid]   {Path(out_dir).name}: {lines[-1][:400]}", flush=True)
        except OSError:
            pass
    return returncode, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a grid of train_arm.py configs across GPUs."
    )
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help="Glob of config YAMLs, e.g. 'configs/*.yaml' (quote it).",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=5,
        help="Concurrent runs per GPU (default: 5).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated physical GPU ids (default: 0,1).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON,
        help=f"Python interpreter for the child runs (default: {DEFAULT_PYTHON}).",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default=DEFAULT_TRAINER,
        help=f"Trainer script path (default: {DEFAULT_TRAINER}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run configs even if a valid summary.json already exists.",
    )
    parser.add_argument(
        "--echo_last_line",
        action="store_true",
        help="After each run, print the last line of its log.txt (its [done] summary or error).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List what would run and exit.",
    )
    args = parser.parse_args()

    config_paths = sorted(glob.glob(args.configs))
    if not config_paths:
        print(f"[run_grid] no configs matched {args.configs!r}")
        sys.exit(1)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("[run_grid] --gpus produced no ids")
        sys.exit(1)

    # Build the work list, skipping already-finished runs.
    tasks: List[Tuple[str, str]] = []
    skipped = 0
    for config_path in config_paths:
        out_dir = load_out_dir(config_path)
        if out_dir is None:
            continue
        if not args.force and has_valid_summary(out_dir):
            skipped += 1
            continue
        tasks.append((config_path, out_dir))

    total = len(tasks)
    n_workers = args.workers_per_gpu * len(gpus)
    print(
        f"[run_grid] {len(config_paths)} configs matched | "
        f"{skipped} already done | {total} to run | "
        f"{len(gpus)} gpu(s) x {args.workers_per_gpu} = {n_workers} workers"
    )
    if args.dry_run:
        for config_path, out_dir in tasks:
            print(f"[run_grid] would run {config_path} -> {out_dir}")
        return
    if total == 0:
        return

    # One slot per (gpu, worker); a thread holds a slot for the duration of
    # a run and returns it afterwards, which is what pins the run to a GPU.
    slots: "queue.Queue[str]" = queue.Queue()
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            slots.put(gpu)

    task_q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    for task in tasks:
        task_q.put(task)

    print_lock = threading.Lock()
    fail_lock = threading.Lock()
    state = {"done": 0, "failed": 0, "elapsed_sum": 0.0}
    t_start = time.time()

    def worker() -> None:
        while True:
            try:
                config_path, out_dir = task_q.get_nowait()
            except queue.Empty:
                return
            gpu = slots.get()
            try:
                returncode, elapsed = run_one(
                    config_path, out_dir, gpu, args.python, args.trainer,
                    echo_last_line=args.echo_last_line,
                )
            except Exception as exc:  # noqa: BLE001 - never lose a worker
                returncode, elapsed = -1, 0.0
                with print_lock:
                    print(f"[run_grid] ERROR {config_path}: {exc}")
            finally:
                slots.put(gpu)
                task_q.task_done()

            if returncode != 0:
                record_failure(config_path, out_dir, returncode, fail_lock)

            with print_lock:
                state["done"] += 1
                state["elapsed_sum"] += elapsed
                if returncode != 0:
                    state["failed"] += 1
                done = state["done"]
                mean_dur = state["elapsed_sum"] / max(done, 1)
                remaining = total - done
                # Remaining runs are spread over n_workers lanes.
                eta_sec = mean_dur * remaining / max(n_workers, 1)
                status = "ok" if returncode == 0 else f"FAIL({returncode})"
                print(
                    f"[run_grid] {done}/{total} {status} "
                    f"{Path(out_dir).name} gpu={gpu} {elapsed:.0f}s | "
                    f"mean {mean_dur:.0f}s | failed {state['failed']} | "
                    f"ETA {eta_sec / 60.0:.1f} min",
                    flush=True,
                )

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = time.time() - t_start
    print(
        f"[run_grid] finished {state['done']}/{total} runs in {wall / 60.0:.1f} min "
        f"({state['failed']} failed)"
    )
    if state["failed"]:
        print(f"[run_grid] failures logged to {FAILURES_PATH}")


if __name__ == "__main__":
    main()
