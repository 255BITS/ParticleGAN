"""One bounded sequential controller per physical GPU; logs are easy to tail."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=("0", "1"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/forge/smooth-polar-factorial-v1")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.slot, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    tasks = [task for task, slot in protocol["gpu_assignment"].items() if str(slot) == args.slot]
    reservation = sum(protocol["tasks"][task]["timeout_seconds"] for task in tasks) * len(protocol["arms"])
    print(json.dumps(dict(event="controller_start", slot=args.slot, tasks=tasks,
                          reservation_seconds=reservation, protocol=protocol["id"])), flush=True)
    rows = []
    for task in tasks:
        for arm in protocol["arms"]:
            name = task + "--" + arm["id"]
            output, log = args.output / name, args.output / (name + ".log")
            if output.exists() or log.exists():
                raise ValueError("refusing to overwrite or repeat existing arm: " + name)
            started = time.monotonic()
            print(json.dumps(dict(event="start", task=task, arm=arm["id"], log=str(log))), flush=True)
            with log.open("x") as stream:
                process = subprocess.Popen([sys.executable, "-u", str(HERE / "run.py"),
                    "--task", task, "--arm", arm["id"], "--output", str(output)],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    result = process.wait(timeout=protocol["tasks"][task]["timeout_seconds"] + 30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    result = process.wait()
            row = dict(task=task, arm=arm["id"], exit_code=result,
                       controller_wall_seconds=time.monotonic() - started,
                       receipt_saved=(output / "receipt.json").exists())
            rows.append(row)
            (args.output / ("controller-" + args.slot + ".json")).write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(dict(event="finished", **row)), flush=True)
    print(json.dumps(dict(event="controller_complete", slot=args.slot, runs=len(rows))), flush=True)


if __name__ == "__main__":
    main()
