"""Run a config queue on two GPUs; only summarize a job after it exits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    jobs = iter(args.configs)
    failed = []
    with (args.out/"queue.log").open("w", buffering=1) as stream:
        def log(**row):
            line = json.dumps(row)
            with lock:
                stream.write(line+"\n")
                print(line, flush=True)
        def worker(device):
            while True:
                with lock:
                    config = next(jobs, None)
                if config is None:
                    return
                name = json.loads(config.read_text())["name"]
                directory = args.out/name
                log(event="launch", name=name, device=device, config=str(config))
                with (args.out/f"{name}.console.log").open("w") as console:
                    code = subprocess.call([sys.executable, "-u", "experiments/memory_scout.py",
                        "--config", str(config), "--out", str(directory), "--device", device],
                        stdout=console, stderr=subprocess.STDOUT)
                if code:
                    failed.append(name)
                    log(event="failed", name=name, code=code)
                else:
                    result = json.loads((directory/"summary.json").read_text())
                    log(event="completed", name=name, metrics=result["metrics"]["generated_256"])
        with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
            list(pool.map(worker, args.devices))
        log(event="queue_complete", failed=failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
